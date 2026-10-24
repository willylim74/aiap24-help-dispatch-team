"""
Model training layer.

Implements the two-stage "hurdle" architecture established in eda.ipynb
(Sections 10, 18.5, 19, 21-26):

    stage1_classification -- P(is_late), trained on every row
    stage2_regression      -- E[lateness_minutes | is_late], trained only on
                               the late-labelled subset of each fold, split
                               out AFTER the shared train/val/test split so
                               there is no leakage between what the
                               classifier sees as negatives and what the
                               regressor is trained on

Both stages share one preprocessing pipeline shape (see data_preparation.py)
but each model gets its OWN freshly-built, unfitted pipeline instance --
never a fitted one reused across models or folds.

Class weighting (EDA Section 18.2/21.2) is applied as a standing design
decision for classifiers that support it, not a searched hyperparameter.
Hyperparameter search assigns GridSearchCV to "linear"-family models and
HalvingRandomSearchCV to "tree"-family models, per config; the winning
configuration from every search is always refit on the FULL training fold
(never the search subsample) before being evaluated.

Checkpointing: every tuned model is written to disk (ledger + pickled
pipeline) the moment it finishes, so a re-run or a recovery from an
interrupted run skips whatever's already done instead of refitting it.
"""
from __future__ import annotations

import importlib
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                              mean_absolute_error, mean_squared_error, precision_score,
                              r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, HalvingRandomSearchCV, KFold,
                                      StratifiedKFold, train_test_split)
from sklearn.pipeline import Pipeline

from data_preparation import build_preprocessing_pipeline

_ESTIMATOR_MODULES = {
    "LinearRegression": "sklearn.linear_model", "Ridge": "sklearn.linear_model",
    "Lasso": "sklearn.linear_model", "LogisticRegression": "sklearn.linear_model",
    "RidgeClassifier": "sklearn.linear_model",
    "DecisionTreeRegressor": "sklearn.tree", "DecisionTreeClassifier": "sklearn.tree",
    "RandomForestRegressor": "sklearn.ensemble", "RandomForestClassifier": "sklearn.ensemble",
    "GradientBoostingRegressor": "sklearn.ensemble", "GradientBoostingClassifier": "sklearn.ensemble",
    "KNeighborsRegressor": "sklearn.neighbors", "KNeighborsClassifier": "sklearn.neighbors",
    "DummyRegressor": "sklearn.dummy", "DummyClassifier": "sklearn.dummy",
}


def _instantiate(estimator_name: str, params: dict):
    module = importlib.import_module(_ESTIMATOR_MODULES[estimator_name])
    cls = getattr(module, estimator_name)
    return cls(**(params or {}))


# ============================================================================
# Split
# ============================================================================

def split_data(df: pd.DataFrame, feature_cols: list[str], config: dict) -> dict:
    """70/15/15 train/val/test, stratified on the classification target
    (EDA Section 13), computed once and shared by both stages. Stage 2's
    late-only subsets are derived from these same splits in
    `get_stage2_subset`, never from an independent split.
    """
    target_cfg = config["target"]
    split_cfg = config["split"]
    random_state = config["random_state"]

    X = df[feature_cols].copy()
    y_class = df[target_cfg["classification_col"]].copy()
    y_severity = df[target_cfg["severity_col"]].copy()

    val_test_ratio = split_cfg["val_ratio"] + split_cfg["test_ratio"]
    X_train, X_tmp, y_train, y_tmp, ysev_train, ysev_tmp = train_test_split(
        X, y_class, y_severity, test_size=val_test_ratio,
        stratify=y_class if split_cfg["stratify_on"] else None, random_state=random_state,
    )
    test_share_of_tmp = split_cfg["test_ratio"] / val_test_ratio
    X_val, X_test, y_val, y_test, ysev_val, ysev_test = train_test_split(
        X_tmp, y_tmp, ysev_tmp, test_size=test_share_of_tmp,
        stratify=y_tmp if split_cfg["stratify_on"] else None, random_state=random_state,
    )
    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "ysev_train": ysev_train, "ysev_val": ysev_val, "ysev_test": ysev_test,
    }


def get_stage2_subset(splits: dict, fold: str) -> tuple[pd.DataFrame, pd.Series]:
    """Filter a fold to the late-only subset AFTER splitting (leakage-safe:
    Stage 2 never influences how Stage 1's train/val/test rows were chosen).
    fold is one of 'train', 'val', 'test'.
    """
    is_late = splits[f"y_{fold}"] == 1
    return splits[f"X_{fold}"][is_late], splits[f"ysev_{fold}"][is_late]


# ============================================================================
# Checkpoint ledger
# ============================================================================

class Checkpointer:
    """Resumable on-disk ledger: one JSON entry per model plus its pickled
    pipeline. `is_done`/`checkpoint` are the only two operations tuning
    functions need.
    """

    def __init__(self, config: dict):
        self.output_cfg = config["output"]
        os.makedirs(self.output_cfg["models_dir"], exist_ok=True)
        os.makedirs(self.output_cfg["results_dir"], exist_ok=True)
        os.makedirs(self.output_cfg["checkpoints_dir"], exist_ok=True)
        self.ledger_path = self.output_cfg["ledger_path"]

    def _load(self) -> dict:
        return json.load(open(self.ledger_path)) if os.path.exists(self.ledger_path) else {}

    def is_done(self, key: str) -> bool:
        return self._load().get(key, {}).get("status") == "done"

    def get(self, key: str) -> dict:
        return self._load().get(key)

    def checkpoint(self, key: str, stage: str, pipe, metrics: dict, best_params: dict,
                    search_seconds: float, extra: dict | None = None) -> dict:
        safe_name = key.replace(" ", "_").replace("/", "-").replace(":", "").lower()
        model_path = os.path.join(self.output_cfg["models_dir"], f"{stage}_{safe_name}.joblib")
        joblib.dump(pipe, model_path)
        record = {
            "key": key, "stage": stage, "model_path": model_path, "metrics": metrics,
            "best_params": best_params, "search_seconds": search_seconds,
            "status": "done", "timestamp": time.time(),
        }
        if extra:
            record["extra"] = extra
        ledger = self._load()
        ledger[key] = record
        json.dump(ledger, open(self.ledger_path, "w"), indent=2)
        return record


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_classifier(pipe, X_eval, y_eval) -> dict:
    pred = pipe.predict(X_eval)
    score = pipe.predict_proba(X_eval)[:, 1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X_eval)
    return {
        "accuracy": accuracy_score(y_eval, pred),
        "precision": precision_score(y_eval, pred, zero_division=0),
        "recall": recall_score(y_eval, pred, zero_division=0),
        "f1": f1_score(y_eval, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_eval, score),
        "pr_auc": average_precision_score(y_eval, score),
    }


def evaluate_regressor(pipe, X_eval, y_eval) -> dict:
    pred = pipe.predict(X_eval)
    mse = mean_squared_error(y_eval, pred)
    return {"MAE": mean_absolute_error(y_eval, pred), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_eval, pred)}


def precision_recall_at_k(y_true, score, k_fractions: list[float]) -> pd.DataFrame:
    """Ranked-score evaluation (EDA Section 18.5/19/25, the recommended
    deployment framing): for the top K% of rows by predicted risk score,
    what fraction are actually positive (precision@K) and what fraction of
    all positives does that capture (recall@K)?
    """
    y_true = np.asarray(y_true)
    n = len(y_true)
    order = np.argsort(-score)
    y_sorted = y_true[order]
    total_pos = y_sorted.sum()
    rows = []
    for k_frac in k_fractions:
        k = max(1, int(round(n * k_frac)))
        tp = y_sorted[:k].sum()
        rows.append({"k_pct_reviewed": k_frac * 100, "k_count": k,
                     "precision_at_k": tp / k, "recall_at_k": tp / total_pos})
    return pd.DataFrame(rows)


# ============================================================================
# Class imbalance handling (Section 18.2/21.2)
# ============================================================================

def is_imbalanced(y: pd.Series, config: dict) -> bool:
    positive_rate = y.mean()
    threshold = config["class_imbalance"]["positive_rate_threshold"]
    return positive_rate < threshold or positive_rate > (1 - threshold)


def apply_class_weight(model_name: str, estimator, config: dict):
    """Fix class_weight='balanced' on the constructor for models that
    support it and are configured as applicable (a standing design
    decision, not a searched option -- see config.yaml class_imbalance).
    Returns the estimator unchanged otherwise.
    """
    imbalance_cfg = config["class_imbalance"]
    if model_name in imbalance_cfg.get("applies_to_models", []):
        estimator.set_params(class_weight=imbalance_cfg["strategy"])
    return estimator


def balanced_sample_weight(y: pd.Series) -> pd.Series:
    counts = y.value_counts()
    weight_map = {0: 1.0, 1: counts[0] / counts[1]}
    return y.map(weight_map)


def _stratified_subsample(X: pd.DataFrame, y: pd.Series, n: int, random_state: int):
    rng = np.random.RandomState(random_state)
    idx_pos = y[y == 1].index.values
    idx_neg = y[y == 0].index.values
    n_pos = int(round(n * (len(idx_pos) / len(y))))
    n_neg = n - n_pos
    sel_pos = rng.choice(idx_pos, size=min(n_pos, len(idx_pos)), replace=False)
    sel_neg = rng.choice(idx_neg, size=min(n_neg, len(idx_neg)), replace=False)
    sel = np.concatenate([sel_pos, sel_neg])
    rng.shuffle(sel)
    return X.loc[sel], y.loc[sel]


# ============================================================================
# Baseline fitting
# ============================================================================

def fit_baseline_models(stage: str, splits_or_subset: dict, config: dict,
                          missing_flag_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Fit every model in the configured registry (plus the naive baseline)
    with default hyperparameters, evaluate on the validation set. Returns
    (results_dataframe, fitted_pipelines_dict).
    """
    stage_cfg = config["models"]["classification" if stage == "stage1" else "regression"]
    is_classification = stage == "stage1"
    evaluate_fn = evaluate_classifier if is_classification else evaluate_regressor
    ranking_metric = config["evaluation"]["classification" if is_classification else "regression"]["ranking_metric"]

    X_train, y_train = splits_or_subset["X_train"], splits_or_subset["y_train"]
    X_val, y_val = splits_or_subset["X_val"], splits_or_subset["y_val"]

    all_models = [stage_cfg["naive"]] + stage_cfg["registry"]

    rows, fitted = [], {}
    for spec in all_models:
        name = spec["name"]
        estimator = _instantiate(spec["estimator"], spec.get("params", {}))
        # baseline fitting deliberately uses library defaults throughout
        # (EDA Section 16): class weighting is introduced later, as an
        # explicit tuning-phase decision (Section 18.2/21.2), specifically
        # so the baseline-vs-tuned comparison shows its effect

        pipe = Pipeline([("features", build_preprocessing_pipeline(config, missing_flag_cols)),
                          ("model", estimator)])
        t0 = time.time()
        pipe.fit(X_train, y_train)
        fit_seconds = time.time() - t0

        metrics = evaluate_fn(pipe, X_val, y_val)
        metrics.update({"model": name, "fit_seconds": round(fit_seconds, 1)})
        rows.append(metrics)
        fitted[name] = pipe

    metric_cols = config["evaluation"]["classification" if is_classification else "regression"]["metrics"]
    results = pd.DataFrame(rows).set_index("model")[metric_cols + ["fit_seconds"]]
    results = results.sort_values(ranking_metric, ascending=False)
    return results, fitted


# ============================================================================
# Hyperparameter tuning
# ============================================================================

def tune_model(stage: str, spec: dict, splits_or_subset: dict, config: dict,
                missing_flag_cols: list[str], checkpointer: Checkpointer) -> dict:
    """Tune one model per its configured search_family:
      - 'none'   -> no hyperparameters; nothing to search
      - 'linear' -> GridSearchCV on the full training fold
      - 'tree'   -> HalvingRandomSearchCV, subsampled per config for the
                    classification stage (compute-constrained strategy),
                    then refit on the FULL training fold
    Checkpointed: returns immediately from the ledger if already done.
    """
    name = spec["name"]
    ledger_key = f"{stage}::{name}"
    if checkpointer.is_done(ledger_key):
        return checkpointer.get(ledger_key)

    is_classification = stage == "stage1"
    evaluate_fn = evaluate_classifier if is_classification else evaluate_regressor
    scoring = "average_precision" if is_classification else "r2"
    random_state = config["random_state"]

    X_train, y_train = splits_or_subset["X_train"], splits_or_subset["y_train"]
    X_val, y_val = splits_or_subset["X_val"], splits_or_subset["y_val"]

    estimator = _instantiate(spec["estimator"], spec.get("params", {}))
    imbalanced = is_imbalanced(y_train, config) if is_classification else False
    use_class_weight = (is_classification and imbalanced
                         and name in config["class_imbalance"].get("applies_to_models", []))
    use_sample_weight = (is_classification and imbalanced
                         and name in config["class_imbalance"].get("sample_weight_models", []))
    if use_class_weight:
        estimator = apply_class_weight(name, estimator, config)

    if spec["search_family"] == "none":
        pipe = Pipeline([("features", build_preprocessing_pipeline(config, missing_flag_cols)),
                          ("model", estimator)])
        fit_kwargs = {"model__sample_weight": balanced_sample_weight(y_train)} if use_sample_weight else {}
        t0 = time.time()
        pipe.fit(X_train, y_train, **fit_kwargs)
        metrics = evaluate_fn(pipe, X_val, y_val)
        return checkpointer.checkpoint(ledger_key, stage, pipe, metrics, {}, time.time() - t0,
                                        extra={"note": "no hyperparameters to tune", "display_name": name})

    param_grid = {f"model__{k}": v for k, v in spec["param_grid"].items()}
    cv_folds = config["tuning"]["cv_folds"]
    cv = (StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state) if is_classification
          else KFold(n_splits=cv_folds, shuffle=True, random_state=random_state))

    subsample_cfg = config["tuning"]["subsample"]
    use_subsample = (
        spec["search_family"] == "tree"
        and subsample_cfg["enabled"]
        and subsample_cfg["applies_to_stage"] == ("classification" if is_classification else "regression")
        and len(X_train) > subsample_cfg["n_rows"]
    )

    t0 = time.time()
    pipe = Pipeline([("features", build_preprocessing_pipeline(config, missing_flag_cols)),
                      ("model", estimator)])

    if spec["search_family"] == "linear":
        search = GridSearchCV(pipe, param_grid, scoring=scoring, cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
    else:  # tree -> HalvingRandomSearchCV
        if use_subsample:
            X_search, y_search = _stratified_subsample(X_train, y_train, subsample_cfg["n_rows"], random_state)
        else:
            X_search, y_search = X_train, y_train
        search = HalvingRandomSearchCV(
            pipe, param_grid, resource="n_samples",
            max_resources=len(X_search), min_resources=max(200, len(X_search) // 9),
            factor=config["tuning"]["halving"]["factor"], scoring=scoring, cv=3,
            random_state=random_state, n_jobs=-1,
        )
        search.fit(X_search, y_search)
    search_seconds = time.time() - t0

    t1 = time.time()
    final_pipe = Pipeline([("features", build_preprocessing_pipeline(config, missing_flag_cols)),
                            ("model", _instantiate(spec["estimator"], spec.get("params", {})))])
    if use_class_weight:
        final_pipe.set_params(model__class_weight=config["class_imbalance"]["strategy"])
    final_pipe.set_params(**search.best_params_)
    fit_kwargs = {"model__sample_weight": balanced_sample_weight(y_train)} if use_sample_weight else {}
    final_pipe.fit(X_train, y_train, **fit_kwargs)
    refit_seconds = time.time() - t1

    metrics = evaluate_fn(final_pipe, X_val, y_val)
    return checkpointer.checkpoint(
        ledger_key, stage, final_pipe, metrics, search.best_params_, search_seconds,
        extra={"refit_seconds": refit_seconds, "searched_on_subsample": use_subsample,
               "sample_weight_applied": use_sample_weight, "display_name": name},
    )


def tune_all_models(stage: str, splits_or_subset: dict, config: dict,
                      missing_flag_cols: list[str], checkpointer: Checkpointer) -> pd.DataFrame:
    """Tune every registry model for a stage and return the results table,
    sorted by the stage's ranking metric. `Linear` (no hyperparameters) is
    included with its baseline result carried forward unchanged.
    """
    is_classification = stage == "stage1"
    stage_cfg = config["models"]["classification" if is_classification else "regression"]
    ranking_metric = config["evaluation"]["classification" if is_classification else "regression"]["ranking_metric"]
    metric_cols = config["evaluation"]["classification" if is_classification else "regression"]["metrics"]

    rows = []
    for spec in stage_cfg["registry"]:
        record = tune_model(stage, spec, splits_or_subset, config, missing_flag_cols, checkpointer)
        rows.append({**record["metrics"], "model": record.get("extra", {}).get("display_name", record["key"]),
                     "best_params": json.dumps(record["best_params"])})

    results = pd.DataFrame(rows).set_index("model")[metric_cols + ["best_params"]]
    return results.sort_values(ranking_metric, ascending=False)


# ============================================================================
# Final test-set evaluation
# ============================================================================

def evaluate_best_on_test(stage: str, tuned_results: pd.DataFrame, config: dict,
                            checkpointer: Checkpointer, X_test, y_test) -> dict:
    """Select the top-ranked tuned model and evaluate it ONCE on the held
    -out test set. This must only be called after tuning/validation-based
    selection is finalized.
    """
    is_classification = stage == "stage1"
    ranking_metric = config["evaluation"]["classification" if is_classification else "regression"]["ranking_metric"]
    evaluate_fn = evaluate_classifier if is_classification else evaluate_regressor

    best_name = tuned_results[ranking_metric].idxmax()
    ledger_key = f"{stage}::{best_name}"
    record = checkpointer.get(ledger_key)
    pipe = joblib.load(record["model_path"])
    test_metrics = evaluate_fn(pipe, X_test, y_test)
    return {"model": best_name, "pipeline": pipe, "test_metrics": test_metrics,
            "validation_metrics": record["metrics"]}
