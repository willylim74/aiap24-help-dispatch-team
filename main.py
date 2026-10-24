"""
Orchestration layer for the MoveEasy delivery risk-scoring pipeline.

Run with:  python main.py [--config src/config.yaml] [--skip-tuning]

This can be invoked from any working directory -- all paths (src/, the
config file, data/data.csv, outputs/) are resolved relative to this file's
own location (the project root), not the directory the command happened to
be run from.

Control flow:
    1. load config.yaml
    2. load + clean data.csv (data_preparation)
    3. split into train/val/test (shared across both hurdle-model stages)
    4. for each stage (classification, regression):
         - fit baseline models, evaluate on validation
         - tune every model, evaluate on validation
         - select the best tuned model and evaluate it once on test
    5. ranked-score (Precision@K/Recall@K) demonstration on the test set,
       using the selected Stage 1 model -- the recommended deployment
       framing from eda.ipynb Section 18.5/19/25
    6. print + save a closing summary comparing baseline / tuned / test
       results for both stages
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd
import yaml

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from data_preparation import get_feature_columns, prepare_raw_data
from model_training import (Checkpointer, evaluate_best_on_test, fit_baseline_models,
                             get_stage2_subset, precision_recall_at_k, split_data,
                             tune_all_models)


def load_config(path: str) -> dict:
    with open(path) as f:
        config = yaml.safe_load(f)
    return resolve_config_paths(config)


def resolve_config_paths(config: dict) -> dict:
    """Rewrite every relative path in the config to be absolute, anchored
    at PROJECT_ROOT, so the pipeline behaves identically no matter what
    directory `python main.py` was invoked from.
    """
    raw_path = config["data"]["raw_path"]
    if not os.path.isabs(raw_path):
        config["data"]["raw_path"] = os.path.join(PROJECT_ROOT, raw_path)

    for key in ("base_dir", "models_dir", "results_dir", "checkpoints_dir", "ledger_path"):
        value = config["output"][key]
        if not os.path.isabs(value):
            config["output"][key] = os.path.join(PROJECT_ROOT, value)
    return config


def build_stage_subsets(splits: dict) -> dict:
    """Stage 2's train/val/test subsets, filtered to late-only rows AFTER
    the shared split (see model_training.get_stage2_subset)."""
    X_train_late, ysev_train_late = get_stage2_subset(splits, "train")
    X_val_late, ysev_val_late = get_stage2_subset(splits, "val")
    X_test_late, ysev_test_late = get_stage2_subset(splits, "test")
    return {
        "X_train": X_train_late, "y_train": ysev_train_late,
        "X_val": X_val_late, "y_val": ysev_val_late,
        "X_test": X_test_late, "y_test": ysev_test_late,
    }


def run_stage(stage: str, subset: dict, config: dict, missing_flag_cols: list[str],
              checkpointer: Checkpointer, skip_tuning: bool) -> dict:
    label = "Stage 1 (classification: P(is_late))" if stage == "stage1" else \
        "Stage 2 (regression: E[lateness_minutes | is_late])"
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")

    print("\n--- Baseline models (default hyperparameters) ---")
    t0 = time.time()
    baseline_results, _ = fit_baseline_models(stage, subset, config, missing_flag_cols)
    print(baseline_results.round(4).to_string())
    print(f"[baseline fitting took {time.time() - t0:.1f}s]")

    if skip_tuning:
        print("\n--skip-tuning set: leaving tuned/test evaluation to a later run.")
        return {"baseline_results": baseline_results, "tuned_results": None, "final": None}

    print("\n--- Hyperparameter tuning (checkpointed; resumes if interrupted) ---")
    t0 = time.time()
    tuned_results = tune_all_models(stage, subset, config, missing_flag_cols, checkpointer)
    print(tuned_results.drop(columns="best_params").round(4).to_string())
    print(f"[tuning took {time.time() - t0:.1f}s]")

    print("\n--- Final one-time test-set evaluation ---")
    final = evaluate_best_on_test(stage, tuned_results, config, checkpointer,
                                    subset["X_test"], subset["y_test"])
    print(f"Selected model: {final['model']}")
    for k, v in final["test_metrics"].items():
        print(f"    {k:10s} {v:.4f}")

    return {"baseline_results": baseline_results, "tuned_results": tuned_results, "final": final}


def print_closing_summary(stage1_out: dict, stage2_out: dict, config: dict):
    print(f"\n{'=' * 70}\nClosing summary\n{'=' * 70}")
    if stage1_out["final"] is None or stage2_out["final"] is None:
        print("(tuning was skipped -- no final test comparison to show)")
        return

    rank1 = config["evaluation"]["classification"]["ranking_metric"]
    rank2 = config["evaluation"]["regression"]["ranking_metric"]
    rows = [
        {"stage": "Stage 1", "metric": rank1,
         "baseline": stage1_out["baseline_results"][rank1].max(),
         "tuned (validation)": stage1_out["tuned_results"][rank1].max(),
         "final (test)": stage1_out["final"]["test_metrics"][rank1]},
        {"stage": "Stage 2", "metric": rank2,
         "baseline": stage2_out["baseline_results"][rank2].max(),
         "tuned (validation)": stage2_out["tuned_results"][rank2].max(),
         "final (test)": stage2_out["final"]["test_metrics"][rank2]},
    ]
    summary = pd.DataFrame(rows).set_index(["stage", "metric"])
    print(summary.round(4).to_string())
    return summary


def run_ranked_score_demo(stage1_out: dict, subset1: dict, config: dict) -> pd.DataFrame | None:
    """Precision@K / Recall@K on the test set for the selected Stage 1
    model -- the recommended deployment framing (EDA Section 18.5/19/25):
    sort deliveries by risk score and let dispatch capacity decide how far
    down the list to work, rather than thresholding at 0.5.
    """
    if stage1_out["final"] is None:
        return None
    ranked_cfg = config["evaluation"]["ranked_score"]
    if not ranked_cfg["enabled"]:
        return None

    print(f"\n{'=' * 70}\nRanked-score evaluation on the test set ({stage1_out['final']['model']})\n{'=' * 70}")
    pipe = stage1_out["final"]["pipeline"]
    scores = pipe.predict_proba(subset1["X_test"])[:, 1]
    pk_table = precision_recall_at_k(subset1["y_test"].values, scores, ranked_cfg["k_fractions"])
    print(pk_table.round(4).to_string(index=False))
    return pk_table


def main():
    parser = argparse.ArgumentParser(description="MoveEasy delivery risk-scoring pipeline")
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "src", "config.yaml"))
    parser.add_argument("--skip-tuning", action="store_true",
                         help="fit baselines only; skip hyperparameter tuning and test evaluation")
    args = parser.parse_args()

    # 1. load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # 2. load + clean data
    df, missing_flag_cols = prepare_raw_data(config)
    feature_cols = get_feature_columns(config, missing_flag_cols)
    print(f"Prepared data: {df.shape[0]:,} rows, {len(feature_cols)} feature columns, "
          f"is_late positive rate {df[config['target']['classification_col']].mean():.2%}")

    # 3. split (shared across both stages)
    splits = split_data(df, feature_cols, config)
    stage2_subset = build_stage_subsets(splits)
    print(f"Split -> train: {len(splits['X_train']):,}  val: {len(splits['X_val']):,}  "
          f"test: {len(splits['X_test']):,}  (Stage 2 late-only train: {len(stage2_subset['X_train']):,})")

    checkpointer = Checkpointer(config)

    # 4. train + tune both stages
    stage1_out = run_stage("stage1", splits, config, missing_flag_cols, checkpointer, args.skip_tuning)
    stage2_out = run_stage("stage2", stage2_subset, config, missing_flag_cols, checkpointer, args.skip_tuning)

    # 5. ranked-score demonstration
    pk_table = run_ranked_score_demo(stage1_out, splits, config)

    # 6. closing summary
    summary = print_closing_summary(stage1_out, stage2_out, config)

    # persist results
    results_dir = config["output"]["results_dir"]
    stage1_out["baseline_results"].to_csv(f"{results_dir}/baseline_results_stage1.csv")
    stage2_out["baseline_results"].to_csv(f"{results_dir}/baseline_results_stage2.csv")
    if stage1_out["tuned_results"] is not None:
        stage1_out["tuned_results"].drop(columns="best_params").to_csv(f"{results_dir}/tuned_results_stage1.csv")
        stage2_out["tuned_results"].drop(columns="best_params").to_csv(f"{results_dir}/tuned_results_stage2.csv")
        json.dump({
            "stage1_model": stage1_out["final"]["model"], "stage1_test_metrics": stage1_out["final"]["test_metrics"],
            "stage2_model": stage2_out["final"]["model"], "stage2_test_metrics": stage2_out["final"]["test_metrics"],
        }, open(f"{results_dir}/final_test_results.json", "w"), indent=2)
    if pk_table is not None:
        pk_table.to_csv(f"{results_dir}/final_test_precision_recall_at_k.csv", index=False)
    if summary is not None:
        summary.to_csv(f"{results_dir}/closing_summary.csv")

    print(f"\nAll results saved to {results_dir}/")


if __name__ == "__main__":
    main()
