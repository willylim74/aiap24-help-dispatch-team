"""
Data preparation layer.

Two categories of logic, kept deliberately separate:

1. Row-level functions (`clean_structural_errors`, `remove_duplicates`,
   `add_missingness_indicators`, `engineer_targets`) run ONCE on the whole
   dataset, before any train/val/test split. Everything here is
   deterministic and statistics-free -- fixing a typo or dropping an exact
   duplicate row doesn't depend on which fold a row ends up in, so doing it
   before the split carries no leakage risk.

2. `build_preprocessing_pipeline` returns an unfitted sklearn Pipeline
   (imputation, outlier flagging, feature engineering, scaling/encoding).
   This part DOES involve statistics learned from data (imputation medians,
   IQR fences, encoder categories), so it is only ever fit on a training
   fold -- see model_training.py, which builds a fresh instance per model
   and calls .fit() only on X_train.

Every column name, cleaning rule, and imputation/outlier/FE choice below is
read from config.yaml rather than hardcoded, per Section 11-14 of eda.ipynb.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================================
# Row-level cleaning (run once, before the split)
# ============================================================================

def load_raw_data(config: dict) -> pd.DataFrame:
    """Load the raw CSV and drop identifier columns not used as features."""
    path = config["data"]["raw_path"]
    df = pd.read_csv(path)
    id_cols = [c for c in config["data"]["id_columns_to_drop"] if c in df.columns]
    return df.drop(columns=id_cols)


def clean_structural_errors(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Fix casing/typo inconsistencies and impossible values. Deterministic
    string/value corrections only -- no statistics are learned here, so this
    is safe to run on the full dataset before splitting.
    """
    df = df.copy()
    cleaning_cfg = config["cleaning"]

    for col in cleaning_cfg["datetime_columns"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    for col, mapping in cleaning_cfg["casing_maps"].items():
        df[col] = df[col].astype(str).str.strip().str.lower().map(mapping)

    for col in cleaning_cfg.get("lowercase_strip_columns", []):
        df[col] = df[col].astype(str).str.strip().str.lower()

    for col in cleaning_cfg.get("strip_only_columns", []):
        df[col] = df[col].astype(str).str.strip()

    for col in cleaning_cfg["numeric_as_text_columns"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan

    postal_col = cleaning_cfg["postal_code_column"]
    df[postal_col] = df[postal_col].apply(lambda v: str(int(v)) if pd.notna(v) else np.nan)

    if cleaning_cfg.get("drop_impossible_timestamp_rows", True):
        delivery_col = config["target"]["delivery_datetime_col"]
        pickup_col = "pickup_datetime"
        df = df[df[delivery_col] >= df[pickup_col]].reset_index(drop=True)

    return df


def remove_duplicates(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Drop exact duplicate rows, if configured to."""
    if config["cleaning"].get("drop_exact_duplicates", True):
        df = df.drop_duplicates().reset_index(drop=True)
    return df


def add_missingness_indicators(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[str]]:
    """Add a `<col>_was_missing` binary flag for every column configured as
    a missingness-indicator source, BEFORE those columns get imputed later
    in the pipeline. Returns the frame and the list of new flag columns.
    """
    df = df.copy()
    flag_cols = []
    for col in config["features"]["missing_indicator_source_columns"]:
        flag_col = f"{col}_was_missing"
        df[flag_col] = df[col].isna().astype(int)
        flag_cols.append(flag_col)
    return df, flag_cols


def engineer_targets(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Derive the two hurdle-model targets from the raw timestamps:
    lateness_minutes (continuous severity) and is_late (binary, thresholded
    at 0 minutes). See eda.ipynb Section 1 / 10 for the rationale.
    """
    df = df.copy()
    target_cfg = config["target"]
    delivery_col = target_cfg["delivery_datetime_col"]
    promised_col = target_cfg["promised_datetime_col"]

    df[target_cfg["severity_col"]] = (
        df[delivery_col] - df[promised_col]
    ).dt.total_seconds() / 60
    df[target_cfg["classification_col"]] = (df[target_cfg["severity_col"]] > 0).astype(int)
    return df


def prepare_raw_data(config: dict) -> tuple[pd.DataFrame, list[str]]:
    """Run the full row-level preparation sequence in order and return the
    cleaned frame plus the list of missingness-indicator columns it created
    (needed downstream to build the feature column list).
    """
    df = load_raw_data(config)
    df = clean_structural_errors(df, config)
    df = remove_duplicates(df, config)
    df, missing_flag_cols = add_missingness_indicators(df, config)
    df = engineer_targets(df, config)
    return df, missing_flag_cols


def get_feature_columns(config: dict, missing_flag_cols: list[str]) -> list[str]:
    """The full input feature-column list fed into the preprocessing
    pipeline: dispatch-time-safe datetime inputs, base numeric columns,
    categorical columns, and missingness flags. Deliberately excludes
    delivery_datetime (the outcome timestamp the target is derived from) and
    the identifier columns already dropped in load_raw_data.
    """
    f = config["features"]
    return (
        f["datetime_input_columns"]
        + f["numeric_base_columns"]
        + f["categorical_columns"]
        + missing_flag_cols
    )


# ============================================================================
# Preprocessing pipeline (fit only on a training fold -- see model_training.py)
# ============================================================================

class DatetimeFeatureTransformer(BaseEstimator, TransformerMixin):
    """Derives dispatch-time-available numeric features from the
    booking/pickup/promised timestamps. Stateless (every output depends
    only on that row's own inputs), so fit() is a no-op and there is no
    leakage risk regardless of which fold this is fit or transformed on.
    """

    def __init__(self, booking_col="booking_datetime", pickup_col="pickup_datetime",
                 promised_col="promised_delivery_datetime"):
        self.booking_col = booking_col
        self.pickup_col = pickup_col
        self.promised_col = promised_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        pickup = X[self.pickup_col]
        X["pickup_hour"] = pickup.dt.hour
        X["pickup_dayofweek"] = pickup.dt.dayofweek
        X["pickup_is_weekend"] = (pickup.dt.dayofweek >= 5).astype(int)
        X["promised_window_minutes"] = (X[self.promised_col] - pickup).dt.total_seconds() / 60
        X["booking_lead_minutes"] = (pickup - X[self.booking_col]).dt.total_seconds() / 60
        return X

    def get_feature_names_out(self, input_features=None):
        extra = ["pickup_hour", "pickup_dayofweek", "pickup_is_weekend",
                 "promised_window_minutes", "booking_lead_minutes"]
        base = list(input_features) if input_features is not None else []
        return np.array(base + extra)


class OutlierFlagTransformer(BaseEstimator, TransformerMixin):
    """Adds a `<col>_is_outlier` binary flag per configured numeric column,
    using IQR fences learned only from whatever data this is fit on (the
    training fold). Original values are left untouched: per eda.ipynb
    Section 8, outliers here are informative and should be kept and flagged,
    not capped or removed. A missing value is never flagged as an outlier.
    """

    def __init__(self, numeric_cols=None, iqr_multiplier=1.5):
        self.numeric_cols = numeric_cols or []
        self.iqr_multiplier = iqr_multiplier

    def fit(self, X, y=None):
        self.fences_ = {}
        for c in self.numeric_cols:
            q1, q3 = np.nanpercentile(X[c].astype(float), [25, 75])
            iqr = q3 - q1
            self.fences_[c] = (q1 - self.iqr_multiplier * iqr, q3 + self.iqr_multiplier * iqr)
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.numeric_cols:
            lo, hi = self.fences_[c]
            vals = X[c].astype(float)
            X[f"{c}_is_outlier"] = ((vals < lo) | (vals > hi)).fillna(False).astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else []
        extra = [f"{c}_is_outlier" for c in self.numeric_cols]
        return np.array(base + extra)


_SCALERS = {"standard": StandardScaler}
_ENCODERS = {"onehot": OneHotEncoder}


def build_preprocessing_pipeline(config: dict, missing_flag_cols: list[str]) -> Pipeline:
    """Build a fresh, UNFITTED preprocessing pipeline. Call this once per
    model/tuning run and fit it only on that run's training fold -- never
    share a fitted instance across folds or reuse across models, or
    imputation/scaling statistics will leak between them.
    """
    numeric_base_cols = config["features"]["numeric_base_columns"]
    categorical_cols = config["features"]["categorical_columns"]
    engineered_cols = config["feature_engineering"]["temporal"]["derived_features"]
    outlier_flag_cols = [f"{c}_is_outlier" for c in config["outlier_handling"]["apply_to_columns"]]

    numeric_for_scaling = numeric_base_cols + engineered_cols + missing_flag_cols + outlier_flag_cols

    impute_cfg = config["missing_value_imputation"]
    scaler_cls = _SCALERS[config["preprocessing"]["numeric_scaler"]]
    encoder_cls = _ENCODERS[config["preprocessing"]["categorical_encoder"]]

    return Pipeline([
        ("datetime_fe", DatetimeFeatureTransformer()),
        ("outlier_flags", OutlierFlagTransformer(
            numeric_cols=config["outlier_handling"]["apply_to_columns"],
            iqr_multiplier=config["outlier_handling"]["iqr_multiplier"],
        )),
        ("coltrans", ColumnTransformer([
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy=impute_cfg["numeric_strategy"])),
                ("scale", scaler_cls()),
            ]), numeric_for_scaling),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy=impute_cfg["categorical_strategy"],
                                          fill_value=impute_cfg["categorical_fill_value"])),
                ("encode", encoder_cls(handle_unknown=config["preprocessing"]["categorical_encoder_handle_unknown"])),
            ]), categorical_cols),
        ])),
    ])


# ============================================================================
# Embedded sanity checks -- run with `python src/data_preparation.py`
# ============================================================================

def _run_self_tests():
    tiny_config = {
        "cleaning": {
            "datetime_columns": ["booking_datetime", "pickup_datetime",
                                  "promised_delivery_datetime", "delivery_datetime"],
            "casing_maps": {"branch": {"west": "West", "east": "East"}},
            "lowercase_strip_columns": ["vehicle_type"],
            "strip_only_columns": ["delivery_priority"],
            "numeric_as_text_columns": ["parcel_weight_kg"],
            "postal_code_column": "recipient_postal_code",
            "drop_impossible_timestamp_rows": True,
            "drop_exact_duplicates": True,
        },
        "target": {
            "delivery_datetime_col": "delivery_datetime",
            "promised_datetime_col": "promised_delivery_datetime",
            "severity_col": "lateness_minutes",
            "classification_col": "is_late",
        },
        "features": {"missing_indicator_source_columns": ["parcel_weight_kg"]},
    }

    raw = pd.DataFrame({
        "branch": ["west", "west", "east"],
        "vehicle_type": [" Van", "bike", "bike"],
        "delivery_priority": [" Standard", "VIP", "VIP"],
        "parcel_weight_kg": ["2.5", "-1", "2.5"],
        "recipient_postal_code": ["1", "2", "2"],
        "booking_datetime": ["2026-01-01 08:00"] * 3,
        "pickup_datetime": ["2026-01-01 09:00", "2026-01-01 09:00", "2026-01-01 09:00"],
        "promised_delivery_datetime": ["2026-01-01 11:00"] * 3,
        # row 0: delivered before pickup (impossible, must be dropped);
        # row 1: legitimate late delivery (west); row 2: on-time delivery (east)
        "delivery_datetime": ["2026-01-01 08:30", "2026-01-01 12:00", "2026-01-01 10:30"],
    })

    cleaned = clean_structural_errors(raw, tiny_config)
    assert len(cleaned) == 2, f"expected the impossible-timestamp row dropped, got {len(cleaned)} rows"
    assert set(cleaned["branch"]) == {"West", "East"}, "casing map did not normalize branch"
    assert cleaned["parcel_weight_kg"].isna().sum() == 1, "the -1 sentinel should become NaN"

    deduped = remove_duplicates(cleaned, tiny_config)
    assert len(deduped) == 2, f"expected no duplicates among the surviving rows, got {len(deduped)} rows"

    with_flags, flag_cols = add_missingness_indicators(cleaned, tiny_config)
    assert flag_cols == ["parcel_weight_kg_was_missing"]
    assert with_flags["parcel_weight_kg_was_missing"].tolist() == [1, 0]

    targeted = engineer_targets(with_flags, tiny_config)
    assert targeted["is_late"].tolist() == [1, 0]
    assert abs(targeted.loc[0, "lateness_minutes"] - 60.0) < 1e-6

    dt_out = DatetimeFeatureTransformer().fit_transform(pd.DataFrame({
        "booking_datetime": pd.to_datetime(["2026-01-05 08:00"]),  # a Monday
        "pickup_datetime": pd.to_datetime(["2026-01-05 09:00"]),
        "promised_delivery_datetime": pd.to_datetime(["2026-01-05 11:00"]),
    }))
    assert dt_out["pickup_hour"].iloc[0] == 9
    assert dt_out["pickup_is_weekend"].iloc[0] == 0
    assert dt_out["promised_window_minutes"].iloc[0] == 120
    assert dt_out["booking_lead_minutes"].iloc[0] == 60

    outlier_check = pd.DataFrame({"distance_km": [3.0, 4.0, 5.0, 500.0]})
    of = OutlierFlagTransformer(numeric_cols=["distance_km"]).fit(outlier_check)
    flagged = of.transform(outlier_check)
    assert flagged["distance_km_is_outlier"].tolist() == [0, 0, 0, 1]

    print("data_preparation.py: all self-tests passed.")


if __name__ == "__main__":
    _run_self_tests()
