"""
Step 02 - Feature Engineering
=============================

Loads the cleaned Parquet from ``data/processed/paysim_cleaned.parquet``,
decomposes the temporal signal, one-hot encodes ``type``, and writes
``X_features.parquet`` + ``y_target.parquet`` to ``data/processed/``.

Run from the project root:
    python src/02_feature_eng.py
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAYSIM_TYPES: Tuple[str, ...] = (
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mem_mb(df: pd.DataFrame) -> float:
    """Return the deep memory usage of a DataFrame in megabytes."""
    return float(df.memory_usage(deep=True).sum() / (1024 ** 2))


def _print_X_report(X: pd.DataFrame) -> None:
    """Print the final feature matrix shape, memory, dtypes and columns."""
    mem = _mem_mb(X)
    cols = list(X.columns)

    print("=" * 72)
    print(f"[FEATURE MATRIX X] shape={X.shape[0]:,} x {X.shape[1]} | "
          f"memory={mem:,.2f} MB")
    print("-" * 72)
    print(f"Columns ({len(cols)}):")
    for c in cols:
        print(f"   {c:<22} -> {X[c].dtype}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Step-level transformations
# ---------------------------------------------------------------------------
def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Decompose `step` (1..744) into hour-of-day / day-of-week."""
    df["hour_of_day"] = (df["step"] % 24).astype("int8")
    df["day_of_week"] = ((df["step"] // 24) % 7).astype("int8")
    return df


def _one_hot_encode_type(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode `type` into int8 columns (full PAYSIM_TYPES taxonomy)."""
    type_dummies = pd.get_dummies(df["type"], dtype="int8")
    type_dummies = type_dummies.reindex(
        columns=[f"type_{t}" for t in PAYSIM_TYPES], fill_value=0
    ).astype("int8")
    df = pd.concat([df.drop(columns=["type"]), type_dummies], axis=1)
    return df


def _assert_high_cardinality_dropped(df: pd.DataFrame) -> None:
    """Safety net: high-cardinality string IDs must not survive cleaning."""
    leaks = [c for c in ("nameOrig", "nameDest") if c in df.columns]
    if leaks:
        raise RuntimeError(
            f"High-cardinality identifiers still present after cleaning: {leaks}"
        )


def _drop_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are no longer needed once time features exist."""
    drop_cols = ["step", "isFlaggedFraud"]
    existing = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=existing)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def engineer_features(parquet_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Build the final feature matrix X and target vector y."""
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Parquet not found at: {parquet_path}")

    print(f"Loading cleaned parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"   loaded shape={df.shape[0]:,} x {df.shape[1]} | "
          f"memory={_mem_mb(df):,.2f} MB")

    _assert_high_cardinality_dropped(df)
    df = _add_temporal_features(df)
    df = _one_hot_encode_type(df)
    df = _drop_redundant_columns(df)

    if "isFraud" not in df.columns:
        raise KeyError("`isFraud` target column missing from cleaned dataset.")

    y = df["isFraud"].astype("int8")
    X = df.drop(columns=["isFraud"])

    _print_X_report(X)

    fraud_rate = float(y.mean()) * 100
    print(f"Target distribution: positives={int(y.sum()):,} | "
          f"negatives={int((1 - y).sum()):,} | fraud_rate={fraud_rate:.4f}%")

    return X, y


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc_dir = os.path.join(project_root, "data", "processed")
    in_path = os.path.join(proc_dir, "paysim_cleaned.parquet")

    X, y = engineer_features(in_path)

    out_X = os.path.join(proc_dir, "X_features.parquet")
    out_y = os.path.join(proc_dir, "y_target.parquet")
    X.to_parquet(out_X, index=False)
    y.to_frame("isFraud").to_parquet(out_y, index=False)
    print(f"\nSaved X -> {out_X}")
    print(f"Saved y -> {out_y}")
