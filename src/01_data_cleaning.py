"""
Step 01 - Data Cleaning & Memory Optimization
============================================

Reads the raw PaySim CSV from ``data/raw/``, produces a leakage-safe,
memory-optimized Parquet at ``data/processed/paysim_cleaned.parquet``.

Run from the project root:
    python src/01_data_cleaning.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def _mem_mb(df: pd.DataFrame) -> float:
    """Return the deep memory usage of a DataFrame in megabytes."""
    return float(df.memory_usage(deep=True).sum() / (1024 ** 2))


def _print_report(label: str, df: pd.DataFrame) -> None:
    """Print a compact structural / memory report for the given DataFrame."""
    rows, cols = df.shape
    mem = _mem_mb(df)
    fraud_counts = (
        df["isFraud"].value_counts(dropna=False).sort_index()
        if "isFraud" in df.columns else pd.Series(dtype=int)
    )
    fraud_pct = (
        (fraud_counts / fraud_counts.sum() * 100).round(4)
        if not fraud_counts.empty else fraud_counts
    )

    print("=" * 72)
    print(f"[{label}] shape={rows:,} x {cols} | memory={mem:,.2f} MB")
    print("-" * 72)
    if not fraud_counts.empty:
        print("isFraud distribution:")
        for cls, n in fraud_counts.items():
            pct = fraud_pct.get(cls, 0.0)
            print(f"   class={cls!s:>4}  count={n:>12,}  share={pct:>7.4f}%")
    else:
        print("isFraud distribution: <column not present>")
    print("dtypes:")
    for col, dt in df.dtypes.items():
        print(f"   {col:<18} -> {dt}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Step-level transformations
# ---------------------------------------------------------------------------
BALANCE_COLS = (
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
)


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe features BEFORE dropping balance columns."""
    # Fraudsters frequently break conservation of money. The residuals below
    # are exactly the kind of signal that survives the *moment a transaction
    # is initiated* (i.e. available pre-authorization in a real system),
    # unlike the raw post-balance figures which only exist post-settlement.
    df["errorBalanceOrig"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

    # Merchants (IDs starting with 'M') have zero balances in PaySim by design.
    is_merchant = df["nameDest"].astype(str).str.startswith("M")
    df["is_merchant"] = is_merchant.astype("int8")

    return df


def _drop_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the raw balance columns and free-text identifiers that leak."""
    drop_cols = list(BALANCE_COLS) + ["nameOrig", "nameDest"]
    existing = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=existing)


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Compress in-memory dtypes without losing precision for our purposes."""
    for col in df.select_dtypes(include=["int64", "int32"]).columns:
        col_min, col_max = df[col].min(), df[col].max()
        if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
            df[col] = df[col].astype("int8")
        elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
            df[col] = df[col].astype("int16")
        elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
            df[col] = df[col].astype("int32")

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")

    if "type" in df.columns and df["type"].dtype != "category":
        df["type"] = df["type"].astype("category")

    return df


def _handle_missing_and_duplicates(
    df: pd.DataFrame, label: str, drop: bool
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Audit NaNs / duplicates on raw float64 data and (optionally) drop real dups."""
    na_counts = df.isna().sum()
    n_dup = int(df.duplicated().sum())
    report = {
        "na_total": int(na_counts.sum()),
        "na_per_col": {c: int(n) for c, n in na_counts.items() if n > 0},
        "duplicates": n_dup,
    }

    print(f"[{label}] Missing-value audit (per column, only non-zero shown):")
    if report["na_per_col"]:
        for c, n in report["na_per_col"].items():
            print(f"   {c:<22} NaN count = {n:,}")
    else:
        print("   <none>")

    print(f"[{label}] Fully-duplicated rows: {n_dup:,}")

    if drop and n_dup:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"[{label}] Dropped {n_dup:,} duplicated rows.")

    return df, report


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def clean_paysim(csv_path: str, chunksize: int = 250_000) -> pd.DataFrame:
    """Clean and optimize the PaySim dataset end-to-end."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    print(f"Reading {csv_path!r} in chunks of {chunksize:,} rows ...")
    raw_frames = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        raw_frames.append(chunk)
    raw = pd.concat(raw_frames, axis=0, ignore_index=True)

    before_mem = _mem_mb(raw)
    before_shape = raw.shape
    print("\n>>> BEFORE cleaning")
    print(f"    shape={before_shape[0]:,} x {before_shape[1]} | "
          f"memory={before_mem:,.2f} MB")

    raw, audit_report = _handle_missing_and_duplicates(
        raw, label="BEFORE cleaning", drop=True
    )

    raw = _engineer_features(raw)
    cleaned = _drop_leakage(raw)
    cleaned = _downcast(cleaned)
    cleaned, _ = _handle_missing_and_duplicates(
        cleaned, label="AFTER cleaning", drop=False
    )

    print("\n>>> AFTER cleaning")
    _print_report("AFTER cleaning", cleaned)

    after_mem = _mem_mb(cleaned)
    savings = 1.0 - (after_mem / before_mem if before_mem else 0.0)
    print(f"\nMemory saved: {savings * 100:,.2f}% "
          f"({before_mem - after_mem:,.2f} MB)")
    print(f"Audit summary: missing_values={audit_report['na_total']:,} | "
          f"duplicates_dropped={audit_report['duplicates']:,}")

    return cleaned


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(project_root, "data", "raw")
    proc_dir = os.path.join(project_root, "data", "processed")

    # Default: first CSV found in data/raw/.
    csv_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(".csv")]
    if not csv_files:
        print(f"No CSV files found in {raw_dir!r}.")
        sys.exit(1)

    csv_path = os.path.join(raw_dir, csv_files[0])
    df = clean_paysim(csv_path)

    out_path = os.path.join(proc_dir, "paysim_cleaned.parquet")
    df.to_parquet(out_path, index=False)
    print(f"\nSaved cleaned dataset -> {out_path}")
