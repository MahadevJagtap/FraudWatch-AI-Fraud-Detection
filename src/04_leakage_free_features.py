"""
Leakage-Free Feature Engineering & Chronological Dataset Preparation
=====================================================================

Builds a strictly pre-authorization, leakage-free feature set from the raw PaySim CSV:
- Drops post-settlement balances: newbalanceOrig, newbalanceDest.
- Drops derived leakage residuals: errorBalanceOrig, errorBalanceDest.
- Drops system rule flag: isFlaggedFraud.
- Derives strictly pre-transaction features:
  - Temporal: hour_of_day, day_of_week
  - Amount: amount, log_amount
  - Account types: is_merchant, dest_is_customer
  - Initial balances: orig_old_balance, dest_old_balance, orig_drain_ratio, orig_is_zero, dest_is_zero
  - Transaction type one-hot: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER
- Enforces time-aware chronological split (no look-ahead shuffle):
  - Train:      step <= 576  (Days 1 to 24)
  - Validation: 576 < step <= 648 (Days 25 to 27)
  - Test (OOT): step > 648   (Days 28 to 31)

Run from project root:
    python src/04_leakage_free_features.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PAYSIM_TYPES: Tuple[str, ...] = (
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
)


def extract_pre_authorization_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features legitimately available at pre-authorization time."""
    feat = pd.DataFrame(index=df.index)

    # 1. Temporal signals
    feat["hour_of_day"] = (df["step"] % 24).astype("int8")
    feat["day_of_week"] = ((df["step"] // 24) % 7).astype("int8")

    # 2. Transaction types (one-hot)
    type_series = df["type"].astype(str)
    for t in PAYSIM_TYPES:
        feat[f"type_{t}"] = (type_series == t).astype("int8")

    # 3. Amount signals
    amount = df["amount"].astype("float32")
    feat["amount"] = amount
    feat["log_amount"] = np.log1p(np.maximum(0.0, amount)).astype("float32")

    # 4. Destination account signals (pre-authorization)
    dest_str = df["nameDest"].astype(str)
    is_merchant = dest_str.str.startswith("M")
    feat["is_merchant"] = is_merchant.astype("int8")
    feat["dest_is_customer"] = (~is_merchant).astype("int8")

    # 5. Pre-transaction balance signals (strictly prior to settlement)
    old_orig = df["oldbalanceOrg"].astype("float32")
    old_dest = df["oldbalanceDest"].astype("float32")
    feat["orig_old_balance"] = old_orig
    feat["dest_old_balance"] = old_dest
    feat["orig_is_zero"] = (old_orig == 0.0).astype("int8")
    feat["dest_is_zero"] = (old_dest == 0.0).astype("int8")

    # Drain ratio: fraction of origin funds being moved (capped to 10.0 to prevent inf)
    drain_ratio = np.clip(amount / (old_orig + 1.0), 0.0, 10.0).astype("float32")
    feat["orig_drain_ratio"] = drain_ratio

    # Keep step and target for splitting
    feat["step"] = df["step"].astype("int16")
    feat["isFraud"] = df["isFraud"].astype("int8")

    return feat


def process_dataset(raw_csv_path: str, out_dir: str, chunksize: int = 500_000) -> Dict[str, Any]:
    """Process raw CSV in chunks and write chronological Parquet partitions."""
    if not os.path.isfile(raw_csv_path):
        raise FileNotFoundError(f"Raw CSV not found at: {raw_csv_path}")

    os.makedirs(out_dir, exist_ok=True)
    print(f"Reading {raw_csv_path!r} in chunks of {chunksize:,} rows...")

    cols_to_use = [
        "step", "type", "amount", "nameOrig", "oldbalanceOrg",
        "nameDest", "oldbalanceDest", "isFraud"
    ]

    processed_chunks = []
    total_rows = 0
    t0 = time.perf_counter()

    for i, chunk in enumerate(pd.read_csv(raw_csv_path, usecols=cols_to_use, chunksize=chunksize)):
        clean_chunk = extract_pre_authorization_features(chunk)
        processed_chunks.append(clean_chunk)
        total_rows += len(chunk)
        print(f"   Processed chunk {i + 1}: cumulative {total_rows:,} rows")

    full_df = pd.concat(processed_chunks, axis=0, ignore_index=True)
    elapsed = time.perf_counter() - t0
    print(f"\nCompleted feature extraction in {elapsed:.2f}s: {len(full_df):,} total rows.")

    # Chronological Split:
    # Train:      Days 1-24  (step <= 576)
    # Validation: Days 25-27 (576 < step <= 648)
    # Test (OOT): Days 28-31 (step > 648)
    train_mask = full_df["step"] <= 576
    val_mask = (full_df["step"] > 576) & (full_df["step"] <= 648)
    test_mask = full_df["step"] > 648

    df_train = full_df[train_mask].reset_index(drop=True)
    df_val = full_df[val_mask].reset_index(drop=True)
    df_test = full_df[test_mask].reset_index(drop=True)

    print("\n" + "=" * 72)
    print("CHRONOLOGICAL PARTITIONS AUDIT")
    print("=" * 72)
    for name, split_df in [("Train (Days 1-24)", df_train), ("Validation (Days 25-27)", df_val), ("Test OOT (Days 28-31)", df_test)]:
        fraud_cnt = int(split_df["isFraud"].sum())
        total_cnt = len(split_df)
        rate = (fraud_cnt / total_cnt) * 100 if total_cnt > 0 else 0.0
        step_min = int(split_df["step"].min()) if total_cnt > 0 else 0
        step_max = int(split_df["step"].max()) if total_cnt > 0 else 0
        print(f"{name:<25}: rows={total_cnt:>10,} | frauds={fraud_cnt:>6,} ({rate:.4f}%) | steps={step_min}..{step_max}")
    print("=" * 72)

    # Save to Parquet
    p_train = os.path.join(out_dir, "leakage_free_train.parquet")
    p_val = os.path.join(out_dir, "leakage_free_val.parquet")
    p_test = os.path.join(out_dir, "leakage_free_test.parquet")

    df_train.to_parquet(p_train, index=False)
    df_val.to_parquet(p_val, index=False)
    df_test.to_parquet(p_test, index=False)

    print(f"\nSaved Train partition -> {p_train}")
    print(f"Saved Validation partition -> {p_val}")
    print(f"Saved Out-of-Time Test partition -> {p_test}")

    # Feature column names (excluding step and isFraud)
    feature_cols = [c for c in df_train.columns if c not in ("step", "isFraud")]
    print(f"\nEngineered Pre-Authorization Features ({len(feature_cols)}): {feature_cols}")

    return {
        "feature_names": feature_cols,
        "train_rows": len(df_train),
        "val_rows": len(df_val),
        "test_rows": len(df_test),
        "total_rows": total_rows,
    }


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_csv = os.path.join(project_root, "data", "raw", "PS_20174392719_1491204439457_log.csv")
    proc_dir = os.path.join(project_root, "data", "processed")
    process_dataset(raw_csv, proc_dir)
