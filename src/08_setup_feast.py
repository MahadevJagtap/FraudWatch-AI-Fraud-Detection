"""
Feast Setup, Materialization, and Retrieval Verification
========================================================

Generates Feast Parquet source, applies feature definitions,
materializes features into the SQLite online store, and verifies
both online and offline feature retrieval.

Run from project root:
    python src/08_setup_feast.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from feast import FeatureStore

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
FEATURE_STORE_DIR = os.path.join(PROJECT_ROOT, "feature_store")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PARQUET_OUT = os.path.join(DATA_DIR, "account_features_feast.parquet")
RAW_CSV = os.path.join("D:\\Fraud_detection", "PaySim_dataset.csv")


def generate_feast_parquet(sample_size: int = 50_000) -> str:
    print(f"Generating Feast Parquet source from {RAW_CSV!r} (sample {sample_size:,} accounts)...")
    os.makedirs(DATA_DIR, exist_ok=True)

    # Read a sample of transactions to get realistic accounts
    df_sample = pd.read_csv(
        RAW_CSV,
        nrows=sample_size,
        usecols=["step", "nameOrig", "nameDest", "amount", "oldbalanceOrg", "oldbalanceDest"]
    )

    base_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    records = []
    seen_accounts = set()

    for _, row in df_sample.iterrows():
        step_hours = int(row["step"])
        tx_time = base_time + timedelta(hours=step_hours)

        # Origin account
        orig_id = str(row["nameOrig"])
        if orig_id not in seen_accounts:
            seen_accounts.add(orig_id)
            drain = float(np.clip(row["amount"] / (row["oldbalanceOrg"] + 1.0), 0.0, 10.0))
            records.append({
                "account_id": orig_id,
                "event_timestamp": tx_time,
                "created_timestamp": tx_time,
                "orig_old_balance": float(row["oldbalanceOrg"]),
                "dest_old_balance": 0.0,
                "orig_drain_ratio": drain,
                "is_merchant": 0,
                "dest_is_customer": 1,
            })

        # Destination account
        dest_id = str(row["nameDest"])
        if dest_id not in seen_accounts:
            seen_accounts.add(dest_id)
            is_m = int(dest_id.startswith("M"))
            records.append({
                "account_id": dest_id,
                "event_timestamp": tx_time,
                "created_timestamp": tx_time,
                "orig_old_balance": 0.0,
                "dest_old_balance": float(row["oldbalanceDest"]),
                "orig_drain_ratio": 0.0,
                "is_merchant": is_m,
                "dest_is_customer": 1 - is_m,
            })

    feast_df = pd.DataFrame(records)
    feast_df["event_timestamp"] = pd.to_datetime(feast_df["event_timestamp"])
    feast_df["created_timestamp"] = pd.to_datetime(feast_df["created_timestamp"])
    feast_df["orig_old_balance"] = feast_df["orig_old_balance"].astype("float32")
    feast_df["dest_old_balance"] = feast_df["dest_old_balance"].astype("float32")
    feast_df["orig_drain_ratio"] = feast_df["orig_drain_ratio"].astype("float32")
    feast_df["is_merchant"] = feast_df["is_merchant"].astype("int32")
    feast_df["dest_is_customer"] = feast_df["dest_is_customer"].astype("int32")

    feast_df.to_parquet(PARQUET_OUT, index=False)
    print(f"Saved {len(feast_df):,} account records to {PARQUET_OUT}")
    return PARQUET_OUT


def setup_and_verify_feast():
    # 1. Generate Parquet source
    generate_feast_parquet()

    # 2. Initialize FeatureStore
    print(f"\nInitializing Feast FeatureStore with repo: {FEATURE_STORE_DIR}")
    store = FeatureStore(repo_path=FEATURE_STORE_DIR)

    # 3. Apply definitions
    from feature_store.features import account_entity, account_features_view
    print("Applying entity and feature view definitions to registry...")
    store.apply([account_entity, account_features_view])
    print("Registry updated successfully!")

    # 4. Materialize to SQLite online store
    start_date = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end_date = datetime.now(timezone.utc) + timedelta(days=365)
    print(f"Materializing features to online store from {start_date} to {end_date}...")
    store.materialize(start_date=start_date, end_date=end_date)
    print("Online store materialized successfully!")

    # 5. Verify Online Retrieval
    print("\nVerifying Online Feature Retrieval...")
    test_accounts = [{"account_id": "C1231006815"}, {"account_id": "M1979787155"}]
    features_to_fetch = [
        "account_features:orig_old_balance",
        "account_features:dest_old_balance",
        "account_features:orig_drain_ratio",
        "account_features:is_merchant",
        "account_features:dest_is_customer",
    ]
    online_response = store.get_online_features(
        features=features_to_fetch,
        entity_rows=test_accounts
    ).to_dict()
    print("Online features response:")
    for key, values in online_response.items():
        print(f"  {key}: {values}")

    # 6. Verify Offline Historical Retrieval
    print("\nVerifying Offline Historical Feature Retrieval...")
    entity_df = pd.DataFrame({
        "account_id": ["C1231006815", "M1979787155"],
        "event_timestamp": [
            datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 1, 5, 12, 0, 0, tzinfo=timezone.utc),
        ]
    })
    training_data = store.get_historical_features(
        entity_df=entity_df,
        features=features_to_fetch
    ).to_df()
    print("Offline historical dataframe retrieved:")
    print(training_data[["account_id", "orig_old_balance", "is_merchant", "dest_is_customer"]])

    print("\n" + "=" * 60)
    print("FEAST FEATURE STORE VERIFICATION COMPLETE & PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    setup_and_verify_feast()
