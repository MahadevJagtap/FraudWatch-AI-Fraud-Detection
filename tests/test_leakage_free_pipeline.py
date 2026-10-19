"""
Automated tests for the leakage-free pre-authorization pipeline.
Milestone 2 test suite.
"""

from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

import importlib
leakage_module = importlib.import_module("src.04_leakage_free_features")
extract_pre_authorization_features = leakage_module.extract_pre_authorization_features
PAYSIM_TYPES = leakage_module.PAYSIM_TYPES


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "leakage_free_xgb_model.json")
METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "leakage_free_metadata.json")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "leakage_free_evaluation_report.json")


def test_leakage_free_artifacts_exist():
    assert os.path.isfile(MODEL_PATH), f"Model missing: {MODEL_PATH}"
    assert os.path.getsize(MODEL_PATH) > 0, "Model file is empty"
    assert os.path.isfile(METADATA_PATH), f"Metadata missing: {METADATA_PATH}"

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["model_name"] == "leakage_free_xgb_model"
    assert "features" in meta
    assert len(meta["features"]) == 16

    # Verify no post-settlement leakage in schema
    leaky_columns = {"newbalanceOrig", "newbalanceDest", "errorBalanceOrig", "errorBalanceDest", "isFlaggedFraud"}
    for col in meta["features"]:
        assert col not in leaky_columns, f"Leaky column {col} found in feature schema!"


def test_leakage_free_model_inference():
    clf = XGBClassifier()
    clf.load_model(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    features = meta["features"]

    # Test Case 1: High-risk account-draining transfer
    # 100,000 transferred out of 100,000 balance -> drain_ratio = 1.0
    fraud_tx = {
        "hour_of_day": 12,
        "day_of_week": 2,
        "type_CASH_IN": 0,
        "type_CASH_OUT": 0,
        "type_DEBIT": 0,
        "type_PAYMENT": 0,
        "type_TRANSFER": 1,
        "amount": 100000.0,
        "log_amount": float(np.log1p(100000.0)),
        "is_merchant": 0,
        "dest_is_customer": 1,
        "orig_old_balance": 100000.0,
        "dest_old_balance": 0.0,
        "orig_is_zero": 0,
        "dest_is_zero": 1,
        "orig_drain_ratio": 100000.0 / 100001.0,
    }
    row_fraud = np.array([[fraud_tx[col] for col in features]], dtype=np.float32)
    proba_fraud = float(clf.predict_proba(row_fraud)[0, 1])
    assert proba_fraud > 0.50, f"Expected high fraud probability, got {proba_fraud}"

    # Test Case 2: Low-risk merchant payment
    # 50.0 payment from 500.0 balance to merchant -> drain_ratio = 0.10
    legit_tx = {
        "hour_of_day": 14,
        "day_of_week": 3,
        "type_CASH_IN": 0,
        "type_CASH_OUT": 0,
        "type_DEBIT": 0,
        "type_PAYMENT": 1,
        "type_TRANSFER": 0,
        "amount": 50.0,
        "log_amount": float(np.log1p(50.0)),
        "is_merchant": 1,
        "dest_is_customer": 0,
        "orig_old_balance": 500.0,
        "dest_old_balance": 0.0,
        "orig_is_zero": 0,
        "dest_is_zero": 1,
        "orig_drain_ratio": 50.0 / 501.0,
    }
    row_legit = np.array([[legit_tx[col] for col in features]], dtype=np.float32)
    proba_legit = float(clf.predict_proba(row_legit)[0, 1])
    assert proba_legit < 0.20, f"Expected low fraud probability, got {proba_legit}"


def test_extract_pre_authorization_features_function():
    sample_raw = pd.DataFrame([
        {
            "step": 49,
            "type": "TRANSFER",
            "amount": 250000.0,
            "nameOrig": "C12345",
            "oldbalanceOrg": 250000.0,
            "newbalanceOrig": 0.0,  # post-settlement, should be excluded
            "nameDest": "C67890",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,  # post-settlement, should be excluded
            "isFraud": 1,
            "isFlaggedFraud": 0,
        },
        {
            "step": 12,
            "type": "PAYMENT",
            "amount": 45.5,
            "nameOrig": "C99999",
            "oldbalanceOrg": 1000.0,
            "newbalanceOrig": 954.5,
            "nameDest": "M11111",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
            "isFraud": 0,
            "isFlaggedFraud": 0,
        }
    ])

    extracted = extract_pre_authorization_features(sample_raw)

    # 1. Verify shape and row count
    assert len(extracted) == 2

    # 2. Verify leaky columns are not extracted
    for forbidden in ["newbalanceOrig", "newbalanceDest", "errorBalanceOrig", "errorBalanceDest", "isFlaggedFraud"]:
        assert forbidden not in extracted.columns

    # 3. Verify temporal extraction
    assert extracted.loc[0, "hour_of_day"] == 49 % 24  # 1
    assert extracted.loc[0, "day_of_week"] == (49 // 24) % 7  # 2

    # 4. Verify merchant vs customer destination flag
    assert extracted.loc[0, "is_merchant"] == 0
    assert extracted.loc[0, "dest_is_customer"] == 1
    assert extracted.loc[1, "is_merchant"] == 1
    assert extracted.loc[1, "dest_is_customer"] == 0

    # 5. Verify drain ratio
    assert abs(extracted.loc[0, "orig_drain_ratio"] - (250000.0 / 250001.0)) < 1e-4


def test_chronological_split_integrity():
    """Verify chronological split partitions have strict temporal boundaries and zero step overlap."""
    assert os.path.isfile(METADATA_PATH), f"Metadata missing: {METADATA_PATH}"
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    splits = meta["chronological_splits"]
    train_split = splits["train"]
    val_split = splits["val"]
    test_split = splits["test"]

    # Verify step ranges
    assert train_split["step_range"] == [1, 576]
    assert val_split["step_range"] == [577, 648]
    assert test_split["step_range"] == [649, 743]

    train_steps = set(range(train_split["step_range"][0], train_split["step_range"][1] + 1))
    val_steps = set(range(val_split["step_range"][0], val_split["step_range"][1] + 1))
    test_steps = set(range(test_split["step_range"][0], test_split["step_range"][1] + 1))

    # Strict zero overlap
    assert len(train_steps & val_steps) == 0, "Train and Val steps overlap!"
    assert len(val_steps & test_steps) == 0, "Val and Test steps overlap!"
    assert len(train_steps & test_steps) == 0, "Train and Test steps overlap!"

    # Strict monotonic temporal ordering
    assert max(train_steps) < min(val_steps), "Train split is not strictly before validation split!"
    assert max(val_steps) < min(test_steps), "Validation split is not strictly before test split!"

    # Verify recorded row counts match PaySim 6,362,620 total
    total_split_rows = train_split["rows"] + val_split["rows"] + test_split["rows"]
    assert total_split_rows == 6362620, f"Expected 6,362,620 rows total, got {total_split_rows}"

    # Also verify on-disk partition files if present
    val_parquet = os.path.join(PROJECT_ROOT, "data", "processed", "leakage_free_val.parquet")
    test_parquet = os.path.join(PROJECT_ROOT, "data", "processed", "leakage_free_test.parquet")
    if os.path.isfile(val_parquet) and os.path.isfile(test_parquet):
        df_val_step = pd.read_parquet(val_parquet, columns=["step"])
        df_test_step = pd.read_parquet(test_parquet, columns=["step"])
        assert int(df_val_step["step"].min()) >= 577
        assert int(df_val_step["step"].max()) <= 648
        assert int(df_test_step["step"].min()) >= 649
        assert int(df_test_step["step"].max()) <= 743


def test_leakage_free_schema_enforcement():
    """Verify that the 16 pre-authorization feature contract is strictly enforced."""
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    features = meta["features"]
    assert len(features) == 16, f"Expected exactly 16 features, got {len(features)}"

    expected_features = [
        "hour_of_day", "day_of_week",
        "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
        "amount", "log_amount",
        "is_merchant", "dest_is_customer",
        "orig_old_balance", "dest_old_balance",
        "orig_is_zero", "dest_is_zero",
        "orig_drain_ratio",
    ]
    assert features == expected_features, "Feature list order or names mismatch the 16-feature contract"

    # Ensure prohibited post-settlement terms are not present in any feature substring
    for f in features:
        assert "newbalance" not in f.lower(), f"Prohibited feature substring 'newbalance' in {f}"
        assert "errorbalance" not in f.lower(), f"Prohibited feature substring 'errorbalance' in {f}"


def test_xgboost_save_reload_consistency(tmp_path):
    """Verify saving and re-loading XGBoost model produces numerically identical probabilities."""
    clf_orig = XGBClassifier()
    clf_orig.load_model(MODEL_PATH)

    temp_model_path = tmp_path / "temp_xgb_check.json"
    clf_orig.save_model(str(temp_model_path))

    clf_reloaded = XGBClassifier()
    clf_reloaded.load_model(str(temp_model_path))

    # Evaluate on synthetic batch of 10 rows
    np.random.seed(123)
    dummy_input = np.random.randn(10, 16).astype(np.float32)

    p_orig = clf_orig.predict_proba(dummy_input)
    p_reloaded = clf_reloaded.predict_proba(dummy_input)

    assert np.allclose(p_orig, p_reloaded, atol=1e-6), "Reloaded XGBoost predictions diverged!"
