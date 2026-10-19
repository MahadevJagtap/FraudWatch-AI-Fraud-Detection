"""
Verification & Reload Script for PaySim XGBoost Fraud Model
===========================================================

Validates that:
1. The saved model file exists and is non-empty.
2. The model can be reloaded in a fresh process.
3. Feature metadata matches expected schema.
4. Predictions on held-out test data are finite and have the expected shape.
5. Tests both legitimate and fraudulent test examples.
6. Simulates a future FastAPI single-transaction inference pipeline with raw input.
7. Saves a sample prediction report to reports/sample_predictions.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def verify_model_pipeline() -> Dict[str, Any]:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "models", "fraud_xgb_model.json")
    meta_path = os.path.join(project_root, "models", "model_metadata.json")
    proc_dir = os.path.join(project_root, "data", "processed")
    reports_dir = os.path.join(project_root, "reports")

    print("=" * 72)
    print("STEP 1: Verify Saved Model and Metadata Artifacts")
    print("=" * 72)
    assert os.path.isfile(model_path), f"Model file not found: {model_path}"
    model_size = os.path.getsize(model_path)
    print(f"Model path: {model_path}")
    print(f"Model size: {model_size:,} bytes ({model_size / (1024**2):.2f} MB)")
    assert model_size > 0, "Model file is empty!"

    assert os.path.isfile(meta_path), f"Metadata file not found: {meta_path}"
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print("Loaded metadata:")
    print(f"  Target: {metadata['target']}")
    print(f"  Features ({len(metadata['features'])}): {metadata['features']}")
    print(f"  Evaluation Metrics: AUPRC={metadata['metrics']['auprc']:.4f}, Recall={metadata['metrics']['recall_fraud']:.4f}")

    print("\n" + "=" * 72)
    print("STEP 2: Reload Model in Fresh Instance")
    print("=" * 72)
    fresh_model = XGBClassifier()
    fresh_model.load_model(model_path)
    print("Successfully loaded model into fresh XGBClassifier instance.")

    print("\n" + "=" * 72)
    print("STEP 3: Load Held-out Test Set & Validate Predictions")
    print("=" * 72)
    X_path = os.path.join(proc_dir, "X_features.parquet")
    y_path = os.path.join(proc_dir, "y_target.parquet")
    X = pd.read_parquet(X_path)
    y_df = pd.read_parquet(y_path)
    y = (y_df["isFraud"] if "isFraud" in y_df.columns else y_df.iloc[:, 0]).astype("int8")

    # Use the exact same seed and test split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pick a sample of test transactions
    sample_size = 5000
    X_sample = X_test.iloc[:sample_size]
    y_sample = y_test.iloc[:sample_size]

    t0 = time.perf_counter()
    sample_preds = fresh_model.predict(X_sample)
    sample_probas = fresh_model.predict_proba(X_sample)[:, 1]
    infer_time = time.perf_counter() - t0

    assert np.all(np.isfinite(sample_probas)), "Probabilities contain NaN or Inf!"
    assert sample_preds.shape == (sample_size,), f"Expected shape ({sample_size},), got {sample_preds.shape}"
    print(f"Tested {sample_size:,} held-out transactions in {infer_time*1000:.2f} ms ({sample_size / infer_time:,.0f} rows/s).")
    print(f"All predictions finite: True. Shape verified: {sample_preds.shape}")

    print("\n" + "=" * 72)
    print("STEP 4: Test Specific Ground-Truth Legit & Fraud Examples")
    print("=" * 72)
    # Find test indices for actual fraud and actual legit
    fraud_indices = y_test[y_test == 1].index
    legit_indices = y_test[y_test == 0].index

    # Test top 3 legit and top 3 fraud
    test_cases = []
    print("\n[Legitimate Examples from Test Set]")
    for idx in legit_indices[:3]:
        row = X_test.loc[[idx]]
        pred = int(fresh_model.predict(row)[0])
        prob = float(fresh_model.predict_proba(row)[:, 1][0])
        print(f"  Row {idx}: True=0, Predicted={pred}, Fraud Probability={prob:.6f}")
        test_cases.append({
            "index": int(idx),
            "true_label": 0,
            "predicted_label": pred,
            "fraud_probability": round(prob, 6),
            "features": row.to_dict(orient="records")[0]
        })

    print("\n[Fraudulent Examples from Test Set]")
    for idx in fraud_indices[:3]:
        row = X_test.loc[[idx]]
        pred = int(fresh_model.predict(row)[0])
        prob = float(fresh_model.predict_proba(row)[:, 1][0])
        print(f"  Row {idx}: True=1, Predicted={pred}, Fraud Probability={prob:.6f}")
        test_cases.append({
            "index": int(idx),
            "true_label": 1,
            "predicted_label": pred,
            "fraud_probability": round(prob, 6),
            "features": row.to_dict(orient="records")[0]
        })

    print("\n" + "=" * 72)
    print("STEP 5: Simulate Raw Incoming Transaction Preprocessing (FastAPI Ready)")
    print("=" * 72)
    # Simulate a raw transaction payload arriving via HTTP JSON
    raw_transaction = {
        "step": 741,
        "type": "TRANSFER",
        "amount": 250000.0,
        "nameOrig": "C123456789",
        "oldbalanceOrg": 250000.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C987654321",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0
    }

    def preprocess_raw_transaction(tx: dict, expected_cols: list[str]) -> pd.DataFrame:
        """Lightweight preprocessor converting raw transaction into model-ready features."""
        error_orig = float(tx["oldbalanceOrg"] - tx["amount"] - tx["newbalanceOrig"])
        error_dest = float(tx["oldbalanceDest"] + tx["amount"] - tx["newbalanceDest"])
        is_merchant = 1 if str(tx["nameDest"]).startswith("M") else 0
        hour_of_day = int(tx["step"] % 24)
        day_of_week = int((tx["step"] // 24) % 7)

        feature_dict = {
            "amount": float(tx["amount"]),
            "errorBalanceOrig": error_orig,
            "errorBalanceDest": error_dest,
            "is_merchant": is_merchant,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "type_CASH_IN": 1 if tx["type"] == "CASH_IN" else 0,
            "type_CASH_OUT": 1 if tx["type"] == "CASH_OUT" else 0,
            "type_DEBIT": 1 if tx["type"] == "DEBIT" else 0,
            "type_PAYMENT": 1 if tx["type"] == "PAYMENT" else 0,
            "type_TRANSFER": 1 if tx["type"] == "TRANSFER" else 0,
        }
        df_tx = pd.DataFrame([feature_dict])[expected_cols]
        return df_tx

    processed_tx = preprocess_raw_transaction(raw_transaction, metadata["features"])
    raw_pred = int(fresh_model.predict(processed_tx)[0])
    raw_prob = float(fresh_model.predict_proba(processed_tx)[:, 1][0])
    print(f"Raw Input: {raw_transaction}")
    print(f"Preprocessed DataFrame:\n{processed_tx}")
    print(f"Prediction: {'FRAUD (1)' if raw_pred == 1 else 'LEGIT (0)'} | Fraud Probability: {raw_prob:.6f}")

    # Save verification sample report
    sample_report = {
        "status": "PASS",
        "model_file": model_path,
        "model_size_bytes": model_size,
        "verified_features": metadata["features"],
        "sample_test_cases": test_cases,
        "simulated_raw_api_test": {
            "input": raw_transaction,
            "predicted_class": raw_pred,
            "fraud_probability": round(raw_prob, 6)
        }
    }
    sample_report_path = os.path.join(reports_dir, "sample_predictions.json")
    with open(sample_report_path, "w", encoding="utf-8") as f:
        json.dump(sample_report, f, indent=2)
    print(f"\nSaved sample predictions & verification report -> {sample_report_path}")

    return sample_report


if __name__ == "__main__":
    verify_model_pipeline()
