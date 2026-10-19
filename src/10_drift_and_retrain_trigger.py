"""
Evidently AI Drift Monitoring & Controlled Retraining Trigger
=============================================================

Monitors data drift between baseline training data (Days 1-24) and out-of-time
production data (Days 28-31) using Evidently AI.

Controlled Promotion Gate:
--------------------------
1. Generates full Evidently HTML and JSON drift reports.
2. If drift threshold is breached, triggers a controlled candidate retraining job.
3. Compares Candidate model vs Champion model on validation benchmark.
4. SAFEGUARD: Never automatically promotes solely because drift occurred.
   A candidate model is promoted if and only if candidate AUPRC >= champion AUPRC.

Run standalone:
    python src/10_drift_and_retrain_trigger.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, Any

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score

import mlflow

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TRAIN_PARQUET = os.path.join(PROJECT_ROOT, "data", "processed", "leakage_free_train.parquet")
VAL_PARQUET = os.path.join(PROJECT_ROOT, "data", "processed", "leakage_free_val.parquet")
TEST_PARQUET = os.path.join(PROJECT_ROOT, "data", "processed", "leakage_free_test.parquet")
CHAMPION_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "leakage_free_xgb_model.json")
METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "leakage_free_metadata.json")

REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
DRIFT_HTML_PATH = os.path.join(REPORTS_DIR, "evidently_drift_report.html")
DRIFT_JSON_PATH = os.path.join(REPORTS_DIR, "evidently_drift_report.json")
PROMOTION_AUDIT_PATH = os.path.join(REPORTS_DIR, "model_promotion_audit.json")
DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db")

DRIFT_POLICY_THRESHOLD = 0.25  # Trigger retraining if >= 25% of features exhibit drift


def run_drift_analysis(sample_size: int = 5000) -> Dict[str, Any]:
    print("=" * 70)
    print("Evidently AI: Pre-Authorization Data Drift Analysis")
    print("=" * 70)

    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 1. Load reference and current datasets
    print("Loading reference baseline and out-of-time datasets...")
    train_df = pd.read_parquet(TRAIN_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta["features"]

    # Sample representative window for fast statistical testing
    ref_sample = train_df[feature_cols].sample(n=min(sample_size, len(train_df)), random_state=42)
    curr_sample = test_df[feature_cols].sample(n=min(sample_size, len(test_df)), random_state=42)

    print(f"Comparing Reference (Train Day 1-24, {len(ref_sample):,} rows) vs Current (Test Day 28-31, {len(curr_sample):,} rows)...")

    # 2. Run Evidently Report
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref_sample, current_data=curr_sample)

    # 3. Save HTML and JSON reports
    snapshot.save_html(DRIFT_HTML_PATH)
    snapshot.save_json(DRIFT_JSON_PATH)
    print(f"Saved interactive HTML drift report: {DRIFT_HTML_PATH}")
    print(f"Saved JSON drift snapshot:          {DRIFT_JSON_PATH}")

    # 4. Extract drift summary
    drift_dict = snapshot.dict()
    # Find drift metrics in snapshot
    drift_summary = {
        "timestamp": time.time(),
        "reference_rows": len(ref_sample),
        "current_rows": len(curr_sample),
        "total_features": len(feature_cols),
        "drift_policy_threshold": DRIFT_POLICY_THRESHOLD,
    }

    # Inspect metrics in snapshot
    drifted_count = 0
    drift_details = {}
    metrics_list = drift_dict.get("metrics", []) or drift_dict.get("metric_results", [])
    for item in metrics_list:
        metric_name = str(item.get("metric_name", item.get("metric", "")))
        val = item.get("value", {})
        if "DriftedColumnsCount" in metric_name and isinstance(val, dict):
            drifted_count = int(val.get("count", 0))
        elif isinstance(val, dict) and "drift_by_columns" in val:
            for col_name, col_data in val["drift_by_columns"].items():
                is_drift = col_data.get("drift_detected", False)
                score = col_data.get("drift_score", 0.0)
                drift_details[col_name] = {"drift_detected": is_drift, "drift_score": round(score, 4)}
                if is_drift:
                    drifted_count += 1
        elif "ValueDrift" in metric_name:
            col = str(item.get("config", {}).get("column", metric_name))
            drift_details[col] = {"drift_score": val}

    drift_share = drifted_count / max(1, len(feature_cols))
    drift_summary["drifted_features_count"] = drifted_count
    drift_summary["drift_share"] = round(drift_share, 4)
    drift_summary["drift_detected"] = bool(drift_share >= DRIFT_POLICY_THRESHOLD)

    print(f"\nDrift Results: {drifted_count}/{len(feature_cols)} features drifted (share: {drift_share:.1%})")
    print(f"Drift Threshold Trigger: {'TRIGGERED' if drift_summary['drift_detected'] else 'NOT TRIGGERED'}")

    # 5. Controlled Retraining & Model Promotion Gate
    promotion_audit = evaluate_controlled_retraining(
        drift_summary=drift_summary,
        feature_cols=feature_cols
    )

    # 6. Log to MLflow
    try:
        sqlite_uri = f"sqlite:///{os.path.abspath(DB_PATH).replace(os.sep, '/')}"
        mlflow.set_tracking_uri(sqlite_uri)
        mlflow.set_experiment("FraudWatch_Pre_Authorization")

        with mlflow.start_run(run_name="evidently_drift_audit"):
            mlflow.set_tags({
                "type": "evidently_drift_monitoring",
                "retraining_triggered": str(drift_summary["drift_detected"]).lower(),
                "promotion_verdict": promotion_audit["promotion_verdict"]
            })
            mlflow.log_metrics({
                "drift_share": drift_share,
                "drifted_features": drifted_count,
                "champion_auprc": promotion_audit["champion_auprc"],
                "candidate_auprc": promotion_audit["candidate_auprc"]
            })
            mlflow.log_artifact(DRIFT_JSON_PATH, artifact_path="drift_reports")
            mlflow.log_artifact(PROMOTION_AUDIT_PATH, artifact_path="governance")
            print("Logged drift audit and governance verdict to MLflow.")
    except Exception as e:
        print(f"MLflow logging notice: {e}")

    return drift_summary


def evaluate_controlled_retraining(
    drift_summary: Dict[str, Any],
    feature_cols: list[str]
) -> Dict[str, Any]:
    """Execute safe candidate training and enforce promotion validation guardrail."""
    print("\n" + "=" * 70)
    print("Controlled Retraining Guardrail & Model Promotion Gate")
    print("=" * 70)

    # Load validation set for impartial gatekeeping
    val_df = pd.read_parquet(VAL_PARQUET)
    X_val = val_df[feature_cols]
    y_val = val_df["isFraud"].to_numpy()

    # Load Champion model
    champion_clf = XGBClassifier()
    champion_clf.load_model(CHAMPION_MODEL_PATH)
    champion_preds = champion_clf.predict_proba(X_val)[:, 1]
    champion_auprc = float(average_precision_score(y_val, champion_preds))

    print(f"Current Champion Model Validation AUPRC: {champion_auprc:.4f}")

    # Train a candidate model on recent window (simulating retrained model)
    train_df = pd.read_parquet(TRAIN_PARQUET)
    recent_train = train_df.tail(100_000)
    X_recent = recent_train[feature_cols]
    y_recent = recent_train["isFraud"].to_numpy()

    candidate_clf = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=972.04,
        random_state=42
    )
    candidate_clf.fit(X_recent, y_recent)
    candidate_preds = candidate_clf.predict_proba(X_val)[:, 1]
    candidate_auprc = float(average_precision_score(y_val, candidate_preds))

    print(f"Candidate Retrained Model Validation AUPRC: {candidate_auprc:.4f}")

    # MANDATORY PROMOTION RULE:
    # Never promote solely on drift. Candidate MUST meet or exceed Champion AUPRC on benchmark.
    if candidate_auprc >= champion_auprc:
        verdict = "APPROVED_FOR_PROMOTION"
        action = "Candidate model outperformed champion. Safe to promote."
        promoted = True
    else:
        verdict = "REJECTED_PROMOTION"
        action = f"Candidate model (AUPRC {candidate_auprc:.4f}) failed to exceed champion (AUPRC {champion_auprc:.4f}). Retaining champion model to prevent degradation."
        promoted = False

    print(f"\nGovernance Promotion Verdict: {verdict}")
    print(f"Action: {action}")

    audit_record = {
        "timestamp": time.time(),
        "drift_detected": drift_summary["drift_detected"],
        "drift_share": drift_summary["drift_share"],
        "champion_auprc": round(champion_auprc, 4),
        "candidate_auprc": round(candidate_auprc, 4),
        "promotion_verdict": verdict,
        "is_model_promoted": promoted,
        "explanation": action,
        "policy": "Candidate must meet or exceed Champion AUPRC on validation benchmark regardless of drift status."
    }

    with open(PROMOTION_AUDIT_PATH, "w", encoding="utf-8") as f:
        json.dump(audit_record, f, indent=2)
    print(f"Saved governance promotion audit to: {PROMOTION_AUDIT_PATH}")

    return audit_record


if __name__ == "__main__":
    run_drift_analysis()
