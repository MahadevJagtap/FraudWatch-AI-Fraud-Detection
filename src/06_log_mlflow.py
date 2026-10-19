"""
MLflow Experiment Tracking for FraudWatch
==========================================

Logs pre-authorization XGBoost models, hyperparameters, evaluation metrics,
and artifacts to a local MLflow tracking store.

Run from project root:
    python src/06_log_mlflow.py
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import mlflow
import mlflow.xgboost

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "mlruns")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "leakage_free_xgb_model.json")
METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "leakage_free_metadata.json")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "leakage_free_evaluation_report.json")
IMPORTANCE_PATH = os.path.join(PROJECT_ROOT, "reports", "leakage_free_feature_importance.csv")


def log_experiment():
    sqlite_uri = f"sqlite:///{os.path.abspath(DB_PATH).replace(os.sep, '/')}"
    print(f"Configuring MLflow tracking URI: {sqlite_uri}")
    mlflow.set_tracking_uri(sqlite_uri)
    experiment_name = "FraudWatch_Pre_Authorization"
    mlflow.set_experiment(experiment_name)

    if not os.path.isfile(REPORT_PATH) or not os.path.isfile(METADATA_PATH):
        raise FileNotFoundError("Evaluation report or metadata missing. Run 05_train_leakage_free_model.py first.")

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"Starting MLflow run under experiment {experiment_name!r}...")
    with mlflow.start_run(run_name="xgboost_leakage_free_v1") as run:
        run_id = run.info.run_id
        print(f"Active MLflow Run ID: {run_id}")

        # 1. Tags
        mlflow.set_tags({
            "model_type": "xgboost",
            "leakage_free": "true",
            "pre_authorization": "true",
            "split_type": "chronological_time_aware",
            "dataset": "PaySim_6.36M",
            "target": "isFraud"
        })

        # 2. Parameters
        params = metadata.get("hyperparameters", {})
        mlflow.log_params(params)
        mlflow.log_params({
            "scale_pos_weight": metadata.get("scale_pos_weight", 972.04),
            "train_rows": metadata.get("train_rows", 6201194),
            "val_rows": metadata.get("val_rows", 80316),
            "test_rows": metadata.get("test_rows", 81110),
            "feature_count": len(metadata.get("features", [])),
            "decision_threshold": report.get("decision_threshold", 0.70)
        })

        # 3. Metrics
        mlflow.log_metrics({
            "test_auprc": report.get("test_auprc", 0.0),
            "test_roc_auc": report.get("test_roc_auc", 0.0),
            "optimal_threshold": report.get("decision_threshold", 0.70),
            "test_f1": report.get("test_f1_at_optimal_threshold", 0.0),
            "test_precision": report.get("test_precision", 0.0),
            "test_recall": report.get("test_recall", 0.0),
            "test_tp": report.get("test_true_positives", 0),
            "test_fp": report.get("test_false_positives", 0),
            "test_fn": report.get("test_false_negatives", 0),
            "test_tn": report.get("test_true_negatives", 0),
            "training_time_seconds": report.get("training_time_seconds", 0.0)
        })

        # 4. Artifacts
        mlflow.log_artifact(METADATA_PATH, artifact_path="metadata")
        mlflow.log_artifact(REPORT_PATH, artifact_path="evaluation")
        if os.path.isfile(IMPORTANCE_PATH):
            mlflow.log_artifact(IMPORTANCE_PATH, artifact_path="feature_importance")

        # 5. Model artifact
        clf = XGBClassifier()
        clf.load_model(MODEL_PATH)
        mlflow.xgboost.log_model(clf, artifact_path="model")

        print(f"\nMLflow run logged successfully!")
        print(f"Run ID: {run_id}")
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"View runs using: python -m mlflow ui --backend-store-uri {sqlite_uri}")
        return run_id


if __name__ == "__main__":
    log_experiment()
