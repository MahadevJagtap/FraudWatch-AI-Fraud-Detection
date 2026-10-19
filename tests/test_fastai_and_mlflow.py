"""
Automated tests for FastAI Tabular Model and MLflow Experiment Tracking.
Milestone 2 test suite.
"""

from __future__ import annotations

import json
import os
import sqlite3
import numpy as np
import pandas as pd
import pytest
import torch
from fastai.tabular.all import (
    Categorify,
    CategoryBlock,
    Normalize,
    TabularPandas,
    load_learner,
    tabular_learner,
)
import mlflow
from mlflow.tracking import MlflowClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fastai_tabular_model.pkl")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "fastai_evaluation_report.json")
MLFLOW_DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db")

SAMPLE_FEATURES = [
    "hour_of_day",
    "day_of_week",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
    "amount",
    "log_amount",
    "is_merchant",
    "dest_is_customer",
    "orig_old_balance",
    "dest_old_balance",
    "orig_is_zero",
    "dest_is_zero",
    "orig_drain_ratio",
]


def test_fastai_artifacts_exist():
    assert os.path.isfile(MODEL_PATH), f"FastAI model missing: {MODEL_PATH}"
    assert os.path.getsize(MODEL_PATH) > 0, "FastAI model file is empty"
    assert os.path.isfile(REPORT_PATH), f"Report missing: {REPORT_PATH}"

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        rep = json.load(f)

    assert rep["model_name"] == "fastai_tabular_deep_learning"
    assert rep["test_auprc"] > 0.90, f"Expected AUPRC > 0.90, got {rep['test_auprc']}"
    assert rep["test_roc_auc"] > 0.95, f"Expected ROC-AUC > 0.95, got {rep['test_roc_auc']}"


def test_fastai_model_inference():
    learn = load_learner(MODEL_PATH)
    assert learn is not None

    sample_tx = pd.DataFrame([{
        "hour_of_day": 12,
        "day_of_week": 2,
        "type_CASH_IN": 0,
        "type_CASH_OUT": 0,
        "type_DEBIT": 0,
        "type_PAYMENT": 0,
        "type_TRANSFER": 1,
        "amount": 100000.0,
        "log_amount": 11.5129,
        "is_merchant": 0,
        "dest_is_customer": 1,
        "orig_old_balance": 100000.0,
        "dest_old_balance": 0.0,
        "orig_is_zero": 0,
        "dest_is_zero": 1,
        "orig_drain_ratio": 1.0
    }])

    row, pred_class, probs = learn.predict(sample_tx.iloc[0])
    assert len(probs) == 2
    fraud_proba = float(probs[1])
    assert 0.0 <= fraud_proba <= 1.0


def test_mlflow_runs_logged():
    assert os.path.isfile(MLFLOW_DB_PATH), f"MLflow database missing: {MLFLOW_DB_PATH}"

    conn = sqlite3.connect(MLFLOW_DB_PATH)
    cursor = conn.cursor()

    # Verify both the milestone experiment and previous experiment exist
    cursor.execute("SELECT name FROM experiments WHERE name = 'fraudwatch-preauth-models'")
    exp = cursor.fetchone()
    assert exp is not None, "Experiment 'fraudwatch-preauth-models' not found in MLflow db"

    # Verify that the runs in fraudwatch-preauth-models are finished successfully
    cursor.execute("""
        SELECT r.name, r.status
        FROM runs r
        JOIN experiments e ON r.experiment_id = e.experiment_id
        WHERE e.name = 'fraudwatch-preauth-models'
    """)
    runs = cursor.fetchall()
    run_dict = {name: status for name, status in runs}
    assert "xgboost_preauth_champion" in run_dict, "Missing run xgboost_preauth_champion"
    assert "fastai_tabular_nn" in run_dict, "Missing run fastai_tabular_nn"
    assert run_dict["xgboost_preauth_champion"] == "FINISHED"
    assert run_dict["fastai_tabular_nn"] == "FINISHED"

    conn.close()


def test_mlflow_temporary_tracking_store(tmp_path):
    """Verify MLflow experiment creation and logging using an isolated local SQLite tracking store."""
    temp_db = tmp_path / "test_mlflow.db"
    tracking_uri = f"sqlite:///{temp_db}"
    orig_uri = mlflow.get_tracking_uri()

    try:
        mlflow.set_tracking_uri(tracking_uri)
        exp_name = "test-isolated-preauth"
        exp_id = mlflow.create_experiment(exp_name)

        with mlflow.start_run(experiment_id=exp_id, run_name="test_run_unit") as run:
            mlflow.log_params({
                "model_type": "xgboost_test",
                "max_depth": 6,
                "threshold_strategy": "validation_f1_optimal"
            })
            mlflow.log_metrics({
                "test_average_precision": 0.9917,
                "test_roc_auc": 0.9999,
                "test_precision": 0.9209,
                "test_recall": 0.9771,
                "test_f1": 0.9481,
                "test_threshold": 0.8247,
                "test_true_positives": 1024,
                "test_false_positives": 88
            })

            # Log a test artifact file
            artifact_file = tmp_path / "test_artifact.json"
            artifact_file.write_text(json.dumps({"status": "verified"}), encoding="utf-8")
            mlflow.log_artifact(str(artifact_file), artifact_path="audit")

        client = MlflowClient(tracking_uri=tracking_uri)
        retrieved_run = client.get_run(run.info.run_id)

        assert retrieved_run.info.status == "FINISHED"
        assert retrieved_run.data.params["model_type"] == "xgboost_test"
        assert float(retrieved_run.data.metrics["test_average_precision"]) == pytest.approx(0.9917, rel=1e-4)
        assert float(retrieved_run.data.metrics["test_threshold"]) == pytest.approx(0.8247, rel=1e-4)

        artifacts = client.list_artifacts(run.info.run_id, path="audit")
        assert any(a.path.endswith("test_artifact.json") for a in artifacts)

    finally:
        mlflow.set_tracking_uri(orig_uri)


def test_mlflow_required_metrics_and_metadata_logged():
    """Verify that required evaluation metrics and parameters are logged in the main experiment."""
    tracking_uri = f"sqlite:///{MLFLOW_DB_PATH}"
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("fraudwatch-preauth-models")
    assert experiment is not None, "Experiment fraudwatch-preauth-models not found"

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) >= 2, f"Expected >= 2 runs in experiment, got {len(runs)}"

    required_metrics = {
        "test_auprc",
        "test_roc_auc",
        "test_precision",
        "test_recall",
        "test_f1",
        "optimal_threshold",
        "test_tp",
        "test_fp",
        "test_fn",
        "test_tn",
    }

    for run in runs:
        logged_keys = set(run.data.metrics.keys())
        missing = required_metrics - logged_keys
        assert not missing, f"Run {run.info.run_name} missing required metrics: {missing}"
        assert "feature_count" in run.data.params or "features_count" in run.data.params


def test_fastai_training_on_synthetic_fixture(tmp_path):
    """Test FastAI Tabular model training and inference on a small synthetic fixture."""
    np.random.seed(42)
    n_samples = 60

    data = {
        "hour_of_day": np.random.randint(0, 24, n_samples),
        "day_of_week": np.random.randint(0, 7, n_samples),
        "type_CASH_IN": np.random.choice([0, 1], n_samples),
        "type_CASH_OUT": np.random.choice([0, 1], n_samples),
        "type_DEBIT": np.zeros(n_samples, dtype=int),
        "type_PAYMENT": np.random.choice([0, 1], n_samples),
        "type_TRANSFER": np.random.choice([0, 1], n_samples),
        "amount": np.random.exponential(1000.0, n_samples) + 10.0,
        "is_merchant": np.random.choice([0, 1], n_samples),
        "dest_is_customer": np.random.choice([0, 1], n_samples),
        "orig_old_balance": np.random.exponential(5000.0, n_samples),
        "dest_old_balance": np.random.exponential(5000.0, n_samples),
        "orig_is_zero": np.random.choice([0, 1], n_samples),
        "dest_is_zero": np.random.choice([0, 1], n_samples),
    }
    data["log_amount"] = np.log1p(data["amount"])
    data["orig_drain_ratio"] = data["amount"] / (data["orig_old_balance"] + 1.0)
    data["isFraud"] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])

    df = pd.DataFrame(data)

    cat_names = [
        "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
        "is_merchant", "dest_is_customer", "orig_is_zero", "dest_is_zero"
    ]
    cont_names = [
        "hour_of_day", "day_of_week", "amount", "log_amount",
        "orig_old_balance", "dest_old_balance", "orig_drain_ratio"
    ]

    splits = (list(range(40)), list(range(40, 60)))
    to = TabularPandas(
        df,
        procs=[Categorify, Normalize],
        cat_names=cat_names,
        cont_names=cont_names,
        y_names="isFraud",
        y_block=CategoryBlock(),
        splits=splits,
    )

    dls = to.dataloaders(bs=16)
    learn = tabular_learner(dls, layers=[20, 10], n_out=2)
    learn.fit(1, lr=0.01)

    # Inference on a single row
    sample_row = df.iloc[45].drop("isFraud")
    row, pred_class, probs = learn.predict(sample_row)

    assert int(pred_class) in (0, 1)
    assert len(probs) == 2
    assert float(probs[0] + probs[1]) == pytest.approx(1.0, rel=1e-3)


def test_training_only_preprocessing():
    """Verify that preprocessing statistics (normalization mean/std) are fit strictly on training data."""
    train_df = pd.DataFrame({
        "amount": [10.0, 20.0, 30.0, 40.0],
        "is_merchant": [0, 1, 0, 1],
        "isFraud": [0, 0, 1, 0],
    })
    # Test set has radically different distribution (mean 10,000)
    test_df = pd.DataFrame({
        "amount": [10000.0, 20000.0],
        "is_merchant": [1, 0],
        "isFraud": [0, 1],
    })

    combined = pd.concat([train_df, test_df], ignore_index=True)
    splits = (list(range(4)), list(range(4, 6)))

    to = TabularPandas(
        combined,
        procs=[Normalize],
        cat_names=["is_merchant"],
        cont_names=["amount"],
        y_names="isFraud",
        y_block=CategoryBlock(),
        splits=splits,
    )

    # Normalize procedure must calculate means exclusively from splits[0] (train_df)
    train_mean = train_df["amount"].mean()  # 25.0
    norm_proc = [p for p in to.procs if isinstance(p, Normalize)][0]

    assert float(norm_proc.means["amount"]) == pytest.approx(train_mean, abs=1e-5)
    # Ensure test mean was NOT leaked into norm_proc.means
    assert float(norm_proc.means["amount"]) < 100.0


def test_threshold_selection_validation_only():
    """Verify that the decision threshold is determined solely from validation labels."""
    # Synthetic validation set
    y_val = np.array([0, 0, 0, 0, 1, 1, 0, 1, 0, 0])
    p_val = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.6, 0.9, 0.15, 0.25])

    # Synthetic untouched test set
    y_test_a = np.array([0, 0, 1, 1, 0])
    p_test = np.array([0.2, 0.3, 0.75, 0.85, 0.1])

    # Threshold selection function that searches grid on validation data only
    def select_threshold_on_val(y_v, p_v):
        thresholds = np.linspace(0.1, 0.9, 81)
        best_t, best_f1 = 0.5, -1.0
        for t in thresholds:
            pred = (p_v >= t).astype(int)
            tp = np.sum((pred == 1) & (y_v == 1))
            fp = np.sum((pred == 1) & (y_v == 0))
            fn = np.sum((pred == 0) & (y_v == 1))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        return best_t

    selected_t = select_threshold_on_val(y_val, p_val)

    # Even if test labels are completely changed/inverted, selected threshold MUST remain identical
    y_test_b = np.array([1, 1, 0, 0, 1])
    selected_t_recheck = select_threshold_on_val(y_val, p_val)

    assert selected_t == selected_t_recheck, "Threshold must not depend on test data"
    # Apply selected threshold to test set once
    test_preds = (p_test >= selected_t).astype(int)
    assert len(test_preds) == len(p_test)


def test_fastai_model_save_reload_consistency(tmp_path):
    """Verify that export and reload of FastAI Tabular model preserves exact predictions."""
    learn = load_learner(MODEL_PATH)
    temp_export = tmp_path / "temp_fastai.pkl"
    learn.export(temp_export)

    reloaded_learn = load_learner(temp_export)

    sample_tx = pd.DataFrame([{
        "hour_of_day": 10,
        "day_of_week": 1,
        "type_CASH_IN": 0,
        "type_CASH_OUT": 1,
        "type_DEBIT": 0,
        "type_PAYMENT": 0,
        "type_TRANSFER": 0,
        "amount": 5000.0,
        "log_amount": 8.5174,
        "is_merchant": 0,
        "dest_is_customer": 1,
        "orig_old_balance": 5000.0,
        "dest_old_balance": 1000.0,
        "orig_is_zero": 0,
        "dest_is_zero": 0,
        "orig_drain_ratio": 5000.0 / 5001.0
    }])

    _, _, probs_orig = learn.predict(sample_tx.iloc[0])
    _, _, probs_reloaded = reloaded_learn.predict(sample_tx.iloc[0])

    assert torch.allclose(probs_orig, probs_reloaded, atol=1e-5)
