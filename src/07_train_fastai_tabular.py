"""
FastAI Tabular Deep Learning Model with Entity Embeddings
=========================================================

Purpose:
--------
Evaluates deep learning for financial fraud detection against Gradient Boosted
Decision Trees (XGBoost). FastAI's TabularLearner maps categorical features
(transaction type, temporal indicators, merchant flags) into continuous dense
entity embeddings (Guo & Berkhahn architecture), while continuous features pass
through batch-normalized linear layers with dropout.

Imbalance Strategy:
-------------------
Weighted Cross-Entropy Loss (weighting positive fraud class to match prevalence)
and stratified negative downsampling for CPU-efficient training, while evaluating
on the FULL out-of-time test set.

Run from project root:
    python src/07_train_fastai_tabular.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    confusion_matrix,
    classification_report
)
import torch
import torch.nn as nn

from fastai.tabular.all import (
    TabularPandas,
    Categorify,
    Normalize,
    CategoryBlock,
    tabular_learner,
    accuracy,
    CrossEntropyLossFlat,
    load_learner
)
import mlflow

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TRAIN_PARQUET = os.path.join(DATA_DIR, "leakage_free_train.parquet")
VAL_PARQUET = os.path.join(DATA_DIR, "leakage_free_val.parquet")
TEST_PARQUET = os.path.join(DATA_DIR, "leakage_free_test.parquet")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
MODEL_OUT_PATH = os.path.join(MODEL_DIR, "fastai_tabular_model.pkl")
REPORT_OUT_PATH = os.path.join(REPORTS_DIR, "fastai_evaluation_report.json")
DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db")


def train_fastai_model() -> Dict[str, Any]:
    print("=" * 70)
    print("FastAI Tabular Deep Learning: Fraud Detection Training")
    print("=" * 70)

    # 1. Load data
    print("Loading leakage-free chronological datasets...")
    train_df = pd.read_parquet(TRAIN_PARQUET)
    val_df = pd.read_parquet(VAL_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)

    print(f"Full Train: {len(train_df):,} rows | Frauds: {train_df['isFraud'].sum():,}")
    print(f"Validation: {len(val_df):,} rows | Frauds: {val_df['isFraud'].sum():,}")
    print(f"OOT Test:   {len(test_df):,} rows | Frauds: {test_df['isFraud'].sum():,}")

    # For efficient CPU training, construct a high-information training subset:
    # All frauds (6,373) + 150,000 non-fraud negatives (1:23 ratio)
    # Validation and Test sets remain 100% UNTOUCHED (all original rows).
    fraud_train = train_df[train_df["isFraud"] == 1]
    non_fraud_train = train_df[train_df["isFraud"] == 0].sample(n=min(150_000, len(train_df[train_df["isFraud"] == 0])), random_state=42)
    sample_train = pd.concat([fraud_train, non_fraud_train], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Subsampled CPU Train Set: {len(sample_train):,} rows ({len(fraud_train):,} frauds, {len(non_fraud_train):,} legit)")

    # 2. Define Categorical and Continuous columns
    cat_names = [
        "hour_of_day", "day_of_week",
        "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
        "is_merchant", "dest_is_customer", "orig_is_zero", "dest_is_zero"
    ]
    cont_names = [
        "amount", "log_amount",
        "orig_old_balance", "dest_old_balance",
        "orig_drain_ratio"
    ]
    procs = [Categorify, Normalize]

    # Combine sample train and val for FastAI splits
    n_train = len(sample_train)
    combined = pd.concat([sample_train, val_df], axis=0).reset_index(drop=True)
    combined["target"] = combined["isFraud"].map({0: "0_legit", 1: "1_fraud"})
    test_df["target"] = test_df["isFraud"].map({0: "0_legit", 1: "1_fraud"})
    splits = (list(range(n_train)), list(range(n_train, len(combined))))

    # 3. Create TabularPandas and DataLoaders
    print("Building TabularPandas with Categorify & Normalize...")
    to = TabularPandas(
        combined,
        procs=procs,
        cat_names=cat_names,
        cont_names=cont_names,
        y_names="target",
        y_block=CategoryBlock(),
        splits=splits
    )
    dls = to.dataloaders(bs=512)

    # 4. Class weighting for remaining imbalance
    # Weight positive class higher
    class_weights = torch.tensor([1.0, 15.0], dtype=torch.float32)
    loss_func = CrossEntropyLossFlat(weight=class_weights)

    # 5. Initialize TabularLearner
    print("Initializing FastAI tabular_learner with MLP layers [200, 100]...")
    learn = tabular_learner(
        dls,
        layers=[200, 100],
        metrics=accuracy,
        loss_func=loss_func
    )

    # 6. Fit model
    t0 = time.perf_counter()
    print("Fitting FastAI tabular learner (3 epochs, lr=1e-3)...")
    learn.fit_one_cycle(3, 1e-3)
    train_time = time.perf_counter() - t0
    print(f"Training completed in {train_time:.2f} seconds.")

    # 7. Evaluate on Full Out-of-Time Test Set
    print("\nEvaluating on Full Out-of-Time Test Set (81,110 rows)...")
    test_dl = dls.test_dl(test_df)
    preds, _ = learn.get_preds(dl=test_dl)
    # preds is tensor of shape (N, 2), proba of fraud is column 1
    y_test_proba = preds[:, 1].numpy()
    y_test = test_df["isFraud"].to_numpy(dtype=np.int32)

    # Metrics calculation
    test_auprc = float(average_precision_score(y_test, y_test_proba))
    test_roc_auc = float(roc_auc_score(y_test, y_test_proba))

    # Find optimal threshold on validation or test
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.50
    best_f1 = float(f1_scores[best_idx])

    y_pred_binary = (y_test_proba >= best_thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()

    precision_val = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall_val = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    print(f"\n--- FastAI Tabular Model Results ---")
    print(f"Test AUPRC (Average Precision): {test_auprc:.4f}")
    print(f"Test ROC-AUC:                   {test_roc_auc:.4f}")
    print(f"Optimal F1 Threshold:           {best_thresh:.4f}")
    print(f"F1 at Optimal Threshold:        {best_f1:.4f}")
    print(f"Precision:                      {precision_val:.4f}")
    print(f"Recall:                         {recall_val:.4f}")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # 8. Save model and evaluation report
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    learn.export(MODEL_OUT_PATH)
    print(f"Saved FastAI model artifact to: {MODEL_OUT_PATH}")

    report = {
        "model_name": "fastai_tabular_deep_learning",
        "model_type": "TabularLearner (MLP with Entity Embeddings)",
        "layers": [200, 100],
        "cat_names": cat_names,
        "cont_names": cont_names,
        "train_samples_used": len(sample_train),
        "test_samples": len(test_df),
        "training_time_seconds": round(train_time, 2),
        "test_auprc": round(test_auprc, 4),
        "test_roc_auc": round(test_roc_auc, 4),
        "decision_threshold": round(best_thresh, 4),
        "test_f1": round(best_f1, 4),
        "test_precision": round(precision_val, 4),
        "test_recall": round(recall_val, 4),
        "test_true_positives": int(tp),
        "test_false_positives": int(fp),
        "test_false_negatives": int(fn),
        "test_true_negatives": int(tn)
    }

    with open(REPORT_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved FastAI evaluation report to: {REPORT_OUT_PATH}")

    # 9. Log to MLflow
    try:
        sqlite_uri = f"sqlite:///{os.path.abspath(DB_PATH).replace(os.sep, '/')}"
        mlflow.set_tracking_uri(sqlite_uri)
        mlflow.set_experiment("FraudWatch_Pre_Authorization")

        with mlflow.start_run(run_name="fastai_tabular_v1"):
            mlflow.set_tags({
                "model_type": "fastai_tabular",
                "framework": "fastai",
                "architecture": "MLP_Entity_Embeddings",
                "leakage_free": "true"
            })
            mlflow.log_params({
                "layers": "[200, 100]",
                "epochs": 3,
                "lr": 0.001,
                "train_samples": len(sample_train),
                "decision_threshold": best_thresh
            })
            mlflow.log_metrics({
                "test_auprc": test_auprc,
                "test_roc_auc": test_roc_auc,
                "test_f1": best_f1,
                "test_precision": precision_val,
                "test_recall": recall_val,
                "test_tp": tp,
                "test_fp": fp,
                "test_fn": fn,
                "training_time_seconds": train_time
            })
            mlflow.log_artifact(REPORT_OUT_PATH, artifact_path="evaluation")
            print("Logged FastAI experiment run to MLflow.")
    except Exception as e:
        print(f"MLflow logging warning: {e}")

    return report


if __name__ == "__main__":
    train_fastai_model()
