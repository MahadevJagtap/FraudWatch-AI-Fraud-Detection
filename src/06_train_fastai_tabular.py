"""
FastAI Tabular Neural Network Baseline for Fraud Detection
==========================================================

Evaluates deep learning with Entity Embeddings for financial fraud detection
under the exact same authorization-time feature contract and chronological
evaluation protocol as the Pre-Authorization XGBoost model.

Methodology:
------------
1. Features: 16 pre-authorization features (zero post-settlement leakage).
2. Partitions:
   - Train: Days 1-24 (step <= 576), 6,201,194 rows.
     (Subsampled to all 6,373 frauds + 150,000 non-frauds for reproducible CPU training).
   - Validation: Days 25-27 (step 577-648), 80,316 rows (100% full, used for threshold tuning).
   - Test (Out-of-Time): Days 28-31 (step > 648), 81,110 rows (100% full untouched test).
3. Preprocessing: Categorify and Normalize fit strictly on training indices.
4. Class Imbalance: Weighted Cross-Entropy Loss [1.0, 15.0] on training data.
5. Threshold Selection: Tuned strictly on validation set to maximize F1, then applied ONCE to test.
6. MLflow Tracking: Logged to experiment 'fraudwatch-preauth-models'.

Run from project root:
    python src/06_train_fastai_tabular.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
import torch
import torch.nn as nn
from fastai.tabular.all import (
    CategoryBlock,
    Categorify,
    CrossEntropyLossFlat,
    EarlyStoppingCallback,
    Normalize,
    SaveModelCallback,
    TabularPandas,
    accuracy,
    load_learner,
    tabular_learner,
)
import mlflow

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db")

MODEL_OUT_PATH = os.path.join(MODELS_DIR, "fastai_tabular_model.pkl")
REPORT_OUT_PATH = os.path.join(REPORTS_DIR, "fastai_evaluation_report.json")

EXPERIMENT_NAME = "fraudwatch-preauth-models"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{os.path.abspath(DB_PATH).replace(os.sep, '/')}")


def train_fastai_tabular_pipeline() -> Dict[str, Any]:
    print("=" * 72)
    print("STEP 1: Load Chronological Datasets for FastAI Baseline")
    print("=" * 72)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    train_df = pd.read_parquet(os.path.join(DATA_DIR, "leakage_free_train.parquet"))
    val_df = pd.read_parquet(os.path.join(DATA_DIR, "leakage_free_val.parquet"))
    test_df = pd.read_parquet(os.path.join(DATA_DIR, "leakage_free_test.parquet"))

    print(f"Full Train Dataset : {len(train_df):,} rows | Frauds: {train_df['isFraud'].sum():,}")
    print(f"Validation Dataset : {len(val_df):,} rows | Frauds: {val_df['isFraud'].sum():,}")
    print(f"Out-of-Time Test   : {len(test_df):,} rows | Frauds: {test_df['isFraud'].sum():,}")

    # Reproducible CPU training subset:
    # All 6,373 frauds + 150,000 non-fraud negatives (1:23 ratio)
    # Validation and Test sets remain 100% FULL and UNTOUCHED.
    fraud_train = train_df[train_df["isFraud"] == 1]
    non_fraud_train = train_df[train_df["isFraud"] == 0].sample(n=min(150_000, len(train_df[train_df["isFraud"] == 0])), random_state=42)
    sample_train = pd.concat([fraud_train, non_fraud_train], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Training Subset (CPU): {len(sample_train):,} rows ({len(fraud_train):,} frauds, {len(non_fraud_train):,} non-frauds)")

    # 2. Define Feature Contract
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
    all_features = cat_names + cont_names
    print(f"Feature Contract ({len(all_features)} features): {len(cat_names)} categorical embeddings, {len(cont_names)} continuous")

    # Target mapping
    sample_train["target"] = sample_train["isFraud"].map({0: "0_legit", 1: "1_fraud"})
    val_df["target"] = val_df["isFraud"].map({0: "0_legit", 1: "1_fraud"})
    test_df["target"] = test_df["isFraud"].map({0: "0_legit", 1: "1_fraud"})

    n_train = len(sample_train)
    combined = pd.concat([sample_train, val_df], axis=0).reset_index(drop=True)
    splits = (list(range(n_train)), list(range(n_train, len(combined))))

    # 3. Build TabularPandas with Preprocessing Fit Strictly on Train
    print("Building TabularPandas (Categorify & Normalize fit strictly on train split)...")
    to = TabularPandas(
        combined,
        procs=[Categorify, Normalize],
        cat_names=cat_names,
        cont_names=cont_names,
        y_names="target",
        y_block=CategoryBlock(),
        splits=splits
    )
    dls = to.dataloaders(bs=512)

    # 4. Loss Function with Class Weighting
    class_weights = torch.tensor([1.0, 15.0], dtype=torch.float32)
    loss_func = CrossEntropyLossFlat(weight=class_weights)

    # 5. Initialize Learner with Entity Embeddings & MLP
    print("\n" + "=" * 72)
    print("STEP 2: Initialize & Train FastAI Tabular Learner (MLP [200, 100])")
    print("=" * 72)
    learn = tabular_learner(
        dls,
        layers=[200, 100],
        metrics=accuracy,
        loss_func=loss_func,
        cbs=[EarlyStoppingCallback(monitor='valid_loss', min_delta=0.001, patience=2)]
    )

    t0 = time.perf_counter()
    learn.fit_one_cycle(3, 1e-3)
    train_time = time.perf_counter() - t0
    print(f"FastAI Training completed in {train_time:.2f}s.")

    # 6. STEP 3: Tune Decision Threshold Strictly on Validation Set
    print("\n" + "=" * 72)
    print("STEP 3: Tune Decision Threshold Strictly on Validation Set")
    print("=" * 72)
    val_dl = dls.test_dl(val_df)
    val_preds, _ = learn.get_preds(dl=val_dl)
    val_probs = val_preds[:, 1].numpy()
    y_val = val_df["isFraud"].to_numpy(dtype=int)

    val_auprc = float(average_precision_score(y_val, val_probs))
    val_roc_auc = float(roc_auc_score(y_val, val_probs))

    prec_v, rec_v, thresh_v = precision_recall_curve(y_val, val_probs)
    f1_v = 2 * (prec_v * rec_v) / (prec_v + rec_v + 1e-10)
    best_idx = np.argmax(f1_v)
    chosen_threshold = float(thresh_v[best_idx]) if best_idx < len(thresh_v) else 0.50
    val_best_f1 = float(f1_v[best_idx])
    val_prec = float(prec_v[best_idx])
    val_rec = float(rec_v[best_idx])

    print(f"Validation AUPRC:                  {val_auprc:.4f}")
    print(f"Validation ROC-AUC:                {val_roc_auc:.4f}")
    print(f"Validation Chosen Threshold (max F1): {chosen_threshold:.4f}")
    print(f"Validation F1 at Threshold:        {val_best_f1:.4f} (Prec: {val_prec:.2%}, Rec: {val_rec:.2%})")

    # 7. STEP 4: Evaluate Once on Untouched Out-of-Time Test Set
    print("\n" + "=" * 72)
    print("STEP 4: Evaluate Once on Untouched Out-of-Time Test Set (81,110 rows)")
    print("=" * 72)
    test_dl = dls.test_dl(test_df)
    t_inf = time.perf_counter()
    test_preds, _ = learn.get_preds(dl=test_dl)
    inf_time = time.perf_counter() - t_inf
    inf_latency_ms = (inf_time / len(test_df)) * 1000.0

    test_probs = test_preds[:, 1].numpy()
    y_test = test_df["isFraud"].to_numpy(dtype=int)

    test_auprc = float(average_precision_score(y_test, test_probs))
    test_roc_auc = float(roc_auc_score(y_test, test_probs))

    # Apply validation-chosen threshold
    y_test_pred = (test_probs >= chosen_threshold).astype(int)
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    test_prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    test_rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    test_f1 = float(2 * test_prec * test_rec / (test_prec + test_rec)) if (test_prec + test_rec) > 0 else 0.0

    print(f"Test Inference Latency:            {inf_latency_ms:.4f} ms/row ({len(test_df)/inf_time:,.0f} rows/s)")
    print(f"Test AUPRC (Average Precision):    {test_auprc:.4f}")
    print(f"Test ROC-AUC:                      {test_roc_auc:.4f}")
    print(f"Evaluated Threshold (from Val):    {chosen_threshold:.4f}")
    print(f"Test Precision:                    {test_prec:.4f} ({test_prec*100:.2f}%)")
    print(f"Test Recall:                       {test_rec:.4f} ({test_rec*100:.2f}%)")
    print(f"Test F1 Score:                     {test_f1:.4f}")
    print(f"Confusion Matrix:                  TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # 8. Visual Artifacts
    pr_curve_path = os.path.join(REPORTS_DIR, "fastai_pr_curve.png")
    plt.figure(figsize=(8, 6))
    prec_t, rec_t, _ = precision_recall_curve(y_test, test_probs)
    plt.plot(rec_t, prec_t, color="#7C3AED", lw=2, label=f"FastAI Tabular MLP (AUPRC = {test_auprc:.4f})")
    plt.scatter([test_rec], [test_prec], color="#DC2626", s=100, zorder=5, label=f"Val Threshold {chosen_threshold:.2f} (F1 = {test_f1:.4f})")
    plt.title("FastAI Out-of-Time Test Precision-Recall Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left", fontsize=11)
    plt.tight_layout()
    plt.savefig(pr_curve_path, dpi=150)
    plt.close()

    cm_plot_path = os.path.join(REPORTS_DIR, "fastai_confusion_matrix.png")
    plt.figure(figsize=(6, 5))
    cm_matrix = np.array([[tn, fp], [fn, tp]])
    plt.imshow(cm_matrix, cmap="Purples", interpolation="nearest")
    plt.title(f"FastAI Confusion Matrix (Thresh = {chosen_threshold:.2f})", fontsize=12, fontweight="bold")
    plt.colorbar()
    ticks = [0, 1]
    plt.xticks(ticks, ["Legitimate", "Fraud"])
    plt.yticks(ticks, ["Legitimate", "Fraud"])
    for i in range(2):
        for j in range(2):
            color = "white" if cm_matrix[i, j] > cm_matrix.max() / 2 else "black"
            plt.text(j, i, f"{cm_matrix[i, j]:,}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()

    # 9. Save Model & Reports
    learn.export(MODEL_OUT_PATH)
    print(f"Saved FastAI Model -> {MODEL_OUT_PATH}")

    report = {
        "model_name": "fastai_tabular_deep_learning",
        "model_architecture": "TabularLearner (Entity Embeddings + MLP [200, 100])",
        "cat_names": cat_names,
        "cont_names": cont_names,
        "feature_count": len(all_features),
        "train_samples_used": len(sample_train),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "threshold_selection_dataset": "validation (steps 577-648)",
        "training_time_seconds": round(train_time, 2),
        "inference_latency_ms": round(inf_latency_ms, 4),
        "val_auprc": round(val_auprc, 4),
        "val_roc_auc": round(val_roc_auc, 4),
        "val_f1": round(val_best_f1, 4),
        "decision_threshold": round(chosen_threshold, 4),
        "test_auprc": round(test_auprc, 4),
        "test_roc_auc": round(test_roc_auc, 4),
        "test_precision": round(test_prec, 4),
        "test_recall": round(test_rec, 4),
        "test_f1_at_optimal_threshold": round(test_f1, 4),
        "test_true_positives": int(tp),
        "test_false_positives": int(fp),
        "test_false_negatives": int(fn),
        "test_true_negatives": int(tn),
    }

    with open(REPORT_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved FastAI Report -> {REPORT_OUT_PATH}")

    # 10. STEP 5: Log to MLflow
    print("\n" + "=" * 72)
    print("STEP 5: Log FastAI Run to MLflow Tracking Server")
    print("=" * 72)
    try:
        print(f"MLflow Tracking URI: {TRACKING_URI}")
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name="fastai_tabular_nn") as run:
            run_id = run.info.run_id
            print(f"Active MLflow Run ID: {run_id}")

            # Tags
            mlflow.set_tags({
                "model_type": "fastai_tabular",
                "framework": "fastai",
                "architecture": "MLP_Entity_Embeddings",
                "role": "deep_learning_tabular_benchmark",
                "leakage_free": "true",
                "threshold_tuning": "validation_set_only",
            })

            # Parameters
            mlflow.log_params({
                "layers": "[200, 100]",
                "epochs": 3,
                "learning_rate": 0.001,
                "batch_size": 512,
                "train_samples_used": len(sample_train),
                "val_samples": len(val_df),
                "test_samples": len(test_df),
                "feature_count": len(all_features),
                "threshold_selection_dataset": "validation",
                "decision_threshold": chosen_threshold,
            })

            # Validation metrics
            mlflow.log_metrics({
                "val_auprc": val_auprc,
                "val_roc_auc": val_roc_auc,
                "val_f1": val_best_f1,
                "val_precision": val_prec,
                "val_recall": val_rec,
            })

            # Final Out-of-Time Test metrics
            mlflow.log_metrics({
                "test_auprc": test_auprc,
                "test_roc_auc": test_roc_auc,
                "optimal_threshold": chosen_threshold,
                "test_precision": test_prec,
                "test_recall": test_rec,
                "test_f1": test_f1,
                "test_tp": tp,
                "test_fp": fp,
                "test_fn": fn,
                "test_tn": tn,
                "training_time_seconds": train_time,
                "inference_latency_ms": inf_latency_ms,
            })

            # Artifacts
            mlflow.log_artifact(pr_curve_path, artifact_path="plots")
            mlflow.log_artifact(cm_plot_path, artifact_path="plots")
            mlflow.log_artifact(REPORT_OUT_PATH, artifact_path="evaluation")
            mlflow.log_artifact(MODEL_OUT_PATH, artifact_path="model")
            print("Successfully logged FastAI model, parameters, metrics, and visual artifacts to MLflow!")
    except Exception as e:
        print(f"MLflow logging notice: {e}")

    return report


if __name__ == "__main__":
    train_fastai_tabular_pipeline()
