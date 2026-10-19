"""
Train & Evaluate Leakage-Free XGBoost Fraud Detection Model with MLflow Tracking
================================================================================

Strict Evaluation Methodology:
------------------------------
1. Exclusively pre-authorization features (zero post-settlement leakage).
2. Chronological time-aware dataset splits without future look-ahead bias:
   - Train: Days 1-24 (step <= 576), 6,201,194 rows.
   - Validation: Days 25-27 (576 < step <= 648), 80,316 rows (used for early stopping & threshold tuning).
   - Test (Out-of-Time): Days 28-31 (step > 648), 81,110 rows (pure final evaluation).
3. The decision threshold is selected STRICTLY on validation data to maximize F1,
   and applied ONCE to the untouched test set.
4. Experiment, hyperparameters, metrics, and visual artifacts logged to MLflow
   under experiment 'fraudwatch-preauth-models'.

Run from project root:
    python src/05_train_leakage_free_model.py
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
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db")

EXPERIMENT_NAME = "fraudwatch-preauth-models"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{os.path.abspath(DB_PATH).replace(os.sep, '/')}")


def train_leakage_free_pipeline() -> Dict[str, Any]:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    print("=" * 72)
    print("STEP 1: Load Chronological Dataset Partitions")
    print("=" * 72)
    df_train = pd.read_parquet(os.path.join(DATA_DIR, "leakage_free_train.parquet"))
    df_val = pd.read_parquet(os.path.join(DATA_DIR, "leakage_free_val.parquet"))
    df_test = pd.read_parquet(os.path.join(DATA_DIR, "leakage_free_test.parquet"))

    drop_cols = ["step", "isFraud"]
    feature_cols = [c for c in df_train.columns if c not in drop_cols]

    X_train, y_train = df_train[feature_cols], df_train["isFraud"].to_numpy()
    X_val, y_val = df_val[feature_cols], df_val["isFraud"].to_numpy()
    X_test, y_test = df_test[feature_cols], df_test["isFraud"].to_numpy()

    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Train rows : {len(X_train):,} | Frauds: {y_train.sum():,} ({y_train.mean()*100:.4f}%) | Steps: {df_train['step'].min()}..{df_train['step'].max()}")
    print(f"Val rows   : {len(X_val):,}   | Frauds: {y_val.sum():,}   ({y_val.mean()*100:.4f}%) | Steps: {df_val['step'].min()}..{df_val['step'].max()}")
    print(f"Test rows  : {len(X_test):,}  | Frauds: {y_test.sum():,}  ({y_test.mean()*100:.4f}%) | Steps: {df_test['step'].min()}..{df_test['step'].max()}")

    # Class weighting
    pos_train = int(y_train.sum())
    neg_train = int(len(y_train) - pos_train)
    scale_pos_weight = neg_train / pos_train if pos_train > 0 else 1.0
    print(f"\nClass imbalance scale_pos_weight = {scale_pos_weight:.2f}")

    print("\n" + "=" * 72)
    print("STEP 2: Train Pre-Authorization XGBoost Classifier")
    print("=" * 72)
    hyperparams = {
        "n_estimators": 300,
        "learning_rate": 0.08,
        "max_depth": 6,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": float(scale_pos_weight),
        "eval_metric": "aucpr",
        "early_stopping_rounds": 30,
        "n_jobs": -1,
        "random_state": 42,
    }

    model = XGBClassifier(**hyperparams)

    t0 = time.perf_counter()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    train_time = time.perf_counter() - t0
    best_iter = getattr(model, "best_iteration", model.n_estimators)
    print(f"\nTraining completed in {train_time:.2f}s (Best iteration: {best_iter})")

    print("\n" + "=" * 72)
    print("STEP 3: Tune Decision Threshold Strictly on Validation Set")
    print("=" * 72)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    val_auprc = float(average_precision_score(y_val, y_val_proba))
    val_roc_auc = float(roc_auc_score(y_val, y_val_proba))

    prec_v, rec_v, thresh_v = precision_recall_curve(y_val, y_val_proba)
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

    print("\n" + "=" * 72)
    print("STEP 4: Evaluate Once on Untouched Out-of-Time Test Set (Days 28-31)")
    print("=" * 72)
    t_inf = time.perf_counter()
    y_test_proba = model.predict_proba(X_test)[:, 1]
    inf_time = time.perf_counter() - t_inf
    inf_latency_ms = (inf_time / len(X_test)) * 1000.0

    test_auprc = float(average_precision_score(y_test, y_test_proba))
    test_roc_auc = float(roc_auc_score(y_test, y_test_proba))
    test_prevalence = float(y_test.mean())

    # Apply the untouched validation threshold
    y_test_pred = (y_test_proba >= chosen_threshold).astype(int)
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    test_prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    test_rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    test_f1 = float(2 * test_prec * test_rec / (test_prec + test_rec)) if (test_prec + test_rec) > 0 else 0.0

    print(f"Test Inference Latency:            {inf_latency_ms:.4f} ms/row ({len(X_test)/inf_time:,.0f} rows/s)")
    print(f"Test AUPRC (Average Precision):    {test_auprc:.4f} (Base fraud rate = {test_prevalence*100:.2f}%)")
    print(f"Test ROC-AUC:                      {test_roc_auc:.4f}")
    print(f"Evaluated Threshold (from Val):    {chosen_threshold:.4f}")
    print(f"Test Precision:                    {test_prec:.4f} ({test_prec*100:.2f}%)")
    print(f"Test Recall:                       {test_rec:.4f} ({test_rec*100:.2f}%)")
    print(f"Test F1 Score:                     {test_f1:.4f}")
    print(f"Confusion Matrix:                  TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # Feature importances
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 72)
    print("Top Feature Importances (Pre-Authorization):")
    print("=" * 72)
    print(fi.to_string(index=False))

    # Step 5: Generate Plot Artifacts
    pr_curve_path = os.path.join(REPORTS_DIR, "leakage_free_pr_curve.png")
    plt.figure(figsize=(8, 6))
    prec_t, rec_t, _ = precision_recall_curve(y_test, y_test_proba)
    plt.plot(rec_t, prec_t, color="#1E3A8A", lw=2, label=f"Pre-Auth XGBoost (AUPRC = {test_auprc:.4f})")
    plt.scatter([test_rec], [test_prec], color="#DC2626", s=100, zorder=5, label=f"Val Threshold {chosen_threshold:.2f} (F1 = {test_f1:.4f})")
    plt.title("Out-of-Time Test Precision-Recall Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left", fontsize=11)
    plt.tight_layout()
    plt.savefig(pr_curve_path, dpi=150)
    plt.close()

    cm_plot_path = os.path.join(REPORTS_DIR, "leakage_free_confusion_matrix.png")
    plt.figure(figsize=(6, 5))
    cm_matrix = np.array([[tn, fp], [fn, tp]])
    plt.imshow(cm_matrix, cmap="Blues", interpolation="nearest")
    plt.title(f"Test Confusion Matrix (Thresh = {chosen_threshold:.2f})", fontsize=12, fontweight="bold")
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

    fi_plot_path = os.path.join(REPORTS_DIR, "leakage_free_feature_importance.png")
    plt.figure(figsize=(10, 6))
    plt.barh(fi["feature"][::-1], fi["importance"][::-1], color="#2563EB")
    plt.title("Pre-Authorization Feature Importance (XGBoost)", fontsize=13, fontweight="bold")
    plt.xlabel("Relative Importance", fontsize=11)
    plt.tight_layout()
    plt.savefig(fi_plot_path, dpi=150)
    plt.close()

    # Save artifacts
    out_model = os.path.join(MODELS_DIR, "leakage_free_xgb_model.json")
    model.save_model(out_model)
    print(f"\nSaved Leakage-Free Model -> {out_model}")

    fi_path = os.path.join(REPORTS_DIR, "leakage_free_feature_importance.csv")
    fi.to_csv(fi_path, index=False)
    print(f"Saved Feature Importance -> {fi_path}")

    metadata = {
        "model_name": "leakage_free_xgb_model",
        "description": "Pre-authorization XGBoost model evaluated on chronological out-of-time test set with validation-tuned threshold",
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "feature_dtypes": {c: str(X_train[c].dtype) for c in feature_cols},
        "target": "isFraud",
        "scale_pos_weight": float(scale_pos_weight),
        "best_iteration": int(best_iter),
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
        "test_fraud_prevalence": test_prevalence,
        "decision_threshold": round(chosen_threshold, 4),
        "threshold_selection_dataset": "validation (steps 577-648)",
        "hyperparameters": {k: v for k, v in hyperparams.items() if k != "scale_pos_weight"},
        "metrics": {
            "val_auprc": round(val_auprc, 4),
            "val_roc_auc": round(val_roc_auc, 4),
            "val_f1_at_optimal_threshold": round(val_best_f1, 4),
            "test_auprc": round(test_auprc, 4),
            "test_roc_auc": round(test_roc_auc, 4),
            "decision_threshold": round(chosen_threshold, 4),
            "test_precision": round(test_prec, 4),
            "test_recall": round(test_rec, 4),
            "test_f1_at_optimal_threshold": round(test_f1, 4),
            "test_true_positives": int(tp),
            "test_false_positives": int(fp),
            "test_false_negatives": int(fn),
            "test_true_negatives": int(tn),
            "training_time_seconds": round(train_time, 2),
            "inference_latency_ms": round(inf_latency_ms, 4),
        },
    }
    meta_path = os.path.join(MODELS_DIR, "leakage_free_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved Metadata -> {meta_path}")

    eval_path = os.path.join(REPORTS_DIR, "leakage_free_evaluation_report.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(metadata["metrics"], f, indent=2)
    print(f"Saved Evaluation Report -> {eval_path}")

    # STEP 6: Log to MLflow
    print("\n" + "=" * 72)
    print("STEP 6: Log Run to MLflow Tracking Server")
    print("=" * 72)
    try:
        print(f"MLflow Tracking URI: {TRACKING_URI}")
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name="xgboost_preauth_champion") as run:
            run_id = run.info.run_id
            print(f"Active MLflow Run ID: {run_id}")

            # Tags
            mlflow.set_tags({
                "model_type": "xgboost",
                "role": "champion_gatekeeper",
                "leakage_free": "true",
                "pre_authorization": "true",
                "split_strategy": "chronological_time_aware",
                "threshold_tuning": "validation_set_only",
                "dataset": "PaySim_6.36M",
            })

            # Parameters
            mlflow.log_params(metadata["hyperparameters"])
            mlflow.log_params({
                "scale_pos_weight": metadata["scale_pos_weight"],
                "train_rows": metadata["train_rows"],
                "val_rows": metadata["val_rows"],
                "test_rows": metadata["test_rows"],
                "feature_count": metadata["feature_count"],
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

            # Visual Artifacts
            mlflow.log_artifact(pr_curve_path, artifact_path="plots")
            mlflow.log_artifact(cm_plot_path, artifact_path="plots")
            mlflow.log_artifact(fi_plot_path, artifact_path="plots")
            mlflow.log_artifact(fi_path, artifact_path="feature_importance")
            mlflow.log_artifact(meta_path, artifact_path="metadata")
            mlflow.log_artifact(eval_path, artifact_path="evaluation")

            # Model Artifact
            mlflow.xgboost.log_model(model, artifact_path="model")
            print("Successfully logged XGBoost model, parameters, metrics, and visual artifacts to MLflow!")
    except Exception as e:
        print(f"MLflow logging notice: {e}")

    return metadata


if __name__ == "__main__":
    train_leakage_free_pipeline()
