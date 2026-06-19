"""
Step 03 - Train & Evaluate XGBoost
==================================

Trains an XGBoost classifier on the engineered PaySim features, evaluates
it with fraud-class metrics (Precision / Recall / F1 / AUPRC), and writes
the trained model to ``models/fraud_xgb_model.json`` along with a feature
importance CSV to ``reports/feature_importance.csv``.

Run from the project root:
    python src/03_train_model.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    precision_recall_fscore_support,
)

from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mem_mb(df: pd.DataFrame) -> float:
    """Return the deep memory usage of a DataFrame in megabytes."""
    return float(df.memory_usage(deep=True).sum() / (1024 ** 2))


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------
def _load_xy(X_path: str, y_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load engineered feature matrix X and target vector y from Parquet."""
    if not os.path.isfile(X_path):
        raise FileNotFoundError(f"X_features not found at: {X_path}")
    if not os.path.isfile(y_path):
        raise FileNotFoundError(f"y_target not found at: {y_path}")

    print(f"Loading X from: {X_path}")
    X = pd.read_parquet(X_path)
    print(f"   shape={X.shape[0]:,} x {X.shape[1]} | "
          f"memory={_mem_mb(X):,.2f} MB")

    print(f"Loading y from: {y_path}")
    y_df = pd.read_parquet(y_path)
    if isinstance(y_df, pd.DataFrame):
        y = (y_df["isFraud"] if "isFraud" in y_df.columns else y_df.iloc[:, 0]).astype("int8")
    else:
        y = y_df.astype("int8")

    if len(X) != len(y):
        raise ValueError(
            f"X rows ({len(X):,}) and y rows ({len(y):,}) do not match."
        )

    fraud_rate = float(y.mean()) * 100
    print(f"   positives={int(y.sum()):,} | "
          f"negatives={int((1 - y).sum()):,} | "
          f"fraud_rate={fraud_rate:.4f}%")
    return X, y


def _split(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 train/test split - preserves fraud ratio on both sides."""
    print(f"\nSplitting with stratify=y, test_size={test_size}, seed={seed} ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    print(f"   X_train: {X_train.shape[0]:,} rows | fraud={int(y_train.sum()):,} "
          f"({y_train.mean() * 100:.4f}%)")
    print(f"   X_test : {X_test.shape[0]:,} rows | fraud={int(y_test.sum()):,} "
          f"({y_test.mean() * 100:.4f}%)")
    return X_train, X_test, y_train, y_test


def _build_model(scale_pos_weight: float) -> XGBClassifier:
    """Construct an XGBClassifier tuned for tabular fraud detection."""
    return XGBClassifier(
        n_estimators=400,
        learning_rate=0.1,
        max_depth=8,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        device="cuda",
        n_jobs=-1,
        random_state=42,
    )


def _evaluate(
    model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, Any]:
    """Compute fraud-class metrics + AUPRC and print a detailed report."""
    print("\nGenerating predictions on the test set ...")
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    inference_s = time.perf_counter() - t0
    print(f"Inference took {inference_s:.2f}s "
          f"({len(X_test) / inference_s:,.0f} rows/s)")

    print("\n" + "=" * 72)
    print("Classification report (test set)")
    print("=" * 72)
    print(classification_report(
        y_test, y_pred,
        target_names=["Legit (0)", "Fraud (1)"],
        digits=4,
        zero_division=0,
    ))

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[1], zero_division=0
    )
    auprc = float(average_precision_score(y_test, y_proba))

    print("=" * 72)
    print("Fraud-class metrics (positive class = 1)")
    print("-" * 72)
    print(f"   Precision : {prec[0]:.4f}")
    print(f"   Recall    : {rec[0]:.4f}")
    print(f"   F1-Score  : {f1[0]:.4f}")
    print(f"   AUPRC     : {auprc:.4f}   <-- PRIMARY METRIC")
    print("=" * 72)

    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "precision_fraud": float(prec[0]),
        "recall_fraud": float(rec[0]),
        "f1_fraud": float(f1[0]),
        "auprc": auprc,
        "inference_seconds": inference_s,
    }


def _feature_importance(
    model: XGBClassifier, X: pd.DataFrame, top_k: int = 5
) -> pd.DataFrame:
    """Extract XGBoost feature importances and return the top_k rows."""
    fi = (
        pd.DataFrame({
            "feature": list(X.columns),
            "importance": model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\n" + "=" * 72)
    print(f"Top {top_k} features driving fraud detection")
    print("=" * 72)
    print(fi.head(top_k).to_string(index=False))
    print("=" * 72)
    return fi


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def train_fraud_model(X_path: str, y_path: str) -> Dict[str, Any]:
    """Train an XGBoost classifier on the engineered PaySim features."""
    X, y = _load_xy(X_path, y_path)
    X_train, X_test, y_train, y_test = _split(X, y)

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / n_pos if n_pos else 1.0
    print(f"\nClass-imbalance handling:")
    print(f"   train positives={n_pos:,} | negatives={n_neg:,}")
    print(f"   scale_pos_weight = neg / pos = {scale_pos_weight:.2f}")

    model = _build_model(scale_pos_weight)
    print(f"\nTraining XGBoost on {len(X_train):,} rows x {X_train.shape[1]} cols ...")
    t0 = time.perf_counter()
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )
    train_s = time.perf_counter() - t0
    print(f"Training took {train_s:.2f}s")

    metrics = _evaluate(model, X_test, y_test)
    fi = _feature_importance(model, X, top_k=5)

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "metrics": metrics,
        "feature_importance": fi,
        "scale_pos_weight": scale_pos_weight,
        "training_seconds": train_s,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc_dir = os.path.join(project_root, "data", "processed")
    models_dir = os.path.join(project_root, "models")
    reports_dir = os.path.join(project_root, "reports")

    X_path = os.path.join(proc_dir, "X_features.parquet")
    y_path = os.path.join(proc_dir, "y_target.parquet")

    artifact = train_fraud_model(X_path, y_path)

    out_model = os.path.join(models_dir, "fraud_xgb_model.json")
    artifact["model"].save_model(out_model)
    print(f"\nSaved trained model -> {out_model}")

    fi_path = os.path.join(reports_dir, "feature_importance.csv")
    artifact["feature_importance"].to_csv(fi_path, index=False)
    print(f"Saved feature importance -> {fi_path}")
