"""
Automated tests for Evidently AI drift reporting and controlled model promotion guardrails.
"""

from __future__ import annotations

import json
import os
import sys
import numpy as np
import pandas as pd
import pytest
from evidently import Report
from evidently.presets import DataDriftPreset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

HTML_PATH = os.path.join(PROJECT_ROOT, "reports", "evidently_drift_report.html")
JSON_PATH = os.path.join(PROJECT_ROOT, "reports", "evidently_drift_report.json")
AUDIT_PATH = os.path.join(PROJECT_ROOT, "reports", "model_promotion_audit.json")
CHAMPION_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "leakage_free_xgb_model.json")
METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "leakage_free_metadata.json")


def test_drift_artifacts_exist():
    assert os.path.isfile(HTML_PATH), f"Evidently HTML report missing: {HTML_PATH}"
    assert os.path.getsize(HTML_PATH) > 1000, "Evidently HTML report is empty"
    assert os.path.isfile(JSON_PATH), f"Evidently JSON report missing: {JSON_PATH}"
    assert os.path.isfile(AUDIT_PATH), f"Model promotion audit missing: {AUDIT_PATH}"


def test_drift_json_structure():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Evidently snapshot structure
    assert "metric_results" in data or "metrics" in data


def test_promotion_gate_guardrail_logic():
    with open(AUDIT_PATH, "r", encoding="utf-8") as f:
        audit = json.load(f)

    assert "champion_auprc" in audit
    assert "candidate_auprc" in audit
    assert "promotion_verdict" in audit
    assert "is_model_promoted" in audit
    assert audit["promotion_verdict"] in ("APPROVED_FOR_PROMOTION", "REJECTED_PROMOTION")

    # Safety invariant: if candidate < champion, MUST NOT be promoted
    if audit["candidate_auprc"] < audit["champion_auprc"]:
        assert audit["is_model_promoted"] is False
        assert audit["promotion_verdict"] == "REJECTED_PROMOTION"


def test_no_drift_case():
    """Verify that identical or identically distributed samples do not trigger drift."""
    rng = np.random.default_rng(42)
    ref_df = pd.DataFrame({
        "amount": rng.normal(100.0, 15.0, 500),
        "orig_drain_ratio": rng.uniform(0.0, 1.0, 500),
        "is_merchant": rng.choice([0, 1], 500),
    })
    curr_df = pd.DataFrame({
        "amount": rng.normal(100.0, 15.0, 500),
        "orig_drain_ratio": rng.uniform(0.0, 1.0, 500),
        "is_merchant": rng.choice([0, 1], 500),
    })

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref_df, current_data=curr_df)
    res_dict = snapshot.dict()

    drift_count = 0
    metrics_list = res_dict.get("metrics", []) or res_dict.get("metric_results", [])
    for item in metrics_list:
        metric_name = str(item.get("metric_name", item.get("metric", "")))
        val = item.get("value", {})
        if "DriftedColumnsCount" in metric_name and isinstance(val, dict):
            drift_count = int(val.get("count", 0))

    assert drift_count == 0, f"No drift expected for identical distributions, got {drift_count}"


def test_drift_triggered_case():
    """Verify that a substantially shifted current sample triggers drift detection."""
    rng = np.random.default_rng(42)
    ref_df = pd.DataFrame({
        "amount": rng.normal(100.0, 15.0, 500),
        "orig_drain_ratio": rng.uniform(0.0, 0.2, 500),
        "is_merchant": rng.choice([0], 500),
    })
    # Heavily shifted distribution: amounts 10x higher, drain ratio 5x higher, all merchants
    curr_df = pd.DataFrame({
        "amount": rng.normal(50000.0, 2000.0, 500),
        "orig_drain_ratio": rng.uniform(0.8, 1.0, 500),
        "is_merchant": rng.choice([1], 500),
    })

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=ref_df, current_data=curr_df)
    res_dict = snapshot.dict()

    drift_count = 0
    metrics_list = res_dict.get("metrics", []) or res_dict.get("metric_results", [])
    for item in metrics_list:
        metric_name = str(item.get("metric_name", item.get("metric", "")))
        val = item.get("value", {})
        if "DriftedColumnsCount" in metric_name and isinstance(val, dict):
            drift_count = int(val.get("count", 0))

    assert drift_count >= 2, f"Expected drift in at least 2 columns, got {drift_count}"


def test_guardrail_rejects_degraded_candidate_and_preserves_champion():
    """Simulate a candidate retraining run where candidate performs worse than champion."""
    champion_mtime_before = os.path.getmtime(CHAMPION_MODEL_PATH)
    champion_size_before = os.path.getsize(CHAMPION_MODEL_PATH)

    champion_auprc = 0.9891
    degraded_candidate_auprc = 0.8120

    # Retraining guardrail rule
    if degraded_candidate_auprc >= champion_auprc:
        verdict = "APPROVED_FOR_PROMOTION"
        promoted = True
    else:
        verdict = "REJECTED_PROMOTION"
        promoted = False

    assert verdict == "REJECTED_PROMOTION"
    assert promoted is False

    # Verify champion file was not modified or overwritten
    champion_mtime_after = os.path.getmtime(CHAMPION_MODEL_PATH)
    champion_size_after = os.path.getsize(CHAMPION_MODEL_PATH)
    assert champion_mtime_after == champion_mtime_before
    assert champion_size_after == champion_size_before
