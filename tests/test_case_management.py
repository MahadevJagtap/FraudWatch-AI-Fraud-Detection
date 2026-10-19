"""
Automated tests for Case Management & Analyst Feedback persistence.
"""

from __future__ import annotations

import os
import sqlite3
import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.case_store import (
    init_case_db,
    insert_or_update_alert,
    get_cases,
    get_case_by_id,
    update_case_review,
    get_case_metrics,
    sync_alerts_from_jsonl,
)


@pytest.fixture
def temp_case_db(tmp_path):
    db_path = str(tmp_path / "test_cases.db")
    init_case_db(db_path)
    return db_path


def test_case_db_initialization(temp_case_db):
    conn = sqlite3.connect(temp_case_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cursor.fetchall()}
    assert "cases" in tables
    assert "case_audit_log" in tables
    conn.close()


def test_insert_and_query_case(temp_case_db):
    alert = {
        "transaction_id": "tx_test_1001",
        "timestamp": 1700000000.0,
        "step": 45,
        "type": "TRANSFER",
        "amount": 250000.0,
        "nameOrig": "C111222333",
        "nameDest": "C444555666",
        "orig_drain_ratio": 1.0,
        "fraud_probability": 0.985,
        "risk_level": "CRITICAL",
        "decision_threshold": 0.70,
    }

    insert_or_update_alert(alert, db_path=temp_case_db)

    # Query without filter
    cases = get_cases(db_path=temp_case_db)
    assert len(cases) == 1
    assert cases[0]["transaction_id"] == "tx_test_1001"
    assert cases[0]["case_status"] == "OPEN"
    assert cases[0]["amount"] == 250000.0

    # Query with status filter
    open_cases = get_cases(status="OPEN", db_path=temp_case_db)
    assert len(open_cases) == 1
    closed_cases = get_cases(status="CONFIRMED_FRAUD", db_path=temp_case_db)
    assert len(closed_cases) == 0

    # Query with search
    search_cases = get_cases(search="C111222", db_path=temp_case_db)
    assert len(search_cases) == 1


def test_update_case_review_and_audit_trail(temp_case_db):
    alert = {
        "transaction_id": "tx_audit_test",
        "timestamp": 1700000000.0,
        "step": 50,
        "type": "TRANSFER",
        "amount": 75000.0,
        "nameOrig": "C999",
        "nameDest": "C888",
        "orig_drain_ratio": 0.99,
        "fraud_probability": 0.95,
        "risk_level": "HIGH",
    }
    insert_or_update_alert(alert, db_path=temp_case_db)

    # 1. First review: transition to INVESTIGATING
    ok1 = update_case_review(
        transaction_id="tx_audit_test",
        new_status="INVESTIGATING",
        disposition="PENDING_CUSTOMER_CONTACT",
        notes="Called cardholder for verification",
        analyst_id="analyst_alice",
        db_path=temp_case_db
    )
    assert ok1 is True

    # 2. Second review: transition to CONFIRMED_FRAUD
    ok2 = update_case_review(
        transaction_id="tx_audit_test",
        new_status="CONFIRMED_FRAUD",
        disposition="ACCOUNT_TAKEOVER_FULL_DRAIN",
        notes="Customer confirmed transaction was unauthorized",
        analyst_id="analyst_alice",
        db_path=temp_case_db
    )
    assert ok2 is True

    # Verify case record and audit log
    case_detail = get_case_by_id("tx_audit_test", db_path=temp_case_db)
    assert case_detail is not None
    assert case_detail["case_status"] == "CONFIRMED_FRAUD"
    assert case_detail["analyst_disposition"] == "ACCOUNT_TAKEOVER_FULL_DRAIN"

    history = case_detail["audit_history"]
    assert len(history) == 2
    assert history[0]["previous_status"] == "OPEN"
    assert history[0]["new_status"] == "INVESTIGATING"
    assert history[1]["previous_status"] == "INVESTIGATING"
    assert history[1]["new_status"] == "CONFIRMED_FRAUD"


def test_case_metrics(temp_case_db):
    for i in range(3):
        insert_or_update_alert({
            "transaction_id": f"tx_open_{i}",
            "amount": 1000.0 * (i + 1),
            "fraud_probability": 0.8
        }, db_path=temp_case_db)

    insert_or_update_alert({
        "transaction_id": "tx_confirmed_1",
        "amount": 50000.0,
        "fraud_probability": 0.99
    }, db_path=temp_case_db)
    update_case_review(
        "tx_confirmed_1", "CONFIRMED_FRAUD", "ATO", "Confirmed", db_path=temp_case_db
    )

    metrics = get_case_metrics(db_path=temp_case_db)
    assert metrics["total"] == 4
    assert metrics["open"] == 3
    assert metrics["confirmed_fraud"] == 1
    assert metrics["investigating"] == 0
    assert metrics["false_positive"] == 0


def test_api_case_management_endpoints():
    client = TestClient(app)

    # 1. Fetch cases
    resp = client.get("/platform/cases")
    assert resp.status_code == 200
    data = resp.json()
    assert "cases" in data
    assert "count" in data

    # 2. Fetch metrics
    resp_m = client.get("/platform/cases/summary/metrics")
    assert resp_m.status_code == 200
    metrics = resp_m.json()
    assert "total" in metrics
    assert "open" in metrics

    # 3. Test 404 on nonexistent case
    resp_404 = client.get("/platform/cases/nonexistent_tx_999")
    assert resp_404.status_code == 404


def test_end_to_end_scoring_to_analyst_case_review_flow():
    from api.service import scoring_service
    import time
    if not scoring_service.is_loaded:
        scoring_service.load()

    client = TestClient(app)
    tx_id = f"TX-E2E-{time.time():.4f}"

    # 1. Post a fraudulent transaction to scoring API
    fraud_payload = {
        "step": 741,
        "type": "TRANSFER",
        "amount": 450000.0,
        "nameOrig": "C_VICTIM_99",
        "oldbalanceOrg": 450000.0,
        "nameDest": "C_FRAUD_99",
        "oldbalanceDest": 0.0,
        "mode": "live_authorization",
        "transaction_id": tx_id,
        "use_feature_store": True,
    }
    score_resp = client.post("/predict/preauth", json=fraud_payload)
    assert score_resp.status_code == 200
    score_data = score_resp.json()
    assert score_data["is_fraud"] is True
    assert score_data["risk_level"] in ("HIGH", "CRITICAL")

    # 2. Case should automatically exist in case management store
    case_resp = client.get(f"/platform/cases/{tx_id}")
    assert case_resp.status_code == 200
    case_data = case_resp.json()
    assert case_data["transaction_id"] == tx_id
    assert case_data["case_status"] == "OPEN"
    assert case_data["amount"] == 450000.0

    # 3. Analyst steps in: marks case as INVESTIGATING
    review_resp1 = client.post(
        f"/platform/cases/{tx_id}/review",
        json={
            "status": "INVESTIGATING",
            "disposition": "CONTACTING_CUSTOMER",
            "notes": "Large drain detected; contacting cardholder",
            "analyst_id": "analyst_bob",
        }
    )
    assert review_resp1.status_code == 200

    # 4. Analyst confirms fraud
    review_resp2 = client.post(
        f"/platform/cases/{tx_id}/review",
        json={
            "status": "CONFIRMED_FRAUD",
            "disposition": "ACCOUNT_TAKEOVER_FULL_DRAIN",
            "notes": "Customer confirmed unauthorized transaction",
            "analyst_id": "analyst_bob",
        }
    )
    assert review_resp2.status_code == 200

    # 5. Fetch updated case and verify persistence & audit trail
    final_case_resp = client.get(f"/platform/cases/{tx_id}")
    assert final_case_resp.status_code == 200
    final_case = final_case_resp.json()
    assert final_case["case_status"] == "CONFIRMED_FRAUD"
    assert final_case["analyst_disposition"] == "ACCOUNT_TAKEOVER_FULL_DRAIN"

    history = final_case["audit_history"]
    assert len(history) >= 2
    assert history[-2]["new_status"] == "INVESTIGATING"
    assert history[-1]["new_status"] == "CONFIRMED_FRAUD"

