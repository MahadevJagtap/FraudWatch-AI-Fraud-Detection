"""
Automated unit tests for Kafka Streaming components, invalid-event recovery, and idempotency.
"""

from __future__ import annotations

import json
import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from streaming.config import BOOTSTRAP_SERVERS, TRANSACTIONS_TOPIC, ALERTS_TOPIC
from streaming.producer import transaction_generator
from streaming.consumer_worker import (
    load_scoring_artifacts,
    score_transaction,
    record_alert,
    ALERTS_LOG,
    _SEEN_ALERT_IDS,
)
from api.case_store import init_case_db, get_case_by_id, update_case_review, get_connection


def test_kafka_config():
    assert BOOTSTRAP_SERVERS is not None
    assert "raw" in TRANSACTIONS_TOPIC
    assert "alerts" in ALERTS_TOPIC or "fraud" in ALERTS_TOPIC


def test_kafka_transaction_generator():
    gen = transaction_generator(max_events=5)
    events = list(gen)
    assert len(events) == 5

    sample = events[0]
    assert "transaction_id" in sample
    assert "step" in sample
    assert "type" in sample
    assert "amount" in sample
    assert "nameOrig" in sample
    assert "nameDest" in sample
    assert "oldbalanceOrg" in sample
    assert "oldbalanceDest" in sample

    # Verify no post-settlement leaky keys
    for forbidden in ["newbalanceOrig", "newbalanceDest", "errorBalanceOrig", "isFlaggedFraud"]:
        assert forbidden not in sample


def test_scoring_worker_fraud_detection():
    clf, features, threshold = load_scoring_artifacts()

    # Case 1: High-risk account-takeover drain
    fraud_tx = {
        "transaction_id": "tx_test_fraud_001",
        "step": 48,
        "type": "TRANSFER",
        "amount": 250000.0,
        "nameOrig": "C1111111",
        "oldbalanceOrg": 250000.0,
        "nameDest": "C2222222",
        "oldbalanceDest": 0.0
    }
    result_fraud = score_transaction(fraud_tx, clf, features, threshold)
    assert result_fraud["is_fraud_predicted"] is True
    assert result_fraud["fraud_probability"] >= threshold
    assert result_fraud["risk_level"] in ("HIGH", "CRITICAL")
    assert result_fraud["orig_drain_ratio"] > 0.95

    # Case 2: Low-risk merchant payment
    legit_tx = {
        "transaction_id": "tx_test_legit_001",
        "step": 14,
        "type": "PAYMENT",
        "amount": 35.0,
        "nameOrig": "C3333333",
        "oldbalanceOrg": 1200.0,
        "nameDest": "M9999999",
        "oldbalanceDest": 0.0
    }
    result_legit = score_transaction(legit_tx, clf, features, threshold)
    assert result_legit["is_fraud_predicted"] is False
    assert result_legit["fraud_probability"] < 0.25
    assert result_legit["risk_level"] == "LOW"


def test_record_alert_log_and_duplicate_idempotency():
    import time
    _SEEN_ALERT_IDS.clear()
    tx_id = f"tx_unit_test_dedup_{time.time():.4f}"
    sample_alert = {
        "transaction_id": tx_id,
        "timestamp": 123456789.0,
        "fraud_probability": 0.995,
        "decision_threshold": 0.70,
        "is_fraud_predicted": True,
        "risk_level": "CRITICAL",
        "amount": 500000.0,
        "type": "TRANSFER",
        "orig_drain_ratio": 1.0,
        "nameOrig": "C999",
        "nameDest": "C888"
    }

    # First record should succeed
    is_new = record_alert(sample_alert, producer=None)
    assert is_new is True

    # Analyst marks this case as INVESTIGATING
    update_case_review(
        transaction_id=tx_id,
        new_status="INVESTIGATING",
        disposition="SUSPECTED_TAKEOVER",
        notes="Under analyst investigation",
        analyst_id="lead_analyst_1",
    )
    case_before = get_case_by_id(tx_id)
    assert case_before["case_status"] == "INVESTIGATING"

    # Second record (replay or duplicate event) should be detected as duplicate
    is_new_again = record_alert(sample_alert, producer=None)
    assert is_new_again is False

    # Analyst case in database must NOT be overwritten or reset to OPEN
    case_after = get_case_by_id(tx_id)
    assert case_after["case_status"] == "INVESTIGATING"
    assert case_after["analyst_notes"] == "Under analyst investigation"


def test_invalid_event_handling():
    clf, features, threshold = load_scoring_artifacts()

    # Empty payload
    invalid_empty = {}
    res = score_transaction(invalid_empty, clf, features, threshold)
    assert "transaction_id" in res
    assert res["amount"] == 0.0

    # Malformed non-numeric values
    malformed = {
        "transaction_id": "tx_malformed",
        "step": "bad_step",
        "type": 12345,
        "amount": "not_a_number",
    }
    with pytest.raises(Exception):
        score_transaction(malformed, clf, features, threshold)


def test_bounded_simulated_stream_processing():
    """Verify end-to-end event scoring loop on a bounded batch including invalid and duplicate events."""
    import time
    clf, features, threshold = load_scoring_artifacts()
    uid = f"{time.time():.4f}"

    batch_events = [
        # 1. High-risk fraud
        {
            "transaction_id": f"tx_stream_01_{uid}",
            "step": 50,
            "type": "TRANSFER",
            "amount": 300000.0,
            "nameOrig": "C_VICTIM_1",
            "oldbalanceOrg": 300000.0,
            "nameDest": "C_MULE_1",
            "oldbalanceDest": 0.0,
        },
        # 2. Duplicate of event 1
        {
            "transaction_id": f"tx_stream_01_{uid}",
            "step": 50,
            "type": "TRANSFER",
            "amount": 300000.0,
            "nameOrig": "C_VICTIM_1",
            "oldbalanceOrg": 300000.0,
            "nameDest": "C_MULE_1",
            "oldbalanceDest": 0.0,
        },
        # 3. Normal merchant payment
        {
            "transaction_id": "tx_stream_02",
            "step": 2,
            "type": "PAYMENT",
            "amount": 45.0,
            "nameOrig": "C_USER_2",
            "oldbalanceOrg": 500.0,
            "nameDest": "M_STORE_2",
            "oldbalanceDest": 0.0,
        },
        # 4. Malformed event that raises an exception
        "INVALID_STRING_EVENT_NOT_A_DICT",
        # 5. Normal cash in
        {
            "transaction_id": "tx_stream_03",
            "step": 10,
            "type": "CASH_IN",
            "amount": 1000.0,
            "nameOrig": "C_USER_3",
            "oldbalanceOrg": 200.0,
            "nameDest": "C_USER_4",
            "oldbalanceDest": 500.0,
        },
    ]

    processed = 0
    failed = 0
    alerts = 0
    new_alerts = 0

    for item in batch_events:
        try:
            if not isinstance(item, dict):
                raise ValueError("Expected dict event")
            res = score_transaction(item, clf, features, threshold)
            processed += 1
            if res["is_fraud_predicted"]:
                alerts += 1
                if record_alert(res, producer=None):
                    new_alerts += 1
        except Exception:
            failed += 1

    assert processed == 4
    assert failed == 1
    assert alerts == 2  # event 1 and its duplicate both predicted fraud
    assert new_alerts == 1  # but only 1 unique alert recorded (idempotent!)


def test_broker_probe_and_simulation_smoke_test():
    from src.kafka_smoke_test import is_broker_reachable, run_simulated_stream_test
    reachable = is_broker_reachable("127.0.0.1:9092", timeout_sec=0.5)
    assert isinstance(reachable, bool)

    sim_results = run_simulated_stream_test(num_events=6)
    assert sim_results["mode"] == "BOUNDED_SIMULATION"
    assert sim_results["processed"] >= 4
    assert sim_results["failed"] == 1

