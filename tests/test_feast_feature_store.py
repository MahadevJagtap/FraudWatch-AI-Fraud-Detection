"""
Automated tests for Feast Feature Store integration and API inference enrichment.
"""

from __future__ import annotations

import os
import sys
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from api.app import app
from api.service import scoring_service

FEATURE_STORE_DIR = os.path.join(PROJECT_ROOT, "feature_store")
PARQUET_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "account_features_feast.parquet")
ONLINE_DB_PATH = os.path.join(FEATURE_STORE_DIR, "data", "online_store.db")

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def ensure_models_loaded():
    if not scoring_service.is_loaded:
        scoring_service.load()


def test_feast_artifacts_exist():
    assert os.path.isdir(FEATURE_STORE_DIR), "feature_store directory missing"
    assert os.path.isfile(os.path.join(FEATURE_STORE_DIR, "feature_store.yaml")), "feature_store.yaml missing"
    assert os.path.isfile(PARQUET_PATH), f"Feast parquet source missing: {PARQUET_PATH}"
    assert os.path.isfile(ONLINE_DB_PATH), f"Feast online store missing: {ONLINE_DB_PATH}"
    assert os.path.getsize(ONLINE_DB_PATH) > 0, "Feast online store is empty"


def test_feast_online_feature_retrieval():
    from feature_store.client import get_online_account_features

    account_ids = ["C1231006815", "M1979787155"]
    features = get_online_account_features(account_ids)

    assert "account_id" in features
    assert features["account_id"] == account_ids
    assert "orig_old_balance" in features
    assert "dest_old_balance" in features
    assert "orig_drain_ratio" in features
    assert "is_merchant" in features
    assert "dest_is_customer" in features

    assert features["orig_old_balance"][0] == 170136.0
    assert features["is_merchant"][1] == 1


def test_feast_account_profile_endpoint():
    # Known account HIT
    resp_hit = client.get("/platform/feature-store/account/C1231006815")
    assert resp_hit.status_code == 200
    data_hit = resp_hit.json()
    assert data_hit["found"] is True
    assert data_hit["status"] == "HIT"
    assert data_hit["features"]["orig_old_balance"] == 170136.0

    # Unknown account MISS
    resp_miss = client.get("/platform/feature-store/account/C9999999999")
    assert resp_miss.status_code == 200
    data_miss = resp_miss.json()
    assert data_miss["found"] is False
    assert data_miss["status"] == "MISS"
    assert data_miss["features"] is None


def test_predict_with_feast_enrichment_hit():
    payload = {
        "step": 1,
        "type": "PAYMENT",
        "amount": 9839.64,
        "nameOrig": "C1231006815",
        "oldbalanceOrg": 170136.0,
        "newbalanceOrig": 160296.36,
        "nameDest": "M1979787155",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "mode": "historical",
        "transaction_id": "TX-FEAST-HIT",
        "use_feature_store": True,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    result = resp.json()
    assert result["features_source"] == "feast_online_store"
    assert result["feast_lookup_status"] == "HIT"
    assert result["retrieved_features"] is not None
    assert "orig" in result["retrieved_features"]
    assert result["retrieved_features"]["orig"]["orig_old_balance"] == 170136.0


def test_predict_with_feast_enrichment_unknown_account_fallback():
    payload = {
        "step": 10,
        "type": "CASH_OUT",
        "amount": 5000.0,
        "nameOrig": "C9999999999",  # Non-existent in Feast
        "oldbalanceOrg": 5000.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C8888888888",
        "oldbalanceDest": 1000.0,
        "newbalanceDest": 6000.0,
        "mode": "historical",
        "transaction_id": "TX-FEAST-MISS",
        "use_feature_store": True,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    result = resp.json()
    assert result["features_source"] == "payload_fallback"
    assert result["feast_lookup_status"] == "MISS"
    assert "fell back to client-supplied payload" in result["audit_note"]


def test_predict_preauth_with_feast():
    payload = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 50000.0,
        "nameOrig": "C1231006815",
        "oldbalanceOrg": 170136.0,
        "nameDest": "C987654321",
        "oldbalanceDest": 0.0,
        "mode": "live_authorization",
        "transaction_id": "TX-PREAUTH-1",
        "use_feature_store": True,
    }
    resp = client.post("/predict/preauth", json=payload)
    assert resp.status_code == 200
    result = resp.json()
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert result["features_source"] == "feast_online_store"
    assert result["feast_lookup_status"] == "HIT"


def test_feast_outage_fallback_handling(monkeypatch):
    # Simulate an outage or exception in Feast client
    def mock_broken_feast(*args, **kwargs):
        raise ConnectionError("Simulated Feast Online Store Connection Outage")

    monkeypatch.setattr("feature_store.client.get_online_account_features", mock_broken_feast)

    payload = {
        "step": 5,
        "type": "PAYMENT",
        "amount": 100.0,
        "nameOrig": "C1231006815",
        "oldbalanceOrg": 500.0,
        "newbalanceOrig": 400.0,
        "nameDest": "M123456",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "mode": "historical",
        "transaction_id": "TX-FEAST-ERROR",
        "use_feature_store": True,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    result = resp.json()
    assert result["features_source"] == "payload_fallback"
    assert result["feast_lookup_status"] == "ERROR"
    assert "Feast online store query failed" in result["audit_note"]
