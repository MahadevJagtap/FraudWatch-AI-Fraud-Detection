"""
Automated test suite for the PaySim Fraud Scoring FastAPI service.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.config import settings
from api.service import FraudScoringService, ModelLoadError


@pytest.fixture(scope="module")
def client():
    """Create a test client that initializes the app and lifespan."""
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Test Root & Health Endpoints
# ---------------------------------------------------------------------------
def test_root_endpoint(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "documentation" in data
    assert data["documentation"]["swagger_ui"] == "/docs"
    assert "endpoints" in data


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["feature_count"] == 11
    assert "errorBalanceOrig" in data["expected_features"]
    assert "is_merchant" in data["expected_features"]
    assert data["training_auprc"] is not None


# ---------------------------------------------------------------------------
# Test Valid Predictions (Historical & Live Authorization)
# ---------------------------------------------------------------------------
def test_predict_legitimate_transaction(client: TestClient):
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
        "transaction_id": "TX-LEGIT-101",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "TX-LEGIT-101"
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["fraud_probability"] < 0.20
    assert data["risk_level"] == "LOW"
    assert data["recommended_action"] == "APPROVE"
    assert data["is_fraud"] is False
    assert data["input_mode"] == "historical"


def test_predict_fraudulent_transaction(client: TestClient):
    payload = {
        "step": 741,
        "type": "TRANSFER",
        "amount": 250000.0,
        "nameOrig": "C123456789",
        "oldbalanceOrg": 250000.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C987654321",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "mode": "historical",
        "transaction_id": "TX-FRAUD-101",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "TX-FRAUD-101"
    assert data["fraud_probability"] >= 0.90
    assert data["risk_level"] == "HIGH"
    assert data["recommended_action"] == "BLOCK"
    assert data["is_fraud"] is True


def test_predict_live_authorization_mode(client: TestClient):
    # In live authorization mode, new balances are omitted
    payload = {
        "step": 50,
        "type": "TRANSFER",
        "amount": 5000.0,
        "nameOrig": "C111111111",
        "oldbalanceOrg": 50000.0,
        "nameDest": "C222222222",
        "oldbalanceDest": 1000.0,
        "mode": "live_authorization",
        "transaction_id": "TX-LIVE-001",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "TX-LIVE-001"
    assert data["input_mode"] == "live_authorization"
    assert data["audit_note"] is not None
    assert "live_authorization" in data["audit_note"]


# ---------------------------------------------------------------------------
# Test Invalid Inputs & Error Handling
# ---------------------------------------------------------------------------
def test_predict_missing_balances_in_historical_mode(client: TestClient):
    # Missing new balances in historical mode should fail validation
    payload = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 5000.0,
        "nameOrig": "C111111111",
        "oldbalanceOrg": 50000.0,
        "nameDest": "C222222222",
        "oldbalanceDest": 1000.0,
        "mode": "historical",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "Validation Error"
    assert any("newbalanceOrig" in str(d) for d in data["details"])


def test_predict_negative_amount(client: TestClient):
    payload = {
        "step": 1,
        "type": "TRANSFER",
        "amount": -500.0,
        "nameOrig": "C111111111",
        "oldbalanceOrg": 50000.0,
        "newbalanceOrig": 49500.0,
        "nameDest": "C222222222",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 500.0,
        "mode": "historical",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_transaction_type(client: TestClient):
    payload = {
        "step": 1,
        "type": "CRYPTO_SWAP",
        "amount": 500.0,
        "nameOrig": "C111111111",
        "oldbalanceOrg": 50000.0,
        "newbalanceOrig": 49500.0,
        "nameDest": "C222222222",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 500.0,
        "mode": "historical",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test Configurable Threshold Overrides
# ---------------------------------------------------------------------------
def test_predict_threshold_overrides(client: TestClient):
    payload = {
        "step": 741,
        "type": "TRANSFER",
        "amount": 250000.0,
        "nameOrig": "C123456789",
        "oldbalanceOrg": 250000.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C987654321",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "mode": "historical",
    }
    # Pass custom query thresholds
    response = client.post("/predict?block_threshold=0.9999&review_threshold=0.80", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["block_threshold"] == 0.9999
    assert data["review_threshold"] == 0.80


# ---------------------------------------------------------------------------
# Test Batch Scoring
# ---------------------------------------------------------------------------
def test_predict_batch_success(client: TestClient):
    payload = {
        "transactions": [
            {
                "step": 1,
                "type": "PAYMENT",
                "amount": 100.0,
                "nameOrig": "C1000",
                "oldbalanceOrg": 1000.0,
                "newbalanceOrig": 900.0,
                "nameDest": "M2000",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "mode": "historical",
                "transaction_id": "BATCH-1",
            },
            {
                "step": 200,
                "type": "TRANSFER",
                "amount": 500000.0,
                "nameOrig": "C3000",
                "oldbalanceOrg": 500000.0,
                "newbalanceOrig": 0.0,
                "nameDest": "C4000",
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "mode": "historical",
                "transaction_id": "BATCH-2",
            },
        ],
        "block_threshold": 0.90,
        "review_threshold": 0.50,
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_scored"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["transaction_id"] == "BATCH-1"
    assert data["results"][1]["transaction_id"] == "BATCH-2"
    assert data["latency_ms"] >= 0.0


def test_predict_batch_empty_list(client: TestClient):
    payload = {"transactions": []}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test Model Loading Error Handling
# ---------------------------------------------------------------------------
def test_model_loading_missing_file():
    service = FraudScoringService(
        model_path="nonexistent_model.json",
        metadata_path="nonexistent_meta.json",
    )
    with pytest.raises(ModelLoadError) as exc_info:
        service.load()
    assert "Metadata file not found" in str(exc_info.value)


def test_platform_status_endpoint(client: TestClient):
    response = client.get("/platform/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OPERATIONAL"
    assert "technologies" in data
    techs = data["technologies"]
    assert "xgboost" in techs
    assert "feast" in techs
    assert "fastai" in techs
    assert "pytorch_geometric" in techs
    assert "mlflow" in techs
    assert "kafka_streaming" in techs
    assert "evidently_ai" in techs


def test_platform_alerts_endpoint(client: TestClient):
    response = client.get("/platform/alerts?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "alerts" in data
    assert "count" in data

