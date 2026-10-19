"""
Pydantic schemas and enums for the PaySim Fraud Scoring API.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class TransactionType(str, Enum):
    """Allowed transaction types from the PaySim dataset taxonomy."""
    CASH_IN = "CASH_IN"
    CASH_OUT = "CASH_OUT"
    DEBIT = "DEBIT"
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"


class InputMode(str, Enum):
    """
    Feature ingestion mode:
    - historical: Uses raw PaySim balance records including post-settlement new balances.
    - live_authorization: Simulates an in-flight transaction where new balances are not yet settled.
    """
    HISTORICAL = "historical"
    LIVE_AUTHORIZATION = "live_authorization"


class RiskLevel(str, Enum):
    """Categorized transaction risk tier."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ActionDecision(str, Enum):
    """Recommended automated policy decision (demo policy)."""
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"


class TransactionInput(BaseModel):
    """Input payload representing a financial transaction."""
    step: int = Field(default=1, ge=1, description="Hour of simulation/transaction step (1..744)")
    type: TransactionType = Field(..., description="Transaction category (e.g. TRANSFER, CASH_OUT)")
    amount: float = Field(..., gt=0.0, description="Amount of the transaction (must be positive)")
    nameOrig: str = Field(..., min_length=1, max_length=60, description="Origin account ID (e.g. C123456789)")
    oldbalanceOrg: float = Field(..., ge=0.0, description="Initial balance in origin account prior to transaction")
    newbalanceOrig: Optional[float] = Field(
        default=None, ge=0.0, description="New balance in origin account (required in historical mode)"
    )
    nameDest: str = Field(..., min_length=1, max_length=60, description="Recipient account ID (e.g. M123456789 or C987654321)")
    oldbalanceDest: float = Field(default=0.0, ge=0.0, description="Initial balance in recipient account prior to transaction")
    newbalanceDest: Optional[float] = Field(
        default=None, ge=0.0, description="New balance in recipient account (required in historical mode)"
    )
    mode: InputMode = Field(
        default=InputMode.HISTORICAL,
        description="Mode: 'historical' uses post-settlement balances; 'live_authorization' estimates them pre-authorization."
    )
    transaction_id: Optional[str] = Field(
        default=None, description="Optional client transaction tracking identifier"
    )
    use_feature_store: bool = Field(
        default=False,
        description="If True, query Feast online store using nameOrig/nameDest to enrich account features"
    )

    @model_validator(mode="after")
    def validate_mode_requirements(self) -> TransactionInput:
        if self.mode == InputMode.HISTORICAL:
            if self.newbalanceOrig is None or self.newbalanceDest is None:
                raise ValueError(
                    "In 'historical' mode, both 'newbalanceOrig' and 'newbalanceDest' must be provided. "
                    "If you are submitting an in-flight authorization request without settled balances, "
                    "set 'mode' to 'live_authorization'."
                )
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
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
                "transaction_id": "TX-987654",
                "use_feature_store": False
            }
        }
    }


class PredictionResult(BaseModel):
    """Output prediction for a scored transaction."""
    transaction_id: Optional[str] = None
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Model-predicted probability of fraud")
    is_fraud: bool = Field(..., description="True if fraud_probability >= block_threshold")
    risk_level: RiskLevel = Field(..., description="Risk tier: LOW, MEDIUM, or HIGH")
    recommended_action: ActionDecision = Field(..., description="Action: APPROVE, REVIEW, or BLOCK")
    input_mode: InputMode = Field(..., description="Input mode used for feature engineering")
    block_threshold: float = Field(..., description="Decision threshold used for BLOCK")
    review_threshold: float = Field(..., description="Decision threshold used for REVIEW")
    audit_note: Optional[str] = Field(default=None, description="Audit/disclaimer note regarding feature derivation")
    features_source: Optional[str] = Field(
        default="payload", description="Origin of features: 'payload', 'feast_online_store', or 'feast_fallback'"
    )
    feast_lookup_status: Optional[str] = Field(
        default=None, description="Feast online retrieval status: 'HIT', 'MISS', 'ERROR', or 'DISABLED'"
    )
    retrieved_features: Optional[Dict[str, Any]] = Field(
        default=None, description="Features retrieved from Feast online store if queried"
    )


class BatchTransactionRequest(BaseModel):
    """Batch scoring request payload."""
    transactions: List[TransactionInput] = Field(
        ..., min_length=1, description="List of transactions to score (max allowed defined by server config)"
    )
    block_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Optional override for block threshold"
    )
    review_threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Optional override for review threshold"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "transactions": [
                    {
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
                        "transaction_id": "TX-LEGIT-1"
                    },
                    {
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
                        "transaction_id": "TX-FRAUD-1"
                    }
                ]
            }
        }
    }


class BatchPredictionResponse(BaseModel):
    """Batch scoring response summary."""
    total_scored: int
    fraud_count: int
    review_count: int
    approved_count: int
    block_threshold: float
    review_threshold: float
    latency_ms: float
    results: List[PredictionResult]


class HealthResponse(BaseModel):
    """Health check status response."""
    status: str
    model_loaded: bool
    model_path: str
    model_type: str
    expected_features: List[str]
    feature_count: int
    training_auprc: Optional[float] = None
    training_recall: Optional[float] = None
    timestamp: str
