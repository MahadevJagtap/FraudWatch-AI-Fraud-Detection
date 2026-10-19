"""
Configuration module for the PaySim Fraud Scoring API.
"""

from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, Field

# Base directories
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PACKAGE_ROOT / "models" / "fraud_xgb_model.json"
DEFAULT_METADATA_PATH = PACKAGE_ROOT / "models" / "model_metadata.json"


class Settings(BaseModel):
    """Application settings with environment variable overrides."""
    api_title: str = "PaySim Real-Time Fraud Scoring Service"
    api_version: str = "1.0.0"
    api_description: str = (
        "Production-ready FastAPI scoring service for financial transaction fraud detection "
        "powered by an XGBoost model trained on the PaySim benchmark dataset."
    )
    model_path: str = os.getenv("FRAUD_MODEL_PATH", str(DEFAULT_MODEL_PATH))
    metadata_path: str = os.getenv("FRAUD_METADATA_PATH", str(DEFAULT_METADATA_PATH))
    preauth_model_path: str = os.getenv(
        "PREAUTH_MODEL_PATH", str(PACKAGE_ROOT / "models" / "leakage_free_xgb_model.json")
    )
    preauth_metadata_path: str = os.getenv(
        "PREAUTH_METADATA_PATH", str(PACKAGE_ROOT / "models" / "leakage_free_metadata.json")
    )

    # Feast Feature Store configuration
    enable_feast_online_store: bool = os.getenv("ENABLE_FEAST_ONLINE_STORE", "true").lower() == "true"
    feast_repo_path: str = os.getenv("FEAST_REPO_PATH", str(PACKAGE_ROOT / "feature_store"))

    # Decision thresholds (demo risk policy)
    # Scores >= block_threshold -> HIGH Risk, BLOCK
    # Scores >= review_threshold -> MEDIUM Risk, REVIEW
    # Scores < review_threshold  -> LOW Risk, APPROVE
    default_block_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    default_review_threshold: float = Field(default=0.50, ge=0.0, le=1.0)

    max_batch_size: int = Field(default=500, ge=1, le=5000)


settings = Settings()
