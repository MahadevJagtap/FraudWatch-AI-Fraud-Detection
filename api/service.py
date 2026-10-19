"""
Fraud Scoring Service - loads the trained models and handles feature transformation & inference.
Supports both legacy baseline XGBoost (11 features) and pre-authorization leakage-free XGBoost (16 features),
with native Feast SQLite online feature store integration and fallback handling.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from api.config import settings
from api.schemas import (
    ActionDecision,
    BatchPredictionResponse,
    InputMode,
    PredictionResult,
    RiskLevel,
    TransactionInput,
)

logger = logging.getLogger("fraud_api.service")

# All 5 PaySim transaction types in categorical order
PAYSIM_TYPES: Tuple[str, ...] = (
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
)


class ModelLoadError(RuntimeError):
    """Raised when the model or metadata fails to load or validate."""
    pass


class FraudScoringService:
    """Singleton service for loading models and executing real-time fraud scoring."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        preauth_model_path: Optional[str] = None,
        preauth_metadata_path: Optional[str] = None,
    ) -> None:
        self.model_path = model_path or settings.model_path
        self.metadata_path = metadata_path or settings.metadata_path
        self.preauth_model_path = preauth_model_path or settings.preauth_model_path
        self.preauth_metadata_path = preauth_metadata_path or settings.preauth_metadata_path

        # Baseline 11-feature model
        self.model: Optional[XGBClassifier] = None
        self.metadata: Dict[str, Any] = {}
        self.features: List[str] = []
        self.is_loaded: bool = False

        # Pre-authorization 16-feature champion model
        self.preauth_model: Optional[XGBClassifier] = None
        self.preauth_metadata: Dict[str, Any] = {}
        self.preauth_features: List[str] = []
        self.preauth_threshold: float = 0.8247
        self.preauth_is_loaded: bool = False

    def load(self) -> None:
        """Load and validate both the baseline model and pre-authorization champion model."""
        # 1. Load Baseline Model
        logger.info(f"Loading metadata from {self.metadata_path}...")
        if not os.path.isfile(self.metadata_path):
            raise ModelLoadError(f"Metadata file not found at: {self.metadata_path}")

        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        except Exception as e:
            raise ModelLoadError(f"Failed to parse metadata JSON: {e}") from e

        self.features = self.metadata.get("features", [])
        if not self.features:
            raise ModelLoadError("Metadata does not contain expected 'features' list.")

        logger.info(f"Loading model from {self.model_path}...")
        if not os.path.isfile(self.model_path):
            raise ModelLoadError(f"Model file not found at: {self.model_path}")

        if os.path.getsize(self.model_path) == 0:
            raise ModelLoadError(f"Model file at {self.model_path} is empty (0 bytes).")

        try:
            clf = XGBClassifier()
            clf.load_model(self.model_path)
            self.model = clf
        except Exception as e:
            raise ModelLoadError(f"Failed to load baseline XGBoost model: {e}") from e

        if hasattr(self.model, "n_features_in_") and self.model.n_features_in_ != len(self.features):
            raise ModelLoadError(
                f"Baseline model expects {self.model.n_features_in_} features, but metadata has {len(self.features)}."
            )

        self.is_loaded = True
        logger.info(
            f"Successfully loaded baseline XGBoost model ({len(self.features)} features). "
            f"Training AUPRC: {self.metadata.get('metrics', {}).get('auprc', 'N/A')}"
        )

        # 2. Load Pre-authorization Model (if present)
        if os.path.isfile(self.preauth_model_path) and os.path.isfile(self.preauth_metadata_path):
            try:
                with open(self.preauth_metadata_path, "r", encoding="utf-8") as f:
                    self.preauth_metadata = json.load(f)
                self.preauth_features = self.preauth_metadata.get("features", [])
                self.preauth_threshold = float(
                    self.preauth_metadata.get("decision_threshold", 0.8247)
                )

                clf_pre = XGBClassifier()
                clf_pre.load_model(self.preauth_model_path)
                self.preauth_model = clf_pre
                self.preauth_is_loaded = True
                logger.info(
                    f"Successfully loaded pre-authorization XGBoost model ({len(self.preauth_features)} features). "
                    f"Validation-tuned threshold: {self.preauth_threshold}"
                )
            except Exception as e:
                logger.warning(f"Could not load pre-authorization model: {e}")
                self.preauth_is_loaded = False
        else:
            logger.info("Pre-authorization model files not detected; skipping optional preauth model.")

    def _lookup_feast_account_features(
        self, account_ids: List[str]
    ) -> Tuple[Dict[str, Dict[str, Any]], str]:
        """
        Query Feast online store for account features.
        Returns:
            Tuple of (features_by_account_dict, status_string)
            status_string is one of: 'HIT', 'MISS', 'DISABLED', 'ERROR'
        """
        if not settings.enable_feast_online_store:
            return {}, "DISABLED"

        clean_ids = [str(aid) for aid in account_ids if aid]
        if not clean_ids:
            return {}, "MISS"

        try:
            from feature_store.client import get_online_account_features
            raw = get_online_account_features(clean_ids)
            retrieved: Dict[str, Dict[str, Any]] = {}
            found_any = False
            aids = raw.get("account_id", [])
            for idx, aid in enumerate(aids):
                acct_data = {}
                for feat_name, val_list in raw.items():
                    if feat_name != "account_id" and idx < len(val_list):
                        val = val_list[idx]
                        if val is not None:
                            acct_data[feat_name] = val
                            found_any = True
                if acct_data:
                    retrieved[aid] = acct_data

            status = "HIT" if found_any else "MISS"
            return retrieved, status
        except Exception as exc:
            logger.warning(f"Feast online store lookup error: {exc}", exc_info=True)
            return {}, "ERROR"

    def _transform_single(self, tx: TransactionInput) -> Tuple[Dict[str, float], Optional[str]]:
        """
        Derive the exact 11 features expected by the baseline model.
        Returns a dictionary of features and an optional audit note.
        """
        is_merchant = 1.0 if str(tx.nameDest).startswith("M") else 0.0
        hour_of_day = float(tx.step % 24)
        day_of_week = float((tx.step // 24) % 7)

        if tx.mode == InputMode.HISTORICAL:
            assert tx.newbalanceOrig is not None and tx.newbalanceDest is not None
            error_orig = float(tx.oldbalanceOrg - tx.amount - tx.newbalanceOrig)
            error_dest = float(tx.oldbalanceDest + tx.amount - tx.newbalanceDest)
            note = None
        else:
            # LIVE_AUTHORIZATION mode:
            new_orig_est = (
                tx.newbalanceOrig
                if tx.newbalanceOrig is not None
                else max(0.0, float(tx.oldbalanceOrg - tx.amount))
            )
            new_dest_est = (
                tx.newbalanceDest
                if tx.newbalanceDest is not None
                else (float(tx.oldbalanceDest + tx.amount) if not str(tx.nameDest).startswith("M") else 0.0)
            )
            error_orig = float(tx.oldbalanceOrg - tx.amount - new_orig_est)
            error_dest = float(tx.oldbalanceDest + tx.amount - new_dest_est)
            note = (
                "Mode: live_authorization. Post-settlement balances were estimated. "
                "Notice: Model was trained on PaySim historical post-settlement residuals; "
                "retraining with pre-authorization features only is recommended for production live-authorization."
            )

        feature_map: Dict[str, float] = {
            "amount": float(tx.amount),
            "errorBalanceOrig": error_orig,
            "errorBalanceDest": error_dest,
            "is_merchant": is_merchant,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "type_CASH_IN": 1.0 if tx.type == "CASH_IN" else 0.0,
            "type_CASH_OUT": 1.0 if tx.type == "CASH_OUT" else 0.0,
            "type_DEBIT": 1.0 if tx.type == "DEBIT" else 0.0,
            "type_PAYMENT": 1.0 if tx.type == "PAYMENT" else 0.0,
            "type_TRANSFER": 1.0 if tx.type == "TRANSFER" else 0.0,
        }
        return feature_map, note

    def score_transaction(
        self,
        tx: TransactionInput,
        block_threshold: Optional[float] = None,
        review_threshold: Optional[float] = None,
    ) -> PredictionResult:
        """Score a single transaction using the baseline XGBoost model, with optional Feast online enrichment."""
        if not self.is_loaded or self.model is None:
            raise ModelLoadError("Model is not loaded. Call load() before scoring.")

        b_thresh = block_threshold if block_threshold is not None else settings.default_block_threshold
        r_thresh = review_threshold if review_threshold is not None else settings.default_review_threshold

        retrieved_feast: Dict[str, Any] = {}
        feast_status = "DISABLED"
        source = "payload"
        tx_to_score = tx

        if tx.use_feature_store:
            feast_map, feast_status = self._lookup_feast_account_features([tx.nameOrig, tx.nameDest])
            if feast_status == "HIT":
                source = "feast_online_store"
                orig_feats = feast_map.get(tx.nameOrig, {})
                dest_feats = feast_map.get(tx.nameDest, {})
                retrieved_feast = {"orig": orig_feats, "dest": dest_feats}
                new_old_orig = float(orig_feats.get("orig_old_balance", tx.oldbalanceOrg))
                new_old_dest = float(dest_feats.get("dest_old_balance", tx.oldbalanceDest))
                tx_to_score = tx.model_copy(update={"oldbalanceOrg": new_old_orig, "oldbalanceDest": new_old_dest})
            elif feast_status == "MISS":
                source = "payload_fallback"
            elif feast_status == "ERROR":
                source = "payload_fallback"

        feature_map, note = self._transform_single(tx_to_score)
        if tx.use_feature_store:
            if feast_status == "HIT":
                enrich_note = "Enriched with pre-computed features from Feast online store (SQLite)."
                note = f"{note} | {enrich_note}" if note else enrich_note
            elif feast_status == "MISS":
                enrich_note = f"Account {tx.nameOrig} not found in Feast online store; fell back to client-supplied payload balance."
                note = f"{note} | {enrich_note}" if note else enrich_note
            elif feast_status == "ERROR":
                enrich_note = "Feast online store query failed; fell back to client-supplied payload balance."
                note = f"{note} | {enrich_note}" if note else enrich_note

        feature_row = np.array([[feature_map[col] for col in self.features]], dtype=np.float32)
        proba = float(self.model.predict_proba(feature_row)[0, 1])

        if proba >= b_thresh:
            risk = RiskLevel.HIGH
            action = ActionDecision.BLOCK
            is_fraud = True
        elif proba >= r_thresh:
            risk = RiskLevel.MEDIUM
            action = ActionDecision.REVIEW
            is_fraud = False
        else:
            risk = RiskLevel.LOW
            action = ActionDecision.APPROVE
            is_fraud = False

        return PredictionResult(
            transaction_id=tx.transaction_id,
            fraud_probability=round(proba, 6),
            is_fraud=is_fraud,
            risk_level=risk,
            recommended_action=action,
            input_mode=tx.mode,
            block_threshold=b_thresh,
            review_threshold=r_thresh,
            audit_note=note,
            features_source=source,
            feast_lookup_status=feast_status,
            retrieved_features=retrieved_feast if retrieved_feast else None,
        )

    def score_batch(
        self,
        transactions: List[TransactionInput],
        block_threshold: Optional[float] = None,
        review_threshold: Optional[float] = None,
    ) -> BatchPredictionResponse:
        """Batch scoring with vector inference for maximum throughput."""
        if not self.is_loaded or self.model is None:
            raise ModelLoadError("Model is not loaded. Call load() before scoring.")

        if len(transactions) > settings.max_batch_size:
            raise ValueError(
                f"Batch size {len(transactions)} exceeds maximum allowed limit of {settings.max_batch_size}."
            )

        b_thresh = block_threshold if block_threshold is not None else settings.default_block_threshold
        r_thresh = review_threshold if review_threshold is not None else settings.default_review_threshold

        t0 = time.perf_counter()

        rows = []
        notes = []
        for tx in transactions:
            feat_map, note = self._transform_single(tx)
            rows.append([feat_map[col] for col in self.features])
            notes.append(note)

        X_matrix = np.array(rows, dtype=np.float32)
        probas = self.model.predict_proba(X_matrix)[:, 1]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        results: List[PredictionResult] = []
        fraud_count = 0
        review_count = 0
        approved_count = 0

        for i, tx in enumerate(transactions):
            proba = float(probas[i])
            if proba >= b_thresh:
                risk = RiskLevel.HIGH
                action = ActionDecision.BLOCK
                is_fraud = True
                fraud_count += 1
            elif proba >= r_thresh:
                risk = RiskLevel.MEDIUM
                action = ActionDecision.REVIEW
                is_fraud = False
                review_count += 1
            else:
                risk = RiskLevel.LOW
                action = ActionDecision.APPROVE
                is_fraud = False
                approved_count += 1

            results.append(
                PredictionResult(
                    transaction_id=tx.transaction_id,
                    fraud_probability=round(proba, 6),
                    is_fraud=is_fraud,
                    risk_level=risk,
                    recommended_action=action,
                    input_mode=tx.mode,
                    block_threshold=b_thresh,
                    review_threshold=r_thresh,
                    audit_note=notes[i],
                    features_source="payload",
                    feast_lookup_status="DISABLED",
                )
            )

        return BatchPredictionResponse(
            total_scored=len(transactions),
            fraud_count=fraud_count,
            review_count=review_count,
            approved_count=approved_count,
            block_threshold=b_thresh,
            review_threshold=r_thresh,
            latency_ms=round(latency_ms, 2),
            results=results,
        )

    def _transform_preauth_single(
        self,
        tx: TransactionInput,
        feast_orig: Optional[Dict[str, Any]] = None,
        feast_dest: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Derive the 16 pre-authorization features strictly free of post-settlement leakage."""
        orig_old = (
            float(feast_orig["orig_old_balance"])
            if feast_orig and "orig_old_balance" in feast_orig and feast_orig["orig_old_balance"] is not None
            else float(tx.oldbalanceOrg)
        )
        dest_old = (
            float(feast_dest["dest_old_balance"])
            if feast_dest and "dest_old_balance" in feast_dest and feast_dest["dest_old_balance"] is not None
            else float(tx.oldbalanceDest)
        )
        is_m = 1.0 if str(tx.nameDest).startswith("M") else 0.0
        dest_is_cust = 1.0 - is_m
        drain = float(np.clip(tx.amount / (orig_old + 1.0), 0.0, 10.0))

        return {
            "hour_of_day": float(tx.step % 24),
            "day_of_week": float((tx.step // 24) % 7),
            "type_CASH_IN": 1.0 if tx.type == "CASH_IN" else 0.0,
            "type_CASH_OUT": 1.0 if tx.type == "CASH_OUT" else 0.0,
            "type_DEBIT": 1.0 if tx.type == "DEBIT" else 0.0,
            "type_PAYMENT": 1.0 if tx.type == "PAYMENT" else 0.0,
            "type_TRANSFER": 1.0 if tx.type == "TRANSFER" else 0.0,
            "amount": float(tx.amount),
            "log_amount": float(np.log1p(tx.amount)),
            "is_merchant": is_m,
            "dest_is_customer": dest_is_cust,
            "orig_old_balance": orig_old,
            "dest_old_balance": dest_old,
            "orig_is_zero": 1.0 if orig_old == 0.0 else 0.0,
            "dest_is_zero": 1.0 if dest_old == 0.0 else 0.0,
            "orig_drain_ratio": drain,
        }

    def score_preauth_transaction(
        self,
        tx: TransactionInput,
        block_threshold: Optional[float] = None,
        review_threshold: Optional[float] = None,
    ) -> PredictionResult:
        """
        Score a transaction using the pre-authorization leakage-free XGBoost model (16 features),
        enriched with Feast online feature store.
        """
        if not self.preauth_is_loaded or self.preauth_model is None:
            raise ModelLoadError("Pre-authorization model is not loaded.")

        b_thresh = block_threshold if block_threshold is not None else self.preauth_threshold
        r_thresh = review_threshold if review_threshold is not None else 0.50

        feast_map, feast_status = self._lookup_feast_account_features([tx.nameOrig, tx.nameDest])
        retrieved_feast: Dict[str, Any] = {}
        source = "payload"
        note = "Pre-authorization model inference (16 features, strictly leakage-free)."

        if feast_status == "HIT":
            source = "feast_online_store"
            orig_feats = feast_map.get(tx.nameOrig, {})
            dest_feats = feast_map.get(tx.nameDest, {})
            retrieved_feast = {"orig": orig_feats, "dest": dest_feats}
            note += " Online account balances retrieved from Feast online store."
            feature_map = self._transform_preauth_single(tx, orig_feats, dest_feats)
        elif feast_status == "MISS":
            source = "payload_fallback"
            note += f" Account {tx.nameOrig} not in Feast store; used payload initial balances."
            feature_map = self._transform_preauth_single(tx)
        else:
            source = "payload_fallback"
            note += " Feast store lookup unavailable; used payload initial balances."
            feature_map = self._transform_preauth_single(tx)

        feature_row = np.array([[feature_map[col] for col in self.preauth_features]], dtype=np.float32)
        proba = float(self.preauth_model.predict_proba(feature_row)[0, 1])

        if proba >= b_thresh:
            risk = RiskLevel.HIGH
            action = ActionDecision.BLOCK
            is_fraud = True
        elif proba >= r_thresh:
            risk = RiskLevel.MEDIUM
            action = ActionDecision.REVIEW
            is_fraud = False
        else:
            risk = RiskLevel.LOW
            action = ActionDecision.APPROVE
            is_fraud = False

        return PredictionResult(
            transaction_id=tx.transaction_id,
            fraud_probability=round(proba, 6),
            is_fraud=is_fraud,
            risk_level=risk,
            recommended_action=action,
            input_mode=tx.mode,
            block_threshold=b_thresh,
            review_threshold=r_thresh,
            audit_note=note,
            features_source=source,
            feast_lookup_status=feast_status,
            retrieved_features=retrieved_feast if retrieved_feast else None,
        )


# Global service instance
scoring_service = FraudScoringService()
