"""
Kafka Real-Time Scoring Worker & Alert Producer
===============================================

Consumes transactions from Kafka raw topic (`fraudwatch.transactions.raw`),
retrieves pre-authorization features, scores transactions via XGBoost,
and emits high-priority alerts to `fraudwatch.alerts.fraud`.

Run standalone:
    python streaming/consumer_worker.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from kafka import KafkaConsumer, KafkaProducer

from streaming.config import (
    BOOTSTRAP_SERVERS,
    TRANSACTIONS_TOPIC,
    ALERTS_TOPIC,
    CONSUMER_GROUP_ID,
    AUTO_OFFSET_RESET
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "leakage_free_xgb_model.json")
METADATA_PATH = os.path.join(PROJECT_ROOT, "models", "leakage_free_metadata.json")
ALERTS_LOG = os.path.join(PROJECT_ROOT, "reports", "kafka_fraud_alerts.jsonl")


def load_scoring_artifacts() -> Tuple[XGBClassifier, list[str], float]:
    """Load pre-authorization model, feature names, and decision threshold."""
    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(METADATA_PATH):
        raise FileNotFoundError("Pre-authorization model or metadata missing. Run 05_train_leakage_free_model.py first.")

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    clf = XGBClassifier()
    clf.load_model(MODEL_PATH)
    features = meta["features"]
    threshold = float(meta.get("decision_threshold", 0.70))
    return clf, features, threshold


def score_transaction(
    tx: Dict[str, Any],
    clf: XGBClassifier,
    feature_names: list[str],
    threshold: float = 0.70
) -> Dict[str, Any]:
    """Score an individual transaction without target leakage."""
    step = int(tx.get("step", 1))
    tx_type = str(tx.get("type", "PAYMENT")).upper()
    amount = float(tx.get("amount", 0.0))
    orig_bal = float(tx.get("oldbalanceOrg", 0.0))
    dest_bal = float(tx.get("oldbalanceDest", 0.0))
    dest_id = str(tx.get("nameDest", ""))

    is_merchant = 1 if dest_id.startswith("M") else 0
    dest_is_customer = 1 - is_merchant
    orig_drain_ratio = float(np.clip(amount / (orig_bal + 1.0), 0.0, 10.0))

    feature_map = {
        "hour_of_day": step % 24,
        "day_of_week": (step // 24) % 7,
        "type_CASH_IN": 1 if tx_type == "CASH_IN" else 0,
        "type_CASH_OUT": 1 if tx_type == "CASH_OUT" else 0,
        "type_DEBIT": 1 if tx_type == "DEBIT" else 0,
        "type_PAYMENT": 1 if tx_type == "PAYMENT" else 0,
        "type_TRANSFER": 1 if tx_type == "TRANSFER" else 0,
        "amount": amount,
        "log_amount": float(np.log1p(max(0.0, amount))),
        "is_merchant": is_merchant,
        "dest_is_customer": dest_is_customer,
        "orig_old_balance": orig_bal,
        "dest_old_balance": dest_bal,
        "orig_is_zero": 1 if orig_bal == 0.0 else 0,
        "dest_is_zero": 1 if dest_bal == 0.0 else 0,
        "orig_drain_ratio": orig_drain_ratio,
    }

    row_vec = np.array([[feature_map[f] for f in feature_names]], dtype=np.float32)
    fraud_prob = float(clf.predict_proba(row_vec)[0, 1])
    is_fraud = bool(fraud_prob >= threshold)

    risk_level = "CRITICAL" if fraud_prob >= 0.90 else ("HIGH" if fraud_prob >= threshold else "LOW")

    return {
        "transaction_id": tx.get("transaction_id", f"tx_{int(time.time()*1000)}"),
        "timestamp": time.time(),
        "fraud_probability": round(fraud_prob, 4),
        "decision_threshold": threshold,
        "is_fraud_predicted": is_fraud,
        "risk_level": risk_level,
        "amount": amount,
        "type": tx_type,
        "orig_drain_ratio": round(orig_drain_ratio, 4),
        "nameOrig": tx.get("nameOrig", ""),
        "nameDest": dest_id,
    }


_SEEN_ALERT_IDS: set = set()


def record_alert(alert: Dict[str, Any], producer: KafkaProducer | None = None) -> bool:
    """
    Save alert to disk log, cases.db database, and optionally emit to Kafka alert topic.
    Returns True if newly recorded, False if duplicate.
    """
    tx_id = alert.get("transaction_id", "")
    if tx_id in _SEEN_ALERT_IDS:
        return False

    os.makedirs(os.path.dirname(ALERTS_LOG), exist_ok=True)
    # Check existing file if cold start
    if not _SEEN_ALERT_IDS and os.path.isfile(ALERTS_LOG):
        try:
            with open(ALERTS_LOG, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            _SEEN_ALERT_IDS.add(json.loads(line).get("transaction_id"))
                        except Exception:
                            pass
        except Exception:
            pass

    if tx_id in _SEEN_ALERT_IDS:
        return False

    _SEEN_ALERT_IDS.add(tx_id)
    with open(ALERTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(alert) + "\n")

    # Persist to SQLite case management store (idempotent via PRIMARY KEY)
    try:
        from api.case_store import insert_or_update_alert
        insert_or_update_alert(alert)
    except Exception as e:
        print(f"Notice: Could not persist to cases.db: {e}")

    if producer:
        try:
            producer.send(ALERTS_TOPIC, key=alert["transaction_id"], value=alert)
        except Exception as e:
            print(f"Warning: could not forward alert to Kafka: {e}")

    return True


def run_scoring_worker(
    bootstrap_servers: str = BOOTSTRAP_SERVERS,
    max_messages: int | None = None
) -> Dict[str, int]:
    print(f"Loading pre-authorization scoring model from {MODEL_PATH}...")
    clf, features, threshold = load_scoring_artifacts()
    print(f"Model loaded with {len(features)} features. Decision threshold: {threshold:.2f}")

    print(f"Connecting to Kafka at {bootstrap_servers}...")
    consumer = KafkaConsumer(
        TRANSACTIONS_TOPIC,
        bootstrap_servers=bootstrap_servers,
        group_id=CONSUMER_GROUP_ID,
        auto_offset_reset=AUTO_OFFSET_RESET,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        consumer_timeout_ms=10000,
        request_timeout_ms=15000,
        api_version_auto_timeout_ms=3000,
    )

    alert_producer = None
    try:
        alert_producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8"),
            request_timeout_ms=5000,
            api_version_auto_timeout_ms=3000,
        )
    except Exception as e:
        print(f"Notice: Alert producer could not connect to Kafka: {e}")

    print(f"Scoring worker listening on topic {TRANSACTIONS_TOPIC!r}...")
    processed = 0
    failed = 0
    alerts_emitted = 0

    try:
        for message in consumer:
            try:
                tx = message.value
                if not isinstance(tx, dict):
                    raise ValueError(f"Expected dict message, got {type(tx)}")
                result = score_transaction(tx, clf, features, threshold)
                processed += 1

                if result["is_fraud_predicted"]:
                    alerts_emitted += 1
                    is_new = record_alert(result, producer=alert_producer)
                    if is_new:
                        print(f"[ALERT] Fraud Detected! ID: {result['transaction_id']}, Score: {result['fraud_probability']:.4f}, Amount: ${result['amount']:,.2f}")
                    else:
                        print(f"[INFO] Duplicate event ignored for alert persistence: {result['transaction_id']}")
            except Exception as exc:
                failed += 1
                print(f"[WARNING] Skipping invalid transaction event: {exc}")

            if max_messages and (processed + failed) >= max_messages:
                break
    finally:
        consumer.close()
        if alert_producer:
            alert_producer.close()

    print(f"\nWorker stopped. Processed: {processed}, Failed: {failed}, Alerts emitted: {alerts_emitted}")
    return {"processed": processed, "failed": failed, "alerts_emitted": alerts_emitted}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Kafka fraud scoring worker")
    parser.add_argument("--bootstrap-servers", default=BOOTSTRAP_SERVERS, help="Kafka bootstrap servers")
    parser.add_argument("--max-messages", type=int, default=None, help="Max messages to process before exiting")
    args = parser.parse_args()

    try:
        run_scoring_worker(bootstrap_servers=args.bootstrap_servers, max_messages=args.max_messages)
    except Exception as e:
        print(f"Consumer worker error: {e}", file=sys.stderr)
        sys.exit(1)

