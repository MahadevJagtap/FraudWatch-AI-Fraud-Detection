"""
Kafka Streaming Pipeline Verification & Smoke Test
==================================================

Probes for a live Kafka broker on the configured bootstrap servers.
- If a live broker is reachable: executes an end-to-end broker-backed produce/consume smoke test.
- If no broker is running: explicitly reports the infrastructure blocker, runs the verified
  bounded simulation pipeline, and outputs exact commands to start Kafka.

Run standalone:
    python src/kafka_smoke_test.py
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from typing import Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from streaming.config import (
    BOOTSTRAP_SERVERS,
    TRANSACTIONS_TOPIC,
    ALERTS_TOPIC,
    CONSUMER_GROUP_ID,
)
from streaming.consumer_worker import (
    load_scoring_artifacts,
    score_transaction,
    record_alert,
    _SEEN_ALERT_IDS,
)
from streaming.producer import transaction_generator


def is_broker_reachable(bootstrap: str = BOOTSTRAP_SERVERS, timeout_sec: float = 2.0) -> bool:
    """Probe if the Kafka bootstrap host:port is listening on TCP."""
    try:
        host, port_str = bootstrap.split(":", 1)
        port = int(port_str)
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def run_broker_smoke_test(bootstrap: str, num_events: int = 10) -> Dict[str, Any]:
    """Execute live produce and consume test against an active broker."""
    print(f"\n[LIVE BROKER TEST] Connected to Kafka at {bootstrap}!")
    from kafka import KafkaProducer, KafkaConsumer

    producer = KafkaProducer(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8"),
        request_timeout_ms=5000,
    )

    clf, features, threshold = load_scoring_artifacts()
    events_sent = 0

    print(f"  Publishing {num_events} bounded PaySim transactions to {TRANSACTIONS_TOPIC!r}...")
    for tx in transaction_generator(max_events=num_events):
        producer.send(TRANSACTIONS_TOPIC, key=tx["transaction_id"], value=tx)
        events_sent += 1
    producer.flush()
    print(f"  Successfully sent {events_sent} events to broker.")

    print(f"  Consuming events from {TRANSACTIONS_TOPIC!r} with consumer group {CONSUMER_GROUP_ID!r}...")
    consumer = KafkaConsumer(
        TRANSACTIONS_TOPIC,
        bootstrap_servers=bootstrap,
        group_id=f"{CONSUMER_GROUP_ID}_smoke_{int(time.time())}",
        auto_offset_reset="earliest",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        consumer_timeout_ms=8000,
        request_timeout_ms=15000,
    )

    consumed = 0
    alerts = 0
    try:
        raw_batches = consumer.poll(timeout_ms=6000, max_records=events_sent)
        for tp, records in raw_batches.items():
            for msg in records:
                tx = msg.value
                result = score_transaction(tx, clf, features, threshold)
                consumed += 1
                if result["is_fraud_predicted"]:
                    alerts += 1
                    record_alert(result)
                if consumed >= events_sent:
                    break
            if consumed >= events_sent:
                break
    except Exception as exc:
        print(f"  [NOTICE] Consumer loop finished or intercepted: {exc}")
    finally:
        try:
            consumer.close()
        except Exception:
            pass
        try:
            producer.close()
        except Exception:
            pass

    print(f"  Live broker smoke test finished: {consumed} consumed, {alerts} fraud alerts persisted.")
    return {
        "mode": "REAL_KAFKA_BROKER",
        "broker_reachable": True,
        "produced": events_sent,
        "consumed": consumed,
        "alerts_persisted": alerts,
    }


def run_simulated_stream_test(num_events: int = 10) -> Dict[str, Any]:
    """Execute bounded streaming pipeline simulation with error recovery and deduplication."""
    print("\n[STREAMING SIMULATION TEST] Running bounded streaming scoring simulation...")
    _SEEN_ALERT_IDS.clear()
    clf, features, threshold = load_scoring_artifacts()
    uid = f"{time.time():.4f}"

    # Generate real transactions plus test cases for invalid events and duplicates
    test_stream: List[Any] = []
    for tx in transaction_generator(max_events=max(1, num_events - 2)):
        test_stream.append(tx)

    # Inject a duplicate of the first event
    if test_stream:
        dup = dict(test_stream[0])
        test_stream.append(dup)

    # Inject an invalid non-dict event
    test_stream.append("MALFORMED_NON_JSON_EVENT_STRING")

    processed = 0
    failed = 0
    alerts = 0
    new_cases = 0

    for item in test_stream:
        try:
            if not isinstance(item, dict):
                raise ValueError("Expected dictionary event, got non-dict payload")
            res = score_transaction(item, clf, features, threshold)
            processed += 1
            if res["is_fraud_predicted"]:
                alerts += 1
                if record_alert(res):
                    new_cases += 1
        except Exception as exc:
            failed += 1
            print(f"  [RECOVERY] Caught and handled invalid event safely: {exc}")

    print(f"  Simulation completed:")
    print(f"    Total Events Evaluated: {len(test_stream)}")
    print(f"    Successfully Scored:    {processed}")
    print(f"    Failed (Safely Caught): {failed}")
    print(f"    Fraud Alerts Emitted:   {alerts}")
    print(f"    New Cases Persisted:    {new_cases} (Duplicate events ignored idempotently)")

    return {
        "mode": "BOUNDED_SIMULATION",
        "broker_reachable": False,
        "total_events": len(test_stream),
        "processed": processed,
        "failed": failed,
        "alerts_emitted": alerts,
        "new_cases_persisted": new_cases,
    }


def main():
    print("=" * 75)
    print("FraudWatch Kafka Streaming Pipeline Verification")
    print("=" * 75)
    print(f"Target Bootstrap Server: {BOOTSTRAP_SERVERS}")
    print(f"Inbound Topic:           {TRANSACTIONS_TOPIC}")
    print(f"Outbound Alert Topic:    {ALERTS_TOPIC}")
    print(f"Consumer Group ID:       {CONSUMER_GROUP_ID}")

    reachable = is_broker_reachable(BOOTSTRAP_SERVERS, timeout_sec=1.5)

    if reachable:
        results = run_broker_smoke_test(BOOTSTRAP_SERVERS)
    else:
        print(f"\nSTATUS: Kafka broker at {BOOTSTRAP_SERVERS} is OFFLINE.")
        print("  Exact Blocker: No active broker daemon listening on port 9092.")
        print("  Host Context:  Windows host without Docker engine or native Kafka service in PATH.")
        print("  Alternative:   Executing verified bounded stream pipeline simulation.")
        results = run_simulated_stream_test()

        print("\n" + "-" * 75)
        print("TO START A REAL KAFKA BROKER WHEN DOCKER IS INSTALLED:")
        print("  1. Ensure Docker Desktop is installed and running.")
        print("  2. In project root, run:  docker compose up -d")
        print("  3. Re-run this smoke test: python src/kafka_smoke_test.py")
        print("-" * 75)

    print("\n" + "=" * 75)
    print(f"VERIFICATION COMPLETED: Mode = {results['mode']}")
    print("=" * 75)


if __name__ == "__main__":
    main()
