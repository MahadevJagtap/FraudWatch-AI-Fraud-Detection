"""
Kafka Transaction Replay Producer
=================================

Streams pre-authorization transactions from PaySim dataset into the
Kafka raw transactions topic (`fraudwatch.transactions.raw`).

Run standalone:
    python streaming/producer.py --max-events 100 --delay-ms 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, Generator

import pandas as pd
from kafka import KafkaProducer

from streaming.config import BOOTSTRAP_SERVERS, TRANSACTIONS_TOPIC

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join("D:\\Fraud_detection", "PaySim_dataset.csv")


def create_producer(bootstrap_servers: str = BOOTSTRAP_SERVERS) -> KafkaProducer:
    """Initialize Kafka producer with JSON serializer."""
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8"),
        acks="all",
        retries=3,
        request_timeout_ms=5000,
        api_version_auto_timeout_ms=3000,
    )


def transaction_generator(csv_path: str = DEFAULT_CSV, max_events: int = 1000) -> Generator[Dict[str, Any], None, None]:
    """Yield transactions from CSV without post-settlement leakage columns."""
    cols = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "nameDest", "oldbalanceDest", "isFraud"]
    count = 0
    for chunk in pd.read_csv(csv_path, usecols=cols, chunksize=1000):
        for _, row in chunk.iterrows():
            count += 1
            yield {
                "transaction_id": f"tx_{count:08d}",
                "step": int(row["step"]),
                "type": str(row["type"]),
                "amount": float(row["amount"]),
                "nameOrig": str(row["nameOrig"]),
                "oldbalanceOrg": float(row["oldbalanceOrg"]),
                "nameDest": str(row["nameDest"]),
                "oldbalanceDest": float(row["oldbalanceDest"]),
                "ground_truth_isFraud": int(row["isFraud"]),
                "timestamp": time.time(),
            }
            if max_events and count >= max_events:
                return


def replay_transactions(
    csv_path: str = DEFAULT_CSV,
    max_events: int = 100,
    delay_ms: int = 50,
    bootstrap_servers: str = BOOTSTRAP_SERVERS
) -> int:
    """Publish transactions to Kafka."""
    print(f"Connecting to Kafka broker at {bootstrap_servers}...")
    producer = create_producer(bootstrap_servers)
    topic = TRANSACTIONS_TOPIC
    print(f"Replaying up to {max_events} transactions to topic {topic!r} (delay: {delay_ms}ms)...")

    sent = 0
    t0 = time.perf_counter()

    for tx in transaction_generator(csv_path, max_events=max_events):
        key = tx["transaction_id"]
        producer.send(topic, key=key, value=tx)
        sent += 1
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
        if sent % 50 == 0 or sent == max_events:
            print(f"  Sent {sent}/{max_events} events...")

    producer.flush()
    elapsed = time.perf_counter() - t0
    print(f"Finished sending {sent} events in {elapsed:.2f}s ({sent/max(0.01, elapsed):.1f} msg/sec).")
    return sent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay PaySim transactions to Kafka")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to PaySim CSV")
    parser.add_argument("--max-events", type=int, default=100, help="Max events to stream")
    parser.add_argument("--delay-ms", type=int, default=20, help="Delay between events in milliseconds")
    parser.add_argument("--bootstrap-servers", default=BOOTSTRAP_SERVERS, help="Kafka bootstrap servers")
    args = parser.parse_args()

    try:
        replay_transactions(
            csv_path=args.csv,
            max_events=args.max_events,
            delay_ms=args.delay_ms,
            bootstrap_servers=args.bootstrap_servers
        )
    except Exception as e:
        print(f"Producer error: {e}", file=sys.stderr)
        sys.exit(1)
