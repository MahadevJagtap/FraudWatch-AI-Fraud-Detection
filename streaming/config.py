"""
Kafka & Event Streaming Configuration Settings
==============================================
"""

from __future__ import annotations

import os

BOOTSTRAP_SERVERS: str = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TRANSACTIONS_TOPIC: str = os.environ.get("KAFKA_TRANSACTIONS_TOPIC", "fraudwatch.transactions.raw")
ALERTS_TOPIC: str = os.environ.get("KAFKA_ALERTS_TOPIC", "fraudwatch.alerts.fraud")
CONSUMER_GROUP_ID: str = os.environ.get("KAFKA_GROUP_ID", "fraudwatch_scoring_group")
AUTO_OFFSET_RESET: str = os.environ.get("KAFKA_AUTO_OFFSET_RESET", "earliest")
