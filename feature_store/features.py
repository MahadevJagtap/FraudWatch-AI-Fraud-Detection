"""
Feast Feature Definitions for FraudWatch
=======================================

Defines entities, data sources, and feature views for accounts
(both origin and destination) in the PaySim dataset.
"""

from __future__ import annotations

import os
from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int32, String

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(REPO_DIR)
PARQUET_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "account_features_feast.parquet").replace("\\", "/")

# 1. Entity Definition
account_entity = Entity(
    name="account_id",
    value_type=String,
    join_keys=["account_id"],
    description="Financial account identifier (origin or destination)",
)

# 2. File Data Source
account_stats_source = FileSource(
    name="account_stats_source",
    path=PARQUET_PATH,
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# 3. Feature View Definition
account_features_view = FeatureView(
    name="account_features",
    entities=[account_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="orig_old_balance", dtype=Float32),
        Field(name="dest_old_balance", dtype=Float32),
        Field(name="orig_drain_ratio", dtype=Float32),
        Field(name="is_merchant", dtype=Int32),
        Field(name="dest_is_customer", dtype=Int32),
    ],
    online=True,
    source=account_stats_source,
)
