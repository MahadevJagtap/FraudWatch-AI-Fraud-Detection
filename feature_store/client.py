"""
Feast Online Feature Store Client Helper
========================================

Provides fast online feature lookup for real-time transaction scoring.
"""

from __future__ import annotations

import os
from typing import Dict, List, Any
from feast import FeatureStore

REPO_DIR = os.path.dirname(os.path.abspath(__file__))

_STORE: FeatureStore | None = None


def get_feature_store() -> FeatureStore:
    global _STORE
    if _STORE is None:
        _STORE = FeatureStore(repo_path=REPO_DIR)
    return _STORE


def get_online_account_features(account_ids: List[str]) -> Dict[str, Any]:
    """Retrieve pre-computed account features from Feast SQLite online store."""
    store = get_feature_store()
    entity_rows = [{"account_id": aid} for aid in account_ids]
    features = [
        "account_features:orig_old_balance",
        "account_features:dest_old_balance",
        "account_features:orig_drain_ratio",
        "account_features:is_merchant",
        "account_features:dest_is_customer",
    ]
    return store.get_online_features(
        features=features,
        entity_rows=entity_rows
    ).to_dict()
