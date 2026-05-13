"""
ModelServe — Feast Feature Client

Provides feature lookups from the Feast online store (Redis).
Uses the Feast SDK — does NOT query Redis directly.

Key design:
  - Initializes FeatureStore on startup
  - get_features(entity_id) calls store.get_online_features()
  - Tracks hit/miss counts for Prometheus metrics
  - Returns features as a dict suitable for DataFrame conversion
"""

import os
import logging
import pandas as pd
from typing import Optional

from feast import FeatureStore

from app.metrics import (
    feast_online_store_hits_total,
    feast_online_store_misses_total,
)

logger = logging.getLogger(__name__)

# Feature list must match the FeatureView schema in feast_repo/feature_definitions.py
FEATURE_SERVICE_FEATURES = [
    "fraud_features:merchant",
    "fraud_features:category",
    "fraud_features:gender",
    "fraud_features:state",
    "fraud_features:amt",
    "fraud_features:hour",
    "fraud_features:day_of_week",
    "fraud_features:month",
    "fraud_features:is_weekend",
    "fraud_features:time_of_day",
    "fraud_features:amt_log",
    "fraud_features:amt_squared",
    "fraud_features:amt_x_category",
    "fraud_features:amt_x_merchant",
    "fraud_features:merchant_avg_amt",
    "fraud_features:merchant_std_amt",
    "fraud_features:category_avg_amt",
    "fraud_features:category_std_amt",
]

# Feature names without the view prefix (what the model expects)
FEATURE_NAMES = [f.split(":")[1] for f in FEATURE_SERVICE_FEATURES]


class FeatureClient:
    """Wraps Feast FeatureStore for online feature retrieval."""

    def __init__(self):
        self.store: Optional[FeatureStore] = None
        self.repo_path = os.getenv("FEAST_REPO_PATH", "feast_repo")
        self.is_connected = False

    def connect(self) -> bool:
        """Initialize the Feast FeatureStore connection."""
        try:
            logger.info(f"Connecting to Feast repo at: {self.repo_path}")
            self.store = FeatureStore(repo_path=self.repo_path)
            self.is_connected = True
            logger.info("Feast FeatureStore connected successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Feast: {e}", exc_info=True)
            self.is_connected = False
            return False

    def get_features(self, entity_id: int) -> Optional[dict]:
        """
        Fetch features for a given entity_id (cc_num) from the Feast online store.

        Args:
            entity_id: Credit card number to look up

        Returns:
            dict of feature_name -> value, or None if lookup fails
        """
        if not self.is_connected or self.store is None:
            raise RuntimeError("Feast not connected. Call connect() first.")

        try:
            entity_rows = [{"cc_num": entity_id}]

            online_features = self.store.get_online_features(
                features=FEATURE_SERVICE_FEATURES,
                entity_rows=entity_rows,
            )

            feature_dict = online_features.to_dict()

            # Check if features were actually found (not all None)
            feature_values = {}
            has_values = False

            for name in FEATURE_NAMES:
                values = feature_dict.get(name, [None])
                val = values[0] if values else None
                feature_values[name] = val
                if val is not None:
                    has_values = True

            if has_values:
                feast_online_store_hits_total.inc()
                logger.debug(f"Feast hit for entity_id={entity_id}")
                return feature_values
            else:
                feast_online_store_misses_total.inc()
                logger.warning(f"Feast miss for entity_id={entity_id}: all features None")
                return None

        except Exception as e:
            feast_online_store_misses_total.inc()
            logger.error(f"Feast lookup failed for entity_id={entity_id}: {e}")
            return None

    def get_features_as_dataframe(self, entity_id: int) -> Optional[pd.DataFrame]:
        """
        Fetch features and return as a single-row DataFrame
        suitable for model.predict().
        """
        features = self.get_features(entity_id)
        if features is None:
            return None

        # Replace None values with 0.0 (safe default for scaled features)
        for k, v in features.items():
            if v is None:
                features[k] = 0.0

        return pd.DataFrame([features])
