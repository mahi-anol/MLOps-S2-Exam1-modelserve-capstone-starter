"""
Feast Preprocess — Converts feature-engineered data to Feast-compatible parquet.

Takes the output of feature engineering (train_features.parquet) and produces
a parquet file suitable for Feast ingestion:
  - Keeps cc_num as the entity key
  - Adds event_timestamp column
  - Drops the target column (is_fraud)
  - Saves to training/features.parquet
"""

import yaml
import pandas as pd
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from src.Logger import Logger

logger = Logger.get_logger(__file__)


def load_params(stage: str) -> dict:
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)[stage]


@dataclass
class FeastPreprocessConfig:
    _p = load_params("feast_preprocess")
    feature_data_path: str = _p["feature_data_path"]
    output_path: str = _p["output_path"]
    entity_column: str = _p["entity_column"]
    target_column: str = _p["target_column"]


class FeastPreprocessor:
    def __init__(self):
        self.config = FeastPreprocessConfig()

    def run(self):
        try:
            logger.info(f"Loading feature data from {self.config.feature_data_path}")
            df = pd.read_parquet(self.config.feature_data_path)
            logger.info(f"Loaded shape: {df.shape}")

            # Verify entity column exists
            if self.config.entity_column not in df.columns:
                raise ValueError(
                    f"Entity column '{self.config.entity_column}' not found. "
                    f"Available: {list(df.columns)}"
                )

            # Drop target column — features only
            if self.config.target_column in df.columns:
                df = df.drop(columns=[self.config.target_column])
                logger.info(f"Dropped target column: {self.config.target_column}")

            # Add event_timestamp — Feast requires this for point-in-time joins
            df["event_timestamp"] = pd.Timestamp(datetime.now(), tz="UTC")
            logger.info("Added event_timestamp column")

            # Ensure entity column is int64 for consistent lookups
            df[self.config.entity_column] = df[self.config.entity_column].astype("int64")

            # Deduplicate by entity — keep last occurrence
            # (multiple transactions per cc_num; Feast needs one row per entity)
            before = len(df)
            df = df.drop_duplicates(subset=[self.config.entity_column], keep="last")
            logger.info(
                f"Deduplicated by {self.config.entity_column}: {before} -> {len(df)} rows"
            )

            # Save
            output_path = Path(self.config.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, index=False, engine="pyarrow")
            logger.info(f"Saved Feast-compatible features -> {output_path}")
            logger.info(f"Feature columns: {[c for c in df.columns if c not in ['cc_num', 'event_timestamp']]}")

        except Exception as e:
            logger.error(f"Feast preprocessing failed: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    FeastPreprocessor().run()
