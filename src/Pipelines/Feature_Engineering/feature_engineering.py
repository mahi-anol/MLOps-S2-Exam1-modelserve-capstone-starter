import os
import pickle
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler

from src.Logger import Logger

logger = Logger.get_logger(__file__)


def load_params(stage: str) -> dict:
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)[stage]


@dataclass
class FeatureConfig:
    _p = load_params("feature_engineering")
    processed_train_path: str = _p["processed_train_path"]
    processed_test_path: str = _p["processed_test_path"]
    output_dir: str = _p["output_dir"]
    artifacts_dir: str = _p["artifacts_dir"]
    target_column: str = _p["target_column"]
    time_of_day_bins: list = field(default_factory=lambda: load_params("feature_engineering")["time_of_day_bins"])
    time_of_day_labels: list = field(default_factory=lambda: load_params("feature_engineering")["time_of_day_labels"])


class FeatureEngineer:
    def __init__(self):
        self.config = FeatureConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.artifacts_dir, exist_ok=True)
        self.scaler = StandardScaler()

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Loading processed train and test CSVs...")
        train = pd.read_csv(self.config.processed_train_path)
        test = pd.read_csv(self.config.processed_test_path)
        logger.info(f"Train shape: {train.shape} | Test shape: {test.shape}")
        return train, test

    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Engineering time-based features...")
        df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

        df["time_of_day"] = pd.cut(
            df["hour"],
            bins=self.config.time_of_day_bins,
            labels=self.config.time_of_day_labels,
            right=False,
        ).astype(str)

        time_map = {
            label: i for i, label in enumerate(self.config.time_of_day_labels)
        }
        df["time_of_day"] = df["time_of_day"].map(time_map).fillna(-1).astype(int)
        return df

    def _amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Engineering amount-based features...")
        df["amt_log"] = np.log1p(np.abs(df["amt"]))
        df["amt_squared"] = df["amt"] ** 2
        return df

    def _interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Engineering interaction features...")
        df["amt_x_category"] = df["amt"] * df["category"]
        df["amt_x_merchant"] = df["amt"] * df["merchant"]
        return df

    def _aggregation_features(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Engineering aggregation features...")

        merchant_stats = (
            train.groupby("merchant")["amt"]
            .agg(merchant_avg_amt="mean", merchant_std_amt="std")
            .reset_index()
        )
        category_stats = (
            train.groupby("category")["amt"]
            .agg(category_avg_amt="mean", category_std_amt="std")
            .reset_index()
        )

        train = train.merge(merchant_stats, on="merchant", how="left")
        train = train.merge(category_stats, on="category", how="left")
        test = test.merge(merchant_stats, on="merchant", how="left")
        test = test.merge(category_stats, on="category", how="left")

        for col in [
            "merchant_avg_amt",
            "merchant_std_amt",
            "category_avg_amt",
            "category_std_amt",
        ]:
            median_val = train[col].median()
            train[col] = train[col].fillna(median_val)
            test[col] = test[col].fillna(median_val)

        logger.info("Added merchant and category aggregation features")
        return train, test

    def _scale_new_features(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        new_num_cols = [
            "amt_log",
            "amt_squared",
            "amt_x_category",
            "amt_x_merchant",
            "merchant_avg_amt",
            "merchant_std_amt",
            "category_avg_amt",
            "category_std_amt",
        ]
        cols = [c for c in new_num_cols if c in train.columns]
        logger.info(f"Scaling new numerical features: {cols}")
        train[cols] = self.scaler.fit_transform(train[cols])
        test[cols] = self.scaler.transform(test[cols])
        return train, test

    def _save_artifacts(self):
        scaler_path = Path(self.config.artifacts_dir) / "feature_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Saved scaler artifact -> {scaler_path}")

    def _save(self, train: pd.DataFrame, test: pd.DataFrame):
        train_out = Path(self.config.output_dir) / "train_features.parquet"
        test_out = Path(self.config.output_dir) / "test_features.parquet"
        train.to_parquet(train_out, index=False, engine="pyarrow")
        test.to_parquet(test_out, index=False, engine="pyarrow")
        logger.info(f"Saved feature data -> {train_out} | {test_out}")
        logger.info(
            f"Final train shape: {train.shape} | Final test shape: {test.shape}"
        )
        logger.info(f"Columns: {list(train.columns)}")

    def run(self):
        try:
            train, test = self._load_data()

            train = self._time_features(train)
            test = self._time_features(test)

            train = self._amount_features(train)
            test = self._amount_features(test)

            train = self._interaction_features(train)
            test = self._interaction_features(test)

            train, test = self._aggregation_features(train, test)
            train, test = self._scale_new_features(train, test)

            self._save_artifacts()
            self._save(train, test)
            logger.info("Feature engineering complete.")

        except Exception as e:
            logger.error(f"Feature engineering failed: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    FeatureEngineer().run()
