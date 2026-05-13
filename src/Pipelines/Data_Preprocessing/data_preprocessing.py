import os
import pickle
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample

from src.Logger import Logger

logger = Logger.get_logger(__file__)


def load_params(stage: str) -> dict:
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)[stage]


@dataclass
class PreprocessConfig:
    _p = load_params("preprocessing")
    raw_train_path: str = _p["raw_train_path"]
    raw_test_path: str = _p["raw_test_path"]
    output_dir: str = _p["output_dir"]
    target_column: str = _p["target_column"]
    handle_imbalance: bool = _p["handle_imbalance"]
    scale_features: bool = _p["scale_features"]
    drop_columns: list = field(default_factory=lambda: load_params("preprocessing")["drop_columns"])
    categorical_columns: list = field(default_factory=lambda: load_params("preprocessing")["categorical_columns"])
    numerical_columns: list = field(default_factory=lambda: load_params("preprocessing")["numerical_columns"])
    passthrough_columns: list = field(default_factory=lambda: load_params("preprocessing").get("passthrough_columns", []))


class DataPreprocessor:
    def __init__(self):
        self.config = PreprocessConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()

    def _load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Loading raw train and test CSVs...")
        train = pd.read_csv(self.config.raw_train_path)
        test = pd.read_csv(self.config.raw_test_path)
        logger.info(f"Train shape: {train.shape} | Test shape: {test.shape}")
        return train, test

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = [c for c in self.config.drop_columns if c in df.columns]
        logger.info(f"Dropping columns: {cols_to_drop}")
        return df.drop(columns=cols_to_drop)

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {before - len(df)} duplicate rows")
        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = df.isnull().sum().sum()
        if missing:
            logger.info(f"Filling {missing} missing values")
            for col in self.config.numerical_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].median())
            for col in self.config.categorical_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].mode()[0])
        return df

    def _extract_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "trans_date_trans_time" in df.columns:
            dt = pd.to_datetime(df["trans_date_trans_time"])
            df["hour"] = dt.dt.hour
            df["day_of_week"] = dt.dt.dayofweek
            df["month"] = dt.dt.month
            df = df.drop(columns=["trans_date_trans_time"])
            logger.info("Extracted datetime features: hour, day_of_week, month")
        return df

    def _encode_categoricals(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Label-encoding categoricals: {self.config.categorical_columns}")
        for col in self.config.categorical_columns:
            if col not in train.columns:
                continue
            le = LabelEncoder()
            train[col] = le.fit_transform(train[col].astype(str))

            classes_map = {v: i for i, v in enumerate(le.classes_)}
            test[col] = test[col].astype(str).map(classes_map).fillna(-1).astype(int)

            self.label_encoders[col] = le
        return train, test

    def _scale_features(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        num_cols = [c for c in self.config.numerical_columns if c in train.columns]
        logger.info(f"Scaling numerical features: {num_cols}")
        train[num_cols] = self.scaler.fit_transform(train[num_cols])
        test[num_cols] = self.scaler.transform(test[num_cols])
        return train, test

    def _handle_imbalance(self, train: pd.DataFrame) -> pd.DataFrame:
        target = self.config.target_column
        majority = train[train[target] == 0]
        minority = train[train[target] == 1]
        logger.info(
            f"Class counts before resampling -> 0: {len(majority)}, 1: {len(minority)}"
        )
        minority_upsampled = resample(
            minority, replace=True, n_samples=len(majority), random_state=42
        )
        train = pd.concat([majority, minority_upsampled]).sample(
            frac=1, random_state=42
        )
        logger.info(
            f"Class counts after resampling -> 0: {len(majority)}, 1: {len(minority_upsampled)}"
        )
        return train

    def _save(self, train: pd.DataFrame, test: pd.DataFrame):
        train_out = Path(self.config.output_dir) / "train_processed.csv"
        test_out = Path(self.config.output_dir) / "test_processed.csv"
        train.to_csv(train_out, index=False)
        test.to_csv(test_out, index=False)
        logger.info(f"Saved processed data -> {train_out} | {test_out}")

        # Save encoders and scaler for reproducibility
        artifacts_dir = Path("artifacts/preprocessing")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        with open(artifacts_dir / "label_encoders.pkl", "wb") as f:
            pickle.dump(self.label_encoders, f)
        with open(artifacts_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Saved preprocessing artifacts -> {artifacts_dir}")

    def run(self):
        try:
            train, test = self._load_data()

            train = self._drop_columns(train)
            test = self._drop_columns(test)

            train = self._remove_duplicates(train)
            test = self._remove_duplicates(test)

            train = self._handle_missing(train)
            test = self._handle_missing(test)

            train = self._extract_datetime_features(train)
            test = self._extract_datetime_features(test)

            train, test = self._encode_categoricals(train, test)

            if self.config.scale_features:
                train, test = self._scale_features(train, test)

            if self.config.handle_imbalance:
                train = self._handle_imbalance(train)

            self._save(train, test)
            logger.info("Preprocessing complete.")

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    DataPreprocessor().run()
