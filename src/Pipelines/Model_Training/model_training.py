"""
ModelServe — Model Training & MLflow Registration

Loads feature-engineered data, trains an XGBoost classifier for fraud detection,
logs metrics/params/artifacts to MLflow, and registers the model in the
MLflow Model Registry with stage "Production".

Also exports sample_request.json for API testing.

Usage:
    python -m src.Pipelines.Model_Training.model_training

Prerequisites:
    - MLflow and Postgres must be running (docker compose up postgres mlflow)
    - Feature engineering pipeline must have run (data/features/ must exist)
"""

import os
import json
import yaml
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from xgboost import XGBClassifier

from src.Logger import Logger

logger = Logger.get_logger(__file__)


def load_params(stage: str) -> dict:
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)[stage]


@dataclass
class TrainingConfig:
    _p = load_params("model_training")
    features_path: str = _p["features_path"]
    test_features_path: str = _p.get("test_features_path", "")
    target_column: str = _p["target_column"]
    entity_column: str = _p["entity_column"]
    model_name: str = _p["model_name"]
    algorithm: str = _p["algorithm"]
    mlflow_tracking_uri: str = _p["mlflow_tracking_uri"]
    sample_request_path: str = _p["sample_request_path"]
    hyperparameters: dict = field(
        default_factory=lambda: load_params("model_training")["hyperparameters"]
    )


class ModelTrainer:
    def __init__(self):
        self.config = TrainingConfig()

    def _load_data(self) -> pd.DataFrame:
        logger.info(f"Loading training features from {self.config.features_path}")
        df = pd.read_parquet(self.config.features_path)
        logger.info(f"Loaded shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        return df

    def _prepare_data(self, df: pd.DataFrame):
        """Split into features and target, then train/test split."""
        # Drop entity column — not a feature for the model
        drop_cols = [self.config.entity_column, self.config.target_column]
        drop_cols = [c for c in drop_cols if c in df.columns]

        feature_cols = [c for c in df.columns if c not in drop_cols]
        logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

        X = df[feature_cols]
        y = df[self.config.target_column]

        logger.info(f"Target distribution:\n{y.value_counts()}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        logger.info(
            f"Train: {X_train.shape}, Test: {X_test.shape}"
        )
        return X_train, X_test, y_train, y_test, feature_cols

    def _train_model(self, X_train, y_train):
        """Train XGBoost classifier."""
        params = self.config.hyperparameters.copy()
        logger.info(f"Training {self.config.algorithm} with params: {params}")

        model = XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            scale_pos_weight=params["scale_pos_weight"],
            eval_metric=params["eval_metric"],
            random_state=params["random_state"],
            use_label_encoder=False,
        )
        model.fit(X_train, y_train)
        logger.info("Model training complete.")
        return model

    def _evaluate_model(self, model, X_test, y_test) -> dict:
        """Compute evaluation metrics."""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        logger.info("Evaluation metrics:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        logger.info(
            f"\nClassification Report:\n"
            f"{classification_report(y_test, y_pred)}"
        )
        return metrics

    def _log_to_mlflow(self, model, metrics: dict, feature_cols: list):
        """Log everything to MLflow and register the model."""
        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment("fraud-detection")

        with mlflow.start_run(run_name="xgboost-fraud-baseline") as run:
            # Log parameters
            mlflow.log_params(self.config.hyperparameters)
            mlflow.log_param("algorithm", self.config.algorithm)
            mlflow.log_param("n_features", len(feature_cols))
            mlflow.log_param("feature_columns", json.dumps(feature_cols))

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=self.config.model_name,
            )

            run_id = run.info.run_id
            logger.info(f"MLflow run ID: {run_id}")

        # Transition latest version to Production
        client = MlflowClient(tracking_uri=self.config.mlflow_tracking_uri)
        versions = client.search_model_versions(f"name='{self.config.model_name}'")
        if versions:
            latest = max(versions, key=lambda v: int(v.version))
            client.transition_model_version_stage(
                name=self.config.model_name,
                version=latest.version,
                stage="Production",
                archive_existing_versions=True,
            )
            logger.info(
                f"Model '{self.config.model_name}' v{latest.version} -> Production"
            )

    def _export_sample_request(self, df: pd.DataFrame):
        """Export a sample_request.json with a valid entity_id for testing."""
        # Pick a valid cc_num from the dataset
        sample_entity = int(df[self.config.entity_column].iloc[0])
        sample = {"entity_id": sample_entity}

        output_path = Path(self.config.sample_request_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(sample, f, indent=2)
        logger.info(f"Saved sample request -> {output_path}: {sample}")

    def run(self):
        try:
            df = self._load_data()
            X_train, X_test, y_train, y_test, feature_cols = self._prepare_data(df)
            model = self._train_model(X_train, y_train)
            metrics = self._evaluate_model(model, X_test, y_test)
            self._log_to_mlflow(model, metrics, feature_cols)
            self._export_sample_request(df)
            logger.info("Model training pipeline complete.")
        except Exception as e:
            logger.error(f"Model training failed: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    ModelTrainer().run()
