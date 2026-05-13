import os
import logging
import mlflow
import mlflow.pyfunc
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self):
        self.model = None
        self.raw_model = None
        self.model_version: str = "unknown"
        self.model_name: str = os.getenv("MLFLOW_MODEL_NAME", "fraud-detection-model")
        self.tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        self.is_loaded: bool = False
        self.feature_order: list = []

    def load(self) -> bool:
        try:
            logger.info(f"Connecting to MLflow at {self.tracking_uri}")
            mlflow.set_tracking_uri(self.tracking_uri)

            model_uri = f"models:/{self.model_name}/Production"
            logger.info(f"Loading model: {model_uri}")

            self.model = mlflow.pyfunc.load_model(model_uri)

            # Extract the raw sklearn/xgboost model to get feature order
            try:
                unwrapped = self.model._model_impl
                if hasattr(unwrapped, 'sklearn_model'):
                    self.raw_model = unwrapped.sklearn_model
                elif hasattr(unwrapped, 'get_booster'):
                    self.raw_model = unwrapped
                else:
                    self.raw_model = unwrapped

                # Get feature names the model was trained on
                if hasattr(self.raw_model, 'get_booster'):
                    self.feature_order = self.raw_model.get_booster().feature_names or []
                elif hasattr(self.raw_model, 'feature_names_in_'):
                    self.feature_order = list(self.raw_model.feature_names_in_)

                logger.info(f"Model feature order: {self.feature_order}")
            except Exception as e:
                logger.warning(f"Could not extract raw model: {e}")

            # Get model version
            client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
            versions = client.search_model_versions(f"name='{self.model_name}'")
            production_versions = [v for v in versions if v.current_stage == "Production"]
            if production_versions:
                self.model_version = production_versions[0].version
            else:
                self.model_version = "latest"

            self.is_loaded = True
            logger.info(f"Model loaded: {self.model_name} v{self.model_version}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            self.is_loaded = False
            return False

    def predict(self, features: pd.DataFrame) -> dict:
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Reorder columns to match training order
        if self.feature_order:
            features = features[self.feature_order]

        # Use the raw model directly to avoid pyfunc column order issues
        if self.raw_model is not None:
            prediction = self.raw_model.predict(features)
            result = {"prediction": int(prediction[0]), "probability": 0.0}
            try:
                if hasattr(self.raw_model, "predict_proba"):
                    proba = self.raw_model.predict_proba(features)
                    result["probability"] = float(proba[0][1])
            except Exception:
                logger.warning("Could not get probability", exc_info=True)
        else:
            prediction = self.model.predict(features)
            result = {"prediction": int(prediction[0]), "probability": float(prediction[0])}

        return result
