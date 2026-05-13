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

    # ─── Internal: load a specific version URI ──────────────────────────────
    def _load_uri(self, model_uri: str, version_label: str) -> bool:
        """
        Load a model from a fully-qualified MLflow URI (e.g.
        `models:/fraud-detection-model/3` or `.../Production`).
        Updates self.model, self.raw_model, self.feature_order, self.model_version.
        """
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

        self.model_version = version_label
        self.is_loaded = True
        return True

    def load(self) -> bool:
        try:
            logger.info(f"Connecting to MLflow at {self.tracking_uri}")
            mlflow.set_tracking_uri(self.tracking_uri)

            model_uri = f"models:/{self.model_name}/Production"

            # Resolve the concrete version number for Production so we can
            # report it on /health and in the Prometheus gauge.
            client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
            versions = client.search_model_versions(f"name='{self.model_name}'")
            production_versions = [v for v in versions if v.current_stage == "Production"]
            if production_versions:
                version_label = production_versions[0].version
            else:
                version_label = "latest"

            return self._load_uri(model_uri, version_label)

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            self.is_loaded = False
            return False

    # ─── Rollback ───────────────────────────────────────────────────────────
    def rollback(self, target_version: Optional[str] = None) -> dict:
        """
        Switch the in-memory model to a previous version.

        - If `target_version` is given, load that exact version.
        - Otherwise, pick the version immediately below the current one
          (the most recent prior version that is NOT the currently loaded one).

        Promotes the target version to MLflow stage `Production` and demotes
        the currently-served version to `Archived`, so a restart of the
        container will also come up on the rolled-back version.

        Returns: { previous_version, current_version, model_name }
        Raises:  RuntimeError if no rollback target is available.
        """
        mlflow.set_tracking_uri(self.tracking_uri)
        client = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)

        all_versions = client.search_model_versions(f"name='{self.model_name}'")
        if not all_versions:
            raise RuntimeError(f"No versions registered for model '{self.model_name}'")

        # Sort numerically by version (MLflow returns version as a string).
        sorted_versions = sorted(
            all_versions,
            key=lambda v: int(v.version),
            reverse=True,
        )

        previous_version = self.model_version

        # Pick the target version
        if target_version is not None:
            target = next(
                (v for v in sorted_versions if str(v.version) == str(target_version)),
                None,
            )
            if target is None:
                raise RuntimeError(
                    f"Version {target_version} not found for model '{self.model_name}'"
                )
        else:
            # Auto: most recent version that isn't the currently-loaded one
            target = next(
                (v for v in sorted_versions if str(v.version) != str(self.model_version)),
                None,
            )
            if target is None:
                raise RuntimeError(
                    f"No prior version available to roll back to "
                    f"(only one version of '{self.model_name}' exists)"
                )

        # Load the target version into memory first — if this fails we
        # don't want to have touched any stage transitions.
        model_uri = f"models:/{self.model_name}/{target.version}"
        self._load_uri(model_uri, str(target.version))

        # Update MLflow registry stages so future restarts also use this version.
        try:
            # Archive the previous Production version (if it's still Production)
            if previous_version not in ("unknown", "latest"):
                try:
                    client.transition_model_version_stage(
                        name=self.model_name,
                        version=previous_version,
                        stage="Archived",
                        archive_existing_versions=False,
                    )
                except Exception as e:
                    logger.warning(f"Could not archive v{previous_version}: {e}")

            # Promote the rollback target to Production
            client.transition_model_version_stage(
                name=self.model_name,
                version=target.version,
                stage="Production",
                archive_existing_versions=True,
            )
        except Exception as e:
            # In-memory rollback already succeeded; surface the registry
            # update failure but don't undo the load.
            logger.error(f"Stage transition failed after in-memory rollback: {e}")

        logger.info(
            f"Rollback complete: {self.model_name} v{previous_version} → v{target.version}"
        )

        return {
            "model_name": self.model_name,
            "previous_version": previous_version,
            "current_version": str(target.version),
        }

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