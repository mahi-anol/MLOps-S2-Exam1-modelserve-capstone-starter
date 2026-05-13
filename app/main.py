"""
ModelServe — FastAPI Inference Service

Production-grade ML serving API for fraud detection.

Endpoints:
  GET  /health                          → Health check with model version
  POST /predict                         → Predict fraud for entity_id
  GET  /predict/{entity_id}?explain=true → Predict with feature explanation
  GET  /metrics                         → Prometheus metrics

Design:
  - Model loaded from MLflow Registry ONCE on startup
  - Features fetched via Feast SDK (not direct Redis)
  - Prometheus metrics tracked for all requests
  - Structured JSON errors with proper HTTP status codes
"""

import time
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from app.schemas import (
    PredictRequest,
    PredictResponse,
    ExplainResponse,
    HealthResponse,
    ErrorResponse,
)
from app.model_loader import ModelLoader
from app.feature_client import FeatureClient
from app.metrics import (
    prediction_requests_total,
    prediction_duration_seconds,
    prediction_errors_total,
    model_version_info,
    get_metrics,
    get_metrics_content_type,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances — initialized on startup
model_loader = ModelLoader()
feature_client = FeatureClient()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and connect to Feast on startup."""
    logger.info("Starting ModelServe inference service...")

    # Load model from MLflow
    if not model_loader.load():
        logger.error("Failed to load model on startup — service will be unhealthy")
    else:
        # Set the model version gauge for Prometheus
        model_version_info.labels(version=model_loader.model_version).set(1)

    # Connect to Feast
    if not feature_client.connect():
        logger.error("Failed to connect to Feast — predictions will fail")

    yield

    logger.info("Shutting down ModelServe inference service...")


app = FastAPI(
    title="ModelServe",
    description="Production ML serving platform for fraud detection",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Health ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint. Returns model version."""
    return HealthResponse(
        status="healthy" if model_loader.is_loaded else "unhealthy",
        model_version=model_loader.model_version,
    )


# ─── POST /predict ───────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def predict(request: PredictRequest):
    """
    Run fraud prediction for a given entity_id (credit card number).
    Fetches features from Feast, runs inference through MLflow model.
    """
    start_time = time.time()

    try:
        # Fetch features from Feast
        features_df = feature_client.get_features_as_dataframe(request.entity_id)
        if features_df is None:
            prediction_errors_total.labels(error_type="feature_not_found").inc()
            raise HTTPException(
                status_code=400,
                detail=f"No features found for entity_id={request.entity_id}. "
                       f"Entity may not exist in the feature store.",
            )

        # Run prediction
        result = model_loader.predict(features_df)

        # Build response
        response = PredictResponse(
            prediction=result["prediction"],
            probability=round(result["probability"], 6),
            model_version=model_loader.model_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Record metrics
        duration = time.time() - start_time
        prediction_duration_seconds.observe(duration)
        prediction_requests_total.labels(
            method="POST", endpoint="/predict", status="success"
        ).inc()

        return response

    except HTTPException:
        raise
    except Exception as e:
        duration = time.time() - start_time
        prediction_duration_seconds.observe(duration)
        prediction_errors_total.labels(error_type="prediction_error").inc()
        prediction_requests_total.labels(
            method="POST", endpoint="/predict", status="error"
        ).inc()
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── GET /predict/{entity_id} ────────────────────────────────────────────────

@app.get(
    "/predict/{entity_id}",
    response_model=ExplainResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def predict_explain(entity_id: int, explain: bool = False):
    """
    Predict with optional feature explanation.
    When explain=true, returns the feature values used for prediction.
    """
    start_time = time.time()

    try:
        # Fetch features
        features_df = feature_client.get_features_as_dataframe(entity_id)
        if features_df is None:
            prediction_errors_total.labels(error_type="feature_not_found").inc()
            raise HTTPException(
                status_code=400,
                detail=f"No features found for entity_id={entity_id}",
            )

        # Run prediction
        result = model_loader.predict(features_df)

        # Get feature values as dict for explanation
        feature_values = features_df.iloc[0].to_dict()
        # Convert numpy types to native Python for JSON serialization
        feature_values = {k: float(v) if v is not None else None for k, v in feature_values.items()}

        response = ExplainResponse(
            prediction=result["prediction"],
            probability=round(result["probability"], 6),
            model_version=model_loader.model_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            features=feature_values,
        )

        duration = time.time() - start_time
        prediction_duration_seconds.observe(duration)
        prediction_requests_total.labels(
            method="GET", endpoint="/predict/{entity_id}", status="success"
        ).inc()

        return response

    except HTTPException:
        raise
    except Exception as e:
        duration = time.time() - start_time
        prediction_duration_seconds.observe(duration)
        prediction_errors_total.labels(error_type="prediction_error").inc()
        prediction_requests_total.labels(
            method="GET", endpoint="/predict/{entity_id}", status="error"
        ).inc()
        logger.error(f"Explain prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Metrics ─────────────────────────────────────────────────────────────────

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics in text exposition format."""
    return Response(
        content=get_metrics(),
        media_type=get_metrics_content_type(),
    )
