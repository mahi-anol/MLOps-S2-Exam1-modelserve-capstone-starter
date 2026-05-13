"""
ModelServe — Pydantic Request/Response Schemas

Defines the data contracts for the FastAPI inference service endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class PredictRequest(BaseModel):
    """POST /predict request body."""
    entity_id: int = Field(..., description="Credit card number (cc_num) for feature lookup")


class PredictResponse(BaseModel):
    """POST /predict response body."""
    prediction: int = Field(..., description="Predicted class (0=legit, 1=fraud)")
    probability: float = Field(..., description="Fraud probability score")
    model_version: str = Field(..., description="MLflow model version used")
    timestamp: str = Field(..., description="Prediction timestamp in ISO 8601")


class ExplainResponse(PredictResponse):
    """GET /predict/{entity_id}?explain=true response body."""
    features: dict = Field(..., description="Feature values used for prediction")


class HealthResponse(BaseModel):
    """GET /health response body."""
    status: str = Field(default="healthy")
    model_version: str = Field(..., description="Currently loaded model version")


class ErrorResponse(BaseModel):
    """Error response body."""
    error: str
    detail: Optional[str] = None
