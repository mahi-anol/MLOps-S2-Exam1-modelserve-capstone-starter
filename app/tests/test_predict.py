"""
ModelServe — Tests for FastAPI Inference Service

Tests all endpoints with mocked MLflow and Feast dependencies.
These tests run in CI (GitHub Actions) and during the TA demo.

Usage:
    pytest app/tests/test_predict.py -v
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# ─── Mock Setup ──────────────────────────────────────────────────────────────
# We need to mock model_loader and feature_client BEFORE importing the app,
# because the app initializes them at module level.


@pytest.fixture
def mock_model_loader():
    """Mock the ModelLoader to return predictions without MLflow."""
    with patch("app.main.model_loader") as mock_ml:
        mock_ml.is_loaded = True
        mock_ml.model_version = "1"
        mock_ml.load.return_value = True
        mock_ml.predict.return_value = {
            "prediction": 0,
            "probability": 0.123456,
        }
        yield mock_ml


@pytest.fixture
def mock_feature_client():
    """Mock the FeatureClient to return features without Feast/Redis."""
    with patch("app.main.feature_client") as mock_fc:
        mock_fc.is_connected = True
        mock_fc.connect.return_value = True

        # Return a realistic feature DataFrame
        mock_features = pd.DataFrame(
            [
                {
                    "merchant": 42,
                    "category": 5,
                    "gender": 1,
                    "state": 10,
                    "amt": 0.5,
                    "hour": -0.3,
                    "day_of_week": 0.2,
                    "month": 0.1,
                    "is_weekend": 0,
                    "time_of_day": 1,
                    "amt_log": 0.4,
                    "amt_squared": 0.25,
                    "amt_x_category": 2.5,
                    "amt_x_merchant": 21.0,
                    "merchant_avg_amt": 0.3,
                    "merchant_std_amt": 0.1,
                    "category_avg_amt": 0.4,
                    "category_std_amt": 0.2,
                }
            ]
        )
        mock_fc.get_features_as_dataframe.return_value = mock_features
        mock_fc.get_features.return_value = mock_features.iloc[0].to_dict()
        yield mock_fc


@pytest.fixture
def client(mock_model_loader, mock_feature_client):
    """Create a TestClient with mocked dependencies."""
    from app.main import app

    return TestClient(app)


# ─── Health Endpoint ─────────────────────────────────────────────────────────


class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status_field(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_has_model_version(self, client):
        data = client.get("/health").json()
        assert "model_version" in data
        assert data["model_version"] == "1"


# ─── POST /predict ───────────────────────────────────────────────────────────


class TestPredict:
    def test_predict_returns_200(self, client):
        response = client.post("/predict", json={"entity_id": 12345})
        assert response.status_code == 200

    def test_predict_response_schema(self, client):
        data = client.post("/predict", json={"entity_id": 12345}).json()
        assert "prediction" in data
        assert "probability" in data
        assert "model_version" in data
        assert "timestamp" in data

    def test_predict_returns_valid_prediction(self, client):
        data = client.post("/predict", json={"entity_id": 12345}).json()
        assert data["prediction"] in [0, 1]
        assert 0.0 <= data["probability"] <= 1.0

    def test_predict_invalid_input_missing_field(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_invalid_input_wrong_type(self, client):
        response = client.post("/predict", json={"entity_id": "not_a_number"})
        assert response.status_code == 422

    def test_predict_entity_not_found(self, client, mock_feature_client):
        mock_feature_client.get_features_as_dataframe.return_value = None
        response = client.post("/predict", json={"entity_id": 99999})
        assert response.status_code == 400

    def test_predict_calls_feast(self, client, mock_feature_client):
        client.post("/predict", json={"entity_id": 12345})
        mock_feature_client.get_features_as_dataframe.assert_called_once_with(12345)

    def test_predict_calls_model(self, client, mock_model_loader):
        client.post("/predict", json={"entity_id": 12345})
        mock_model_loader.predict.assert_called_once()


# ─── GET /predict/{entity_id} ────────────────────────────────────────────────


class TestPredictExplain:
    def test_explain_returns_200(self, client):
        response = client.get("/predict/12345?explain=true")
        assert response.status_code == 200

    def test_explain_includes_features(self, client):
        data = client.get("/predict/12345?explain=true").json()
        assert "features" in data
        assert isinstance(data["features"], dict)
        assert "amt" in data["features"]
        assert "merchant" in data["features"]

    def test_explain_has_prediction_fields(self, client):
        data = client.get("/predict/12345?explain=true").json()
        assert "prediction" in data
        assert "probability" in data
        assert "model_version" in data


# ─── Metrics Endpoint ────────────────────────────────────────────────────────


class TestMetrics:
    def test_metrics_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_has_prometheus_format(self, client):
        response = client.get("/metrics")
        content = response.text
        assert "prediction_requests_total" in content

    def test_metrics_has_required_metrics(self, client):
        # Make a prediction first to generate metrics
        client.post("/predict", json={"entity_id": 12345})
        response = client.get("/metrics")
        content = response.text
        assert "prediction_requests_total" in content
        assert "prediction_duration_seconds" in content
        assert "prediction_errors_total" in content
        assert "model_version_info" in content
