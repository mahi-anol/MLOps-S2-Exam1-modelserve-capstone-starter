"""
ModelServe — Prometheus Metrics

Defines all Prometheus metrics for the inference service.
Exposed at GET /metrics in Prometheus text format.

Required metrics (from exam spec):
  - prediction_requests_total (Counter)
  - prediction_duration_seconds (Histogram)
  - prediction_errors_total (Counter)
  - model_version_info (Gauge with version label)
  - feast_online_store_hits_total (Counter)
  - feast_online_store_misses_total (Counter)
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST


# --- Prediction metrics ---
prediction_requests_total = Counter(
    "prediction_requests_total",
    "Total number of prediction requests received",
    labelnames=["method", "endpoint", "status"],
)

prediction_duration_seconds = Histogram(
    "prediction_duration_seconds",
    "Time taken to process each prediction request (seconds)",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

prediction_errors_total = Counter(
    "prediction_errors_total",
    "Total number of failed prediction requests",
    labelnames=["error_type"],
)

# --- Model info ---
model_version_info = Gauge(
    "model_version_info",
    "Currently served model version",
    labelnames=["version"],
)

# --- Feast metrics ---
feast_online_store_hits_total = Counter(
    "feast_online_store_hits_total",
    "Successful feature lookups from Feast online store",
)

feast_online_store_misses_total = Counter(
    "feast_online_store_misses_total",
    "Failed or empty feature lookups from Feast online store",
)


def get_metrics() -> bytes:
    """Generate Prometheus metrics in text exposition format."""
    return generate_latest()


def get_metrics_content_type() -> str:
    """Return the correct Content-Type header for Prometheus metrics."""
    return CONTENT_TYPE_LATEST
