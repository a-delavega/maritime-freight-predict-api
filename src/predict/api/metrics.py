"""
Prometheus metrics registry and telemetry collectors for model inference.
"""

from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Shared custom registry to avoid polluting with default python metrics if desired
REGISTRY = CollectorRegistry(auto_describe=True)

# 1. Total prediction calls grouped by predicted class and endpoint
PREDICTIONS_TOTAL = Counter(
    "model_predictions_total",
    "Total count of predictions produced by the model",
    ["predicted_class", "endpoint"],
    registry=REGISTRY,
)

# 2. Prediction latency histogram (in seconds)
INFERENCE_DURATION_SECONDS = Histogram(
    "model_inference_duration_seconds",
    "Latency of pure XGBoost model inference in seconds",
    buckets=[0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25],
    registry=REGISTRY,
)

# 3. HTTP Request latency histogram
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "Latency of HTTP requests in seconds",
    ["method", "path", "status_code"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=REGISTRY,
)

# 4. Service health & model loaded gauge
MODEL_LOADED_GAUGE = Gauge(
    "model_loaded_status",
    "Indicates if model pipeline is currently loaded in memory (1=loaded, 0=not loaded)",
    registry=REGISTRY,
)


def get_metrics_snapshot() -> bytes:
    """Format and return all collected Prometheus metrics in standard exposition text format."""
    return generate_latest(REGISTRY)
