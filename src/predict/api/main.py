"""
FastAPI application entrypoint with lifespan events.

The model is loaded once at startup via the lifespan context manager
and shared across all requests through app.state.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse

from predict.infrastructure.model_loader import load_pipeline, ModelLoadError
from predict.infrastructure.xgboost_adapter import XGBoostAdapter
from predict.application.use_cases import ClassifyExerciseDifficultyUseCase
from predict.api.metrics import MODEL_LOADED_GAUGE
from predict.api.middleware import RequestLoggingMiddleware
from predict.api.routes import router


logger = logging.getLogger(__name__)

# Default model path — can be overridden via environment variable
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "model_pipeline.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load model at startup, cleanup at shutdown."""
    import os

    model_path = os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH))
    logger.info("Loading model from: %s", model_path)

    try:
        pipeline = load_pipeline(model_path)
        adapter = XGBoostAdapter(pipeline)
        use_case = ClassifyExerciseDifficultyUseCase(adapter)
        app.state.use_case = use_case
        MODEL_LOADED_GAUGE.set(1)
        logger.info("Model loaded successfully")
    except ModelLoadError as e:
        logger.error("Failed to load model: %s", e)
        app.state.use_case = None
        MODEL_LOADED_GAUGE.set(0)

    yield

    # Cleanup
    if hasattr(app.state, "use_case"):
        del app.state.use_case
    MODEL_LOADED_GAUGE.set(0)
    logger.info("Application shutdown complete")


app = FastAPI(
    title="🚢 Maritime Freight & Navigation Inference Service",
    description="""
### Production-Grade Machine Learning Microservice

Welcome to the **Maritime Predict API documentation**. This service hosts an optimized **XGBoost classification pipeline** (`StandardScaler + XGBClassifier`) to accurately assess maritime encounter complexity and predict exercise difficulty levels (`LOW`, `MEDIUM`, `HIGH`).

---

#### 🌟 Key Architectural Capabilities
* **Clean Architecture**: Complete separation of business rules, interface adapters, and presentation layers.
* **Non-blocking Concurrency**: CPU-bound model evaluation executed across worker threadpools to protect event loop responsiveness.
* **Observability First**: Live telemetry, Prometheus scraping endpoints, and interactive Chart.js dashboards.
* **Strict Type Safety**: Endpoints validated using Pydantic v2 schemas and contract-driven schemas.

---
* **Live Telemetry Dashboard**: [View Metrics Dashboard](/dashboard)
* **Prometheus Exposition Endpoint**: [View Prometheus Text Metrics](/metrics)
    """,
    version="1.0.0",
    docs_url=None,  # Custom modern styled docs served below
    redoc_url="/redoc",
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "Inference Services",
            "description": "High-throughput model serving endpoints for single and batch predictions.",
        },
        {
            "name": "System & Observability",
            "description": "Service health probes, Prometheus telemetry, and real-time visual dashboards.",
        },
    ],
)

app.add_middleware(RequestLoggingMiddleware)

app.include_router(router)


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html() -> HTMLResponse:
    """Renders a customized, modern dark-themed Swagger UI documentation portal."""
    swagger_html = get_swagger_ui_html(
        openapi_url=app.openapi_url or "/openapi.json",
        title=f"{app.title} - API Reference",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
    )
    
    # Modern Dark Theme & Typography Injection for a Big Tech / Developer Portal aesthetic
    custom_theme = """
    <style>
      body {
        background-color: #0b0f19 !important;
        color: #e2e8f0 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
      }
      .swagger-ui {
        filter: invert(88%) hue-rotate(180deg) !important;
      }
      .swagger-ui .topbar {
        background-color: #111827 !important;
        border-bottom: 1px solid #1f2937 !important;
      }
      .swagger-ui img {
        filter: invert(100%) hue-rotate(180deg) !important;
      }
      .swagger-ui .info {
        margin: 30px 0 !important;
      }
      .swagger-ui .info .title {
        color: #0f172a !important;
        font-weight: 800 !important;
        letter-spacing: -0.025em !important;
      }
      .swagger-ui .scheme-container {
        background: transparent !important;
        box-shadow: none !important;
      }
      .swagger-ui .btn.execute {
        background-color: #0284c7 !important;
        border-color: #0284c7 !important;
        color: #ffffff !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
      }
      .swagger-ui .opblock {
        border-radius: 10px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        margin-bottom: 16px !important;
      }
    </style>
    """
    
    html_content = swagger_html.body.decode("utf-8").replace("</head>", f"{custom_theme}</head>")
    return HTMLResponse(content=html_content)
