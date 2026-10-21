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

from predict.infrastructure.model_loader import load_pipeline, ModelLoadError
from predict.infrastructure.xgboost_adapter import XGBoostAdapter
from predict.application.use_cases import ClassifyExerciseDifficultyUseCase
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
        logger.info("Model loaded successfully")
    except ModelLoadError as e:
        logger.error("Failed to load model: %s", e)
        app.state.use_case = None

    yield

    # Cleanup
    if hasattr(app.state, "use_case"):
        del app.state.use_case
    logger.info("Application shutdown complete")


app = FastAPI(
    title="Maritime Navigation Predict API",
    description=(
        "Production-ready inference microservice for maritime navigation "
        "exercise difficulty classification"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(RequestLoggingMiddleware)

app.include_router(router)
