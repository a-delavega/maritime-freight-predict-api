"""
API routes for the maritime navigation predict service.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from predict.api.dependencies import get_use_case
from predict.api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from predict.application.use_cases import ClassifyExerciseDifficultyUseCase
from predict.domain.entities import SituationDifficulty
from predict.infrastructure.xgboost_adapter import PredictionError


MODEL_VERSION = "1.0.0"

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(
    use_case: ClassifyExerciseDifficultyUseCase | None = Depends(get_use_case),
) -> HealthResponse:
    """Health check endpoint. Returns model loading status."""
    return HealthResponse(
        status="healthy" if use_case is not None else "degraded",
        model_loaded=use_case is not None,
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    use_case: ClassifyExerciseDifficultyUseCase | None = Depends(get_use_case),
) -> PredictionResponse:
    """Classify the difficulty of a navigation exercise.

    Receives the difficulty of each situation in the exercise,
    aggregates them, and predicts the overall exercise difficulty.
    """
    if use_case is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is unavailable.",
        )

    # Map schema enum to domain enum
    domain_situations = [
        SituationDifficulty(s.value) for s in request.situations
    ]

    try:
        prediction = use_case.execute(domain_situations)
    except PredictionError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e}",
        )

    return PredictionResponse(
        exercise_difficulty=prediction.difficulty.value,
        probabilities=prediction.probabilities,
        situation_count=prediction.situation_count,
        model_version=MODEL_VERSION,
    )
