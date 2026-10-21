"""
API dependency injection.

Provides the use case instance to route handlers via FastAPI Depends.
"""

from __future__ import annotations

from fastapi import Request

from predict.application.use_cases import ClassifyExerciseDifficultyUseCase


def get_use_case(
    request: Request,
) -> ClassifyExerciseDifficultyUseCase | None:
    """Retrieve the use case from app state (set during lifespan).

    Returns None if the model failed to load, allowing the health
    endpoint to report a degraded state instead of crashing.
    """
    return getattr(request.app.state, "use_case", None)
