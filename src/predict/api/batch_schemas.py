"""
Batch prediction Pydantic v2 schemas.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from predict.api.schemas import PredictionResponse, SituationDifficultySchema


class BatchPredictionRequest(BaseModel):
    """Batch request containing multiple exercises to classify."""

    exercises: dict[str, list[SituationDifficultySchema]] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Dictionary mapping exercise ID/name to list of situation difficulties",
        json_schema_extra={
            "examples": [
                {
                    "exercises": {
                        "exercise_101": ["LOW", "MEDIUM", "HIGH"],
                        "exercise_102": ["HIGH", "HIGH", "MEDIUM"],
                    }
                }
            ]
        },
    )


class BatchPredictionResponse(BaseModel):
    """Batch response returning predictions mapped by exercise ID/name."""

    predictions: dict[str, PredictionResponse]
    total_exercises: int
    model_version: str
