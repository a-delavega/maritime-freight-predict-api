"""
Pydantic v2 schemas for API request/response validation.
"""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class SituationDifficultySchema(str, Enum):
    """Valid difficulty values for a navigation situation."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class PredictionRequest(BaseModel):
    """Request body for exercise difficulty prediction.

    The client sends the list of difficulty levels for each situation
    in the exercise. The API aggregates them internally and classifies.
    """

    situations: list[SituationDifficultySchema] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Difficulty level of each situation in the exercise",
        json_schema_extra={
            "examples": [["LOW", "MEDIUM", "HIGH", "MEDIUM", "MEDIUM", "LOW"]]
        },
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "situations": [
                        "LOW",
                        "MEDIUM",
                        "HIGH",
                        "MEDIUM",
                        "MEDIUM",
                        "LOW",
                    ]
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response body with the exercise difficulty prediction."""

    exercise_difficulty: str = Field(
        ..., description="Predicted difficulty class: LOW, MEDIUM, or HIGH"
    )
    probabilities: dict[str, float] = Field(
        ...,
        description="Probability for each class",
        json_schema_extra={
            "examples": [{"HIGH": 0.6, "LOW": 0.1, "MEDIUM": 0.3}]
        },
    )
    situation_count: int = Field(
        ..., ge=1, description="Number of situations in the exercise"
    )
    model_version: str = Field(
        ..., description="Version of the model used for prediction"
    )


class HealthResponse(BaseModel):
    """Response body for the health check endpoint."""

    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(
        ..., description="Whether the model is loaded and ready"
    )
