"""
Shared test fixtures for the maritime-predict-api test suite.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from predict.application.use_cases import ClassifyExerciseDifficultyUseCase
from predict.domain.entities import (
    ExerciseDifficulty,
    ExercisePrediction,
)
from predict.domain.ports import DifficultyClassifierPort


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@pytest.fixture
def models_dir() -> Path:
    """Path to the models directory."""
    return MODELS_DIR


@pytest.fixture
def dummy_classifier() -> MagicMock:
    """A mock classifier that always predicts MEDIUM."""
    mock = MagicMock(spec=DifficultyClassifierPort)
    mock.predict.return_value = ExercisePrediction(
        difficulty=ExerciseDifficulty.MEDIUM,
        probabilities={"HIGH": 0.2, "LOW": 0.1, "MEDIUM": 0.7},
        situation_count=3,
    )
    return mock


@pytest.fixture
def dummy_use_case(dummy_classifier: MagicMock) -> ClassifyExerciseDifficultyUseCase:
    """A use case with a mocked classifier."""
    return ClassifyExerciseDifficultyUseCase(dummy_classifier)
