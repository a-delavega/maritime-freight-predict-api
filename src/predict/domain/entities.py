"""
Domain entities for maritime navigation exercise difficulty classification.

These entities represent the core domain concepts, independent of any
framework or infrastructure (Clean Architecture - innermost layer).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SituationDifficulty(str, Enum):
    """Difficulty level of an individual navigation situation.

    The numeric values represent the encoding used by the trained model.
    Order: HIGH(0) < LOW(1) < MEDIUM(2) — matching training label encoder.
    """

    HIGH = "HIGH"
    LOW = "LOW"
    MEDIUM = "MEDIUM"

    @property
    def encoded_value(self) -> int:
        """Return the numeric encoding matching model training order."""
        return _SITUATION_ENCODING[self]


class ExerciseDifficulty(str, Enum):
    """Difficulty level of a full navigation exercise (prediction target)."""

    HIGH = "HIGH"
    LOW = "LOW"
    MEDIUM = "MEDIUM"


# Encoding maps
_SITUATION_ENCODING: dict[SituationDifficulty, int] = {
    SituationDifficulty.HIGH: 0,
    SituationDifficulty.LOW: 1,
    SituationDifficulty.MEDIUM: 2,
}

EXERCISE_DECODING: dict[int, ExerciseDifficulty] = {
    0: ExerciseDifficulty.HIGH,
    1: ExerciseDifficulty.LOW,
    2: ExerciseDifficulty.MEDIUM,
}


@dataclass(frozen=True)
class AggregatedFeatures:
    """Aggregated features of an exercise's situations, ready for model input.

    These are the 4 features the trained XGBClassifier expects:
    - mean_difficulty: mean of encoded SituationDifficulty values
    - max_difficulty: max of encoded SituationDifficulty values
    - min_difficulty: min of encoded SituationDifficulty values
    - situation_count: count of situations in the exercise
    """

    mean_difficulty: float
    max_difficulty: float
    min_difficulty: float
    situation_count: int

    def to_feature_list(self) -> list[float]:
        """Convert to ordered list matching model training feature order."""
        return [
            self.mean_difficulty,
            self.max_difficulty,
            self.min_difficulty,
            float(self.situation_count),
        ]


@dataclass(frozen=True)
class ExercisePrediction:
    """Result of classifying an exercise's difficulty."""

    difficulty: ExerciseDifficulty
    probabilities: dict[str, float]  # {"HIGH": 0.6, "LOW": 0.1, "MEDIUM": 0.3}
    situation_count: int
