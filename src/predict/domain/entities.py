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

    Semantic ordinal encoding:
    LOW(1) < MEDIUM(2) < HIGH(3)
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

    @property
    def encoded_value(self) -> int:
        """Return the numeric encoding matching model training order."""
        return _SITUATION_ENCODING[self]


class ExerciseDifficulty(str, Enum):
    """Difficulty level of a full navigation exercise (prediction target)."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# Semantic encoding maps:
# Situations: LOW=1, MEDIUM=2, HIGH=3
_SITUATION_ENCODING: dict[SituationDifficulty, int] = {
    SituationDifficulty.LOW: 1,
    SituationDifficulty.MEDIUM: 2,
    SituationDifficulty.HIGH: 3,
}

# Exercise output classes: Class 0=LOW, Class 1=MEDIUM, Class 2=HIGH
EXERCISE_DECODING: dict[int, ExerciseDifficulty] = {
    0: ExerciseDifficulty.LOW,
    1: ExerciseDifficulty.MEDIUM,
    2: ExerciseDifficulty.HIGH,
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
