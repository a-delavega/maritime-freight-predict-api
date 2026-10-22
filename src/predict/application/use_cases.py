"""
Application use case: Classify the difficulty of a navigation exercise.

This is the core business logic orchestrator. It encodes situation
difficulties, aggregates them, and delegates to the classifier port.
"""

from __future__ import annotations

from predict.domain.entities import (
    AggregatedFeatures,
    SituationDifficulty,
    ExercisePrediction,
)
from predict.domain.ports import DifficultyClassifierPort


class ClassifyExerciseDifficultyUseCase:
    """Orchestrates the classification of an exercise's difficulty.

    Receives a list of situation difficulties, encodes and aggregates them
    into the 4 features expected by the model, then delegates prediction
    to the injected classifier port.
    """

    def __init__(self, classifier: DifficultyClassifierPort) -> None:
        self._classifier = classifier

    def execute(
        self, situations: list[SituationDifficulty]
    ) -> ExercisePrediction:
        """Classify an exercise given its situation difficulties.

        Args:
            situations: List of difficulty levels for each situation.

        Returns:
            ExercisePrediction with class, probabilities, and metadata.

        Raises:
            ValueError: If situations is empty.
        """
        if not situations:
            raise ValueError("At least one situation difficulty is required")

        features = self._aggregate(situations)
        prediction = self._classifier.predict(features)
        return prediction

    @staticmethod
    def _aggregate(situations: list[SituationDifficulty]) -> AggregatedFeatures:
        """Encode situation difficulties and compute aggregated features."""
        encoded = [s.encoded_value for s in situations]
        high_risk_count = float(sum(1 for s in situations if s == SituationDifficulty.HIGH))
        return AggregatedFeatures(
            mean_difficulty=sum(encoded) / len(encoded),
            max_difficulty=float(max(encoded)),
            high_risk_count=high_risk_count,
            situation_count=len(encoded),
        )
