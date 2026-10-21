"""
Domain ports (interfaces) for the difficulty classifier.

These abstract base classes define the contracts that infrastructure
adapters must implement (Dependency Inversion Principle).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from predict.domain.entities import AggregatedFeatures, ExercisePrediction


class DifficultyClassifierPort(ABC):
    """Port for classifying exercise difficulty from aggregated features."""

    @abstractmethod
    def predict(self, features: AggregatedFeatures) -> ExercisePrediction:
        """Predict the difficulty of an exercise.

        Args:
            features: Aggregated situation features for a single exercise.

        Returns:
            ExercisePrediction with the predicted class and probabilities.

        Raises:
            PredictionError: If the model fails to produce a prediction.
        """
        ...
