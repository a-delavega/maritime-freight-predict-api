"""
Infrastructure: XGBoost adapter.

Concrete implementation of DifficultyClassifierPort that uses
a serialized sklearn Pipeline (StandardScaler + XGBClassifier).
"""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline

from predict.domain.entities import (
    AggregatedFeatures,
    ExerciseDifficulty,
    EXERCISE_DECODING,
    ExercisePrediction,
)
from predict.domain.ports import DifficultyClassifierPort


class PredictionError(Exception):
    """Raised when model inference fails."""

    pass


class XGBoostAdapter(DifficultyClassifierPort):
    """Adapter that wraps a sklearn Pipeline for exercise difficulty prediction.

    The Pipeline must contain a StandardScaler and XGBClassifier,
    expecting 4 features: [mean_difficulty, max_difficulty,
    min_difficulty, situation_count].
    """

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    def predict(self, features: AggregatedFeatures) -> ExercisePrediction:
        """Predict exercise difficulty from aggregated features.

        Args:
            features: The 4 aggregated features for a single exercise.

        Returns:
            ExercisePrediction with predicted class and probabilities.

        Raises:
            PredictionError: If the model fails to produce a prediction.
        """
        try:
            X = np.array([features.to_feature_list()])
            probabilities = self._pipeline.predict_proba(X)[0]
            predicted_class_idx = int(np.argmax(probabilities))

            # Decode class index to ExerciseDifficulty enum
            difficulty = EXERCISE_DECODING.get(predicted_class_idx)
            if difficulty is None:
                raise PredictionError(
                    f"Unknown class index from model: {predicted_class_idx}"
                )

            # Build probability dict with class names
            prob_dict = {
                EXERCISE_DECODING[i].value: round(float(p), 4)
                for i, p in enumerate(probabilities)
                if i in EXERCISE_DECODING
            }

            return ExercisePrediction(
                difficulty=difficulty,
                probabilities=prob_dict,
                situation_count=features.situation_count,
            )

        except PredictionError:
            raise
        except Exception as e:
            raise PredictionError(f"Model inference failed: {e}") from e
