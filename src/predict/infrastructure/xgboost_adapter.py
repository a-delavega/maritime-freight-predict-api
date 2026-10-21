"""
Infrastructure: XGBoost adapter.

Concrete implementation of DifficultyClassifierPort that uses
a securely loaded native XGBoost Booster and explicit standard scaler.
Zero joblib / pickle dependency.
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb

from predict.domain.entities import (
    AggregatedFeatures,
    ExerciseDifficulty,
    EXERCISE_DECODING,
    ExercisePrediction,
)
from predict.domain.ports import DifficultyClassifierPort
from predict.infrastructure.model_loader import LoadedModelAssets


class PredictionError(Exception):
    """Raised when model inference fails."""

    pass


class XGBoostAdapter(DifficultyClassifierPort):
    """Adapter wrapping a native XGBoost Booster and explicit ScalerParameters.

    Features expected: [mean_difficulty, max_difficulty, min_difficulty, situation_count].
    """

    def __init__(self, assets: LoadedModelAssets) -> None:
        self._booster = assets.booster
        self._scaler = assets.scaler

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
            # 1. Convert domain features to raw array
            raw_features = np.array([features.to_feature_list()], dtype=np.float32)

            # 2. Standard scale strictly without external pickle/pipeline dependencies
            scaled_features = self._scaler.transform(raw_features)

            # 3. Predict probabilities using native DMatrix
            dmatrix = xgb.DMatrix(scaled_features)
            raw_probs = self._booster.predict(dmatrix)[0]

            # In multi:softprob, raw_probs is an array of shape (num_classes,)
            probabilities = np.array(raw_probs, dtype=np.float32)
            predicted_class_idx = int(np.argmax(probabilities))

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
