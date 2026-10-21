"""
Unit tests for ClassifyExerciseDifficultyUseCase.
"""

from unittest.mock import MagicMock
import pytest

from predict.application.use_cases import ClassifyExerciseDifficultyUseCase
from predict.domain.entities import (
    AggregatedFeatures,
    ExerciseDifficulty,
    ExercisePrediction,
    SituationDifficulty,
)
from predict.domain.ports import DifficultyClassifierPort


@pytest.fixture
def mock_classifier() -> MagicMock:
    """Create a mock classifier that returns a fixed prediction."""
    mock = MagicMock(spec=DifficultyClassifierPort)
    mock.predict.return_value = ExercisePrediction(
        difficulty=ExerciseDifficulty.MEDIUM,
        probabilities={"HIGH": 0.2, "LOW": 0.1, "MEDIUM": 0.7},
        situation_count=3,
    )
    return mock


@pytest.fixture
def use_case(mock_classifier: MagicMock) -> ClassifyExerciseDifficultyUseCase:
    """Create a use case with a mocked classifier."""
    return ClassifyExerciseDifficultyUseCase(mock_classifier)


# --- Happy Path ---


class TestUseCaseHappyPath:
    """Tests for the main prediction flow."""

    def test_use_case_happy_path(
        self, use_case: ClassifyExerciseDifficultyUseCase, mock_classifier: MagicMock
    ):
        """Use case orchestrates correctly with mock port."""
        situations = [
            SituationDifficulty.LOW,
            SituationDifficulty.MEDIUM,
            SituationDifficulty.HIGH,
        ]

        result = use_case.execute(situations)

        mock_classifier.predict.assert_called_once()
        assert result.difficulty == ExerciseDifficulty.MEDIUM
        assert "HIGH" in result.probabilities
        assert "LOW" in result.probabilities
        assert "MEDIUM" in result.probabilities

    def test_use_case_encoding_correct(
        self, use_case: ClassifyExerciseDifficultyUseCase, mock_classifier: MagicMock
    ):
        """Verify semantic ordinal encoding LOW→1, MEDIUM→2, HIGH→3 and aggregation."""
        situations = [
            SituationDifficulty.LOW,     # encoded: 1
            SituationDifficulty.MEDIUM,  # encoded: 2
            SituationDifficulty.HIGH,    # encoded: 3
        ]

        use_case.execute(situations)

        call_args = mock_classifier.predict.call_args
        features: AggregatedFeatures = call_args[0][0]

        # Encoded values: [1, 2, 3] -> sum=6, mean=2.0, max=3, min=1, count=3
        assert features.mean_difficulty == pytest.approx(2.0)
        assert features.max_difficulty == 3.0
        assert features.min_difficulty == 1.0
        assert features.situation_count == 3

    def test_use_case_passes_correct_features_shape(
        self, use_case: ClassifyExerciseDifficultyUseCase, mock_classifier: MagicMock
    ):
        """Features passed to classifier have all 4 required fields."""
        situations = [SituationDifficulty.MEDIUM, SituationDifficulty.MEDIUM]
        use_case.execute(situations)

        features: AggregatedFeatures = mock_classifier.predict.call_args[0][0]
        feature_list = features.to_feature_list()
        assert len(feature_list) == 4
        assert all(isinstance(f, float) for f in feature_list)


# --- Edge Cases ---


class TestUseCaseEdgeCases:
    """Edge case tests for boundary conditions."""

    def test_all_same_difficulty(
        self, use_case: ClassifyExerciseDifficultyUseCase, mock_classifier: MagicMock
    ):
        """All HIGH (3) → mean=max=min=3.0."""
        situations = [SituationDifficulty.HIGH] * 5

        use_case.execute(situations)

        features: AggregatedFeatures = mock_classifier.predict.call_args[0][0]
        assert features.mean_difficulty == 3.0
        assert features.max_difficulty == 3.0
        assert features.min_difficulty == 3.0
        assert features.situation_count == 5

    def test_single_situation_features(
        self, use_case: ClassifyExerciseDifficultyUseCase, mock_classifier: MagicMock
    ):
        """Single situation → mean=max=min, count=1."""
        situations = [SituationDifficulty.MEDIUM]

        use_case.execute(situations)

        features: AggregatedFeatures = mock_classifier.predict.call_args[0][0]
        assert features.mean_difficulty == 2.0
        assert features.max_difficulty == 2.0
        assert features.min_difficulty == 2.0
        assert features.situation_count == 1

    def test_empty_situations_raises(
        self, use_case: ClassifyExerciseDifficultyUseCase
    ):
        """Empty list raises ValueError."""
        with pytest.raises(ValueError, match="At least one situation"):
            use_case.execute([])

    def test_prediction_probabilities_sum_to_one(
        self, use_case: ClassifyExerciseDifficultyUseCase, mock_classifier: MagicMock
    ):
        """Probabilities from mock sum to ~1.0."""
        situations = [SituationDifficulty.MEDIUM, SituationDifficulty.LOW]
        result = use_case.execute(situations)
        total = sum(result.probabilities.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_response_contains_all_classes(
        self, use_case: ClassifyExerciseDifficultyUseCase, mock_classifier: MagicMock
    ):
        """Response contains probabilities for all 3 classes."""
        situations = [SituationDifficulty.HIGH]
        result = use_case.execute(situations)
        assert set(result.probabilities.keys()) == {"HIGH", "LOW", "MEDIUM"}


# --- Domain Entities ---


class TestAggregatedFeatures:
    """Tests for AggregatedFeatures value object."""

    def test_to_feature_list_order(self):
        """Feature list follows the training column order."""
        features = AggregatedFeatures(
            mean_difficulty=1.5,
            max_difficulty=2.0,
            min_difficulty=0.0,
            situation_count=10,
        )
        result = features.to_feature_list()
        assert result == [1.5, 2.0, 0.0, 10.0]

    def test_frozen_dataclass(self):
        """AggregatedFeatures is immutable."""
        features = AggregatedFeatures(
            mean_difficulty=1.0,
            max_difficulty=2.0,
            min_difficulty=0.0,
            situation_count=5,
        )
        with pytest.raises(AttributeError):
            features.mean_difficulty = 999  # type: ignore[misc]


class TestSituationDifficultyEncoding:
    """Tests for the domain encoding of situation difficulty."""

    def test_low_encodes_to_1(self):
        assert SituationDifficulty.LOW.encoded_value == 1

    def test_medium_encodes_to_2(self):
        assert SituationDifficulty.MEDIUM.encoded_value == 2

    def test_high_encodes_to_3(self):
        assert SituationDifficulty.HIGH.encoded_value == 3
