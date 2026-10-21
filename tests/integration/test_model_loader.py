"""
Integration tests for model_loader and XGBoostAdapter.
"""

from pathlib import Path
import pytest

from predict.domain.entities import AggregatedFeatures
from predict.infrastructure.model_loader import load_pipeline, ModelLoadError
from predict.infrastructure.xgboost_adapter import XGBoostAdapter


# --- Model Loading Errors ---


class TestModelLoader:
    """Tests for the model loading infrastructure."""

    def test_model_file_not_found(self, tmp_path: Path):
        """Non-existent path raises ModelLoadError."""
        fake_path = tmp_path / "nonexistent.joblib"
        with pytest.raises(ModelLoadError, match="not found"):
            load_pipeline(fake_path)

    def test_corrupted_model_file(self, tmp_path: Path):
        """Non-joblib file raises ModelLoadError."""
        bad_file = tmp_path / "corrupted.joblib"
        bad_file.write_text("this is not a joblib file")
        with pytest.raises(ModelLoadError):
            load_pipeline(bad_file)

    def test_wrong_type_raises_error(self, tmp_path: Path):
        """A joblib file containing a non-Pipeline object raises error."""
        import joblib

        wrong_obj = {"not": "a pipeline"}
        path = tmp_path / "wrong_type.joblib"
        joblib.dump(wrong_obj, path)
        with pytest.raises(ModelLoadError, match="Expected sklearn Pipeline"):
            load_pipeline(path)

    def test_loads_real_model_if_exists(self, models_dir: Path):
        """If the real model exists, it loads successfully."""
        model_path = models_dir / "model_pipeline.joblib"
        if not model_path.exists():
            pytest.skip("Real model not yet serialized")
        pipeline = load_pipeline(model_path)
        assert hasattr(pipeline, "predict")
        assert hasattr(pipeline, "predict_proba")


# --- XGBoostAdapter with real model ---


class TestXGBoostAdapterIntegration:
    """Integration tests using the real serialized model."""

    @pytest.fixture
    def adapter(self, models_dir: Path) -> XGBoostAdapter:
        model_path = models_dir / "model_pipeline.joblib"
        if not model_path.exists():
            pytest.skip("Real model not yet serialized")
        pipeline = load_pipeline(model_path)
        return XGBoostAdapter(pipeline)

    def test_predict_returns_valid_result(self, adapter: XGBoostAdapter):
        """Adapter produces a valid ExercisePrediction."""
        features = AggregatedFeatures(
            mean_difficulty=1.0,
            max_difficulty=2.0,
            min_difficulty=0.0,
            situation_count=6,
        )
        result = adapter.predict(features)
        assert result.difficulty.value in ("HIGH", "LOW", "MEDIUM")
        assert len(result.probabilities) == 3
        assert result.situation_count == 6

    def test_probabilities_sum_to_one(self, adapter: XGBoostAdapter):
        """Probabilities from the real model sum to ~1.0."""
        features = AggregatedFeatures(
            mean_difficulty=1.5,
            max_difficulty=2.0,
            min_difficulty=1.0,
            situation_count=10,
        )
        result = adapter.predict(features)
        total = sum(result.probabilities.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_all_high_features(self, adapter: XGBoostAdapter):
        """All HIGH encoded (3,3,3) produces a valid prediction."""
        features = AggregatedFeatures(
            mean_difficulty=3.0,
            max_difficulty=3.0,
            min_difficulty=3.0,
            situation_count=3,
        )
        result = adapter.predict(features)
        assert result.difficulty.value in ("HIGH", "LOW", "MEDIUM")
