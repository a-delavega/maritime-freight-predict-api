"""
Integration tests for model_loader and XGBoostAdapter using secure native JSON assets.
"""

import json
from pathlib import Path
import pytest
import numpy as np

from predict.domain.entities import AggregatedFeatures
from predict.infrastructure.model_loader import (
    LoadedModelAssets,
    ModelLoadError,
    ScalerParameters,
    load_secure_model,
)
from predict.infrastructure.xgboost_adapter import XGBoostAdapter


# --- Model Loading Errors ---


class TestModelLoader:
    """Tests for the secure JSON model loading infrastructure."""

    def test_model_file_not_found(self, tmp_path: Path):
        """Non-existent directory raises ModelLoadError."""
        fake_dir = tmp_path / "nonexistent"
        with pytest.raises(ModelLoadError, match="not found"):
            load_secure_model(fake_dir)

    def test_corrupted_model_file(self, tmp_path: Path):
        """Invalid JSON / corrupt model file raises ModelLoadError."""
        bad_dir = tmp_path / "corrupt_dir"
        bad_dir.mkdir()
        bad_model = bad_dir / "model.json"
        bad_model.write_text("not a valid json booster")
        scaler_file = bad_dir / "scaler_params.json"
        scaler_file.write_text(json.dumps({"mean": [1, 2, 3, 4], "scale": [1, 1, 1, 1]}))

        with pytest.raises(ModelLoadError):
            load_secure_model(bad_dir)

    def test_missing_scaler_file(self, tmp_path: Path):
        """Directory with model.json but missing scaler_params.json raises error."""
        test_dir = tmp_path / "missing_scaler"
        test_dir.mkdir()
        (test_dir / "model.json").write_text("{}")
        with pytest.raises(ModelLoadError, match="Scaler parameters file not found"):
            load_secure_model(test_dir)

    def test_loads_real_model_if_exists(self, models_dir: Path):
        """If the real native JSON assets exist, they load successfully."""
        model_path = models_dir / "model.json"
        if not model_path.exists():
            pytest.skip("Real native model.json not found")
        assets = load_secure_model(models_dir)
        assert hasattr(assets.booster, "predict")
        assert len(assets.scaler.mean) == 4
        assert len(assets.scaler.scale) == 4


# --- XGBoostAdapter with real native model ---


class TestXGBoostAdapterIntegration:
    """Integration tests using the native XGBoost model."""

    @pytest.fixture
    def adapter(self, models_dir: Path) -> XGBoostAdapter:
        model_path = models_dir / "model.json"
        if not model_path.exists():
            pytest.skip("Real native model.json not found")
        assets = load_secure_model(models_dir)
        return XGBoostAdapter(assets)

    def test_predict_returns_valid_result(self, adapter: XGBoostAdapter):
        """Adapter produces a valid ExercisePrediction."""
        features = AggregatedFeatures(
            mean_difficulty=2.0,
            max_difficulty=3.0,
            min_difficulty=1.0,
            situation_count=6,
        )
        result = adapter.predict(features)
        assert result.difficulty.value in ("HIGH", "LOW", "MEDIUM")
        assert len(result.probabilities) == 3
        assert result.situation_count == 6

    def test_probabilities_sum_to_one(self, adapter: XGBoostAdapter):
        """Probabilities from the native model sum to ~1.0."""
        features = AggregatedFeatures(
            mean_difficulty=2.5,
            max_difficulty=3.0,
            min_difficulty=2.0,
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
            situation_count=4,
        )
        result = adapter.predict(features)
        assert result.difficulty.value in ("HIGH", "LOW", "MEDIUM")
