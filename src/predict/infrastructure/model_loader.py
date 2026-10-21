"""
Infrastructure: Secure Model and Scaler Loader.

Loads the native XGBoost model from a secure JSON file and
standard scaling parameters from JSON. Zero pickle / joblib execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import xgboost as xgb


class ModelLoadError(Exception):
    """Raised when the model or scaler configuration cannot be safely loaded."""

    pass


@dataclass(frozen=True)
class ScalerParameters:
    """Explicit parameters for standard feature scaling: (x - mean) / scale."""

    mean: np.ndarray
    scale: np.ndarray
    feature_names: list[str]

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply standard scaling: (X - mean) / scale."""
        return (x - self.mean) / self.scale


@dataclass(frozen=True)
class LoadedModelAssets:
    """Encapsulates the secure native XGBoost Booster and explicit Scaler."""

    booster: xgb.Booster
    scaler: ScalerParameters
    metadata: dict


def load_secure_model(models_dir: str | Path) -> LoadedModelAssets:
    """Load native XGBoost model.json, scaler_params.json, and metadata.json.

    Args:
        models_dir: Path to directory containing the JSON artifacts.

    Returns:
        LoadedModelAssets instance ready for high-speed, secure CPU inference.

    Raises:
        ModelLoadError: If required JSON files are missing or unparseable.
    """
    models_dir = Path(models_dir)

    model_path = models_dir / "model.json"
    scaler_path = models_dir / "scaler_params.json"
    metadata_path = models_dir / "metadata.json"

    if not model_path.exists():
        raise ModelLoadError(f"Native model file not found: {model_path}")
    if not scaler_path.exists():
        raise ModelLoadError(f"Scaler parameters file not found: {scaler_path}")

    # 1. Load native XGBoost Booster from JSON
    try:
        booster = xgb.Booster()
        booster.load_model(str(model_path))
    except Exception as e:
        raise ModelLoadError(f"Failed to parse native XGBoost model at {model_path}: {e}") from e

    # 2. Load Scaler parameters from JSON
    try:
        with open(scaler_path, "r", encoding="utf-8") as f:
            scaler_data = json.load(f)
        scaler = ScalerParameters(
            mean=np.array(scaler_data["mean"], dtype=np.float32),
            scale=np.array(scaler_data["scale"], dtype=np.float32),
            feature_names=scaler_data.get("features", []),
        )
    except Exception as e:
        raise ModelLoadError(f"Failed to parse scaler parameters at {scaler_path}: {e}") from e

    # 3. Load Metadata (optional or defaults)
    metadata: dict = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    return LoadedModelAssets(booster=booster, scaler=scaler, metadata=metadata)
