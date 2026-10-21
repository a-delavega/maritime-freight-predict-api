"""
Infrastructure: Model loader.

Loads the serialized sklearn Pipeline (StandardScaler + XGBClassifier)
from a joblib file.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


class ModelLoadError(Exception):
    """Raised when the model file cannot be loaded."""

    pass


def load_pipeline(path: str | Path) -> Pipeline:
    """Load a serialized sklearn Pipeline from a joblib file.

    Args:
        path: Path to the .joblib file containing the Pipeline.

    Returns:
        The deserialized sklearn Pipeline.

    Raises:
        ModelLoadError: If the file does not exist, is corrupted,
            or does not contain a Pipeline.
    """
    path = Path(path)

    if not path.exists():
        raise ModelLoadError(f"Model file not found: {path}")

    try:
        model = joblib.load(path)
    except Exception as e:
        raise ModelLoadError(f"Failed to load model from {path}: {e}") from e

    if not isinstance(model, Pipeline):
        raise ModelLoadError(
            f"Expected sklearn Pipeline, got {type(model).__name__}"
        )

    return model
