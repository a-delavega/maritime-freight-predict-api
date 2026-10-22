"""
Additional integration tests for extended API routes, batch prediction,
metrics, and dashboard to ensure strict coverage > 90%.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from predict.api.main import app
from predict.infrastructure.model_loader import load_secure_model
from predict.infrastructure.xgboost_adapter import XGBoostAdapter
from predict.application.use_cases import ClassifyExerciseDifficultyUseCase


@pytest.fixture
def live_client(models_dir: Path):
    """TestClient that runs full FastAPI application with real model loaded."""
    if not (models_dir / "model.json").exists():
        pytest.skip("Model assets not available")
    assets = load_secure_model(models_dir)
    app.state.use_case = ClassifyExerciseDifficultyUseCase(XGBoostAdapter(assets))
    with TestClient(app) as client:
        yield client


class TestExtendedEndpoints:
    """Tests for batch prediction, metrics, dashboard, playground, and custom docs."""

    def test_predict_batch_success(self, live_client: TestClient):
        payload = {
            "exercises": {
                "ex1": ["LOW", "LOW", "MEDIUM"],
                "ex2": ["HIGH", "HIGH", "HIGH"],
            }
        }
        res = live_client.post("/predict/batch", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["total_exercises"] == 2
        assert "ex1" in data["predictions"]
        assert "ex2" in data["predictions"]
        assert data["predictions"]["ex1"]["exercise_difficulty"] in ("LOW", "MEDIUM", "HIGH")

    def test_predict_batch_empty_situations(self, live_client: TestClient):
        payload = {"exercises": {"empty_ex": []}}
        res = live_client.post("/predict/batch", json=payload)
        assert res.status_code == 422

    def test_predict_batch_invalid_value(self, live_client: TestClient):
        payload = {"exercises": {"bad_ex": ["INVALID_VAL"]}}
        res = live_client.post("/predict/batch", json=payload)
        assert res.status_code == 422

    def test_metrics_endpoint(self, live_client: TestClient):
        res = live_client.get("/metrics")
        assert res.status_code == 200
        assert "model_predictions_total" in res.text

    def test_dashboard_endpoint(self, live_client: TestClient):
        res = live_client.get("/dashboard")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        assert "Maritime Inference Telemetry" in res.text

    def test_playground_endpoint(self, live_client: TestClient):
        res = live_client.get("/playground")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        assert "Maritime AI - Interactive Prediction Studio" in res.text

    def test_custom_swagger_ui(self, live_client: TestClient):
        res = live_client.get("/docs")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
