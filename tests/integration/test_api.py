"""
Integration tests for the FastAPI application.
"""

from contextlib import asynccontextmanager
from unittest.mock import MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from predict.api.routes import router
from predict.application.use_cases import ClassifyExerciseDifficultyUseCase
from predict.domain.entities import ExerciseDifficulty, ExercisePrediction
from predict.domain.ports import DifficultyClassifierPort


def _create_app_with_use_case(
    use_case: ClassifyExerciseDifficultyUseCase | None,
) -> FastAPI:
    """Create a FastAPI app with a specific use case (or None for degraded)."""

    @asynccontextmanager
    async def test_lifespan(app: FastAPI):
        app.state.use_case = use_case
        yield

    test_app = FastAPI(lifespan=test_lifespan)
    test_app.include_router(router)
    return test_app


@pytest.fixture
def mock_use_case() -> ClassifyExerciseDifficultyUseCase:
    """Create a use case with a mock classifier for API testing."""
    mock_classifier = MagicMock(spec=DifficultyClassifierPort)
    mock_classifier.predict.return_value = ExercisePrediction(
        difficulty=ExerciseDifficulty.MEDIUM,
        probabilities={"HIGH": 0.23, "LOW": 0.12, "MEDIUM": 0.65},
        situation_count=6,
    )
    return ClassifyExerciseDifficultyUseCase(mock_classifier)


@pytest.fixture
def client_with_model(mock_use_case: ClassifyExerciseDifficultyUseCase):
    """TestClient with a mock model loaded (bypasses real lifespan)."""
    test_app = _create_app_with_use_case(mock_use_case)
    with TestClient(test_app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def client_without_model():
    """TestClient with no model loaded (degraded state)."""
    test_app = _create_app_with_use_case(None)
    with TestClient(test_app, raise_server_exceptions=False) as client:
        yield client


# --- API Endpoints ---


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_endpoint_healthy(self, client_with_model: TestClient):
        """Health check returns 200 when model is loaded."""
        response = client_with_model.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_health_endpoint_degraded(self, client_without_model: TestClient):
        """Health check returns 200 but degraded when model is not loaded."""
        response = client_without_model.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_loaded"] is False


class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_predict_endpoint_200(self, client_with_model: TestClient):
        """Valid payload returns 200 with prediction."""
        payload = {
            "situations": ["LOW", "MEDIUM", "HIGH", "MEDIUM", "MEDIUM", "LOW"]
        }
        response = client_with_model.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["exercise_difficulty"] in ("HIGH", "LOW", "MEDIUM")
        assert "probabilities" in data
        assert len(data["probabilities"]) == 3
        assert data["situation_count"] == 6
        assert data["model_version"] == "1.0.0"

    def test_predict_endpoint_422_missing_field(
        self, client_with_model: TestClient
    ):
        """Missing required field returns 422."""
        response = client_with_model.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_endpoint_422_invalid_value(
        self, client_with_model: TestClient
    ):
        """Invalid difficulty value returns 422."""
        payload = {"situations": ["INVALID"]}
        response = client_with_model.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_endpoint_422_empty_list(
        self, client_with_model: TestClient
    ):
        """Empty situations list returns 422."""
        payload = {"situations": []}
        response = client_with_model.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_endpoint_422_wrong_type(
        self, client_with_model: TestClient
    ):
        """Non-list value returns 422."""
        payload = {"situations": "MEDIUM"}
        response = client_with_model.post("/predict", json=payload)
        assert response.status_code == 422

    def test_predict_endpoint_single_situation(
        self, client_with_model: TestClient
    ):
        """Single situation prediction works and returns 200."""
        payload = {"situations": ["HIGH"]}
        response = client_with_model.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["exercise_difficulty"] in ("HIGH", "LOW", "MEDIUM")
        assert "model_version" in data


# --- System Failures ---


class TestPredictEndpointFailures:
    """Tests for system failure scenarios."""

    def test_predict_when_model_not_loaded(
        self, client_without_model: TestClient
    ):
        """Model not loaded returns 503."""
        payload = {"situations": ["MEDIUM", "HIGH"]}
        response = client_without_model.post("/predict", json=payload)
        assert response.status_code == 503
        assert "not loaded" in response.json()["detail"].lower()
