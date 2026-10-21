"""
Unit tests for Pydantic v2 schemas.

Tests P0: validation rules
Tests P1: edge cases
"""

from pydantic import ValidationError
import pytest

from predict.api.schemas import HealthResponse, PredictionRequest, PredictionResponse


# --- Valid Inputs ---


class TestPredictionRequestValid:
    """Tests for valid PredictionRequest payloads."""

    def test_valid_request_single_situation(self):
        """Single situation is valid."""
        req = PredictionRequest(situations=["MEDIUM"])
        assert len(req.situations) == 1
        assert req.situations[0].value == "MEDIUM"

    def test_valid_request_multiple_situations(self):
        """Multiple situations are valid."""
        req = PredictionRequest(
            situations=["LOW", "MEDIUM", "HIGH", "MEDIUM", "MEDIUM", "LOW"]
        )
        assert len(req.situations) == 6

    def test_valid_request_all_same(self):
        """All same difficulty is valid."""
        req = PredictionRequest(situations=["HIGH", "HIGH", "HIGH"])
        assert all(s.value == "HIGH" for s in req.situations)

    def test_valid_request_from_json(self):
        """Deserialization from JSON-like dict works."""
        req = PredictionRequest.model_validate(
            {"situations": ["LOW", "MEDIUM"]}
        )
        assert len(req.situations) == 2


# --- Invalid Inputs ---


class TestPredictionRequestInvalid:
    """Tests for invalid PredictionRequest payloads."""

    def test_empty_situations_rejected(self):
        """Empty list is rejected (min_length=1)."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(situations=[])
        assert "situations" in str(exc_info.value)

    def test_invalid_difficulty_value(self):
        """Unknown difficulty value is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(situations=["CRITICAL"])
        assert "situations" in str(exc_info.value)

    def test_missing_situations_field(self):
        """Missing required field is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest.model_validate({})
        assert "situations" in str(exc_info.value)

    def test_wrong_type_in_list(self):
        """Non-string values in list are rejected."""
        with pytest.raises(ValidationError):
            PredictionRequest(situations=[123])

    def test_none_value_rejected(self):
        """None as situations value is rejected."""
        with pytest.raises(ValidationError):
            PredictionRequest(situations=None)

    def test_string_instead_of_list(self):
        """A plain string instead of a list is rejected."""
        with pytest.raises(ValidationError):
            PredictionRequest(situations="MEDIUM")


# --- Edge Cases ---


class TestPredictionRequestEdgeCases:
    """Edge case tests for boundary conditions."""

    def test_max_situations_accepted(self):
        """Exactly 50 situations (max_length) is valid."""
        req = PredictionRequest(situations=["MEDIUM"] * 50)
        assert len(req.situations) == 50

    def test_exceeds_max_situations_rejected(self):
        """51 situations exceeds max_length."""
        with pytest.raises(ValidationError) as exc_info:
            PredictionRequest(situations=["MEDIUM"] * 51)
        assert "situations" in str(exc_info.value)

    def test_case_sensitive_rejection(self):
        """Lowercase 'medium' should be rejected (enum is case-sensitive)."""
        with pytest.raises(ValidationError):
            PredictionRequest(situations=["medium"])


# --- Response Schema ---


class TestPredictionResponse:
    """Tests for PredictionResponse serialization."""

    def test_valid_response(self):
        """Response serializes correctly."""
        resp = PredictionResponse(
            exercise_difficulty="MEDIUM",
            probabilities={"HIGH": 0.6, "LOW": 0.1, "MEDIUM": 0.3},
            situation_count=6,
            model_version="1.0.0",
        )
        data = resp.model_dump()
        assert data["exercise_difficulty"] == "MEDIUM"
        assert len(data["probabilities"]) == 3
        assert data["situation_count"] == 6

    def test_response_invalid_situation_count(self):
        """situation_count must be >= 1."""
        with pytest.raises(ValidationError):
            PredictionResponse(
                exercise_difficulty="MEDIUM",
                probabilities={"HIGH": 0.5, "LOW": 0.2, "MEDIUM": 0.3},
                situation_count=0,
                model_version="1.0.0",
            )


class TestHealthResponse:
    """Tests for HealthResponse."""

    def test_valid_health(self):
        resp = HealthResponse(status="healthy", model_loaded=True)
        assert resp.status == "healthy"
        assert resp.model_loaded is True
