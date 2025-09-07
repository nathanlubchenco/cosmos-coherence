"""Tests for OpenAI client models."""

from datetime import datetime

import pytest
from cosmos_coherence.llm.models import (
    BatchError,
    BatchJob,
    BatchJobStatus,
    BatchRequest,
    ModelResponse,
    TokenUsage,
)
from pydantic import ValidationError


class TestTokenUsage:
    """Test TokenUsage model."""

    def test_token_usage_creation(self):
        """Test creating TokenUsage with valid data."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            estimated_cost=0.0025,
        )
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.estimated_cost == 0.0025

    def test_token_usage_total_calculation(self):
        """Test that total tokens must equal sum of prompt and completion."""
        with pytest.raises(ValidationError):
            TokenUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=200,  # Should be 150
                estimated_cost=0.0025,
            )

    def test_token_usage_negative_values(self):
        """Test that token counts cannot be negative."""
        with pytest.raises(ValidationError):
            TokenUsage(
                prompt_tokens=-10,
                completion_tokens=50,
                total_tokens=40,
                estimated_cost=0.001,
            )


class TestModelResponse:
    """Test ModelResponse model."""

    def test_model_response_creation(self):
        """Test creating ModelResponse with valid data."""
        usage = TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            estimated_cost=0.0001,
        )
        response = ModelResponse(
            content="Test response",
            usage=usage,
            model="gpt-3.5-turbo",
            request_id="req-123",
            latency_ms=150.5,
            temperature=0.7,
            finish_reason="stop",
        )
        assert response.content == "Test response"
        assert response.usage == usage
        assert response.model == "gpt-3.5-turbo"
        assert response.request_id == "req-123"
        assert response.latency_ms == 150.5
        assert response.temperature == 0.7
        assert response.finish_reason == "stop"
        assert response.cached is False  # Default value

    def test_model_response_with_cache(self):
        """Test ModelResponse with cached flag."""
        usage = TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            estimated_cost=0.0,  # Cached responses may have 0 cost
        )
        response = ModelResponse(
            content="Cached response",
            usage=usage,
            model="gpt-3.5-turbo",
            request_id="req-456",
            latency_ms=10.0,
            temperature=0.0,
            finish_reason="stop",
            cached=True,
        )
        assert response.cached is True
        assert response.latency_ms == 10.0  # Faster for cached

    def test_model_response_finish_reasons(self):
        """Test different finish reasons."""
        usage = TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            estimated_cost=0.0001,
        )

        # Valid finish reasons
        for reason in ["stop", "length", "content_filter", "function_call"]:
            response = ModelResponse(
                content="Test",
                usage=usage,
                model="gpt-3.5-turbo",
                request_id="req-test",
                latency_ms=100.0,
                temperature=0.5,
                finish_reason=reason,
            )
            assert response.finish_reason == reason


class TestBatchRequest:
    """Test BatchRequest model."""

    def test_batch_request_minimal(self):
        """Test creating BatchRequest with minimal data."""
        request = BatchRequest(
            prompt="Test prompt",
            temperature=0.7,
        )
        assert request.prompt == "Test prompt"
        assert request.temperature == 0.7
        assert request.model is None
        assert request.max_tokens is None
        assert request.custom_id is not None  # Should be auto-generated

    def test_batch_request_full(self):
        """Test creating BatchRequest with all fields."""
        request = BatchRequest(
            prompt="Test prompt",
            temperature=0.5,
            model="gpt-4",
            max_tokens=100,
            custom_id="custom-123",
        )
        assert request.prompt == "Test prompt"
        assert request.temperature == 0.5
        assert request.model == "gpt-4"
        assert request.max_tokens == 100
        assert request.custom_id == "custom-123"

    def test_batch_request_auto_id(self):
        """Test that custom_id is auto-generated if not provided."""
        request1 = BatchRequest(prompt="Test 1", temperature=0.7)
        request2 = BatchRequest(prompt="Test 2", temperature=0.7)

        assert request1.custom_id is not None
        assert request2.custom_id is not None
        assert request1.custom_id != request2.custom_id  # Should be unique

    def test_batch_request_temperature_validation(self):
        """Test temperature validation."""
        with pytest.raises(ValidationError):
            BatchRequest(prompt="Test", temperature=-0.1)

        with pytest.raises(ValidationError):
            BatchRequest(prompt="Test", temperature=2.1)

        # Valid edge cases
        request = BatchRequest(prompt="Test", temperature=0.0)
        assert request.temperature == 0.0
        request = BatchRequest(prompt="Test", temperature=2.0)
        assert request.temperature == 2.0


class TestBatchJob:
    """Test BatchJob model."""

    def test_batch_job_creation(self):
        """Test creating BatchJob with valid data."""
        now = datetime.now()
        job = BatchJob(
            job_id="batch-123",
            status="validating",
            created_at=now,
            request_count=100,
        )
        assert job.job_id == "batch-123"
        assert job.status == "validating"
        assert job.created_at == now
        assert job.request_count == 100

    def test_batch_job_status_values(self):
        """Test valid status values for BatchJob."""
        now = datetime.now()
        valid_statuses = ["validating", "in_progress", "completed", "failed", "expired"]

        for status in valid_statuses:
            job = BatchJob(
                job_id="batch-test",
                status=status,
                created_at=now,
                request_count=10,
            )
            assert job.status == status


class TestBatchJobStatus:
    """Test BatchJobStatus model."""

    def test_batch_job_status_minimal(self):
        """Test creating BatchJobStatus with minimal data."""
        now = datetime.now()
        status = BatchJobStatus(
            job_id="batch-123",
            status="in_progress",
            created_at=now,
            completed_at=None,
            request_count=100,
            completed_count=50,
            failed_count=0,
            errors=[],
        )
        assert status.job_id == "batch-123"
        assert status.status == "in_progress"
        assert status.created_at == now
        assert status.completed_at is None
        assert status.request_count == 100
        assert status.completed_count == 50
        assert status.failed_count == 0
        assert len(status.errors) == 0

    def test_batch_job_status_completed(self):
        """Test BatchJobStatus for completed job."""
        created = datetime.now()
        completed = datetime.now()
        status = BatchJobStatus(
            job_id="batch-456",
            status="completed",
            created_at=created,
            completed_at=completed,
            request_count=100,
            completed_count=98,
            failed_count=2,
            errors=[
                BatchError(
                    code="rate_limit_exceeded",
                    message="Rate limit hit",
                    request_id="req-fail-1",
                ),
                BatchError(
                    code="invalid_request",
                    message="Invalid prompt",
                    request_id="req-fail-2",
                ),
            ],
        )
        assert status.status == "completed"
        assert status.completed_at == completed
        assert status.completed_count == 98
        assert status.failed_count == 2
        assert len(status.errors) == 2

    def test_batch_job_status_validation(self):
        """Test BatchJobStatus validation rules."""
        now = datetime.now()

        # Completed count cannot exceed request count
        with pytest.raises(ValidationError):
            BatchJobStatus(
                job_id="batch-bad",
                status="in_progress",
                created_at=now,
                completed_at=None,
                request_count=100,
                completed_count=101,  # Exceeds total
                failed_count=0,
                errors=[],
            )

    def test_batch_job_status_literal(self):
        """Test that status must be a valid literal value."""
        now = datetime.now()

        with pytest.raises(ValidationError):
            BatchJobStatus(
                job_id="batch-bad",
                status="invalid_status",  # Not in literal values
                created_at=now,
                completed_at=None,
                request_count=100,
                completed_count=0,
                failed_count=0,
                errors=[],
            )


class TestBatchError:
    """Test BatchError model."""

    def test_batch_error_creation(self):
        """Test creating BatchError with valid data."""
        error = BatchError(
            code="rate_limit_exceeded",
            message="Too many requests",
            request_id="req-123",
        )
        assert error.code == "rate_limit_exceeded"
        assert error.message == "Too many requests"
        assert error.request_id == "req-123"

    def test_batch_error_optional_request_id(self):
        """Test BatchError with optional request_id."""
        error = BatchError(
            code="batch_validation_failed",
            message="Invalid batch format",
            request_id=None,
        )
        assert error.code == "batch_validation_failed"
        assert error.message == "Invalid batch format"
        assert error.request_id is None
