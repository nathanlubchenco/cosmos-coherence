"""Tests for SimpleQA AI-based grading implementation."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from cosmos_coherence.benchmarks.implementations.simpleqa_grader import SimpleQAGrader
from cosmos_coherence.llm.models import ModelResponse, TokenUsage


class TestSimpleQAGrader:
    """Test AI-based grading for SimpleQA benchmark."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock OpenAI client."""
        client = MagicMock()
        client.generate_response = AsyncMock()
        return client

    @pytest.fixture
    def grader(self, mock_client):
        """Create a SimpleQA grader instance."""
        return SimpleQAGrader(mock_client)

    @pytest.mark.asyncio
    async def test_grade_correct_response(self, grader, mock_client):
        """Test grading a correct response."""
        # Mock the grader response
        mock_client.generate_response.return_value = ModelResponse(
            content="CORRECT",
            model="gpt-4o-mini",
            usage=TokenUsage(
                prompt_tokens=100, completion_tokens=1, total_tokens=101, estimated_cost=0.001
            ),
            request_id="test-id",
            latency_ms=100.0,
            temperature=0.0,
            finish_reason="stop",
            cached=False,
        )

        grade, metadata = await grader.grade_response(
            question="What is the capital of France?",
            expert_answer="Paris",
            submission="Paris",
        )

        assert grade == "CORRECT"
        assert metadata["normalized_grade"] == "CORRECT"
        assert metadata["grader_model"] == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_grade_incorrect_response(self, grader, mock_client):
        """Test grading an incorrect response."""
        mock_client.generate_response.return_value = ModelResponse(
            content="INCORRECT",
            model="gpt-4o-mini",
            usage=TokenUsage(
                prompt_tokens=100, completion_tokens=1, total_tokens=101, estimated_cost=0.001
            ),
            request_id="test-id",
            latency_ms=100.0,
            temperature=0.0,
            finish_reason="stop",
            cached=False,
        )

        grade, metadata = await grader.grade_response(
            question="What is the capital of France?",
            expert_answer="Paris",
            submission="London",
        )

        assert grade == "INCORRECT"
        assert metadata["normalized_grade"] == "INCORRECT"

    @pytest.mark.asyncio
    async def test_grade_not_attempted_response(self, grader, mock_client):
        """Test grading a not attempted response."""
        mock_client.generate_response.return_value = ModelResponse(
            content="NOT_ATTEMPTED",
            model="gpt-4o-mini",
            usage=TokenUsage(
                prompt_tokens=100, completion_tokens=1, total_tokens=101, estimated_cost=0.001
            ),
            request_id="test-id",
            latency_ms=100.0,
            temperature=0.0,
            finish_reason="stop",
            cached=False,
        )

        grade, metadata = await grader.grade_response(
            question="What is the capital of France?",
            expert_answer="Paris",
            submission="I don't know",
        )

        assert grade == "NOT_ATTEMPTED"
        assert metadata["normalized_grade"] == "NOT_ATTEMPTED"

    @pytest.mark.asyncio
    async def test_grade_with_extra_text(self, grader, mock_client):
        """Test grading when response contains extra text."""
        mock_client.generate_response.return_value = ModelResponse(
            content="The answer is CORRECT.",
            model="gpt-4o-mini",
            usage=TokenUsage(
                prompt_tokens=100, completion_tokens=5, total_tokens=105, estimated_cost=0.001
            ),
            request_id="test-id",
            latency_ms=100.0,
            temperature=0.0,
            finish_reason="stop",
            cached=False,
        )

        grade, metadata = await grader.grade_response(
            question="What is 2+2?",
            expert_answer="4",
            submission="4",
        )

        assert grade == "CORRECT"
        assert "CORRECT" in metadata["raw_grade"]

    @pytest.mark.asyncio
    async def test_grade_unparseable_response(self, grader, mock_client):
        """Test grading with unparseable response defaults to INCORRECT."""
        mock_client.generate_response.return_value = ModelResponse(
            content="Unable to determine",
            model="gpt-4o-mini",
            usage=TokenUsage(
                prompt_tokens=100, completion_tokens=3, total_tokens=103, estimated_cost=0.001
            ),
            request_id="test-id",
            latency_ms=100.0,
            temperature=0.0,
            finish_reason="stop",
            cached=False,
        )

        grade, metadata = await grader.grade_response(
            question="What is the capital of France?",
            expert_answer="Paris",
            submission="Maybe Paris?",
        )

        assert grade == "INCORRECT"
        assert metadata["normalized_grade"] == "INCORRECT"

    def test_calculate_metrics_all_correct(self):
        """Test metrics calculation with all correct answers."""
        grades = ["CORRECT", "CORRECT", "CORRECT"]
        metrics = SimpleQAGrader.calculate_metrics(grades)

        assert metrics["accuracy"] == 1.0
        assert metrics["accuracy_given_attempted"] == 1.0
        assert metrics["correct_percentage"] == 100.0
        assert metrics["incorrect_percentage"] == 0.0
        assert metrics["not_attempted_percentage"] == 0.0
        assert metrics["total_questions"] == 3
        assert metrics["correct_count"] == 3

    def test_calculate_metrics_mixed(self):
        """Test metrics calculation with mixed results."""
        grades = ["CORRECT", "INCORRECT", "NOT_ATTEMPTED", "CORRECT", "INCORRECT"]
        metrics = SimpleQAGrader.calculate_metrics(grades)

        assert metrics["accuracy"] == 0.4  # 2/5
        assert metrics["accuracy_given_attempted"] == 0.5  # 2/4
        assert metrics["correct_percentage"] == 40.0
        assert metrics["incorrect_percentage"] == 40.0
        assert metrics["not_attempted_percentage"] == 20.0
        assert metrics["total_questions"] == 5
        assert metrics["correct_count"] == 2
        assert metrics["incorrect_count"] == 2
        assert metrics["not_attempted_count"] == 1

    def test_calculate_metrics_all_not_attempted(self):
        """Test metrics calculation with all not attempted."""
        grades = ["NOT_ATTEMPTED", "NOT_ATTEMPTED"]
        metrics = SimpleQAGrader.calculate_metrics(grades)

        assert metrics["accuracy"] == 0.0
        assert metrics["accuracy_given_attempted"] == 0.0  # No attempts
        assert metrics["correct_percentage"] == 0.0
        assert metrics["incorrect_percentage"] == 0.0
        assert metrics["not_attempted_percentage"] == 100.0
        assert metrics["total_questions"] == 2

    def test_calculate_metrics_empty_list(self):
        """Test metrics calculation with empty list."""
        grades = []
        metrics = SimpleQAGrader.calculate_metrics(grades)

        assert metrics["accuracy"] == 0.0
        assert metrics["accuracy_given_attempted"] == 0.0
        assert metrics["correct_percentage"] == 0.0
        assert metrics["incorrect_percentage"] == 0.0
        assert metrics["not_attempted_percentage"] == 0.0
