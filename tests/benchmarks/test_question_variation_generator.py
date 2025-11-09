"""Tests for QuestionVariationGenerator."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from cosmos_coherence.benchmarks.implementations.question_variation_generator import (
    QuestionVariationGenerator,
)
from cosmos_coherence.llm.models import ModelResponse, TokenUsage


@pytest.fixture
def mock_client():
    """Create a mock OpenAI client."""
    client = MagicMock()
    client.generate_response = AsyncMock()
    return client


@pytest.fixture
def generator(mock_client):
    """Create QuestionVariationGenerator instance with mock client."""
    return QuestionVariationGenerator(
        client=mock_client,
        model="gpt-4o-mini",
        temperature_range=[0.7, 1.0, 1.3],
        num_variations=5,
    )


class TestQuestionVariationGenerator:
    """Test QuestionVariationGenerator functionality."""

    @pytest.mark.asyncio
    async def test_generates_correct_number_of_variations(self, generator, mock_client):
        """Test that generator produces the correct number of variations."""
        question = "What is the capital of France?"

        # Mock responses
        mock_client.generate_response.side_effect = [
            ModelResponse(
                content=f"Which city serves as France's capital?",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="gpt-4o-mini",
                request_id="req-1",
                latency_ms=100.0,
                temperature=0.7,
                finish_reason="stop",
                cached=False,
            ),
            ModelResponse(
                content=f"What city is the French capital?",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="gpt-4o-mini",
                request_id="req-2",
                latency_ms=100.0,
                temperature=1.0,
                finish_reason="stop",
                cached=False,
            ),
            ModelResponse(
                content=f"France's capital is located in which city?",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="gpt-4o-mini",
                request_id="req-3",
                latency_ms=100.0,
                temperature=1.3,
                finish_reason="stop",
                cached=False,
            ),
            ModelResponse(
                content=f"Name the capital of France",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="gpt-4o-mini",
                request_id="req-4",
                latency_ms=100.0,
                temperature=0.7,
                finish_reason="stop",
                cached=False,
            ),
            ModelResponse(
                content=f"What's the capital city of France?",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="gpt-4o-mini",
                request_id="req-5",
                latency_ms=100.0,
                temperature=1.0,
                finish_reason="stop",
                cached=False,
            ),
        ]

        variations = await generator.generate_variations(question)

        assert len(variations) == 5
        assert mock_client.generate_response.call_count == 5

    @pytest.mark.asyncio
    async def test_distributes_across_temperature_range(self, generator, mock_client):
        """Test that variations are generated with different temperatures."""
        question = "What is the capital of France?"

        # Mock responses
        mock_client.generate_response.side_effect = [
            ModelResponse(
                content=f"Variation {i}",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="gpt-4o-mini",
                request_id=f"req-{i}",
                latency_ms=100.0,
                temperature=0.7,
                finish_reason="stop",
                cached=False,
            )
            for i in range(5)
        ]

        await generator.generate_variations(question)

        # Check that temperature values were cycled through
        calls = mock_client.generate_response.call_args_list
        temperatures_used = [call.kwargs["temperature"] for call in calls]

        # With 5 variations and 3 temperature values, expect: [0.7, 1.0, 1.3, 0.7, 1.0]
        expected_temps = [0.7, 1.0, 1.3, 0.7, 1.0]
        assert temperatures_used == expected_temps

    @pytest.mark.asyncio
    async def test_handles_api_errors_gracefully(self, generator, mock_client):
        """Test that generator handles API errors with retries."""
        question = "What is the capital of France?"

        # First call fails, second succeeds
        mock_client.generate_response.side_effect = [
            Exception("API Error"),
            ModelResponse(
                content="Which city serves as France's capital?",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="gpt-4o-mini",
                request_id="req-1",
                latency_ms=100.0,
                temperature=0.7,
                finish_reason="stop",
                cached=False,
            ),
            ModelResponse(
                content="What city is the French capital?",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="gpt-4o-mini",
                request_id="req-2",
                latency_ms=100.0,
                temperature=1.0,
                finish_reason="stop",
                cached=False,
            ),
            ModelResponse(
                content="France's capital is located in which city?",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="gpt-4o-mini",
                request_id="req-3",
                latency_ms=100.0,
                temperature=1.3,
                finish_reason="stop",
                cached=False,
            ),
            ModelResponse(
                content="Name the capital of France",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="gpt-4o-mini",
                request_id="req-4",
                latency_ms=100.0,
                temperature=0.7,
                finish_reason="stop",
                cached=False,
            ),
            ModelResponse(
                content="What's the capital city of France?",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="gpt-4o-mini",
                request_id="req-5",
                latency_ms=100.0,
                temperature=1.0,
                finish_reason="stop",
                cached=False,
            ),
        ]

        variations = await generator.generate_variations(question)

        # Should still get 5 variations (with retry)
        assert len(variations) == 5
        # Should have made 6 calls (1 failed + 5 successful)
        assert mock_client.generate_response.call_count == 6

    @pytest.mark.asyncio
    async def test_filters_identical_variations(self, generator, mock_client):
        """Test that identical variations are filtered out."""
        question = "What is the capital of France?"

        # Mock responses with one duplicate
        mock_client.generate_response.side_effect = [
            ModelResponse(
                content="Which city serves as France's capital?",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="gpt-4o-mini",
                request_id="req-1",
                latency_ms=100.0,
                temperature=0.7,
                finish_reason="stop",
                cached=False,
            ),
            ModelResponse(
                content="What is the capital of France?",  # Identical to original
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="gpt-4o-mini",
                request_id="req-2",
                latency_ms=100.0,
                temperature=1.0,
                finish_reason="stop",
                cached=False,
            ),
            ModelResponse(
                content="France's capital is located in which city?",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="gpt-4o-mini",
                request_id="req-3",
                latency_ms=100.0,
                temperature=1.3,
                finish_reason="stop",
                cached=False,
            ),
            ModelResponse(
                content="Name the capital of France",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="gpt-4o-mini",
                request_id="req-4",
                latency_ms=100.0,
                temperature=0.7,
                finish_reason="stop",
                cached=False,
            ),
            ModelResponse(
                content="What's the capital city of France?",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="gpt-4o-mini",
                request_id="req-5",
                latency_ms=100.0,
                temperature=1.0,
                finish_reason="stop",
                cached=False,
            ),
            # Retry for the filtered duplicate
            ModelResponse(
                content="What city is the French capital?",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="gpt-4o-mini",
                request_id="req-6",
                latency_ms=100.0,
                temperature=1.3,
                finish_reason="stop",
                cached=False,
            ),
        ]

        variations = await generator.generate_variations(question)

        # Should get 5 unique variations (duplicate filtered and replaced)
        assert len(variations) == 5
        assert question not in variations  # Original question filtered out

    @pytest.mark.asyncio
    async def test_validates_paraphrases_are_meaningful(self, generator, mock_client):
        """Test that near-identical variations are filtered out."""
        question = "What is the capital of France?"

        # Mock responses with one that's too similar (minimal edit distance)
        mock_client.generate_response.side_effect = [
            ModelResponse(
                content="Which city serves as France's capital?",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="gpt-4o-mini",
                request_id="req-1",
                latency_ms=100.0,
                temperature=0.7,
                finish_reason="stop",
                cached=False,
            ),
            ModelResponse(
                content="What is the capital of France",  # Too similar (missing ?)
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="gpt-4o-mini",
                request_id="req-2",
                latency_ms=100.0,
                temperature=1.0,
                finish_reason="stop",
                cached=False,
            ),
            ModelResponse(
                content="France's capital is located in which city?",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="gpt-4o-mini",
                request_id="req-3",
                latency_ms=100.0,
                temperature=1.3,
                finish_reason="stop",
                cached=False,
            ),
            ModelResponse(
                content="Name the capital of France",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="gpt-4o-mini",
                request_id="req-4",
                latency_ms=100.0,
                temperature=0.7,
                finish_reason="stop",
                cached=False,
            ),
            ModelResponse(
                content="What's the capital city of France?",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="gpt-4o-mini",
                request_id="req-5",
                latency_ms=100.0,
                temperature=1.0,
                finish_reason="stop",
                cached=False,
            ),
            # Retry for the filtered near-duplicate
            ModelResponse(
                content="What city is the French capital?",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="gpt-4o-mini",
                request_id="req-6",
                latency_ms=100.0,
                temperature=1.3,
                finish_reason="stop",
                cached=False,
            ),
        ]

        variations = await generator.generate_variations(question)

        # Should get 5 unique, sufficiently different variations
        assert len(variations) == 5
        # Near-duplicate should be filtered
        assert "What is the capital of France" not in variations
