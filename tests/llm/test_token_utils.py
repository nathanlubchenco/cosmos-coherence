"""Tests for token counting utilities."""


import pytest
from cosmos_coherence.llm.token_utils import (
    TokenCounter,
    count_tokens,
    estimate_cost,
    get_model_pricing,
)


class TestTokenCounting:
    """Test token counting functionality."""

    def test_count_tokens_gpt35(self):
        """Test token counting for GPT-3.5-turbo."""
        text = "Hello, world! This is a test message."
        tokens = count_tokens(text, model="gpt-3.5-turbo")

        assert isinstance(tokens, int)
        assert tokens > 0
        # Approximate range for this text
        assert 5 <= tokens <= 15

    def test_count_tokens_gpt4(self):
        """Test token counting for GPT-4."""
        text = "Hello, world! This is a test message."
        tokens = count_tokens(text, model="gpt-4")

        assert isinstance(tokens, int)
        assert tokens > 0
        # Should be similar to GPT-3.5 count
        assert 5 <= tokens <= 15

    def test_count_tokens_empty_string(self):
        """Test token counting for empty string."""
        tokens = count_tokens("", model="gpt-3.5-turbo")
        assert tokens == 0

    def test_count_tokens_special_characters(self):
        """Test token counting with special characters."""
        text = "Test with émojis 😀 and symbols: @#$%^&*()"
        tokens = count_tokens(text, model="gpt-3.5-turbo")

        assert isinstance(tokens, int)
        assert tokens > 0

    def test_count_tokens_long_text(self):
        """Test token counting for longer text."""
        text = " ".join(["word"] * 1000)
        tokens = count_tokens(text, model="gpt-3.5-turbo")

        assert isinstance(tokens, int)
        # Roughly 1 token per word for simple words
        assert 800 <= tokens <= 1200

    def test_count_tokens_with_messages_format(self):
        """Test token counting with chat messages format."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help?"},
        ]

        tokens = count_tokens(messages, model="gpt-3.5-turbo", is_messages=True)

        assert isinstance(tokens, int)
        assert tokens > 0
        # Should include overhead for message formatting
        assert tokens > 10


class TestCostEstimation:
    """Test cost estimation functionality."""

    def test_get_model_pricing_gpt35(self):
        """Test getting pricing for GPT-3.5-turbo."""
        pricing = get_model_pricing("gpt-3.5-turbo")

        assert "input" in pricing
        assert "output" in pricing
        assert pricing["input"] > 0
        assert pricing["output"] > 0
        assert pricing["output"] > pricing["input"]  # Output usually more expensive

    def test_get_model_pricing_gpt4(self):
        """Test getting pricing for GPT-4."""
        pricing = get_model_pricing("gpt-4")

        assert "input" in pricing
        assert "output" in pricing
        assert pricing["input"] > 0
        assert pricing["output"] > 0

        # GPT-4 should be more expensive than GPT-3.5
        gpt35_pricing = get_model_pricing("gpt-3.5-turbo")
        assert pricing["input"] > gpt35_pricing["input"]
        assert pricing["output"] > gpt35_pricing["output"]

    def test_get_model_pricing_unknown_model(self):
        """Test pricing for unknown model returns default."""
        pricing = get_model_pricing("unknown-model")

        assert "input" in pricing
        assert "output" in pricing
        # Should return some default pricing
        assert pricing["input"] > 0
        assert pricing["output"] > 0

    def test_estimate_cost_gpt35(self):
        """Test cost estimation for GPT-3.5-turbo."""
        cost = estimate_cost(
            prompt_tokens=1000,
            completion_tokens=500,
            model="gpt-3.5-turbo",
        )

        assert isinstance(cost, float)
        assert cost > 0
        # Rough estimate (prices may vary)
        assert 0.0001 <= cost <= 0.01

    def test_estimate_cost_gpt4(self):
        """Test cost estimation for GPT-4."""
        cost = estimate_cost(
            prompt_tokens=1000,
            completion_tokens=500,
            model="gpt-4",
        )

        assert isinstance(cost, float)
        assert cost > 0

        # Should be more expensive than GPT-3.5
        gpt35_cost = estimate_cost(
            prompt_tokens=1000,
            completion_tokens=500,
            model="gpt-3.5-turbo",
        )
        assert cost > gpt35_cost

    def test_estimate_cost_zero_tokens(self):
        """Test cost estimation with zero tokens."""
        cost = estimate_cost(
            prompt_tokens=0,
            completion_tokens=0,
            model="gpt-3.5-turbo",
        )

        assert cost == 0.0

    def test_estimate_cost_batch_discount(self):
        """Test cost estimation with batch API discount."""
        regular_cost = estimate_cost(
            prompt_tokens=1000,
            completion_tokens=500,
            model="gpt-3.5-turbo",
        )

        batch_cost = estimate_cost(
            prompt_tokens=1000,
            completion_tokens=500,
            model="gpt-3.5-turbo",
            batch_api=True,
        )

        assert batch_cost < regular_cost
        # Batch API typically offers 50% discount
        assert batch_cost == pytest.approx(regular_cost * 0.5)


class TestTokenCounter:
    """Test TokenCounter class for tracking usage."""

    def test_token_counter_initialization(self):
        """Test TokenCounter initialization."""
        counter = TokenCounter()

        assert counter.total_prompt_tokens == 0
        assert counter.total_completion_tokens == 0
        assert counter.total_tokens == 0
        assert counter.total_cost == 0.0
        assert counter.request_count == 0

    def test_token_counter_add_usage(self):
        """Test adding usage to TokenCounter."""
        counter = TokenCounter()

        counter.add_usage(
            prompt_tokens=100,
            completion_tokens=50,
            model="gpt-3.5-turbo",
        )

        assert counter.total_prompt_tokens == 100
        assert counter.total_completion_tokens == 50
        assert counter.total_tokens == 150
        assert counter.total_cost > 0
        assert counter.request_count == 1

    def test_token_counter_multiple_requests(self):
        """Test TokenCounter with multiple requests."""
        counter = TokenCounter()

        counter.add_usage(100, 50, "gpt-3.5-turbo")
        counter.add_usage(200, 100, "gpt-3.5-turbo")
        counter.add_usage(150, 75, "gpt-4")

        assert counter.total_prompt_tokens == 450
        assert counter.total_completion_tokens == 225
        assert counter.total_tokens == 675
        assert counter.request_count == 3
        assert counter.total_cost > 0

    def test_token_counter_reset(self):
        """Test resetting TokenCounter."""
        counter = TokenCounter()

        counter.add_usage(100, 50, "gpt-3.5-turbo")
        assert counter.total_tokens == 150

        counter.reset()

        assert counter.total_prompt_tokens == 0
        assert counter.total_completion_tokens == 0
        assert counter.total_tokens == 0
        assert counter.total_cost == 0.0
        assert counter.request_count == 0

    def test_token_counter_get_summary(self):
        """Test getting summary from TokenCounter."""
        counter = TokenCounter()

        counter.add_usage(1000, 500, "gpt-3.5-turbo")
        counter.add_usage(2000, 1000, "gpt-4")

        summary = counter.get_summary()

        assert "total_tokens" in summary
        assert "total_cost" in summary
        assert "request_count" in summary
        assert "average_tokens_per_request" in summary

        assert summary["total_tokens"] == 4500
        assert summary["request_count"] == 2
        assert summary["average_tokens_per_request"] == 2250

    def test_token_counter_model_breakdown(self):
        """Test TokenCounter tracking per model."""
        counter = TokenCounter()

        counter.add_usage(100, 50, "gpt-3.5-turbo")
        counter.add_usage(200, 100, "gpt-3.5-turbo")
        counter.add_usage(300, 150, "gpt-4")

        breakdown = counter.get_model_breakdown()

        assert "gpt-3.5-turbo" in breakdown
        assert "gpt-4" in breakdown

        assert breakdown["gpt-3.5-turbo"]["tokens"] == 450
        assert breakdown["gpt-3.5-turbo"]["requests"] == 2
        assert breakdown["gpt-4"]["tokens"] == 450
        assert breakdown["gpt-4"]["requests"] == 1
