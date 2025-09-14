"""Tests for SimpleQA benchmark implementation."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from cosmos_coherence.benchmarks.implementations.simpleqa_benchmark import SimpleQABenchmark
from cosmos_coherence.benchmarks.models.datasets import SimpleQAItem
from cosmos_coherence.llm.models import ModelResponse, TokenUsage


class TestSimpleQAEvaluationMetrics:
    """Test evaluation metrics for SimpleQA benchmark."""

    @pytest.fixture
    def benchmark(self):
        """Create a SimpleQA benchmark instance."""
        return SimpleQABenchmark(use_huggingface=False)

    @pytest.fixture
    def sample_item(self):
        """Create a sample SimpleQA item."""
        return SimpleQAItem(question="What is the capital of France?", best_answer="Paris")

    def test_exact_match_identical(self, benchmark, sample_item):
        """Test exact match with identical strings."""
        result = benchmark.evaluate_response("Paris", "Paris", sample_item)
        assert result.is_correct is True
        assert result.metadata["exact_match"] is True
        assert result.metadata["f1_score"] == 1.0
        assert result.score == 1.0

    def test_exact_match_case_insensitive(self, benchmark, sample_item):
        """Test exact match is case-insensitive."""
        result = benchmark.evaluate_response("paris", "Paris", sample_item)
        assert result.is_correct is True
        assert result.metadata["exact_match"] is True
        assert result.metadata["f1_score"] == 1.0

    def test_exact_match_whitespace_normalized(self, benchmark, sample_item):
        """Test exact match with whitespace normalization."""
        result = benchmark.evaluate_response("  Paris  ", "Paris", sample_item)
        assert result.is_correct is True
        assert result.metadata["exact_match"] is True

    def test_exact_match_fails_different_answers(self, benchmark, sample_item):
        """Test exact match fails for different answers."""
        result = benchmark.evaluate_response("London", "Paris", sample_item)
        assert result.is_correct is False
        assert result.metadata["exact_match"] is False
        assert result.metadata["f1_score"] == 0.0
        assert result.score == 0.0

    def test_f1_score_perfect_match(self, benchmark, sample_item):
        """Test F1 score for perfect match."""
        result = benchmark.evaluate_response("Paris", "Paris", sample_item)
        assert result.metadata["f1_score"] == 1.0

    def test_f1_score_partial_match(self, benchmark, sample_item):
        """Test F1 score for partial token overlap."""
        # "The capital is Paris" vs "Paris"
        # Tokens: {the, capital, is, paris} vs {paris}
        # Intersection: {paris}
        # Precision: 1/4 = 0.25, Recall: 1/1 = 1.0
        # F1 = 2 * (0.25 * 1.0) / (0.25 + 1.0) = 0.4
        result = benchmark.evaluate_response("The capital is Paris", "Paris", sample_item)
        assert result.metadata["exact_match"] is False
        assert result.metadata["f1_score"] == 0.4
        assert result.score == 0.2  # (0 + 0.4) / 2

    def test_f1_score_no_overlap(self, benchmark, sample_item):
        """Test F1 score with no token overlap."""
        result = benchmark.evaluate_response("London", "Paris", sample_item)
        assert result.metadata["f1_score"] == 0.0
        assert result.score == 0.0

    def test_f1_score_multi_word_answer(self, benchmark):
        """Test F1 score with multi-word answers."""
        item = SimpleQAItem(
            question="What is the speed of light?", best_answer="299,792,458 meters per second"
        )
        # Exact match
        result = benchmark.evaluate_response(
            "299,792,458 meters per second", "299,792,458 meters per second", item
        )
        assert result.metadata["f1_score"] == 1.0

        # Partial match: "299,792,458 m/s" vs "299,792,458 meters per second"
        # Tokens: {299,792,458, m/s} vs {299,792,458, meters, per, second}
        # Intersection: {299,792,458}
        # Precision: 1/2 = 0.5, Recall: 1/4 = 0.25
        # F1 = 2 * (0.5 * 0.25) / (0.5 + 0.25) = 0.333...
        result = benchmark.evaluate_response(
            "299,792,458 m/s", "299,792,458 meters per second", item
        )
        assert result.metadata["exact_match"] is False
        assert pytest.approx(result.metadata["f1_score"], 0.01) == 0.333

    def test_f1_score_empty_response(self, benchmark, sample_item):
        """Test F1 score with empty response."""
        result = benchmark.evaluate_response("", "Paris", sample_item)
        assert result.metadata["f1_score"] == 0.0
        assert result.is_correct is False

    def test_f1_score_empty_ground_truth(self, benchmark, sample_item):
        """Test F1 score with empty ground truth (edge case)."""
        result = benchmark.evaluate_response("Paris", "", sample_item)
        assert result.metadata["f1_score"] == 0.0

    def test_f1_score_both_empty(self, benchmark, sample_item):
        """Test F1 score when both strings are empty."""
        result = benchmark.evaluate_response("", "", sample_item)
        assert result.metadata["f1_score"] == 1.0
        assert result.is_correct is True

    def test_evaluation_result_metadata(self, benchmark, sample_item):
        """Test that evaluation result contains expected metadata."""
        result = benchmark.evaluate_response("Paris, France", "Paris", sample_item)

        assert "exact_match" in result.metadata
        assert "f1_score" in result.metadata
        assert "response_length" in result.metadata
        assert "ground_truth_length" in result.metadata

        assert result.metadata["response_length"] == 2  # {paris,, france}
        assert result.metadata["ground_truth_length"] == 1  # {paris}

    def test_normalization_removes_punctuation(self, benchmark, sample_item):
        """Test that normalization handles punctuation properly."""
        # Note: Current implementation doesn't remove punctuation, just lowercases
        # "Paris." vs "Paris" would not match exactly
        result = benchmark.evaluate_response("Paris.", "Paris", sample_item)
        # Tokens: {paris.} vs {paris} - different tokens
        assert result.metadata["exact_match"] is False

    def test_score_calculation(self, benchmark, sample_item):
        """Test overall score calculation (average of exact match and F1)."""
        # Test 1: Perfect match
        result = benchmark.evaluate_response("Paris", "Paris", sample_item)
        assert result.score == 1.0  # (1.0 + 1.0) / 2

        # Test 2: No match
        result = benchmark.evaluate_response("London", "Paris", sample_item)
        assert result.score == 0.0  # (0.0 + 0.0) / 2

        # Test 3: Partial match
        result = benchmark.evaluate_response("Paris France", "Paris", sample_item)
        # Tokens: {paris, france} vs {paris}
        # Precision: 1/2 = 0.5, Recall: 1/1 = 1.0
        # F1 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 0.667
        # Score = (0 + 0.667) / 2 = 0.333
        assert result.metadata["exact_match"] is False
        assert pytest.approx(result.metadata["f1_score"], 0.01) == 0.667
        assert pytest.approx(result.score, 0.01) == 0.333


class TestSimpleQABenchmark:
    """Test SimpleQA benchmark class functionality."""

    @pytest.fixture
    def benchmark(self):
        """Create a SimpleQA benchmark instance."""
        return SimpleQABenchmark(use_huggingface=False)

    def test_get_prompt_formatting(self, benchmark):
        """Test prompt formatting for SimpleQA items."""
        item = SimpleQAItem(question="What is the capital of France?", best_answer="Paris")
        prompt = benchmark.get_prompt(item)
        assert prompt == "Question: What is the capital of France?\nAnswer:"

    def test_get_baseline_metrics(self, benchmark):
        """Test baseline metrics match paper results."""
        baselines = benchmark.get_baseline_metrics()
        assert baselines["gpt-4_accuracy"] == 0.82
        assert baselines["gpt-3.5_accuracy"] == 0.68
        assert "human_accuracy" in baselines

    def test_benchmark_name(self, benchmark):
        """Test benchmark name property."""
        assert benchmark.benchmark_name == "SimpleQA"

    def test_paper_reference(self, benchmark):
        """Test paper reference property."""
        ref = benchmark.paper_reference
        assert "SimpleQA" in ref
        assert "2024" in ref

    def test_get_evaluation_method(self, benchmark):
        """Test evaluation method description."""
        method = benchmark.get_evaluation_method()
        assert "exact match" in method.lower()
        assert "f1 score" in method.lower()

    def test_validate_config_high_temperature_warning(self, benchmark, caplog):
        """Test that high temperature triggers a warning."""
        config = {"model": {"temperature": 0.8}}
        benchmark.validate_config(config)
        assert "High temperature" in caplog.text
        assert "0.8" in caplog.text

    def test_validate_config_low_temperature_no_warning(self, benchmark, caplog):
        """Test that low temperature doesn't trigger warning."""
        config = {"model": {"temperature": 0.1}}
        benchmark.validate_config(config)
        assert "High temperature" not in caplog.text

    def test_get_original_prompts(self, benchmark):
        """Test getting example prompts."""
        prompts = benchmark.get_original_prompts()
        assert len(prompts) > 0
        assert all("Question:" in p for p in prompts)
        assert all("Answer:" in p for p in prompts)

    @pytest.mark.asyncio
    async def test_load_custom_dataset_not_implemented(self, benchmark):
        """Test that custom dataset loading raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await benchmark._load_custom_dataset()


class TestAIGrading:
    """Test AI-based grading integration in SimpleQA benchmark."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock OpenAI client."""
        client = MagicMock()
        client.generate_response = AsyncMock()
        return client

    @pytest.fixture
    def benchmark_with_ai(self, mock_client):
        """Create a SimpleQA benchmark with AI grading enabled."""
        return SimpleQABenchmark(client=mock_client, use_ai_grading=True, use_huggingface=False)

    @pytest.fixture
    def sample_item(self):
        """Create a sample SimpleQA item."""
        return SimpleQAItem(question="What is the capital of France?", best_answer="Paris")

    @pytest.mark.asyncio
    async def test_ai_grading_correct(self, benchmark_with_ai, mock_client, sample_item):
        """Test AI grading for correct answer."""
        # Mock the grader response
        mock_client.generate_response.return_value = ModelResponse(
            content="A",
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

        result = await benchmark_with_ai.evaluate_response_with_ai("Paris", "Paris", sample_item)

        assert result.is_correct is True
        assert result.score == 1.0
        assert result.metadata["grade"] == "CORRECT"

    @pytest.mark.asyncio
    async def test_ai_grading_incorrect(self, benchmark_with_ai, mock_client, sample_item):
        """Test AI grading for incorrect answer."""
        mock_client.generate_response.return_value = ModelResponse(
            content="B",
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

        result = await benchmark_with_ai.evaluate_response_with_ai("London", "Paris", sample_item)

        assert result.is_correct is False
        assert result.score == 0.0
        assert result.metadata["grade"] == "INCORRECT"

    @pytest.mark.asyncio
    async def test_ai_grading_semantic_match(self, benchmark_with_ai, mock_client, sample_item):
        """Test AI grading recognizes semantic matches."""
        mock_client.generate_response.return_value = ModelResponse(
            content="A",
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

        # AI grading should recognize "The capital is Paris" as correct
        result = await benchmark_with_ai.evaluate_response_with_ai(
            "The capital is Paris", "Paris", sample_item
        )

        assert result.is_correct is True
        assert result.metadata["grade"] == "CORRECT"

    @pytest.mark.asyncio
    async def test_ai_grading_not_attempted(self, benchmark_with_ai, mock_client, sample_item):
        """Test AI grading for not attempted answer."""
        mock_client.generate_response.return_value = ModelResponse(
            content="C",
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

        result = await benchmark_with_ai.evaluate_response_with_ai(
            "I don't know", "Paris", sample_item
        )

        assert result.is_correct is False
        assert result.score == 0.0
        assert result.metadata["grade"] == "NOT_ATTEMPTED"

    @pytest.mark.asyncio
    async def test_ai_grading_requires_client(self):
        """Test that AI grading requires an OpenAI client."""
        benchmark = SimpleQABenchmark(client=None, use_ai_grading=True, use_huggingface=False)
        item = SimpleQAItem(question="Test", best_answer="Answer")

        with pytest.raises(ValueError, match="AI grading requires an OpenAI client"):
            await benchmark.evaluate_response_with_ai("Response", "Answer", item)

    def test_prompt_format_matches_reference(self):
        """Test that prompt format matches OpenAI reference (just the question)."""
        benchmark = SimpleQABenchmark(use_huggingface=False)
        item = SimpleQAItem(question="What is the capital of France?", best_answer="Paris")

        prompt = benchmark.get_prompt(item)

        # OpenAI reference sends just the question, no formatting
        assert prompt == "What is the capital of France?"
        assert "Question:" not in prompt
        assert "Answer:" not in prompt

    def test_evaluation_method_description(self):
        """Test that evaluation method description changes based on grading type."""
        # With AI grading
        benchmark_ai = SimpleQABenchmark(use_ai_grading=True, use_huggingface=False)
        description = benchmark_ai.get_evaluation_method()
        assert "AI-based grading" in description
        assert "CORRECT, INCORRECT, or NOT_ATTEMPTED" in description

        # Without AI grading
        benchmark_exact = SimpleQABenchmark(use_ai_grading=False, use_huggingface=False)
        description = benchmark_exact.get_evaluation_method()
        assert "exact match and F1 score" in description


class TestAdvancedNormalization:
    """Test advanced normalization for exact match scoring."""

    @pytest.fixture
    def benchmark(self):
        """Create a SimpleQA benchmark instance without AI grading."""
        return SimpleQABenchmark(use_ai_grading=False, use_huggingface=False)

    def test_articles_not_removed(self, benchmark):
        """Test that articles are not automatically removed (current behavior)."""
        item = SimpleQAItem(question="Test", best_answer="The Paris")
        result = benchmark.evaluate_response("Paris", "The Paris", item)
        # Current implementation doesn't remove articles
        assert result.metadata["exact_match"] is False

    def test_numeric_equivalence(self, benchmark):
        """Test numeric answer variations."""
        item = SimpleQAItem(question="Test", best_answer="1945")

        # Exact match
        result = benchmark.evaluate_response("1945", "1945", item)
        assert result.metadata["exact_match"] is True

        # Different format (not handled by current implementation)
        result = benchmark.evaluate_response("1,945", "1945", item)
        assert result.metadata["exact_match"] is False

    def test_abbreviation_handling(self, benchmark):
        """Test abbreviation handling."""
        item = SimpleQAItem(question="Test", best_answer="United States")

        # Current implementation doesn't handle abbreviations
        result = benchmark.evaluate_response("US", "United States", item)
        assert result.metadata["exact_match"] is False
        # But F1 score is 0 since tokens don't overlap
        assert result.metadata["f1_score"] == 0.0
