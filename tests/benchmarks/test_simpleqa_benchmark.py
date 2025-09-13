"""Tests for SimpleQA benchmark implementation."""

import pytest
from cosmos_coherence.benchmarks.implementations.simpleqa_benchmark import SimpleQABenchmark
from cosmos_coherence.benchmarks.models.datasets import SimpleQAItem


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


class TestAdvancedNormalization:
    """Test advanced normalization for exact match scoring."""

    @pytest.fixture
    def benchmark(self):
        """Create a SimpleQA benchmark instance."""
        return SimpleQABenchmark(use_huggingface=False)

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
