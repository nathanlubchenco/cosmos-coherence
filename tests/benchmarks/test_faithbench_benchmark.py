"""Tests for FaithBench benchmark implementation."""

import uuid
from typing import List
from unittest.mock import AsyncMock, patch

import pytest
from cosmos_coherence.benchmarks.models.datasets import (
    FaithBenchAnnotation,
    FaithBenchItem,
)
from cosmos_coherence.harness.base_benchmark import (
    BenchmarkEvaluationResult,
    BenchmarkMetadata,
)


class TestFaithBenchBenchmark:
    """Test suite for FaithBench benchmark implementation."""

    @pytest.fixture
    def sample_faithbench_items(self) -> List[FaithBenchItem]:
        """Create sample FaithBench items for testing."""
        return [
            FaithBenchItem(
                id=str(uuid.uuid4()),
                sample_id="fb_001",
                source=(
                    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
                    "in Paris, France. It was built from 1887 to 1889."
                ),
                summary="The Eiffel Tower is a steel structure in Paris built in 1885.",
                annotation_label=FaithBenchAnnotation.HALLUCINATED,
                annotation_spans=["steel structure", "1885"],
                entropy_score=0.67,
                question="Summarize the text about the Eiffel Tower.",
            ),
            FaithBenchItem(
                id=str(uuid.uuid4()),
                sample_id="fb_002",
                source=(
                    "Machine learning is a subset of artificial intelligence that enables "
                    "systems to learn from experience."
                ),
                summary="Machine learning is a subset of AI that allows systems to learn.",
                annotation_label=FaithBenchAnnotation.CONSISTENT,
                annotation_spans=[],
                entropy_score=0.0,
                question="Summarize the text about machine learning.",
            ),
        ]

    @pytest.fixture
    def faithbench_benchmark(self):
        """Create FaithBenchBenchmark instance (to be implemented)."""
        # This will be imported once implemented
        from cosmos_coherence.benchmarks.faithbench import FaithBenchBenchmark

        return FaithBenchBenchmark()

    @pytest.mark.asyncio
    async def test_load_dataset(self, faithbench_benchmark, sample_faithbench_items):
        """Test loading FaithBench dataset."""
        # Just set the cache directly for testing instead of mocking aiohttp
        faithbench_benchmark._dataset_cache = sample_faithbench_items

        dataset = await faithbench_benchmark.load_dataset()

        assert len(dataset) == 2
        assert all(isinstance(item, FaithBenchItem) for item in dataset)
        assert dataset[0].sample_id == "fb_001"

    @pytest.mark.asyncio
    async def test_load_dataset_with_sample_size(self, faithbench_benchmark):
        """Test loading dataset with sample size."""
        # Create 200 sample items
        large_dataset = [
            FaithBenchItem(
                id=str(uuid.uuid4()),
                sample_id=f"fb_{i:03d}",
                source=f"Source {i}",
                summary=f"Summary {i}",
                question="Summarize.",
            )
            for i in range(200)
        ]

        faithbench_benchmark._dataset_cache = large_dataset
        dataset = await faithbench_benchmark.load_dataset(sample_size=100)

        # Should return only 100 items
        assert len(dataset) == 100

    def test_get_prompt_summarization(self, faithbench_benchmark, sample_faithbench_items):
        """Test prompt generation for summarization task."""
        item = sample_faithbench_items[0]
        prompt = faithbench_benchmark.get_prompt(item)

        # Check prompt follows FaithBench format
        assert "summarize" in prompt.lower() or "summary" in prompt.lower()
        assert item.source in prompt
        assert "Eiffel Tower" in prompt

    def test_get_prompt_follows_paper_format(self, faithbench_benchmark):
        """Test that prompts match the format from the FaithBench paper."""
        item = FaithBenchItem(
            sample_id="test",
            source="This is the source text to be summarized.",
            summary="This is a summary.",
            question="Summarize the text.",
        )

        prompt = faithbench_benchmark.get_prompt(item)

        # According to paper, should be a simple summarization prompt
        assert item.source in prompt
        # Should not include the summary in the prompt
        assert item.summary not in prompt

    def test_evaluate_response_consistent(self, faithbench_benchmark):
        """Test evaluating a consistent (no hallucination) response."""
        item = FaithBenchItem(
            sample_id="test",
            source="The sky is blue during the day.",
            summary="The sky is blue during daytime.",
            annotation_label=FaithBenchAnnotation.CONSISTENT,
            question="Summarize.",
        )

        # Response says "yes" meaning it's consistent
        result = faithbench_benchmark.evaluate_response(
            response="yes",  # Model predicts consistent
            ground_truth="The sky is blue during daytime.",
            item=item,
        )

        assert isinstance(result, BenchmarkEvaluationResult)
        assert result.is_correct is True
        assert result.score == 1.0  # Correct prediction
        assert "consistent" in result.metadata.get("annotation_label", "").lower()

    def test_evaluate_response_hallucinated(self, faithbench_benchmark):
        """Test evaluating a hallucinated response."""
        item = FaithBenchItem(
            sample_id="test",
            source="The Earth orbits the Sun.",
            summary="The Sun orbits the Earth.",
            annotation_label=FaithBenchAnnotation.HALLUCINATED,
            annotation_spans=["Sun orbits the Earth"],
            question="Summarize.",
        )

        # Response says "no" meaning it's inconsistent/hallucinated
        result = faithbench_benchmark.evaluate_response(
            response="no",  # Model correctly detects hallucination
            ground_truth="The Sun orbits the Earth.",
            item=item,
        )

        assert isinstance(result, BenchmarkEvaluationResult)
        # Model correctly identified hallucination
        assert result.is_correct is True
        assert result.score == 1.0  # Correct detection
        assert "hallucinated" in result.metadata.get("annotation_label", "").lower()

    def test_evaluate_response_with_entropy_score(self, faithbench_benchmark):
        """Test that entropy score is included in evaluation metadata."""
        item = FaithBenchItem(
            sample_id="test",
            source="Test source.",
            summary="Test summary.",
            entropy_score=0.75,  # High entropy = challenging sample
            question="Summarize.",
        )

        result = faithbench_benchmark.evaluate_response(
            response="Test summary.", ground_truth="Test summary.", item=item
        )

        assert "entropy_score" in result.metadata
        assert result.metadata["entropy_score"] == 0.75
        assert result.metadata.get("is_challenging") is True  # High entropy = challenging

    def test_get_baseline_metrics(self, faithbench_benchmark):
        """Test getting baseline metrics from FaithBench paper."""
        metrics = faithbench_benchmark.get_baseline_metrics()

        assert isinstance(metrics, dict)
        # Should include metrics for GPT-4-Turbo, GPT-4o, GPT-3.5-Turbo
        assert "gpt-4-turbo_accuracy" in metrics
        assert "gpt-4o_accuracy" in metrics
        assert "gpt-3.5-turbo_accuracy" in metrics

        # Check F1 scores are also included
        assert "gpt-4-turbo_f1" in metrics
        assert "gpt-4o_f1" in metrics

        # Verify the actual values from the paper
        assert metrics["gpt-4-turbo_accuracy"] == 0.5765  # 57.65%
        assert metrics["gpt-4o_accuracy"] == 0.5629  # 56.29%
        assert metrics["gpt-3.5-turbo_accuracy"] == 0.4491  # 44.91%

        # Verify metric ranges for non-None values
        for key, value in metrics.items():
            if value is not None:
                assert 0.0 <= value <= 1.0

    def test_get_original_prompts(self, faithbench_benchmark):
        """Test getting example prompts from paper."""
        prompts = faithbench_benchmark.get_original_prompts()

        assert isinstance(prompts, list)
        assert len(prompts) >= 1
        # Each prompt should be for summarization
        for prompt in prompts:
            assert "summar" in prompt.lower()

    def test_validate_config_valid(self, faithbench_benchmark):
        """Test config validation with valid config."""
        valid_config = {
            "model": "gpt-4-turbo",
            "temperature": 0.0,
            "max_tokens": 150,
            "sample_size": 100,
        }

        # Should not raise
        faithbench_benchmark.validate_config(valid_config)

    def test_validate_config_invalid_model(self, faithbench_benchmark):
        """Test config validation with unsupported model."""
        invalid_config = {"model": "unsupported-model", "temperature": 0.0}

        with pytest.raises(ValueError) as exc:
            faithbench_benchmark.validate_config(invalid_config)
        assert "model" in str(exc.value).lower()

    def test_validate_config_invalid_temperature(self, faithbench_benchmark):
        """Test config validation with invalid temperature."""
        # Temperature must be between 0 and 2
        invalid_config = {"model": "gpt-4-turbo", "temperature": 2.5}

        with pytest.raises(ValueError) as exc:
            faithbench_benchmark.validate_config(invalid_config)
        assert "temperature" in str(exc.value).lower() or "Temperature" in str(exc.value)

    def test_benchmark_name(self, faithbench_benchmark):
        """Test benchmark name property."""
        assert faithbench_benchmark.benchmark_name == "faithbench"

    def test_paper_reference(self, faithbench_benchmark):
        """Test paper reference property."""
        ref = faithbench_benchmark.paper_reference
        assert "FaithBench" in ref
        assert "2024" in ref or "arXiv" in ref

    def test_get_evaluation_method(self, faithbench_benchmark):
        """Test evaluation method description."""
        method = faithbench_benchmark.get_evaluation_method()
        assert "binary classification" in method.lower()
        # The method description should mention challenging samples or entropy
        assert "challenging" in method.lower() or "entropy" in method.lower()

    def test_get_metadata(self, faithbench_benchmark):
        """Test getting benchmark metadata."""
        metadata = faithbench_benchmark.get_metadata()

        assert isinstance(metadata, BenchmarkMetadata)
        assert metadata.name == "faithbench"
        assert "FaithBench" in metadata.paper_reference
        assert metadata.baseline_metrics

    def test_format_for_api(self, faithbench_benchmark, sample_faithbench_items):
        """Test formatting items for API calls."""
        item = sample_faithbench_items[0]
        messages = faithbench_benchmark.format_for_api(item)

        assert isinstance(messages, list)
        assert len(messages) >= 1
        assert messages[-1]["role"] == "user"
        assert item.source in messages[-1]["content"]

    def test_format_for_api_with_system_prompt(self, faithbench_benchmark, sample_faithbench_items):
        """Test formatting with system prompt."""
        item = sample_faithbench_items[0]
        system_prompt = "You are a helpful summarization assistant."
        messages = faithbench_benchmark.format_for_api(item, system_prompt=system_prompt)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == system_prompt
        assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_validate_dataset(self, faithbench_benchmark, sample_faithbench_items):
        """Test dataset validation."""
        # Mock load_dataset to return sample items
        with patch.object(
            faithbench_benchmark, "load_dataset", new_callable=AsyncMock
        ) as mock_load:
            mock_load.return_value = sample_faithbench_items

            is_valid, issues = await faithbench_benchmark.validate_dataset()

            assert is_valid is True
            assert len(issues) == 0

    def test_supports_original_evaluation(self, faithbench_benchmark):
        """Test that FaithBench supports original evaluation."""
        assert faithbench_benchmark.supports_original_evaluation() is True

    def test_get_required_model_capabilities(self, faithbench_benchmark):
        """Test required model capabilities."""
        capabilities = faithbench_benchmark.get_required_model_capabilities()

        assert capabilities["text_generation"] is True
        assert capabilities["max_tokens"] >= 100  # Need enough for summaries
        assert "temperature_control" in capabilities

    def test_preprocess_response(self, faithbench_benchmark):
        """Test response preprocessing."""
        response = "  This is a summary.  \n\n"
        processed = faithbench_benchmark.preprocess_response(response)
        assert processed == "This is a summary."

    def test_postprocess_results(self, faithbench_benchmark):
        """Test aggregating results."""
        results = [
            BenchmarkEvaluationResult(
                is_correct=True,
                score=0.9,
                original_metric_score=0.85,
                metadata={"annotation_label": "consistent"},
            ),
            BenchmarkEvaluationResult(
                is_correct=False,
                score=0.2,
                original_metric_score=0.3,
                metadata={"annotation_label": "hallucinated"},
            ),
            BenchmarkEvaluationResult(
                is_correct=True,
                score=0.7,
                original_metric_score=0.75,
                metadata={"annotation_label": "benign"},
            ),
        ]

        aggregated = faithbench_benchmark.postprocess_results(results)

        assert aggregated["accuracy"] == 2 / 3
        assert aggregated["average_score"] == pytest.approx((0.9 + 0.2 + 0.7) / 3)
        assert aggregated["total_evaluated"] == 3
        assert aggregated["total_correct"] == 2

    def test_challenging_samples_handling(self, faithbench_benchmark):
        """Test handling of challenging samples (high entropy)."""
        challenging_item = FaithBenchItem(
            sample_id="challenging",
            source="Complex source text.",
            summary="Summary with disagreement.",
            entropy_score=0.8,  # High entropy
            detector_predictions={"model1": 1, "model2": 0, "model3": 1},
            question="Summarize.",
        )

        result = faithbench_benchmark.evaluate_response(
            response="Summary with disagreement.",
            ground_truth="Summary with disagreement.",
            item=challenging_item,
        )

        # Should flag as challenging
        assert result.metadata.get("is_challenging") is True
        assert result.metadata.get("entropy_score") == 0.8
        assert "detector_predictions" in result.metadata

    def test_four_level_taxonomy_in_evaluation(self, faithbench_benchmark):
        """Test that all four annotation levels are handled in evaluation."""
        levels = [
            FaithBenchAnnotation.CONSISTENT,
            FaithBenchAnnotation.QUESTIONABLE,
            FaithBenchAnnotation.BENIGN,
            FaithBenchAnnotation.HALLUCINATED,
        ]

        for level in levels:
            item = FaithBenchItem(
                sample_id=f"test_{level.value}",
                source="Test source.",
                summary="Test summary.",
                annotation_label=level,
                question="Summarize.",
            )

            result = faithbench_benchmark.evaluate_response(
                response="Test summary.", ground_truth="Test summary.", item=item
            )

            assert result.metadata["annotation_label"] == level.value

    @pytest.mark.asyncio
    async def test_integration_with_github_loader(self, faithbench_benchmark):
        """Test integration with GitHub dataset loading."""
        # Create test dataset
        test_dataset = [
            FaithBenchItem(
                id=str(uuid.uuid4()),
                sample_id=f"fb_{i:03d}",
                source=f"Source {i}",
                summary=f"Summary {i}",
                question="Summarize.",
            )
            for i in range(50)
        ]

        faithbench_benchmark._dataset_cache = test_dataset
        dataset = await faithbench_benchmark.load_dataset(sample_size=50)

        assert len(dataset) == 50
