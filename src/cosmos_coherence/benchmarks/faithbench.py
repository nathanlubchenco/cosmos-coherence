"""FaithBench benchmark implementation for hallucination detection in summarization."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from cosmos_coherence.benchmarks.faithbench_metrics import FaithBenchMetrics
from cosmos_coherence.benchmarks.models.base import BaseDatasetItem
from cosmos_coherence.benchmarks.models.datasets import FaithBenchAnnotation, FaithBenchItem
from cosmos_coherence.harness.base_benchmark import (
    BaseBenchmark,
    BenchmarkEvaluationResult,
)
from cosmos_coherence.harness.huggingface_loader import HuggingFaceDatasetLoader

logger = logging.getLogger(__name__)


class FaithBenchBenchmark(BaseBenchmark):
    """FaithBench benchmark for evaluating hallucination detection in summarization.

    FaithBench focuses on "challenging" samples where multiple detectors disagree,
    using entropy scores to identify these challenging cases.
    """

    # Supported models for Phase 1 (OpenAI only)
    SUPPORTED_MODELS = {
        "gpt-4-turbo",
        "gpt-4o",
        "o1-mini",
        "o3-mini",
    }

    # Models that don't support temperature variation (reasoning models)
    NO_TEMPERATURE_MODELS = {"o1-mini", "o3-mini"}

    # Baseline metrics from the FaithBench paper (placeholder values)
    # These should be updated with actual values from the paper
    BASELINE_METRICS = {
        "gpt-4-turbo_accuracy": 0.75,
        "gpt-4o_accuracy": 0.78,
        "o1-mini_accuracy": 0.82,
        "human_accuracy": 0.89,
        "average_entropy_challenging": 0.67,
    }

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """Initialize FaithBench benchmark.

        Args:
            cache_dir: Optional directory for caching datasets
        """
        super().__init__()
        cache_path = Path(cache_dir) if cache_dir else None
        self.loader = HuggingFaceDatasetLoader(cache_dir=cache_path)
        self.metrics_calculator = FaithBenchMetrics()
        self._dataset_cache: Optional[List[FaithBenchItem]] = None

    async def load_dataset(self, sample_size: Optional[int] = None) -> List[BaseDatasetItem]:
        """Load FaithBench dataset.

        Args:
            sample_size: Optional number of samples to load

        Returns:
            List of FaithBenchItem instances
        """
        if self._dataset_cache is None or sample_size is not None:
            logger.info(f"Loading FaithBench dataset (sample_size={sample_size})")
            self._dataset_cache = await self.loader.load_dataset(
                "faithbench", sample_size=sample_size
            )
            self._dataset = self._dataset_cache  # type: ignore
            logger.info(f"Loaded {len(self._dataset_cache)} FaithBench items")

        # Cast to base type for compatibility with parent class
        return self._dataset_cache  # type: ignore

    def get_prompt(self, item: BaseDatasetItem) -> str:
        """Generate summarization prompt for FaithBench item.

        According to the paper, FaithBench uses straightforward summarization prompts.

        Args:
            item: FaithBenchItem to generate prompt for

        Returns:
            Formatted prompt string
        """
        # Simple summarization prompt format from the paper
        # Cast to FaithBenchItem to access specific fields
        if not isinstance(item, FaithBenchItem):
            raise ValueError(f"Expected FaithBenchItem, got {type(item).__name__}")
        prompt = f"Please provide a concise summary of the following text:\n\n{item.source}"
        return prompt

    def evaluate_response(
        self, response: str, ground_truth: str, item: BaseDatasetItem
    ) -> BenchmarkEvaluationResult:
        """Evaluate model response for hallucination detection.

        Args:
            response: Model's generated summary
            ground_truth: Expected summary (from dataset)
            item: Original FaithBenchItem

        Returns:
            Evaluation result with hallucination detection scores
        """
        # Cast to FaithBenchItem to access specific fields
        if not isinstance(item, FaithBenchItem):
            raise ValueError(f"Expected FaithBenchItem, got {type(item).__name__}")

        # Normalize responses
        response = response.strip().lower()
        ground_truth = ground_truth.strip().lower()

        # Check for exact match (simple baseline)
        is_exact_match = response == ground_truth

        # Calculate similarity score (simple word overlap for now)
        response_words = set(response.split())
        truth_words = set(ground_truth.split())

        if len(response_words) == 0 and len(truth_words) == 0:
            similarity = 1.0
        elif len(response_words) == 0 or len(truth_words) == 0:
            similarity = 0.0
        else:
            overlap = len(response_words & truth_words)
            total = len(response_words | truth_words)
            similarity = overlap / total if total > 0 else 0.0

        # Determine score based on annotation label
        if item.annotation_label == FaithBenchAnnotation.CONSISTENT:
            # For consistent summaries, high similarity is good
            score = similarity
            is_correct = similarity >= 0.7
        elif item.annotation_label == FaithBenchAnnotation.HALLUCINATED:
            # For hallucinated summaries, we want to detect the hallucination
            # In practice, this would involve more sophisticated detection
            score = 1.0 - similarity  # Inverse for hallucination detection
            is_correct = similarity < 0.5  # Correctly identified as different
        elif item.annotation_label == FaithBenchAnnotation.QUESTIONABLE:
            # Questionable cases are in between
            score = 0.5 + (similarity - 0.5) * 0.5
            is_correct = 0.3 <= similarity <= 0.7
        else:  # BENIGN
            # Benign hallucinations are minor
            score = 0.7 + similarity * 0.3
            is_correct = similarity >= 0.6

        # Build metadata
        metadata = {
            "annotation_label": item.annotation_label.value if item.annotation_label else None,
            "entropy_score": item.entropy_score,
            "is_challenging": item.entropy_score is not None and item.entropy_score > 0.6,
            "similarity_score": similarity,
            "exact_match": is_exact_match,
        }

        if item.annotation_spans:
            metadata["annotation_spans"] = item.annotation_spans  # type: ignore

        if item.detector_predictions:
            metadata["detector_predictions"] = item.detector_predictions  # type: ignore

        return BenchmarkEvaluationResult(
            is_correct=is_correct,
            score=score,
            original_metric_score=similarity,
            explanation=(
                f"Evaluated {item.annotation_label.value if item.annotation_label else 'unknown'} "
                f"summary with similarity {similarity:.2f}"
            ),
            metadata=metadata,
        )

    def get_baseline_metrics(self) -> Dict[str, float]:
        """Return baseline metrics from the FaithBench paper.

        Returns:
            Dictionary of baseline metrics
        """
        return self.BASELINE_METRICS.copy()

    def get_original_prompts(self) -> List[str]:
        """Return example prompts from the FaithBench paper.

        Returns:
            List of example prompts
        """
        return [
            (
                "Please provide a concise summary of the following text:\n\n"
                "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
                "in Paris, France."
            ),
            (
                "Please provide a concise summary of the following text:\n\n"
                "Machine learning is a subset of artificial intelligence that enables systems "
                "to learn and improve from experience without being explicitly programmed."
            ),
            (
                "Please provide a concise summary of the following text:\n\n"
                "Climate change refers to long-term shifts in global temperatures and "
                "weather patterns."
            ),
        ]

    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate FaithBench-specific configuration.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        # Check model is supported
        model = config.get("model")
        if model and model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{model}' not supported for FaithBench. "
                f"Supported models: {', '.join(sorted(self.SUPPORTED_MODELS))}"
            )

        # Check temperature settings for reasoning models
        temperature = config.get("temperature")
        if model in self.NO_TEMPERATURE_MODELS and temperature is not None and temperature != 0.0:
            raise ValueError(
                f"Model '{model}' does not support temperature variation. "
                f"Temperature must be 0.0 or omitted."
            )

        # Validate sample size
        sample_size = config.get("sample_size")
        if sample_size is not None and sample_size <= 0:
            raise ValueError(f"Sample size must be positive, got {sample_size}")

        # Validate max tokens
        max_tokens = config.get("max_tokens")
        if max_tokens is not None and max_tokens < 50:
            raise ValueError(
                f"Max tokens too small for summarization: {max_tokens}. "
                f"Recommend at least 100 tokens."
            )

    @property
    def benchmark_name(self) -> str:
        """Return the benchmark identifier.

        Returns:
            Benchmark name
        """
        return "faithbench"

    @property
    def paper_reference(self) -> str:
        """Return the FaithBench paper reference.

        Returns:
            Paper citation
        """
        return (
            "FaithBench: A Benchmark for Hallucination Detection in Summarization "
            "with Challenging Samples (2024)"
        )

    def get_evaluation_method(self) -> str:
        """Return description of the evaluation methodology.

        Returns:
            Evaluation method description
        """
        return (
            "Evaluates hallucination detection in summarization using a 4-level taxonomy "
            "(consistent, questionable, benign, hallucinated) with focus on challenging "
            "samples identified by high entropy scores from detector disagreement."
        )

    def get_required_model_capabilities(self) -> Dict[str, Any]:
        """Get required model capabilities for FaithBench.

        Returns:
            Dictionary of required capabilities
        """
        return {
            "text_generation": True,
            "max_tokens": 150,  # Summaries are typically short
            "temperature_control": True,  # For most models
            "system_prompt": False,  # Not required for FaithBench
            "summarization": True,  # Key capability
        }

    def preprocess_response(self, response: str) -> str:
        """Preprocess model response before evaluation.

        Args:
            response: Raw model response

        Returns:
            Preprocessed response
        """
        # Remove common prefixes that models might add
        response = response.strip()

        # Remove common summary prefixes
        prefixes_to_remove = [
            "Summary:",
            "Summary: ",
            "Here's a summary:",
            "Here is a summary:",
            "The summary is:",
        ]

        for prefix in prefixes_to_remove:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix) :].strip()
                break

        return response

    def postprocess_results(self, results: List[BenchmarkEvaluationResult]) -> Dict[str, float]:
        """Aggregate results with FaithBench-specific metrics.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary of aggregated metrics
        """
        # Get base metrics from parent class
        base_metrics = super().postprocess_results(results)

        if not results:
            return base_metrics

        # Calculate comprehensive FaithBench metrics
        faithbench_metrics = self.metrics_calculator.calculate_aggregate_metrics(results)

        # Merge with base metrics
        base_metrics.update(faithbench_metrics)

        return base_metrics

    def calculate_detailed_metrics(
        self, results: List[BenchmarkEvaluationResult]
    ) -> Dict[str, Any]:
        """Calculate detailed FaithBench metrics with all categories.

        Args:
            results: List of evaluation results

        Returns:
            Structured dictionary with categorized metrics
        """
        # Calculate all metrics
        aggregate = self.metrics_calculator.calculate_aggregate_metrics(results)

        # Export in structured format
        return self.metrics_calculator.export_metrics(aggregate)

    def compare_with_paper_baseline(
        self, results: List[BenchmarkEvaluationResult]
    ) -> Dict[str, float]:
        """Compare results with baseline metrics from the FaithBench paper.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary containing metric comparisons
        """
        current_metrics = self.metrics_calculator.calculate_aggregate_metrics(results)

        # Use paper baselines for comparison
        baseline = {
            "overall_accuracy": self.BASELINE_METRICS.get("human_accuracy", 0.89),
            "challenging_accuracy": self.BASELINE_METRICS.get("average_entropy_challenging", 0.67),
        }

        return self.metrics_calculator.compare_with_baseline(current_metrics, baseline)
