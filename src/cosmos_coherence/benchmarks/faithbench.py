"""FaithBench benchmark implementation for hallucination detection in summarization."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from cosmos_coherence.benchmarks.faithbench_metrics import FaithBenchMetrics
from cosmos_coherence.benchmarks.models.base import BaseDatasetItem
from cosmos_coherence.benchmarks.models.datasets import FaithBenchAnnotation, FaithBenchItem
from cosmos_coherence.harness.base_benchmark import (
    BaseBenchmark,
    BenchmarkEvaluationResult,
    BenchmarkMetadata,
)
from cosmos_coherence.llm.config import OpenAIConfig, RateLimitConfig
from cosmos_coherence.llm.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class FaithBenchBenchmark(BaseBenchmark):
    """FaithBench benchmark for evaluating hallucination detection in summarization.

    FaithBench focuses on "challenging" samples where multiple detectors disagree,
    using entropy scores to identify these challenging cases.
    """

    # Supported models for evaluation
    SUPPORTED_MODELS: Set[str] = {
        "gpt-4-turbo",
        "gpt-4o",
        "gpt-3.5-turbo",
        "claude-3-opus",
        "claude-3-sonnet",
        "llama-3-70b",
        "gemini-1.5-pro",
        "mistral-large",
    }

    # Models that don't support temperature variation
    NO_TEMPERATURE_MODELS: Set[str] = set()

    # Actual metrics from the FaithBench paper (Table 3)
    # These are balanced accuracy scores for hallucination detection
    BASELINE_METRICS = {
        "gpt-4-turbo_accuracy": 0.5765,  # 57.65% balanced accuracy
        "gpt-4o_accuracy": 0.5629,  # 56.29% balanced accuracy
        "gpt-3.5-turbo_accuracy": 0.4491,  # 44.91% balanced accuracy
        "gpt-4-turbo_f1": 0.4361,  # 43.61% F1-macro
        "gpt-4o_f1": 0.4075,  # 40.75% F1-macro
        "gpt-3.5-turbo_f1": 0.3741,  # 37.41% F1-macro
        "human_accuracy": None,  # Not reported in the paper
        "average_entropy_challenging": 0.67,  # Threshold for challenging samples
    }

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize FaithBench benchmark.

        Args:
            cache_dir: Optional directory for caching datasets
            api_key: OpenAI API key for evaluations
        """
        super().__init__()
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./data/faithbench_cache")
        self.metrics_calculator = FaithBenchMetrics()
        self._dataset_cache: Optional[List[FaithBenchItem]] = None

        # Initialize OpenAI client if API key provided
        self.openai_client: Optional[OpenAIClient] = None
        if api_key:
            config = OpenAIConfig(api_key=api_key)  # type: ignore

            # Check for cache configuration from environment
            cache_path = os.environ.get("COSMOS_CACHE_DIR")
            # If COSMOS_CACHE_DIR is set, use it directly as the cache file path
            cache_file = Path(cache_path) if cache_path else None

            self.openai_client = OpenAIClient(
                openai_config=config,
                rate_limit_config=RateLimitConfig(  # type: ignore
                    requests_per_minute=50,
                    max_concurrent=5,
                ),
                enable_cache=True,  # Enable caching by default
                cache_file=cache_file,  # Use configured cache file if available
            )

    async def load_dataset(self, sample_size: Optional[int] = None) -> List[BaseDatasetItem]:
        """Load FaithBench dataset from GitHub.

        Note: FaithBench is not on HuggingFace. It's available from:
        https://github.com/vectara/FaithBench

        Args:
            sample_size: Optional number of samples to load

        Returns:
            List of FaithBenchItem instances
        """
        if self._dataset_cache is None:
            logger.info("Loading FaithBench dataset from GitHub")

            # For now, create mock data with the correct structure
            # In production, download from: https://github.com/vectara/FaithBench/tree/main/data_for_release
            import json

            import aiohttp

            # FaithBench has 16 batch files (1-16, except 13)
            batch_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16]
            all_items = []

            # GitHub raw URL base
            base_url = "https://raw.githubusercontent.com/vectara/FaithBench/main/data_for_release"

            async with aiohttp.ClientSession() as session:
                for batch_id in batch_ids:
                    url = f"{base_url}/batch_{batch_id}.json"
                    try:
                        async with session.get(url) as response:
                            if response.status == 200:
                                # GitHub raw returns text/plain, so we need to parse it manually
                                text = await response.text()
                                data = json.loads(text)
                                # Data has "samples" key at top level
                                samples = data.get("samples", [])
                                for item in samples:
                                    # Convert to FaithBenchItem
                                    faithbench_item = self._convert_raw_item(item)
                                    if faithbench_item:
                                        all_items.append(faithbench_item)
                    except Exception as e:
                        logger.error(f"Failed to load batch {batch_id}: {e}")

            if not all_items:
                raise ValueError(
                    "Failed to load FaithBench dataset from GitHub. "
                    "Please check your internet connection and try again."
                )

            self._dataset_cache = all_items
            logger.info(f"Loaded {len(self._dataset_cache)} FaithBench items")

        # Apply sample size if specified
        if sample_size and self._dataset_cache:
            return self._dataset_cache[:sample_size]  # type: ignore
        return self._dataset_cache  # type: ignore

    def get_prompt(self, item: BaseDatasetItem) -> str:
        """Generate summarization prompt for FaithBench item.

        FaithBench is about detecting hallucinations in existing summaries.

        Args:
            item: FaithBenchItem with source and summary to evaluate

        Returns:
            Formatted prompt for hallucination detection
        """
        # FaithBench evaluates existing summaries for hallucinations
        # The prompt should ask to evaluate if the summary is consistent with the source
        if not isinstance(item, FaithBenchItem):
            raise ValueError(f"Expected FaithBenchItem, got {type(item).__name__}")

        prompt = (
            f"Decide if the following summary is consistent with the corresponding article. "
            f"Note that consistency means all information in the summary is supported "
            f"by the article.\n"
            f"Article: {item.source}\n"
            f"Summary: {item.summary}\n"
            f"Answer (Yes or No):"
        )
        return prompt

    def _convert_raw_item(self, raw_item: Dict[str, Any]) -> Optional[FaithBenchItem]:
        """Convert raw FaithBench JSON item to FaithBenchItem."""
        try:
            # Map the actual FaithBench structure
            annotations = raw_item.get("annotations", [])

            # Determine annotation label from annotations
            if not annotations:
                label = FaithBenchAnnotation.CONSISTENT
            else:
                # Check annotation labels - they are lists like ["Unwanted", "Unwanted.Intrinsic"]
                has_unwanted = False
                has_questionable = False
                has_benign = False

                for annot in annotations:
                    labels = annot.get("label", [])
                    if isinstance(labels, list) and labels:
                        main_label = labels[0].split(".")[0]  # Get main label before dot
                        if main_label == "Unwanted":
                            has_unwanted = True
                        elif main_label == "Questionable":
                            has_questionable = True
                        elif main_label == "Benign":
                            has_benign = True

                if has_unwanted:
                    label = FaithBenchAnnotation.HALLUCINATED
                elif has_questionable:
                    label = FaithBenchAnnotation.QUESTIONABLE
                elif has_benign:
                    label = FaithBenchAnnotation.BENIGN
                else:
                    label = FaithBenchAnnotation.CONSISTENT

            # Collect annotation spans
            annotation_spans = []
            for annot in annotations:
                if annot.get("summary_span"):
                    annotation_spans.append(annot["summary_span"])

            # Get metadata for detector predictions
            metadata = raw_item.get("metadata", {})
            detector_preds = {}

            if metadata:
                # Collect detector predictions
                detectors = [
                    "hhemv1",
                    "hhem-2.1",
                    "trueteacher",
                    "true_nli",
                    "gpt-3.5-turbo",
                    "gpt-4-turbo",
                    "gpt_4o",
                ]
                for detector in detectors:
                    if detector in metadata:
                        detector_preds[detector] = metadata[detector]

            # Calculate entropy score from detector predictions if not provided
            entropy_score = None
            if detector_preds:
                # Simple entropy calculation based on agreement
                values = list(detector_preds.values())
                if values:
                    # Count how many detectors say it's hallucinated (score < 0.5 or 0)
                    hall_count = sum(1 for v in values if v < 0.5)
                    total = len(values)
                    # Entropy is higher when there's disagreement
                    p_hall = hall_count / total if total > 0 else 0
                    p_cons = 1 - p_hall
                    if p_hall > 0 and p_cons > 0:
                        import math

                        entropy_score = -(p_hall * math.log2(p_hall) + p_cons * math.log2(p_cons))
                    else:
                        entropy_score = 0.0

            return FaithBenchItem(
                sample_id=str(raw_item.get("sample_id", "")),
                source=raw_item.get("source", ""),
                summary=raw_item.get("summary", ""),
                annotation_label=label,
                annotation_spans=annotation_spans,
                annotation_justification=raw_item.get("annotation_justification"),
                entropy_score=entropy_score,
                detector_predictions=detector_preds,
                question="Summarize the text.",  # FaithBench is summarization-focused
                summarizer_model=metadata.get("summarizer") if metadata else None,
            )
        except Exception as e:
            logger.error(f"Failed to convert item: {e}")
            return None

    async def evaluate_response_with_llm(
        self, response: str, source: str, model: str = "gpt-4-turbo"
    ) -> bool:
        """Evaluate if a summary contains hallucinations using LLM.

        This implements the actual FaithBench evaluation methodology.

        Args:
            response: Generated summary to evaluate
            source: Original source text
            model: Model to use for evaluation

        Returns:
            True if summary is consistent (no hallucination), False otherwise
        """
        if not self.openai_client:
            raise ValueError(
                "OpenAI client not configured. Please provide an API key when "
                "initializing FaithBenchBenchmark."
            )

        # Use the exact prompt from the paper (https://arxiv.org/pdf/2303.15621)
        prompt = (
            f"Decide if the following summary is consistent with the corresponding article. "
            f"Note that consistency means all information in the summary is supported "
            f"by the article.\n"
            f"Article: {source}\n"
            f"Summary: {response}\n"
            f"Answer (Yes or No):"
        )

        try:
            # Use the OpenAI client to evaluate
            result = await self.openai_client.generate_response(
                prompt=prompt,
                model=model,
                temperature=0.0,  # Deterministic for evaluation
                max_tokens=10,
            )

            # Parse the response - now asking if consistent, so "yes" = consistent
            answer = result.content.strip().lower() if result else "no"
            return "yes" in answer  # Yes = consistent (no hallucination)

        except Exception as e:
            logger.error(f"Failed to evaluate with LLM: {e}")
            return False

    def evaluate_response(
        self, response: str, ground_truth: str, item: BaseDatasetItem
    ) -> BenchmarkEvaluationResult:
        """Evaluate model response for hallucination detection.

        Args:
            response: Model's response to hallucination detection prompt (e.g., "yes" or "no")
            ground_truth: Not used for FaithBench (we use annotation_label instead)
            item: Original FaithBenchItem with annotation label

        Returns:
            Evaluation result with hallucination detection scores
        """
        # Cast to FaithBenchItem to access specific fields
        if not isinstance(item, FaithBenchItem):
            raise ValueError(f"Expected FaithBenchItem, got {type(item).__name__}")

        # Parse the model's response - looking for yes/no answer
        response = response.strip().lower()

        # Model predicts "yes" = consistent (no hallucination)
        # Model predicts "no" = inconsistent (has hallucination)
        model_predicts_consistent = "yes" in response

        # Map ground truth labels to binary classification
        # According to paper Table 2:
        # Positive class (no unwanted hallucination): benign + consistent
        # Negative class (has unwanted content): unwanted (hallucinated) + questionable

        # Initialize ground_truth_consistent to handle None case
        ground_truth_consistent = True

        if item.annotation_label:
            if item.annotation_label in [
                FaithBenchAnnotation.CONSISTENT,
                FaithBenchAnnotation.BENIGN,
            ]:
                # Ground truth: No unwanted hallucination (positive class)
                ground_truth_consistent = True
                is_correct = model_predicts_consistent == ground_truth_consistent
            elif item.annotation_label in [
                FaithBenchAnnotation.HALLUCINATED,
                FaithBenchAnnotation.QUESTIONABLE,
            ]:
                # Ground truth: Has unwanted content (negative class)
                ground_truth_consistent = False
                is_correct = model_predicts_consistent == ground_truth_consistent
            else:
                # Unknown label, treat as consistent
                ground_truth_consistent = True
                is_correct = model_predicts_consistent == ground_truth_consistent
        else:
            # No annotation label, assume consistent
            ground_truth_consistent = True
            is_correct = model_predicts_consistent == ground_truth_consistent

        # Score based on confidence (1.0 for correct, 0.0 for incorrect)
        score = 1.0 if is_correct else 0.0

        # Build metadata
        metadata = {
            "annotation_label": item.annotation_label.value if item.annotation_label else None,
            "entropy_score": item.entropy_score,
            "is_challenging": item.entropy_score is not None and item.entropy_score > 0.6,
            "model_prediction": "consistent" if model_predicts_consistent else "hallucinated",
            "ground_truth_binary": "consistent" if ground_truth_consistent else "hallucinated",
            "detected_hallucination": not model_predicts_consistent,
        }

        if item.annotation_spans:
            metadata["annotation_spans"] = item.annotation_spans  # type: ignore

        if item.detector_predictions:
            metadata["detector_predictions"] = item.detector_predictions  # type: ignore

        return BenchmarkEvaluationResult(
            is_correct=is_correct,
            score=score,
            original_metric_score=score,
            explanation=(
                f"Model predicted {'consistent' if model_predicts_consistent else 'hallucinated'} "
                f"for {item.annotation_label.value if item.annotation_label else 'unknown'} label. "
                f"Binary GT: {'consistent' if ground_truth_consistent else 'hallucinated'}. "
                f"Correct: {is_correct}"
            ),
            metadata=metadata,
        )

    def get_baseline_metrics(self) -> Dict[str, float]:
        """Return baseline metrics from the FaithBench paper.

        Returns:
            Dictionary of baseline metrics
        """
        # Filter out None values for compatibility with BenchmarkMetadata
        return {k: v for k, v in self.BASELINE_METRICS.items() if v is not None}

    def get_metadata(self) -> BenchmarkMetadata:
        """Get benchmark metadata.

        Returns:
            BenchmarkMetadata instance with benchmark information
        """
        return BenchmarkMetadata(
            name=self.benchmark_name,
            paper_reference=self.paper_reference,
            evaluation_method=self.get_evaluation_method(),
            baseline_metrics=self.get_baseline_metrics(),
        )

    def get_original_prompts(self) -> List[str]:
        """Return example prompts from the FaithBench paper.

        Returns:
            List of example prompts
        """
        return [
            (
                "Decide if the following summary is consistent with the corresponding article. "
                "Note that consistency means all information in the summary is supported "
                "by the article.\n"
                "Article: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
                "in Paris, France.\n"
                "Summary: The Eiffel Tower is located in Paris, France.\n"
                "Answer (Yes or No):"
            ),
            (
                "Decide if the following summary is consistent with the corresponding article. "
                "Note that consistency means all information in the summary is supported "
                "by the article.\n"
                "Article: Machine learning is a subset of AI that enables "
                "systems to learn and improve without being explicitly programmed.\n"
                "Summary: Machine learning requires explicit programming for every task.\n"
                "Answer (Yes or No):"
            ),
            (
                "Decide if the following summary is consistent with the corresponding article. "
                "Note that consistency means all information in the summary is supported "
                "by the article.\n"
                "Article: Climate change refers to long-term shifts in global temperatures and "
                "weather patterns.\n"
                "Summary: Climate change causes immediate daily weather changes.\n"
                "Answer (Yes or No):"
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

        # Validate temperature range
        temperature = config.get("temperature")
        if temperature is not None:
            if not 0.0 <= temperature <= 2.0:
                raise ValueError(f"Temperature must be between 0.0 and 2.0, got {temperature}")

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
            "Bao et al. FaithBench: A Diverse Hallucination Benchmark for "
            "Summarization by Modern LLMs. arXiv:2410.13210 (2024)"
        )

    def get_evaluation_method(self) -> str:
        """Return description of the evaluation methodology.

        Returns:
            Evaluation method description
        """
        return (
            "Binary classification of summaries as hallucinated or consistent. "
            "FaithBench focuses on challenging samples with high entropy scores "
            "(>0.67) where multiple SOTA detectors disagree. Human expert annotations "
            "provide ground truth labels using a nuanced taxonomy: Consistent, "
            "Questionable, Benign, and Unwanted (hallucinated)."
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
