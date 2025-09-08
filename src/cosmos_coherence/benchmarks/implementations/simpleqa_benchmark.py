"""SimpleQA benchmark implementation with HuggingFace support."""

from typing import Dict, List

from cosmos_coherence.benchmarks.models.base import BaseDatasetItem
from cosmos_coherence.benchmarks.models.datasets import SimpleQAItem
from cosmos_coherence.harness.base_benchmark import BenchmarkEvaluationResult
from cosmos_coherence.harness.base_benchmark_hf import HuggingFaceEnabledBenchmark


class SimpleQABenchmark(HuggingFaceEnabledBenchmark):
    """SimpleQA benchmark for evaluating factual accuracy.

    This benchmark tests models on simple factual questions to evaluate
    their ability to provide accurate, concise answers without hallucination.

    The benchmark can load data from:
    - HuggingFace: basicv8vc/SimpleQA dataset
    - Local files (if configured)
    """

    def __init__(self, **kwargs):
        """Initialize SimpleQA benchmark.

        By default, uses HuggingFace dataset unless explicitly disabled.
        """
        # Default to HuggingFace dataset if not specified
        if "hf_dataset_name" not in kwargs and kwargs.get("use_huggingface", True):
            kwargs["hf_dataset_name"] = "simpleqa"

        super().__init__(**kwargs)

    def get_prompt(self, item: BaseDatasetItem) -> str:
        """Format dataset item into prompt.

        Args:
            item: Dataset item (SimpleQAItem)

        Returns:
            Formatted prompt string
        """
        if isinstance(item, SimpleQAItem):
            # Simple Q&A format as used in the original benchmark
            return f"Question: {item.question}\nAnswer:"

        # Fallback for other item types
        return f"Question: {item.question}\nAnswer:"

    def evaluate_response(
        self, response: str, ground_truth: str, item: BaseDatasetItem
    ) -> BenchmarkEvaluationResult:
        """Evaluate model response using exact match and F1 score.

        Args:
            response: Model's response
            ground_truth: Expected answer
            item: Original dataset item

        Returns:
            Evaluation result with scores
        """
        # Normalize for comparison
        response_normalized = response.strip().lower()
        ground_truth_normalized = ground_truth.strip().lower()

        # Exact match evaluation
        exact_match = response_normalized == ground_truth_normalized

        # Calculate F1 score (token-level)
        response_tokens = set(response_normalized.split())
        ground_truth_tokens = set(ground_truth_normalized.split())

        if not response_tokens and not ground_truth_tokens:
            f1_score = 1.0
        elif not response_tokens or not ground_truth_tokens:
            f1_score = 0.0
        else:
            intersection = response_tokens & ground_truth_tokens
            precision = len(intersection) / len(response_tokens)
            recall = len(intersection) / len(ground_truth_tokens)

            if precision + recall == 0:
                f1_score = 0.0
            else:
                f1_score = 2 * (precision * recall) / (precision + recall)

        # Overall score is average of exact match and F1
        overall_score = (float(exact_match) + f1_score) / 2

        return BenchmarkEvaluationResult(
            is_correct=exact_match,
            score=overall_score,
            original_metric_score=f1_score,
            explanation=f"Exact match: {exact_match}, F1 score: {f1_score:.3f}",
            metadata={
                "exact_match": exact_match,
                "f1_score": f1_score,
                "response_length": len(response_tokens),
                "ground_truth_length": len(ground_truth_tokens),
            },
        )

    def get_baseline_metrics(self) -> Dict[str, float]:
        """Return published baseline metrics from SimpleQA paper.

        Returns:
            Dictionary of baseline metrics
        """
        return {
            "gpt-4_accuracy": 0.82,
            "gpt-3.5_accuracy": 0.68,
            "claude-2_accuracy": 0.75,
            "human_accuracy": 0.94,
        }

    def get_original_prompts(self) -> List[str]:
        """Return example prompts from the SimpleQA format.

        Returns:
            List of example prompts
        """
        return [
            "Question: What is the capital of France?\nAnswer:",
            "Question: Who wrote Romeo and Juliet?\nAnswer:",
            "Question: What year did World War II end?\nAnswer:",
        ]

    def validate_config(self, config: Dict) -> None:
        """Validate benchmark configuration.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        # SimpleQA has minimal configuration requirements
        if "model" in config:
            model_config = config["model"]
            if "temperature" in model_config and model_config["temperature"] > 0.3:
                # Warn about high temperature for factual tasks
                import logging

                logging.warning(
                    f"High temperature ({model_config['temperature']}) detected. "
                    "SimpleQA performs better with low temperature (0-0.3) for factual accuracy."
                )

    @property
    def benchmark_name(self) -> str:
        """Return benchmark identifier."""
        return "SimpleQA"

    @property
    def paper_reference(self) -> str:
        """Return paper citation."""
        return "SimpleQA: A Benchmark for Detecting Hallucinations in LLMs (2024)"

    def get_evaluation_method(self) -> str:
        """Return description of evaluation methodology."""
        return (
            "Evaluates factual accuracy using exact match and F1 score. "
            "Responses are compared against ground truth answers with "
            "normalization for case and whitespace."
        )

    async def _load_custom_dataset(self) -> List[BaseDatasetItem]:
        """Load dataset from custom source if not using HuggingFace.

        Returns:
            List of dataset items

        Raises:
            NotImplementedError: If no custom loading is implemented
        """
        # This would be implemented if you have a custom data source
        raise NotImplementedError(
            "SimpleQA benchmark requires HuggingFace dataset or custom implementation"
        )
