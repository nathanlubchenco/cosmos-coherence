"""SimpleQA benchmark implementation with HuggingFace support."""

from typing import TYPE_CHECKING, Dict, List, Optional

from cosmos_coherence.benchmarks.models.base import BaseDatasetItem
from cosmos_coherence.harness.base_benchmark import BenchmarkEvaluationResult
from cosmos_coherence.harness.base_benchmark_hf import HuggingFaceEnabledBenchmark
from cosmos_coherence.llm.openai_client import OpenAIClient

if TYPE_CHECKING:
    from cosmos_coherence.benchmarks.implementations.simpleqa_grader import SimpleQAGrader


class SimpleQABenchmark(HuggingFaceEnabledBenchmark):
    """SimpleQA benchmark for evaluating factual accuracy.

    This benchmark tests models on simple factual questions to evaluate
    their ability to provide accurate, concise answers without hallucination.

    The benchmark can load data from:
    - HuggingFace: basicv8vc/SimpleQA dataset
    - Local files (if configured)
    """

    def __init__(
        self, client: Optional[OpenAIClient] = None, use_ai_grading: bool = True, **kwargs
    ):
        """Initialize SimpleQA benchmark.

        By default, uses HuggingFace dataset unless explicitly disabled.

        Args:
            client: OpenAI client for AI-based grading (optional)
            use_ai_grading: Whether to use AI grading (True) or exact match (False)
            **kwargs: Additional arguments for parent class
        """
        # Extract use_huggingface flag if present (for compatibility)
        use_huggingface = kwargs.pop("use_huggingface", True)

        # Default to HuggingFace dataset if not specified and use_huggingface is True
        if "hf_dataset_name" not in kwargs and use_huggingface:
            kwargs["hf_dataset_name"] = "simpleqa"

        super().__init__(**kwargs)

        self.client = client
        self.use_ai_grading = use_ai_grading
        self._grader: Optional["SimpleQAGrader"] = None  # Lazy initialization

    def get_prompt(self, item: BaseDatasetItem) -> str:
        """Format dataset item into prompt.

        Following OpenAI's reference implementation, we send just the question
        without any additional formatting or instructions.

        Args:
            item: Dataset item (SimpleQAItem)

        Returns:
            The question as a string
        """
        # Match OpenAI reference: just send the question directly
        return item.question

    async def evaluate_response_with_ai(
        self, response: str, ground_truth: str, item: BaseDatasetItem
    ) -> BenchmarkEvaluationResult:
        """Evaluate model response using AI grading (OpenAI reference method).

        Args:
            response: Model's response
            ground_truth: Expected answer
            item: Original dataset item

        Returns:
            Evaluation result with scores
        """
        if not self.client:
            raise ValueError("AI grading requires an OpenAI client to be provided")

        # Lazy initialization of grader
        if not self._grader:
            from cosmos_coherence.benchmarks.implementations.simpleqa_grader import (
                SimpleQAGrader,
            )

            self._grader = SimpleQAGrader(self.client)

        # Get AI grade
        grade, metadata = await self._grader.grade_response(
            question=item.question, expert_answer=ground_truth, submission=response
        )

        is_correct = grade == "CORRECT"
        score = 1.0 if is_correct else 0.0

        return BenchmarkEvaluationResult(
            is_correct=is_correct,
            score=score,
            original_metric_score=score,
            explanation=f"AI Grade: {grade}",
            metadata={
                "grade": grade,
                "grading_metadata": metadata,
                "response_length": len(response.split()),
                "ground_truth_length": len(ground_truth.split()),
            },
        )

    def evaluate_response(
        self, response: str, ground_truth: str, item: BaseDatasetItem
    ) -> BenchmarkEvaluationResult:
        """Evaluate model response using exact match (fallback method).

        This is the synchronous fallback when AI grading is not available.

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

    def get_baseline_metrics(self) -> Dict:
        """Return published baseline metrics from SimpleQA paper.

        Returns:
            Dictionary of baseline metrics
        """
        return {
            "gpt-4_accuracy": 0.82,
            "gpt-3.5_accuracy": 0.68,
            "claude-2_accuracy": 0.75,
            "human_accuracy": 0.94,
            "paper_reference": "https://arxiv.org/abs/2410.02034",
            "benchmark_version": "v1.0",
        }

    def compare_to_baseline(self, model: str, accuracy: float) -> Dict:
        """Compare model accuracy to published baselines.

        Args:
            model: Model name (e.g., 'gpt-4', 'gpt-3.5-turbo')
            accuracy: Model's accuracy score (0-1)

        Returns:
            Dictionary with comparison results including:
            - model: Model name
            - your_score: Provided accuracy
            - baseline_score: Published baseline if available
            - difference: Difference from baseline
            - within_tolerance: Whether within 5% tolerance
        """
        baselines = self.get_baseline_metrics()

        # Normalize model name for comparison
        model_lower = model.lower()
        baseline_key = None
        baseline_score = None

        if "gpt-4" in model_lower:
            baseline_key = "gpt-4_accuracy"
            baseline_score = baselines.get(baseline_key)
        elif "gpt-3.5" in model_lower or "gpt-35" in model_lower:
            baseline_key = "gpt-3.5_accuracy"
            baseline_score = baselines.get(baseline_key)
        elif "claude-2" in model_lower:
            baseline_key = "claude-2_accuracy"
            baseline_score = baselines.get(baseline_key)

        result = {
            "model": model,
            "your_score": accuracy,
            "baseline_score": baseline_score,
            "difference": None,
            "within_tolerance": None,
        }

        if baseline_score is not None:
            difference = accuracy - baseline_score
            result["difference"] = difference
            # Within 5% tolerance (0.05 absolute difference) with small epsilon for float comparison
            result["within_tolerance"] = (
                abs(difference) <= 0.0501
            )  # Small epsilon for float precision

        return result

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
        if self.use_ai_grading:
            return (
                "Evaluates factual accuracy using AI-based grading "
                "(matching OpenAI reference). A grading model assesses whether responses "
                "are CORRECT, INCORRECT, or NOT_ATTEMPTED based on semantic meaning, "
                "allowing for variations in wording and minor typos."
            )
        else:
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
