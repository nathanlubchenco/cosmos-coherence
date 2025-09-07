"""Base benchmark framework for reproducible evaluation."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from cosmos_coherence.benchmarks.models.base import BaseDatasetItem

logger = logging.getLogger(__name__)


class BenchmarkEvaluationResult(BaseModel):
    """Result of evaluating a single response."""

    is_correct: bool
    score: float = Field(ge=0.0, le=1.0)
    original_metric_score: float = Field(
        ge=0.0, le=1.0, description="Score using original paper's metric"
    )
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkMetadata(BaseModel):
    """Metadata about a benchmark."""

    name: str
    paper_reference: str
    evaluation_method: str
    baseline_metrics: Dict[str, float]
    dataset_size: Optional[int] = None
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.now)


class OriginalMetrics(BaseModel):
    """Original metrics from benchmark paper."""

    exact_match: Optional[float] = None
    f1_score: Optional[float] = None
    accuracy: Optional[float] = None
    additional_metrics: Dict[str, float] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary."""
        result = {}
        if self.exact_match is not None:
            result["exact_match"] = self.exact_match
        if self.f1_score is not None:
            result["f1_score"] = self.f1_score
        if self.accuracy is not None:
            result["accuracy"] = self.accuracy
        result.update(self.additional_metrics)
        return result

    def compare_to(self, other: "OriginalMetrics") -> Dict[str, float]:
        """Compare to another set of metrics."""
        diff = {}
        if self.exact_match is not None and other.exact_match is not None:
            diff["exact_match"] = self.exact_match - other.exact_match
        if self.f1_score is not None and other.f1_score is not None:
            diff["f1_score"] = self.f1_score - other.f1_score
        if self.accuracy is not None and other.accuracy is not None:
            diff["accuracy"] = self.accuracy - other.accuracy

        # Compare additional metrics
        for key in self.additional_metrics:
            if key in other.additional_metrics:
                diff[key] = self.additional_metrics[key] - other.additional_metrics[key]

        return diff


class BaseBenchmark(ABC):
    """Abstract base class for benchmark implementations."""

    def __init__(self):
        """Initialize the benchmark."""
        self._metadata: Optional[BenchmarkMetadata] = None
        self._dataset: Optional[List[BaseDatasetItem]] = None
        logger.info(f"Initializing benchmark: {self.benchmark_name}")

    @abstractmethod
    async def load_dataset(self) -> List[BaseDatasetItem]:
        """Load and return the benchmark dataset.

        Returns:
            List of dataset items
        """
        pass

    @abstractmethod
    def get_prompt(self, item: BaseDatasetItem) -> str:
        """Format dataset item into LLM prompt using original benchmark format.

        Args:
            item: Dataset item to format

        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def evaluate_response(
        self, response: str, ground_truth: str, item: BaseDatasetItem
    ) -> BenchmarkEvaluationResult:
        """Evaluate model response using original benchmark metrics.

        Args:
            response: Model's response
            ground_truth: Expected answer
            item: Original dataset item

        Returns:
            Evaluation result with scores
        """
        pass

    @abstractmethod
    def get_baseline_metrics(self) -> Dict[str, float]:
        """Return published baseline metrics for reproducibility validation.

        Returns:
            Dictionary of metric names to values
        """
        pass

    @abstractmethod
    def get_original_prompts(self) -> List[str]:
        """Return example prompts from original paper for format validation.

        Returns:
            List of example prompts
        """
        pass

    @abstractmethod
    def validate_config(self, config: Dict) -> None:
        """Validate benchmark-specific configuration.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    @property
    @abstractmethod
    def benchmark_name(self) -> str:
        """Return the benchmark identifier.

        Returns:
            Benchmark name
        """
        pass

    @property
    @abstractmethod
    def paper_reference(self) -> str:
        """Return the original paper reference for this benchmark.

        Returns:
            Paper citation
        """
        pass

    @abstractmethod
    def get_evaluation_method(self) -> str:
        """Return description of the evaluation methodology.

        Returns:
            Evaluation method description
        """
        pass

    def get_metadata(self) -> BenchmarkMetadata:
        """Get benchmark metadata.

        Returns:
            Benchmark metadata object
        """
        if self._metadata is None:
            dataset_size = None
            if self._dataset is not None:
                dataset_size = len(self._dataset)

            self._metadata = BenchmarkMetadata(
                name=self.benchmark_name,
                paper_reference=self.paper_reference,
                evaluation_method=self.get_evaluation_method(),
                baseline_metrics=self.get_baseline_metrics(),
                dataset_size=dataset_size,
            )

        return self._metadata

    def format_for_api(
        self, item: BaseDatasetItem, system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Format item for API call.

        Args:
            item: Dataset item
            system_prompt: Optional system prompt

        Returns:
            List of message dictionaries for API
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_prompt = self.get_prompt(item)
        messages.append({"role": "user", "content": user_prompt})

        return messages

    async def validate_dataset(self) -> Tuple[bool, List[str]]:
        """Validate the dataset for consistency.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        try:
            dataset = await self.load_dataset()

            if not dataset:
                issues.append("Dataset is empty")
                return False, issues

            # Check for required fields
            for i, item in enumerate(dataset):
                if not hasattr(item, "id"):
                    issues.append(f"Item {i} missing id")
                if not hasattr(item, "question"):
                    issues.append(f"Item {i} missing question")

            # Check prompt generation
            try:
                sample_prompt = self.get_prompt(dataset[0])
                if not sample_prompt:
                    issues.append("Generated prompt is empty")
            except Exception as e:
                issues.append(f"Error generating prompt: {e}")

            # Check baseline metrics
            baseline = self.get_baseline_metrics()
            if not baseline:
                issues.append("No baseline metrics provided")

        except Exception as e:
            issues.append(f"Error loading dataset: {e}")
            return False, issues

        return len(issues) == 0, issues

    async def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get statistics about the dataset.

        Returns:
            Dictionary of statistics
        """
        dataset = await self.load_dataset()

        if not dataset:
            return {"total_items": 0}

        stats: Dict[str, Any] = {
            "total_items": len(dataset),
            "has_context": 0,
            "no_context": 0,
            "average_question_length": 0.0,
            "average_answer_length": 0.0,
        }

        total_q_length = 0
        total_a_length = 0

        for item in dataset:
            # Check for context
            if hasattr(item, "context") and item.context:
                stats["has_context"] += 1
            else:
                stats["no_context"] += 1

            # Calculate lengths
            if hasattr(item, "question"):
                total_q_length += len(str(item.question))
            if hasattr(item, "answer"):
                total_a_length += len(str(item.answer))

        if dataset:
            stats["average_question_length"] = total_q_length / len(dataset)
            stats["average_answer_length"] = total_a_length / len(dataset)

        return stats

    def supports_original_evaluation(self) -> bool:
        """Check if benchmark supports original evaluation metrics.

        Returns:
            True if original evaluation is supported
        """
        return True

    def get_required_model_capabilities(self) -> Dict[str, Any]:
        """Get required model capabilities for this benchmark.

        Returns:
            Dictionary of required capabilities
        """
        return {
            "text_generation": True,
            "max_tokens": 2048,
            "temperature_control": True,
            "system_prompt": False,
        }

    def preprocess_response(self, response: str) -> str:
        """Preprocess model response before evaluation.

        Args:
            response: Raw model response

        Returns:
            Preprocessed response
        """
        # Default: strip whitespace
        return response.strip()

    def postprocess_results(self, results: List[BenchmarkEvaluationResult]) -> Dict[str, float]:
        """Aggregate individual results into summary metrics.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary of aggregated metrics
        """
        if not results:
            return {}

        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        total_score = sum(r.score for r in results)
        total_original = sum(r.original_metric_score for r in results)

        return {
            "accuracy": correct / total,
            "average_score": total_score / total,
            "average_original_score": total_original / total,
            "total_evaluated": total,
            "total_correct": correct,
        }
