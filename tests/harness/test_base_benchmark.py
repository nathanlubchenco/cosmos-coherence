"""Tests for the base benchmark framework."""

import json
from typing import Dict, List, Optional

import pytest
from cosmos_coherence.benchmarks.models.base import BaseDatasetItem
from cosmos_coherence.harness.base_benchmark import (
    BaseBenchmark,
    BenchmarkEvaluationResult,
    BenchmarkMetadata,
    OriginalMetrics,
)


class MockDatasetItem(BaseDatasetItem):
    """Mock dataset item for testing."""

    question: str
    answer: str
    context: Optional[str] = None

    def validate_content(self) -> bool:
        """Validate the content of this dataset item."""
        return bool(self.question and self.answer)


class MockBenchmark(BaseBenchmark):
    """Mock benchmark implementation for testing."""

    def __init__(self):
        super().__init__()
        from uuid import UUID

        self._dataset = [
            MockDatasetItem(
                id=UUID("00000000-0000-0000-0000-000000000001"),
                question="What is 2+2?",
                answer="4",
                context=None,
            ),
            MockDatasetItem(
                id=UUID("00000000-0000-0000-0000-000000000002"),
                question="What is the capital of France?",
                answer="Paris",
                context="France is a country in Europe.",
            ),
        ]

    async def load_dataset(self) -> List[BaseDatasetItem]:
        """Load mock dataset."""
        return self._dataset

    def get_prompt(self, item: BaseDatasetItem) -> str:
        """Format item into prompt."""
        mock_item = item  # type: ignore
        if hasattr(mock_item, "context") and mock_item.context:
            return f"Context: {mock_item.context}\nQuestion: {mock_item.question}"
        return f"Question: {mock_item.question}"

    def evaluate_response(
        self, response: str, ground_truth: str, item: BaseDatasetItem
    ) -> BenchmarkEvaluationResult:
        """Evaluate response against ground truth."""
        is_correct = response.strip().lower() == ground_truth.strip().lower()
        return BenchmarkEvaluationResult(
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            original_metric_score=1.0 if is_correct else 0.0,
            explanation=f"Exact match: {is_correct}",
            metadata={"response": response, "ground_truth": ground_truth},
        )

    def get_baseline_metrics(self) -> Dict[str, float]:
        """Return mock baseline metrics."""
        return {
            "accuracy": 0.85,
            "f1_score": 0.82,
            "exact_match_rate": 0.79,
        }

    def get_original_prompts(self) -> List[str]:
        """Return example prompts from original paper."""
        return [
            "Question: What is 2+2?",
            "Context: France is a country in Europe.\nQuestion: What is the capital of France?",
        ]

    def validate_config(self, config: Dict) -> None:
        """Validate benchmark configuration."""
        if "model" not in config:
            raise ValueError("Model must be specified in config")

    @property
    def benchmark_name(self) -> str:
        """Return benchmark name."""
        return "mock_benchmark"

    @property
    def paper_reference(self) -> str:
        """Return paper reference."""
        return "Mock et al. (2024). A Mock Benchmark for Testing."

    def get_evaluation_method(self) -> str:
        """Return evaluation method description."""
        return "Exact string match (case-insensitive)"


class TestBaseBenchmark:
    """Test the base benchmark functionality."""

    @pytest.fixture
    def mock_benchmark(self):
        """Create a mock benchmark instance."""
        return MockBenchmark()

    def test_benchmark_initialization(self, mock_benchmark):
        """Test that benchmark initializes correctly."""
        assert mock_benchmark.benchmark_name == "mock_benchmark"
        assert mock_benchmark.paper_reference == "Mock et al. (2024). A Mock Benchmark for Testing."
        assert mock_benchmark.get_evaluation_method() == "Exact string match (case-insensitive)"

    @pytest.mark.asyncio
    async def test_load_dataset(self, mock_benchmark):
        """Test dataset loading."""
        from uuid import UUID

        dataset = await mock_benchmark.load_dataset()
        assert len(dataset) == 2
        assert dataset[0].id == UUID("00000000-0000-0000-0000-000000000001")
        assert dataset[0].question == "What is 2+2?"  # type: ignore
        assert dataset[1].id == UUID("00000000-0000-0000-0000-000000000002")
        assert dataset[1].answer == "Paris"  # type: ignore

    def test_get_prompt_without_context(self, mock_benchmark):
        """Test prompt formatting without context."""
        item = MockDatasetItem(question="Test question?", answer="Test answer")
        prompt = mock_benchmark.get_prompt(item)
        assert prompt == "Question: Test question?"

    def test_get_prompt_with_context(self, mock_benchmark):
        """Test prompt formatting with context."""
        item = MockDatasetItem(
            question="Test question?",
            answer="Test answer",
            context="Test context.",
        )
        prompt = mock_benchmark.get_prompt(item)
        assert prompt == "Context: Test context.\nQuestion: Test question?"

    def test_evaluate_response_correct(self, mock_benchmark):
        """Test evaluation with correct response."""
        item = MockDatasetItem(question="What is 2+2?", answer="4")
        result = mock_benchmark.evaluate_response("4", "4", item)

        assert result.is_correct is True
        assert result.score == 1.0
        assert result.original_metric_score == 1.0
        assert "Exact match: True" in result.explanation

    def test_evaluate_response_incorrect(self, mock_benchmark):
        """Test evaluation with incorrect response."""
        item = MockDatasetItem(question="What is 2+2?", answer="4")
        result = mock_benchmark.evaluate_response("5", "4", item)

        assert result.is_correct is False
        assert result.score == 0.0
        assert result.original_metric_score == 0.0
        assert "Exact match: False" in result.explanation

    def test_evaluate_response_case_insensitive(self, mock_benchmark):
        """Test that evaluation is case-insensitive."""
        item = MockDatasetItem(question="Capital?", answer="Paris")
        result = mock_benchmark.evaluate_response("PARIS", "Paris", item)

        assert result.is_correct is True
        assert result.score == 1.0

    def test_get_baseline_metrics(self, mock_benchmark):
        """Test retrieving baseline metrics."""
        metrics = mock_benchmark.get_baseline_metrics()

        assert "accuracy" in metrics
        assert metrics["accuracy"] == 0.85
        assert "f1_score" in metrics
        assert metrics["f1_score"] == 0.82
        assert "exact_match_rate" in metrics
        assert metrics["exact_match_rate"] == 0.79

    def test_get_original_prompts(self, mock_benchmark):
        """Test retrieving original prompt examples."""
        prompts = mock_benchmark.get_original_prompts()

        assert len(prompts) == 2
        assert "Question: What is 2+2?" in prompts
        assert "Context: France is a country in Europe." in prompts[1]

    def test_validate_config_valid(self, mock_benchmark):
        """Test config validation with valid config."""
        config = {"model": "gpt-4o-mini", "temperature": 0.0}
        # Should not raise
        mock_benchmark.validate_config(config)

    def test_validate_config_invalid(self, mock_benchmark):
        """Test config validation with invalid config."""
        config = {"temperature": 0.0}  # Missing model
        with pytest.raises(ValueError, match="Model must be specified"):
            mock_benchmark.validate_config(config)

    def test_benchmark_metadata(self, mock_benchmark):
        """Test benchmark metadata generation."""
        metadata = mock_benchmark.get_metadata()

        assert isinstance(metadata, BenchmarkMetadata)
        assert metadata.name == "mock_benchmark"
        assert metadata.paper_reference == "Mock et al. (2024). A Mock Benchmark for Testing."
        assert metadata.evaluation_method == "Exact string match (case-insensitive)"
        assert metadata.baseline_metrics == mock_benchmark.get_baseline_metrics()

    def test_format_for_api(self, mock_benchmark):
        """Test formatting prompt for API call."""
        item = MockDatasetItem(question="Test?", answer="Answer")
        messages = mock_benchmark.format_for_api(item)

        assert isinstance(messages, list)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "Question: Test?" in messages[0]["content"]

    def test_format_for_api_with_system_prompt(self, mock_benchmark):
        """Test formatting with system prompt."""
        item = MockDatasetItem(question="Test?", answer="Answer")
        system_prompt = "You are a helpful assistant."
        messages = mock_benchmark.format_for_api(item, system_prompt=system_prompt)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == system_prompt
        assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_validate_dataset(self, mock_benchmark):
        """Test dataset validation."""
        is_valid, issues = await mock_benchmark.validate_dataset()

        assert is_valid is True
        assert len(issues) == 0

    @pytest.mark.asyncio
    async def test_get_dataset_statistics(self, mock_benchmark):
        """Test getting dataset statistics."""
        stats = await mock_benchmark.get_dataset_statistics()

        assert stats["total_items"] == 2
        assert stats["has_context"] == 1
        assert stats["no_context"] == 1
        assert stats["average_question_length"] > 0

    def test_abstract_class_enforcement(self):
        """Test that abstract methods must be implemented."""

        class IncompleteBenchmark(BaseBenchmark):
            """Incomplete benchmark missing required methods."""

            pass

        # Should not be able to instantiate
        with pytest.raises(TypeError):
            IncompleteBenchmark()

    def test_supports_original_evaluation(self, mock_benchmark):
        """Test that benchmark supports original evaluation metrics."""
        assert mock_benchmark.supports_original_evaluation() is True

    def test_get_required_model_capabilities(self, mock_benchmark):
        """Test getting required model capabilities."""
        capabilities = mock_benchmark.get_required_model_capabilities()

        assert "text_generation" in capabilities
        assert capabilities["max_tokens"] >= 100
        assert capabilities["temperature_control"] is True


class TestBenchmarkMetadata:
    """Test benchmark metadata functionality."""

    def test_metadata_creation(self):
        """Test creating benchmark metadata."""
        metadata = BenchmarkMetadata(
            name="test_benchmark",
            paper_reference="Test et al. (2024)",
            evaluation_method="F1 score",
            baseline_metrics={"f1": 0.85},
            dataset_size=1000,
            version="1.0.0",
        )

        assert metadata.name == "test_benchmark"
        assert metadata.paper_reference == "Test et al. (2024)"
        assert metadata.evaluation_method == "F1 score"
        assert metadata.baseline_metrics["f1"] == 0.85
        assert metadata.dataset_size == 1000
        assert metadata.version == "1.0.0"

    def test_metadata_serialization(self):
        """Test metadata serialization to JSON."""
        metadata = BenchmarkMetadata(
            name="test_benchmark",
            paper_reference="Test et al. (2024)",
            evaluation_method="Accuracy",
            baseline_metrics={"accuracy": 0.9},
        )

        json_str = metadata.model_dump_json()
        loaded = json.loads(json_str)

        assert loaded["name"] == "test_benchmark"
        assert loaded["baseline_metrics"]["accuracy"] == 0.9


class TestOriginalMetrics:
    """Test original metrics tracking."""

    def test_original_metrics_creation(self):
        """Test creating original metrics object."""
        metrics = OriginalMetrics(
            exact_match=0.85,
            f1_score=0.88,
            accuracy=0.86,
            additional_metrics={"bleu": 0.75, "rouge": 0.72},
        )

        assert metrics.exact_match == 0.85
        assert metrics.f1_score == 0.88
        assert metrics.accuracy == 0.86
        assert metrics.additional_metrics["bleu"] == 0.75

    def test_original_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = OriginalMetrics(
            exact_match=0.85, f1_score=0.88, additional_metrics={"custom": 0.9}
        )

        metrics_dict = metrics.to_dict()

        assert metrics_dict["exact_match"] == 0.85
        assert metrics_dict["f1_score"] == 0.88
        assert metrics_dict["custom"] == 0.9

    def test_original_metrics_comparison(self):
        """Test comparing original metrics."""
        metrics1 = OriginalMetrics(exact_match=0.85, f1_score=0.88)
        metrics2 = OriginalMetrics(exact_match=0.84, f1_score=0.87)

        diff = metrics1.compare_to(metrics2)

        assert diff["exact_match"] == pytest.approx(0.01)
        assert diff["f1_score"] == pytest.approx(0.01)
