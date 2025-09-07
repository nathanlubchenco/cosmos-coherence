"""Tests for the benchmark runner system."""

import asyncio
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from cosmos_coherence.benchmarks.models.base import BaseDatasetItem
from cosmos_coherence.harness.base_benchmark import (
    BaseBenchmark,
    BenchmarkEvaluationResult,
)
from cosmos_coherence.harness.benchmark_runner import (
    BenchmarkRunner,
    ExecutionConfig,
    ExecutionResult,
    ProgressTracker,
    RunnerError,
)


class MockDatasetItem(BaseDatasetItem):
    """Mock dataset item for testing."""

    question: str
    answer: str
    context: str = ""

    def validate_content(self) -> bool:
        """Validate the content."""
        return bool(self.question and self.answer)


class MockBenchmark(BaseBenchmark):
    """Mock benchmark for testing."""

    def __init__(self, dataset_size: int = 10):
        super().__init__()
        self.dataset_size = dataset_size
        self.load_dataset_called = False
        self.evaluate_count = 0

    async def load_dataset(self) -> List[BaseDatasetItem]:
        """Load mock dataset."""
        self.load_dataset_called = True
        return [
            MockDatasetItem(
                id=UUID(f"00000000-0000-0000-0000-{i:012d}"),
                question=f"Question {i}",
                answer=f"Answer {i}",
            )
            for i in range(self.dataset_size)
        ]

    def get_prompt(self, item: BaseDatasetItem) -> str:
        """Get prompt for item."""
        return f"Q: {item.question}"

    def evaluate_response(
        self, response: str, ground_truth: str, item: BaseDatasetItem
    ) -> BenchmarkEvaluationResult:
        """Evaluate response."""
        self.evaluate_count += 1
        is_correct = response.strip().lower() == ground_truth.strip().lower()
        return BenchmarkEvaluationResult(
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            original_metric_score=1.0 if is_correct else 0.0,
            explanation=f"Match: {is_correct}",
        )

    def get_baseline_metrics(self) -> Dict[str, float]:
        """Get baseline metrics."""
        return {"accuracy": 0.85, "f1_score": 0.82}

    def get_original_prompts(self) -> List[str]:
        """Get original prompts."""
        return ["Q: Question 0", "Q: Question 1"]

    def validate_config(self, config: Dict) -> None:
        """Validate config."""
        if "model" not in config:
            raise ValueError("Model required")

    @property
    def benchmark_name(self) -> str:
        """Get benchmark name."""
        return "mock_benchmark"

    @property
    def paper_reference(self) -> str:
        """Get paper reference."""
        return "Mock et al. (2024)"

    def get_evaluation_method(self) -> str:
        """Get evaluation method."""
        return "Exact match"


class TestBenchmarkRunner:
    """Test the benchmark runner functionality."""

    @pytest.fixture
    def mock_benchmark(self):
        """Create a mock benchmark."""
        return MockBenchmark(dataset_size=5)

    @pytest.fixture
    def execution_config(self):
        """Create execution config."""
        return ExecutionConfig(
            max_parallel=2,
            timeout_seconds=30,
            retry_attempts=2,
            temperature=0.7,
            save_results=True,
            results_dir=Path("/tmp/results"),
        )

    @pytest.fixture
    def runner(self, mock_benchmark, execution_config):
        """Create a benchmark runner."""
        return BenchmarkRunner(mock_benchmark, execution_config)

    @pytest.mark.asyncio
    async def test_runner_initialization(self, runner, mock_benchmark, execution_config):
        """Test runner initialization."""
        assert runner.benchmark == mock_benchmark
        assert runner.config == execution_config
        assert runner.context is not None
        assert runner.progress is not None

    @pytest.mark.asyncio
    async def test_load_dataset(self, runner):
        """Test dataset loading."""
        dataset = await runner.load_dataset()
        assert len(dataset) == 5
        assert runner.benchmark.load_dataset_called
        assert all(isinstance(item, MockDatasetItem) for item in dataset)

    @pytest.mark.asyncio
    async def test_process_single_item(self, runner):
        """Test processing a single item."""
        item = MockDatasetItem(question="Test question?", answer="Test answer")

        # Mock the model call
        with patch.object(runner, "call_model", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "Test answer"

            result = await runner.process_item(item)

            assert result.item_id == item.id
            assert result.prediction == "Test answer"
            assert result.ground_truth == "Test answer"
            assert result.evaluation.is_correct is True
            assert result.evaluation.score == 1.0

    @pytest.mark.asyncio
    async def test_process_item_with_error(self, runner):
        """Test processing item with error."""
        item = MockDatasetItem(question="Test question?", answer="Test answer")

        # Mock the model call to raise an error
        with patch.object(runner, "call_model", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = Exception("API Error")

            result = await runner.process_item(item)

            assert result.item_id == item.id
            assert result.error == "API Error"
            assert result.evaluation is None

    @pytest.mark.asyncio
    async def test_process_batch(self, runner):
        """Test processing a batch of items."""
        items = [
            MockDatasetItem(
                id=UUID(f"00000000-0000-0000-0000-{i:012d}"),
                question=f"Question {i}",
                answer=f"Answer {i}",
            )
            for i in range(3)
        ]

        with patch.object(runner, "call_model", new_callable=AsyncMock) as mock_call:
            mock_call.side_effect = ["Answer 0", "Wrong", "Answer 2"]

            results = await runner.process_batch(items)

            assert len(results) == 3
            assert results[0].evaluation.is_correct is True
            assert results[1].evaluation.is_correct is False
            assert results[2].evaluation.is_correct is True

    @pytest.mark.asyncio
    async def test_run_benchmark_full(self, runner):
        """Test running full benchmark."""
        with patch.object(runner, "call_model", new_callable=AsyncMock) as mock_call:
            # Return correct answers for 3 out of 5 items
            mock_call.side_effect = ["Answer 0", "Wrong", "Answer 2", "Wrong", "Answer 4"]

            results = await runner.run()

            assert len(results.item_results) == 5
            assert results.total_items == 5
            assert results.successful_items == 5
            assert results.failed_items == 0

            # Check metrics
            assert results.metrics["accuracy"] == 0.6  # 3 out of 5 correct
            assert results.metrics["average_score"] == 0.6

    @pytest.mark.asyncio
    async def test_run_with_progress_tracking(self, runner):
        """Test progress tracking during run."""
        progress_updates = []

        def progress_callback(tracker: ProgressTracker):
            progress_updates.append(
                {
                    "processed": tracker.processed_items,
                    "total": tracker.total_items,
                    "percentage": tracker.get_percentage(),
                }
            )

        runner.progress.add_callback(progress_callback)

        with patch.object(runner, "call_model", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "Answer"
            await runner.run()

        # Should have progress updates
        assert len(progress_updates) > 0
        assert progress_updates[-1]["percentage"] == 100.0

    @pytest.mark.asyncio
    async def test_parallel_execution(self, execution_config):
        """Test parallel execution of items."""
        execution_config.max_parallel = 3
        benchmark = MockBenchmark(dataset_size=10)
        runner = BenchmarkRunner(benchmark, execution_config)

        call_times = []

        async def mock_model_call(prompt: str) -> str:
            """Mock model call with delay."""
            start = asyncio.get_event_loop().time()
            await asyncio.sleep(0.1)  # Simulate API delay
            call_times.append(asyncio.get_event_loop().time() - start)
            return "Answer"

        with patch.object(runner, "call_model", side_effect=mock_model_call):
            results = await runner.run()

        # With parallelism, total time should be less than sequential
        assert len(results.item_results) == 10
        # Check that multiple calls overlapped (parallel execution)
        assert results.execution_time < 1.0  # Should be much less than 10 * 0.1

    @pytest.mark.asyncio
    async def test_retry_logic(self, runner):
        """Test retry logic on failures."""
        item = MockDatasetItem(question="Test?", answer="Answer")

        call_count = 0

        async def flaky_model_call(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return "Answer"

        # Patch the model_caller directly to test retry in call_model
        runner.model_caller = flaky_model_call
        result = await runner.process_item(item)

        assert call_count == 2  # First attempt failed, second succeeded
        assert result.evaluation.is_correct is True

    @pytest.mark.asyncio
    async def test_timeout_handling(self, execution_config):
        """Test timeout handling."""
        execution_config.timeout_seconds = 0.1
        benchmark = MockBenchmark(dataset_size=1)
        runner = BenchmarkRunner(benchmark, execution_config)

        async def slow_model_call(prompt: str) -> str:
            await asyncio.sleep(1.0)  # Longer than timeout
            return "Answer"

        # Replace the model caller to test actual timeout
        runner.model_caller = slow_model_call
        results = await runner.run()

        # Check that item has an error (timeout)
        assert results.item_results[0].error is not None
        assert (
            "timeout" in results.item_results[0].error.lower()
            or "retry attempts failed" in results.item_results[0].error.lower()
        )

    @pytest.mark.asyncio
    async def test_save_results(self, runner, tmp_path):
        """Test saving results to file."""
        runner.config.results_dir = tmp_path

        with patch.object(runner, "call_model", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = "Answer"
            await runner.run()

        # Check that results were saved
        result_files = list(tmp_path.glob("*.json"))
        assert len(result_files) > 0

        # Verify content
        import json

        with open(result_files[0]) as f:
            saved_data = json.load(f)
            assert saved_data["benchmark_name"] == "mock_benchmark"
            assert saved_data["total_items"] == 5

    @pytest.mark.asyncio
    async def test_execution_context(self, runner):
        """Test execution context management."""
        context = runner.context

        assert context.start_time is not None
        assert context.benchmark_name == "mock_benchmark"
        assert context.config == runner.config.model_dump()

        # Update context during execution
        context.add_metadata("test_key", "test_value")
        assert context.metadata["test_key"] == "test_value"

        # Complete execution
        context.complete()
        assert context.end_time is not None
        assert context.duration > 0

    def test_progress_tracker(self):
        """Test progress tracker functionality."""
        tracker = ProgressTracker(total_items=100)

        assert tracker.total_items == 100
        assert tracker.processed_items == 0
        assert tracker.get_percentage() == 0.0

        # Update progress
        tracker.update(25)
        assert tracker.processed_items == 25
        assert tracker.get_percentage() == 25.0

        # Test with callback
        callback_called = False

        def callback(t: ProgressTracker):
            nonlocal callback_called
            callback_called = True

        tracker.add_callback(callback)
        tracker.update(50)

        assert callback_called
        assert tracker.processed_items == 50

    @pytest.mark.asyncio
    async def test_runner_error_handling(self):
        """Test runner error handling."""
        # Test with invalid benchmark
        with pytest.raises(RunnerError):
            BenchmarkRunner(None, ExecutionConfig())

        # Test with invalid config
        benchmark = MockBenchmark()
        with pytest.raises(RunnerError):
            BenchmarkRunner(benchmark, None)

    @pytest.mark.asyncio
    async def test_partial_results_on_error(self, runner):
        """Test that partial results are returned on error."""
        call_count = 0

        async def partial_failure_model(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise Exception("Critical failure")
            return f"Answer {call_count}"

        with patch.object(runner, "call_model", side_effect=partial_failure_model):
            results = await runner.run()

        # Should have results for items before failure
        assert results.total_items == 5
        assert results.successful_items < 5
        assert results.failed_items > 0

    @pytest.mark.asyncio
    async def test_custom_model_caller(self):
        """Test using custom model caller."""
        benchmark = MockBenchmark(dataset_size=2)
        config = ExecutionConfig()

        async def custom_caller(prompt: str) -> str:
            return f"Custom: {prompt}"

        runner = BenchmarkRunner(benchmark, config, model_caller=custom_caller)

        results = await runner.run()

        assert len(results.item_results) == 2
        # Custom caller returns different format, so won't match
        assert all(not r.evaluation.is_correct for r in results.item_results)

    @pytest.mark.asyncio
    async def test_benchmark_validation(self, runner):
        """Test benchmark configuration validation."""
        # Valid config
        valid_config = {"model": "gpt-4"}
        runner.validate_benchmark_config(valid_config)

        # Invalid config
        with pytest.raises(ValueError):
            runner.validate_benchmark_config({})


class TestExecutionConfig:
    """Test execution configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExecutionConfig()

        assert config.max_parallel == 5
        assert config.timeout_seconds == 60
        assert config.retry_attempts == 3
        assert config.temperature == 0.0
        assert config.save_results is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = ExecutionConfig(
            max_parallel=10,
            timeout_seconds=120,
            retry_attempts=5,
            temperature=0.5,
            save_results=False,
            results_dir=Path("/custom/path"),
        )

        assert config.max_parallel == 10
        assert config.timeout_seconds == 120
        assert config.retry_attempts == 5
        assert config.temperature == 0.5
        assert config.save_results is False
        assert config.results_dir == Path("/custom/path")

    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid max_parallel
        with pytest.raises(ValueError):
            ExecutionConfig(max_parallel=0)

        # Invalid timeout
        with pytest.raises(ValueError):
            ExecutionConfig(timeout_seconds=-1)

        # Invalid temperature
        with pytest.raises(ValueError):
            ExecutionConfig(temperature=2.1)  # Beyond max range


class TestExecutionResult:
    """Test execution result model."""

    def test_result_creation(self):
        """Test creating execution result."""
        item_results = [
            ExecutionResult.ItemResult(
                item_id=UUID("00000000-0000-0000-0000-000000000001"),
                prediction="Answer",
                ground_truth="Answer",
                evaluation=BenchmarkEvaluationResult(
                    is_correct=True,
                    score=1.0,
                    original_metric_score=1.0,
                ),
            )
        ]

        result = ExecutionResult(
            benchmark_name="test",
            total_items=1,
            successful_items=1,
            failed_items=0,
            metrics={"accuracy": 1.0},
            execution_time=10.5,
            item_results=item_results,
        )

        assert result.benchmark_name == "test"
        assert result.total_items == 1
        assert result.successful_items == 1
        assert result.metrics["accuracy"] == 1.0

    def test_result_serialization(self):
        """Test result serialization."""
        result = ExecutionResult(
            benchmark_name="test",
            total_items=10,
            successful_items=8,
            failed_items=2,
            metrics={"accuracy": 0.8},
            execution_time=30.0,
            item_results=[],
        )

        # Convert to dict
        result_dict = result.model_dump()
        assert result_dict["benchmark_name"] == "test"
        assert result_dict["metrics"]["accuracy"] == 0.8

        # Convert to JSON
        json_str = result.model_dump_json()
        assert "test" in json_str
        assert "0.8" in json_str
