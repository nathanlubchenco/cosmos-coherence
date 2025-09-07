"""Benchmark runner for executing benchmarks with async support."""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from cosmos_coherence.benchmarks.models.base import BaseDatasetItem
from cosmos_coherence.harness.base_benchmark import (
    BaseBenchmark,
    BenchmarkEvaluationResult,
)

logger = logging.getLogger(__name__)


class RunnerError(Exception):
    """Exception raised by benchmark runner."""

    pass


class ExecutionConfig(BaseModel):
    """Configuration for benchmark execution."""

    max_parallel: int = Field(default=5, ge=1, le=100, description="Max parallel executions")
    timeout_seconds: int = Field(default=60, ge=1, description="Timeout per item in seconds")
    retry_attempts: int = Field(default=3, ge=1, le=10, description="Number of retry attempts")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Model temperature")
    save_results: bool = Field(default=True, description="Save results to file")
    results_dir: Path = Field(default=Path("./results"), description="Directory for results")
    model_settings: Dict[str, Any] = Field(default_factory=dict, description="Model configuration")

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature range."""
        if v < 0.0 or v > 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {v}")
        return v

    @field_validator("max_parallel")
    @classmethod
    def validate_parallel(cls, v: int) -> int:
        """Validate max parallel setting."""
        if v < 1:
            raise ValueError(f"max_parallel must be at least 1, got {v}")
        return v

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout setting."""
        if v < 1:
            raise ValueError(f"timeout_seconds must be positive, got {v}")
        return v


class ProgressTracker:
    """Track execution progress."""

    def __init__(self, total_items: int):
        """Initialize progress tracker."""
        self.total_items = total_items
        self.processed_items = 0
        self.failed_items = 0
        self.start_time = time.time()
        self._callbacks: List[Callable] = []
        self._lock = asyncio.Lock()

    def update(self, processed: int, failed: int = 0) -> None:
        """Update progress."""
        self.processed_items = processed
        self.failed_items = failed
        self._notify_callbacks()

    def get_percentage(self) -> float:
        """Get completion percentage."""
        if self.total_items == 0:
            return 100.0
        return (self.processed_items / self.total_items) * 100

    def get_eta_seconds(self) -> Optional[float]:
        """Get estimated time to completion in seconds."""
        if self.processed_items == 0:
            return None

        elapsed = time.time() - self.start_time
        rate = self.processed_items / elapsed
        remaining = self.total_items - self.processed_items

        if rate > 0:
            return remaining / rate
        return None

    def add_callback(self, callback: Callable) -> None:
        """Add progress callback."""
        self._callbacks.append(callback)

    def _notify_callbacks(self) -> None:
        """Notify all callbacks of progress update."""
        for callback in self._callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")


class ExecutionContext:
    """Execution context for tracking runtime information."""

    def __init__(self, benchmark_name: str, config: Dict[str, Any]):
        """Initialize execution context."""
        self.benchmark_name = benchmark_name
        self.config = config
        self.start_time = datetime.utcnow()
        self.end_time: Optional[datetime] = None
        self.metadata: Dict[str, Any] = {}
        self.duration: float = 0.0

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to context."""
        self.metadata[key] = value

    def complete(self) -> None:
        """Mark execution as complete."""
        self.end_time = datetime.utcnow()
        self.duration = (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "config": self.config,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "metadata": self.metadata,
        }


class ExecutionResult(BaseModel):
    """Result of benchmark execution."""

    class ItemResult(BaseModel):
        """Result for a single item."""

        item_id: UUID
        prediction: Optional[str] = None
        ground_truth: Optional[str] = None
        evaluation: Optional[BenchmarkEvaluationResult] = None
        error: Optional[str] = None
        execution_time: float = 0.0

    benchmark_name: str
    total_items: int
    successful_items: int
    failed_items: int
    metrics: Dict[str, float]
    execution_time: float
    item_results: List[ItemResult]
    context: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def save_to_file(self, path: Path) -> None:
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, default=str)

        logger.info(f"Results saved to {path}")


class BenchmarkRunner:
    """Runner for executing benchmarks with async support."""

    def __init__(
        self,
        benchmark: BaseBenchmark,
        config: ExecutionConfig,
        model_caller: Optional[Callable] = None,
    ):
        """Initialize benchmark runner."""
        if not benchmark:
            raise RunnerError("Benchmark cannot be None")
        if not config:
            raise RunnerError("Config cannot be None")

        self.benchmark = benchmark
        self.config = config
        self.model_caller = model_caller or self._default_model_caller
        self.context = ExecutionContext(benchmark.benchmark_name, config.model_dump())
        self.progress = ProgressTracker(0)
        self._semaphore = asyncio.Semaphore(config.max_parallel)

    async def _default_model_caller(self, prompt: str) -> str:
        """Default model caller (mock implementation)."""
        # This would be replaced with actual model API call
        await asyncio.sleep(0.1)  # Simulate API delay
        return "Mock response"

    async def call_model(self, prompt: str) -> str:
        """Call the model with retry logic."""
        last_error = None

        for attempt in range(self.config.retry_attempts):
            try:
                # Apply timeout
                result = await asyncio.wait_for(
                    self.model_caller(prompt),
                    timeout=self.config.timeout_seconds,
                )
                return str(result)
            except asyncio.TimeoutError:
                last_error = "Request timeout"
                logger.warning(f"Attempt {attempt + 1} timed out")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

            if attempt < self.config.retry_attempts - 1:
                await asyncio.sleep(2**attempt)  # Exponential backoff

        raise RunnerError(f"All retry attempts failed: {last_error}")

    async def load_dataset(self) -> List[BaseDatasetItem]:
        """Load the benchmark dataset."""
        logger.info(f"Loading dataset for {self.benchmark.benchmark_name}")
        dataset = await self.benchmark.load_dataset()
        self.progress.total_items = len(dataset)
        logger.info(f"Loaded {len(dataset)} items")
        return dataset

    async def process_item(self, item: BaseDatasetItem) -> ExecutionResult.ItemResult:
        """Process a single dataset item."""
        start_time = time.time()

        try:
            # Get prompt
            prompt = self.benchmark.get_prompt(item)

            # Call model
            prediction = await self.call_model(prompt)

            # Get ground truth (assuming item has answer attribute)
            ground_truth = getattr(item, "answer", "")

            # Evaluate response
            evaluation = self.benchmark.evaluate_response(prediction, ground_truth, item)

            return ExecutionResult.ItemResult(
                item_id=item.id,
                prediction=prediction,
                ground_truth=ground_truth,
                evaluation=evaluation,
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            logger.error(f"Error processing item {item.id}: {e}")
            return ExecutionResult.ItemResult(
                item_id=item.id,
                error=str(e),
                execution_time=time.time() - start_time,
            )

    async def process_batch(self, items: List[BaseDatasetItem]) -> List[ExecutionResult.ItemResult]:
        """Process a batch of items with parallelism."""

        async def process_with_semaphore(item: BaseDatasetItem) -> ExecutionResult.ItemResult:
            async with self._semaphore:
                result = await self.process_item(item)
                # Update progress
                processed = self.progress.processed_items + 1
                failed = self.progress.failed_items + (1 if result.error else 0)
                self.progress.update(processed, failed)
                return result

        # Process items in parallel
        tasks = [process_with_semaphore(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions from gather
        processed_results: List[ExecutionResult.ItemResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    ExecutionResult.ItemResult(
                        item_id=items[i].id,
                        error=str(result),
                    )
                )
            elif isinstance(result, ExecutionResult.ItemResult):
                processed_results.append(result)

        return processed_results

    async def run(self) -> ExecutionResult:
        """Run the benchmark."""
        logger.info(f"Starting benchmark run: {self.benchmark.benchmark_name}")
        start_time = time.time()

        try:
            # Load dataset
            dataset = await self.load_dataset()

            # Process all items
            item_results = await self.process_batch(dataset)

            # Calculate metrics
            successful_results = [r for r in item_results if r.evaluation is not None]
            failed_results = [r for r in item_results if r.error is not None]

            metrics = self._calculate_metrics(successful_results)

            # Complete context
            self.context.complete()

            # Create execution result
            result = ExecutionResult(
                benchmark_name=self.benchmark.benchmark_name,
                total_items=len(dataset),
                successful_items=len(successful_results),
                failed_items=len(failed_results),
                metrics=metrics,
                execution_time=time.time() - start_time,
                item_results=item_results,
                context=self.context.to_dict(),
            )

            # Save results if configured
            if self.config.save_results:
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.benchmark.benchmark_name}_{timestamp}.json"
                result_path = self.config.results_dir / filename
                result.save_to_file(result_path)

            logger.info(
                f"Benchmark complete: {result.successful_items}/{result.total_items} successful"
            )

            return result

        except Exception as e:
            logger.error(f"Benchmark run failed: {e}")
            raise RunnerError(f"Benchmark execution failed: {e}")

    def _calculate_metrics(self, results: List[ExecutionResult.ItemResult]) -> Dict[str, float]:
        """Calculate aggregate metrics from results."""
        if not results:
            return {}

        total = len(results)
        correct = sum(1 for r in results if r.evaluation and r.evaluation.is_correct)
        total_score = sum(r.evaluation.score for r in results if r.evaluation)
        total_original = sum(r.evaluation.original_metric_score for r in results if r.evaluation)

        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "average_score": total_score / total if total > 0 else 0.0,
            "average_original_score": total_original / total if total > 0 else 0.0,
            "total_evaluated": total,
            "total_correct": correct,
        }

    def validate_benchmark_config(self, config: Dict[str, Any]) -> None:
        """Validate benchmark-specific configuration."""
        self.benchmark.validate_config(config)
