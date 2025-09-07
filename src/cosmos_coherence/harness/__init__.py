"""Benchmark harness for reproducible evaluation."""

from cosmos_coherence.harness.base_benchmark import (
    BaseBenchmark,
    BenchmarkEvaluationResult,
    BenchmarkMetadata,
    OriginalMetrics,
)
from cosmos_coherence.harness.benchmark_runner import (
    BenchmarkRunner,
    ExecutionConfig,
    ExecutionContext,
    ExecutionResult,
    ProgressTracker,
    RunnerError,
)
from cosmos_coherence.harness.reproducibility import (
    BaselineMetrics,
    ComparisonReport,
    DeviationDetail,
    ReproducibilityConfig,
    ReproducibilityValidator,
    ValidationResult,
)

__all__ = [
    # Base benchmark classes
    "BaseBenchmark",
    "BenchmarkEvaluationResult",
    "BenchmarkMetadata",
    "OriginalMetrics",
    # Benchmark runner classes
    "BenchmarkRunner",
    "ExecutionConfig",
    "ExecutionContext",
    "ExecutionResult",
    "ProgressTracker",
    "RunnerError",
    # Reproducibility classes
    "BaselineMetrics",
    "ComparisonReport",
    "DeviationDetail",
    "ReproducibilityConfig",
    "ReproducibilityValidator",
    "ValidationResult",
]
