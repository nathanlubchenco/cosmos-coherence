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

# Temporarily comment out CLI import to fix tests
# from cosmos_coherence.harness.cli import BenchmarkCLI
from cosmos_coherence.harness.reproducibility import (
    BaselineMetrics,
    ComparisonReport,
    DeviationDetail,
    ReproducibilityConfig,
    ReproducibilityValidator,
    ValidationResult,
)
from cosmos_coherence.harness.result_collection import (
    BenchmarkComparison,
    BenchmarkReport,
    ExportFormat,
    ResultCollector,
    ResultFilter,
    ResultReporter,
    ResultStorage,
    StatisticalSummary,
    SummaryReport,
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
    # Result collection classes
    "BenchmarkComparison",
    "BenchmarkReport",
    "ExportFormat",
    "ResultCollector",
    "ResultFilter",
    "ResultReporter",
    "ResultStorage",
    "StatisticalSummary",
    "SummaryReport",
    # CLI classes
    # "BenchmarkCLI",  # Temporarily commented out
]
