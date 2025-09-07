"""Benchmark data models package.

This module provides data models for benchmark execution and tracking.
Configuration enums are imported from config.models to maintain a single source of truth.
"""

from cosmos_coherence.config.models import (
    BenchmarkType,
    CoherenceMeasure,
)
from cosmos_coherence.config.models import (
    StrategyType as EvaluationStrategy,  # Alias for backward compatibility
)

from .base import (
    BaseDatasetItem,
    BaseExperiment,
    BaseResult,
    BenchmarkRunConfig,  # Renamed from BenchmarkConfig
    BenchmarkValidationError,
    ConfigurationError,
    DataPoint,
    DatasetValidationError,
    ValidationMixin,
)
from .datasets import (
    FaithBenchItem,
    FEVERItem,
    FEVERLabel,
    HaluEvalItem,
    HaluEvalTaskType,
    SimpleQACategory,
    SimpleQADifficulty,
    SimpleQAItem,
    TruthfulQACategory,
    TruthfulQAItem,
)
from .experiments import (
    AggregationResult,
    BenchmarkMetrics,
    ExperimentComparison,
    ExperimentResult,
    ExperimentRun,
    ExperimentStatus,
    ExperimentTracker,
    MetricType,
    ResultDiff,
    RunMetadata,
    StatisticalSummary,
)

__all__ = [
    # Base models
    "BaseDatasetItem",
    "BaseExperiment",
    "BaseResult",
    "BenchmarkRunConfig",
    # Dataset models
    "FaithBenchItem",
    "SimpleQAItem",
    "TruthfulQAItem",
    "FEVERItem",
    "HaluEvalItem",
    # Dataset enums
    "SimpleQACategory",
    "SimpleQADifficulty",
    "TruthfulQACategory",
    "FEVERLabel",
    "HaluEvalTaskType",
    # Experiment models
    "ExperimentTracker",
    "ExperimentRun",
    "ExperimentResult",
    "BenchmarkMetrics",
    "StatisticalSummary",
    "AggregationResult",
    "ExperimentComparison",
    "ResultDiff",
    "RunMetadata",
    # Experiment enums
    "ExperimentStatus",
    "MetricType",
    # Enums (imported from config.models)
    "BenchmarkType",
    "CoherenceMeasure",
    "EvaluationStrategy",
    # Exceptions
    "BenchmarkValidationError",
    "ConfigurationError",
    "DatasetValidationError",
    # Utilities
    "DataPoint",
    "ValidationMixin",
]
