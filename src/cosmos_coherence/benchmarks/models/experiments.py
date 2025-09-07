"""Experiment tracking and result models for benchmark execution."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

from cosmos_coherence.benchmarks.models.base import ValidationMixin
from cosmos_coherence.config.models import (
    BenchmarkType,
    CoherenceMeasure,
)
from cosmos_coherence.config.models import (
    StrategyType as EvaluationStrategy,
)


class ExperimentStatus(str, Enum):
    """Status of an experiment or run."""

    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MetricType(str, Enum):
    """Types of metrics tracked in experiments."""

    # Performance metrics
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"

    # Hallucination metrics
    HALLUCINATION_RATE = "hallucination_rate"
    FACTUALITY_SCORE = "factuality_score"
    CONSISTENCY_SCORE = "consistency_score"

    # Efficiency metrics
    RESPONSE_TIME = "response_time"
    TOKENS_PER_SECOND = "tokens_per_second"
    TOTAL_TOKENS = "total_tokens"

    # Resource metrics
    COST = "cost"
    MEMORY_USAGE = "memory_usage"
    GPU_UTILIZATION = "gpu_utilization"

    # Error metrics
    ERROR_RATE = "error_rate"
    TIMEOUT_RATE = "timeout_rate"

    @property
    def category(self) -> str:
        """Get the category of this metric."""
        if self in [self.ACCURACY, self.PRECISION, self.RECALL, self.F1_SCORE]:
            return "performance"
        elif self in [self.HALLUCINATION_RATE, self.FACTUALITY_SCORE, self.CONSISTENCY_SCORE]:
            return "quality"
        elif self in [self.RESPONSE_TIME, self.TOKENS_PER_SECOND, self.TOTAL_TOKENS]:
            return "efficiency"
        elif self in [self.COST, self.MEMORY_USAGE, self.GPU_UTILIZATION]:
            return "resource"
        else:
            return "error"

    @property
    def aggregation(self) -> str:
        """Get the aggregation method for this metric."""
        if self in [self.RESPONSE_TIME]:
            return "percentile"
        elif self in [self.TOTAL_TOKENS, self.COST]:
            return "sum"
        else:
            return "mean"

    @property
    def display_name(self) -> str:
        """Get the display name for this metric."""
        return self.value.replace("_", " ").title()

    @property
    def unit(self) -> str:
        """Get the unit for this metric."""
        if self in [
            self.ACCURACY,
            self.PRECISION,
            self.RECALL,
            self.F1_SCORE,
            self.HALLUCINATION_RATE,
            self.FACTUALITY_SCORE,
            self.CONSISTENCY_SCORE,
            self.ERROR_RATE,
            self.TIMEOUT_RATE,
            self.GPU_UTILIZATION,
        ]:
            return "%"
        elif self == self.RESPONSE_TIME:
            return "ms"
        elif self == self.TOKENS_PER_SECOND:
            return "tokens/s"
        elif self == self.TOTAL_TOKENS:
            return "tokens"
        elif self == self.COST:
            return "USD"
        elif self == self.MEMORY_USAGE:
            return "MB"
        else:
            return ""

    @property
    def good_threshold(self) -> Optional[float]:
        """Get the threshold for a 'good' value of this metric."""
        thresholds = {
            self.ACCURACY: 0.90,
            self.PRECISION: 0.90,
            self.RECALL: 0.85,
            self.F1_SCORE: 0.85,
            self.HALLUCINATION_RATE: 0.05,
            self.FACTUALITY_SCORE: 0.90,
            self.CONSISTENCY_SCORE: 0.90,
            self.RESPONSE_TIME: 1000,
            self.ERROR_RATE: 0.01,
            self.TIMEOUT_RATE: 0.01,
        }
        return thresholds.get(self)


class RunMetadata(BaseModel):
    """Metadata about an experiment run environment."""

    hostname: Optional[str] = Field(None, description="Hostname where run executed")
    gpu_info: Optional[str] = Field(None, description="GPU information if available")
    python_version: Optional[str] = Field(None, description="Python version used")
    package_versions: Dict[str, str] = Field(
        default_factory=dict,
        description="Versions of key packages",
    )
    environment_vars: Dict[str, str] = Field(
        default_factory=dict,
        description="Relevant environment variables",
    )
    git_commit: Optional[str] = Field(None, description="Git commit hash")
    git_branch: Optional[str] = Field(None, description="Git branch name")


class ExperimentTracker(BaseModel, ValidationMixin):
    """Main experiment tracking model."""

    id: UUID = Field(default_factory=uuid4, description="Unique experiment ID")
    name: str = Field(..., description="Experiment name", min_length=1, max_length=255)
    description: Optional[str] = Field(None, description="Experiment description", max_length=2000)
    benchmark_type: BenchmarkType = Field(..., description="Type of benchmark")
    strategy: EvaluationStrategy = Field(..., description="Evaluation strategy")
    model_name: str = Field(..., description="Model being tested")

    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Experiment configuration",
    )
    tags: List[str] = Field(default_factory=list, description="Experiment tags")

    status: ExperimentStatus = Field(
        default=ExperimentStatus.CREATED,
        description="Current experiment status",
    )
    total_runs: int = Field(default=0, description="Total number of runs")
    completed_runs: int = Field(default=0, description="Number of completed runs")

    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Experiment creation time",
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Last update time",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate experiment name."""
        if not v or not v.strip():
            raise ValueError("Experiment name cannot be empty")
        return v.strip()


class ExperimentRun(BaseModel, ValidationMixin):
    """Individual run within an experiment."""

    id: UUID = Field(default_factory=uuid4, description="Unique run ID")
    experiment_id: UUID = Field(..., description="Parent experiment ID")
    run_number: int = Field(..., description="Run number within experiment", gt=0)

    dataset_size: int = Field(..., description="Number of items in dataset", gt=0)
    processed_items: int = Field(default=0, description="Items processed so far")
    failed_items: int = Field(default=0, description="Items that failed")

    status: ExperimentStatus = Field(
        default=ExperimentStatus.CREATED,
        description="Run status",
    )

    started_at: Optional[datetime] = Field(None, description="Run start time")
    completed_at: Optional[datetime] = Field(None, description="Run completion time")

    checkpoint_path: Optional[Path] = Field(None, description="Checkpoint file path")
    last_checkpoint_at: Optional[datetime] = Field(None, description="Last checkpoint time")

    error_message: Optional[str] = Field(None, description="Error message if failed")
    error_traceback: Optional[str] = Field(None, description="Error traceback")

    metadata: Optional[RunMetadata] = Field(None, description="Run environment metadata")

    @property
    def duration(self) -> Optional[float]:
        """Calculate run duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def progress_percentage(self) -> float:
        """Calculate progress as percentage."""
        if self.dataset_size > 0:
            return (self.processed_items / self.dataset_size) * 100
        return 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.processed_items > 0:
            successful = self.processed_items - self.failed_items
            return (successful / self.processed_items) * 100
        return 0.0

    @field_validator("dataset_size", "processed_items", "failed_items")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        """Validate non-negative integers."""
        if v < 0:
            raise ValueError("Value must be non-negative")
        return v


class ExperimentResult(BaseModel, ValidationMixin):
    """Individual result from an experiment run."""

    id: UUID = Field(default_factory=uuid4, description="Unique result ID")
    experiment_id: UUID = Field(..., description="Parent experiment ID")
    run_id: UUID = Field(..., description="Parent run ID")
    dataset_item_id: str = Field(..., description="Dataset item identifier")

    model_response: Optional[str] = Field(None, description="Model's response")
    is_correct: Optional[bool] = Field(None, description="Whether response is correct")
    confidence_score: Optional[float] = Field(
        None,
        description="Confidence score [0, 1]",
        ge=0.0,
        le=1.0,
    )

    all_responses: Optional[List[str]] = Field(
        None,
        description="All responses for k-response strategy",
    )
    response_scores: Optional[List[float]] = Field(
        None,
        description="Scores for each response",
    )

    coherence_scores: Optional[Dict[CoherenceMeasure, float]] = Field(
        None,
        description="Coherence measure scores",
    )

    response_time_ms: Optional[int] = Field(
        None,
        description="Response time in milliseconds",
        gt=0,
    )
    tokens_used: Optional[int] = Field(
        None,
        description="Number of tokens used",
        gt=0,
    )

    error: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional result metadata",
    )

    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Result creation time",
    )

    @model_validator(mode="after")
    def validate_responses(self) -> "ExperimentResult":
        """Validate response arrays have matching lengths."""
        if self.all_responses and self.response_scores:
            if len(self.all_responses) != len(self.response_scores):
                raise ValueError("all_responses and response_scores must have same length")
        return self


class StatisticalSummary(BaseModel, ValidationMixin):
    """Statistical summary of a metric."""

    metric_name: str = Field(..., description="Name of the metric")
    mean: float = Field(..., description="Mean value")
    std_dev: Optional[float] = Field(None, description="Standard deviation")
    min_value: Optional[float] = Field(None, description="Minimum value")
    max_value: Optional[float] = Field(None, description="Maximum value")
    median: Optional[float] = Field(None, description="Median value")

    percentiles: Optional[Dict[int, float]] = Field(
        None,
        description="Percentile values (e.g., {25: 0.25, 50: 0.5, 75: 0.75})",
    )

    confidence_interval_95: Optional[Tuple[float, float]] = Field(
        None,
        description="95% confidence interval",
    )
    confidence_interval_99: Optional[Tuple[float, float]] = Field(
        None,
        description="99% confidence interval",
    )

    skewness: Optional[float] = Field(None, description="Skewness of distribution")
    kurtosis: Optional[float] = Field(None, description="Kurtosis of distribution")
    distribution_test: Optional[str] = Field(
        None,
        description="Distribution test performed",
    )
    p_value: Optional[float] = Field(None, description="P-value from distribution test")

    count: int = Field(..., description="Number of samples", gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> "StatisticalSummary":
        """Validate statistical consistency."""
        if self.min_value is not None and self.max_value is not None:
            if self.min_value > self.max_value:
                raise ValueError("min_value cannot be greater than max_value")
            if self.min_value > self.mean or self.max_value < self.mean:
                raise ValueError("mean must be between min and max values")
            if self.median is not None:
                if self.min_value > self.median or self.max_value < self.median:
                    raise ValueError("median must be between min and max values")

        # Validate percentiles
        if self.percentiles:
            for percentile, value in self.percentiles.items():
                if not 0 <= percentile <= 100:
                    raise ValueError(f"Percentile {percentile} must be between 0 and 100")
                if self.min_value is not None and value < self.min_value:
                    raise ValueError(f"Percentile {percentile} value cannot be less than min_value")
                if self.max_value is not None and value > self.max_value:
                    raise ValueError(
                        f"Percentile {percentile} value cannot be greater than max_value"
                    )

        # Validate standard deviation
        if self.std_dev is not None and self.std_dev < 0:
            raise ValueError("Standard deviation cannot be negative")

        return self


class BenchmarkMetrics(BaseModel, ValidationMixin):
    """Metrics calculated for a benchmark run."""

    experiment_id: UUID = Field(..., description="Parent experiment ID")
    run_id: UUID = Field(..., description="Parent run ID")

    total_items: int = Field(..., description="Total items evaluated", gt=0)
    correct_items: int = Field(..., description="Correctly answered items", ge=0)
    failed_items: int = Field(default=0, description="Failed items", ge=0)

    # Core metrics
    accuracy: Optional[float] = Field(None, description="Overall accuracy", ge=0.0, le=1.0)
    precision: Optional[float] = Field(None, description="Precision score", ge=0.0, le=1.0)
    recall: Optional[float] = Field(None, description="Recall score", ge=0.0, le=1.0)
    f1_score: Optional[float] = Field(None, description="F1 score", ge=0.0, le=1.0)
    cohen_kappa: Optional[float] = Field(None, description="Cohen's kappa score")

    # Hallucination metrics
    hallucination_rate: Optional[float] = Field(
        None, description="Rate of hallucinations", ge=0.0, le=1.0
    )
    factuality_score: Optional[float] = Field(None, description="Factuality score", ge=0.0, le=1.0)
    consistency_score: Optional[float] = Field(
        None, description="Consistency score", ge=0.0, le=1.0
    )

    # Performance metrics
    avg_response_time_ms: Optional[float] = Field(None, description="Average response time")
    p95_response_time_ms: Optional[float] = Field(None, description="95th percentile response time")
    p99_response_time_ms: Optional[float] = Field(None, description="99th percentile response time")

    total_tokens_used: Optional[int] = Field(None, description="Total tokens consumed")
    total_cost_usd: Optional[float] = Field(None, description="Total cost in USD")

    # Confidence intervals
    accuracy_confidence_interval: Optional[Tuple[float, float]] = Field(
        None,
        description="Confidence interval for accuracy",
    )

    # Breakdown by category
    metrics_by_category: Optional[Dict[str, Dict[str, Any]]] = Field(
        None,
        description="Metrics broken down by category",
    )

    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Metrics calculation time",
    )

    @model_validator(mode="before")
    @classmethod
    def calculate_basic_metrics(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate basic metrics if not provided."""
        if "accuracy" not in values or values.get("accuracy") is None:
            total = values.get("total_items", 0)
            correct = values.get("correct_items", 0)
            if total > 0:
                values["accuracy"] = correct / total
        return values

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_items > 0:
            return self.failed_items / self.total_items
        return 0.0

    @property
    def completion_rate(self) -> float:
        """Calculate completion rate."""
        if self.total_items > 0:
            completed = self.total_items - self.failed_items
            return completed / self.total_items
        return 0.0

    @model_validator(mode="after")
    def validate_counts(self) -> "BenchmarkMetrics":
        """Validate count consistency."""
        if self.correct_items > self.total_items:
            raise ValueError("correct_items cannot exceed total_items")
        if self.failed_items > self.total_items:
            raise ValueError("failed_items cannot exceed total_items")

        # Validate F1 score consistency with precision and recall
        if self.precision is not None and self.recall is not None and self.f1_score is not None:
            if self.precision > 0 and self.recall > 0:
                expected_f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)
                # Allow small tolerance for floating point errors
                if abs(self.f1_score - expected_f1) > 0.01:
                    # Just note the inconsistency, don't fail
                    pass

        return self

    def validate_consistency(self) -> list[str]:
        """Check metric consistency and return warnings."""
        warnings = []
        if self.precision is not None and self.recall is not None and self.f1_score is not None:
            if self.precision > 0 and self.recall > 0:
                expected_f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)
                if abs(self.f1_score - expected_f1) > 0.01:
                    warnings.append(f"F1 score {self.f1_score} inconsistent with precision/recall")
        return warnings


class AggregationResult(BaseModel, ValidationMixin):
    """Aggregated results across multiple runs."""

    experiment_id: UUID = Field(..., description="Parent experiment ID")
    num_runs: int = Field(..., description="Number of runs aggregated", gt=0)
    total_items: int = Field(..., description="Total items across all runs", gt=0)

    overall_metrics: Optional[Dict[str, StatisticalSummary]] = Field(
        None,
        description="Statistical summary of each metric",
    )

    metrics_by_strategy: Optional[Dict[EvaluationStrategy, Dict[str, float]]] = Field(
        None,
        description="Metrics grouped by strategy",
    )

    metrics_by_model: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="Metrics grouped by model",
    )

    temporal_metrics: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Metrics over time",
    )

    best_run_id: Optional[UUID] = Field(None, description="ID of best performing run")
    worst_run_id: Optional[UUID] = Field(None, description="ID of worst performing run")

    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Aggregation time",
    )


class ExperimentComparison(BaseModel, ValidationMixin):
    """Comparison between multiple experiments."""

    experiment_ids: List[UUID] = Field(
        ...,
        description="IDs of experiments being compared",
        min_length=2,
    )

    metric_comparisons: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Comparison of metrics across experiments",
    )

    winner_by_metric: Optional[Dict[str, str]] = Field(
        None,
        description="Winner for each metric",
    )

    statistical_tests: Optional[Dict[str, Dict[str, Any]]] = Field(
        None,
        description="Statistical test results",
    )

    recommendations: Optional[List[str]] = Field(
        None,
        description="Recommendations based on comparison",
    )

    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Comparison time",
    )


class ResultDiff(BaseModel, ValidationMixin):
    """Difference between two results."""

    old_result_id: UUID = Field(..., description="ID of old result")
    new_result_id: UUID = Field(..., description="ID of new result")

    field_changes: Optional[Dict[str, Dict[str, Any]]] = Field(
        None,
        description="Changes in fields",
    )

    metric_changes: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="Changes in metrics",
    )

    response_diff: Optional[Dict[str, Any]] = Field(
        None,
        description="Difference in model responses",
    )

    coherence_changes: Optional[Dict[CoherenceMeasure, Dict[str, float]]] = Field(
        None,
        description="Changes in coherence scores",
    )

    improvement_summary: Optional[str] = Field(
        None,
        description="Summary of improvements",
    )

    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Diff creation time",
    )
