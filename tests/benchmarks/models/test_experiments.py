"""Tests for experiment tracking and result models."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest
from cosmos_coherence.benchmarks.models import (
    BenchmarkType,
    CoherenceMeasure,
    EvaluationStrategy,
)
from cosmos_coherence.benchmarks.models.experiments import (
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
from pydantic import ValidationError


class TestExperimentTracker:
    """Test ExperimentTracker model for overall experiment management."""

    def test_experiment_tracker_creation(self):
        """Test creating an experiment tracker."""
        tracker = ExperimentTracker(
            name="GPT-5 Hallucination Study",
            description="Testing GPT-5 on hallucination benchmarks",
            benchmark_type=BenchmarkType.FAITHBENCH,
            strategy=EvaluationStrategy.COHERENCE,
            model_name="gpt-5",
            tags=["gpt-5", "hallucination", "coherence"],
        )
        assert tracker.name == "GPT-5 Hallucination Study"
        assert tracker.status == ExperimentStatus.CREATED
        assert isinstance(tracker.id, UUID)
        assert tracker.total_runs == 0

    def test_experiment_tracker_with_config(self):
        """Test tracker with full configuration."""
        tracker = ExperimentTracker(
            name="Test Experiment",
            benchmark_type=BenchmarkType.SIMPLEQA,
            strategy=EvaluationStrategy.K_RESPONSE,
            model_name="gpt-4",
            config={
                "k_responses": 5,
                "temperature": 0.7,
                "max_tokens": 1000,
            },
        )
        assert tracker.config["k_responses"] == 5
        assert tracker.config["temperature"] == 0.7

    def test_experiment_tracker_status_transitions(self):
        """Test valid status transitions."""
        tracker = ExperimentTracker(
            name="Test",
            benchmark_type=BenchmarkType.FEVER,
            strategy=EvaluationStrategy.BASELINE,
            model_name="gpt-4",
        )

        # Valid transitions
        tracker.status = ExperimentStatus.RUNNING
        assert tracker.status == ExperimentStatus.RUNNING

        tracker.status = ExperimentStatus.COMPLETED
        assert tracker.status == ExperimentStatus.COMPLETED

    def test_experiment_tracker_validation(self):
        """Test tracker validation rules."""
        with pytest.raises(ValidationError):
            # Name cannot be empty
            ExperimentTracker(
                name="",
                benchmark_type=BenchmarkType.TRUTHFULQA,
                strategy=EvaluationStrategy.COHERENCE,
                model_name="gpt-4",
            )

    def test_experiment_tracker_serialization(self):
        """Test JSON serialization of tracker."""
        tracker = ExperimentTracker(
            name="Serialization Test",
            benchmark_type=BenchmarkType.HALUEVAL,
            strategy=EvaluationStrategy.COHERENCE,
            model_name="gpt-4",
            tags=["test"],
        )
        json_str = tracker.model_dump_json()
        data = json.loads(json_str)
        assert data["name"] == "Serialization Test"
        assert data["benchmark_type"] == "halueval"

    def test_experiment_tracker_comparison(self):
        """Test comparing two experiment trackers."""
        tracker1 = ExperimentTracker(
            name="Experiment 1",
            benchmark_type=BenchmarkType.FAITHBENCH,
            strategy=EvaluationStrategy.COHERENCE,
            model_name="gpt-4",
        )
        tracker2 = ExperimentTracker(
            name="Experiment 2",
            benchmark_type=BenchmarkType.FAITHBENCH,
            strategy=EvaluationStrategy.BASELINE,
            model_name="gpt-4",
        )
        assert tracker1.id != tracker2.id
        assert tracker1.strategy != tracker2.strategy


class TestExperimentRun:
    """Test ExperimentRun model for individual executions."""

    def test_experiment_run_basic(self):
        """Test basic experiment run creation."""
        run = ExperimentRun(
            experiment_id=UUID("12345678-1234-5678-1234-567812345678"),
            run_number=1,
            dataset_size=100,
            status=ExperimentStatus.RUNNING,
        )
        assert run.run_number == 1
        assert run.dataset_size == 100
        assert run.status == ExperimentStatus.RUNNING
        assert run.duration is None

    def test_experiment_run_with_metadata(self):
        """Test run with metadata."""
        metadata = RunMetadata(
            hostname="gpu-server-01",
            gpu_info="NVIDIA A100 80GB",
            python_version="3.11.5",
            package_versions={"openai": "1.0.0", "pydantic": "2.5.0"},
            environment_vars={"CUDA_VISIBLE_DEVICES": "0,1"},
        )
        run = ExperimentRun(
            experiment_id=UUID("12345678-1234-5678-1234-567812345678"),
            run_number=1,
            dataset_size=1000,
            metadata=metadata,
        )
        assert run.metadata.hostname == "gpu-server-01"
        assert run.metadata.gpu_info == "NVIDIA A100 80GB"
        assert "openai" in run.metadata.package_versions

    def test_experiment_run_timing(self):
        """Test run timing calculations."""
        start = datetime.now()
        run = ExperimentRun(
            experiment_id=UUID("12345678-1234-5678-1234-567812345678"),
            run_number=1,
            dataset_size=100,
            started_at=start,
            completed_at=start + timedelta(minutes=5),
        )
        assert run.duration == 300.0  # 5 minutes in seconds

    def test_experiment_run_error_tracking(self):
        """Test error tracking in runs."""
        run = ExperimentRun(
            experiment_id=UUID("12345678-1234-5678-1234-567812345678"),
            run_number=1,
            dataset_size=100,
            status=ExperimentStatus.FAILED,
            error_message="API rate limit exceeded",
            error_traceback="Traceback (most recent call last):\n...",
        )
        assert run.status == ExperimentStatus.FAILED
        assert "rate limit" in run.error_message

    def test_experiment_run_progress_tracking(self):
        """Test progress tracking during run."""
        run = ExperimentRun(
            experiment_id=UUID("12345678-1234-5678-1234-567812345678"),
            run_number=1,
            dataset_size=1000,
            processed_items=250,
            failed_items=5,
        )
        assert run.progress_percentage == 25.0
        assert run.success_rate == 98.0  # (250-5)/250

    def test_experiment_run_checkpointing(self):
        """Test checkpoint support for resumable runs."""
        run = ExperimentRun(
            experiment_id=UUID("12345678-1234-5678-1234-567812345678"),
            run_number=1,
            dataset_size=10000,
            checkpoint_path=Path("/tmp/checkpoint_001.pkl"),
            last_checkpoint_at=datetime.now(),
            processed_items=5000,
        )
        assert run.checkpoint_path == Path("/tmp/checkpoint_001.pkl")
        assert run.processed_items == 5000


class TestExperimentResult:
    """Test ExperimentResult model for storing outcomes."""

    def test_experiment_result_basic(self):
        """Test basic result creation."""
        result = ExperimentResult(
            experiment_id=UUID("12345678-1234-5678-1234-567812345678"),
            run_id=UUID("87654321-4321-8765-4321-876543218765"),
            dataset_item_id="item_001",
            model_response="Paris is the capital of France.",
            is_correct=True,
            confidence_score=0.95,
        )
        assert result.is_correct is True
        assert result.confidence_score == 0.95

    def test_experiment_result_with_multiple_responses(self):
        """Test result with k-response strategy."""
        result = ExperimentResult(
            experiment_id=UUID("12345678-1234-5678-1234-567812345678"),
            run_id=UUID("87654321-4321-8765-4321-876543218765"),
            dataset_item_id="item_002",
            model_response="Final answer",
            all_responses=["Answer 1", "Answer 2", "Answer 3"],
            response_scores=[0.8, 0.9, 0.85],
            is_correct=True,
        )
        assert len(result.all_responses) == 3
        assert len(result.response_scores) == 3

    def test_experiment_result_with_coherence(self):
        """Test result with coherence measures."""
        result = ExperimentResult(
            experiment_id=UUID("12345678-1234-5678-1234-567812345678"),
            run_id=UUID("87654321-4321-8765-4321-876543218765"),
            dataset_item_id="item_003",
            model_response="Test response",
            coherence_scores={
                CoherenceMeasure.SHOGENJI: 0.75,
                CoherenceMeasure.FITELSON: 0.82,
                CoherenceMeasure.OLSSON: 0.69,
            },
            is_correct=False,
        )
        assert result.coherence_scores[CoherenceMeasure.SHOGENJI] == 0.75

    def test_experiment_result_timing(self):
        """Test result timing metrics."""
        result = ExperimentResult(
            experiment_id=UUID("12345678-1234-5678-1234-567812345678"),
            run_id=UUID("87654321-4321-8765-4321-876543218765"),
            dataset_item_id="item_004",
            model_response="Response",
            response_time_ms=1250,
            tokens_used=150,
            is_correct=True,
        )
        assert result.response_time_ms == 1250
        assert result.tokens_used == 150

    def test_experiment_result_error_handling(self):
        """Test result with error."""
        result = ExperimentResult(
            experiment_id=UUID("12345678-1234-5678-1234-567812345678"),
            run_id=UUID("87654321-4321-8765-4321-876543218765"),
            dataset_item_id="item_005",
            model_response=None,
            is_correct=False,
            error="API timeout",
        )
        assert result.model_response is None
        assert result.error == "API timeout"

    def test_experiment_result_metadata(self):
        """Test result with additional metadata."""
        result = ExperimentResult(
            experiment_id=UUID("12345678-1234-5678-1234-567812345678"),
            run_id=UUID("87654321-4321-8765-4321-876543218765"),
            dataset_item_id="item_006",
            model_response="Response",
            is_correct=True,
            metadata={
                "prompt_template": "template_v1",
                "temperature": 0.7,
                "top_p": 0.9,
            },
        )
        assert result.metadata["temperature"] == 0.7


class TestBenchmarkMetrics:
    """Test BenchmarkMetrics model for metric calculations."""

    def test_benchmark_metrics_basic(self):
        """Test basic metrics calculation."""
        metrics = BenchmarkMetrics(
            experiment_id=UUID("12345678-1234-5678-1234-567812345678"),
            run_id=UUID("87654321-4321-8765-4321-876543218765"),
            total_items=100,
            correct_items=85,
            failed_items=5,
        )
        assert metrics.accuracy == 0.85
        assert metrics.error_rate == 0.05
        assert metrics.completion_rate == 0.95

    def test_benchmark_metrics_with_confidence(self):
        """Test metrics with confidence intervals."""
        metrics = BenchmarkMetrics(
            experiment_id=UUID("12345678-1234-5678-1234-567812345678"),
            run_id=UUID("87654321-4321-8765-4321-876543218765"),
            total_items=1000,
            correct_items=750,
            accuracy_confidence_interval=(0.72, 0.78),
        )
        assert metrics.accuracy == 0.75
        assert metrics.accuracy_confidence_interval == (0.72, 0.78)

    def test_benchmark_metrics_hallucination_specific(self):
        """Test hallucination-specific metrics."""
        metrics = BenchmarkMetrics(
            experiment_id=UUID("12345678-1234-5678-1234-567812345678"),
            run_id=UUID("87654321-4321-8765-4321-876543218765"),
            total_items=200,
            correct_items=160,
            hallucination_rate=0.15,
            factuality_score=0.85,
            consistency_score=0.92,
        )
        assert metrics.hallucination_rate == 0.15
        assert metrics.factuality_score == 0.85

    def test_benchmark_metrics_by_category(self):
        """Test metrics broken down by category."""
        metrics = BenchmarkMetrics(
            experiment_id=UUID("12345678-1234-5678-1234-567812345678"),
            run_id=UUID("87654321-4321-8765-4321-876543218765"),
            total_items=300,
            correct_items=240,
            metrics_by_category={
                "science": {"accuracy": 0.85, "total": 100},
                "history": {"accuracy": 0.75, "total": 100},
                "general": {"accuracy": 0.82, "total": 100},
            },
        )
        assert metrics.metrics_by_category["science"]["accuracy"] == 0.85

    def test_benchmark_metrics_performance(self):
        """Test performance metrics."""
        metrics = BenchmarkMetrics(
            experiment_id=UUID("12345678-1234-5678-1234-567812345678"),
            run_id=UUID("87654321-4321-8765-4321-876543218765"),
            total_items=100,
            correct_items=80,
            avg_response_time_ms=1500,
            p95_response_time_ms=3000,
            p99_response_time_ms=5000,
            total_tokens_used=15000,
            total_cost_usd=0.45,
        )
        assert metrics.avg_response_time_ms == 1500
        assert metrics.p95_response_time_ms == 3000
        assert metrics.total_cost_usd == 0.45

    def test_benchmark_metrics_statistical(self):
        """Test statistical metrics."""
        metrics = BenchmarkMetrics(
            experiment_id=UUID("12345678-1234-5678-1234-567812345678"),
            run_id=UUID("87654321-4321-8765-4321-876543218765"),
            total_items=500,
            correct_items=400,
            precision=0.82,
            recall=0.78,
            f1_score=0.80,
            cohen_kappa=0.65,
        )
        assert metrics.precision == 0.82
        assert metrics.recall == 0.78
        assert metrics.f1_score == 0.80


class TestStatisticalSummary:
    """Test StatisticalSummary for aggregated statistics."""

    def test_statistical_summary_basic(self):
        """Test basic statistical summary."""
        summary = StatisticalSummary(
            metric_name="accuracy",
            mean=0.85,
            std_dev=0.05,
            min_value=0.75,
            max_value=0.95,
            median=0.85,
            count=10,
        )
        assert summary.mean == 0.85
        assert summary.std_dev == 0.05

    def test_statistical_summary_percentiles(self):
        """Test summary with percentiles."""
        summary = StatisticalSummary(
            metric_name="response_time",
            mean=1500,
            std_dev=500,
            min_value=500,
            max_value=5000,
            median=1400,
            percentiles={
                25: 1000,
                50: 1400,
                75: 1800,
                90: 2500,
                95: 3000,
                99: 4500,
            },
            count=1000,
        )
        assert summary.percentiles[95] == 3000

    def test_statistical_summary_confidence_interval(self):
        """Test summary with confidence intervals."""
        summary = StatisticalSummary(
            metric_name="hallucination_rate",
            mean=0.15,
            std_dev=0.03,
            confidence_interval_95=(0.12, 0.18),
            confidence_interval_99=(0.11, 0.19),
            count=500,
        )
        assert summary.confidence_interval_95 == (0.12, 0.18)

    def test_statistical_summary_distribution(self):
        """Test summary with distribution info."""
        summary = StatisticalSummary(
            metric_name="coherence_score",
            mean=0.75,
            std_dev=0.10,
            skewness=-0.5,
            kurtosis=2.8,
            distribution_test="normal",
            p_value=0.12,
            count=200,
        )
        assert summary.skewness == -0.5
        assert summary.distribution_test == "normal"


class TestAggregationResult:
    """Test AggregationResult for batch processing."""

    def test_aggregation_result_basic(self):
        """Test basic aggregation result."""
        agg = AggregationResult(
            experiment_id=UUID("12345678-1234-5678-1234-567812345678"),
            num_runs=5,
            total_items=5000,
            overall_metrics={
                "accuracy": StatisticalSummary(
                    metric_name="accuracy",
                    mean=0.82,
                    std_dev=0.03,
                    min_value=0.78,
                    max_value=0.86,
                    median=0.82,
                    count=5,
                ),
            },
        )
        assert agg.num_runs == 5
        assert agg.overall_metrics["accuracy"].mean == 0.82

    def test_aggregation_result_by_strategy(self):
        """Test aggregation grouped by strategy."""
        agg = AggregationResult(
            experiment_id=UUID("12345678-1234-5678-1234-567812345678"),
            num_runs=15,
            total_items=15000,
            metrics_by_strategy={
                EvaluationStrategy.BASELINE: {
                    "accuracy": 0.75,
                    "hallucination_rate": 0.20,
                },
                EvaluationStrategy.K_RESPONSE: {
                    "accuracy": 0.82,
                    "hallucination_rate": 0.15,
                },
                EvaluationStrategy.COHERENCE: {
                    "accuracy": 0.88,
                    "hallucination_rate": 0.10,
                },
            },
        )
        assert agg.metrics_by_strategy[EvaluationStrategy.COHERENCE]["accuracy"] == 0.88

    def test_aggregation_result_by_model(self):
        """Test aggregation grouped by model."""
        agg = AggregationResult(
            experiment_id=UUID("12345678-1234-5678-1234-567812345678"),
            num_runs=20,
            total_items=20000,
            metrics_by_model={
                "gpt-4": {"accuracy": 0.85, "cost": 1.50},
                "gpt-5": {"accuracy": 0.92, "cost": 2.00},
                "claude-3": {"accuracy": 0.88, "cost": 1.75},
            },
        )
        assert agg.metrics_by_model["gpt-5"]["accuracy"] == 0.92

    def test_aggregation_result_temporal(self):
        """Test temporal aggregation."""
        agg = AggregationResult(
            experiment_id=UUID("12345678-1234-5678-1234-567812345678"),
            num_runs=30,
            total_items=30000,
            temporal_metrics=[
                {"timestamp": datetime.now() - timedelta(days=7), "accuracy": 0.80},
                {"timestamp": datetime.now() - timedelta(days=3), "accuracy": 0.83},
                {"timestamp": datetime.now(), "accuracy": 0.85},
            ],
        )
        assert len(agg.temporal_metrics) == 3
        assert agg.temporal_metrics[-1]["accuracy"] == 0.85


class TestExperimentComparison:
    """Test ExperimentComparison for comparing experiments."""

    def test_experiment_comparison_basic(self):
        """Test basic experiment comparison."""
        comp = ExperimentComparison(
            experiment_ids=[
                UUID("12345678-1234-5678-1234-567812345678"),
                UUID("87654321-4321-8765-4321-876543218765"),
            ],
            metric_comparisons={
                "accuracy": {
                    "exp1": 0.82,
                    "exp2": 0.85,
                    "difference": 0.03,
                    "percent_change": 3.66,
                },
            },
        )
        assert comp.metric_comparisons["accuracy"]["difference"] == 0.03

    def test_experiment_comparison_statistical_significance(self):
        """Test comparison with statistical significance."""
        comp = ExperimentComparison(
            experiment_ids=[
                UUID("12345678-1234-5678-1234-567812345678"),
                UUID("87654321-4321-8765-4321-876543218765"),
            ],
            metric_comparisons={
                "accuracy": {
                    "exp1": 0.82,
                    "exp2": 0.85,
                    "difference": 0.03,
                    "p_value": 0.02,
                    "is_significant": True,
                    "effect_size": 0.45,
                },
            },
        )
        assert comp.metric_comparisons["accuracy"]["is_significant"] is True

    def test_experiment_comparison_multiple_metrics(self):
        """Test comparison across multiple metrics."""
        comp = ExperimentComparison(
            experiment_ids=[
                UUID("12345678-1234-5678-1234-567812345678"),
                UUID("87654321-4321-8765-4321-876543218765"),
            ],
            metric_comparisons={
                "accuracy": {"exp1": 0.82, "exp2": 0.85},
                "hallucination_rate": {"exp1": 0.15, "exp2": 0.12},
                "response_time": {"exp1": 1500, "exp2": 1200},
                "cost": {"exp1": 0.50, "exp2": 0.60},
            },
            winner_by_metric={
                "accuracy": "exp2",
                "hallucination_rate": "exp2",
                "response_time": "exp2",
                "cost": "exp1",
            },
        )
        assert comp.winner_by_metric["cost"] == "exp1"

    def test_experiment_comparison_recommendations(self):
        """Test comparison with recommendations."""
        comp = ExperimentComparison(
            experiment_ids=[
                UUID("12345678-1234-5678-1234-567812345678"),
                UUID("87654321-4321-8765-4321-876543218765"),
            ],
            metric_comparisons={
                "accuracy": {"exp1": 0.82, "exp2": 0.85},
            },
            recommendations=[
                "Experiment 2 shows significant improvement in accuracy",
                "Consider the 20% increase in cost for 3% accuracy gain",
                "Run additional trials to confirm results",
            ],
        )
        assert len(comp.recommendations) == 3


class TestResultDiff:
    """Test ResultDiff for tracking changes between results."""

    def test_result_diff_basic(self):
        """Test basic result diff."""
        diff = ResultDiff(
            old_result_id=UUID("12345678-1234-5678-1234-567812345678"),
            new_result_id=UUID("87654321-4321-8765-4321-876543218765"),
            field_changes={
                "is_correct": {"old": False, "new": True},
                "confidence_score": {"old": 0.65, "new": 0.85},
            },
        )
        assert diff.field_changes["is_correct"]["new"] is True

    def test_result_diff_metric_changes(self):
        """Test diff with metric changes."""
        diff = ResultDiff(
            old_result_id=UUID("12345678-1234-5678-1234-567812345678"),
            new_result_id=UUID("87654321-4321-8765-4321-876543218765"),
            metric_changes={
                "accuracy": {"old": 0.80, "new": 0.85, "delta": 0.05},
                "f1_score": {"old": 0.78, "new": 0.83, "delta": 0.05},
            },
            improvement_summary="5% improvement across key metrics",
        )
        assert diff.metric_changes["accuracy"]["delta"] == 0.05

    def test_result_diff_response_changes(self):
        """Test diff with response changes."""
        diff = ResultDiff(
            old_result_id=UUID("12345678-1234-5678-1234-567812345678"),
            new_result_id=UUID("87654321-4321-8765-4321-876543218765"),
            response_diff={
                "old_response": "Paris is the capital of Italy",
                "new_response": "Paris is the capital of France",
                "changes": [
                    {"type": "substitution", "old": "Italy", "new": "France"},
                ],
            },
        )
        assert diff.response_diff["changes"][0]["new"] == "France"

    def test_result_diff_coherence_changes(self):
        """Test diff with coherence measure changes."""
        diff = ResultDiff(
            old_result_id=UUID("12345678-1234-5678-1234-567812345678"),
            new_result_id=UUID("87654321-4321-8765-4321-876543218765"),
            coherence_changes={
                CoherenceMeasure.SHOGENJI: {"old": 0.65, "new": 0.78, "delta": 0.13},
                CoherenceMeasure.FITELSON: {"old": 0.70, "new": 0.82, "delta": 0.12},
            },
        )
        assert diff.coherence_changes[CoherenceMeasure.SHOGENJI]["delta"] == 0.13


class TestMetricType:
    """Test MetricType enum and related functionality."""

    def test_metric_type_categories(self):
        """Test metric type categorization."""
        assert MetricType.ACCURACY.category == "performance"
        assert MetricType.HALLUCINATION_RATE.category == "quality"
        assert MetricType.RESPONSE_TIME.category == "efficiency"
        assert MetricType.COST.category == "resource"

    def test_metric_type_aggregation(self):
        """Test metric aggregation rules."""
        assert MetricType.ACCURACY.aggregation == "mean"
        assert MetricType.RESPONSE_TIME.aggregation == "percentile"
        assert MetricType.TOTAL_TOKENS.aggregation == "sum"
        assert MetricType.ERROR_RATE.aggregation == "mean"

    def test_metric_type_display(self):
        """Test metric display properties."""
        assert MetricType.ACCURACY.display_name == "Accuracy"
        assert MetricType.ACCURACY.unit == "%"
        assert MetricType.RESPONSE_TIME.unit == "ms"
        assert MetricType.COST.unit == "USD"

    def test_metric_type_thresholds(self):
        """Test metric threshold definitions."""
        assert MetricType.ACCURACY.good_threshold == 0.90
        assert MetricType.HALLUCINATION_RATE.good_threshold == 0.05
        assert MetricType.RESPONSE_TIME.good_threshold == 1000


class TestExperimentSerialization:
    """Test serialization and deserialization of experiment models."""

    def test_full_experiment_serialization(self):
        """Test serializing a complete experiment with all components."""
        # Create a full experiment
        tracker = ExperimentTracker(
            name="Full Serialization Test",
            benchmark_type=BenchmarkType.FAITHBENCH,
            strategy=EvaluationStrategy.COHERENCE,
            model_name="gpt-5",
        )

        run = ExperimentRun(
            experiment_id=tracker.id,
            run_number=1,
            dataset_size=100,
            status=ExperimentStatus.COMPLETED,
        )

        result = ExperimentResult(
            experiment_id=tracker.id,
            run_id=run.id,
            dataset_item_id="item_001",
            model_response="Test response",
            is_correct=True,
        )

        metrics = BenchmarkMetrics(
            experiment_id=tracker.id,
            run_id=run.id,
            total_items=100,
            correct_items=85,
        )

        # Serialize everything
        data = {
            "tracker": tracker.model_dump(),
            "run": run.model_dump(),
            "result": result.model_dump(),
            "metrics": metrics.model_dump(),
        }

        json_str = json.dumps(data, default=str)
        loaded_data = json.loads(json_str)

        # Verify deserialization
        assert loaded_data["tracker"]["name"] == "Full Serialization Test"
        assert loaded_data["metrics"]["accuracy"] == 0.85

    def test_experiment_file_persistence(self):
        """Test saving and loading experiments from files."""
        tracker = ExperimentTracker(
            name="Persistence Test",
            benchmark_type=BenchmarkType.SIMPLEQA,
            strategy=EvaluationStrategy.K_RESPONSE,
            model_name="gpt-4",
        )

        # Test JSON export
        json_data = tracker.model_dump_json(indent=2)
        assert "Persistence Test" in json_data

        # Test loading from JSON
        loaded = ExperimentTracker.model_validate_json(json_data)
        assert loaded.name == tracker.name
        assert loaded.id == tracker.id


class TestExperimentValidation:
    """Test validation rules for experiment models."""

    def test_experiment_tracker_name_validation(self):
        """Test experiment name validation."""
        with pytest.raises(ValidationError):
            ExperimentTracker(
                name="a" * 256,  # Too long
                benchmark_type=BenchmarkType.FEVER,
                strategy=EvaluationStrategy.BASELINE,
                model_name="gpt-4",
            )

    def test_experiment_run_dataset_size_validation(self):
        """Test dataset size validation."""
        with pytest.raises(ValidationError):
            ExperimentRun(
                experiment_id=UUID("12345678-1234-5678-1234-567812345678"),
                run_number=1,
                dataset_size=-10,  # Negative size
            )

    def test_experiment_result_score_validation(self):
        """Test score range validation."""
        with pytest.raises(ValidationError):
            ExperimentResult(
                experiment_id=UUID("12345678-1234-5678-1234-567812345678"),
                run_id=UUID("87654321-4321-8765-4321-876543218765"),
                dataset_item_id="item_001",
                model_response="Response",
                confidence_score=1.5,  # Out of range [0, 1]
            )

    def test_benchmark_metrics_percentage_validation(self):
        """Test percentage metric validation."""
        with pytest.raises(ValidationError):
            BenchmarkMetrics(
                experiment_id=UUID("12345678-1234-5678-1234-567812345678"),
                run_id=UUID("87654321-4321-8765-4321-876543218765"),
                total_items=100,
                correct_items=150,  # More than total
            )

    def test_statistical_summary_consistency(self):
        """Test statistical summary consistency validation."""
        with pytest.raises(ValidationError):
            StatisticalSummary(
                metric_name="test",
                mean=0.5,
                min_value=0.6,  # Min > mean
                max_value=0.8,
                count=10,
            )
