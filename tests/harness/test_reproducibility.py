"""Tests for the reproducibility validation system."""

import json

import pytest
from cosmos_coherence.harness.reproducibility import (
    BaselineMetrics,
    ReproducibilityConfig,
    ReproducibilityValidator,
)


class TestReproducibilityValidator:
    """Test the reproducibility validator functionality."""

    @pytest.fixture
    def validator_config(self):
        """Create a test configuration for the validator."""
        return ReproducibilityConfig(
            tolerance_percentage=1.0,  # 1% tolerance
            use_deterministic_seed=True,
            random_seed=42,
            compare_to_published=True,
            save_validation_report=True,
        )

    @pytest.fixture
    def validator(self, validator_config):
        """Create a ReproducibilityValidator instance."""
        return ReproducibilityValidator(validator_config)

    @pytest.fixture
    def sample_baseline_metrics(self):
        """Create sample baseline metrics for testing."""
        return BaselineMetrics(
            benchmark_name="faithbench",
            accuracy=0.854,
            f1_score=0.823,
            exact_match_rate=0.796,
            metrics={
                "precision": 0.867,
                "recall": 0.782,
                "perplexity": 12.45,
            },
            model_name="gpt-4o-mini",
            dataset_size=1000,
            paper_reference="FaithBench: Evaluating Hallucinations in LLMs (2024)",
        )

    @pytest.fixture
    def sample_current_metrics(self):
        """Create sample current run metrics for testing."""
        return BaselineMetrics(
            benchmark_name="faithbench",
            accuracy=0.848,  # Within 1% tolerance
            f1_score=0.819,  # Within 1% tolerance
            exact_match_rate=0.790,  # Within 1% tolerance
            metrics={
                "precision": 0.862,
                "recall": 0.778,
                "perplexity": 12.52,  # Within 1% tolerance of 12.45
            },
            model_name="gpt-4o-mini",
            dataset_size=1000,
            paper_reference="FaithBench: Evaluating Hallucinations in LLMs (2024)",
        )

    def test_validator_initialization(self, validator_config):
        """Test that the validator initializes correctly."""
        validator = ReproducibilityValidator(validator_config)
        assert validator.config == validator_config
        assert validator.baseline_storage_path.exists()

    def test_store_baseline_metrics(self, validator, sample_baseline_metrics, tmp_path):
        """Test storing baseline metrics to JSONL file."""
        validator.baseline_storage_path = tmp_path

        # Store the baseline metrics
        storage_path = validator.store_baseline_metrics(sample_baseline_metrics)
        assert storage_path  # Variable is used for assertion

        # Verify the file was created
        assert storage_path.exists()
        assert storage_path.suffix == ".jsonl"

        # Verify the content
        with open(storage_path, "r") as f:
            stored_data = json.loads(f.readline())
            assert stored_data["benchmark_name"] == "faithbench"
            assert stored_data["accuracy"] == 0.854
            assert stored_data["f1_score"] == 0.823

    def test_load_baseline_metrics(self, validator, sample_baseline_metrics, tmp_path):
        """Test loading baseline metrics from JSONL file."""
        validator.baseline_storage_path = tmp_path

        # Store baseline first
        validator.store_baseline_metrics(sample_baseline_metrics)

        # Load it back
        loaded_metrics = validator.load_baseline_metrics("faithbench")

        assert loaded_metrics is not None
        assert loaded_metrics.benchmark_name == sample_baseline_metrics.benchmark_name
        assert loaded_metrics.accuracy == sample_baseline_metrics.accuracy
        assert loaded_metrics.f1_score == sample_baseline_metrics.f1_score

    def test_calculate_deviation(self, validator):
        """Test deviation calculation between two values."""
        # Test positive deviation
        deviation = validator.calculate_deviation(100, 102)
        assert deviation == pytest.approx(2.0)

        # Test negative deviation
        deviation = validator.calculate_deviation(100, 98)
        assert deviation == pytest.approx(-2.0)

        # Test zero deviation
        deviation = validator.calculate_deviation(100, 100)
        assert deviation == 0.0

        # Test with zero baseline (should handle gracefully)
        deviation = validator.calculate_deviation(0, 10)
        assert deviation == float("inf")

    def test_check_tolerance(self, validator):
        """Test tolerance checking for metric deviations."""
        # Within tolerance (1%)
        assert validator.check_tolerance(0.854, 0.848, tolerance=1.0) is True
        assert validator.check_tolerance(0.854, 0.862, tolerance=1.0) is True

        # Outside tolerance
        assert validator.check_tolerance(0.854, 0.840, tolerance=1.0) is False
        assert validator.check_tolerance(0.854, 0.870, tolerance=1.0) is False

        # Exact match
        assert validator.check_tolerance(0.854, 0.854, tolerance=1.0) is True

    def test_compare_metrics(self, validator, sample_baseline_metrics, sample_current_metrics):
        """Test comprehensive metric comparison."""
        deviations = validator.compare_metrics(sample_baseline_metrics, sample_current_metrics)

        # Check that deviations are calculated for all metrics
        assert "accuracy" in deviations
        assert "f1_score" in deviations
        assert "exact_match_rate" in deviations
        assert "precision" in deviations
        assert "recall" in deviations

        # Verify deviation values
        assert deviations["accuracy"].baseline_value == 0.854
        assert deviations["accuracy"].current_value == 0.848
        assert abs(deviations["accuracy"].deviation_percentage) < 1.0
        assert deviations["accuracy"].within_tolerance is True

    def test_validate_reproducibility_pass(
        self, validator, sample_baseline_metrics, sample_current_metrics
    ):
        """Test successful reproducibility validation."""
        result = validator.validate_reproducibility(sample_baseline_metrics, sample_current_metrics)

        assert result.validation_passed is True
        assert result.overall_deviation < 1.0
        assert len(result.failed_metrics) == 0
        assert "All metrics within tolerance" in result.summary

    def test_validate_reproducibility_fail(self, validator, sample_baseline_metrics):
        """Test failed reproducibility validation."""
        # Create metrics with large deviations
        bad_metrics = BaselineMetrics(
            benchmark_name="faithbench",
            accuracy=0.750,  # >10% deviation
            f1_score=0.700,  # >10% deviation
            exact_match_rate=0.690,  # >10% deviation
            metrics={
                "precision": 0.750,
                "recall": 0.650,
                "perplexity": 15.0,
            },
            model_name="gpt-4o-mini",
            dataset_size=1000,
            paper_reference="FaithBench: Evaluating Hallucinations in LLMs (2024)",
        )

        result = validator.validate_reproducibility(sample_baseline_metrics, bad_metrics)

        assert result.validation_passed is False
        assert result.overall_deviation > 1.0
        assert len(result.failed_metrics) > 0
        assert "accuracy" in result.failed_metrics
        assert "Validation failed" in result.summary

    def test_generate_comparison_report(
        self, validator, sample_baseline_metrics, sample_current_metrics
    ):
        """Test comparison report generation."""
        validation_result = validator.validate_reproducibility(
            sample_baseline_metrics, sample_current_metrics
        )

        report = validator.generate_comparison_report(
            sample_baseline_metrics,
            sample_current_metrics,
            validation_result,
        )

        assert report.benchmark_name == "faithbench"
        assert report.validation_passed is True
        assert report.tolerance_used == 1.0
        assert len(report.metric_comparisons) > 0
        assert len(report.recommendations) >= 0

    def test_deterministic_execution_settings(self, validator):
        """Test that deterministic execution settings are properly configured."""
        settings = validator.get_deterministic_settings()

        assert settings["temperature"] == 0.0
        assert settings["top_p"] == 1.0
        assert settings["seed"] == 42
        assert settings["max_tokens"] is not None

    def test_save_validation_report(
        self, validator, sample_baseline_metrics, sample_current_metrics, tmp_path
    ):
        """Test saving validation report to file."""
        validator.report_storage_path = tmp_path

        validation_result = validator.validate_reproducibility(
            sample_baseline_metrics, sample_current_metrics
        )

        report = validator.generate_comparison_report(
            sample_baseline_metrics,
            sample_current_metrics,
            validation_result,
        )

        report_path = validator.save_validation_report(report)

        assert report_path.exists()
        assert report_path.suffix == ".json"

        # Verify content
        with open(report_path, "r") as f:
            saved_report = json.load(f)
            assert saved_report["benchmark_name"] == "faithbench"
            assert saved_report["validation_passed"] is True

    def test_statistical_significance(self, validator):
        """Test statistical significance testing for deviations."""
        # Small sample size - not significant
        is_significant = validator.test_statistical_significance(
            baseline_mean=0.85,
            current_mean=0.84,
            baseline_std=0.05,
            current_std=0.05,
            n_samples=10,
        )
        assert is_significant is False

        # Large sample size with clear difference - significant
        is_significant = validator.test_statistical_significance(
            baseline_mean=0.85,
            current_mean=0.80,
            baseline_std=0.02,
            current_std=0.02,
            n_samples=1000,
        )
        assert is_significant is True

    def test_handle_missing_baseline(self, validator):
        """Test handling of missing baseline metrics."""
        # Try to load non-existent baseline
        metrics = validator.load_baseline_metrics("non_existent_benchmark")
        assert metrics is None

        # Validation should fail gracefully
        current_metrics = BaselineMetrics(
            benchmark_name="non_existent",
            accuracy=0.85,
            f1_score=0.82,
            exact_match_rate=0.79,
            metrics={},
            model_name="gpt-4o-mini",
            dataset_size=100,
            paper_reference="Test",
        )

        with pytest.raises(ValueError, match="No baseline metrics found"):
            validator.validate_against_published(current_metrics)

    def test_multiple_benchmark_validation(self, validator, tmp_path):
        """Test validating multiple benchmarks in sequence."""
        validator.baseline_storage_path = tmp_path

        benchmarks = ["faithbench", "simpleqa", "truthfulqa"]

        for benchmark in benchmarks:
            baseline = BaselineMetrics(
                benchmark_name=benchmark,
                accuracy=0.85,
                f1_score=0.82,
                exact_match_rate=0.79,
                metrics={"precision": 0.86, "recall": 0.78},
                model_name="gpt-4o-mini",
                dataset_size=1000,
                paper_reference=f"{benchmark} paper",
            )

            validator.store_baseline_metrics(baseline)

        # Verify all baselines are stored
        for benchmark in benchmarks:
            loaded = validator.load_baseline_metrics(benchmark)
            assert loaded is not None
            assert loaded.benchmark_name == benchmark

    def test_edge_cases(self, validator):
        """Test edge cases in validation."""
        # Test with None values
        assert validator.calculate_deviation(None, 0.85) == float("inf")
        assert validator.calculate_deviation(0.85, None) == float("inf")

        # Test with negative values (shouldn't happen but handle gracefully)
        deviation = validator.calculate_deviation(-0.5, 0.5)
        # From -0.5 to 0.5 is a change of 1.0, divided by abs(-0.5) = 0.5, gives 200%
        assert deviation == 200.0

        # Test with very small values
        deviation = validator.calculate_deviation(0.001, 0.0011)
        assert abs(deviation - 10.0) < 0.01
