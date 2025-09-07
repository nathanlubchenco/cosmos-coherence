"""Reproducibility validation system for benchmark evaluation."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats

logger = logging.getLogger(__name__)


class ReproducibilityConfig(BaseModel):
    """Configuration for reproducibility validation."""

    validate_before_experiments: bool = Field(
        default=True, description="Require validation before allowing experiments"
    )
    tolerance_percentage: float = Field(
        default=1.0, description="Maximum allowed deviation from baseline (%)"
    )
    use_deterministic_seed: bool = Field(
        default=True, description="Use fixed random seed for reproducibility"
    )
    random_seed: int = Field(default=42, description="Random seed for deterministic execution")
    compare_to_published: bool = Field(
        default=True, description="Compare results to published baselines"
    )
    save_validation_report: bool = Field(
        default=True, description="Save validation reports to disk"
    )
    significance_level: float = Field(
        default=0.05, description="Statistical significance level for deviation testing"
    )


class BaselineMetrics(BaseModel):
    """Baseline metrics for a benchmark."""

    benchmark_name: str
    accuracy: float
    f1_score: Optional[float] = None
    exact_match_rate: float
    metrics: Dict[str, float] = Field(default_factory=dict)
    model_name: str
    dataset_size: int
    paper_reference: str
    timestamp: datetime = Field(default_factory=datetime.now)

    def to_jsonl(self) -> str:
        """Convert to JSONL format."""
        data = self.model_dump()
        data["timestamp"] = data["timestamp"].isoformat()
        return json.dumps(data)

    @classmethod
    def from_jsonl(cls, jsonl_str: str) -> "BaselineMetrics":
        """Create from JSONL string."""
        data = json.loads(jsonl_str)
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class DeviationDetail(BaseModel):
    """Details about a metric deviation."""

    metric_name: str
    baseline_value: float
    current_value: float
    deviation_percentage: float
    within_tolerance: bool
    is_statistically_significant: bool = False


class ValidationResult(BaseModel):
    """Result of reproducibility validation."""

    validation_passed: bool
    overall_deviation: float
    metric_deviations: Dict[str, DeviationDetail]
    failed_metrics: List[str]
    summary: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ComparisonReport(BaseModel):
    """Detailed comparison report for reproducibility validation."""

    benchmark_name: str
    validation_passed: bool
    our_metrics: Dict[str, float]
    published_metrics: Dict[str, float]
    deviations: Dict[str, float]
    tolerance_used: float
    metric_comparisons: List[DeviationDetail]
    recommendations: List[str]
    timestamp: datetime = Field(default_factory=datetime.now)


class ReproducibilityValidator:
    """Validates benchmark reproducibility against published baselines."""

    def __init__(self, config: ReproducibilityConfig):
        """Initialize the reproducibility validator.

        Args:
            config: Configuration for validation
        """
        self.config = config
        self.baseline_storage_path = Path("data/baselines")
        self.baseline_storage_path.mkdir(parents=True, exist_ok=True)
        self.report_storage_path = Path("data/validation_reports")
        self.report_storage_path.mkdir(parents=True, exist_ok=True)

    def store_baseline_metrics(self, metrics: BaselineMetrics) -> Path:
        """Store baseline metrics to JSONL file.

        Args:
            metrics: Baseline metrics to store

        Returns:
            Path to the stored file
        """
        filename = f"{metrics.benchmark_name}_baseline.jsonl"
        filepath = self.baseline_storage_path / filename

        with open(filepath, "a") as f:
            f.write(metrics.to_jsonl() + "\n")

        logger.info(f"Stored baseline metrics for {metrics.benchmark_name} to {filepath}")
        return filepath

    def load_baseline_metrics(self, benchmark_name: str) -> Optional[BaselineMetrics]:
        """Load baseline metrics from storage.

        Args:
            benchmark_name: Name of the benchmark

        Returns:
            Baseline metrics if found, None otherwise
        """
        filename = f"{benchmark_name}_baseline.jsonl"
        filepath = self.baseline_storage_path / filename

        if not filepath.exists():
            logger.warning(f"No baseline found for {benchmark_name}")
            return None

        # Get the most recent baseline
        with open(filepath, "r") as f:
            lines = f.readlines()
            if lines:
                return BaselineMetrics.from_jsonl(lines[-1])

        return None

    def calculate_deviation(self, baseline: Optional[float], current: Optional[float]) -> float:
        """Calculate percentage deviation between baseline and current value.

        Args:
            baseline: Baseline value
            current: Current value

        Returns:
            Percentage deviation
        """
        if baseline is None or current is None:
            return float("inf")

        if baseline == 0:
            return float("inf") if current != 0 else 0.0

        return ((current - baseline) / abs(baseline)) * 100

    def check_tolerance(self, baseline: float, current: float, tolerance: float) -> bool:
        """Check if deviation is within tolerance.

        Args:
            baseline: Baseline value
            current: Current value
            tolerance: Tolerance percentage

        Returns:
            True if within tolerance
        """
        deviation = abs(self.calculate_deviation(baseline, current))
        return deviation <= tolerance

    def test_statistical_significance(
        self,
        baseline_mean: float,
        current_mean: float,
        baseline_std: float,
        current_std: float,
        n_samples: int,
    ) -> bool:
        """Test if deviation is statistically significant.

        Args:
            baseline_mean: Mean of baseline distribution
            current_mean: Mean of current distribution
            baseline_std: Standard deviation of baseline
            current_std: Standard deviation of current
            n_samples: Number of samples

        Returns:
            True if statistically significant
        """
        if n_samples < 2:
            return False

        # Perform Welch's t-test (doesn't assume equal variances)
        t_stat = (current_mean - baseline_mean) / np.sqrt(
            (baseline_std**2 / n_samples) + (current_std**2 / n_samples)
        )

        # Calculate degrees of freedom using Welch-Satterthwaite equation
        df = ((baseline_std**2 / n_samples) + (current_std**2 / n_samples)) ** 2 / (
            (baseline_std**2 / n_samples) ** 2 / (n_samples - 1)
            + (current_std**2 / n_samples) ** 2 / (n_samples - 1)
        )

        # Two-tailed test
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

        return bool(p_value < self.config.significance_level)

    def compare_metrics(
        self, baseline: BaselineMetrics, current: BaselineMetrics
    ) -> Dict[str, DeviationDetail]:
        """Compare baseline and current metrics in detail.

        Args:
            baseline: Baseline metrics
            current: Current metrics

        Returns:
            Dictionary of deviation details by metric name
        """
        deviations = {}

        # Compare primary metrics
        primary_metrics = [
            ("accuracy", baseline.accuracy, current.accuracy),
            ("f1_score", baseline.f1_score, current.f1_score),
            ("exact_match_rate", baseline.exact_match_rate, current.exact_match_rate),
        ]

        for metric_name, baseline_val, current_val in primary_metrics:
            if baseline_val is not None and current_val is not None:
                deviation_pct = self.calculate_deviation(baseline_val, current_val)
                deviations[metric_name] = DeviationDetail(
                    metric_name=metric_name,
                    baseline_value=baseline_val,
                    current_value=current_val,
                    deviation_percentage=deviation_pct,
                    within_tolerance=self.check_tolerance(
                        baseline_val, current_val, self.config.tolerance_percentage
                    ),
                )

        # Compare additional metrics
        for metric_name, baseline_val in baseline.metrics.items():
            if metric_name in current.metrics:
                current_val = current.metrics[metric_name]
                deviation_pct = self.calculate_deviation(baseline_val, current_val)
                deviations[metric_name] = DeviationDetail(
                    metric_name=metric_name,
                    baseline_value=baseline_val,
                    current_value=current_val,
                    deviation_percentage=deviation_pct,
                    within_tolerance=self.check_tolerance(
                        baseline_val, current_val, self.config.tolerance_percentage
                    ),
                )

        return deviations

    def validate_reproducibility(
        self, baseline: BaselineMetrics, current: BaselineMetrics
    ) -> ValidationResult:
        """Validate reproducibility of current results against baseline.

        Args:
            baseline: Baseline metrics
            current: Current metrics

        Returns:
            Validation result
        """
        deviations = self.compare_metrics(baseline, current)

        # Check which metrics failed tolerance
        failed_metrics = [
            name for name, detail in deviations.items() if not detail.within_tolerance
        ]

        # Calculate overall deviation (average of absolute deviations)
        deviation_values = [abs(detail.deviation_percentage) for detail in deviations.values()]
        overall_deviation = np.mean(deviation_values) if deviation_values else 0.0

        validation_passed = len(failed_metrics) == 0

        if validation_passed:
            summary = f"All metrics within tolerance ({self.config.tolerance_percentage}%)"
        else:
            summary = f"Validation failed: {len(failed_metrics)} metrics exceed tolerance"

        return ValidationResult(
            validation_passed=validation_passed,
            overall_deviation=overall_deviation,
            metric_deviations=deviations,
            failed_metrics=failed_metrics,
            summary=summary,
        )

    def validate_against_published(self, current: BaselineMetrics) -> ValidationResult:
        """Validate current results against published baseline.

        Args:
            current: Current metrics

        Returns:
            Validation result

        Raises:
            ValueError: If no baseline found for benchmark
        """
        baseline = self.load_baseline_metrics(current.benchmark_name)

        if baseline is None:
            raise ValueError(f"No baseline metrics found for {current.benchmark_name}")

        return self.validate_reproducibility(baseline, current)

    def generate_comparison_report(
        self,
        baseline: BaselineMetrics,
        current: BaselineMetrics,
        validation_result: ValidationResult,
    ) -> ComparisonReport:
        """Generate detailed comparison report.

        Args:
            baseline: Baseline metrics
            current: Current metrics
            validation_result: Validation result

        Returns:
            Comparison report
        """
        # Extract metrics for report
        our_metrics = {
            "accuracy": current.accuracy,
            "f1_score": current.f1_score,
            "exact_match_rate": current.exact_match_rate,
            **current.metrics,
        }

        published_metrics = {
            "accuracy": baseline.accuracy,
            "f1_score": baseline.f1_score,
            "exact_match_rate": baseline.exact_match_rate,
            **baseline.metrics,
        }

        # Calculate deviations
        deviations = {
            name: detail.deviation_percentage
            for name, detail in validation_result.metric_deviations.items()
        }

        # Generate recommendations
        recommendations = []
        if not validation_result.validation_passed:
            recommendations.append(
                f"Reproducibility validation failed with {len(validation_result.failed_metrics)} "
                f"metrics exceeding {self.config.tolerance_percentage}% tolerance"
            )

            for metric in validation_result.failed_metrics:
                detail = validation_result.metric_deviations[metric]
                recommendations.append(
                    f"- {metric}: {detail.deviation_percentage:.2f}% deviation "
                    f"(baseline: {detail.baseline_value:.3f}, current: {detail.current_value:.3f})"
                )

            recommendations.append(
                "Consider reviewing: prompt formatting, evaluation logic, "
                "model parameters, and dataset preprocessing"
            )
        else:
            recommendations.append(
                f"Successfully reproduced baseline within "
                f"{self.config.tolerance_percentage}% tolerance"
            )
            recommendations.append("Ready to proceed with temperature-variant experiments")

        return ComparisonReport(
            benchmark_name=current.benchmark_name,
            validation_passed=validation_result.validation_passed,
            our_metrics=our_metrics,
            published_metrics=published_metrics,
            deviations=deviations,
            tolerance_used=self.config.tolerance_percentage,
            metric_comparisons=list(validation_result.metric_deviations.values()),
            recommendations=recommendations,
        )

    def save_validation_report(self, report: ComparisonReport) -> Path:
        """Save validation report to file.

        Args:
            report: Comparison report to save

        Returns:
            Path to saved report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report.benchmark_name}_validation_{timestamp}.json"
        filepath = self.report_storage_path / filename

        report_dict = report.model_dump()
        report_dict["timestamp"] = report_dict["timestamp"].isoformat()

        # Convert DeviationDetail objects to dicts
        report_dict["metric_comparisons"] = [
            detail.model_dump() if hasattr(detail, "model_dump") else detail
            for detail in report_dict["metric_comparisons"]
        ]

        with open(filepath, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)

        logger.info(f"Saved validation report to {filepath}")
        return filepath

    def get_deterministic_settings(self) -> Dict[str, Any]:
        """Get settings for deterministic execution.

        Returns:
            Dictionary of deterministic execution settings
        """
        return {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": self.config.random_seed,
            "max_tokens": 2048,  # Fixed to ensure consistency
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
