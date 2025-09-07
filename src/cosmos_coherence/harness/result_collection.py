"""Result collection and reporting system for benchmark harness."""

import csv
import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats

from cosmos_coherence.harness.benchmark_runner import ExecutionResult

logger = logging.getLogger(__name__)


class ExportFormat(str, Enum):
    """Supported export formats."""

    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    MARKDOWN = "markdown"
    HTML = "html"


class ResultFilter(BaseModel):
    """Filter criteria for result selection."""

    benchmark_pattern: Optional[str] = None
    min_accuracy: Optional[float] = None
    max_accuracy: Optional[float] = None
    min_items: Optional[int] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    temperature: Optional[float] = None


class StatisticalSummary(BaseModel):
    """Statistical summary of results."""

    count: int
    mean_accuracy: float
    std_accuracy: float
    min_accuracy: float
    max_accuracy: float
    median: float
    percentile_25: float
    percentile_75: float
    mean_f1: Optional[float] = None
    mean_precision: Optional[float] = None
    mean_recall: Optional[float] = None
    mean_execution_time: float
    total_items_processed: int

    @classmethod
    def from_results(cls, results: List[ExecutionResult]) -> "StatisticalSummary":
        """Create summary from execution results."""
        if not results:
            raise ValueError("Cannot create summary from empty results")

        accuracies = [r.metrics.get("accuracy", 0) for r in results]
        f1_scores = [r.metrics.get("f1_score") for r in results if "f1_score" in r.metrics]
        precisions = [r.metrics.get("precision") for r in results if "precision" in r.metrics]
        recalls = [r.metrics.get("recall") for r in results if "recall" in r.metrics]
        exec_times = [r.execution_time for r in results]
        total_items = sum(r.total_items for r in results)

        return cls(
            count=len(results),
            mean_accuracy=float(np.mean(accuracies)),
            std_accuracy=float(np.std(accuracies)),
            min_accuracy=float(np.min(accuracies)),
            max_accuracy=float(np.max(accuracies)),
            median=float(np.median(accuracies)),
            percentile_25=float(np.percentile(accuracies, 25)),
            percentile_75=float(np.percentile(accuracies, 75)),
            mean_f1=float(np.mean(f1_scores)) if f1_scores else None,
            mean_precision=float(np.mean(precisions)) if precisions else None,
            mean_recall=float(np.mean(recalls)) if recalls else None,
            mean_execution_time=float(np.mean(exec_times)),
            total_items_processed=total_items,
        )

    def get_confidence_interval(self, confidence: float = 0.95) -> Dict[str, float]:
        """Calculate confidence interval for mean accuracy."""
        # Using t-distribution for small samples
        sem = self.std_accuracy / np.sqrt(self.count)
        margin = sem * stats.t.ppf((1 + confidence) / 2, self.count - 1)

        return {
            "lower": self.mean_accuracy - margin,
            "upper": self.mean_accuracy + margin,
            "confidence": confidence,
            "margin_of_error": margin,
        }

    def compare_to(self, other: "StatisticalSummary") -> Dict[str, Any]:
        """Compare this summary to another using statistical tests."""
        # Welch's t-test for unequal variances
        t_stat, p_value = stats.ttest_ind_from_stats(
            self.mean_accuracy,
            self.std_accuracy,
            self.count,
            other.mean_accuracy,
            other.std_accuracy,
            other.count,
            equal_var=False,
        )

        mean_diff = self.mean_accuracy - other.mean_accuracy

        return {
            "mean_difference": mean_diff,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "effect_size": mean_diff
            / np.sqrt((self.std_accuracy**2 + other.std_accuracy**2) / 2),
        }


class ResultCollector:
    """Collects and manages benchmark execution results."""

    def __init__(self):
        """Initialize result collector."""
        self.results: List[ExecutionResult] = []
        self.metadata: Dict[str, Any] = {}

    def add_result(self, result: ExecutionResult) -> None:
        """Add a result to the collection."""
        self.results.append(result)
        logger.info(
            f"Added result for {result.benchmark_name}: accuracy={result.metrics.get('accuracy')}"
        )

    def get_results_by_benchmark(self, benchmark_name: str) -> List[ExecutionResult]:
        """Get all results for a specific benchmark."""
        return [r for r in self.results if r.benchmark_name == benchmark_name]

    def get_latest_result(self, benchmark_name: str) -> Optional[ExecutionResult]:
        """Get the most recent result for a benchmark."""
        benchmark_results = self.get_results_by_benchmark(benchmark_name)
        if not benchmark_results:
            return None
        return max(benchmark_results, key=lambda r: r.timestamp)

    def calculate_aggregate_metrics(self, benchmark_name: str) -> Dict[str, float]:
        """Calculate aggregate metrics for a benchmark."""
        results = self.get_results_by_benchmark(benchmark_name)
        if not results:
            return {}

        metrics_dict: Dict[str, List[float]] = {}

        # Collect all metric values
        for result in results:
            for metric_name, value in result.metrics.items():
                if metric_name not in metrics_dict:
                    metrics_dict[metric_name] = []
                metrics_dict[metric_name].append(value)

        # Calculate statistics for each metric
        aggregates = {}
        for metric_name, values in metrics_dict.items():
            aggregates[f"mean_{metric_name}"] = float(np.mean(values))
            aggregates[f"std_{metric_name}"] = float(np.std(values))
            aggregates[f"min_{metric_name}"] = float(np.min(values))
            aggregates[f"max_{metric_name}"] = float(np.max(values))

        return aggregates

    def filter_results(self, filter_config: ResultFilter) -> List[ExecutionResult]:
        """Filter results based on criteria."""
        filtered = self.results.copy()

        if filter_config.benchmark_pattern:
            filtered = [r for r in filtered if filter_config.benchmark_pattern in r.benchmark_name]

        if filter_config.min_accuracy is not None:
            filtered = [
                r for r in filtered if r.metrics.get("accuracy", 0) >= filter_config.min_accuracy
            ]

        if filter_config.max_accuracy is not None:
            filtered = [
                r for r in filtered if r.metrics.get("accuracy", 0) <= filter_config.max_accuracy
            ]

        if filter_config.min_items is not None:
            filtered = [r for r in filtered if r.total_items >= filter_config.min_items]

        if filter_config.date_from:
            filtered = [r for r in filtered if r.timestamp >= filter_config.date_from]

        if filter_config.date_to:
            filtered = [r for r in filtered if r.timestamp <= filter_config.date_to]

        if filter_config.temperature is not None:
            filtered = [
                r
                for r in filtered
                if r.context and r.context.get("temperature") == filter_config.temperature
            ]

        return filtered

    def clear(self) -> None:
        """Clear all results and metadata."""
        self.results.clear()
        self.metadata.clear()


class ResultStorage:
    """Handles storage and retrieval of results."""

    def __init__(self, base_path: Path):
        """Initialize storage with base path."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_result(
        self, result: ExecutionResult, format: ExportFormat = ExportFormat.JSON
    ) -> Path:
        """Save a single result to file."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.benchmark_name}_{timestamp}.{format.value}"
        file_path = self.base_path / filename

        if format == ExportFormat.JSON:
            with open(file_path, "w") as f:
                json.dump(result.model_dump(mode="json"), f, indent=2, default=str)
        elif format == ExportFormat.JSONL:
            with open(file_path, "w") as f:
                f.write(result.model_dump_json() + "\n")
        else:
            raise ValueError(f"Unsupported format for single result: {format}")

        logger.info(f"Saved result to {file_path}")
        return file_path

    def save_multiple_results(
        self, results: List[ExecutionResult], format: ExportFormat = ExportFormat.JSONL
    ) -> Path:
        """Save multiple results to file."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.{format.value}"
        file_path = self.base_path / filename

        if format == ExportFormat.JSONL:
            with open(file_path, "w") as f:
                for result in results:
                    f.write(result.model_dump_json() + "\n")
        elif format == ExportFormat.JSON:
            with open(file_path, "w") as f:
                data = [r.model_dump(mode="json") for r in results]
                json.dump(data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format for multiple results: {format}")

        logger.info(f"Saved {len(results)} results to {file_path}")
        return file_path

    def load_result(self, file_path: Path) -> ExecutionResult:
        """Load a result from file."""
        with open(file_path) as f:
            if file_path.suffix == ".jsonl":
                line = f.readline()
                data = json.loads(line)
            else:
                data = json.load(f)

            # Handle the first item if it's a list
            if isinstance(data, list):
                data = data[0]

            return ExecutionResult(**data)

    def list_results(self) -> List[Path]:
        """List all stored result files."""
        patterns = ["*.json", "*.jsonl"]
        files: List[Path] = []
        for pattern in patterns:
            files.extend(self.base_path.glob(pattern))
        return sorted(files)

    def export_to_csv(self, results: List[ExecutionResult], filename: str) -> Path:
        """Export results to CSV format."""
        csv_path = self.base_path / filename

        if not results:
            raise ValueError("No results to export")

        # Flatten result data for CSV
        rows = []
        for result in results:
            row = {
                "benchmark_name": result.benchmark_name,
                "timestamp": result.timestamp.isoformat(),
                "total_items": result.total_items,
                "successful_items": result.successful_items,
                "failed_items": result.failed_items,
                "execution_time": result.execution_time,
            }
            # Add metrics
            for metric_name, value in result.metrics.items():
                row[metric_name] = value
            # Add context if available
            if result.context:
                for key, value in result.context.items():
                    if isinstance(value, (str, int, float)):
                        row[f"context_{key}"] = value
            rows.append(row)

        # Write CSV
        if rows:
            fieldnames = list(rows[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        logger.info(f"Exported {len(results)} results to {csv_path}")
        return csv_path


class ComparisonReport(BaseModel):
    """Report comparing two benchmarks."""

    benchmark_a: str
    benchmark_b: str
    metric_comparisons: Dict[str, Dict[str, Any]]  # Changed to Any to allow mixed types
    performance_difference: Dict[str, float]
    statistical_significance: Dict[str, bool]


class BenchmarkReport(BaseModel):
    """Report for a single benchmark."""

    benchmark_name: str
    num_runs: int
    best_result: Optional[ExecutionResult]
    worst_result: Optional[ExecutionResult]
    statistics: Optional[StatisticalSummary]
    trends: Dict[str, Any] = Field(default_factory=dict)


class SummaryReport(BaseModel):
    """Overall summary report."""

    total_experiments: int
    benchmarks_tested: List[str]
    overall_statistics: Optional[StatisticalSummary]
    per_benchmark_stats: Optional[Dict[str, StatisticalSummary]]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BenchmarkComparison:
    """Utilities for comparing benchmark results."""

    @staticmethod
    def compare(
        results_a: List[ExecutionResult], results_b: List[ExecutionResult]
    ) -> Dict[str, Any]:
        """Compare two sets of benchmark results."""
        if not results_a or not results_b:
            raise ValueError("Cannot compare empty result sets")

        summary_a = StatisticalSummary.from_results(results_a)
        summary_b = StatisticalSummary.from_results(results_b)

        # Determine winner and improvements
        metrics_better = {}
        improvement_percentage = {}

        for metric in ["accuracy", "f1", "precision", "recall"]:
            if hasattr(summary_a, f"mean_{metric}") and hasattr(summary_b, f"mean_{metric}"):
                val_a = getattr(summary_a, f"mean_{metric}")
                val_b = getattr(summary_b, f"mean_{metric}")
                if val_a and val_b:
                    metrics_better[metric] = "a" if val_a > val_b else "b"
                    improvement_percentage[metric] = ((val_a - val_b) / val_b) * 100

        winner = "a" if summary_a.mean_accuracy > summary_b.mean_accuracy else "b"

        return {
            "winner": winner,
            "metrics_better": metrics_better,
            "improvement_percentage": improvement_percentage,
            "statistical_comparison": summary_a.compare_to(summary_b),
        }

    @staticmethod
    def compare_multiple(all_results: Dict[str, List[ExecutionResult]]) -> Dict[str, Any]:
        """Compare multiple benchmark result sets."""
        summaries = {
            name: StatisticalSummary.from_results(results) for name, results in all_results.items()
        }

        # Rank by accuracy
        rankings = {
            "accuracy": sorted(
                summaries.keys(), key=lambda k: summaries[k].mean_accuracy, reverse=True
            )
        }

        # Pairwise comparisons
        pairwise = []
        names = list(all_results.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                comparison = BenchmarkComparison.compare(
                    all_results[names[i]], all_results[names[j]]
                )
                pairwise.append(
                    {
                        "pair": (names[i], names[j]),
                        "winner": names[i] if comparison["winner"] == "a" else names[j],
                        "details": comparison,
                    }
                )

        return {
            "rankings": rankings,
            "summaries": summaries,
            "pairwise_comparisons": pairwise,
        }


class ResultReporter:
    """Generates reports from benchmark results."""

    def generate_summary_report(self, results: List[ExecutionResult]) -> SummaryReport:
        """Generate overall summary report."""
        if not results:
            raise ValueError("No results to report")

        benchmarks = list(set(r.benchmark_name for r in results))

        per_benchmark_stats = {}
        for benchmark in benchmarks:
            benchmark_results = [r for r in results if r.benchmark_name == benchmark]
            if benchmark_results:
                per_benchmark_stats[benchmark] = StatisticalSummary.from_results(benchmark_results)

        return SummaryReport(
            total_experiments=len(results),
            benchmarks_tested=benchmarks,
            overall_statistics=StatisticalSummary.from_results(results),
            per_benchmark_stats=per_benchmark_stats,
        )

    def generate_benchmark_report(
        self, benchmark_name: str, results: List[ExecutionResult]
    ) -> BenchmarkReport:
        """Generate report for a specific benchmark."""
        if not results:
            raise ValueError("No results for benchmark")

        # Find best and worst by accuracy
        sorted_results = sorted(results, key=lambda r: r.metrics.get("accuracy", 0))

        return BenchmarkReport(
            benchmark_name=benchmark_name,
            num_runs=len(results),
            best_result=sorted_results[-1] if sorted_results else None,
            worst_result=sorted_results[0] if sorted_results else None,
            statistics=StatisticalSummary.from_results(results),
        )

    def generate_comparison_report(
        self,
        name_a: str,
        results_a: List[ExecutionResult],
        name_b: str,
        results_b: List[ExecutionResult],
    ) -> ComparisonReport:
        """Generate comparison report between two benchmarks."""
        comparison = BenchmarkComparison.compare(results_a, results_b)

        return ComparisonReport(
            benchmark_a=name_a,
            benchmark_b=name_b,
            metric_comparisons={
                metric: {"winner": winner, "difference": diff}
                for metric, winner in comparison.get("metrics_better", {}).items()
                for diff in [comparison.get("improvement_percentage", {}).get(metric, 0)]
            },
            performance_difference=comparison.get("improvement_percentage", {}),
            statistical_significance={
                "overall": comparison.get("statistical_comparison", {}).get("significant", False)
            },
        )

    def format_as_markdown(self, report: SummaryReport) -> str:
        """Format report as markdown."""
        lines = ["# Benchmark Report", ""]
        lines.append(f"Generated: {report.timestamp.isoformat()}")
        lines.append(f"Total Experiments: {report.total_experiments}")
        lines.append("")

        lines.append("## Overall Statistics")
        if report.overall_statistics:
            stats = report.overall_statistics
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Mean Accuracy | {stats.mean_accuracy:.4f} |")
            lines.append(f"| Std Deviation | {stats.std_accuracy:.4f} |")
            lines.append(f"| Min Accuracy | {stats.min_accuracy:.4f} |")
            lines.append(f"| Max Accuracy | {stats.max_accuracy:.4f} |")
            lines.append(f"| Median | {stats.median:.4f} |")
            lines.append("")

        lines.append("## Per-Benchmark Results")
        if report.per_benchmark_stats:
            for benchmark, stats in report.per_benchmark_stats.items():
                lines.append(f"\n### {benchmark}")
                lines.append(f"- Mean Accuracy: {stats.mean_accuracy:.4f}")
                lines.append(f"- Items Processed: {stats.total_items_processed}")
                lines.append(f"- Mean Execution Time: {stats.mean_execution_time:.2f}s")

        return "\n".join(lines)

    def format_as_html(self, report: SummaryReport) -> str:
        """Format report as HTML."""
        html = ["<html><head><title>Benchmark Report</title></head><body>"]
        html.append("<h1>Benchmark Report</h1>")
        html.append(f"<p>Generated: {report.timestamp.isoformat()}</p>")
        html.append(f"<p>Total Experiments: {report.total_experiments}</p>")

        if report.overall_statistics:
            stats = report.overall_statistics
            html.append("<h2>Overall Statistics</h2>")
            html.append("<table border='1'>")
            html.append("<tr><th>Metric</th><th>Value</th></tr>")
            html.append(f"<tr><td>Mean Accuracy</td><td>{stats.mean_accuracy:.4f}</td></tr>")
            html.append(f"<tr><td>Std Deviation</td><td>{stats.std_accuracy:.4f}</td></tr>")
            html.append(f"<tr><td>Min Accuracy</td><td>{stats.min_accuracy:.4f}</td></tr>")
            html.append(f"<tr><td>Max Accuracy</td><td>{stats.max_accuracy:.4f}</td></tr>")
            html.append("</table>")

        html.append("</body></html>")
        return "\n".join(html)

    def save_report(self, report: SummaryReport, path: Path, format: str = "markdown") -> None:
        """Save report to file."""
        if format == "markdown":
            content = self.format_as_markdown(report)
        elif format == "html":
            content = self.format_as_html(report)
        elif format == "json":
            content = report.model_dump_json(indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        with open(path, "w") as f:
            f.write(content)

    def analyze_by_temperature(
        self, results: List[ExecutionResult]
    ) -> Dict[float, Dict[str, float]]:
        """Analyze results grouped by temperature setting."""
        temp_groups: Dict[float, List[ExecutionResult]] = {}

        for result in results:
            if result.context and "temperature" in result.context:
                temp = result.context["temperature"]
                if temp not in temp_groups:
                    temp_groups[temp] = []
                temp_groups[temp].append(result)

        analysis = {}
        for temp, temp_results in temp_groups.items():
            if temp_results:
                summary = StatisticalSummary.from_results(temp_results)
                analysis[temp] = {
                    "mean_accuracy": summary.mean_accuracy,
                    "std_accuracy": summary.std_accuracy,
                    "count": summary.count,
                }

        return analysis
