"""Tests for result collection and reporting system."""

import json
from datetime import datetime

import pytest
from cosmos_coherence.harness.benchmark_runner import ExecutionResult
from cosmos_coherence.harness.result_collection import (
    BenchmarkComparison,
    ExportFormat,
    ResultCollector,
    ResultFilter,
    ResultReporter,
    ResultStorage,
    StatisticalSummary,
)


class TestResultCollector:
    """Test the result collector functionality."""

    @pytest.fixture
    def sample_execution_result(self):
        """Create a sample execution result."""
        return ExecutionResult(
            benchmark_name="test_benchmark",
            total_items=100,
            successful_items=95,
            failed_items=5,
            metrics={
                "accuracy": 0.85,
                "f1_score": 0.82,
                "precision": 0.88,
                "recall": 0.79,
            },
            execution_time=120.5,
            item_results=[],
            context={
                "model": "gpt-4o-mini",
                "temperature": 0.7,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    @pytest.fixture
    def collector(self):
        """Create a result collector instance."""
        return ResultCollector()

    def test_collector_initialization(self, collector):
        """Test collector initializes correctly."""
        assert collector.results == []
        assert collector.metadata == {}

    def test_add_result(self, collector, sample_execution_result):
        """Test adding a result to collector."""
        collector.add_result(sample_execution_result)
        assert len(collector.results) == 1
        assert collector.results[0] == sample_execution_result

    def test_add_multiple_results(self, collector):
        """Test adding multiple results."""
        for i in range(5):
            result = ExecutionResult(
                benchmark_name="test",
                total_items=100,
                successful_items=85 + i,
                failed_items=15 - i,
                metrics={"accuracy": 0.8 + i * 0.01},
                execution_time=10.0,
                item_results=[],
            )
            collector.add_result(result)

        assert len(collector.results) == 5
        # Check the modified accuracies with appropriate tolerance for floating point
        assert collector.results[0].metrics["accuracy"] == pytest.approx(0.8, rel=1e-6)
        assert collector.results[4].metrics["accuracy"] == pytest.approx(0.84, rel=1e-6)

    def test_get_results_by_benchmark(self, collector):
        """Test filtering results by benchmark name."""
        # Add results for different benchmarks
        for name in ["benchmark_a", "benchmark_b", "benchmark_a"]:
            result = ExecutionResult(
                benchmark_name=name,
                total_items=10,
                successful_items=10,
                failed_items=0,
                metrics={"accuracy": 0.9},
                execution_time=10.0,
                item_results=[],
            )
            collector.add_result(result)

        # Filter by benchmark
        results_a = collector.get_results_by_benchmark("benchmark_a")
        assert len(results_a) == 2
        assert all(r.benchmark_name == "benchmark_a" for r in results_a)

    def test_get_latest_result(self, collector):
        """Test getting the most recent result for a benchmark."""
        # Add results with different timestamps
        for i in range(3):
            result = ExecutionResult(
                benchmark_name="test",
                total_items=10,
                successful_items=10,
                failed_items=0,
                metrics={"accuracy": 0.8 + i * 0.05},
                execution_time=10.0,
                item_results=[],
                timestamp=datetime(2024, 1, i + 1),
            )
            collector.add_result(result)

        latest = collector.get_latest_result("test")
        assert latest is not None
        assert latest.metrics["accuracy"] == 0.9  # Last one added

    def test_calculate_aggregate_metrics(self, collector):
        """Test calculating aggregate metrics across results."""
        # Add multiple results
        for i in range(10):
            result = ExecutionResult(
                benchmark_name="test",
                total_items=100,
                successful_items=85 + i,
                failed_items=15 - i,
                metrics={
                    "accuracy": 0.8 + i * 0.01,
                    "f1_score": 0.75 + i * 0.01,
                },
                execution_time=100 + i * 5,
                item_results=[],
            )
            collector.add_result(result)

        aggregates = collector.calculate_aggregate_metrics("test")

        assert "mean_accuracy" in aggregates
        assert "std_accuracy" in aggregates
        assert "min_accuracy" in aggregates
        assert "max_accuracy" in aggregates
        assert aggregates["mean_accuracy"] == pytest.approx(0.845, rel=1e-3)

    def test_filter_results(self, collector):
        """Test filtering results by various criteria."""
        # Add diverse results
        for i in range(20):
            result = ExecutionResult(
                benchmark_name=f"benchmark_{i % 3}",
                total_items=100,
                successful_items=80 + i,
                failed_items=20 - i,
                metrics={"accuracy": 0.7 + (i * 0.01)},
                execution_time=50 + i * 2,
                item_results=[],
                context={"temperature": 0.5 + (i % 3) * 0.2},
            )
            collector.add_result(result)

        # Filter by accuracy threshold
        filter_config = ResultFilter(min_accuracy=0.8)
        filtered = collector.filter_results(filter_config)
        assert all(r.metrics["accuracy"] >= 0.8 for r in filtered)

        # Filter by benchmark name pattern
        filter_config = ResultFilter(benchmark_pattern="benchmark_1")
        filtered = collector.filter_results(filter_config)
        assert all("1" in r.benchmark_name for r in filtered)

    def test_clear_results(self, collector, sample_execution_result):
        """Test clearing all results."""
        collector.add_result(sample_execution_result)
        assert len(collector.results) > 0

        collector.clear()
        assert len(collector.results) == 0
        assert collector.metadata == {}


class TestResultStorage:
    """Test result storage functionality."""

    @pytest.fixture
    def storage(self, tmp_path):
        """Create a result storage instance."""
        return ResultStorage(base_path=tmp_path)

    @pytest.fixture
    def sample_result(self):
        """Create a sample result for testing."""
        return ExecutionResult(
            benchmark_name="test",
            total_items=10,
            successful_items=9,
            failed_items=1,
            metrics={"accuracy": 0.9},
            execution_time=5.0,
            item_results=[],
        )

    def test_storage_initialization(self, storage, tmp_path):
        """Test storage initializes with correct path."""
        assert storage.base_path == tmp_path
        assert storage.base_path.exists()

    def test_save_result_json(self, storage, sample_result):
        """Test saving result as JSON."""
        file_path = storage.save_result(sample_result, format=ExportFormat.JSON)

        assert file_path.exists()
        assert file_path.suffix == ".json"

        # Verify content
        with open(file_path) as f:
            data = json.load(f)
            assert data["benchmark_name"] == "test"
            assert data["metrics"]["accuracy"] == 0.9

    def test_save_result_jsonl(self, storage, sample_result):
        """Test saving result as JSONL."""
        file_path = storage.save_result(sample_result, format=ExportFormat.JSONL)

        assert file_path.exists()
        assert file_path.suffix == ".jsonl"

        # Verify content
        with open(file_path) as f:
            line = f.readline()
            data = json.loads(line)
            assert data["benchmark_name"] == "test"

    def test_save_multiple_results(self, storage):
        """Test saving multiple results to JSONL."""
        results = []
        for i in range(5):
            results.append(
                ExecutionResult(
                    benchmark_name=f"test_{i}",
                    total_items=10,
                    successful_items=10,
                    failed_items=0,
                    metrics={"accuracy": 0.8 + i * 0.02},
                    execution_time=10.0,
                    item_results=[],
                )
            )

        file_path = storage.save_multiple_results(results, format=ExportFormat.JSONL)

        assert file_path.exists()
        with open(file_path) as f:
            lines = f.readlines()
            assert len(lines) == 5

    def test_load_results(self, storage, sample_result):
        """Test loading results from storage."""
        # Save a result first
        file_path = storage.save_result(sample_result)

        # Load it back
        loaded = storage.load_result(file_path)
        assert loaded.benchmark_name == sample_result.benchmark_name
        assert loaded.metrics == sample_result.metrics

    def test_list_stored_results(self, storage):
        """Test listing all stored results."""
        # Save multiple results
        for i in range(3):
            result = ExecutionResult(
                benchmark_name=f"test_{i}",
                total_items=10,
                successful_items=10,
                failed_items=0,
                metrics={"accuracy": 0.9},
                execution_time=10.0,
                item_results=[],
            )
            storage.save_result(result)

        # List results
        files = storage.list_results()
        assert len(files) >= 3

    def test_export_to_csv(self, storage):
        """Test exporting results to CSV format."""
        results = []
        for i in range(5):
            results.append(
                ExecutionResult(
                    benchmark_name="test",
                    total_items=100,
                    successful_items=85 + i,
                    failed_items=15 - i,
                    metrics={
                        "accuracy": 0.8 + i * 0.01,
                        "f1_score": 0.75 + i * 0.01,
                    },
                    execution_time=100.0,
                    item_results=[],
                )
            )

        csv_path = storage.export_to_csv(results, "test_results.csv")

        assert csv_path.exists()
        assert csv_path.suffix == ".csv"

        # Verify CSV content
        import csv

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 5
            assert "benchmark_name" in rows[0]
            assert "accuracy" in rows[0]


class TestStatisticalSummary:
    """Test statistical summary generation."""

    @pytest.fixture
    def results(self):
        """Create sample results for analysis."""
        results = []
        for i in range(100):
            results.append(
                ExecutionResult(
                    benchmark_name="test",
                    total_items=1000,
                    successful_items=850 + (i % 20),
                    failed_items=150 - (i % 20),
                    metrics={
                        "accuracy": 0.85 + (i % 20) * 0.005,
                        "f1_score": 0.82 + (i % 20) * 0.004,
                        "precision": 0.88 + (i % 15) * 0.003,
                        "recall": 0.79 + (i % 10) * 0.006,
                    },
                    execution_time=120 + (i % 30),
                    item_results=[],
                )
            )
        return results

    def test_calculate_summary_statistics(self, results):
        """Test calculating comprehensive statistics."""
        summary = StatisticalSummary.from_results(results)

        assert summary.count == 100
        assert summary.mean_accuracy > 0
        assert summary.std_accuracy > 0
        assert summary.min_accuracy <= summary.mean_accuracy <= summary.max_accuracy
        assert summary.percentile_25 < summary.median < summary.percentile_75

    def test_confidence_intervals(self, results):
        """Test calculating confidence intervals."""
        summary = StatisticalSummary.from_results(results)
        ci = summary.get_confidence_interval(confidence=0.95)

        assert ci["lower"] < summary.mean_accuracy < ci["upper"]
        assert ci["confidence"] == 0.95

    def test_compare_distributions(self):
        """Test comparing two result distributions."""
        # Create two sets of results with different characteristics
        results_a = []
        results_b = []

        for i in range(50):
            results_a.append(
                ExecutionResult(
                    benchmark_name="a",
                    total_items=100,
                    successful_items=85,
                    failed_items=15,
                    metrics={"accuracy": 0.85 + (i % 10) * 0.001},
                    execution_time=10.0,
                    item_results=[],
                )
            )
            results_b.append(
                ExecutionResult(
                    benchmark_name="b",
                    total_items=100,
                    successful_items=80,
                    failed_items=20,
                    metrics={"accuracy": 0.80 + (i % 10) * 0.001},
                    execution_time=10.0,
                    item_results=[],
                )
            )

        summary_a = StatisticalSummary.from_results(results_a)
        summary_b = StatisticalSummary.from_results(results_b)

        comparison = summary_a.compare_to(summary_b)

        assert comparison["mean_difference"] > 0  # A has higher accuracy
        assert "p_value" in comparison
        assert "significant" in comparison


class TestResultReporter:
    """Test result reporting functionality."""

    @pytest.fixture
    def reporter(self):
        """Create a result reporter instance."""
        return ResultReporter()

    @pytest.fixture
    def sample_results(self):
        """Create sample results for reporting."""
        results = []
        for benchmark in ["faith", "simple", "truthful"]:
            for i in range(5):
                results.append(
                    ExecutionResult(
                        benchmark_name=benchmark,
                        total_items=100,
                        successful_items=80 + i * 2,
                        failed_items=20 - i * 2,
                        metrics={
                            "accuracy": 0.8 + i * 0.02,
                            "f1_score": 0.75 + i * 0.02,
                        },
                        execution_time=60 + i * 5,
                        item_results=[],
                        context={"temperature": 0.3 + i * 0.1},
                    )
                )
        return results

    def test_generate_summary_report(self, reporter, sample_results):
        """Test generating a summary report."""
        report = reporter.generate_summary_report(sample_results)

        assert report.total_experiments == 15
        assert sorted(report.benchmarks_tested) == ["faith", "simple", "truthful"]
        assert report.overall_statistics is not None
        assert report.per_benchmark_stats is not None
        assert len(report.per_benchmark_stats) == 3

    def test_generate_benchmark_report(self, reporter, sample_results):
        """Test generating a report for specific benchmark."""
        faith_results = [r for r in sample_results if r.benchmark_name == "faith"]
        report = reporter.generate_benchmark_report("faith", faith_results)

        assert report.benchmark_name == "faith"
        assert report.num_runs == 5
        assert report.best_result is not None
        assert report.worst_result is not None
        assert report.statistics is not None

    def test_generate_comparison_report(self, reporter):
        """Test generating comparison between benchmarks."""
        results_a = []
        results_b = []

        for i in range(10):
            results_a.append(
                ExecutionResult(
                    benchmark_name="benchmark_a",
                    total_items=100,
                    successful_items=85,
                    failed_items=15,
                    metrics={"accuracy": 0.85 + i * 0.001},
                    execution_time=10.0,
                    item_results=[],
                )
            )
            results_b.append(
                ExecutionResult(
                    benchmark_name="benchmark_b",
                    total_items=100,
                    successful_items=82,
                    failed_items=18,
                    metrics={"accuracy": 0.82 + i * 0.001},
                    execution_time=12.0,
                    item_results=[],
                )
            )

        comparison = reporter.generate_comparison_report(
            "benchmark_a", results_a, "benchmark_b", results_b
        )

        assert comparison.benchmark_a == "benchmark_a"
        assert comparison.benchmark_b == "benchmark_b"
        assert comparison.metric_comparisons is not None
        assert "accuracy" in comparison.metric_comparisons
        assert comparison.performance_difference is not None

    def test_format_report_as_markdown(self, reporter, sample_results):
        """Test formatting report as markdown."""
        report = reporter.generate_summary_report(sample_results)
        markdown = reporter.format_as_markdown(report)

        assert "# Benchmark Report" in markdown
        assert "## Overall Statistics" in markdown
        assert "## Per-Benchmark Results" in markdown
        assert "| Metric |" in markdown  # Table formatting

    def test_format_report_as_html(self, reporter, sample_results):
        """Test formatting report as HTML."""
        report = reporter.generate_summary_report(sample_results)
        html = reporter.format_as_html(report)

        assert "<html>" in html
        assert "<table" in html  # Check for table tag (may have attributes)
        assert "Benchmark Report" in html

    def test_save_report(self, reporter, sample_results, tmp_path):
        """Test saving report to file."""
        report = reporter.generate_summary_report(sample_results)

        # Save as markdown
        md_path = tmp_path / "report.md"
        reporter.save_report(report, md_path, format="markdown")
        assert md_path.exists()

        # Save as HTML
        html_path = tmp_path / "report.html"
        reporter.save_report(report, html_path, format="html")
        assert html_path.exists()

        # Save as JSON
        json_path = tmp_path / "report.json"
        reporter.save_report(report, json_path, format="json")
        assert json_path.exists()

    def test_temperature_analysis(self, reporter):
        """Test analyzing results by temperature settings."""
        results = []
        for temp in [0.3, 0.5, 0.7, 1.0]:
            for i in range(10):
                results.append(
                    ExecutionResult(
                        benchmark_name="test",
                        total_items=100,
                        successful_items=int(90 - temp * 20 + i),
                        failed_items=int(10 + temp * 20 - i),
                        metrics={"accuracy": 0.9 - temp * 0.2 + i * 0.01},
                        execution_time=10.0,
                        item_results=[],
                        context={"temperature": temp},
                    )
                )

        analysis = reporter.analyze_by_temperature(results)

        assert len(analysis) == 4
        assert all(temp in analysis for temp in [0.3, 0.5, 0.7, 1.0])
        # Lower temperatures should have higher accuracy in this test
        assert analysis[0.3]["mean_accuracy"] > analysis[1.0]["mean_accuracy"]


class TestBenchmarkComparison:
    """Test benchmark comparison functionality."""

    def test_compare_two_benchmarks(self):
        """Test comparing results from two benchmarks."""
        results_a = [
            ExecutionResult(
                benchmark_name="a",
                total_items=100,
                successful_items=85,
                failed_items=15,
                metrics={"accuracy": 0.85, "f1": 0.83},
                execution_time=10.0,
                item_results=[],
            )
            for _ in range(10)
        ]

        results_b = [
            ExecutionResult(
                benchmark_name="b",
                total_items=100,
                successful_items=80,
                failed_items=20,
                metrics={"accuracy": 0.80, "f1": 0.78},
                execution_time=12.0,
                item_results=[],
            )
            for _ in range(10)
        ]

        comparison = BenchmarkComparison.compare(results_a, results_b)

        assert comparison["winner"] == "a"
        assert comparison["metrics_better"]["accuracy"] == "a"
        assert comparison["improvement_percentage"]["accuracy"] > 0

    def test_multi_benchmark_comparison(self):
        """Test comparing multiple benchmarks."""
        all_results = {}

        for name in ["a", "b", "c"]:
            all_results[name] = [
                ExecutionResult(
                    benchmark_name=name,
                    total_items=100,
                    successful_items=80 + ord(name) - ord("a") * 2,
                    failed_items=20 - ord(name) + ord("a") * 2,
                    metrics={"accuracy": 0.8 + (ord(name) - ord("a")) * 0.02},
                    execution_time=10.0,
                    item_results=[],
                )
                for _ in range(5)
            ]

        comparison = BenchmarkComparison.compare_multiple(all_results)

        assert comparison["rankings"]["accuracy"][0] == "c"  # Highest accuracy
        assert len(comparison["pairwise_comparisons"]) == 3  # C(3,2) = 3 pairs
