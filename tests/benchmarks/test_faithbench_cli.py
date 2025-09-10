"""Tests for FaithBench CLI commands."""

import json
from unittest.mock import MagicMock, patch

import pytest
from cosmos_coherence.benchmarks.faithbench_cli import app
from cosmos_coherence.benchmarks.models.datasets import FaithBenchAnnotation, FaithBenchItem
from cosmos_coherence.harness.base_benchmark import BenchmarkEvaluationResult
from typer.testing import CliRunner

runner = CliRunner()


class TestFaithBenchCLI:
    """Test suite for FaithBench CLI commands."""

    @pytest.fixture
    def sample_faithbench_item(self):
        """Create a sample FaithBench item."""
        return FaithBenchItem(
            sample_id="test_001",
            source="The Earth orbits the Sun once every 365.25 days.",
            summary="The Earth orbits the Sun annually.",
            annotation_label=FaithBenchAnnotation.CONSISTENT,
            entropy_score=0.1,
            question="Summarize the orbital period.",
        )

    @pytest.fixture
    def sample_results(self):
        """Create sample evaluation results."""
        return [
            BenchmarkEvaluationResult(
                is_correct=True,
                score=0.9,
                original_metric_score=0.85,
                metadata={
                    "annotation_label": "consistent",
                    "entropy_score": 0.1,
                    "is_challenging": False,
                },
            ),
            BenchmarkEvaluationResult(
                is_correct=False,
                score=0.3,
                original_metric_score=0.4,
                metadata={
                    "annotation_label": "hallucinated",
                    "entropy_score": 0.8,
                    "is_challenging": True,
                },
            ),
        ]

    def test_info_command(self):
        """Test the info command displays FaithBench information."""
        result = runner.invoke(app, ["info"])

        assert result.exit_code == 0
        assert "FaithBench" in result.stdout
        assert "Hallucination Detection" in result.stdout
        assert "Consistent" in result.stdout
        assert "Hallucinated" in result.stdout
        assert "gpt-4-turbo" in result.stdout

    @patch("cosmos_coherence.benchmarks.faithbench_cli.FaithBenchMetrics")
    @patch("cosmos_coherence.benchmarks.faithbench_cli.asyncio.run")
    @patch("cosmos_coherence.benchmarks.faithbench_cli.FaithBenchBenchmark")
    def test_run_command_basic(
        self, mock_benchmark_class, mock_asyncio_run, mock_metrics_class, sample_results
    ):
        """Test basic run command."""
        # Setup mocks
        mock_benchmark = MagicMock()
        mock_benchmark_class.return_value = mock_benchmark
        mock_benchmark_class.SUPPORTED_MODELS = {"gpt-4-turbo", "gpt-4o"}
        mock_benchmark_class.NO_TEMPERATURE_MODELS = set()

        # Mock the async load_dataset - return the results directly
        mock_dataset = [MagicMock() for _ in range(5)]
        for item in mock_dataset:
            item.summary = "Test summary"
        mock_asyncio_run.return_value = mock_dataset

        # Mock evaluate_response
        mock_benchmark.evaluate_response.return_value = sample_results[0]
        mock_benchmark.validate_config.return_value = None

        # Mock metrics calculator
        mock_metrics = MagicMock()
        mock_metrics_class.return_value = mock_metrics
        mock_metrics.calculate_aggregate_metrics.return_value = {
            "overall_accuracy": 0.85,
            "average_score": 0.8,
            "precision_hallucinated": 0.7,
            "recall_hallucinated": 0.6,
            "f1_hallucinated": 0.65,
        }

        result = runner.invoke(app, ["run", "--model", "gpt-4-turbo"])

        assert result.exit_code == 0
        assert "Running FaithBench with gpt-4-turbo" in result.stdout
        assert "Evaluation Complete" in result.stdout

    def test_run_command_invalid_model(self):
        """Test run command with invalid model."""
        with patch(
            "cosmos_coherence.benchmarks.faithbench_cli.FaithBenchBenchmark.SUPPORTED_MODELS",
            {"gpt-4-turbo"},
        ):
            result = runner.invoke(app, ["run", "--model", "invalid-model"])

            assert result.exit_code == 1
            assert "not supported" in result.stdout

    def test_run_command_invalid_temperature_for_reasoning_model(self):
        """Test run command with invalid temperature for o1-mini."""
        with patch(
            "cosmos_coherence.benchmarks.faithbench_cli.FaithBenchBenchmark.SUPPORTED_MODELS",
            {"o1-mini"},
        ):
            with patch(
                "cosmos_coherence.benchmarks.faithbench_cli.FaithBenchBenchmark.NO_TEMPERATURE_MODELS",
                {"o1-mini"},
            ):
                result = runner.invoke(app, ["run", "--model", "o1-mini", "--temperature", "0.7"])

                assert result.exit_code == 1
                assert "doesn't support temperature" in result.stdout

    @patch("cosmos_coherence.benchmarks.faithbench_cli.FaithBenchMetrics")
    @patch("cosmos_coherence.benchmarks.faithbench_cli.asyncio.run")
    @patch("cosmos_coherence.benchmarks.faithbench_cli.FaithBenchBenchmark")
    def test_run_command_with_sample_size(
        self, mock_benchmark_class, mock_asyncio_run, mock_metrics_class, sample_results
    ):
        """Test run command with sample size option."""
        # Setup mocks
        mock_benchmark = MagicMock()
        mock_benchmark_class.return_value = mock_benchmark
        mock_benchmark_class.SUPPORTED_MODELS = {"gpt-4-turbo"}
        mock_benchmark_class.NO_TEMPERATURE_MODELS = set()

        # Mock dataset loading
        mock_dataset = [MagicMock() for _ in range(10)]
        for item in mock_dataset:
            item.summary = "Test summary"
        mock_asyncio_run.return_value = mock_dataset

        mock_benchmark.evaluate_response.return_value = sample_results[0]
        mock_benchmark.validate_config.return_value = None

        # Mock metrics calculator
        mock_metrics = MagicMock()
        mock_metrics_class.return_value = mock_metrics
        mock_metrics.calculate_aggregate_metrics.return_value = {
            "overall_accuracy": 0.85,
            "average_score": 0.8,
            "precision_hallucinated": 0.7,
            "recall_hallucinated": 0.6,
            "f1_hallucinated": 0.65,
        }

        result = runner.invoke(app, ["run", "--sample-size", "10"])

        assert result.exit_code == 0
        assert "Sample size: 10" in result.stdout

    @patch("cosmos_coherence.benchmarks.faithbench_cli.FaithBenchMetrics")
    @patch("cosmos_coherence.benchmarks.faithbench_cli.asyncio.run")
    @patch("cosmos_coherence.benchmarks.faithbench_cli.FaithBenchBenchmark")
    def test_run_command_with_output(
        self, mock_benchmark_class, mock_asyncio_run, mock_metrics_class, tmp_path, sample_results
    ):
        """Test run command with output file."""
        # Setup mocks
        mock_benchmark = MagicMock()
        mock_benchmark_class.return_value = mock_benchmark
        mock_benchmark_class.SUPPORTED_MODELS = {"gpt-4-turbo"}
        mock_benchmark_class.NO_TEMPERATURE_MODELS = set()
        mock_benchmark.get_baseline_metrics.return_value = {}

        # Mock dataset
        mock_dataset = [MagicMock() for _ in range(5)]
        for item in mock_dataset:
            item.summary = "Test summary"
        mock_asyncio_run.return_value = mock_dataset

        mock_benchmark.evaluate_response.return_value = sample_results[0]
        mock_benchmark.validate_config.return_value = None

        # Mock metrics calculator
        mock_metrics = MagicMock()
        mock_metrics_class.return_value = mock_metrics
        mock_metrics.calculate_aggregate_metrics.return_value = {
            "overall_accuracy": 0.85,
            "average_score": 0.8,
            "precision_hallucinated": 0.7,
            "recall_hallucinated": 0.6,
            "f1_hallucinated": 0.65,
        }
        mock_metrics.export_metrics.return_value = {"summary": {}, "per_label_metrics": {}}

        output_file = tmp_path / "results.json"
        result = runner.invoke(app, ["run", "--output", str(output_file)])

        assert result.exit_code == 0
        assert output_file.exists()
        assert "Results saved to" in result.stdout

        # Check output file content
        with open(output_file) as f:
            data = json.load(f)
            assert "model" in data
            assert "metrics" in data

    @patch("cosmos_coherence.benchmarks.faithbench_cli.FaithBenchMetrics")
    @patch("cosmos_coherence.benchmarks.faithbench_cli.asyncio.run")
    @patch("cosmos_coherence.benchmarks.faithbench_cli.FaithBenchBenchmark")
    def test_run_command_show_challenging(
        self, mock_benchmark_class, mock_asyncio_run, mock_metrics_class, sample_results
    ):
        """Test run command with show-challenging option."""
        # Setup mocks
        mock_benchmark = MagicMock()
        mock_benchmark_class.return_value = mock_benchmark
        mock_benchmark_class.SUPPORTED_MODELS = {"gpt-4-turbo"}
        mock_benchmark_class.NO_TEMPERATURE_MODELS = set()

        # Mock dataset
        mock_dataset = [MagicMock() for _ in range(5)]
        for item in mock_dataset:
            item.summary = "Test summary"
        mock_asyncio_run.return_value = mock_dataset

        mock_benchmark.evaluate_response.return_value = sample_results[0]
        mock_benchmark.validate_config.return_value = None

        # Mock metrics calculator
        mock_metrics = MagicMock()
        mock_metrics_class.return_value = mock_metrics
        mock_metrics.calculate_aggregate_metrics.return_value = {
            "overall_accuracy": 0.85,
            "average_score": 0.8,
            "precision_hallucinated": 0.7,
            "recall_hallucinated": 0.6,
            "f1_hallucinated": 0.65,
            "challenging_accuracy": 0.6,
            "challenging_count": 10,
            "non_challenging_accuracy": 0.9,
            "average_entropy": 0.5,
        }

        result = runner.invoke(app, ["run", "--show-challenging"])

        assert result.exit_code == 0
        assert "Challenging Samples Performance" in result.stdout

    @patch("cosmos_coherence.benchmarks.faithbench_cli.FaithBenchMetrics")
    @patch("cosmos_coherence.benchmarks.faithbench_cli.asyncio.run")
    @patch("cosmos_coherence.benchmarks.faithbench_cli.FaithBenchBenchmark")
    def test_run_command_compare_baseline(
        self, mock_benchmark_class, mock_asyncio_run, mock_metrics_class, sample_results
    ):
        """Test run command with compare-baseline option."""
        # Setup mocks
        mock_benchmark = MagicMock()
        mock_benchmark_class.return_value = mock_benchmark
        mock_benchmark_class.SUPPORTED_MODELS = {"gpt-4-turbo"}
        mock_benchmark_class.NO_TEMPERATURE_MODELS = set()

        # Mock baseline metrics
        mock_benchmark.get_baseline_metrics.return_value = {
            "gpt-4-turbo_accuracy": 0.75,
        }

        # Mock dataset
        mock_dataset = [MagicMock() for _ in range(5)]
        for item in mock_dataset:
            item.summary = "Test summary"
        mock_asyncio_run.return_value = mock_dataset

        mock_benchmark.evaluate_response.return_value = sample_results[0]
        mock_benchmark.validate_config.return_value = None

        # Mock metrics calculator
        mock_metrics = MagicMock()
        mock_metrics_class.return_value = mock_metrics
        mock_metrics.calculate_aggregate_metrics.return_value = {
            "overall_accuracy": 0.85,
            "average_score": 0.8,
            "precision_hallucinated": 0.7,
            "recall_hallucinated": 0.6,
            "f1_hallucinated": 0.65,
        }

        result = runner.invoke(app, ["run", "--compare-baseline"])

        assert result.exit_code == 0
        assert "Comparison with Paper Baseline" in result.stdout

    def test_analyze_command_basic(self, tmp_path):
        """Test analyze command with basic options."""
        # Create a sample results file
        results_file = tmp_path / "results.json"
        results_data = {
            "model": "gpt-4-turbo",
            "metrics": {
                "overall_accuracy": 0.85,
                "overall_precision": 0.88,
                "overall_recall": 0.82,
                "overall_f1": 0.85,
            },
            "detailed_metrics": {},
        }

        with open(results_file, "w") as f:
            json.dump(results_data, f)

        result = runner.invoke(app, ["analyze", str(results_file)])

        assert result.exit_code == 0
        assert "FaithBench Analysis" in result.stdout
        assert "Model: gpt-4-turbo" in result.stdout
        assert "Overall Performance" in result.stdout

    def test_analyze_command_per_label(self, tmp_path):
        """Test analyze command with per-label option."""
        # Create results file with per-label metrics
        results_file = tmp_path / "results.json"
        results_data = {
            "model": "gpt-4-turbo",
            "metrics": {
                "overall_accuracy": 0.85,
                "precision_consistent": 0.9,
                "recall_consistent": 0.88,
                "f1_consistent": 0.89,
                "precision_hallucinated": 0.75,
                "recall_hallucinated": 0.70,
                "f1_hallucinated": 0.72,
            },
            "detailed_metrics": {},
        }

        with open(results_file, "w") as f:
            json.dump(results_data, f)

        result = runner.invoke(app, ["analyze", str(results_file), "--per-label"])

        assert result.exit_code == 0
        assert "Per-Label Performance" in result.stdout
        assert "Consistent" in result.stdout
        assert "Hallucinated" in result.stdout

    def test_analyze_command_entropy(self, tmp_path):
        """Test analyze command with entropy option."""
        # Create results file with entropy metrics
        results_file = tmp_path / "results.json"
        results_data = {
            "model": "gpt-4-turbo",
            "metrics": {
                "overall_accuracy": 0.85,
                "low_entropy_accuracy": 0.95,
                "low_entropy_count": 30,
                "medium_entropy_accuracy": 0.80,
                "medium_entropy_count": 40,
                "high_entropy_accuracy": 0.60,
                "high_entropy_count": 30,
            },
            "detailed_metrics": {},
        }

        with open(results_file, "w") as f:
            json.dump(results_data, f)

        result = runner.invoke(app, ["analyze", str(results_file), "--entropy"])

        assert result.exit_code == 0
        assert "Entropy-Stratified Performance" in result.stdout
        assert "Low" in result.stdout
        assert "Medium" in result.stdout
        assert "High" in result.stdout

    def test_analyze_command_export_report(self, tmp_path):
        """Test analyze command with export option."""
        # Create results file
        results_file = tmp_path / "results.json"
        results_data = {
            "model": "gpt-4-turbo",
            "metrics": {
                "overall_accuracy": 0.65,
                "recall_hallucinated": 0.45,
            },
            "config": {"temperature": 0.0},
            "detailed_metrics": {},
        }

        with open(results_file, "w") as f:
            json.dump(results_data, f)

        report_file = tmp_path / "report.json"
        result = runner.invoke(app, ["analyze", str(results_file), "--export", str(report_file)])

        assert result.exit_code == 0
        assert report_file.exists()
        assert "Report exported to" in result.stdout

        # Check report content
        with open(report_file) as f:
            report = json.load(f)
            assert "recommendations" in report
            assert len(report["recommendations"]) > 0  # Should have recommendations for low scores

    def test_analyze_command_file_not_found(self):
        """Test analyze command with non-existent file."""
        result = runner.invoke(app, ["analyze", "nonexistent.json"])

        assert result.exit_code == 1
        assert "not found" in result.stdout
