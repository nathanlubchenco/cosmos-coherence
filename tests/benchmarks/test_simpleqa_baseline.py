"""Tests for SimpleQA baseline comparison functionality."""

import json
from unittest.mock import patch

import pytest
from cosmos_coherence.benchmarks.simpleqa_cli import app
from typer.testing import CliRunner


class TestSimpleQABaseline:
    """Test baseline comparison functionality."""

    def test_baseline_constants(self):
        """Test that baseline constants are correctly defined."""
        from cosmos_coherence.benchmarks.implementations.simpleqa_benchmark import SimpleQABenchmark

        benchmark = SimpleQABenchmark(use_huggingface=False)
        baselines = benchmark.get_baseline_metrics()

        assert "gpt-4_accuracy" in baselines
        assert "gpt-3.5_accuracy" in baselines
        assert baselines["gpt-4_accuracy"] == 0.82
        assert baselines["gpt-3.5_accuracy"] == 0.68
        assert baselines["paper_reference"] == "https://arxiv.org/abs/2410.02034"
        assert baselines["benchmark_version"] == "v1.0"

    def test_validate_baseline_command_exists(self):
        """Test that validate-baseline command exists in CLI."""
        runner = CliRunner()
        result = runner.invoke(app, ["validate-baseline", "--help"])
        assert result.exit_code == 0
        assert "Compare benchmark results against published baselines" in result.output

    def test_validate_baseline_gpt4_within_tolerance(self):
        """Test validation passes when GPT-4 results are within 5% tolerance."""
        runner = CliRunner()

        # Create a mock results file
        with runner.isolated_filesystem():
            results = {
                "model": "gpt-4",
                "temperature": 0.3,
                "total_questions": 100,
                "correct_answers": 82,
                "metrics": {"exact_match_accuracy": 0.82, "f1_score": 0.85},
            }

            with open("results.json", "w") as f:
                json.dump(results, f)

            result = runner.invoke(app, ["validate-baseline", "results.json"])
            assert result.exit_code == 0
            assert "✓ Within 5%" in result.output
            assert "82.0%" in result.output

    def test_validate_baseline_gpt4_exceeds_tolerance(self):
        """Test validation warns when GPT-4 results exceed 5% tolerance."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            results = {
                "model": "gpt-4",
                "temperature": 0.3,
                "total_questions": 100,
                "correct_answers": 75,  # 75% is 7% below baseline of 82%
                "metrics": {"exact_match_accuracy": 0.75, "f1_score": 0.78},
            }

            with open("results.json", "w") as f:
                json.dump(results, f)

            result = runner.invoke(app, ["validate-baseline", "results.json"])
            assert result.exit_code == 0
            assert "⚠ >5% deviation" in result.output
            assert "75.0%" in result.output
            assert "-7.0%" in result.output

    def test_validate_baseline_gpt35_within_tolerance(self):
        """Test validation for GPT-3.5 results."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            results = {
                "model": "gpt-3.5-turbo",
                "temperature": 0.3,
                "total_questions": 100,
                "correct_answers": 68,
                "metrics": {"exact_match_accuracy": 0.68, "f1_score": 0.72},
            }

            with open("results.json", "w") as f:
                json.dump(results, f)

            result = runner.invoke(app, ["validate-baseline", "results.json"])
            assert result.exit_code == 0
            assert "✓ Within 5%" in result.output
            assert "68.0%" in result.output

    def test_validate_baseline_with_details_flag(self):
        """Test validation with --details flag shows additional metrics."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            results = {
                "model": "gpt-4",
                "temperature": 0.3,
                "total_questions": 100,
                "correct_answers": 82,
                "metrics": {"exact_match_accuracy": 0.82, "f1_score": 0.85},
            }

            with open("results.json", "w") as f:
                json.dump(results, f)

            result = runner.invoke(app, ["validate-baseline", "results.json", "--details"])
            assert result.exit_code == 0
            assert "Detailed Metrics" in result.output
            assert "F1 Score: 0.850" in result.output
            assert "Total Questions: 100" in result.output
            assert "Correct Answers: 82" in result.output

    def test_validate_baseline_file_not_found(self):
        """Test validation handles missing results file."""
        runner = CliRunner()
        result = runner.invoke(app, ["validate-baseline", "nonexistent.json"])
        assert result.exit_code == 1
        assert "Results file not found" in result.output

    def test_validate_baseline_invalid_json(self):
        """Test validation handles invalid JSON."""
        runner = CliRunner()

        with runner.isolated_filesystem():
            with open("invalid.json", "w") as f:
                f.write("not valid json")

            result = runner.invoke(app, ["validate-baseline", "invalid.json"])
            assert result.exit_code == 1
            assert "Invalid JSON" in result.output

    def test_baseline_comparison_in_benchmark_class(self):
        """Test baseline comparison method in SimpleQABenchmark."""
        from cosmos_coherence.benchmarks.implementations.simpleqa_benchmark import SimpleQABenchmark

        benchmark = SimpleQABenchmark(use_huggingface=False)

        # Test within tolerance
        comparison = benchmark.compare_to_baseline("gpt-4", 0.80)
        assert comparison["model"] == "gpt-4"
        assert comparison["your_score"] == 0.80
        assert comparison["baseline_score"] == 0.82
        assert abs(comparison["difference"] - (-0.02)) < 0.001  # Handle floating point precision
        assert comparison["within_tolerance"] is True

        # Test exceeds tolerance
        comparison = benchmark.compare_to_baseline("gpt-4", 0.75)
        assert abs(comparison["difference"] - (-0.07)) < 0.001  # Handle floating point precision
        assert comparison["within_tolerance"] is False

    def test_baseline_comparison_unknown_model(self):
        """Test baseline comparison with unknown model."""
        from cosmos_coherence.benchmarks.implementations.simpleqa_benchmark import SimpleQABenchmark

        benchmark = SimpleQABenchmark(use_huggingface=False)

        comparison = benchmark.compare_to_baseline("claude-3", 0.85)
        assert comparison["model"] == "claude-3"
        assert comparison["your_score"] == 0.85
        assert comparison["baseline_score"] is None
        assert comparison["difference"] is None
        assert comparison["within_tolerance"] is None

    def test_deviation_warning_threshold(self):
        """Test that 5% deviation threshold is correctly applied."""
        from cosmos_coherence.benchmarks.implementations.simpleqa_benchmark import SimpleQABenchmark

        benchmark = SimpleQABenchmark(use_huggingface=False)

        # Exactly 5% absolute difference below (should be within tolerance)
        comparison = benchmark.compare_to_baseline("gpt-4", 0.77)  # 82% - 5% = 77%
        assert comparison["within_tolerance"] is True

        # Just over 5% absolute difference below (should exceed tolerance)
        comparison = benchmark.compare_to_baseline("gpt-4", 0.76)  # 82% - 6% = 76%
        assert comparison["within_tolerance"] is False

        # Exactly 5% absolute difference above (should be within tolerance)
        comparison = benchmark.compare_to_baseline("gpt-4", 0.87)  # 82% + 5% = 87%
        assert comparison["within_tolerance"] is True

        # Just over 5% absolute difference above (should exceed tolerance)
        comparison = benchmark.compare_to_baseline("gpt-4", 0.88)  # 82% + 6% = 88%
        assert comparison["within_tolerance"] is False

    @pytest.mark.asyncio
    async def test_run_command_with_baseline_comparison(self):
        """Test that run command can include baseline comparison."""
        from cosmos_coherence.benchmarks.implementations.simpleqa_benchmark import SimpleQABenchmark
        from cosmos_coherence.benchmarks.models.datasets import SimpleQAItem

        # Mock a small dataset
        mock_items = [
            SimpleQAItem(
                question="What is 2+2?",
                best_answer="4",
                good_answers=["4", "four"],
                bad_answers=["5", "22"],
            ),
            SimpleQAItem(
                question="What is the capital of France?",
                best_answer="Paris",
                good_answers=["Paris"],
                bad_answers=["London", "Berlin"],
            ),
        ]

        with patch.object(SimpleQABenchmark, "load_dataset", return_value=mock_items):
            benchmark = SimpleQABenchmark(use_huggingface=False, sample_size=2)
            await benchmark.load_dataset()  # Load dataset to verify it works

            # Simulate evaluation results with accuracy within tolerance
            results = {
                "model": "gpt-4",
                "accuracy": 0.84,  # Within 5% of baseline (82%)
            }

            comparison = benchmark.compare_to_baseline("gpt-4", results["accuracy"])
            assert comparison["within_tolerance"] is True
            assert comparison["baseline_score"] == 0.82

            # Test with accuracy exceeding tolerance
            results_high = {
                "model": "gpt-4",
                "accuracy": 1.0,  # 100% is 18% above baseline
            }

            comparison_high = benchmark.compare_to_baseline("gpt-4", results_high["accuracy"])
            assert comparison_high["within_tolerance"] is False  # Too high above baseline
            assert comparison_high["baseline_score"] == 0.82
