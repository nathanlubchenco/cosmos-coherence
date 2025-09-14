"""Tests for HaluEval CLI commands."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner


class TestHaluEvalCLI:
    """Test HaluEval CLI functionality."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_results(self):
        """Create sample results for testing."""
        return {
            "model": "gpt-4",
            "temperature": 0.0,
            "total_samples": 100,
            "correct_samples": 65,
            "metrics": {
                "overall_accuracy": 0.65,
                "qa_accuracy": 0.68,
                "dialogue_accuracy": 0.62,
                "summarization_accuracy": 0.72,
                "precision": 0.70,
                "recall": 0.60,
                "f1_score": 0.65,
            },
            "items": [
                {
                    "question": "Is this answer hallucinated?",
                    "task_type": "qa",
                    "prediction": "Yes",
                    "expected": "Yes",
                    "is_correct": True,
                },
                {
                    "question": "Check this dialogue response",
                    "task_type": "dialogue",
                    "prediction": "No",
                    "expected": "Yes",
                    "is_correct": False,
                },
            ],
        }

    def test_run_command_help(self, runner):
        """Test run command help text."""
        # Import here to avoid circular imports
        from cosmos_coherence.benchmarks.halueval_cli import app

        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "Run HaluEval benchmark evaluation" in result.stdout
        assert "--model" in result.stdout
        assert "--sample-size" in result.stdout
        assert "--task-type" in result.stdout

    @patch.dict("os.environ", {}, clear=True)
    def test_run_command_no_api_key(self, runner):
        """Test run command fails without API key."""
        from cosmos_coherence.benchmarks.halueval_cli import app

        result = runner.invoke(app, ["run", "--model", "gpt-4"])
        assert result.exit_code == 1
        assert "OPENAI_API_KEY" in result.stdout

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("cosmos_coherence.benchmarks.halueval_cli.asyncio.run")
    def test_run_command_basic(self, mock_async_run, runner):
        """Test basic run command execution."""
        from cosmos_coherence.benchmarks.halueval_cli import app

        # Mock the async run to return sample results
        mock_async_run.return_value = {
            "model": "gpt-4",
            "temperature": 0.0,
            "total_samples": 10,
            "correct_samples": 6,
            "metrics": {
                "overall_accuracy": 0.6,
                "precision": 0.65,
                "recall": 0.55,
                "f1_score": 0.6,
            },
        }

        result = runner.invoke(
            app,
            [
                "run",
                "--model",
                "gpt-4",
                "--sample-size",
                "10",
                "--no-progress",
            ],
        )

        assert result.exit_code == 0
        assert "HaluEval Results" in result.stdout
        assert "gpt-4" in result.stdout
        mock_async_run.assert_called_once()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("cosmos_coherence.benchmarks.halueval_cli.asyncio.run")
    def test_run_with_task_filter(self, mock_async_run, runner):
        """Test running with specific task type."""
        from cosmos_coherence.benchmarks.halueval_cli import app

        mock_async_run.return_value = {
            "model": "gpt-4",
            "temperature": 0.0,
            "task_type": "qa",
            "total_samples": 5,
            "correct_samples": 4,
            "metrics": {"qa_accuracy": 0.8},
        }

        result = runner.invoke(
            app,
            [
                "run",
                "--model",
                "gpt-4",
                "--task-type",
                "qa",
                "--sample-size",
                "5",
            ],
        )

        assert result.exit_code == 0
        assert "Task Type: qa" in result.stdout

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("cosmos_coherence.benchmarks.halueval_cli.asyncio.run")
    def test_run_with_output_file(self, mock_async_run, runner, sample_results):
        """Test saving results to file."""
        from cosmos_coherence.benchmarks.halueval_cli import app

        mock_async_run.return_value = sample_results

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            output_path = tmp.name

        try:
            result = runner.invoke(
                app,
                [
                    "run",
                    "--model",
                    "gpt-4",
                    "--sample-size",
                    "100",
                    "--output",
                    output_path,
                ],
            )

            assert result.exit_code == 0
            assert Path(output_path).exists()

            # Verify saved content
            with open(output_path) as f:
                saved_data = json.load(f)
                assert saved_data["model"] == "gpt-4"
                assert saved_data["total_samples"] == 100
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_compare_command_help(self, runner):
        """Test compare command help."""
        from cosmos_coherence.benchmarks.halueval_cli import app

        result = runner.invoke(app, ["compare", "--help"])
        assert result.exit_code == 0
        assert "Compare model results" in result.stdout

    def test_compare_results(self, runner):
        """Test comparing two result files."""
        import json
        import tempfile

        from cosmos_coherence.benchmarks.halueval_cli import app

        # Create temporary files with results
        result1 = {
            "model": "gpt-4",
            "metrics": {
                "overall_accuracy": 0.65,
                "f1_score": 0.65,
            },
        }
        result2 = {
            "model": "gpt-3.5-turbo",
            "metrics": {
                "overall_accuracy": 0.55,
                "f1_score": 0.54,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f1:
            json.dump(result1, f1)
            file1 = f1.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f2:
            json.dump(result2, f2)
            file2 = f2.name

        try:
            result = runner.invoke(
                app,
                ["compare", file1, file2],
            )

            assert result.exit_code == 0
            assert "Comparison Results" in result.stdout
            assert "gpt-4" in result.stdout
            assert "gpt-3.5-turbo" in result.stdout
        finally:
            Path(file1).unlink(missing_ok=True)
            Path(file2).unlink(missing_ok=True)

    def test_baseline_command(self, runner):
        """Test baseline comparison command."""
        from cosmos_coherence.benchmarks.halueval_cli import app

        result = runner.invoke(
            app,
            ["baseline", "--accuracy", "0.70"],
        )

        assert result.exit_code == 0
        assert "Baseline Comparison" in result.stdout
        # Check if it shows GPT-3.5 baseline (0.65)
        assert "0.65" in result.stdout

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("cosmos_coherence.benchmarks.halueval_cli.asyncio.run")
    def test_cache_management(self, mock_async_run, runner):
        """Test cache control flags."""
        from cosmos_coherence.benchmarks.halueval_cli import app

        mock_async_run.return_value = {
            "model": "gpt-4",
            "temperature": 0.0,
            "total_samples": 5,
            "correct_samples": 3,
            "metrics": {"overall_accuracy": 0.6},
        }

        # Test with cache disabled
        result = runner.invoke(
            app,
            [
                "run",
                "--model",
                "gpt-4",
                "--sample-size",
                "5",
                "--no-cache",
            ],
        )

        assert result.exit_code == 0
        # Just verify it ran successfully
        assert "HaluEval Results" in result.stdout
        mock_async_run.assert_called_once()

    def test_clear_cache_command(self, runner):
        """Test cache clearing command."""

        from cosmos_coherence.benchmarks.halueval_cli import app

        # Mock the cache directory existence
        with patch("cosmos_coherence.benchmarks.halueval_cli.Path.exists") as mock_exists:
            mock_exists.return_value = True
            with patch(
                "cosmos_coherence.benchmarks.halueval_cli.clear_halueval_cache"
            ) as mock_clear:
                result = runner.invoke(app, ["clear-cache", "--confirm"])
                assert result.exit_code == 0
                mock_clear.assert_called_once()
                assert "Cache cleared" in result.stdout
