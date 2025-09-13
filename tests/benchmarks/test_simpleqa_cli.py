"""Tests for SimpleQA CLI commands."""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cosmos_coherence.benchmarks.simpleqa_cli import app
from typer.testing import CliRunner


class TestSimpleQACLI:
    """Test SimpleQA CLI functionality."""

    @pytest.fixture
    def runner(self):
        """Create a CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_results(self):
        """Create sample results for testing."""
        return {
            "model": "gpt-4",
            "temperature": 0.3,
            "total_questions": 100,
            "correct_answers": 82,
            "metrics": {
                "exact_match_accuracy": 0.82,
                "f1_score": 0.85,
            },
            "items": [
                {
                    "question": "What is the capital of France?",
                    "expected": "Paris",
                    "response": "Paris",
                    "correct": True,
                    "f1_score": 1.0,
                    "exact_match": True,
                },
                {
                    "question": "Who wrote Romeo and Juliet?",
                    "expected": "William Shakespeare",
                    "response": "Shakespeare",
                    "correct": False,
                    "f1_score": 0.5,
                    "exact_match": False,
                },
            ],
        }

    def test_run_command_help(self, runner):
        """Test run command help text."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "Run SimpleQA benchmark evaluation" in result.stdout
        assert "--model" in result.stdout
        assert "--sample-size" in result.stdout
        assert "--temperature" in result.stdout

    @patch.dict("os.environ", {}, clear=True)
    def test_run_command_no_api_key(self, runner):
        """Test run command fails without API key."""
        result = runner.invoke(app, ["run", "--model", "gpt-4"])
        assert result.exit_code == 1
        assert "OPENAI_API_KEY" in result.stdout

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("cosmos_coherence.benchmarks.simpleqa_cli.asyncio.run")
    def test_run_command_basic(self, mock_async_run, runner):
        """Test basic run command execution."""
        # Mock the async run to return sample results
        mock_async_run.return_value = {
            "model": "gpt-4",
            "temperature": 0.3,
            "total_questions": 10,
            "correct_answers": 8,
            "metrics": {
                "exact_match_accuracy": 0.8,
                "f1_score": 0.85,
            },
            "items": [],
        }

        result = runner.invoke(app, ["run", "--model", "gpt-4", "--sample-size", "10"])
        assert result.exit_code == 0
        assert "Running SimpleQA Benchmark" in result.stdout
        assert "Model: gpt-4" in result.stdout
        mock_async_run.assert_called_once()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("cosmos_coherence.benchmarks.simpleqa_cli.asyncio.run")
    def test_run_command_with_output(self, mock_async_run, runner):
        """Test run command with output file."""
        mock_async_run.return_value = {
            "model": "gpt-4",
            "temperature": 0.3,
            "total_questions": 5,
            "correct_answers": 4,
            "metrics": {
                "exact_match_accuracy": 0.8,
                "f1_score": 0.85,
            },
            "items": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.json"
            result = runner.invoke(
                app, ["run", "--model", "gpt-4", "--sample-size", "5", "--output", str(output_path)]
            )

            assert result.exit_code == 0
            assert output_path.exists()

            # Check saved results
            with open(output_path) as f:
                saved_results = json.load(f)
            assert saved_results["model"] == "gpt-4"
            assert saved_results["total_questions"] == 5

    def test_validate_baseline_command_help(self, runner):
        """Test validate-baseline command help text."""
        result = runner.invoke(app, ["validate-baseline", "--help"])
        assert result.exit_code == 0
        assert "Compare benchmark results against published baselines" in result.stdout

    def test_validate_baseline_missing_file(self, runner):
        """Test validate-baseline with missing file."""
        result = runner.invoke(app, ["validate-baseline", "nonexistent.json"])
        assert result.exit_code == 1
        assert "not found" in result.stdout

    def test_validate_baseline_gpt4(self, runner, sample_results):
        """Test baseline validation for GPT-4."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_results, f)
            temp_path = f.name

        try:
            result = runner.invoke(app, ["validate-baseline", temp_path])
            assert result.exit_code == 0
            assert "Baseline Comparison" in result.stdout
            assert "82.0%" in result.stdout  # Exact match accuracy
            assert "Within 5%" in result.stdout  # Status
        finally:
            Path(temp_path).unlink()

    def test_validate_baseline_with_details(self, runner, sample_results):
        """Test baseline validation with detailed output."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_results, f)
            temp_path = f.name

        try:
            result = runner.invoke(app, ["validate-baseline", temp_path, "--details"])
            assert result.exit_code == 0
            assert "Detailed Metrics" in result.stdout
            assert "F1 Score" in result.stdout
            assert "Total Questions" in result.stdout
        finally:
            Path(temp_path).unlink()

    def test_export_command_help(self, runner):
        """Test export command help text."""
        result = runner.invoke(app, ["export", "--help"])
        assert result.exit_code == 0
        assert "Export results to JSONL format" in result.stdout

    def test_export_to_jsonl(self, runner, sample_results):
        """Test exporting results to JSONL format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create input file
            input_path = Path(tmpdir) / "results.json"
            with open(input_path, "w") as f:
                json.dump(sample_results, f)

            # Export to JSONL
            output_path = Path(tmpdir) / "export.jsonl"
            result = runner.invoke(app, ["export", str(input_path), str(output_path)])

            assert result.exit_code == 0
            assert output_path.exists()

            # Check JSONL content
            with open(output_path) as f:
                lines = f.readlines()

            # First line should be metadata
            metadata = json.loads(lines[0])
            assert metadata["type"] == "experiment_metadata"
            assert metadata["benchmark"] == "SimpleQA"

            # Subsequent lines should be items
            assert len(lines) == 3  # metadata + 2 items
            item1 = json.loads(lines[1])
            assert item1["question"] == "What is the capital of France?"

    def test_export_without_metadata(self, runner, sample_results):
        """Test exporting without metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "results.json"
            with open(input_path, "w") as f:
                json.dump(sample_results, f)

            output_path = Path(tmpdir) / "export.jsonl"
            result = runner.invoke(
                app, ["export", str(input_path), str(output_path), "--no-metadata"]
            )

            assert result.exit_code == 0

            with open(output_path) as f:
                lines = f.readlines()

            # Should only have items, no metadata
            assert len(lines) == 2
            item1 = json.loads(lines[0])
            assert "question" in item1  # First line is an item, not metadata

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("cosmos_coherence.benchmarks.simpleqa_cli.load_config")
    @patch("cosmos_coherence.benchmarks.simpleqa_cli.asyncio.run")
    def test_run_with_config_file(self, mock_async_run, mock_load_config, runner):
        """Test run command with configuration file."""
        # Mock config loader
        mock_config = MagicMock()
        mock_config.model = MagicMock()
        mock_config.model.model_type = MagicMock()
        mock_config.model.model_type.value = "gpt-4"
        mock_config.model.temperature = 0.4
        mock_config.benchmark = MagicMock()
        mock_config.benchmark.sample_size = 50
        mock_load_config.return_value = mock_config

        mock_async_run.return_value = {
            "model": "gpt-4",
            "temperature": 0.1,
            "total_questions": 50,
            "correct_answers": 40,
            "metrics": {"exact_match_accuracy": 0.8, "f1_score": 0.85},
            "items": [],
        }

        with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
            result = runner.invoke(app, ["run", "--config", f.name])
            assert result.exit_code == 0
            mock_load_config.assert_called_once()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("cosmos_coherence.benchmarks.simpleqa_cli.asyncio.run")
    def test_run_with_temperature(self, mock_async_run, runner):
        """Test run command with custom temperature."""
        mock_async_run.return_value = {
            "model": "gpt-4",
            "temperature": 0.5,
            "total_questions": 10,
            "correct_answers": 7,
            "metrics": {"exact_match_accuracy": 0.7, "f1_score": 0.75},
            "items": [],
        }

        result = runner.invoke(
            app, ["run", "--model", "gpt-4", "--temperature", "0.5", "--sample-size", "10"]
        )

        assert result.exit_code == 0
        assert "Temperature: 0.5" in result.stdout

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("cosmos_coherence.benchmarks.simpleqa_cli.asyncio.run")
    def test_run_with_verbose(self, mock_async_run, runner):
        """Test run command with verbose output."""
        mock_async_run.return_value = {
            "model": "gpt-4",
            "temperature": 0.3,
            "total_questions": 2,
            "correct_answers": 2,
            "metrics": {"exact_match_accuracy": 1.0, "f1_score": 1.0},
            "items": [],
        }

        result = runner.invoke(app, ["run", "--model", "gpt-4", "--sample-size", "2", "--verbose"])

        assert result.exit_code == 0

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
    @patch("cosmos_coherence.benchmarks.simpleqa_cli.asyncio.run")
    def test_run_no_progress(self, mock_async_run, runner):
        """Test run command without progress bar."""
        mock_async_run.return_value = {
            "model": "gpt-4",
            "temperature": 0.3,
            "total_questions": 5,
            "correct_answers": 4,
            "metrics": {"exact_match_accuracy": 0.8, "f1_score": 0.85},
            "items": [],
        }

        result = runner.invoke(
            app, ["run", "--model", "gpt-4", "--sample-size", "5", "--no-progress"]
        )

        assert result.exit_code == 0


class TestSimpleQACLIIntegration:
    """Integration tests for SimpleQA CLI."""

    @pytest.mark.asyncio
    async def test_evaluate_item(self):
        """Test evaluating a single item."""
        from cosmos_coherence.benchmarks.implementations.simpleqa_benchmark import (
            SimpleQABenchmark,
        )
        from cosmos_coherence.benchmarks.models.datasets import SimpleQAItem
        from cosmos_coherence.benchmarks.simpleqa_cli import _evaluate_item

        benchmark = SimpleQABenchmark(use_huggingface=False)

        # Mock client
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Paris"
        mock_client.generate_response = AsyncMock(return_value=mock_response)

        item = SimpleQAItem(question="What is the capital of France?", best_answer="Paris")

        result = await _evaluate_item(benchmark, mock_client, item, "gpt-4", 0.3, verbose=False)

        assert result["question"] == "What is the capital of France?"
        assert result["expected"] == "Paris"
        assert result["response"] == "Paris"
        assert result["correct"] is True
        assert result["f1_score"] == 1.0
        assert result["exact_match"] is True
