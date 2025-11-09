"""Tests for SimpleQACoherenceBenchmark class."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import numpy as np
import pytest

from cosmos_coherence.benchmarks.implementations.simpleqa_coherence_benchmark import (
    SimpleQACoherenceBenchmark,
)
from cosmos_coherence.benchmarks.models.datasets import SimpleQAItem


class TestSimpleQACoherenceBenchmark:
    """Test suite for SimpleQACoherenceBenchmark."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock OpenAI client."""
        client = AsyncMock()
        client.generate_response = AsyncMock()
        return client

    @pytest.fixture
    def sample_item(self):
        """Create a sample SimpleQA item for testing."""
        return SimpleQAItem(
            question="What is the capital of France?",
            best_answer="Paris",
            metadata={
                "category": "geography",
                "difficulty": "easy"
            }
        )

    @pytest.mark.asyncio
    async def test_benchmark_initialization_with_coherence_variant(self, mock_client):
        """Test: Benchmark initializes with coherence variant correctly."""
        # Test with Shogenji variant
        benchmark = SimpleQACoherenceBenchmark(
            client=mock_client,
            coherence_variant="shogenji",
            coherence_threshold=0.5,
            num_variations=5,
        )

        assert benchmark.coherence_variant_name == "shogenji"
        assert benchmark.coherence_threshold == 0.5
        assert benchmark.num_variations == 5
        assert benchmark.variation_generator is not None
        assert benchmark.coherence_integration is not None
        assert benchmark.answer_strategy is not None

        # Test with Fitelson variant
        benchmark_fitelson = SimpleQACoherenceBenchmark(
            client=mock_client,
            coherence_variant="fitelson",
        )
        assert benchmark_fitelson.coherence_variant_name == "fitelson"

        # Test with Olsson variant
        benchmark_olsson = SimpleQACoherenceBenchmark(
            client=mock_client,
            coherence_variant="olsson",
        )
        assert benchmark_olsson.coherence_variant_name == "olsson"

        # Test invalid variant name
        with pytest.raises(ValueError, match="Unknown coherence variant"):
            SimpleQACoherenceBenchmark(
                client=mock_client,
                coherence_variant="invalid_variant",
            )

    @pytest.mark.asyncio
    async def test_evaluate_with_coherence_returns_expected_structure(
        self, mock_client, sample_item
    ):
        """Test: evaluate_with_coherence() returns expected result structure."""
        # Set up mocks
        mock_variations = [
            "Which city is France's capital?",
            "What city serves as the capital of France?"
        ]

        benchmark = SimpleQACoherenceBenchmark(
            client=mock_client,
            coherence_variant="shogenji",
            coherence_threshold=0.5,
            num_variations=2,
        )

        # Mock the internal components
        with patch.object(
            benchmark.variation_generator, 'generate_variations',
            new_callable=AsyncMock, return_value=mock_variations
        ):
            with patch.object(
                benchmark.coherence_integration, 'score_question_variations',
                return_value=0.8  # High coherence
            ):
                with patch.object(
                    mock_client, 'generate_response',
                    new_callable=AsyncMock
                ) as mock_response:
                    # Mock model answer
                    mock_response.return_value = Mock(content="Paris")

                    # Mock grader
                    with patch.object(
                        benchmark, '_grader'
                    ) as mock_grader:
                        mock_grader.grade_response = AsyncMock(
                            return_value=("CORRECT", {"grader_model": "gpt-4o-mini"})
                        )

                        result = await benchmark.evaluate_with_coherence(sample_item)

        # Verify result structure
        assert "question" in result
        assert "ground_truth" in result
        assert "question_variations" in result
        assert "coherence_score" in result
        assert "threshold_met" in result
        assert "question_type" in result
        assert "final_answer" in result
        assert "grade" in result
        assert "is_correct" in result
        assert "metadata" in result

        # Verify values
        assert result["question"] == sample_item.question
        assert result["ground_truth"] == sample_item.best_answer
        assert result["question_variations"] == mock_variations
        assert result["coherence_score"] == 0.8
        assert result["threshold_met"] is True
        assert result["question_type"] == "factual"  # Not YES/NO
        assert result["final_answer"] == "Paris"
        assert result["grade"] == "CORRECT"
        assert result["is_correct"] is True

    @pytest.mark.asyncio
    async def test_grading_integration_calls_grader_correctly(
        self, mock_client, sample_item
    ):
        """Test: Grading integration works (calls SimpleQAGrader correctly)."""
        benchmark = SimpleQACoherenceBenchmark(
            client=mock_client,
            coherence_variant="shogenji",
        )

        # Mock components
        mock_variations = ["Variation 1", "Variation 2"]

        with patch.object(
            benchmark.variation_generator, 'generate_variations',
            new_callable=AsyncMock, return_value=mock_variations
        ):
            with patch.object(
                benchmark.coherence_integration, 'score_question_variations',
                return_value=0.7
            ):
                with patch.object(
                    mock_client, 'generate_response',
                    new_callable=AsyncMock
                ) as mock_response:
                    mock_response.return_value = Mock(content="Paris")

                    with patch.object(
                        benchmark, '_grader'
                    ) as mock_grader:
                        mock_grader.grade_response = AsyncMock(
                            return_value=("CORRECT", {"grader_model": "gpt-4o-mini"})
                        )

                        await benchmark.evaluate_with_coherence(sample_item)

                        # Verify grader was called with correct arguments
                        mock_grader.grade_response.assert_called_once()
                        call_args = mock_grader.grade_response.call_args
                        assert call_args.kwargs['question'] == sample_item.question
                        assert call_args.kwargs['expert_answer'] == sample_item.best_answer
                        assert call_args.kwargs['submission'] == "Paris"

    @pytest.mark.asyncio
    async def test_metadata_tracking_complete(self, mock_client, sample_item):
        """Test: Metadata tracking complete (variations, scores, decisions)."""
        benchmark = SimpleQACoherenceBenchmark(
            client=mock_client,
            coherence_variant="shogenji",
            coherence_threshold=0.6,
            num_variations=3,
        )

        mock_variations = ["Var1", "Var2", "Var3"]
        coherence_score = 0.4  # Low coherence (below threshold)

        with patch.object(
            benchmark.variation_generator, 'generate_variations',
            new_callable=AsyncMock, return_value=mock_variations
        ):
            with patch.object(
                benchmark.coherence_integration, 'score_question_variations',
                return_value=coherence_score
            ):
                with patch.object(
                    mock_client, 'generate_response',
                    new_callable=AsyncMock
                ) as mock_response:
                    mock_response.return_value = Mock(content="Paris")

                    with patch.object(
                        benchmark, '_grader'
                    ) as mock_grader:
                        mock_grader.grade_response = AsyncMock(
                            return_value=("NOT_ATTEMPTED", {})
                        )

                        result = await benchmark.evaluate_with_coherence(sample_item)

        # Verify comprehensive metadata
        metadata = result["metadata"]
        assert "num_variations" in metadata
        assert "coherence_variant" in metadata
        assert "coherence_threshold" in metadata
        assert "grading_metadata" in metadata

        assert metadata["num_variations"] == 3
        assert metadata["coherence_variant"] == "shogenji"
        assert metadata["coherence_threshold"] == 0.6

        # Verify decision tracking
        assert result["threshold_met"] is False  # Score 0.4 < threshold 0.6
        assert result["final_answer"] == "I don't know"  # Should abstain
        assert result["grade"] == "NOT_ATTEMPTED"

    @pytest.mark.asyncio
    async def test_checkpoint_save_load(self, mock_client, tmp_path):
        """Test: Checkpoint save/load works (serialize/deserialize state)."""
        benchmark = SimpleQACoherenceBenchmark(
            client=mock_client,
            coherence_variant="shogenji",
        )

        # Create sample results
        results = [
            {
                "question": "Question 1",
                "ground_truth": "Answer 1",
                "coherence_score": 0.8,
                "final_answer": "Answer 1",
                "is_correct": True,
            },
            {
                "question": "Question 2",
                "ground_truth": "Answer 2",
                "coherence_score": 0.3,
                "final_answer": "I don't know",
                "is_correct": False,
            },
        ]

        checkpoint_path = tmp_path / "test_checkpoint.json"

        # Save checkpoint
        benchmark.save_checkpoint(
            results=results,
            checkpoint_path=checkpoint_path,
            last_index=1,
            config={
                "model": "gpt-4o-mini",
                "coherence_variant": "shogenji",
                "num_variations": 5,
            }
        )

        # Verify checkpoint file was created
        assert checkpoint_path.exists()

        # Load checkpoint
        loaded_results, loaded_index = benchmark.load_checkpoint(checkpoint_path)

        # Verify loaded data matches saved data
        assert loaded_results == results
        assert loaded_index == 1

        # Verify checkpoint structure
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)

        assert "config" in checkpoint_data
        assert "progress" in checkpoint_data
        assert "partial_results" in checkpoint_data
        assert checkpoint_data["progress"]["questions_completed"] == 2
        assert checkpoint_data["progress"]["last_index"] == 1
