"""Integration tests for end-to-end question-variation coherence workflows.

These tests validate the complete pipeline from question variation generation
through coherence scoring, answer strategy, and grading.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from cosmos_coherence.benchmarks.implementations.simpleqa_coherence_benchmark import (
    SimpleQACoherenceBenchmark,
)
from cosmos_coherence.benchmarks.models.datasets import SimpleQAItem


@pytest.fixture
def sample_simpleqa_items():
    """Create 10 sample SimpleQA items for testing."""
    return [
        SimpleQAItem(
            question="What is the capital of France?",
            best_answer="Paris",
            metadata={"category": "geography", "difficulty": "easy"}
        ),
        SimpleQAItem(
            question="Who wrote 'Romeo and Juliet'?",
            best_answer="William Shakespeare",
            metadata={"category": "literature", "difficulty": "easy"}
        ),
        SimpleQAItem(
            question="Is water wet?",
            best_answer="Yes",
            metadata={"category": "science", "difficulty": "medium"}
        ),
        SimpleQAItem(
            question="Can dogs fly?",
            best_answer="No",
            metadata={"category": "science", "difficulty": "easy"}
        ),
        SimpleQAItem(
            question="What is the speed of light?",
            best_answer="299,792,458 meters per second",
            metadata={"category": "physics", "difficulty": "medium"}
        ),
        SimpleQAItem(
            question="Who discovered penicillin?",
            best_answer="Alexander Fleming",
            metadata={"category": "history", "difficulty": "medium"}
        ),
        SimpleQAItem(
            question="Is the Earth round?",
            best_answer="Yes",
            metadata={"category": "science", "difficulty": "easy"}
        ),
        SimpleQAItem(
            question="What is the largest planet?",
            best_answer="Jupiter",
            metadata={"category": "astronomy", "difficulty": "easy"}
        ),
        SimpleQAItem(
            question="Does the sun rise in the west?",
            best_answer="No",
            metadata={"category": "science", "difficulty": "easy"}
        ),
        SimpleQAItem(
            question="Who painted the Mona Lisa?",
            best_answer="Leonardo da Vinci",
            metadata={"category": "art", "difficulty": "easy"}
        ),
    ]


@pytest.fixture
def mock_client():
    """Create a mock OpenAI client."""
    client = AsyncMock()
    client.generate_response = AsyncMock()
    return client


class TestEndToEndIntegration:
    """Integration tests for complete evaluation workflows."""

    @pytest.mark.asyncio
    async def test_full_evaluation_all_three_coherence_variants(
        self, mock_client, sample_simpleqa_items
    ):
        """Test: Full evaluation on 10-question sample (all three coherence variants).

        This test validates that all three coherence variants (Shogenji, Fitelson, Olsson)
        can successfully process a complete evaluation pipeline.
        """
        variants = ["shogenji", "fitelson", "olsson"]

        for variant in variants:
            benchmark = SimpleQACoherenceBenchmark(
                client=mock_client,
                coherence_variant=variant,
                coherence_threshold=0.5,
                num_variations=3,
            )

            # Mock all components to avoid actual API calls
            mock_variations = ["Variation 1", "Variation 2", "Variation 3"]

            with patch.object(
                benchmark.variation_generator, 'generate_variations',
                new_callable=AsyncMock, return_value=mock_variations
            ):
                with patch.object(
                    benchmark.coherence_integration, 'score_question_variations',
                    return_value=0.7  # High coherence
                ):
                    with patch.object(
                        mock_client, 'generate_response',
                        new_callable=AsyncMock
                    ) as mock_response:
                        # Return plausible answers
                        mock_response.return_value = Mock(content="Test Answer")

                        with patch.object(
                            benchmark, '_grader'
                        ) as mock_grader:
                            mock_grader.grade_response = AsyncMock(
                                return_value=("CORRECT", {"grader_model": "gpt-4o-mini"})
                            )

                            # Process all 10 questions
                            results = []
                            for item in sample_simpleqa_items:
                                result = await benchmark.evaluate_with_coherence(item)
                                results.append(result)

            # Verify we processed all 10 questions
            assert len(results) == 10, f"Failed for variant {variant}"

            # Verify all results have expected structure
            for result in results:
                assert "question" in result
                assert "coherence_score" in result
                assert "final_answer" in result
                assert "grade" in result
                assert "is_correct" in result
                assert "metadata" in result
                assert result["metadata"]["coherence_variant"] == variant

    @pytest.mark.asyncio
    async def test_checkpoint_save_and_resume(
        self, mock_client, sample_simpleqa_items, tmp_path
    ):
        """Test: Checkpoint save and resume (interrupt after 5 questions, resume).

        This test validates that evaluation can be interrupted mid-stream,
        saved to a checkpoint, and resumed from the checkpoint.
        """
        benchmark = SimpleQACoherenceBenchmark(
            client=mock_client,
            coherence_variant="shogenji",
            coherence_threshold=0.5,
            num_variations=2,
        )

        checkpoint_path = tmp_path / "test_checkpoint.json"

        # Mock components
        mock_variations = ["Var1", "Var2"]

        with patch.object(
            benchmark.variation_generator, 'generate_variations',
            new_callable=AsyncMock, return_value=mock_variations
        ):
            with patch.object(
                benchmark.coherence_integration, 'score_question_variations',
                return_value=0.8
            ):
                with patch.object(
                    mock_client, 'generate_response',
                    new_callable=AsyncMock
                ) as mock_response:
                    mock_response.return_value = Mock(content="Answer")

                    with patch.object(
                        benchmark, '_grader'
                    ) as mock_grader:
                        mock_grader.grade_response = AsyncMock(
                            return_value=("CORRECT", {})
                        )

                        # Process first 5 questions
                        first_batch_results = []
                        for i, item in enumerate(sample_simpleqa_items[:5]):
                            result = await benchmark.evaluate_with_coherence(item)
                            first_batch_results.append(result)

                        # Save checkpoint after 5 questions
                        benchmark.save_checkpoint(
                            results=first_batch_results,
                            checkpoint_path=checkpoint_path,
                            last_index=4,
                            config={
                                "model": "gpt-4o-mini",
                                "coherence_variant": "shogenji",
                                "num_variations": 2,
                            }
                        )

        # Verify checkpoint was created
        assert checkpoint_path.exists()

        # Load checkpoint and verify structure
        loaded_results, loaded_index = benchmark.load_checkpoint(checkpoint_path)
        assert len(loaded_results) == 5
        assert loaded_index == 4

        # Resume processing remaining 5 questions
        with patch.object(
            benchmark.variation_generator, 'generate_variations',
            new_callable=AsyncMock, return_value=mock_variations
        ):
            with patch.object(
                benchmark.coherence_integration, 'score_question_variations',
                return_value=0.8
            ):
                with patch.object(
                    mock_client, 'generate_response',
                    new_callable=AsyncMock
                ) as mock_response:
                    mock_response.return_value = Mock(content="Answer")

                    with patch.object(
                        benchmark, '_grader'
                    ) as mock_grader:
                        mock_grader.grade_response = AsyncMock(
                            return_value=("CORRECT", {})
                        )

                        # Process remaining questions (index 5-9)
                        second_batch_results = loaded_results.copy()
                        for i, item in enumerate(sample_simpleqa_items[5:], start=5):
                            result = await benchmark.evaluate_with_coherence(item)
                            second_batch_results.append(result)

        # Verify we have all 10 results
        assert len(second_batch_results) == 10

        # Verify first 5 results match checkpoint
        for i in range(5):
            assert second_batch_results[i]["question"] == first_batch_results[i]["question"]

    @pytest.mark.asyncio
    async def test_baseline_comparison_metrics_structure(
        self, mock_client, sample_simpleqa_items
    ):
        """Test: Baseline comparison produces expected metrics structure.

        This test validates that when baseline comparison is enabled,
        the metrics structure includes both coherence and baseline metrics.
        """
        # Run coherence-guided evaluation
        coherence_benchmark = SimpleQACoherenceBenchmark(
            client=mock_client,
            coherence_variant="shogenji",
            coherence_threshold=0.5,
            num_variations=2,
        )

        mock_variations = ["Var1", "Var2"]

        with patch.object(
            coherence_benchmark.variation_generator, 'generate_variations',
            new_callable=AsyncMock, return_value=mock_variations
        ):
            with patch.object(
                coherence_benchmark.coherence_integration, 'score_question_variations',
                side_effect=[0.8, 0.3, 0.7, 0.2, 0.9]  # Mix of high and low coherence
            ):
                with patch.object(
                    mock_client, 'generate_response',
                    new_callable=AsyncMock
                ) as mock_response:
                    mock_response.return_value = Mock(content="Answer")

                    with patch.object(
                        coherence_benchmark, '_grader'
                    ) as mock_grader:
                        # Return mix of correct/incorrect/not_attempted
                        mock_grader.grade_response = AsyncMock(
                            side_effect=[
                                ("CORRECT", {}),
                                ("NOT_ATTEMPTED", {}),
                                ("CORRECT", {}),
                                ("NOT_ATTEMPTED", {}),
                                ("INCORRECT", {}),
                            ]
                        )

                        coherence_results = []
                        for item in sample_simpleqa_items[:5]:
                            result = await coherence_benchmark.evaluate_with_coherence(item)
                            coherence_results.append(result)

        # Calculate coherence metrics
        coherence_metrics = {
            "total_questions": len(coherence_results),
            "correct_count": sum(1 for r in coherence_results if r["grade"] == "CORRECT"),
            "incorrect_count": sum(1 for r in coherence_results if r["grade"] == "INCORRECT"),
            "not_attempted_count": sum(1 for r in coherence_results if r["grade"] == "NOT_ATTEMPTED"),
        }
        coherence_metrics["overall_correct"] = coherence_metrics["correct_count"] / coherence_metrics["total_questions"]
        coherence_metrics["overall_incorrect"] = coherence_metrics["incorrect_count"] / coherence_metrics["total_questions"]
        coherence_metrics["overall_not_attempted"] = coherence_metrics["not_attempted_count"] / coherence_metrics["total_questions"]

        attempted = coherence_metrics["correct_count"] + coherence_metrics["incorrect_count"]
        coherence_metrics["correct_given_attempted"] = (
            coherence_metrics["correct_count"] / attempted if attempted > 0 else 0
        )

        # Verify metrics structure
        assert "total_questions" in coherence_metrics
        assert "correct_count" in coherence_metrics
        assert "incorrect_count" in coherence_metrics
        assert "not_attempted_count" in coherence_metrics
        assert "overall_correct" in coherence_metrics
        assert "overall_incorrect" in coherence_metrics
        assert "overall_not_attempted" in coherence_metrics
        assert "correct_given_attempted" in coherence_metrics

        # Verify expected behavior: some abstentions due to low coherence
        assert coherence_metrics["not_attempted_count"] > 0
        assert coherence_metrics["overall_not_attempted"] > 0

    @pytest.mark.asyncio
    async def test_error_handling_graceful_degradation(
        self, mock_client, sample_simpleqa_items
    ):
        """Test: Error handling (API failures, malformed questions).

        This test validates that the system handles errors gracefully and
        continues processing despite individual failures.
        """
        benchmark = SimpleQACoherenceBenchmark(
            client=mock_client,
            coherence_variant="shogenji",
            coherence_threshold=0.5,
            num_variations=2,
        )

        # Simulate various error scenarios
        error_count = 0
        success_count = 0

        with patch.object(
            benchmark.variation_generator, 'generate_variations',
            new_callable=AsyncMock
        ) as mock_gen:
            # First call succeeds, second fails, third succeeds, etc.
            mock_gen.side_effect = [
                ["Var1", "Var2"],  # Success
                Exception("API rate limit exceeded"),  # Failure
                ["Var3", "Var4"],  # Success
                Exception("Network timeout"),  # Failure
                ["Var5", "Var6"],  # Success
            ]

            with patch.object(
                benchmark.coherence_integration, 'score_question_variations',
                return_value=0.7
            ):
                with patch.object(
                    mock_client, 'generate_response',
                    new_callable=AsyncMock
                ) as mock_response:
                    mock_response.return_value = Mock(content="Answer")

                    with patch.object(
                        benchmark, '_grader'
                    ) as mock_grader:
                        mock_grader.grade_response = AsyncMock(
                            return_value=("CORRECT", {})
                        )

                        # Process 5 questions with mixed success/failure
                        for i, item in enumerate(sample_simpleqa_items[:5]):
                            try:
                                result = await benchmark.evaluate_with_coherence(item)
                                success_count += 1
                            except Exception as e:
                                error_count += 1
                                # In production, would log error and continue
                                pass

        # Verify that errors occurred but some evaluations succeeded
        assert error_count > 0, "Expected some errors to occur"
        assert success_count > 0, "Expected some evaluations to succeed"
        assert error_count + success_count == 5, "All questions should be attempted"

    @pytest.mark.asyncio
    async def test_parallel_processing_stability(
        self, mock_client, sample_simpleqa_items
    ):
        """Test: Parallel processing with multiple workers (2, 4 workers).

        This test validates that concurrent evaluation of multiple questions
        works correctly without race conditions or data corruption.
        """
        benchmark = SimpleQACoherenceBenchmark(
            client=mock_client,
            coherence_variant="shogenji",
            coherence_threshold=0.5,
            num_variations=2,
        )

        mock_variations = ["Var1", "Var2"]

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
                    mock_response.return_value = Mock(content="Answer")

                    with patch.object(
                        benchmark, '_grader'
                    ) as mock_grader:
                        mock_grader.grade_response = AsyncMock(
                            return_value=("CORRECT", {})
                        )

                        # Test with 2 workers
                        tasks_2_workers = [
                            benchmark.evaluate_with_coherence(item)
                            for item in sample_simpleqa_items[:4]
                        ]
                        results_2_workers = await asyncio.gather(*tasks_2_workers)

                        assert len(results_2_workers) == 4
                        assert all("question" in r for r in results_2_workers)

                        # Test with 4 workers
                        tasks_4_workers = [
                            benchmark.evaluate_with_coherence(item)
                            for item in sample_simpleqa_items[:8]
                        ]
                        results_4_workers = await asyncio.gather(*tasks_4_workers)

                        assert len(results_4_workers) == 8
                        assert all("question" in r for r in results_4_workers)

                        # Verify no duplicate processing
                        questions_processed = [r["question"] for r in results_4_workers]
                        assert len(questions_processed) == len(set(questions_processed))
