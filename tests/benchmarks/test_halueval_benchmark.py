"""Tests for HaluEval benchmark implementation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cosmos_coherence.benchmarks.implementations.halueval_benchmark import (
    HaluEvalBenchmark,
)
from cosmos_coherence.benchmarks.models.datasets import HaluEvalItem, HaluEvalTaskType
from cosmos_coherence.llm.models import ModelResponse, TokenUsage


class TestHaluEvalBenchmark:
    """Test HaluEval benchmark implementation."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock OpenAI client."""
        client = MagicMock()
        client.generate_response = AsyncMock()
        return client

    @pytest.fixture
    def benchmark(self, mock_client):
        """Create a HaluEval benchmark instance."""
        return HaluEvalBenchmark(client=mock_client, use_huggingface=False)

    @pytest.fixture
    def sample_qa_item(self):
        """Create a sample QA item."""
        return HaluEvalItem(
            question="What is the capital of France?",
            knowledge="Paris is the capital and largest city of France.",
            task_type=HaluEvalTaskType.QA,
            right_answer="Paris is the capital of France.",
            hallucinated_answer="London is the capital of France.",
        )

    @pytest.fixture
    def sample_dialogue_item(self):
        """Create a sample dialogue item."""
        return HaluEvalItem(
            question="Respond to the user",
            knowledge="Be helpful and friendly.",
            task_type=HaluEvalTaskType.DIALOGUE,
            right_answer="I'd be happy to help you today.",
            hallucinated_answer="Our company was founded in 1985.",
            dialogue_history=["User: Hello", "Agent: Hi there!"],
        )

    @pytest.fixture
    def sample_summarization_item(self):
        """Create a sample summarization item."""
        return HaluEvalItem(
            question="Summarize the document",
            knowledge="Focus on main points.",
            task_type=HaluEvalTaskType.SUMMARIZATION,
            right_answer="The article discusses climate change impacts.",
            hallucinated_answer="The article denies climate change.",
            document="Climate change is affecting global temperatures...",
        )

    def test_initialization(self, mock_client):
        """Test benchmark initialization."""
        benchmark = HaluEvalBenchmark(client=mock_client)
        assert benchmark.client == mock_client
        assert benchmark.benchmark_name == "HaluEval"

    def test_get_prompt_qa(self, benchmark, sample_qa_item):
        """Test prompt generation for QA task."""
        # Mock random choice to select hallucinated answer
        with patch("random.choice", return_value=True):
            prompt, is_hallucinated = benchmark.get_prompt_with_selection(sample_qa_item)

        assert "#Question#:" in prompt
        assert sample_qa_item.question in prompt
        assert "#Answer#:" in prompt
        assert sample_qa_item.hallucinated_answer in prompt
        assert is_hallucinated is True

    def test_get_prompt_dialogue(self, benchmark, sample_dialogue_item):
        """Test prompt generation for dialogue task."""
        with patch("random.choice", return_value=False):
            prompt, is_hallucinated = benchmark.get_prompt_with_selection(sample_dialogue_item)

        assert "#Dialogue History#:" in prompt
        assert "#Response#:" in prompt
        assert sample_dialogue_item.right_answer in prompt
        assert is_hallucinated is False

    def test_get_prompt_summarization(self, benchmark, sample_summarization_item):
        """Test prompt generation for summarization task."""
        with patch("random.choice", return_value=False):
            prompt, is_hallucinated = benchmark.get_prompt_with_selection(sample_summarization_item)

        assert "#Document#:" in prompt
        assert "#Summary#:" in prompt
        assert sample_summarization_item.document in prompt
        assert sample_summarization_item.right_answer in prompt
        assert is_hallucinated is False

    def test_evaluate_response_correct_detection(self, benchmark, mock_client, sample_qa_item):
        """Test evaluating response with correct hallucination detection."""
        # Mock LLM response
        mock_client.generate_response.return_value = ModelResponse(
            content="Yes",
            model="gpt-4",
            usage=TokenUsage(
                prompt_tokens=100,
                completion_tokens=1,
                total_tokens=101,
                estimated_cost=0.001,
            ),
            request_id="test-id",
            latency_ms=100.0,
            temperature=0.0,
            finish_reason="stop",
            cached=False,
        )

        # Evaluate with hallucinated answer
        result = benchmark.evaluate_response(
            response="Yes",
            ground_truth="hallucinated",
            item=sample_qa_item,
        )

        assert result.is_correct is True
        assert result.score == 1.0
        assert result.metadata["prediction"] == "Yes"
        assert result.metadata["expected"] == "Yes"

    def test_evaluate_response_incorrect_detection(self, benchmark, mock_client, sample_qa_item):
        """Test evaluating response with incorrect hallucination detection."""
        # Mock LLM response
        mock_client.generate_response.return_value = ModelResponse(
            content="No",
            model="gpt-4",
            usage=TokenUsage(
                prompt_tokens=100,
                completion_tokens=1,
                total_tokens=101,
                estimated_cost=0.001,
            ),
            request_id="test-id",
            latency_ms=100.0,
            temperature=0.0,
            finish_reason="stop",
            cached=False,
        )

        # Evaluate with hallucinated answer (should say Yes)
        result = benchmark.evaluate_response(
            response="No",
            ground_truth="hallucinated",
            item=sample_qa_item,
        )

        assert result.is_correct is False
        assert result.score == 0.0
        assert result.metadata["prediction"] == "No"
        assert result.metadata["expected"] == "Yes"

    @pytest.mark.asyncio
    async def test_evaluate_response_parsing(self, benchmark, mock_client, sample_qa_item):
        """Test response parsing for various formats."""
        test_cases = [
            ("Yes", "Yes"),
            ("No", "No"),
            ("YES", "Yes"),
            ("no", "No"),
            ("Yes, this is hallucinated", "Yes"),
            ("No hallucination detected", "No"),
            ("Invalid response", "No"),  # Default to No
        ]

        for response_text, expected_parsed in test_cases:
            mock_client.generate_response.return_value = ModelResponse(
                content=response_text,
                model="gpt-4",
                usage=TokenUsage(
                    prompt_tokens=100,
                    completion_tokens=5,
                    total_tokens=105,
                    estimated_cost=0.001,
                ),
                request_id="test-id",
                latency_ms=100.0,
                temperature=0.0,
                finish_reason="stop",
                cached=False,
            )

            result = benchmark.evaluate_response(
                response=response_text,
                ground_truth="hallucinated",
                item=sample_qa_item,
            )

            assert result.metadata["prediction"] == expected_parsed

    def test_calculate_metrics(self, benchmark):
        """Test metrics calculation."""
        results = [
            {"is_correct": True, "task_type": "qa"},
            {"is_correct": True, "task_type": "qa"},
            {"is_correct": False, "task_type": "qa"},
            {"is_correct": True, "task_type": "dialogue"},
            {"is_correct": False, "task_type": "dialogue"},
            {"is_correct": True, "task_type": "summarization"},
        ]

        metrics = benchmark.calculate_metrics(results)

        assert metrics["overall_accuracy"] == 4 / 6  # 4 correct out of 6
        assert metrics["qa_accuracy"] == 2 / 3  # 2 correct out of 3 QA
        assert metrics["dialogue_accuracy"] == 1 / 2  # 1 correct out of 2 dialogue
        assert metrics["summarization_accuracy"] == 1 / 1  # 1 correct out of 1 summ
        assert metrics["total_samples"] == 6

    def test_get_metadata(self, benchmark):
        """Test getting benchmark metadata."""
        metadata = benchmark.get_metadata()

        assert metadata.name == "HaluEval"
        assert "EMNLP 2023" in metadata.paper_reference
        assert metadata.evaluation_method == "Binary hallucination detection"
        assert "accuracy" in metadata.baseline_metrics

    def test_paper_reference(self, benchmark):
        """Test paper reference information."""
        reference = benchmark.get_paper_reference()

        assert "HaluEval" in reference
        assert "Li et al." in reference
        assert "2023" in reference
        assert "EMNLP" in reference
