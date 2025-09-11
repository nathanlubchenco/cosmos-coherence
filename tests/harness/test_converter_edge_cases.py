"""Additional tests for dataset converter edge cases and robustness."""

# Import the loader directly from the module file
import importlib.util
import os
import uuid
from pathlib import Path

import pytest
from cosmos_coherence.benchmarks.models.base import DatasetValidationError
from cosmos_coherence.benchmarks.models.datasets import (
    FaithBenchItem,
    FEVERLabel,
    HaluEvalTaskType,
    TruthfulQACategory,
)

spec = importlib.util.spec_from_file_location(
    "huggingface_loader",
    os.path.join(
        os.path.dirname(__file__), "../../src/cosmos_coherence/harness/huggingface_loader.py"
    ),
)
huggingface_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(huggingface_loader)
DatasetNotFoundError = huggingface_loader.DatasetNotFoundError
HuggingFaceDatasetLoader = huggingface_loader.HuggingFaceDatasetLoader


class TestConverterEdgeCases:
    """Test edge cases and robustness of dataset converters."""

    @pytest.fixture
    def loader(self):
        """Create a loader instance."""
        return HuggingFaceDatasetLoader(cache_dir=Path("/tmp/test_cache"))

    def test_faithbench_missing_optional_fields(self, loader):
        """Test FaithBench converter with missing optional fields."""
        raw_data = [
            {
                "sample_id": "test_001",
                "summary": "Test claim",
                "source": "Test context",
                # No id, annotations, metadata
            }
        ]

        items = loader._convert_to_pydantic(raw_data, "faithbench")

        assert len(items) == 1
        assert isinstance(items[0], FaithBenchItem)
        assert items[0].summary == "Test claim"
        assert items[0].source == "Test context"
        assert items[0].annotation_label is None
        assert items[0].annotation_spans == []
        assert items[0].entropy_score is None
        assert items[0].detector_predictions == {}
        # ID should be auto-generated
        assert isinstance(items[0].id, uuid.UUID)

    def test_faithbench_claim_as_question(self, loader):
        """Test FaithBench converter auto-generates question field."""
        raw_data = [
            {
                "sample_id": "test_002",
                "summary": "The sky is blue",
                "source": "Scientific fact",
                # No explicit question field
            }
        ]

        items = loader._convert_to_pydantic(raw_data, "faithbench")

        assert len(items) == 1
        # FaithBench generates question from source for compatibility
        assert "Scientific fact" in items[0].question or "Summarize" in items[0].question

    def test_simpleqa_with_answer_field(self, loader):
        """Test SimpleQA converter with 'answer' field instead of 'best_answer'."""
        raw_data = [
            {
                "question": "What is 2+2?",
                "answer": "4",  # Using 'answer' instead of 'best_answer'
                "id": str(uuid.uuid4()),
            }
        ]

        items = loader._convert_to_pydantic(raw_data, "simpleqa")

        assert len(items) == 1
        assert items[0].best_answer == "4"

    def test_simpleqa_with_category_and_difficulty(self, loader):
        """Test SimpleQA converter with optional category and difficulty fields."""
        raw_data = [
            {
                "question": "What is the speed of light?",
                "best_answer": "299,792,458 m/s",
                "category": "science",
                "difficulty": "hard",
                "metadata": {"source": "physics"},
            }
        ]

        items = loader._convert_to_pydantic(raw_data, "simpleqa")

        assert len(items) == 1
        assert items[0].category == "science"
        assert items[0].difficulty == "hard"
        assert items[0].metadata == {"source": "physics"}

    def test_truthfulqa_invalid_category_defaults_to_other(self, loader):
        """Test TruthfulQA converter with invalid category defaults to 'other'."""
        raw_data = [
            {
                "question": "Test question",
                "best_answer": "Test answer",
                "correct_answers": ["Answer 1"],
                "incorrect_answers": ["Wrong 1"],
                "category": "invalid_category",  # Invalid category
            }
        ]

        items = loader._convert_to_pydantic(raw_data, "truthfulqa")

        assert len(items) == 1
        assert items[0].category == TruthfulQACategory.OTHER

    def test_truthfulqa_no_category_defaults_to_other(self, loader):
        """Test TruthfulQA converter without category defaults to 'other'."""
        raw_data = [
            {
                "question": "Test question",
                "best_answer": "Test answer",
                "correct_answers": ["Answer 1"],
                "incorrect_answers": ["Wrong 1"],
                # No category field
            }
        ]

        items = loader._convert_to_pydantic(raw_data, "truthfulqa")

        assert len(items) == 1
        assert items[0].category == TruthfulQACategory.OTHER

    def test_fever_invalid_label_defaults_to_notenoughinfo(self, loader):
        """Test FEVER converter with invalid label defaults to NOTENOUGHINFO."""
        raw_data = [
            {
                "claim": "Test claim",
                "label": "INVALID_LABEL",  # Invalid label
                "evidence": [],
            }
        ]

        items = loader._convert_to_pydantic(raw_data, "fever")

        assert len(items) == 1
        assert items[0].label == FEVERLabel.NOTENOUGHINFO

    def test_fever_missing_label_defaults_to_notenoughinfo(self, loader):
        """Test FEVER converter without label defaults to NOTENOUGHINFO."""
        raw_data = [
            {
                "claim": "Test claim",
                # No label field
                "evidence": [],
            }
        ]

        items = loader._convert_to_pydantic(raw_data, "fever")

        assert len(items) == 1
        assert items[0].label == FEVERLabel.NOTENOUGHINFO

    def test_fever_with_optional_fields(self, loader):
        """Test FEVER converter with optional fields."""
        raw_data = [
            {
                "claim": "The Earth is round",
                "label": "SUPPORTED",
                "evidence": [["Earth", 0, "The Earth is a sphere"]],
                "verdict": "Claim is supported by evidence",
                "wikipedia_url": "https://en.wikipedia.org/wiki/Earth",
                "annotation_id": "anno_123",
                "id": str(uuid.uuid4()),
            }
        ]

        items = loader._convert_to_pydantic(raw_data, "fever")

        assert len(items) == 1
        assert items[0].verdict == "Claim is supported by evidence"
        assert items[0].wikipedia_url == "https://en.wikipedia.org/wiki/Earth"
        assert items[0].annotation_id == "anno_123"

    def test_halueval_with_answer_field(self, loader):
        """Test HaluEval converter with 'answer' field instead of 'right_answer'."""
        raw_data = [
            {
                "question": "What is AI?",
                "knowledge": "AI knowledge",
                "answer": "Correct answer",  # Using 'answer' instead of 'right_answer'
                "hallucinated_answer": "Wrong answer",
                "task_type": "qa",
            }
        ]

        items = loader._convert_to_pydantic(raw_data, "halueval")

        assert len(items) == 1
        assert items[0].right_answer == "Correct answer"

    def test_halueval_with_dialogue_history(self, loader):
        """Test HaluEval converter with dialogue history."""
        raw_data = [
            {
                "question": "What did we discuss?",
                "knowledge": "Previous conversation",
                "right_answer": "We discussed AI",
                "hallucinated_answer": "We discussed weather",
                "task_type": "dialogue",
                "dialogue_history": ["Hello", "Hi there", "Let's talk about AI"],
            }
        ]

        items = loader._convert_to_pydantic(raw_data, "halueval")

        assert len(items) == 1
        assert items[0].task_type == HaluEvalTaskType.DIALOGUE
        assert items[0].dialogue_history == ["Hello", "Hi there", "Let's talk about AI"]

    def test_halueval_default_task_type(self, loader):
        """Test HaluEval converter defaults to 'general' task type."""
        raw_data = [
            {
                "question": "Test",
                "knowledge": "Test knowledge",
                "right_answer": "Right",
                "hallucinated_answer": "Wrong",
                # No task_type field
            }
        ]

        items = loader._convert_to_pydantic(raw_data, "halueval")

        assert len(items) == 1
        assert items[0].task_type == HaluEvalTaskType.GENERAL

    def test_empty_strings_handled_gracefully(self, loader):
        """Test that empty strings are handled gracefully."""
        raw_data = [
            {
                "question": "",  # Empty question
                "best_answer": "Answer",
                "id": str(uuid.uuid4()),
            }
        ]

        # SimpleQA should raise validation error for empty question
        with pytest.raises(DatasetValidationError) as exc_info:
            loader._convert_to_pydantic(raw_data, "simpleqa")
        assert "Validation failed for item 0" in str(exc_info.value)

    def test_whitespace_strings_trimmed(self, loader):
        """Test that whitespace strings are properly handled."""
        raw_data = [
            {
                "sample_id": "test_whitespace",
                "summary": "  Test claim with spaces  ",
                "source": "\n\nContext with newlines\n\n",
            }
        ]

        items = loader._convert_to_pydantic(raw_data, "faithbench")

        assert len(items) == 1
        # Check if whitespace is handled (this depends on model validation)
        assert items[0].summary == "Test claim with spaces"
        assert items[0].source == "Context with newlines"

    def test_batch_conversion_with_mixed_validity(self, loader):
        """Test batch conversion continues despite individual failures."""
        raw_data = [
            {"question": "Valid 1", "best_answer": "Answer 1"},
            {"invalid": "data"},  # This will fail
            {"question": "Valid 2", "best_answer": "Answer 2"},
        ]

        # Should fail on the first invalid item
        with pytest.raises(DatasetValidationError) as exc_info:
            loader._convert_to_pydantic(raw_data, "simpleqa")

        assert "Validation failed for item 1" in str(exc_info.value)
        assert exc_info.value.field == "item_1"

    def test_none_values_handled(self, loader):
        """Test that None values are handled appropriately."""
        raw_data = [
            {
                "sample_id": "test_none",
                "summary": "Test",
                "source": "Context",
                "annotation_spans": None,  # Explicit None
                "annotations": None,  # Explicit None
            }
        ]

        items = loader._convert_to_pydantic(raw_data, "faithbench")

        assert len(items) == 1
        assert items[0].annotation_spans == []  # Should default to empty list
        assert items[0].annotation_label is None  # Should remain None

    def test_numeric_ids_converted_to_string(self, loader):
        """Test that numeric IDs are handled."""
        raw_data = [
            {
                "question": "Test",
                "best_answer": "Answer",
                "id": 12345,  # Numeric ID
            }
        ]

        # This should fail because ID must be a valid UUID
        with pytest.raises(DatasetValidationError):
            loader._convert_to_pydantic(raw_data, "simpleqa")

    def test_special_characters_in_text(self, loader):
        """Test handling of special characters in text fields."""
        raw_data = [
            {
                "question": "What is 'this' & \"that\"?",
                "best_answer": "It's something with <special> characters & symbols!",
                "id": str(uuid.uuid4()),
            }
        ]

        items = loader._convert_to_pydantic(raw_data, "simpleqa")

        assert len(items) == 1
        assert items[0].question == "What is 'this' & \"that\"?"
        assert items[0].best_answer == "It's something with <special> characters & symbols!"

    def test_unicode_characters_preserved(self, loader):
        """Test that unicode characters are preserved."""
        raw_data = [
            {
                "question": "什么是人工智能？",  # Chinese
                "best_answer": "AI est l'intelligence artificielle 🤖",  # French + emoji
                "id": str(uuid.uuid4()),
            }
        ]

        items = loader._convert_to_pydantic(raw_data, "simpleqa")

        assert len(items) == 1
        assert items[0].question == "什么是人工智能？"
        assert items[0].best_answer == "AI est l'intelligence artificielle 🤖"

    def test_large_text_fields(self, loader):
        """Test handling of very large text fields."""
        large_text = "A" * 10000  # 10,000 character string
        raw_data = [
            {
                "sample_id": "test_large",
                "summary": large_text,
                "source": large_text,
            }
        ]

        items = loader._convert_to_pydantic(raw_data, "faithbench")

        assert len(items) == 1
        assert len(items[0].summary) == 10000
        assert len(items[0].source) == 10000
