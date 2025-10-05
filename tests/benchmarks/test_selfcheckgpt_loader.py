"""Tests for SelfCheckGPT dataset loading functionality."""

import tempfile
from pathlib import Path

import pytest
from cosmos_coherence.benchmarks.models.base import DatasetValidationError
from cosmos_coherence.benchmarks.models.datasets import SelfCheckGPTItem
from cosmos_coherence.harness.huggingface_loader import (
    HuggingFaceDatasetLoader,
)
from pydantic import ValidationError


class TestSelfCheckGPTDatasetLoader:
    """Test suite for SelfCheckGPT dataset loading."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def loader(self, temp_cache_dir):
        """Create a loader instance with temporary cache."""
        return HuggingFaceDatasetLoader(cache_dir=temp_cache_dir)

    @pytest.fixture
    def sample_selfcheckgpt_data(self):
        """Sample SelfCheckGPT dataset in HuggingFace format.

        Based on potsawee/wiki_bio_gpt3_hallucination structure.
        """
        return [
            {
                "wiki_bio_text": (
                    "Albert Einstein (1879-1955) was a German-born physicist. "
                    "He developed the theory of relativity."
                ),
                "gpt3_text": (
                    "Albert Einstein was a physicist. " "He won the Nobel Prize in Physics."
                ),
                "gpt3_sentences": [
                    "Albert Einstein was a physicist.",
                    "He won the Nobel Prize in Physics.",
                ],
                "annotation": [
                    "accurate",  # Sentence 1
                    "accurate",  # Sentence 2
                ],
            },
            {
                "wiki_bio_text": (
                    "Marie Curie (1867-1934) was a Polish physicist. " "She discovered radium."
                ),
                "gpt3_text": ("Marie Curie was a chemist. " "She won two Nobel Prizes."),
                "gpt3_sentences": [
                    "Marie Curie was a chemist.",
                    "She won two Nobel Prizes.",
                ],
                "annotation": [
                    "minor_inaccurate",  # Sentence 1 (physicist, not chemist)
                    "accurate",  # Sentence 2
                ],
            },
            {
                "wiki_bio_text": (
                    "Isaac Newton (1643-1727) was an English mathematician. "
                    "He formulated laws of motion."
                ),
                "gpt3_text": ("Isaac Newton invented calculus. " "He was born in London."),
                "gpt3_sentences": [
                    "Isaac Newton invented calculus.",
                    "He was born in London.",
                ],
                "annotation": [
                    "accurate",  # Sentence 1
                    "major_inaccurate",  # Sentence 2 (born in Lincolnshire)
                ],
            },
        ]

    def test_loader_recognizes_selfcheckgpt(self, loader):
        """Test that loader recognizes SelfCheckGPT dataset name."""
        assert "selfcheckgpt" in loader.dataset_mapping
        assert loader.dataset_mapping["selfcheckgpt"] == "potsawee/wiki_bio_gpt3_hallucination"

    def test_load_selfcheckgpt_dataset_structure(self, loader, sample_selfcheckgpt_data):
        """Test loading SelfCheckGPT dataset with correct structure."""
        # Convert raw data to Pydantic models
        items = loader._convert_to_pydantic(sample_selfcheckgpt_data, "selfcheckgpt")

        # Verify we got SelfCheckGPTItem instances
        assert len(items) == 3
        assert all(isinstance(item, SelfCheckGPTItem) for item in items)

        # Verify first item structure
        assert items[0].topic == "Albert Einstein"
        assert "physicist" in items[0].wiki_bio_text
        assert "physicist" in items[0].gpt3_text
        assert len(items[0].gpt3_sentences) == 2
        assert len(items[0].annotation) == 2

    def test_load_with_sample_size(self, loader, sample_selfcheckgpt_data):
        """Test loading dataset with sample_size parameter."""
        # Convert all data first, then slice
        items = loader._convert_to_pydantic(sample_selfcheckgpt_data[:2], "selfcheckgpt")

        assert len(items) == 2
        assert items[0].topic == "Albert Einstein"
        assert items[1].topic == "Marie Curie"

    def test_annotation_labels_mapping(self, loader, sample_selfcheckgpt_data):
        """Test that annotation labels are properly mapped."""
        items = loader._convert_to_pydantic(sample_selfcheckgpt_data, "selfcheckgpt")

        # Check annotation labels are preserved
        assert items[0].annotation == ["accurate", "accurate"]
        assert items[1].annotation == [
            "minor_inaccurate",
            "accurate",
        ]
        assert items[2].annotation == ["accurate", "major_inaccurate"]

    def test_sentence_count_validation(self, loader, sample_selfcheckgpt_data):
        """Test validation of sentence count vs annotation count."""
        # Add invalid data (mismatched counts)
        invalid_data = sample_selfcheckgpt_data.copy()
        invalid_data.append(
            {
                "topic": "Test",
                "wiki_bio_text": "Test bio",
                "gpt3_text": "Test text",
                "gpt3_sentences": ["Sentence 1", "Sentence 2"],
                "annotation": ["accurate"],  # Only 1 annotation for 2 sentences
            }
        )

        # Should raise validation error
        with pytest.raises(DatasetValidationError):
            loader._convert_to_pydantic(invalid_data, "selfcheckgpt")


class TestSelfCheckGPTItem:
    """Test suite for SelfCheckGPTItem data model."""

    def test_create_valid_item(self):
        """Test creating a valid SelfCheckGPTItem."""
        item = SelfCheckGPTItem(
            question="Albert Einstein",  # topic becomes question
            topic="Albert Einstein",
            wiki_bio_text="Albert Einstein was a physicist.",
            gpt3_text="Albert Einstein invented relativity.",
            gpt3_sentences=["Albert Einstein invented relativity."],
            annotation=["accurate"],
        )

        assert item.topic == "Albert Einstein"
        assert "physicist" in item.wiki_bio_text
        assert len(item.gpt3_sentences) == 1
        assert len(item.annotation) == 1

    def test_validate_topic_required(self):
        """Test that topic is required."""
        with pytest.raises(ValueError):
            SelfCheckGPTItem(
                question="",
                topic="",
                wiki_bio_text="Some text",
                gpt3_text="Some text",
                gpt3_sentences=["Some text"],
                annotation=["accurate"],
            )

    def test_validate_annotation_count_matches_sentences(self):
        """Test that annotation count must match sentence count."""
        # Pydantic ValidationError is raised during model initialization
        with pytest.raises(ValidationError):
            SelfCheckGPTItem(
                question="Test",
                topic="Test",
                wiki_bio_text="Test",
                gpt3_text="Test",
                gpt3_sentences=["Sentence 1", "Sentence 2"],
                annotation=["accurate"],  # Only 1 for 2 sentences
            )

    def test_validate_content_method(self):
        """Test the validate_content method catches all issues."""
        # Empty topic - field validator catches this
        with pytest.raises(ValueError):
            SelfCheckGPTItem(
                question="",
                topic="",
                wiki_bio_text="Test",
                gpt3_text="Test",
                gpt3_sentences=["Test"],
                annotation=["accurate"],
            )

        # Empty wiki_bio_text - field validator catches this
        with pytest.raises(ValueError):
            SelfCheckGPTItem(
                question="Topic",
                topic="Topic",
                wiki_bio_text="",
                gpt3_text="Test",
                gpt3_sentences=["Test"],
                annotation=["accurate"],
            )

    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        item = SelfCheckGPTItem(
            question="Test Topic",
            topic="Test Topic",
            wiki_bio_text="Bio text",
            gpt3_text="GPT-3 text",
            gpt3_sentences=["GPT-3 text"],
            annotation=["accurate"],
        )

        data = item.to_dict()

        assert data["topic"] == "Test Topic"
        assert data["wiki_bio_text"] == "Bio text"
        assert data["gpt3_text"] == "GPT-3 text"
        assert isinstance(data["id"], str)  # UUID serialized as string
        assert isinstance(data["created_at"], str)  # datetime as ISO string

    def test_from_dict_deserialization(self):
        """Test deserialization from dictionary."""
        data = {
            "question": "Test",
            "topic": "Test",
            "wiki_bio_text": "Bio",
            "gpt3_text": "GPT-3",
            "gpt3_sentences": ["GPT-3"],
            "annotation": ["accurate"],
        }

        item = SelfCheckGPTItem.from_dict(data)

        assert item.topic == "Test"
        assert item.wiki_bio_text == "Bio"
        assert len(item.gpt3_sentences) == 1
