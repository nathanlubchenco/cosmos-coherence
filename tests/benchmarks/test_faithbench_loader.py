"""Tests for FaithBench dataset loader functionality."""

import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from cosmos_coherence.benchmarks.models.datasets import (
    FaithBenchAnnotation,
    FaithBenchItem,
)
from cosmos_coherence.harness.huggingface_loader import (
    HuggingFaceDatasetLoader,
)
from pydantic import ValidationError


class TestFaithBenchDatasetLoader:
    """Test suite for FaithBench dataset loading functionality."""

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
    def sample_faithbench_data(self) -> List[Dict[str, Any]]:
        """Create sample FaithBench data matching repository format."""
        return [
            {
                "sample_id": "fb_001",
                "source": (
                    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
                    "in Paris, France. It is named after the engineer Gustave Eiffel, whose "
                    "company designed and built the tower. Constructed from 1887 to 1889, it "
                    "was initially criticized by some of France's leading artists and "
                    "intellectuals for its design."
                ),
                "summary": (
                    "The Eiffel Tower is a steel structure in Paris built by Gustave Eiffel "
                    "in 1885."
                ),
                "annotations": [
                    {
                        "label": "hallucinated",
                        "spans": ["steel structure", "1885"],
                        "justification": (
                            "The tower is wrought-iron, not steel, and was built from "
                            "1887-1889, not 1885"
                        ),
                    }
                ],
                "metadata": {
                    "summarizer": "gpt-4o",
                    "detector_predictions": {
                        "gpt-4-turbo": 1,
                        "gpt-3.5-turbo": 0,
                        "claude-3": 1,
                    },
                    "entropy_score": 0.67,
                },
            },
            {
                "sample_id": "fb_002",
                "source": (
                    "Machine learning is a subset of artificial intelligence that enables "
                    "systems to learn and improve from experience without being explicitly "
                    "programmed. It focuses on developing computer programs that can access "
                    "data and use it to learn for themselves."
                ),
                "summary": (
                    "Machine learning is a subset of AI that allows systems to learn from "
                    "experience without explicit programming."
                ),
                "annotations": [
                    {
                        "label": "consistent",
                        "spans": [],
                        "justification": (
                            "The summary accurately captures the main points of the " "source text"
                        ),
                    }
                ],
                "metadata": {
                    "summarizer": "gpt-4-turbo",
                    "detector_predictions": {
                        "gpt-4-turbo": 0,
                        "gpt-3.5-turbo": 0,
                        "claude-3": 0,
                    },
                    "entropy_score": 0.0,
                },
            },
            {
                "sample_id": "fb_003",
                "source": (
                    "The Great Wall of China is a series of fortifications that were built "
                    "across the historical northern borders of ancient Chinese states and "
                    "Imperial China as protection against various nomadic groups."
                ),
                "summary": (
                    "The Great Wall of China is a fortification built to protect China from "
                    "invaders, though some sections were also used for trade."
                ),
                "annotations": [
                    {
                        "label": "questionable",
                        "spans": ["used for trade"],
                        "justification": (
                            "While trade may have occurred along the wall, the source "
                            "doesn't mention this"
                        ),
                    }
                ],
                "metadata": {
                    "summarizer": "gpt-3.5-turbo",
                    "detector_predictions": {
                        "gpt-4-turbo": 1,
                        "gpt-3.5-turbo": 0,
                        "claude-3": 0,
                    },
                    "entropy_score": 0.45,
                },
            },
            {
                "sample_id": "fb_004",
                "source": (
                    "Photosynthesis is the process by which green plants and some other "
                    "organisms use sunlight to synthesize foods from carbon dioxide and "
                    "water. Photosynthesis in plants generally involves the green pigment "
                    "chlorophyll."
                ),
                "summary": (
                    "Photosynthesis is how green plants make food using sunlight, CO2, and "
                    "H2O with the help of chlorophyll pigments."
                ),
                "annotations": [
                    {
                        "label": "benign",
                        "spans": ["pigments"],
                        "justification": (
                            "Used plural 'pigments' instead of singular 'pigment', but "
                            "this is a minor inaccuracy"
                        ),
                    }
                ],
                "metadata": {
                    "summarizer": "o1-mini",
                    "detector_predictions": {
                        "gpt-4-turbo": 0,
                        "gpt-3.5-turbo": 0,
                        "claude-3": 1,
                    },
                    "entropy_score": 0.32,
                },
            },
        ]

    def test_faithbench_item_creation(self):
        """Test creating a FaithBenchItem with all fields."""
        item = FaithBenchItem(
            sample_id="test_001",
            source="This is the source text that needs to be summarized.",
            summary="This is a summary of the source.",
            annotation_label=FaithBenchAnnotation.CONSISTENT,
            annotation_spans=[],
            annotation_justification="Summary is accurate",
            detector_predictions={"model1": 0, "model2": 0},
            entropy_score=0.0,
            summarizer_model="gpt-4o",
            question="Summarize the text",
        )

        assert item.sample_id == "test_001"
        assert item.source == "This is the source text that needs to be summarized."
        assert item.summary == "This is a summary of the source."
        assert item.annotation_label == FaithBenchAnnotation.CONSISTENT
        assert item.entropy_score == 0.0

    def test_faithbench_item_validation(self):
        """Test FaithBenchItem validation rules."""
        # Test empty source
        with pytest.raises(ValidationError) as exc:
            FaithBenchItem(sample_id="test", source="", summary="Summary", question="Test")
        assert "source" in str(exc.value).lower()

        # Test empty summary
        with pytest.raises(ValidationError) as exc:
            FaithBenchItem(sample_id="test", source="Source text", summary="", question="Test")
        assert "summary" in str(exc.value).lower()

        # Test invalid annotation label
        with pytest.raises(ValidationError) as exc:
            FaithBenchItem(
                sample_id="test",
                source="Source",
                summary="Summary",
                annotation_label="invalid_label",
                question="Test",
            )
        # Check that the error is related to annotation_label field
        assert "annotation_label" in str(exc.value).lower() or "enum" in str(exc.value).lower()

        # Test invalid entropy score
        with pytest.raises(ValidationError) as exc:
            FaithBenchItem(
                sample_id="test",
                source="Source",
                summary="Summary",
                entropy_score=1.5,  # Should be between 0 and 1
                question="Test",
            )
        assert "entropy" in str(exc.value).lower()

    def test_faithbench_annotation_enum(self):
        """Test FaithBenchAnnotation enum values."""
        assert FaithBenchAnnotation.CONSISTENT.value == "consistent"
        assert FaithBenchAnnotation.QUESTIONABLE.value == "questionable"
        assert FaithBenchAnnotation.BENIGN.value == "benign"
        assert FaithBenchAnnotation.HALLUCINATED.value == "hallucinated"

        # Test all enum values are accessible
        all_values = [e.value for e in FaithBenchAnnotation]
        assert len(all_values) == 4
        assert "consistent" in all_values
        assert "questionable" in all_values
        assert "benign" in all_values
        assert "hallucinated" in all_values

    def test_convert_faithbench_item(self, loader, sample_faithbench_data):
        """Test converting raw FaithBench data to FaithBenchItem."""
        raw_item = sample_faithbench_data[0]
        item = loader._convert_faithbench_item(raw_item)

        assert isinstance(item, FaithBenchItem)
        assert item.sample_id == "fb_001"
        assert "Eiffel Tower" in item.source
        assert "steel structure" in item.summary
        assert item.annotation_label == "hallucinated"
        assert len(item.annotation_spans) == 2
        assert "steel structure" in item.annotation_spans
        assert "1885" in item.annotation_spans
        assert item.entropy_score == 0.67
        assert item.summarizer_model == "gpt-4o"

    def test_convert_faithbench_all_labels(self, loader, sample_faithbench_data):
        """Test converting FaithBench items with all 4 annotation labels."""
        items = [loader._convert_faithbench_item(data) for data in sample_faithbench_data]

        # Check we have all 4 label types
        labels = [item.annotation_label for item in items]
        assert "hallucinated" in labels
        assert "consistent" in labels
        assert "questionable" in labels
        assert "benign" in labels

    def test_convert_faithbench_missing_fields(self, loader):
        """Test converting FaithBench item with missing optional fields."""
        minimal_item = {"sample_id": "test", "source": "Source text", "summary": "Summary text"}

        item = loader._convert_faithbench_item(minimal_item)
        assert item.sample_id == "test"
        assert item.source == "Source text"
        assert item.summary == "Summary text"
        assert item.annotation_label is None
        assert item.annotation_spans == []
        assert item.entropy_score is None

    @pytest.mark.asyncio
    async def test_load_faithbench_dataset(self, loader, sample_faithbench_data):
        """Test loading FaithBench dataset."""
        with patch.object(loader, "_load_from_cache", return_value=sample_faithbench_data):
            items = await loader.load_dataset("faithbench")

            assert len(items) == 4
            assert all(isinstance(item, FaithBenchItem) for item in items)

            # Check specific items
            assert items[0].sample_id == "fb_001"
            assert items[0].annotation_label == "hallucinated"
            assert items[1].annotation_label == "consistent"
            assert items[2].annotation_label == "questionable"
            assert items[3].annotation_label == "benign"

    @pytest.mark.asyncio
    async def test_load_faithbench_with_sampling(self, loader, sample_faithbench_data):
        """Test loading FaithBench dataset with sampling."""
        with patch.object(loader, "_load_from_cache", return_value=sample_faithbench_data):
            # Test sample_size=2
            items = await loader.load_dataset("faithbench", sample_size=2)
            assert len(items) == 2
            assert items[0].sample_id == "fb_001"
            assert items[1].sample_id == "fb_002"

            # Test sample_size larger than dataset
            items = await loader.load_dataset("faithbench", sample_size=10)
            assert len(items) == 4  # Should return all available

    def test_faithbench_cache_path(self, loader):
        """Test FaithBench cache path generation."""
        path = loader._get_cache_path("faithbench", "test")
        assert path.name == "faithbench_test.json"

        path_default = loader._get_cache_path("faithbench")
        assert path_default.name == "faithbench_default.json"

    @pytest.mark.asyncio
    async def test_faithbench_cache_operations(
        self, loader, temp_cache_dir, sample_faithbench_data
    ):
        """Test FaithBench caching functionality."""
        cache_path = temp_cache_dir / "faithbench_test.json"

        # Save to cache
        loader._save_to_cache(sample_faithbench_data, cache_path)
        assert cache_path.exists()

        # Load from cache
        cached_data = loader._load_from_cache(cache_path)
        assert len(cached_data) == 4
        assert cached_data[0]["sample_id"] == "fb_001"

        # Clear cache
        loader.clear_cache("faithbench")
        assert not cache_path.exists()

    def test_faithbench_entropy_validation(self):
        """Test entropy score validation for challenging sample detection."""
        # Valid entropy scores
        item1 = FaithBenchItem(
            sample_id="test",
            source="Source",
            summary="Summary",
            entropy_score=0.0,  # No disagreement
            question="Test",
        )
        assert item1.entropy_score == 0.0

        item2 = FaithBenchItem(
            sample_id="test",
            source="Source",
            summary="Summary",
            entropy_score=1.0,  # Maximum disagreement
            question="Test",
        )
        assert item2.entropy_score == 1.0

        item3 = FaithBenchItem(
            sample_id="test",
            source="Source",
            summary="Summary",
            entropy_score=0.67,  # Moderate disagreement (challenging)
            question="Test",
        )
        assert item3.entropy_score == 0.67

    def test_faithbench_detector_predictions(self, sample_faithbench_data):
        """Test handling of detector predictions metadata."""
        loader = HuggingFaceDatasetLoader()
        item = loader._convert_faithbench_item(sample_faithbench_data[0])

        assert isinstance(item.detector_predictions, dict)
        assert "gpt-4-turbo" in item.detector_predictions
        assert item.detector_predictions["gpt-4-turbo"] == 1
        assert item.detector_predictions["gpt-3.5-turbo"] == 0

    @pytest.mark.asyncio
    async def test_faithbench_integration_with_base_system(self, loader):
        """Test FaithBench integration with base dataset system."""
        # Create mock data that tests inheritance from BaseDatasetItem
        mock_data = [
            {
                "sample_id": "test",
                "source": "Test source",
                "summary": "Test summary",
                "id": str(uuid.uuid4()),  # BaseDatasetItem field
            }
        ]

        with patch.object(loader, "_load_from_cache", return_value=mock_data):
            items = await loader.load_dataset("faithbench")
            assert len(items) == 1
            assert hasattr(items[0], "id")  # From BaseDatasetItem
            assert hasattr(items[0], "created_at")  # From BaseDatasetItem
            assert hasattr(items[0], "sample_id")  # From FaithBenchItem

    def test_faithbench_summarization_focus(self, sample_faithbench_data):
        """Test that FaithBench correctly handles summarization task."""
        loader = HuggingFaceDatasetLoader()

        for raw_item in sample_faithbench_data:
            item = loader._convert_faithbench_item(raw_item)
            # Verify it's treated as summarization, not Q&A
            assert item.source  # Original text to summarize
            assert item.summary  # Generated summary
            # The 'question' field should be auto-generated for BaseDatasetItem compatibility
            assert item.question  # Should be populated
            assert "..." in item.question or "Summarize" in item.question
