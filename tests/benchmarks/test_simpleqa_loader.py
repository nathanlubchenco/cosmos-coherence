"""Tests for SimpleQA dataset loading functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import pytest
from cosmos_coherence.benchmarks.models.base import DatasetValidationError
from cosmos_coherence.benchmarks.models.datasets import (
    SimpleQACategory,
    SimpleQADifficulty,
    SimpleQAItem,
)
from cosmos_coherence.harness.huggingface_loader import (
    DatasetNotFoundError,
    HuggingFaceDatasetLoader,
)


class TestSimpleQADatasetLoader:
    """Test suite for SimpleQA dataset loading."""

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
    def sample_simpleqa_data(self):
        """Sample SimpleQA dataset in HuggingFace format."""
        return [
            {
                "question": "What is the capital of France?",
                "best_answer": "Paris",
                "id": "550e8400-e29b-41d4-a716-446655440001",
            },
            {
                "question": "Who wrote '1984'?",
                "best_answer": "George Orwell",
                "id": "550e8400-e29b-41d4-a716-446655440002",
            },
            {
                "question": "What is the speed of light in vacuum?",
                "best_answer": "299,792,458 meters per second",
                "id": "550e8400-e29b-41d4-a716-446655440003",
            },
        ]

    @pytest.fixture
    def sample_simpleqa_with_metadata(self):
        """Sample SimpleQA data with optional fields."""
        return [
            {
                "question": "What is HTTP?",
                "best_answer": "Hypertext Transfer Protocol",
                "category": "technology",
                "difficulty": "easy",
                "sources": ["https://www.w3.org/Protocols/"],
                "grading_notes": "Accept 'HyperText Transfer Protocol' as correct",
                "id": "550e8400-e29b-41d4-a716-446655440004",
            },
            {
                "question": "Who painted the Mona Lisa?",
                "best_answer": "Leonardo da Vinci",
                "category": "literature",  # Will map to LITERATURE
                "difficulty": "medium",
                "sources": ["https://en.wikipedia.org/wiki/Mona_Lisa"],
                "id": "550e8400-e29b-41d4-a716-446655440005",
            },
        ]

    def test_loader_recognizes_simpleqa(self, loader):
        """Test that loader recognizes SimpleQA dataset."""
        assert "simpleqa" in loader.dataset_mapping
        assert loader.dataset_mapping["simpleqa"] == "basicv8vc/SimpleQA"

    def test_convert_simpleqa_basic(self, loader, sample_simpleqa_data):
        """Test converting basic SimpleQA data to Pydantic models."""
        items = loader._convert_to_pydantic(sample_simpleqa_data, "simpleqa")

        assert len(items) == 3
        assert all(isinstance(item, SimpleQAItem) for item in items)

        # Check first item
        item1 = items[0]
        assert item1.question == "What is the capital of France?"
        assert item1.best_answer == "Paris"
        assert isinstance(item1.id, UUID)

        # Check second item
        item2 = items[1]
        assert item2.question == "Who wrote '1984'?"
        assert item2.best_answer == "George Orwell"

        # Check third item
        item3 = items[2]
        assert item3.question == "What is the speed of light in vacuum?"
        assert item3.best_answer == "299,792,458 meters per second"

    def test_convert_simpleqa_with_metadata(self, loader, sample_simpleqa_with_metadata):
        """Test converting SimpleQA data with optional fields."""
        items = loader._convert_to_pydantic(sample_simpleqa_with_metadata, "simpleqa")

        assert len(items) == 2

        # Check first item with full metadata
        item1 = items[0]
        assert item1.question == "What is HTTP?"
        assert item1.best_answer == "Hypertext Transfer Protocol"
        assert item1.category == SimpleQACategory.TECHNOLOGY
        assert item1.difficulty == SimpleQADifficulty.EASY
        assert item1.sources == ["https://www.w3.org/Protocols/"]
        assert item1.grading_notes == "Accept 'HyperText Transfer Protocol' as correct"

        # Check second item
        item2 = items[1]
        assert item2.question == "Who painted the Mona Lisa?"
        assert item2.best_answer == "Leonardo da Vinci"
        assert item2.category == SimpleQACategory.LITERATURE
        assert item2.difficulty == SimpleQADifficulty.MEDIUM

    @pytest.mark.asyncio
    async def test_load_simpleqa_from_cache(self, loader, temp_cache_dir, sample_simpleqa_data):
        """Test loading SimpleQA from cache."""
        # Create cache file
        cache_path = temp_cache_dir / "simpleqa_default.json"
        cache_path.write_text(json.dumps(sample_simpleqa_data))

        # Load from cache
        items = await loader.load_dataset("simpleqa")

        assert len(items) == 3
        assert all(isinstance(item, SimpleQAItem) for item in items)
        assert items[0].question == "What is the capital of France?"

    @pytest.mark.asyncio
    async def test_load_simpleqa_from_huggingface(self, loader, sample_simpleqa_data):
        """Test loading SimpleQA from HuggingFace when not cached."""
        with patch.object(loader, "_load_from_huggingface") as mock_load:
            mock_load.return_value = sample_simpleqa_data

            items = await loader.load_dataset("simpleqa")

            mock_load.assert_called_once_with("basicv8vc/SimpleQA", None)
            assert len(items) == 3
            assert all(isinstance(item, SimpleQAItem) for item in items)

    @pytest.mark.asyncio
    async def test_load_simpleqa_with_sample_size(self, loader):
        """Test loading SimpleQA with sample_size parameter."""
        # Create larger dataset
        large_data = [
            {
                "question": f"Question {i}?",
                "best_answer": f"Answer {i}",
                "id": f"550e8400-e29b-41d4-a716-44665544{i:04d}",
            }
            for i in range(100)
        ]

        with patch.object(loader, "_load_from_huggingface") as mock_load:
            mock_load.return_value = large_data

            # Load with sample size
            items = await loader.load_dataset("simpleqa", sample_size=10)

            assert len(items) == 10
            assert all(isinstance(item, SimpleQAItem) for item in items)
            # Check that we got the first 10 items
            assert items[0].question == "Question 0?"
            assert items[9].question == "Question 9?"

    @pytest.mark.asyncio
    async def test_load_simpleqa_invalid_dataset_name(self, loader):
        """Test loading with invalid dataset name."""
        with pytest.raises(DatasetNotFoundError) as exc_info:
            await loader.load_dataset("invalid_dataset")
        assert "Unknown dataset" in str(exc_info.value)

    def test_simpleqa_validation_error(self, loader):
        """Test handling of invalid SimpleQA data."""
        invalid_data = [
            {
                "question": "",  # Empty question
                "best_answer": "Paris",
                "id": "550e8400-e29b-41d4-a716-446655440001",
            }
        ]

        with pytest.raises(DatasetValidationError) as exc_info:
            loader._convert_to_pydantic(invalid_data, "simpleqa")
        assert "question" in str(exc_info.value).lower()

    def test_simpleqa_missing_required_field(self, loader):
        """Test handling of missing required fields."""
        invalid_data = [
            {
                "question": "What is the capital of France?",
                # Missing best_answer
                "id": "550e8400-e29b-41d4-a716-446655440001",
            }
        ]

        with pytest.raises(DatasetValidationError) as exc_info:
            loader._convert_to_pydantic(invalid_data, "simpleqa")
        assert "best_answer" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_simpleqa_cache_persistence(self, loader, temp_cache_dir, sample_simpleqa_data):
        """Test that data is cached after loading from HuggingFace."""
        cache_path = temp_cache_dir / "simpleqa_default.json"

        # Ensure cache doesn't exist initially
        assert not cache_path.exists()

        with patch.object(loader, "_load_from_huggingface") as mock_load:
            mock_load.return_value = sample_simpleqa_data

            # First load - should call HuggingFace
            items1 = await loader.load_dataset("simpleqa")
            assert mock_load.call_count == 1

            # Cache should now exist
            assert cache_path.exists()

            # Second load - should use cache
            items2 = await loader.load_dataset("simpleqa")
            assert mock_load.call_count == 1  # Still only called once

            # Data should be the same
            assert len(items1) == len(items2)
            assert items1[0].question == items2[0].question

    def test_simpleqa_category_mapping(self, loader):
        """Test that category strings are correctly mapped to enums."""
        import uuid

        data_with_categories = [
            {"question": "Q1", "best_answer": "A1", "category": "science", "id": str(uuid.uuid4())},
            {
                "question": "Q2",
                "best_answer": "A2",
                "category": "technology",
                "id": str(uuid.uuid4()),
            },
            {"question": "Q3", "best_answer": "A3", "category": "history", "id": str(uuid.uuid4())},
            {
                "question": "Q4",
                "best_answer": "A4",
                "category": "geography",
                "id": str(uuid.uuid4()),
            },
            {"question": "Q5", "best_answer": "A5", "category": "sports", "id": str(uuid.uuid4())},
            {
                "question": "Q6",
                "best_answer": "A6",
                "category": "tv_shows",
                "id": str(uuid.uuid4()),
            },
            {
                "question": "Q7",
                "best_answer": "A7",
                "category": "video_games",
                "id": str(uuid.uuid4()),
            },
            {"question": "Q8", "best_answer": "A8", "category": "music", "id": str(uuid.uuid4())},
            {
                "question": "Q9",
                "best_answer": "A9",
                "category": "literature",
                "id": str(uuid.uuid4()),
            },
            {
                "question": "Q10",
                "best_answer": "A10",
                "category": "general",
                "id": str(uuid.uuid4()),
            },
        ]

        items = loader._convert_to_pydantic(data_with_categories, "simpleqa")

        assert items[0].category == SimpleQACategory.SCIENCE
        assert items[1].category == SimpleQACategory.TECHNOLOGY
        assert items[2].category == SimpleQACategory.HISTORY
        assert items[3].category == SimpleQACategory.GEOGRAPHY
        assert items[4].category == SimpleQACategory.SPORTS
        assert items[5].category == SimpleQACategory.TV_SHOWS
        assert items[6].category == SimpleQACategory.VIDEO_GAMES
        assert items[7].category == SimpleQACategory.MUSIC
        assert items[8].category == SimpleQACategory.LITERATURE
        assert items[9].category == SimpleQACategory.GENERAL

    def test_simpleqa_difficulty_mapping(self, loader):
        """Test that difficulty strings are correctly mapped to enums."""
        import uuid

        data_with_difficulty = [
            {"question": "Q1", "best_answer": "A1", "difficulty": "easy", "id": str(uuid.uuid4())},
            {
                "question": "Q2",
                "best_answer": "A2",
                "difficulty": "medium",
                "id": str(uuid.uuid4()),
            },
            {"question": "Q3", "best_answer": "A3", "difficulty": "hard", "id": str(uuid.uuid4())},
        ]

        items = loader._convert_to_pydantic(data_with_difficulty, "simpleqa")

        assert items[0].difficulty == SimpleQADifficulty.EASY
        assert items[1].difficulty == SimpleQADifficulty.MEDIUM
        assert items[2].difficulty == SimpleQADifficulty.HARD

    def test_simpleqa_sources_handling(self, loader):
        """Test handling of sources field."""
        import uuid

        data_with_sources = [
            {
                "question": "Q1",
                "best_answer": "A1",
                "sources": ["https://example.com", "https://test.org"],
                "id": str(uuid.uuid4()),
            },
            {
                "question": "Q2",
                "best_answer": "A2",
                "sources": [],  # Empty sources
                "id": str(uuid.uuid4()),
            },
            {
                "question": "Q3",
                "best_answer": "A3",
                # No sources field
                "id": str(uuid.uuid4()),
            },
        ]

        items = loader._convert_to_pydantic(data_with_sources, "simpleqa")

        assert items[0].sources == ["https://example.com", "https://test.org"]
        assert items[1].sources is None  # Empty list becomes None
        assert items[2].sources is None  # Missing field is None

    @pytest.mark.asyncio
    async def test_load_simpleqa_without_cache(self, loader):
        """Test loading dataset when caching is disabled."""
        # This would be implemented when no-cache option is added
        pass

    def test_simpleqa_content_validation(self, loader, sample_simpleqa_data):
        """Test that validate_content is called on items."""
        items = loader._convert_to_pydantic(sample_simpleqa_data, "simpleqa")

        # validate_content should not raise any exceptions
        for item in items:
            item.validate_content()
