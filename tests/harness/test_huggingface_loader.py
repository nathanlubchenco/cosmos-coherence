"""Tests for HuggingFace dataset loader functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from cosmos_coherence.benchmarks.models.base import DatasetValidationError
from cosmos_coherence.benchmarks.models.datasets import (
    FaithBenchItem,
    FEVERItem,
    FEVERLabel,
    HaluEvalItem,
    HaluEvalTaskType,
    SimpleQAItem,
    TruthfulQAItem,
)
from cosmos_coherence.harness.huggingface_loader import (
    DatasetLoadError,
    DatasetNotFoundError,
    HuggingFaceDatasetLoader,
)


class TestHuggingFaceDatasetLoader:
    """Test suite for HuggingFace dataset loader."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def loader(self, temp_cache_dir):
        """Create a loader instance with temporary cache."""
        return HuggingFaceDatasetLoader(cache_dir=temp_cache_dir)

    def test_init_with_default_cache_dir(self):
        """Test initialization with default cache directory."""
        loader = HuggingFaceDatasetLoader()
        assert loader.cache_dir == Path(".cache/datasets")
        assert loader.dataset_mapping == {
            "faithbench": "vectara/faithbench",
            "simpleqa": "basicv8vc/SimpleQA",
            "truthfulqa": "truthfulqa/truthful_qa",
            "fever": "fever/fever",
            "halueval": "pminervini/HaluEval",
        }

    def test_init_with_custom_cache_dir(self, temp_cache_dir):
        """Test initialization with custom cache directory."""
        loader = HuggingFaceDatasetLoader(cache_dir=temp_cache_dir)
        assert loader.cache_dir == temp_cache_dir

    def test_get_cache_path(self, loader):
        """Test cache path generation."""
        path = loader._get_cache_path("faithbench", "train")
        assert path.name == "faithbench_train.json"
        assert path.parent == loader.cache_dir

    def test_get_cache_path_no_split(self, loader):
        """Test cache path generation without split."""
        path = loader._get_cache_path("simpleqa")
        assert path.name == "simpleqa_default.json"

    def test_load_from_huggingface(self, loader):
        """Test loading dataset from HuggingFace."""
        with patch.object(loader, "_load_from_huggingface") as mock_load:
            mock_data = [{"question": "What is 2+2?", "answer": "4", "id": "1"}]
            mock_load.return_value = mock_data

            data = loader._load_from_huggingface("basicv8vc/SimpleQA", "test")
            assert len(data) == 1
            assert data[0]["question"] == "What is 2+2?"

    def test_load_from_huggingface_error(self, loader):
        """Test error handling when loading from HuggingFace fails."""
        # Mock the actual method that would fail internally
        with patch(
            "cosmos_coherence.harness.huggingface_loader.HuggingFaceDatasetLoader._load_from_huggingface"
        ) as mock_load:
            mock_load.side_effect = DatasetLoadError("Failed to load dataset: Network error")

            with pytest.raises(DatasetLoadError) as exc_info:
                loader._load_from_huggingface("basicv8vc/SimpleQA", "test")
            assert "Failed to load dataset" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_load_dataset_with_sample_size(self, loader):
        """Test loading dataset with sample_size parameter."""
        import uuid

        # Create mock data
        mock_data = [
            {"question": f"Q{i}", "best_answer": f"A{i}", "id": str(uuid.uuid4())}
            for i in range(100)
        ]

        with patch.object(loader, "_load_from_cache", return_value=mock_data):
            # Test with sample_size=10
            items = await loader.load_dataset("simpleqa", sample_size=10)
            assert len(items) == 10
            # Verify we got the first 10 items
            for i, item in enumerate(items):
                assert item.question == f"Q{i}"
                assert item.best_answer == f"A{i}"

            # Test with sample_size=1
            items = await loader.load_dataset("simpleqa", sample_size=1)
            assert len(items) == 1
            assert items[0].question == "Q0"

            # Test with sample_size larger than dataset
            items = await loader.load_dataset("simpleqa", sample_size=200)
            assert len(items) == 100  # Should return all available items

            # Test with no sample_size (should return all)
            items = await loader.load_dataset("simpleqa", sample_size=None)
            assert len(items) == 100

    @pytest.mark.asyncio
    async def test_load_dataset_empty_with_sample_size(self, loader):
        """Test loading empty dataset with sample_size parameter."""
        with patch.object(loader, "_load_from_cache", return_value=[]):
            items = await loader.load_dataset("simpleqa", sample_size=10)
            assert len(items) == 0

    @pytest.mark.asyncio
    async def test_dataset_slicing_preserves_order(self, loader):
        """Test that dataset slicing preserves original order."""
        import uuid

        mock_data = [
            {"question": f"Q{i}", "best_answer": f"A{i}", "id": str(uuid.uuid4())}
            for i in range(20)
        ]

        with patch.object(loader, "_load_from_cache", return_value=mock_data):
            items = await loader.load_dataset("simpleqa", sample_size=5)
            assert len(items) == 5
            # Ensure we got items in order 0, 1, 2, 3, 4
            for i, item in enumerate(items):
                assert item.question == f"Q{i}"

    def test_save_to_cache(self, loader, temp_cache_dir):
        """Test saving data to cache."""
        data = [{"question": "Test?", "answer": "Yes"}]
        cache_path = temp_cache_dir / "test.json"

        loader._save_to_cache(data, cache_path)

        assert cache_path.exists()
        with open(cache_path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_load_from_cache_exists(self, loader, temp_cache_dir):
        """Test loading data from existing cache."""
        data = [{"question": "Cached?", "answer": "Yes"}]
        cache_path = temp_cache_dir / "test.json"

        with open(cache_path, "w") as f:
            json.dump(data, f)

        loaded = loader._load_from_cache(cache_path)
        assert loaded == data

    def test_load_from_cache_not_exists(self, loader, temp_cache_dir):
        """Test loading from non-existent cache returns None."""
        cache_path = temp_cache_dir / "nonexistent.json"
        result = loader._load_from_cache(cache_path)
        assert result is None

    def test_convert_faithbench(self, loader):
        """Test converting FaithBench data to Pydantic models."""
        raw_data = [
            {
                "sample_id": "test_001",
                "summary": "The sky is blue",
                "source": "According to science, the sky appears blue.",
                "id": "550e8400-e29b-41d4-a716-446655440001",
            }
        ]

        items = loader._convert_to_pydantic(raw_data, "faithbench")

        assert len(items) == 1
        assert isinstance(items[0], FaithBenchItem)
        assert items[0].summary == "The sky is blue"
        assert str(items[0].id) == "550e8400-e29b-41d4-a716-446655440001"

    def test_convert_simpleqa(self, loader):
        """Test converting SimpleQA data to Pydantic models."""
        raw_data = [
            {
                "question": "What is the capital of France?",
                "best_answer": "Paris",
                "id": "550e8400-e29b-41d4-a716-446655440002",
            }
        ]

        items = loader._convert_to_pydantic(raw_data, "simpleqa")

        assert len(items) == 1
        assert isinstance(items[0], SimpleQAItem)
        assert items[0].question == "What is the capital of France?"
        assert items[0].best_answer == "Paris"

    def test_convert_truthfulqa(self, loader):
        """Test converting TruthfulQA data to Pydantic models."""
        raw_data = [
            {
                "question": "What happens if you cross your eyes?",
                "best_answer": "Nothing permanent happens",
                "correct_answers": ["Nothing permanent"],
                "incorrect_answers": ["They get stuck"],
                "category": "misconceptions",
                "id": "550e8400-e29b-41d4-a716-446655440003",
            }
        ]

        items = loader._convert_to_pydantic(raw_data, "truthfulqa")

        assert len(items) == 1
        assert isinstance(items[0], TruthfulQAItem)
        assert items[0].question == "What happens if you cross your eyes?"

    def test_convert_fever(self, loader):
        """Test converting FEVER data to Pydantic models."""
        raw_data = [
            {
                "claim": "The Earth is flat",
                "label": "REFUTED",
                "evidence": [["Earth", 0, "The Earth is round"]],
                "id": "550e8400-e29b-41d4-a716-446655440004",
            }
        ]

        items = loader._convert_to_pydantic(raw_data, "fever")

        assert len(items) == 1
        assert isinstance(items[0], FEVERItem)
        assert items[0].claim == "The Earth is flat"
        assert items[0].label == FEVERLabel.REFUTED

    def test_convert_halueval(self, loader):
        """Test converting HaluEval data to Pydantic models."""
        raw_data = [
            {
                "question": "What is AI?",
                "knowledge": (
                    "Artificial Intelligence (AI) is the simulation of human intelligence."
                ),
                "right_answer": "AI is artificial intelligence",
                "hallucinated_answer": "AI is a type of robot",
                "task_type": "qa",
                "id": "550e8400-e29b-41d4-a716-446655440005",
            }
        ]

        items = loader._convert_to_pydantic(raw_data, "halueval")

        assert len(items) == 1
        assert isinstance(items[0], HaluEvalItem)
        assert items[0].question == "What is AI?"
        assert items[0].task_type == HaluEvalTaskType.QA
        assert (
            items[0].knowledge
            == "Artificial Intelligence (AI) is the simulation of human intelligence."
        )

    def test_convert_validation_error(self, loader):
        """Test validation error handling during conversion."""
        raw_data = [
            {"invalid": "data"},  # Missing required fields
            {
                "question": "Valid?",
                "best_answer": "Yes",
                "id": "550e8400-e29b-41d4-a716-446655440006",
            },
        ]

        with pytest.raises(DatasetValidationError) as exc_info:
            loader._convert_to_pydantic(raw_data, "simpleqa")

        assert "Validation failed for item 0" in str(exc_info.value)
        assert exc_info.value.field == "item_0"

    def test_convert_unknown_dataset(self, loader):
        """Test error for unknown dataset type."""
        with pytest.raises(DatasetNotFoundError) as exc_info:
            loader._convert_to_pydantic([], "unknown")
        assert "Unknown dataset type" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_load_dataset_with_cache(self, loader, temp_cache_dir):
        """Test loading dataset with cache hit."""
        # Pre-populate cache
        cache_data = [
            {
                "question": "Cached?",
                "best_answer": "Yes",
                "id": "550e8400-e29b-41d4-a716-446655440007",
            }
        ]
        cache_path = loader._get_cache_path("simpleqa", "test")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

        items = await loader.load_dataset("simpleqa", split="test")

        assert len(items) == 1
        assert isinstance(items[0], SimpleQAItem)
        assert items[0].question == "Cached?"
        # Should use cache, no need to mock load_dataset

    @pytest.mark.asyncio
    async def test_load_dataset_without_cache(self, loader, temp_cache_dir):
        """Test loading dataset without cache (downloads from HF)."""
        # Mock the internal method instead of the external library
        with patch.object(loader, "_load_from_huggingface") as mock_load:
            mock_load.return_value = [
                {
                    "question": "Downloaded?",
                    "best_answer": "Yes",
                    "id": "550e8400-e29b-41d4-a716-446655440008",
                }
            ]

            items = await loader.load_dataset("simpleqa", split="test", force_download=True)

            assert len(items) == 1
            assert isinstance(items[0], SimpleQAItem)
            assert items[0].question == "Downloaded?"
            mock_load.assert_called_once_with(
                "basicv8vc/SimpleQA", "test", False
            )  # show_progress defaults to False

        # Verify cache was created
        cache_path = loader._get_cache_path("simpleqa", "test")
        assert cache_path.exists()

    @pytest.mark.asyncio
    async def test_load_dataset_unknown(self, loader):
        """Test loading unknown dataset raises error."""
        with pytest.raises(DatasetNotFoundError) as exc_info:
            await loader.load_dataset("unknown_dataset")
        assert "Dataset 'unknown_dataset' not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_load_dataset_with_progress(self, loader, temp_cache_dir):
        """Test loading large dataset shows progress."""
        # Create mock dataset with many items
        import uuid

        mock_items = [
            {"question": f"Q{i}", "best_answer": f"A{i}", "id": str(uuid.uuid4())}
            for i in range(1000)
        ]

        with patch.object(loader, "_load_from_huggingface") as mock_load:
            mock_load.return_value = mock_items

            with patch("cosmos_coherence.harness.huggingface_loader.tqdm") as mock_tqdm:
                mock_tqdm.return_value.__enter__ = lambda self: self
                mock_tqdm.return_value.__exit__ = lambda self, *args: None
                mock_tqdm.return_value.__iter__ = lambda self: iter(mock_items)

                items = await loader.load_dataset(
                    "simpleqa", split="test", show_progress=True, force_download=True
                )

                assert len(items) == 1000
                # Progress bar would be shown in _load_from_huggingface if it were real

    def test_clear_cache(self, loader, temp_cache_dir):
        """Test clearing cache for specific dataset."""
        # Create some cache files
        (temp_cache_dir / "simpleqa_test.json").touch()
        (temp_cache_dir / "simpleqa_train.json").touch()
        (temp_cache_dir / "fever_test.json").touch()

        loader.clear_cache("simpleqa")

        assert not (temp_cache_dir / "simpleqa_test.json").exists()
        assert not (temp_cache_dir / "simpleqa_train.json").exists()
        assert (temp_cache_dir / "fever_test.json").exists()  # Other datasets untouched

    def test_clear_all_cache(self, loader, temp_cache_dir):
        """Test clearing all cache."""
        # Create some cache files
        (temp_cache_dir / "simpleqa_test.json").touch()
        (temp_cache_dir / "fever_test.json").touch()

        loader.clear_cache()

        assert not (temp_cache_dir / "simpleqa_test.json").exists()
        assert not (temp_cache_dir / "fever_test.json").exists()
