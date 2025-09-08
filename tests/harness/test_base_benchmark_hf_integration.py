"""Tests for BaseBenchmark integration with HuggingFace datasets."""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch
from uuid import UUID

import pytest
from cosmos_coherence.benchmarks.models.base import BaseDatasetItem
from cosmos_coherence.benchmarks.models.datasets import SimpleQAItem
from cosmos_coherence.harness.base_benchmark import (
    BaseBenchmark,
    BenchmarkEvaluationResult,
)


class SimpleQABenchmark(BaseBenchmark):
    """SimpleQA benchmark implementation for testing HF integration."""

    def __init__(self, use_huggingface: bool = True, cache_dir: Optional[Path] = None):
        super().__init__()
        self.use_huggingface = use_huggingface
        self.cache_dir = cache_dir
        self._hf_dataset_name = "simpleqa"  # This triggers HF loading
        self._hf_split = "test"

    async def load_dataset(self) -> List[BaseDatasetItem]:
        """Load dataset - will use HF if _hf_dataset_name is set."""
        if hasattr(self, "_hf_dataset_name"):
            # This will be handled by the base class in the modified version
            from cosmos_coherence.harness.huggingface_loader import HuggingFaceDatasetLoader

            loader = HuggingFaceDatasetLoader(cache_dir=self.cache_dir)
            show_progress = getattr(self, "_hf_show_progress", False)
            return await loader.load_dataset(
                self._hf_dataset_name, split=self._hf_split, show_progress=show_progress
            )
        return []

    def get_prompt(self, item: BaseDatasetItem) -> str:
        """Format item into prompt."""
        if isinstance(item, SimpleQAItem):
            return f"Question: {item.question}"
        return str(item.question)

    def evaluate_response(
        self, response: str, ground_truth: str, item: BaseDatasetItem
    ) -> BenchmarkEvaluationResult:
        """Evaluate response."""
        is_correct = response.strip().lower() == ground_truth.strip().lower()
        return BenchmarkEvaluationResult(
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0,
            original_metric_score=1.0 if is_correct else 0.0,
        )

    def get_baseline_metrics(self) -> Dict[str, float]:
        """Return baseline metrics."""
        return {"accuracy": 0.85}

    def get_original_prompts(self) -> List[str]:
        """Return example prompts."""
        return ["Question: What is the capital of France?"]

    def validate_config(self, config: Dict) -> None:
        """Validate config."""
        pass

    @property
    def benchmark_name(self) -> str:
        return "SimpleQA"

    @property
    def paper_reference(self) -> str:
        return "SimpleQA: Test Paper 2024"

    def get_evaluation_method(self) -> str:
        return "Exact match evaluation"


class TestBaseBenchmarkHFIntegration:
    """Test suite for HuggingFace integration with BaseBenchmark."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_hf_data(self):
        """Mock HuggingFace dataset data."""
        return [
            {
                "question": "What is 2+2?",
                "best_answer": "4",
                "id": str(UUID("00000000-0000-0000-0000-000000000001")),
            },
            {
                "question": "What is the capital of France?",
                "best_answer": "Paris",
                "id": str(UUID("00000000-0000-0000-0000-000000000002")),
            },
        ]

    @pytest.mark.asyncio
    async def test_load_dataset_from_huggingface(self, temp_cache_dir, mock_hf_data):
        """Test loading dataset from HuggingFace."""
        benchmark = SimpleQABenchmark(use_huggingface=True, cache_dir=temp_cache_dir)

        # Mock the HF loader
        with patch(
            "cosmos_coherence.harness.huggingface_loader.HuggingFaceDatasetLoader.load_dataset"
        ) as mock_load:
            # Create SimpleQAItem objects from mock data
            mock_items = [
                SimpleQAItem(
                    id=UUID(item["id"]),
                    question=item["question"],
                    best_answer=item["best_answer"],
                )
                for item in mock_hf_data
            ]
            mock_load.return_value = mock_items

            dataset = await benchmark.load_dataset()

            assert len(dataset) == 2
            assert all(isinstance(item, SimpleQAItem) for item in dataset)
            assert dataset[0].question == "What is 2+2?"
            assert dataset[0].best_answer == "4"
            mock_load.assert_called_once_with("simpleqa", split="test", show_progress=False)

    @pytest.mark.asyncio
    async def test_benchmark_with_cached_hf_data(self, temp_cache_dir, mock_hf_data):
        """Test that cached HF data is used when available."""
        # Pre-populate cache
        cache_file = temp_cache_dir / "simpleqa_test.json"
        with open(cache_file, "w") as f:
            json.dump(mock_hf_data, f)

        benchmark = SimpleQABenchmark(use_huggingface=True, cache_dir=temp_cache_dir)

        # The loader should use cached data
        dataset = await benchmark.load_dataset()

        assert len(dataset) == 2
        assert dataset[0].question == "What is 2+2?"

    @pytest.mark.asyncio
    async def test_prompt_generation_with_hf_data(self, temp_cache_dir, mock_hf_data):
        """Test prompt generation with HuggingFace data."""
        benchmark = SimpleQABenchmark(use_huggingface=True, cache_dir=temp_cache_dir)

        with patch(
            "cosmos_coherence.harness.huggingface_loader.HuggingFaceDatasetLoader.load_dataset"
        ) as mock_load:
            mock_items = [
                SimpleQAItem(
                    id=UUID(mock_hf_data[0]["id"]),
                    question=mock_hf_data[0]["question"],
                    best_answer=mock_hf_data[0]["best_answer"],
                )
            ]
            mock_load.return_value = mock_items

            dataset = await benchmark.load_dataset()
            prompt = benchmark.get_prompt(dataset[0])

            assert prompt == "Question: What is 2+2?"

    @pytest.mark.asyncio
    async def test_evaluation_with_hf_data(self, temp_cache_dir, mock_hf_data):
        """Test evaluation with HuggingFace data."""
        benchmark = SimpleQABenchmark(use_huggingface=True, cache_dir=temp_cache_dir)

        with patch(
            "cosmos_coherence.harness.huggingface_loader.HuggingFaceDatasetLoader.load_dataset"
        ) as mock_load:
            mock_items = [
                SimpleQAItem(
                    id=UUID(mock_hf_data[0]["id"]),
                    question=mock_hf_data[0]["question"],
                    best_answer=mock_hf_data[0]["best_answer"],
                )
            ]
            mock_load.return_value = mock_items

            dataset = await benchmark.load_dataset()
            result = benchmark.evaluate_response("4", "4", dataset[0])

            assert result.is_correct
            assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_dataset_statistics_with_hf_data(self, temp_cache_dir, mock_hf_data):
        """Test getting statistics for HuggingFace dataset."""
        benchmark = SimpleQABenchmark(use_huggingface=True, cache_dir=temp_cache_dir)

        with patch(
            "cosmos_coherence.harness.huggingface_loader.HuggingFaceDatasetLoader.load_dataset"
        ) as mock_load:
            mock_items = [
                SimpleQAItem(
                    id=UUID(item["id"]),
                    question=item["question"],
                    best_answer=item["best_answer"],
                )
                for item in mock_hf_data
            ]
            mock_load.return_value = mock_items

            stats = await benchmark.get_dataset_statistics()

            assert stats["total_items"] == 2
            assert stats["average_question_length"] > 0

    @pytest.mark.asyncio
    async def test_hf_configuration_options(self, temp_cache_dir):
        """Test configuration options for HuggingFace datasets."""
        benchmark = SimpleQABenchmark(use_huggingface=True, cache_dir=temp_cache_dir)

        # Test that HF-specific attributes are recognized
        assert hasattr(benchmark, "_hf_dataset_name")
        assert benchmark._hf_dataset_name == "simpleqa"
        assert benchmark._hf_split == "test"

        # Test custom split
        benchmark._hf_split = "validation"

        with patch(
            "cosmos_coherence.harness.huggingface_loader.HuggingFaceDatasetLoader.load_dataset"
        ) as mock_load:
            mock_load.return_value = []
            await benchmark.load_dataset()

            mock_load.assert_called_once_with("simpleqa", split="validation", show_progress=False)

    @pytest.mark.asyncio
    async def test_progress_indicator_for_large_dataset(self, temp_cache_dir):
        """Test that progress indicator is shown for large datasets."""
        benchmark = SimpleQABenchmark(use_huggingface=True, cache_dir=temp_cache_dir)
        benchmark._hf_show_progress = True  # Enable progress

        # Create a large mock dataset
        large_mock_data = [
            {
                "question": f"Question {i}",
                "best_answer": f"Answer {i}",
                "id": str(UUID(f"00000000-0000-0000-0000-{i:012d}")),
            }
            for i in range(1000)
        ]

        with patch(
            "cosmos_coherence.harness.huggingface_loader.HuggingFaceDatasetLoader.load_dataset"
        ) as mock_load:
            mock_items = [
                SimpleQAItem(
                    id=UUID(item["id"]),
                    question=item["question"],
                    best_answer=item["best_answer"],
                )
                for item in large_mock_data
            ]
            mock_load.return_value = mock_items

            dataset = await benchmark.load_dataset()

            assert len(dataset) == 1000
            # Check that progress was requested
            mock_load.assert_called_once_with("simpleqa", split="test", show_progress=True)

    @pytest.mark.asyncio
    async def test_fallback_to_local_if_hf_fails(self, temp_cache_dir):
        """Test fallback behavior when HuggingFace loading fails."""
        benchmark = SimpleQABenchmark(use_huggingface=True, cache_dir=temp_cache_dir)

        with patch(
            "cosmos_coherence.harness.huggingface_loader.HuggingFaceDatasetLoader.load_dataset"
        ) as mock_load:
            mock_load.side_effect = Exception("Network error")

            # Should raise the exception (no silent failures)
            with pytest.raises(Exception, match="Network error"):
                await benchmark.load_dataset()

    @pytest.mark.asyncio
    async def test_mixed_benchmark_types(self, temp_cache_dir):
        """Test that both HF and non-HF benchmarks work together."""
        # Create HF benchmark
        hf_benchmark = SimpleQABenchmark(use_huggingface=True, cache_dir=temp_cache_dir)

        # Create non-HF benchmark (mock from test_base_benchmark.py)
        from tests.harness.test_base_benchmark import MockBenchmark

        non_hf_benchmark = MockBenchmark()

        # Both should work independently
        with patch(
            "cosmos_coherence.harness.huggingface_loader.HuggingFaceDatasetLoader.load_dataset"
        ) as mock_load:
            mock_load.return_value = [
                SimpleQAItem(
                    id=UUID("00000000-0000-0000-0000-000000000001"),
                    question="HF Question",
                    best_answer="HF Answer",
                )
            ]

            hf_dataset = await hf_benchmark.load_dataset()
            non_hf_dataset = await non_hf_benchmark.load_dataset()

            assert len(hf_dataset) == 1
            assert len(non_hf_dataset) == 2
            assert hf_dataset[0].question == "HF Question"
            assert non_hf_dataset[0].question == "What is 2+2?"
