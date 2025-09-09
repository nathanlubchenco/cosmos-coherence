"""Enhanced BaseBenchmark with HuggingFace dataset support."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from cosmos_coherence.benchmarks.models.base import BaseDatasetItem
from cosmos_coherence.harness.base_benchmark import BaseBenchmark

logger = logging.getLogger(__name__)


class HuggingFaceEnabledBenchmark(BaseBenchmark):
    """Base benchmark class with HuggingFace dataset integration.

    Subclasses can use HuggingFace datasets by setting:
    - _hf_dataset_name: The dataset identifier (e.g., "simpleqa", "faithbench")
    - _hf_split: The dataset split to use (default: "test")
    - _hf_cache_dir: Cache directory for datasets (default: .cache/datasets)
    - _hf_show_progress: Show download progress (default: False)
    - _hf_force_download: Force re-download even if cached (default: False)
    """

    def __init__(
        self,
        hf_dataset_name: Optional[str] = None,
        hf_split: str = "test",
        hf_cache_dir: Optional[Path] = None,
        hf_show_progress: bool = False,
        hf_force_download: bool = False,
        sample_size: Optional[int] = None,
    ):
        """Initialize benchmark with optional HuggingFace configuration.

        Args:
            hf_dataset_name: HuggingFace dataset identifier
            hf_split: Dataset split to use
            hf_cache_dir: Cache directory for datasets
            hf_show_progress: Show download progress
            hf_force_download: Force re-download even if cached
            sample_size: Number of items to load (first N items)
        """
        super().__init__()

        # HuggingFace configuration
        self._hf_dataset_name = hf_dataset_name
        self._hf_split = hf_split
        self._hf_cache_dir = hf_cache_dir
        self._hf_show_progress = hf_show_progress
        self._hf_force_download = hf_force_download
        self._sample_size = sample_size

        # Lazy-loaded HF loader
        self._hf_loader = None

        if hf_dataset_name:
            logger.info(
                f"Initialized {self.benchmark_name} with HuggingFace dataset: "
                f"{hf_dataset_name} (split: {hf_split})"
            )

    def _get_hf_loader(self):
        """Get or create HuggingFace dataset loader."""
        if self._hf_loader is None:
            from cosmos_coherence.harness.huggingface_loader import HuggingFaceDatasetLoader

            self._hf_loader = HuggingFaceDatasetLoader(cache_dir=self._hf_cache_dir)
        return self._hf_loader

    async def load_dataset(self) -> List[BaseDatasetItem]:
        """Load dataset from HuggingFace or fallback to custom implementation.

        Returns:
            List of dataset items

        Raises:
            DatasetLoadError: If HuggingFace loading fails
        """
        # Check if HuggingFace dataset is configured
        if self._hf_dataset_name:
            logger.info(f"Loading dataset from HuggingFace: {self._hf_dataset_name}")

            loader = self._get_hf_loader()

            try:
                # Load from HuggingFace
                dataset = await loader.load_dataset(
                    dataset_name=self._hf_dataset_name,
                    split=self._hf_split,
                    force_download=self._hf_force_download,
                    show_progress=self._hf_show_progress,
                    sample_size=self._sample_size,
                )

                logger.info(f"Loaded {len(dataset)} items from HuggingFace")
                self._dataset = dataset
                return cast(List[BaseDatasetItem], dataset)

            except Exception as e:
                logger.error(f"Failed to load HuggingFace dataset: {e}")
                # Re-raise to maintain explicit error handling
                raise

        # Fallback to custom implementation
        return await self._load_custom_dataset()

    async def _load_custom_dataset(self) -> List[BaseDatasetItem]:
        """Load dataset using custom implementation.

        Subclasses should override this method if they need custom loading
        logic when not using HuggingFace datasets.

        Returns:
            List of dataset items
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _load_custom_dataset() "
            "or configure HuggingFace dataset"
        )

    def configure_huggingface(
        self,
        dataset_name: str,
        split: str = "test",
        cache_dir: Optional[Path] = None,
        show_progress: bool = False,
        force_download: bool = False,
    ) -> None:
        """Configure HuggingFace dataset parameters.

        Args:
            dataset_name: HuggingFace dataset identifier
            split: Dataset split to use
            cache_dir: Cache directory for datasets
            show_progress: Show download progress
            force_download: Force re-download even if cached
        """
        self._hf_dataset_name = dataset_name
        self._hf_split = split
        self._hf_cache_dir = cache_dir
        self._hf_show_progress = show_progress
        self._hf_force_download = force_download

        # Reset loader to apply new configuration
        self._hf_loader = None

        logger.info(f"Configured HuggingFace dataset: {dataset_name} (split: {split})")

    def clear_hf_cache(self, dataset_name: Optional[str] = None) -> None:
        """Clear HuggingFace dataset cache.

        Args:
            dataset_name: Specific dataset to clear, or None for all
        """
        if self._hf_loader:
            self._hf_loader.clear_cache(dataset_name)
            logger.info(f"Cleared HuggingFace cache for: {dataset_name or 'all datasets'}")

    def get_hf_dataset_info(self) -> Dict[str, Any]:
        """Get information about configured HuggingFace dataset.

        Returns:
            Dictionary with dataset configuration
        """
        return {
            "enabled": bool(self._hf_dataset_name),
            "dataset_name": self._hf_dataset_name,
            "split": self._hf_split,
            "cache_dir": str(self._hf_cache_dir) if self._hf_cache_dir else None,
            "show_progress": self._hf_show_progress,
            "force_download": self._hf_force_download,
        }

    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str,
        split: str = "test",
        cache_dir: Optional[Path] = None,
        show_progress: bool = False,
        force_download: bool = False,
    ):
        """Create benchmark instance configured for HuggingFace dataset.

        Args:
            dataset_name: HuggingFace dataset identifier
            split: Dataset split to use
            cache_dir: Cache directory for datasets
            show_progress: Show download progress
            force_download: Force re-download even if cached

        Returns:
            Configured benchmark instance
        """
        instance = cls()
        instance.configure_huggingface(
            dataset_name=dataset_name,
            split=split,
            cache_dir=cache_dir,
            show_progress=show_progress,
            force_download=force_download,
        )
        return instance
