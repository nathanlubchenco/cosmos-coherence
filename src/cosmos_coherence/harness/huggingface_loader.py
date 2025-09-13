"""HuggingFace dataset loader with caching support."""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from cosmos_coherence.config.huggingface_config import HuggingFaceConfig

from pydantic import ValidationError

from cosmos_coherence.benchmarks.models.base import DatasetValidationError
from cosmos_coherence.benchmarks.models.datasets import (
    FaithBenchItem,
    FEVERItem,
    FEVERLabel,
    HaluEvalItem,
    SimpleQAItem,
    TruthfulQAItem,
)

logger = logging.getLogger(__name__)

# Progress bar import with graceful fallback
try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("tqdm not available, progress bars will be disabled")


class DatasetLoadError(Exception):
    """Raised when dataset loading fails."""

    pass


class DatasetNotFoundError(Exception):
    """Raised when a dataset is not found or not supported."""

    pass


class HuggingFaceDatasetLoader:
    """Loader for HuggingFace datasets with local caching."""

    # Mapping of dataset names to HuggingFace identifiers
    DEFAULT_DATASET_MAPPING = {
        "faithbench": "vectara/faithbench",
        "simpleqa": "basicv8vc/SimpleQA",
        "truthfulqa": "truthfulqa/truthful_qa",
        "fever": "fever/fever",
        "halueval": "pminervini/HaluEval",
    }

    def __init__(
        self, cache_dir: Optional[Path] = None, config: Optional["HuggingFaceConfig"] = None
    ):
        """Initialize the dataset loader.

        Args:
            cache_dir: Directory for caching datasets. Defaults to config or .cache/datasets/
            config: HuggingFace configuration object
        """
        # Import config here to avoid circular dependencies
        from cosmos_coherence.config.huggingface_config import HuggingFaceConfig

        self.config = config or HuggingFaceConfig.from_env()
        self.cache_dir = cache_dir or self.config.cache_dir
        self.dataset_mapping = self.DEFAULT_DATASET_MAPPING.copy()

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"HuggingFace loader initialized with cache at {self.cache_dir}")

    def _get_cache_path(self, dataset_name: str, split: Optional[str] = None) -> Path:
        """Get the cache file path for a dataset.

        Args:
            dataset_name: Name of the dataset
            split: Dataset split (train/validation/test)

        Returns:
            Path to cache file
        """
        split_suffix = split if split else "default"
        filename = f"{dataset_name}_{split_suffix}.json"
        return self.cache_dir / filename

    def _load_from_cache(self, cache_path: Path) -> Optional[List[Dict[str, Any]]]:
        """Load dataset from cache if it exists.

        Args:
            cache_path: Path to cache file

        Returns:
            Cached data or None if not found
        """
        if cache_path.exists():
            logger.info(f"Loading dataset from cache: {cache_path}")
            try:
                with open(cache_path, "r") as f:
                    data: List[Dict[str, Any]] = json.load(f)
                    return data
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache: {e}")
                return None
        return None

    def _save_to_cache(self, data: List[Dict[str, Any]], cache_path: Path) -> None:
        """Save dataset to cache.

        Args:
            data: Dataset to cache
            cache_path: Path to cache file
        """
        try:
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved dataset to cache: {cache_path}")
        except IOError as e:
            logger.warning(f"Failed to save cache: {e}")

    def _load_from_huggingface(
        self, hf_identifier: str, split: str, show_progress: bool = False
    ) -> List[Dict[str, Any]]:
        """Load dataset from HuggingFace.

        Args:
            hf_identifier: HuggingFace dataset identifier
            split: Dataset split to load
            show_progress: Whether to show progress bar

        Returns:
            List of dataset items

        Raises:
            DatasetLoadError: If loading fails
        """
        try:
            # Import datasets library
            try:
                from datasets import load_dataset
            except ImportError:
                raise DatasetLoadError("datasets library not installed. Run: pip install datasets")

            logger.info(f"Downloading dataset from HuggingFace: {hf_identifier}")

            # Load dataset from HuggingFace
            dataset = load_dataset(hf_identifier, split=split)

            # Convert to list of dictionaries
            items = []

            # Use progress bar if available and requested
            if show_progress and TQDM_AVAILABLE and len(dataset) > 100:
                iterator = tqdm(dataset, desc=f"Loading {hf_identifier}")
            else:
                iterator = dataset

            for item in iterator:
                items.append(dict(item))

            logger.info(f"Downloaded {len(items)} items from {hf_identifier}")
            return items

        except Exception as e:
            raise DatasetLoadError(f"Failed to load dataset {hf_identifier}: {e}") from e

    def _convert_faithbench_item(self, item: Dict[str, Any]) -> FaithBenchItem:
        """Convert raw FaithBench item to Pydantic model.

        FaithBench format from repository: data_for_release/batch_{batch_id}.json
        """

        # Handle ID field - use sample_id or generate
        item_id = item.get("id")

        # Handle annotations structure from FaithBench format
        annotations = item.get("annotations", [])
        annotation_label = None
        annotation_spans = []
        annotation_justification = None

        if annotations and isinstance(annotations, list) and len(annotations) > 0:
            # Take first annotation as primary (FaithBench may have multiple annotators)
            first_annotation = annotations[0] if isinstance(annotations[0], dict) else {}
            annotation_label = first_annotation.get("label")
            annotation_spans = first_annotation.get("spans", [])
            annotation_justification = first_annotation.get("justification")

        # Handle metadata fields
        metadata = item.get("metadata", {})
        detector_predictions = metadata.get("detector_predictions", {})
        entropy_score = metadata.get("entropy_score")
        summarizer_model = metadata.get("summarizer", item.get("summarizer_model"))

        kwargs = {
            "sample_id": item.get("sample_id", item.get("id", "")),
            "source": item.get("source", item.get("context", "")),  # Source text to summarize
            "summary": item.get("summary", item.get("claim", "")),  # Generated summary
            "annotation_label": annotation_label,
            "annotation_spans": annotation_spans if annotation_spans else [],
            "annotation_justification": annotation_justification,
            "detector_predictions": detector_predictions,
            "entropy_score": entropy_score,
            "summarizer_model": summarizer_model,
            # For BaseDatasetItem compatibility
            "question": item.get("source", "")[:100] + "..."
            if item.get("source")
            else "Summarize the text",
        }

        if item_id:
            kwargs["id"] = item_id

        return FaithBenchItem(**kwargs)

    def _convert_simpleqa_item(self, item: Dict[str, Any]) -> SimpleQAItem:
        """Convert raw SimpleQA item to Pydantic model."""
        # Handle ID field
        item_id = item.get("id")
        kwargs = {
            "question": item.get("question", ""),
            "best_answer": item.get(
                "best_answer", item.get("answer", "")
            ),  # Support both field names
            "category": item.get("category"),
            "difficulty": item.get("difficulty"),
            "sources": item.get("sources"),
            "grading_notes": item.get("grading_notes"),
            "metadata": item.get("metadata", {}),
        }
        if item_id:
            kwargs["id"] = item_id
        return SimpleQAItem(**kwargs)

    def _convert_truthfulqa_item(self, item: Dict[str, Any]) -> TruthfulQAItem:
        """Convert raw TruthfulQA item to Pydantic model."""
        from cosmos_coherence.benchmarks.models.datasets import TruthfulQACategory

        # Handle ID field
        item_id = item.get("id")

        # Handle category - default to "other" if not provided or invalid
        category = item.get("category", "other")
        if category and category.lower() not in [c.value for c in TruthfulQACategory]:
            category = "other"

        kwargs = {
            "question": item.get("question", ""),
            "best_answer": item.get("best_answer", ""),
            "correct_answers": item.get("correct_answers", []),
            "incorrect_answers": item.get("incorrect_answers", []),
            "category": category,
            "source": item.get("source"),
        }
        if item_id:
            kwargs["id"] = item_id
        return TruthfulQAItem(**kwargs)

    def _convert_fever_item(self, item: Dict[str, Any]) -> FEVERItem:
        """Convert raw FEVER item to Pydantic model."""
        # Convert label string to enum
        label_str = item.get("label", "NOTENOUGHINFO")
        try:
            label = FEVERLabel(label_str)
        except ValueError:
            label = FEVERLabel.NOTENOUGHINFO

        # Handle ID field
        item_id = item.get("id")
        kwargs = {
            "question": item.get("claim", ""),  # FEVER uses claim as question
            "claim": item.get("claim", ""),
            "label": label,
            "evidence": item.get("evidence", []),
            "verdict": item.get("verdict"),
            "wikipedia_url": item.get("wikipedia_url"),
            "annotation_id": item.get("annotation_id"),
        }
        if item_id:
            kwargs["id"] = item_id
        return FEVERItem(**kwargs)

    def _convert_halueval_item(self, item: Dict[str, Any]) -> HaluEvalItem:
        """Convert raw HaluEval item to Pydantic model."""
        # Handle ID field
        item_id = item.get("id")
        kwargs = {
            "question": item.get("question", ""),
            "knowledge": item.get("knowledge", ""),
            "right_answer": item.get("right_answer", item.get("answer", "")),
            "hallucinated_answer": item.get("hallucinated_answer", ""),
            "task_type": item.get("task_type", "general"),
            "dialogue_history": item.get("dialogue_history"),
        }
        if item_id:
            kwargs["id"] = item_id
        return HaluEvalItem(**kwargs)

    def _convert_to_pydantic(self, raw_data: List[Dict[str, Any]], dataset_name: str) -> List[Any]:
        """Convert raw dataset to Pydantic models.

        Args:
            raw_data: Raw dataset items
            dataset_name: Name of the dataset

        Returns:
            List of Pydantic model instances

        Raises:
            DatasetValidationError: If validation fails
            DatasetNotFoundError: If dataset type is unknown
        """
        converters = {
            "faithbench": self._convert_faithbench_item,
            "simpleqa": self._convert_simpleqa_item,
            "truthfulqa": self._convert_truthfulqa_item,
            "fever": self._convert_fever_item,
            "halueval": self._convert_halueval_item,
        }

        if dataset_name not in converters:
            raise DatasetNotFoundError(f"Unknown dataset type: {dataset_name}")

        converter = converters[dataset_name]
        items = []

        for i, raw_item in enumerate(raw_data):
            try:
                item = converter(raw_item)
                items.append(item)
            except (ValidationError, ValueError) as e:
                raise DatasetValidationError(
                    f"Validation failed for item {i} in {dataset_name}",
                    field=f"item_{i}",
                    value=raw_item,
                    error_code="VALIDATION_ERROR",
                ) from e

        return items

    async def load_dataset(
        self,
        dataset_name: str,
        split: Optional[str] = None,
        force_download: bool = False,
        show_progress: bool = False,
        sample_size: Optional[int] = None,
    ) -> List[Any]:
        """Load a dataset with caching support.

        Args:
            dataset_name: Name of the dataset (e.g., "simpleqa", "fever")
            split: Dataset split (train/validation/test)
            force_download: Force download even if cached
            show_progress: Show progress bar for large datasets
            sample_size: Number of items to return (takes first N items)

        Returns:
            List of Pydantic model instances

        Raises:
            DatasetNotFoundError: If dataset is not supported
            DatasetLoadError: If loading fails
            DatasetValidationError: If validation fails
        """
        # Normalize dataset name
        dataset_name = dataset_name.lower()

        # Check if dataset is supported
        if dataset_name not in self.dataset_mapping:
            raise DatasetNotFoundError(
                f"Dataset '{dataset_name}' not found. "
                f"Supported datasets: {list(self.dataset_mapping.keys())}"
            )

        # Get cache path
        cache_path = self._get_cache_path(dataset_name, split)

        # Try to load from cache unless forced to download
        raw_data = None
        if not force_download:
            raw_data = self._load_from_cache(cache_path)

        # Download if not cached
        if raw_data is None:
            hf_identifier = self.dataset_mapping[dataset_name]

            # Determine split if not provided
            if split is None:
                # Default splits for each dataset
                default_splits = {
                    "simpleqa": "test",
                    "truthfulqa": "validation",
                    "fever": "test",
                    "faithbench": "test",
                    "halueval": "test",
                }
                split = default_splits.get(dataset_name, "test")

            # Check for CI/test mode using config
            if not self.config.is_ci_environment() or force_download:
                raw_data = self._load_from_huggingface(hf_identifier, split, show_progress)
                # Save to cache
                self._save_to_cache(raw_data, cache_path)
            else:
                # In CI mode without cache, return empty list
                logger.warning(f"CI mode: Returning empty dataset for {dataset_name}")
                return []

        # Convert to Pydantic models
        items = self._convert_to_pydantic(raw_data, dataset_name)

        # Apply sampling if requested
        if sample_size is not None and sample_size > 0:
            original_count = len(items)
            items = items[:sample_size]
            logger.info(
                f"SAMPLE MODE: Using first {len(items)} of {original_count} "
                f"items from {dataset_name}"
            )
        else:
            logger.info(f"Loaded {len(items)} items for {dataset_name}")

        return items

    def clear_cache(self, dataset_name: Optional[str] = None) -> None:
        """Clear cache for specific dataset or all datasets.

        Args:
            dataset_name: Dataset to clear cache for, or None for all
        """
        if dataset_name:
            # Clear specific dataset cache
            pattern = f"{dataset_name}_*.json"
            for cache_file in self.cache_dir.glob(pattern):
                cache_file.unlink()
                logger.info(f"Cleared cache: {cache_file}")
        else:
            # Clear all cache
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
                logger.info(f"Cleared cache: {cache_file}")
            logger.info("Cleared all dataset cache")
