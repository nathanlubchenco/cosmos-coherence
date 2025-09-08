"""Configuration for HuggingFace dataset integration."""

import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class HuggingFaceConfig(BaseModel):
    """Configuration for HuggingFace dataset loading."""

    # Cache configuration
    cache_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("HF_CACHE_DIR", ".cache/datasets")),
        description="Directory for caching HuggingFace datasets",
    )

    # Download configuration
    force_download: bool = Field(
        default=False,
        description="Force re-download even if cached",
    )

    show_progress: bool = Field(
        default_factory=lambda: os.getenv("HF_SHOW_PROGRESS", "true").lower() == "true",
        description="Show progress bar during download",
    )

    # CI/Test mode configuration
    use_cached_only: bool = Field(
        default_factory=lambda: os.getenv("CI", "false").lower() == "true"
        or os.getenv("PYTEST_CURRENT_TEST") is not None,
        description="Only use cached data, don't download (for CI/tests)",
    )

    # Network configuration
    max_retries: int = Field(
        default=3,
        description="Maximum retries for network requests",
    )

    timeout_seconds: int = Field(
        default=300,
        description="Timeout for dataset downloads in seconds",
    )

    # Dataset-specific configuration
    default_split: str = Field(
        default="test",
        description="Default dataset split to use",
    )

    # Memory configuration
    load_in_memory: bool = Field(
        default=True,
        description="Load entire dataset into memory",
    )

    max_dataset_size_mb: int = Field(
        default=1000,
        description="Maximum dataset size to load in memory (MB)",
    )

    class Config:
        """Pydantic configuration."""

        env_prefix = "HF_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @classmethod
    def from_env(cls) -> "HuggingFaceConfig":
        """Create configuration from environment variables.

        Environment variables:
        - HF_CACHE_DIR: Cache directory path
        - HF_SHOW_PROGRESS: Show download progress (true/false)
        - HF_FORCE_DOWNLOAD: Force re-download (true/false)
        - CI: Running in CI environment (true/false)
        - PYTEST_CURRENT_TEST: Set by pytest during tests

        Returns:
            HuggingFaceConfig instance
        """
        return cls()

    def get_cache_path(self, dataset_name: str, split: Optional[str] = None) -> Path:
        """Get the cache file path for a specific dataset.

        Args:
            dataset_name: Name of the dataset
            split: Dataset split (train/validation/test)

        Returns:
            Path to cache file
        """
        split = split or self.default_split
        filename = f"{dataset_name}_{split}.json"
        return self.cache_dir / filename

    def ensure_cache_dir(self) -> None:
        """Ensure cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def is_ci_environment(self) -> bool:
        """Check if running in CI/test environment.

        Returns:
            True if in CI or test environment
        """
        return self.use_cached_only

    def should_download(self) -> bool:
        """Check if downloads are allowed.

        Returns:
            True if downloads are allowed
        """
        return not self.use_cached_only and not self.force_download
