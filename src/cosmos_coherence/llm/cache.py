"""LLM response caching implementation."""

import gzip
import json
import logging
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import xxhash

logger = logging.getLogger(__name__)


@dataclass
class CacheStatistics:
    """Cache performance statistics."""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    tokens_saved: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    def estimated_cost_savings(self) -> float:
        """Calculate estimated cost savings based on tokens saved.

        Using approximate GPT-4 pricing:
        - Input: $0.03 per 1K tokens
        - Output: $0.06 per 1K tokens
        - Average: $0.045 per 1K tokens
        """
        price_per_1k_tokens = 0.045
        return (self.tokens_saved / 1000) * price_per_1k_tokens


class LLMCache:
    """Thread-safe in-memory cache for LLM responses with disk persistence."""

    def __init__(self, cache_file: Optional[Union[str, Path]] = None):
        """Initialize the cache.

        Args:
            cache_file: Optional path to cache file for persistence.
                       If provided, will attempt to load existing cache.
        """
        self._cache: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._statistics = CacheStatistics()
        self._cache_file = Path(cache_file) if cache_file else None

        # Load existing cache if file is provided
        if self._cache_file:
            self.load_from_disk(self._cache_file)

    def generate_cache_key(self, params: Dict[str, Any]) -> str:
        """Generate a deterministic hash key from request parameters.

        Args:
            params: Dictionary of request parameters

        Returns:
            xxHash hash of the parameters as a hex string
        """
        # Sort keys to ensure deterministic ordering
        sorted_params = json.dumps(params, sort_keys=True, separators=(",", ":"))

        # Generate xxHash (64-bit) - much faster than SHA256 for cache keys
        # xxh64 is ~10x faster than SHA256 and sufficient for cache deduplication
        hash_object = xxhash.xxh64(sorted_params.encode())
        return str(hash_object.hexdigest())

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """Store a value in the cache.

        Args:
            key: Cache key
            value: Value to store
        """
        with self._lock:
            self._cache[key] = value

    def contains(self, key: str) -> bool:
        """Check if a key exists in the cache.

        Args:
            key: Cache key

        Returns:
            True if key exists, False otherwise
        """
        with self._lock:
            return key in self._cache

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._statistics = CacheStatistics()

    def size(self) -> int:
        """Get the number of entries in the cache.

        Returns:
            Number of cached entries
        """
        with self._lock:
            return len(self._cache)

    def lookup_or_compute(self, key: str, compute_fn: Callable[[], Any]) -> Any:
        """Look up a value in the cache or compute it if not found.

        This method is thread-safe and ensures the compute function is only
        called once even with concurrent requests for the same key.

        Args:
            key: Cache key
            compute_fn: Function to compute the value if not cached

        Returns:
            Cached or computed value
        """
        with self._lock:
            self._statistics.total_requests += 1

            # Check if value is already cached
            if key in self._cache:
                self._statistics.cache_hits += 1

                # Track tokens saved if the response has usage info
                cached_value = self._cache[key]
                if isinstance(cached_value, dict) and "usage" in cached_value:
                    usage = cached_value["usage"]
                    if isinstance(usage, dict) and "total_tokens" in usage:
                        self._statistics.tokens_saved += usage["total_tokens"]

                return cached_value

            # Cache miss - compute the value
            self._statistics.cache_misses += 1
            value = compute_fn()
            self._cache[key] = value
            return value

    def get_statistics(self) -> CacheStatistics:
        """Get cache performance statistics.

        Returns:
            CacheStatistics object with current metrics
        """
        with self._lock:
            # Return a copy to prevent external modification
            return CacheStatistics(
                total_requests=self._statistics.total_requests,
                cache_hits=self._statistics.cache_hits,
                cache_misses=self._statistics.cache_misses,
                tokens_saved=self._statistics.tokens_saved,
            )

    def save_to_disk(self, path: Union[str, Path], compress: bool = False) -> None:
        """Save cache to disk.

        Args:
            path: Path to save the cache file
            compress: Whether to compress the cache file with gzip
        """
        path = Path(path)

        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            # Prepare data for serialization
            data = {"cache": self._cache, "statistics": asdict(self._statistics)}

            # Use atomic write with temp file
            temp_fd = None
            temp_path: Optional[Path] = None
            try:
                # Create temp file in same directory for atomic rename
                temp_fd, temp_path_str = tempfile.mkstemp(
                    dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
                )
                temp_path = Path(temp_path_str)

                # Write to temp file
                if compress or str(path).endswith(".gz"):
                    with gzip.open(temp_path, "wt", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                else:
                    with open(temp_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)

                # Atomic rename
                temp_path.replace(path)
                logger.info(f"Cache saved to {path}")

            except Exception:
                # Clean up temp file on error
                if temp_path and temp_path.exists():
                    temp_path.unlink()
                raise
            finally:
                # Close temp file descriptor if still open
                if temp_fd is not None:
                    try:
                        import os

                        os.close(temp_fd)
                    except OSError:
                        pass

    def load_from_disk(self, path: Union[str, Path]) -> None:
        """Load cache from disk.

        Args:
            path: Path to the cache file
        """
        path = Path(path)

        if not path.exists():
            logger.warning(f"Cache file {path} does not exist")
            return

        try:
            # Detect if file is compressed
            if str(path).endswith(".gz"):
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                # Try gzip first in case it's compressed without .gz extension
                try:
                    with gzip.open(path, "rt", encoding="utf-8") as f:
                        data = json.load(f)
                except (OSError, gzip.BadGzipFile):
                    # Not gzipped, try regular JSON
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)

            with self._lock:
                # Load cache entries
                self._cache = data.get("cache", {})

                # Load statistics
                stats_data = data.get("statistics", {})
                self._statistics = CacheStatistics(
                    total_requests=stats_data.get("total_requests", 0),
                    cache_hits=stats_data.get("cache_hits", 0),
                    cache_misses=stats_data.get("cache_misses", 0),
                    tokens_saved=stats_data.get("tokens_saved", 0),
                )

            logger.info(f"Cache loaded from {path}: {self.size()} entries")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse cache file {path}: {e}")
        except Exception as e:
            logger.error(f"Failed to load cache from {path}: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save cache if file path is set."""
        if self._cache_file:
            try:
                self.save_to_disk(self._cache_file)
            except Exception as e:
                logger.error(f"Failed to save cache on exit: {e}")
        return False
