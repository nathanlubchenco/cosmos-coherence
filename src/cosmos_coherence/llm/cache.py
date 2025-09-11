"""LLM response caching implementation."""

import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


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
    """Thread-safe in-memory cache for LLM responses."""

    def __init__(self):
        """Initialize the cache."""
        self._cache: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._statistics = CacheStatistics()

    def generate_cache_key(self, params: Dict[str, Any]) -> str:
        """Generate a deterministic hash key from request parameters.

        Args:
            params: Dictionary of request parameters

        Returns:
            SHA256 hash of the parameters as a hex string
        """
        # Sort keys to ensure deterministic ordering
        sorted_params = json.dumps(params, sort_keys=True, separators=(",", ":"))

        # Generate SHA256 hash
        hash_object = hashlib.sha256(sorted_params.encode())
        return hash_object.hexdigest()

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
