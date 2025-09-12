"""Tests for LLM response caching functionality."""

import gzip
import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from cosmos_coherence.llm.cache import CacheStatistics, LLMCache


class TestCacheKeyGeneration:
    """Test cache key generation and hashing."""

    def test_generate_cache_key_basic(self):
        """Test basic cache key generation."""
        cache = LLMCache()

        params = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
        }

        key = cache.generate_cache_key(params)

        assert isinstance(key, str)
        assert len(key) == 16  # xxHash64 produces 16 hex characters

    def test_cache_key_deterministic(self):
        """Test that same parameters produce same key."""
        cache = LLMCache()

        params = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Test prompt"}],
            "temperature": 0.5,
            "max_tokens": 100,
        }

        key1 = cache.generate_cache_key(params)
        key2 = cache.generate_cache_key(params)

        assert key1 == key2

    def test_cache_key_different_for_different_params(self):
        """Test that different parameters produce different keys."""
        cache = LLMCache()

        params1 = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
        }

        params2 = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.8,  # Different temperature
        }

        key1 = cache.generate_cache_key(params1)
        key2 = cache.generate_cache_key(params2)

        assert key1 != key2

    def test_cache_key_includes_all_parameters(self):
        """Test that all parameters affect the cache key."""
        cache = LLMCache()

        base_params = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Test"}],
        }

        # Test each parameter affects the key
        test_cases = [
            {"temperature": 0.5},
            {"max_tokens": 100},
            {"top_p": 0.9},
            {"presence_penalty": 0.1},
            {"frequency_penalty": 0.2},
            {"seed": 42},
        ]

        base_key = cache.generate_cache_key(base_params)

        for extra_param in test_cases:
            params = {**base_params, **extra_param}
            key = cache.generate_cache_key(params)
            assert key != base_key, f"Parameter {extra_param} did not affect cache key"

    def test_cache_key_order_independence(self):
        """Test that parameter order doesn't affect cache key."""
        cache = LLMCache()

        params1 = {
            "model": "gpt-4",
            "temperature": 0.7,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }

        params2 = {
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4",
            "temperature": 0.7,
        }

        key1 = cache.generate_cache_key(params1)
        key2 = cache.generate_cache_key(params2)

        assert key1 == key2


class TestInMemoryCache:
    """Test in-memory cache storage functionality."""

    def test_cache_initialization(self):
        """Test cache initializes with empty storage."""
        cache = LLMCache()

        assert cache.size() == 0
        assert cache.get("nonexistent") is None

    def test_cache_set_and_get(self):
        """Test setting and retrieving cache entries."""
        cache = LLMCache()

        key = "test_key"
        value = {"response": "test response", "usage": {"tokens": 10}}

        cache.set(key, value)

        assert cache.size() == 1
        assert cache.get(key) == value

    def test_cache_overwrite(self):
        """Test overwriting existing cache entry."""
        cache = LLMCache()

        key = "test_key"
        value1 = {"response": "first"}
        value2 = {"response": "second"}

        cache.set(key, value1)
        assert cache.get(key) == value1

        cache.set(key, value2)
        assert cache.get(key) == value2
        assert cache.size() == 1  # Still only one entry

    def test_cache_contains(self):
        """Test checking if key exists in cache."""
        cache = LLMCache()

        key = "test_key"
        value = {"response": "test"}

        assert not cache.contains(key)

        cache.set(key, value)

        assert cache.contains(key)

    def test_cache_clear(self):
        """Test clearing the cache."""
        cache = LLMCache()

        cache.set("key1", {"response": "value1"})
        cache.set("key2", {"response": "value2"})

        assert cache.size() == 2

        cache.clear()

        assert cache.size() == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestCacheLookupAndStorage:
    """Test cache lookup and storage operations."""

    def test_lookup_or_compute_cache_hit(self):
        """Test lookup_or_compute returns cached value on hit."""
        cache = LLMCache()

        key = "test_key"
        cached_value = {"response": "cached"}
        cache.set(key, cached_value)

        compute_fn = MagicMock()

        result = cache.lookup_or_compute(key, compute_fn)

        assert result == cached_value
        compute_fn.assert_not_called()  # Should not compute on cache hit

    def test_lookup_or_compute_cache_miss(self):
        """Test lookup_or_compute computes and stores on miss."""
        cache = LLMCache()

        key = "test_key"
        computed_value = {"response": "computed"}

        compute_fn = MagicMock(return_value=computed_value)

        result = cache.lookup_or_compute(key, compute_fn)

        assert result == computed_value
        compute_fn.assert_called_once()
        assert cache.get(key) == computed_value  # Should be stored in cache

    def test_cache_with_request_params(self):
        """Test caching with full request parameters."""
        cache = LLMCache()

        params = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "What is 2+2?"},
            ],
            "temperature": 0.0,
            "max_tokens": 10,
        }

        response = {
            "choices": [{"message": {"content": "4"}}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 1, "total_tokens": 16},
        }

        key = cache.generate_cache_key(params)
        cache.set(key, response)

        # Retrieve with same params
        retrieved = cache.get(key)
        assert retrieved == response

        # Different params should miss
        params["temperature"] = 0.1
        new_key = cache.generate_cache_key(params)
        assert cache.get(new_key) is None


class TestThreadSafety:
    """Test thread-safe cache operations."""

    def test_concurrent_writes(self):
        """Test concurrent write operations are thread-safe."""
        cache = LLMCache()
        results = []

        def write_to_cache(thread_id):
            for i in range(100):
                key = f"key_{thread_id}_{i}"
                value = {"thread": thread_id, "index": i}
                cache.set(key, value)
                results.append((key, value))

        threads = []
        for i in range(5):
            thread = threading.Thread(target=write_to_cache, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Verify all writes succeeded
        assert cache.size() == 500  # 5 threads * 100 writes

        for key, expected_value in results:
            assert cache.get(key) == expected_value

    def test_concurrent_reads_and_writes(self):
        """Test concurrent read and write operations."""
        cache = LLMCache()
        errors = []

        # Pre-populate some data
        for i in range(50):
            cache.set(f"existing_{i}", {"value": i})

        def reader_thread():
            for _ in range(100):
                for i in range(50):
                    key = f"existing_{i}"
                    value = cache.get(key)
                    if value and value != {"value": i}:
                        errors.append(f"Incorrect value for {key}: {value}")

        def writer_thread(thread_id):
            for i in range(100):
                cache.set(f"new_{thread_id}_{i}", {"thread": thread_id, "index": i})

        threads = []

        # Start reader threads
        for _ in range(3):
            thread = threading.Thread(target=reader_thread)
            threads.append(thread)
            thread.start()

        # Start writer threads
        for i in range(2):
            thread = threading.Thread(target=writer_thread, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert cache.size() == 250  # 50 existing + 2 threads * 100 writes

    def test_lookup_or_compute_thread_safety(self):
        """Test lookup_or_compute is thread-safe."""
        cache = LLMCache()
        compute_count = []

        def expensive_compute():
            compute_count.append(1)
            time.sleep(0.01)  # Simulate expensive operation
            return {"result": "computed"}

        def thread_function():
            result = cache.lookup_or_compute("shared_key", expensive_compute)
            assert result == {"result": "computed"}

        threads = []
        for _ in range(10):
            thread = threading.Thread(target=thread_function)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should only compute once despite multiple threads
        assert len(compute_count) == 1
        assert cache.get("shared_key") == {"result": "computed"}


class TestCacheStatistics:
    """Test cache statistics tracking."""

    def test_statistics_initialization(self):
        """Test statistics initialize to zero."""
        stats = CacheStatistics()

        assert stats.total_requests == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.tokens_saved == 0
        assert stats.hit_rate == 0.0

    def test_statistics_tracking(self):
        """Test statistics are tracked correctly."""
        cache = LLMCache()

        # Cache miss
        cache.lookup_or_compute("key1", lambda: {"usage": {"total_tokens": 100}})
        stats = cache.get_statistics()
        assert stats.total_requests == 1
        assert stats.cache_hits == 0
        assert stats.cache_misses == 1
        assert stats.tokens_saved == 0

        # Cache hit
        cache.lookup_or_compute("key1", lambda: {"usage": {"total_tokens": 100}})
        stats = cache.get_statistics()
        assert stats.total_requests == 2
        assert stats.cache_hits == 1
        assert stats.cache_misses == 1
        assert stats.tokens_saved == 100
        assert stats.hit_rate == 0.5

    def test_estimated_savings_calculation(self):
        """Test cost savings calculation."""
        stats = CacheStatistics()
        stats.tokens_saved = 10000

        # Using approximate GPT-4 pricing
        savings = stats.estimated_cost_savings()
        assert savings > 0
        assert isinstance(savings, float)


class TestCachePersistence:
    """Test cache persistence functionality."""

    def test_save_cache_to_json(self):
        """Test saving cache to JSON file."""
        cache = LLMCache()

        # Add some test data
        cache.set("key1", {"response": "value1", "usage": {"total_tokens": 100}})
        cache.set("key2", {"response": "value2", "usage": {"total_tokens": 200}})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            cache.save_to_disk(temp_path)

            # Verify file exists and contains correct data
            assert temp_path.exists()

            with open(temp_path, "r") as f:
                saved_data = json.load(f)

            assert "cache" in saved_data
            assert "statistics" in saved_data
            assert len(saved_data["cache"]) == 2
            assert "key1" in saved_data["cache"]
            assert saved_data["cache"]["key1"] == {
                "response": "value1",
                "usage": {"total_tokens": 100},
            }
        finally:
            temp_path.unlink(missing_ok=True)

    def test_load_cache_from_json(self):
        """Test loading cache from JSON file."""
        # Create test data
        test_data = {
            "cache": {
                "key1": {"response": "value1", "usage": {"total_tokens": 100}},
                "key2": {"response": "value2", "usage": {"total_tokens": 200}},
            },
            "statistics": {
                "total_requests": 10,
                "cache_hits": 5,
                "cache_misses": 5,
                "tokens_saved": 500,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = Path(f.name)

        try:
            cache = LLMCache()
            cache.load_from_disk(temp_path)

            # Verify cache contents
            assert cache.size() == 2
            assert cache.get("key1") == {"response": "value1", "usage": {"total_tokens": 100}}
            assert cache.get("key2") == {"response": "value2", "usage": {"total_tokens": 200}}

            # Verify statistics
            stats = cache.get_statistics()
            assert stats.total_requests == 10
            assert stats.cache_hits == 5
            assert stats.cache_misses == 5
            assert stats.tokens_saved == 500
        finally:
            temp_path.unlink(missing_ok=True)

    def test_save_and_load_compressed(self):
        """Test saving and loading compressed cache files."""
        cache = LLMCache()

        # Add a lot of test data to make compression worthwhile
        for i in range(100):
            cache.set(
                f"key_{i}",
                {
                    "response": f"This is a longer response value {i} " * 10,
                    "usage": {"total_tokens": 100 + i},
                },
            )

        with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as f:
            temp_path = Path(f.name)

        try:
            cache.save_to_disk(temp_path, compress=True)

            # Verify file is compressed
            assert temp_path.exists()

            # Check that it's actually gzipped
            with gzip.open(temp_path, "rt") as f:
                saved_data = json.load(f)

            assert len(saved_data["cache"]) == 100

            # Load into new cache
            new_cache = LLMCache()
            new_cache.load_from_disk(temp_path)

            assert new_cache.size() == 100
            assert new_cache.get("key_50")["response"].startswith(
                "This is a longer response value 50"
            )
        finally:
            temp_path.unlink(missing_ok=True)

    def test_atomic_write_operations(self):
        """Test atomic file writes to prevent corruption."""
        cache = LLMCache()
        cache.set("key1", {"response": "value1"})

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.json"

            # Mock a write failure partway through
            original_open = open
            write_count = [0]

            def mock_open_func(path, *args, **kwargs):
                if "w" in str(args) and str(path).endswith(".tmp"):
                    write_count[0] += 1
                    if write_count[0] == 1:
                        # Let the first write attempt fail
                        raise IOError("Simulated write failure")
                return original_open(path, *args, **kwargs)

            with patch("builtins.open", side_effect=mock_open_func):
                # First attempt should fail but not corrupt existing file
                with pytest.raises(IOError):
                    cache.save_to_disk(cache_path)

            # File should not exist since write failed
            assert not cache_path.exists()

            # Second attempt should succeed
            cache.save_to_disk(cache_path)
            assert cache_path.exists()

    def test_load_nonexistent_file(self):
        """Test loading from a non-existent file."""
        cache = LLMCache()
        non_existent_path = Path("/tmp/non_existent_cache.json")

        # Should not raise an error, just log a warning
        cache.load_from_disk(non_existent_path)

        # Cache should remain empty
        assert cache.size() == 0

    def test_load_corrupted_file(self):
        """Test loading from a corrupted JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json content")
            temp_path = Path(f.name)

        try:
            cache = LLMCache()

            # Should handle the error gracefully
            cache.load_from_disk(temp_path)

            # Cache should remain empty
            assert cache.size() == 0
        finally:
            temp_path.unlink(missing_ok=True)

    def test_cache_directory_creation(self):
        """Test that cache directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "subdir" / "cache.json"

            cache = LLMCache()
            cache.set("key1", {"response": "value1"})

            # Directory doesn't exist yet
            assert not cache_path.parent.exists()

            # Save should create the directory
            cache.save_to_disk(cache_path)

            assert cache_path.parent.exists()
            assert cache_path.exists()

    def test_initialize_with_cache_file(self):
        """Test initializing cache with existing cache file."""
        # Create a cache file
        test_data = {
            "cache": {"key1": {"response": "value1"}, "key2": {"response": "value2"}},
            "statistics": {
                "total_requests": 5,
                "cache_hits": 3,
                "cache_misses": 2,
                "tokens_saved": 150,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_path = Path(f.name)

        try:
            # Initialize cache with the file path
            cache = LLMCache(cache_file=temp_path)

            # Should auto-load the data
            assert cache.size() == 2
            assert cache.get("key1") == {"response": "value1"}

            # Statistics should be loaded
            stats = cache.get_statistics()
            assert stats.total_requests == 5
            assert stats.cache_hits == 3
        finally:
            temp_path.unlink(missing_ok=True)

    def test_save_on_context_exit(self):
        """Test that cache saves on context manager exit."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = Path(f.name)

        try:
            # Use cache as context manager
            with LLMCache(cache_file=temp_path) as cache:
                cache.set("key1", {"response": "value1"})
                cache.set("key2", {"response": "value2"})

            # Cache should be saved after context exit
            assert temp_path.exists()

            # Load and verify
            new_cache = LLMCache()
            new_cache.load_from_disk(temp_path)
            assert new_cache.size() == 2
            assert new_cache.get("key1") == {"response": "value1"}
        finally:
            temp_path.unlink(missing_ok=True)
