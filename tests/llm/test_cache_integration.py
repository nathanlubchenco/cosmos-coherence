"""Integration tests for LLM caching functionality."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from cosmos_coherence.benchmarks.models.base import BenchmarkRunConfig, BenchmarkType
from cosmos_coherence.llm.config import OpenAIConfig
from cosmos_coherence.llm.openai_client import OpenAIClient


class TestCacheIntegration:
    """Test cache integration with configuration and environment variables."""

    def test_benchmark_config_cache_settings(self):
        """Test that BenchmarkRunConfig includes cache settings."""
        config = BenchmarkRunConfig(
            benchmark_type=BenchmarkType.FAITHBENCH,
            dataset_path=Path("/tmp/test"),
            use_cache=False,
            cache_persist=True,
            cache_dir=Path("/tmp/cache"),
        )

        assert config.use_cache is False
        assert config.cache_persist is True
        assert config.cache_dir == Path("/tmp/cache")

    def test_default_cache_settings(self):
        """Test default cache settings in BenchmarkRunConfig."""
        config = BenchmarkRunConfig(
            benchmark_type=BenchmarkType.FAITHBENCH,
            dataset_path=Path("/tmp/test"),
        )

        assert config.use_cache is True  # Default enabled
        assert config.cache_persist is True  # Default enabled
        assert config.cache_dir is None  # Default to None

    @pytest.mark.asyncio
    async def test_environment_variable_disable_cache(self):
        """Test that COSMOS_DISABLE_CACHE environment variable works."""
        # Set environment variable
        os.environ["COSMOS_DISABLE_CACHE"] = "1"

        try:
            openai_config = OpenAIConfig(
                api_key="test-key", default_model="gpt-4", timeout=30.0  # type: ignore[call-arg]
            )
            client = OpenAIClient(openai_config, enable_cache=True)

            # Cache should be disabled despite enable_cache=True
            assert client._cache_enabled is False
            assert client._cache is None

        finally:
            # Clean up environment variable
            del os.environ["COSMOS_DISABLE_CACHE"]

    @pytest.mark.asyncio
    async def test_environment_variable_cache_dir(self):
        """Test that COSMOS_CACHE_DIR environment variable works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "test_cache"
            os.environ["COSMOS_CACHE_DIR"] = str(cache_dir)

            try:
                openai_config = OpenAIConfig(
                    api_key="test-key", default_model="gpt-4", timeout=30.0  # type: ignore[call-arg]
                )
                client = OpenAIClient(openai_config, enable_cache=True)

                # Cache should be enabled with the specified directory
                assert client._cache_enabled is True
                assert client._cache is not None
                assert client._cache._cache_file == cache_dir

            finally:
                # Clean up environment variable
                del os.environ["COSMOS_CACHE_DIR"]

    @pytest.mark.asyncio
    async def test_cache_persistence_integration(self):
        """Test that cache persists across client instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "cache.json"

            openai_config = OpenAIConfig(
                api_key="test-key", default_model="gpt-4", timeout=30.0  # type: ignore[call-arg]
            )

            # First client - make a request
            client1 = OpenAIClient(openai_config, enable_cache=True, cache_file=cache_file)

            with patch.object(
                client1, "_make_request_with_retry", new_callable=AsyncMock
            ) as mock_api:
                mock_api.return_value = {
                    "choices": [{"message": {"content": "Test response"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }

                with patch.object(client1, "_parse_response") as mock_parse:
                    from cosmos_coherence.llm.models import ModelResponse, TokenUsage

                    mock_parse.return_value = ModelResponse(
                        content="Test response",
                        model="gpt-4",
                        temperature=0.7,
                        usage=TokenUsage(
                            prompt_tokens=10,
                            completion_tokens=5,
                            total_tokens=15,
                            estimated_cost=0.001,
                        ),
                        latency_ms=100.0,
                        request_id="test-id",
                        finish_reason="stop",
                        cached=False,
                    )

                    response1 = await client1.generate_response("Test prompt", temperature=0.7)
                    assert response1.content == "Test response"

            # Save cache
            if client1._cache:
                client1._cache.save_to_disk(cache_file)

            # Second client - should use cached response
            client2 = OpenAIClient(openai_config, enable_cache=True, cache_file=cache_file)

            with patch.object(
                client2, "_make_request_with_retry", new_callable=AsyncMock
            ) as mock_api2:
                response2 = await client2.generate_response("Test prompt", temperature=0.7)

                # Should not call API - uses cache
                mock_api2.assert_not_called()
                assert response2.content == "Test response"

                # Check cache statistics
                # Note: stats include the loaded history from client1
                stats = client2.get_cache_statistics()
                assert stats.total_requests == 2  # 1 from client1, 1 from client2
                assert stats.cache_hits == 1  # client2's request was a hit
                assert stats.cache_misses == 1  # client1's request was a miss
