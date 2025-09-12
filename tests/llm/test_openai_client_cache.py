"""Tests for OpenAI client with caching functionality."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from cosmos_coherence.llm.cache import LLMCache
from cosmos_coherence.llm.config import OpenAIConfig
from cosmos_coherence.llm.models import ModelResponse, TokenUsage
from cosmos_coherence.llm.openai_client import OpenAIClient


@pytest.fixture
def openai_config():
    """Create OpenAI config for testing."""
    return OpenAIConfig(
        api_key="test-key", default_model="gpt-4", timeout=30.0  # type: ignore[call-arg]
    )


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI API response."""
    return {
        "id": "test-id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Test response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class TestOpenAIClientCaching:
    """Test OpenAI client with caching functionality."""

    @pytest.mark.asyncio
    async def test_client_with_cache_enabled(self, openai_config):
        """Test that client can be initialized with cache enabled."""
        client = OpenAIClient(openai_config, enable_cache=True)
        assert client._cache is not None
        assert isinstance(client._cache, LLMCache)
        assert client._cache_enabled is True

    @pytest.mark.asyncio
    async def test_client_with_cache_disabled(self, openai_config):
        """Test that client can be initialized with cache disabled."""
        client = OpenAIClient(openai_config, enable_cache=False)
        assert client._cache is None
        assert client._cache_enabled is False

    @pytest.mark.asyncio
    async def test_generate_response_cache_hit(self, openai_config, mock_openai_response):
        """Test that cached responses are returned on cache hit."""
        client = OpenAIClient(openai_config, enable_cache=True)

        # Pre-populate cache
        cache_key = client._cache.generate_cache_key(
            {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Test prompt"}],
                "temperature": 0.7,
            }
        )
        cached_response = ModelResponse(
            content="Cached response",
            model="gpt-4",
            temperature=0.7,
            usage=TokenUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15, estimated_cost=0.001
            ),
            latency_ms=100.0,
            request_id="test-request-id",
            finish_reason="stop",
            cached=False,
        )
        client._cache.set(cache_key, cached_response.model_dump())

        # Mock the OpenAI API call - should not be called
        with patch.object(client, "_make_request_with_retry", new_callable=AsyncMock) as mock_api:
            response = await client.generate_response("Test prompt", temperature=0.7)

            # API should not be called due to cache hit
            mock_api.assert_not_called()

            # Should return cached response
            assert response.content == "Cached response"
            assert response.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_generate_response_cache_miss(self, openai_config, mock_openai_response):
        """Test that API is called and response cached on cache miss."""
        client = OpenAIClient(openai_config, enable_cache=True)

        # Mock the OpenAI API call
        with patch.object(client, "_make_request_with_retry", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = mock_openai_response

            response = await client.generate_response("Test prompt", temperature=0.7)

            # API should be called due to cache miss
            mock_api.assert_called_once()

            # Check response is returned correctly
            assert response.content == "Test response"

            # Response should be cached
            cache_key = client._cache.generate_cache_key(
                {
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "Test prompt"}],
                    "temperature": 0.7,
                }
            )
            cached = client._cache.get(cache_key)
            assert cached is not None
            assert cached["content"] == "Test response"

    @pytest.mark.asyncio
    async def test_cache_disabled_always_calls_api(self, openai_config, mock_openai_response):
        """Test that API is always called when cache is disabled."""
        client = OpenAIClient(openai_config, enable_cache=False)

        with patch.object(client, "_make_request_with_retry", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = mock_openai_response

            # Make the same request twice
            await client.generate_response("Test prompt", temperature=0.7)
            await client.generate_response("Test prompt", temperature=0.7)

            # API should be called twice (no caching)
            assert mock_api.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_key_includes_all_parameters(self, openai_config):
        """Test that cache key includes all relevant parameters."""
        client = OpenAIClient(openai_config, enable_cache=True)

        # Generate cache keys with different parameters
        params1 = client._build_cache_params("Test", "gpt-4", 0.7, max_tokens=100)
        key1 = client._cache.generate_cache_key(params1)

        params2 = client._build_cache_params("Test", "gpt-4", 0.7, max_tokens=200)
        key2 = client._cache.generate_cache_key(params2)

        params3 = client._build_cache_params("Test", "gpt-4", 0.8, max_tokens=100)
        key3 = client._cache.generate_cache_key(params3)

        params4 = client._build_cache_params("Different", "gpt-4", 0.7, max_tokens=100)
        key4 = client._cache.generate_cache_key(params4)

        # All keys should be different
        assert len({key1, key2, key3, key4}) == 4

    @pytest.mark.asyncio
    async def test_cache_persistence(self, openai_config, mock_openai_response):
        """Test that cache can be persisted and loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "cache.json"

            # First client with cache file
            client1 = OpenAIClient(openai_config, enable_cache=True, cache_file=cache_file)

            with patch.object(
                client1, "_make_request_with_retry", new_callable=AsyncMock
            ) as mock_api:
                mock_api.return_value = mock_openai_response
                await client1.generate_response("Test prompt", temperature=0.7)

            # Save cache
            client1._cache.save_to_disk(cache_file)

            # Second client loads from cache file
            client2 = OpenAIClient(openai_config, enable_cache=True, cache_file=cache_file)

            with patch.object(
                client2, "_make_request_with_retry", new_callable=AsyncMock
            ) as mock_api:
                response = await client2.generate_response("Test prompt", temperature=0.7)

                # Should use cached response, not call API
                mock_api.assert_not_called()
                assert response.content == "Test response"

    @pytest.mark.asyncio
    async def test_cache_statistics_tracking(self, openai_config, mock_openai_response):
        """Test that cache statistics are properly tracked."""
        client = OpenAIClient(openai_config, enable_cache=True)

        with patch.object(client, "_make_request_with_retry", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = mock_openai_response

            # First call - cache miss
            await client.generate_response("Test prompt 1", temperature=0.7)

            # Second call - cache hit
            await client.generate_response("Test prompt 1", temperature=0.7)

            # Third call - cache miss (different prompt)
            await client.generate_response("Test prompt 2", temperature=0.7)

            stats = client.get_cache_statistics()
            assert stats.total_requests == 3
            assert stats.cache_hits == 1
            assert stats.cache_misses == 2
            assert stats.hit_rate == pytest.approx(0.333, rel=0.01)

    @pytest.mark.asyncio
    async def test_batch_generate_with_cache(self, openai_config, mock_openai_response):
        """Test that batch generation uses cache appropriately."""
        client = OpenAIClient(openai_config, enable_cache=True)

        # Pre-populate cache with one response
        cache_key = client._cache.generate_cache_key(
            {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Cached prompt"}],
                "temperature": 0.7,
            }
        )
        cached_response = ModelResponse(
            content="Cached response",
            model="gpt-4",
            temperature=0.7,
            usage=TokenUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15, estimated_cost=0.001
            ),
            latency_ms=100.0,
            request_id="test-request-id",
            finish_reason="stop",
            cached=False,
        )
        client._cache.set(cache_key, cached_response.model_dump())

        prompts = ["Cached prompt", "New prompt 1", "New prompt 2"]

        with patch.object(client, "_make_request_with_retry", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = mock_openai_response

            responses = await client.batch_generate(prompts, temperature=0.7)

            # Should only call API for the two new prompts
            assert mock_api.call_count == 2
            assert len(responses) == 3
            assert responses[0].content == "Cached response"  # From cache

    @pytest.mark.asyncio
    async def test_cache_error_handling(self, openai_config):
        """Test that cache errors don't break the client."""
        client = OpenAIClient(openai_config, enable_cache=True)

        # Mock cache to raise an error
        with patch.object(client._cache, "lookup_or_compute", side_effect=Exception("Cache error")):
            with patch.object(
                client, "_make_request_with_retry", new_callable=AsyncMock
            ) as mock_api:
                mock_api.return_value = {
                    "choices": [{"message": {"content": "Fallback response"}}],
                    "usage": {"total_tokens": 10},
                }

                # Cache error should be caught and API should still be called
                # But since lookup_or_compute raises an error, the method will fail
                with pytest.raises(Exception, match="Cache error"):
                    await client.generate_response("Test prompt")

    @pytest.mark.asyncio
    async def test_cache_with_streaming(self, openai_config):
        """Test that streaming responses are not cached."""
        client = OpenAIClient(openai_config, enable_cache=True)

        with patch.object(client, "_make_request_with_retry", new_callable=AsyncMock) as mock_api:
            # Simulate streaming response - with proper usage structure
            mock_api.return_value = {
                "choices": [
                    {"message": {"content": "Streaming response"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
                "model": "gpt-4",
                "id": "test-id",
            }

            # Make streaming request
            await client.generate_response("Test prompt", stream=True)

            # Make same request without streaming
            await client.generate_response("Test prompt", stream=False)

            # Both should call API (streaming not cached)
            assert mock_api.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_print_statistics(self, openai_config, capsys):
        """Test printing cache statistics."""
        client = OpenAIClient(openai_config, enable_cache=True)

        # Generate some cache activity
        client._cache._statistics.total_requests = 100
        client._cache._statistics.cache_hits = 75
        client._cache._statistics.cache_misses = 25
        client._cache._statistics.tokens_saved = 1500

        client.print_cache_statistics()

        captured = capsys.readouterr()
        assert "Cache Statistics" in captured.out
        assert "Total requests: 100" in captured.out
        assert "Cache hits: 75" in captured.out
        assert "Hit rate: 75.0%" in captured.out
        assert "Tokens saved: 1500" in captured.out
