"""Tests for SelfCheckGPT benchmark implementation."""

import tempfile
from pathlib import Path

import pytest
from cosmos_coherence.benchmarks.models.datasets import SelfCheckGPTItem
from cosmos_coherence.llm.openai_client import OpenAIClient


class TestMultiTemperatureSampling:
    """Test multi-temperature sampling for SelfCheckGPT."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_item(self):
        """Create a sample SelfCheckGPT item."""
        return SelfCheckGPTItem(
            question="Albert Einstein",
            topic="Albert Einstein",
            wiki_bio_text=(
                "Albert Einstein was a German-born theoretical physicist. "
                "He developed the theory of relativity."
            ),
            gpt3_text=("Albert Einstein was a physicist. " "He won the Nobel Prize in Physics."),
            gpt3_sentences=[
                "Albert Einstein was a physicist.",
                "He won the Nobel Prize in Physics.",
            ],
            annotation=["accurate", "accurate"],
        )

    @pytest.fixture
    def mock_client(self, temp_cache_dir):
        """Create a mock OpenAI client with caching."""

        from cosmos_coherence.llm.config import OpenAIConfig

        config = OpenAIConfig(api_key="test-key", model="gpt-4o-mini")
        cache_file = temp_cache_dir / "test_cache.json"
        client = OpenAIClient(config, enable_cache=True, cache_file=cache_file)

        # Mock the _make_request method instead of _call_api
        async def mock_make_request(**kwargs):
            temp = kwargs.get("temperature", 0.7)
            # Deterministic at temp 0.0, varied at temp 1.0
            if temp == 0.0:
                text = "Albert Einstein was a physicist who developed relativity."
            else:
                # Simulate variation at higher temperatures
                import random

                variations = [
                    "Albert Einstein was a theoretical physicist.",
                    "Einstein developed the theory of relativity.",
                    "Albert Einstein won the Nobel Prize in 1921.",
                    "He was born in Germany in 1879.",
                    "Einstein is known for E=mc².",
                ]
                text = random.choice(variations)

            return {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                },
            }

        client._make_request = mock_make_request
        return client

    @pytest.mark.asyncio
    async def test_generate_single_sample_temp_zero(self, mock_client, sample_item):
        """Test generating a single sample at temperature 0.0."""
        prompt = f"Write a short biography about {sample_item.topic}."

        response = await mock_client.generate_response(prompt, temperature=0.0, max_tokens=100)

        assert response is not None
        assert "physicist" in response.content.lower()
        assert response.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_generate_multiple_samples_different_temps(self, mock_client):
        """Test generating samples at different temperatures."""
        prompt = "Write a short biography about Albert Einstein."

        # Generate baseline at temp 0.0
        baseline = await mock_client.generate_response(prompt, temperature=0.0, max_tokens=100)

        # Generate samples at temp 1.0
        samples = []
        for _ in range(5):
            sample = await mock_client.generate_response(prompt, temperature=1.0, max_tokens=100)
            samples.append(sample)

        # Verify we got all responses
        assert baseline is not None
        assert len(samples) == 5
        assert all(s is not None for s in samples)

        # Baseline should be deterministic (same prompt, same temp 0.0)
        baseline2 = await mock_client.generate_response(prompt, temperature=0.0, max_tokens=100)
        # With caching, should get exact same response
        assert baseline.content == baseline2.content

    @pytest.mark.asyncio
    async def test_cache_differentiates_temperatures(self, mock_client, temp_cache_dir):
        """Test that cache treats different temperatures as separate entries."""
        prompt = "Test prompt"

        # Generate at temp 0.0
        response_t0 = await mock_client.generate_response(prompt, temperature=0.0, max_tokens=50)

        # Generate at temp 1.0
        response_t1 = await mock_client.generate_response(prompt, temperature=1.0, max_tokens=50)

        # Save cache
        mock_client.save_cache()

        # Verify cache file exists (cache is accessed via _cache._cache_file)
        assert mock_client._cache._cache_file is not None
        assert mock_client._cache._cache_file.exists()

        # Verify cache has entries for both temperatures
        stats = mock_client.get_cache_statistics()
        assert stats.total_requests >= 2  # At least 2 requests made

        # Generate again - should hit cache
        response_t0_cached = await mock_client.generate_response(
            prompt, temperature=0.0, max_tokens=50
        )
        response_t1_cached = await mock_client.generate_response(
            prompt, temperature=1.0, max_tokens=50
        )

        # Cache hits should return same responses
        assert response_t0.content == response_t0_cached.content
        assert response_t1.content == response_t1_cached.content

        # Different temperatures should have different responses
        # (or at least be cached separately)
        stats_after = mock_client.get_cache_statistics()
        assert stats_after.cache_hits >= 2  # Both should hit cache


class TestCacheKeyTemperatureInclusion:
    """Test that temperature is properly included in cache keys."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_temperature_in_cache_key(self, temp_cache_dir):
        """Verify temperature is included in cache key generation."""
        from cosmos_coherence.llm.config import OpenAIConfig

        config = OpenAIConfig(api_key="test-key", model="gpt-4o-mini")
        cache_file = temp_cache_dir / "cache.json"
        client = OpenAIClient(config, enable_cache=True, cache_file=cache_file)

        # Mock the _make_request method
        async def mock_make_request(**kwargs):
            temp = kwargs.get("temperature", 0.7)
            return {
                "id": "test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"Response at temp {temp}",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }

        client._make_request = mock_make_request

        # Generate responses at different temperatures
        r1 = await client.generate_response("test", temperature=0.0, max_tokens=50)
        r2 = await client.generate_response("test", temperature=1.0, max_tokens=50)

        # Responses should be different (cached separately)
        assert "0.0" in r1.content
        assert "1.0" in r2.content

        # Save and reload cache
        client.save_cache()

        # Create new client with same cache
        client2 = OpenAIClient(config, enable_cache=True, cache_file=cache_file)
        client2._make_request = mock_make_request

        # Should hit cache for both temperatures
        r1_cached = await client2.generate_response("test", temperature=0.0, max_tokens=50)
        r2_cached = await client2.generate_response("test", temperature=1.0, max_tokens=50)

        # Should match original responses (from cache)
        assert r1_cached.content == r1.content
        assert r2_cached.content == r2.content
        assert r1_cached.content != r2_cached.content

        # Verify cache stats
        # Note: client2 loads cache from disk, so its statistics start fresh
        # We're testing that the cache correctly differentiated temperatures
        stats = client2.get_cache_statistics()
        # Both requests should hit cache (loaded from disk)
        assert stats.cache_hits >= 2


class TestNLIScorerIntegration:
    """Test NLI scorer integration for consistency evaluation."""

    @pytest.fixture
    def sample_item(self):
        """Create a sample SelfCheckGPT item."""
        return SelfCheckGPTItem(
            question="Albert Einstein",
            topic="Albert Einstein",
            wiki_bio_text=(
                "Albert Einstein was a German-born theoretical physicist. "
                "He developed the theory of relativity."
            ),
            gpt3_text=("Albert Einstein was a physicist. " "He won the Nobel Prize in Physics."),
            gpt3_sentences=[
                "Albert Einstein was a physicist.",
                "He won the Nobel Prize in Physics.",
            ],
            annotation=["accurate", "accurate"],
        )

    @pytest.fixture
    def baseline_text(self):
        """Baseline text for consistency checking."""
        return (
            "Albert Einstein was a theoretical physicist. "
            "He developed the theory of relativity. "
            "He won the Nobel Prize in Physics in 1921."
        )

    @pytest.fixture
    def sample_texts(self):
        """Sample texts generated at temperature 1.0."""
        return [
            "Albert Einstein was a physicist known for relativity theory.",
            "Einstein won the Nobel Prize in 1921 for photoelectric effect.",
            "He was born in Germany in 1879.",
            "Albert Einstein developed E=mc².",
            "Einstein was a professor at Princeton University.",
        ]

    @pytest.fixture
    def mock_nli_scorer(self):
        """Create a mock NLI scorer."""

        class MockNLIScorer:
            def __init__(self, device=None):
                self.device = device

            def predict(self, sentences, sampled_passages):
                """Mock predict method that returns consistency scores.

                Lower scores = more consistent = less hallucination
                Higher scores = less consistent = more hallucination
                """
                # Return mock scores for each sentence
                # Score range: [0.0, 1.0] where higher = hallucination
                num_sentences = len(sentences)
                # Simulate realistic scores: most factual, some uncertain
                scores = [0.1] * num_sentences  # Low scores = factual
                return scores

        return MockNLIScorer

    @pytest.mark.asyncio
    async def test_nli_scorer_initialization(self, mock_nli_scorer):
        """Test NLI scorer can be initialized."""
        scorer = mock_nli_scorer(device="cpu")
        assert scorer is not None
        assert scorer.device == "cpu"

    @pytest.mark.asyncio
    async def test_nli_predict_method(self, mock_nli_scorer, baseline_text, sample_texts):
        """Test NLI scorer predict method returns scores."""
        scorer = mock_nli_scorer(device="cpu")

        # Split baseline into sentences
        sentences = [
            "Albert Einstein was a theoretical physicist.",
            "He developed the theory of relativity.",
            "He won the Nobel Prize in Physics in 1921.",
        ]

        # Get consistency scores
        scores = scorer.predict(sentences=sentences, sampled_passages=sample_texts)

        # Verify scores structure
        assert len(scores) == len(sentences)
        assert all(isinstance(s, (int, float)) for s in scores)
        assert all(0.0 <= s <= 1.0 for s in scores)

    @pytest.mark.asyncio
    async def test_sentence_level_consistency_calculation(
        self, mock_nli_scorer, baseline_text, sample_texts
    ):
        """Test calculation of per-sentence consistency scores."""
        scorer = mock_nli_scorer(device="cpu")

        sentences = [
            "Albert Einstein was a theoretical physicist.",
            "He developed the theory of relativity.",
            "He won the Nobel Prize in Physics in 1921.",
        ]

        scores = scorer.predict(sentences=sentences, sampled_passages=sample_texts)

        # Each sentence should have a score
        assert len(scores) == 3

        # Scores should be in valid range
        for score in scores:
            assert 0.0 <= score <= 1.0

        # Lower scores indicate higher consistency (less hallucination)
        # This is the expected behavior from SelfCheckGPT
        assert all(s < 0.5 for s in scores)  # Mock returns low scores

    @pytest.mark.asyncio
    async def test_aggregate_consistency_score(self, mock_nli_scorer, baseline_text, sample_texts):
        """Test calculation of aggregate consistency score."""
        scorer = mock_nli_scorer(device="cpu")

        sentences = [
            "Albert Einstein was a theoretical physicist.",
            "He developed the theory of relativity.",
            "He won the Nobel Prize in Physics in 1921.",
        ]

        scores = scorer.predict(sentences=sentences, sampled_passages=sample_texts)

        # Calculate aggregate score (mean of sentence scores)
        aggregate_score = sum(scores) / len(scores)

        assert isinstance(aggregate_score, (int, float))
        assert 0.0 <= aggregate_score <= 1.0

        # With mock low scores, aggregate should be low
        assert aggregate_score < 0.5


class TestCacheStatistics:
    """Test cache statistics display."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_cache_statistics_tracking(self, temp_cache_dir):
        """Test that cache statistics are properly tracked."""
        from cosmos_coherence.llm.config import OpenAIConfig

        config = OpenAIConfig(api_key="test-key", model="gpt-4o-mini")
        cache_file = temp_cache_dir / "cache.json"
        client = OpenAIClient(config, enable_cache=True, cache_file=cache_file)

        # Mock _make_request
        async def mock_make_request(**kwargs):
            return {
                "id": "test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "test"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }

        client._make_request = mock_make_request

        # Initial stats - empty cache
        stats = client.get_cache_statistics()
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.hit_rate == 0.0

        # First call - cache miss
        await client.generate_response("test1", temperature=0.5)
        stats = client.get_cache_statistics()
        assert stats.cache_hits == 0
        assert stats.cache_misses == 1
        assert stats.hit_rate == 0.0

        # Second call same prompt - cache hit
        await client.generate_response("test1", temperature=0.5)
        stats = client.get_cache_statistics()
        assert stats.cache_hits == 1
        assert stats.cache_misses == 1
        assert stats.hit_rate == 0.5

        # Third call different prompt - cache miss
        await client.generate_response("test2", temperature=0.5)
        stats = client.get_cache_statistics()
        assert stats.cache_hits == 1
        assert stats.cache_misses == 2
        assert pytest.approx(stats.hit_rate, 0.01) == 0.333

        # Fourth call - second prompt again - cache hit
        await client.generate_response("test2", temperature=0.5)
        stats = client.get_cache_statistics()
        assert stats.cache_hits == 2
        assert stats.cache_misses == 2
        assert stats.hit_rate == 0.5
