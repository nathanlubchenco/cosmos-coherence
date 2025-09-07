"""Tests for OpenAI client configuration and initialization."""

import os
from datetime import datetime
from unittest.mock import patch

import pytest
from cosmos_coherence.llm.config import (
    BatchConfig,
    OpenAIConfig,
    RateLimitConfig,
    RetryConfig,
)
from cosmos_coherence.llm.exceptions import (
    APIError,
    PartialFailureError,
    RateLimitError,
)
from cosmos_coherence.llm.models import (
    BatchJob,
    BatchJobStatus,
    BatchRequest,
    ModelResponse,
    TokenUsage,
)
from cosmos_coherence.llm.openai_client import OpenAIClient
from pydantic import ValidationError


class TestOpenAIConfig:
    """Test OpenAI configuration models."""

    def test_openai_config_with_defaults(self):
        """Test OpenAIConfig with default values."""
        config = OpenAIConfig(api_key="test-key")
        assert config.api_key == "test-key"
        assert config.organization_id is None
        assert config.base_url == "https://api.openai.com/v1"
        assert config.default_model == "gpt-4o-mini"
        assert config.timeout == 30.0
        assert config.max_retries == 3

    def test_openai_config_custom_values(self):
        """Test OpenAIConfig with custom values."""
        config = OpenAIConfig(
            api_key="test-key",
            organization_id="org-123",
            base_url="https://custom.openai.com",
            default_model="gpt-4",
            timeout=60.0,
            max_retries=5,
        )
        assert config.organization_id == "org-123"
        assert config.base_url == "https://custom.openai.com"
        assert config.default_model == "gpt-4"
        assert config.timeout == 60.0
        assert config.max_retries == 5

    def test_openai_config_env_loading(self):
        """Test OpenAIConfig loading from environment variables."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "env-key",
                "OPENAI_ORGANIZATION_ID": "env-org",
                "OPENAI_BASE_URL": "https://env.openai.com",
                "OPENAI_DEFAULT_MODEL": "gpt-4-turbo",
            },
            clear=True,
        ):
            config = OpenAIConfig()
            assert config.api_key == "env-key"
            assert config.organization_id == "env-org"
            assert config.base_url == "https://env.openai.com"
            assert config.default_model == "gpt-4-turbo"

    def test_rate_limit_config_defaults(self):
        """Test RateLimitConfig with default values."""
        config = RateLimitConfig()
        assert config.requests_per_minute == 500
        assert config.tokens_per_minute == 90000
        assert config.max_concurrent == 10
        assert config.adaptive_throttling is True

    def test_rate_limit_config_custom(self):
        """Test RateLimitConfig with custom values."""
        config = RateLimitConfig(
            requests_per_minute=1000,
            tokens_per_minute=150000,
            max_concurrent=20,
            adaptive_throttling=False,
        )
        assert config.requests_per_minute == 1000
        assert config.tokens_per_minute == 150000
        assert config.max_concurrent == 20
        assert config.adaptive_throttling is False

    def test_batch_config_defaults(self):
        """Test BatchConfig with default values."""
        config = BatchConfig()
        assert config.auto_batch_threshold == 100
        assert config.polling_interval == 60.0
        assert config.completion_window == "24h"
        assert config.max_batch_size == 50000

    def test_batch_config_validation(self):
        """Test BatchConfig validation for completion window."""
        with pytest.raises(ValidationError):
            BatchConfig(completion_window="invalid")

        # Valid windows
        config = BatchConfig(completion_window="24h")
        assert config.completion_window == "24h"
        config = BatchConfig(completion_window="7d")
        assert config.completion_window == "7d"

    def test_retry_config_defaults(self):
        """Test RetryConfig with default values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True


class TestOpenAIClient:
    """Test OpenAI client initialization and core functionality."""

    @pytest.fixture
    def mock_openai_config(self):
        """Create a mock OpenAI configuration."""
        return OpenAIConfig(api_key="test-key")

    @pytest.fixture
    def mock_rate_limit_config(self):
        """Create a mock rate limit configuration."""
        return RateLimitConfig()

    @pytest.fixture
    def mock_batch_config(self):
        """Create a mock batch configuration."""
        return BatchConfig()

    @pytest.fixture
    def mock_retry_config(self):
        """Create a mock retry configuration."""
        return RetryConfig()

    @pytest.mark.asyncio
    async def test_client_initialization(self, mock_openai_config):
        """Test OpenAI client initialization."""
        client = OpenAIClient(
            openai_config=mock_openai_config,
        )
        assert client.openai_config == mock_openai_config
        assert client.rate_limit_config is not None
        assert client.batch_config is not None
        assert client.retry_config is not None

    @pytest.mark.asyncio
    async def test_client_initialization_with_all_configs(
        self,
        mock_openai_config,
        mock_rate_limit_config,
        mock_batch_config,
        mock_retry_config,
    ):
        """Test OpenAI client initialization with all configurations."""
        client = OpenAIClient(
            openai_config=mock_openai_config,
            rate_limit_config=mock_rate_limit_config,
            batch_config=mock_batch_config,
            retry_config=mock_retry_config,
        )
        assert client.openai_config == mock_openai_config
        assert client.rate_limit_config == mock_rate_limit_config
        assert client.batch_config == mock_batch_config
        assert client.retry_config == mock_retry_config

    @pytest.mark.asyncio
    async def test_generate_response_success(self, mock_openai_config):
        """Test successful response generation."""
        client = OpenAIClient(openai_config=mock_openai_config)

        with patch.object(client, "_make_request") as mock_request:
            mock_request.return_value = {
                "choices": [{"message": {"content": "Test response"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "model": "gpt-3.5-turbo",
                "id": "test-id",
            }

            response = await client.generate_response(
                prompt="Test prompt",
                temperature=0.7,
            )

            assert isinstance(response, ModelResponse)
            assert response.content == "Test response"
            assert response.usage.total_tokens == 15
            assert response.model == "gpt-3.5-turbo"
            assert response.request_id == "test-id"
            assert response.temperature == 0.7

    @pytest.mark.asyncio
    async def test_generate_response_with_retry(self, mock_openai_config):
        """Test response generation with retry on transient errors."""
        client = OpenAIClient(openai_config=mock_openai_config)

        with patch.object(client, "_make_request") as mock_request:
            # First call fails, second succeeds
            mock_request.side_effect = [
                RateLimitError("Rate limit exceeded"),
                {
                    "choices": [{"message": {"content": "Success after retry"}}],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 6,
                        "total_tokens": 16,
                    },
                    "model": "gpt-3.5-turbo",
                    "id": "retry-id",
                },
            ]

            with patch("asyncio.sleep"):  # Skip actual sleep in tests
                response = await client.generate_response("Test prompt")

            assert response.content == "Success after retry"
            assert mock_request.call_count == 2

    @pytest.mark.asyncio
    async def test_batch_generate_success(self, mock_openai_config):
        """Test batch generation of responses."""
        client = OpenAIClient(openai_config=mock_openai_config)
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]

        with patch.object(client, "generate_response") as mock_generate:
            mock_generate.side_effect = [
                ModelResponse(
                    content=f"Response {i}",
                    usage=TokenUsage(
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                        estimated_cost=0.001,
                    ),
                    model="gpt-3.5-turbo",
                    request_id=f"id-{i}",
                    latency_ms=100.0,
                    temperature=0.7,
                    finish_reason="stop",
                )
                for i in range(1, 4)
            ]

            responses = await client.batch_generate(
                prompts,
                temperature=0.7,
                max_concurrent=2,
            )

            assert len(responses) == 3
            assert responses[0].content == "Response 1"
            assert responses[2].content == "Response 3"
            assert mock_generate.call_count == 3

    @pytest.mark.asyncio
    async def test_batch_generate_with_progress_callback(self, mock_openai_config):
        """Test batch generation with progress callback."""
        client = OpenAIClient(openai_config=mock_openai_config)
        prompts = ["Prompt 1", "Prompt 2"]
        progress_updates = []

        def progress_callback(progress):
            progress_updates.append(progress)

        with patch.object(client, "generate_response") as mock_generate:
            mock_generate.side_effect = [
                ModelResponse(
                    content=f"Response {i}",
                    usage=TokenUsage(
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                        estimated_cost=0.001,
                    ),
                    model="gpt-3.5-turbo",
                    request_id=f"id-{i}",
                    latency_ms=100.0,
                    temperature=0.7,
                    finish_reason="stop",
                )
                for i in range(1, 3)
            ]

            await client.batch_generate(
                prompts,
                progress_callback=progress_callback,
            )

            assert len(progress_updates) > 0
            assert progress_updates[-1] == 1.0  # 100% complete

    @pytest.mark.asyncio
    async def test_batch_generate_partial_failure(self, mock_openai_config):
        """Test batch generation with partial failures."""
        client = OpenAIClient(openai_config=mock_openai_config)
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]

        with patch.object(client, "generate_response") as mock_generate:
            mock_generate.side_effect = [
                ModelResponse(
                    content="Response 1",
                    usage=TokenUsage(
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                        estimated_cost=0.001,
                    ),
                    model="gpt-3.5-turbo",
                    request_id="id-1",
                    latency_ms=100.0,
                    temperature=0.7,
                    finish_reason="stop",
                ),
                APIError("API failure"),
                ModelResponse(
                    content="Response 3",
                    usage=TokenUsage(
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                        estimated_cost=0.001,
                    ),
                    model="gpt-3.5-turbo",
                    request_id="id-3",
                    latency_ms=100.0,
                    temperature=0.7,
                    finish_reason="stop",
                ),
            ]

            with pytest.raises(PartialFailureError) as exc_info:
                await client.batch_generate(prompts)

            error = exc_info.value
            assert len(error.successful_responses) == 2
            assert len(error.failed_indices) == 1
            assert 1 in error.failed_indices

    @pytest.mark.asyncio
    async def test_submit_batch_job(self, mock_openai_config):
        """Test batch job submission."""
        client = OpenAIClient(openai_config=mock_openai_config)
        requests = [
            BatchRequest(prompt="Test 1", temperature=0.5),
            BatchRequest(prompt="Test 2", temperature=0.7),
        ]

        with patch.object(client, "_submit_batch_to_api") as mock_submit:
            mock_submit.return_value = {
                "id": "batch-123",
                "status": "validating",
                "created_at": "2024-01-01T00:00:00Z",
                "request_counts": {"total": 2},
            }

            job = await client.submit_batch_job(requests)

            assert isinstance(job, BatchJob)
            assert job.job_id == "batch-123"
            assert job.status == "validating"
            assert job.request_count == 2

    @pytest.mark.asyncio
    async def test_get_batch_status(self, mock_openai_config):
        """Test getting batch job status."""
        client = OpenAIClient(openai_config=mock_openai_config)

        with patch.object(client, "_get_batch_status_from_api") as mock_status:
            mock_status.return_value = {
                "id": "batch-123",
                "status": "completed",
                "created_at": "2024-01-01T00:00:00Z",
                "completed_at": "2024-01-01T01:00:00Z",
                "request_counts": {
                    "total": 100,
                    "completed": 98,
                    "failed": 2,
                },
                "errors": [],
            }

            status = await client.get_batch_status("batch-123")

            assert isinstance(status, BatchJobStatus)
            assert status.job_id == "batch-123"
            assert status.status == "completed"
            assert status.completed_count == 98
            assert status.failed_count == 2

    @pytest.mark.asyncio
    async def test_retrieve_batch_results(self, mock_openai_config):
        """Test retrieving batch job results."""
        client = OpenAIClient(openai_config=mock_openai_config)

        with patch.object(client, "_retrieve_batch_results_from_api") as mock_retrieve:
            mock_retrieve.return_value = [
                {
                    "custom_id": "req-1",
                    "response": {
                        "body": {
                            "choices": [{"message": {"content": "Result 1"}}],
                            "usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 5,
                                "total_tokens": 15,
                            },
                            "model": "gpt-3.5-turbo",
                            "id": "result-1",
                        }
                    },
                },
                {
                    "custom_id": "req-2",
                    "response": {
                        "body": {
                            "choices": [{"message": {"content": "Result 2"}}],
                            "usage": {
                                "prompt_tokens": 12,
                                "completion_tokens": 6,
                                "total_tokens": 18,
                            },
                            "model": "gpt-3.5-turbo",
                            "id": "result-2",
                        }
                    },
                },
            ]

            results = await client.retrieve_batch_results("batch-123")

            assert len(results) == 2
            assert results[0].content == "Result 1"
            assert results[1].content == "Result 2"
            assert results[0].usage.total_tokens == 15
            assert results[1].usage.total_tokens == 18

    @pytest.mark.asyncio
    async def test_rate_limiting(self, mock_openai_config):
        """Test that rate limiting is enforced."""
        config = RateLimitConfig(
            requests_per_minute=60,
            max_concurrent=2,
        )
        client = OpenAIClient(
            openai_config=mock_openai_config,
            rate_limit_config=config,
        )

        # Test that concurrent requests are limited
        assert client.rate_limit_config.max_concurrent == 2
        assert client._semaphore._value == 2  # Check semaphore limit

    @pytest.mark.asyncio
    async def test_token_counting(self, mock_openai_config):
        """Test token counting functionality."""
        client = OpenAIClient(openai_config=mock_openai_config)

        # Test token counting for a simple prompt
        token_count = client.count_tokens("Hello, world!", model="gpt-3.5-turbo")
        assert token_count > 0
        assert isinstance(token_count, int)

    @pytest.mark.asyncio
    async def test_cost_estimation(self, mock_openai_config):
        """Test cost estimation for token usage."""
        client = OpenAIClient(openai_config=mock_openai_config)

        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            estimated_cost=0.0,
        )

        # Calculate cost for GPT-3.5-turbo
        cost = client.estimate_cost(usage, model="gpt-3.5-turbo")
        assert cost > 0
        assert isinstance(cost, float)

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="AsyncMock handling needs refactoring for batch operations")
    async def test_auto_batch_threshold(self, mock_openai_config, mock_batch_config):
        """Test automatic batch API usage based on threshold."""
        mock_batch_config.auto_batch_threshold = 5
        client = OpenAIClient(
            openai_config=mock_openai_config,
            batch_config=mock_batch_config,
        )

        prompts = ["Prompt " + str(i) for i in range(10)]

        from unittest.mock import AsyncMock

        with patch.object(client, "submit_batch_job", new_callable=AsyncMock) as mock_batch:
            with patch.object(
                client, "retrieve_batch_results", new_callable=AsyncMock
            ) as mock_retrieve:
                mock_batch.return_value = BatchJob(
                    job_id="auto-batch",
                    status="completed",
                    created_at=datetime.now(),
                    request_count=10,
                )
                mock_retrieve.return_value = [
                    ModelResponse(
                        content=f"Response {i}",
                        usage=TokenUsage(
                            prompt_tokens=10,
                            completion_tokens=5,
                            total_tokens=15,
                            estimated_cost=0.001,
                        ),
                        model="gpt-3.5-turbo",
                        request_id=f"id-{i}",
                        latency_ms=100.0,
                        temperature=0.7,
                        finish_reason="stop",
                    )
                    for i in range(10)
                ]

                responses = await client.batch_generate(
                    prompts,
                    use_batch_api=True,  # Should auto-trigger due to threshold
                )

                assert len(responses) == 10
                mock_batch.assert_called_once()
                mock_retrieve.assert_called_once()
