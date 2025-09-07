"""OpenAI client implementation with rate limiting and batch support."""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import aiohttp
from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import BatchConfig, OpenAIConfig, RateLimitConfig, RetryConfig
from .exceptions import (
    APIError,
    PartialFailureError,
    RateLimitError,
    TimeoutError,
    ValidationError,
)
from .exceptions import (
    BatchError as BatchException,
)
from .models import (
    BatchError,
    BatchJob,
    BatchJobStatus,
    BatchRequest,
    ModelResponse,
    TokenUsage,
)
from .token_utils import count_tokens, estimate_cost

logger = logging.getLogger(__name__)


class OpenAIClient:
    """OpenAI client with rate limiting, concurrent processing, and batch API support."""

    def __init__(
        self,
        openai_config: OpenAIConfig,
        rate_limit_config: Optional[RateLimitConfig] = None,
        batch_config: Optional[BatchConfig] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        """Initialize the OpenAI client with configuration."""
        self.openai_config = openai_config
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self.batch_config = batch_config or BatchConfig()
        self.retry_config = retry_config or RetryConfig()

        # Initialize OpenAI client
        self._client = AsyncOpenAI(
            api_key=openai_config.api_key,
            organization=openai_config.organization_id,
            base_url=openai_config.base_url,
            timeout=openai_config.timeout,
            max_retries=0,  # We handle retries ourselves
        )

        # Initialize rate limiter
        self._rate_limiter = AsyncLimiter(
            self.rate_limit_config.requests_per_minute,
            60.0,  # Per minute
        )

        # Initialize semaphore for concurrent connections
        self._semaphore = asyncio.Semaphore(self.rate_limit_config.max_concurrent)

        # Track rate limit headers for adaptive throttling
        self._last_rate_limit_headers: Dict[str, Any] = {}

        # Session for batch API calls
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()

    async def generate_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate a single response with automatic rate limiting and retry logic."""
        model = model or self.openai_config.default_model
        timeout = timeout or self.openai_config.timeout

        # Apply rate limiting
        async with self._rate_limiter:
            async with self._semaphore:
                start_time = time.time()

                # Make request with retry logic
                response = await self._make_request_with_retry(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    **kwargs,
                )

                latency_ms = (time.time() - start_time) * 1000

                # Parse response
                return self._parse_response(response, temperature, latency_ms)

    async def batch_generate(
        self,
        prompts: List[str],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_concurrent: Optional[int] = None,
        use_batch_api: Optional[bool] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
        **kwargs,
    ) -> List[ModelResponse]:
        """Process multiple prompts concurrently with optimal throughput."""
        model = model or self.openai_config.default_model
        max_concurrent = max_concurrent or self.rate_limit_config.max_concurrent

        # Determine whether to use batch API
        if use_batch_api is None:
            use_batch_api = len(prompts) >= self.batch_config.auto_batch_threshold

        if use_batch_api:
            return await self._batch_generate_via_api(
                prompts, model, temperature, progress_callback, **kwargs
            )
        else:
            return await self._batch_generate_concurrent(
                prompts, model, temperature, max_concurrent, progress_callback, **kwargs
            )

    async def _batch_generate_concurrent(
        self,
        prompts: List[str],
        model: str,
        temperature: float,
        max_concurrent: int,
        progress_callback: Optional[Callable[[float], None]],
        **kwargs,
    ) -> List[ModelResponse]:
        """Generate responses concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        completed = 0
        total = len(prompts)
        responses = []
        failures = {}

        async def generate_with_progress(idx: int, prompt: str):
            nonlocal completed
            async with semaphore:
                try:
                    response = await self.generate_response(prompt, model, temperature, **kwargs)
                    responses.append((idx, response))
                except Exception as e:
                    failures[idx] = e
                finally:
                    completed += 1
                    if progress_callback:
                        progress_callback(completed / total)

        # Create tasks for all prompts
        tasks = [generate_with_progress(i, prompt) for i, prompt in enumerate(prompts)]

        await asyncio.gather(*tasks, return_exceptions=True)

        # Check for failures
        if failures:
            # Sort successful responses by index
            responses.sort(key=lambda x: x[0])
            successful = [r for _, r in responses]
            raise PartialFailureError(
                f"Failed to generate {len(failures)} out of {total} responses",
                successful,
                failures,
            )

        # Sort responses by original index and return
        responses.sort(key=lambda x: x[0])
        return [response for _, response in responses]

    async def _batch_generate_via_api(
        self,
        prompts: List[str],
        model: str,
        temperature: float,
        progress_callback: Optional[Callable[[float], None]],
        **kwargs,
    ) -> List[ModelResponse]:
        """Generate responses using the batch API."""
        # Create batch requests
        requests = [
            BatchRequest(prompt=prompt, temperature=temperature, model=model, **kwargs)
            for prompt in prompts
        ]

        # Submit batch job
        job = await self.submit_batch_job(requests)

        # Poll for completion
        while True:
            status = await self.get_batch_status(job.job_id)

            if progress_callback and status.request_count > 0:
                progress = status.completed_count / status.request_count
                progress_callback(progress)

            if status.status == "completed":
                break
            elif status.status in ["failed", "expired"]:
                raise BatchException(
                    f"Batch job {job.job_id} failed with status: {status.status}",
                    job_id=job.job_id,
                    errors=[e.dict() for e in status.errors],
                )

            await asyncio.sleep(self.batch_config.polling_interval)

        # Retrieve results
        return await self.retrieve_batch_results(job.job_id)

    async def submit_batch_job(
        self,
        requests: List[BatchRequest],
        completion_window: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BatchJob:
        """Submit large request batches for asynchronous processing."""
        completion_window = completion_window or self.batch_config.completion_window

        # Validate batch size
        if len(requests) > self.batch_config.max_batch_size:
            raise ValidationError(
                f"Batch size {len(requests)} exceeds maximum {self.batch_config.max_batch_size}"
            )

        # Submit to API
        response = await self._submit_batch_to_api(requests, completion_window, metadata)

        # Parse response
        return BatchJob(
            job_id=response["id"],
            status=response["status"],
            created_at=datetime.fromisoformat(response["created_at"]),
            request_count=response["request_counts"]["total"],
        )

    async def get_batch_status(self, job_id: str) -> BatchJobStatus:
        """Get the status of a batch job."""
        response = await self._get_batch_status_from_api(job_id)

        return BatchJobStatus(
            job_id=response["id"],
            status=response["status"],
            created_at=datetime.fromisoformat(response["created_at"]),
            completed_at=(
                datetime.fromisoformat(response["completed_at"])
                if response.get("completed_at")
                else None
            ),
            request_count=response["request_counts"]["total"],
            completed_count=response["request_counts"].get("completed", 0),
            failed_count=response["request_counts"].get("failed", 0),
            errors=[BatchError(**error) for error in response.get("errors", [])],
        )

    async def retrieve_batch_results(self, job_id: str) -> List[ModelResponse]:
        """Retrieve results from a completed batch job."""
        results = await self._retrieve_batch_results_from_api(job_id)

        responses = []
        for result in results:
            response_body = result["response"]["body"]
            responses.append(
                ModelResponse(
                    content=response_body["choices"][0]["message"]["content"],
                    usage=TokenUsage(
                        prompt_tokens=response_body["usage"]["prompt_tokens"],
                        completion_tokens=response_body["usage"]["completion_tokens"],
                        total_tokens=response_body["usage"]["total_tokens"],
                        estimated_cost=estimate_cost(
                            response_body["usage"]["prompt_tokens"],
                            response_body["usage"]["completion_tokens"],
                            response_body["model"],
                            batch_api=True,
                        ),
                    ),
                    model=response_body["model"],
                    request_id=response_body["id"],
                    latency_ms=1.0,  # Minimal value for batch (not tracked)
                    temperature=0.7,  # Would need to track from original request
                    finish_reason=response_body["choices"][0].get("finish_reason", "stop"),
                    cached=False,
                )
            )

        return responses

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Count tokens in text for the specified model."""
        model = model or self.openai_config.default_model
        return count_tokens(text, model)

    def estimate_cost(self, usage: TokenUsage, model: Optional[str] = None) -> float:
        """Estimate cost for token usage."""
        model = model or self.openai_config.default_model
        return estimate_cost(
            usage.prompt_tokens,
            usage.completion_tokens,
            model,
        )

    # Private methods

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((RateLimitError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.DEBUG),
    )
    async def _make_request_with_retry(self, **kwargs) -> Dict[str, Any]:
        """Make request with retry logic."""
        return await self._make_request(**kwargs)

    async def _make_request(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        timeout: float,
        **kwargs,
    ) -> Dict[str, Any]:
        """Make a request to the OpenAI API."""
        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                ),
                timeout=timeout,
            )

            # Update rate limit headers if present
            if hasattr(response, "headers"):
                self._update_rate_limit_headers(response.headers)

            return response.model_dump()

        except asyncio.TimeoutError:
            raise TimeoutError(f"Request timed out after {timeout} seconds", timeout)
        except Exception as e:
            if "rate_limit" in str(e).lower():
                raise RateLimitError(str(e))
            raise APIError(str(e))

    def _parse_response(
        self,
        response: Dict[str, Any],
        temperature: float,
        latency_ms: float,
    ) -> ModelResponse:
        """Parse API response into ModelResponse."""
        choice = response["choices"][0]
        usage = response["usage"]

        return ModelResponse(
            content=choice["message"]["content"],
            usage=TokenUsage(
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
                estimated_cost=estimate_cost(
                    usage["prompt_tokens"],
                    usage["completion_tokens"],
                    response["model"],
                ),
            ),
            model=response["model"],
            request_id=response["id"],
            latency_ms=latency_ms,
            temperature=temperature,
            finish_reason=choice.get("finish_reason", "stop"),
            cached=False,
        )

    def _update_rate_limit_headers(self, headers: Dict[str, str]):
        """Update rate limit headers for adaptive throttling."""
        if self.rate_limit_config.adaptive_throttling:
            self._last_rate_limit_headers = {
                "requests_remaining": headers.get("x-ratelimit-remaining-requests"),
                "tokens_remaining": headers.get("x-ratelimit-remaining-tokens"),
                "reset_requests": headers.get("x-ratelimit-reset-requests"),
                "reset_tokens": headers.get("x-ratelimit-reset-tokens"),
            }

    async def _submit_batch_to_api(
        self,
        requests: List[BatchRequest],
        completion_window: str,
        metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Submit batch to OpenAI API."""
        # This is a placeholder - actual implementation would use OpenAI's batch API
        # when it becomes available in the SDK
        return {
            "id": f"batch-{datetime.now().timestamp()}",
            "status": "validating",
            "created_at": datetime.now().isoformat(),
            "request_counts": {"total": len(requests)},
        }

    async def _get_batch_status_from_api(self, job_id: str) -> Dict[str, Any]:
        """Get batch status from OpenAI API."""
        # Placeholder implementation
        return {
            "id": job_id,
            "status": "completed",
            "created_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat(),
            "request_counts": {
                "total": 100,
                "completed": 98,
                "failed": 2,
            },
            "errors": [],
        }

    async def _retrieve_batch_results_from_api(self, job_id: str) -> List[Dict[str, Any]]:
        """Retrieve batch results from OpenAI API."""
        # Placeholder implementation
        return []
