"""OpenAI client implementation with rate limiting and batch support."""

import asyncio
import json
import logging
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

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

from .cache import LLMCache
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
        enable_cache: bool = True,
        cache_file: Optional[Union[str, Path]] = None,
    ):
        """Initialize the OpenAI client with configuration.

        Args:
            openai_config: OpenAI API configuration
            rate_limit_config: Rate limiting configuration
            batch_config: Batch API configuration
            retry_config: Retry configuration
            enable_cache: Whether to enable response caching
            cache_file: Optional path to cache file for persistence
        """
        self.openai_config = openai_config
        self.rate_limit_config = rate_limit_config or RateLimitConfig()  # type: ignore[call-arg]
        self.batch_config = batch_config or BatchConfig()  # type: ignore[call-arg]
        self.retry_config = retry_config or RetryConfig()  # type: ignore[call-arg]

        # Initialize cache
        self._cache_enabled = enable_cache
        self._cache: Optional[LLMCache] = None
        if enable_cache:
            self._cache = LLMCache(cache_file=cache_file)

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

    def _generate_cache_key(
        self,
        prompt: str,
        model: str,
        temperature: float,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate cache key for request parameters."""
        # Build parameters dict for cache key
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if max_tokens:
            params["max_tokens"] = max_tokens

        # Add any additional parameters that affect the response
        for key in ["top_p", "presence_penalty", "frequency_penalty", "seed"]:
            if key in kwargs:
                params[key] = kwargs[key]

        return self._cache.generate_cache_key(params) if self._cache else ""

    async def generate_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate a single response with automatic rate limiting and retry logic.

        If caching is enabled, will check cache before making API call.
        Streaming responses (stream=True) are never cached.
        """
        model = model or self.openai_config.default_model
        timeout = timeout or self.openai_config.timeout

        # Don't cache streaming responses
        is_streaming = kwargs.get("stream", False)

        # Check cache if enabled and not streaming
        if self._cache_enabled and self._cache and not is_streaming:
            cache_key = self._generate_cache_key(prompt, model, temperature, max_tokens, **kwargs)

            # Use lookup_or_compute for proper statistics tracking
            cached_response = self._cache.lookup_or_compute(
                cache_key, lambda: None  # Will be None on cache miss
            )

            if cached_response is not None:
                # Cache hit - return cached response
                if isinstance(cached_response, dict):
                    return ModelResponse(**cached_response)
                return ModelResponse(**cached_response)  # type: ignore[arg-type]

            # Cache miss - will make API call and cache the result below

        # Normal API call
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
                model_response = self._parse_response(response, temperature, latency_ms)

                # Cache the response if caching is enabled and not streaming
                if self._cache_enabled and self._cache and not is_streaming:
                    try:
                        cache_key = self._generate_cache_key(
                            prompt, model, temperature, max_tokens, **kwargs
                        )
                        self._cache.set(cache_key, model_response.model_dump())
                    except Exception as e:
                        # Log cache error but don't fail the request
                        logger.warning(f"Failed to cache response: {e}")

                return model_response

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
        errors = []

        for result in results:
            # Check if this result has an error
            if "error" in result:
                errors.append(
                    {"custom_id": result.get("custom_id", "unknown"), "error": result["error"]}
                )
                continue

            # Process successful response
            if "response" in result and result["response"]:
                response_body = result["response"].get("body", result["response"])

                # Handle the response structure
                if "choices" in response_body and response_body["choices"]:
                    content = response_body["choices"][0]["message"]["content"]
                    finish_reason = response_body["choices"][0].get("finish_reason", "stop")
                else:
                    # Fallback for different response formats
                    content = response_body.get("content", "")
                    finish_reason = "stop"

                responses.append(
                    ModelResponse(
                        content=content,
                        usage=TokenUsage(
                            prompt_tokens=response_body["usage"]["prompt_tokens"],
                            completion_tokens=response_body["usage"]["completion_tokens"],
                            total_tokens=response_body["usage"]["total_tokens"],
                            estimated_cost=estimate_cost(
                                response_body["usage"]["prompt_tokens"],
                                response_body["usage"]["completion_tokens"],
                                response_body.get("model", self.openai_config.default_model),
                                batch_api=True,
                            ),
                        ),
                        model=response_body.get("model", self.openai_config.default_model),
                        request_id=result.get("custom_id", response_body.get("id", "")),
                        latency_ms=1.0,  # Minimal value for batch (not tracked)
                        temperature=0.7,  # Would need to track from original request
                        finish_reason=finish_reason,
                        cached=False,
                    )
                )

        # If there were errors, raise a partial failure
        if errors:
            if responses:
                # Convert error dicts to Exception objects for PartialFailureError
                error_exceptions: Dict[int, Exception] = {
                    i: APIError(f"Request {err['custom_id']}: {err['error']}")
                    for i, err in enumerate(errors)
                }
                raise PartialFailureError(
                    f"Batch job {job_id} had {len(errors)} failures out of {len(results)} requests",
                    responses,
                    error_exceptions,
                )
            else:
                raise BatchException(
                    f"Batch job {job_id} failed completely with {len(errors)} errors",
                    job_id=job_id,
                    errors=errors,
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

            return response.model_dump()  # type: ignore[no-any-return]

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
        # Create JSONL file with batch requests
        tasks = []
        for i, request in enumerate(requests):
            task = {
                "custom_id": f"request-{i}-{datetime.now().timestamp()}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": request.model or self.openai_config.default_model,
                    "messages": [{"role": "user", "content": request.prompt}],
                    "temperature": request.temperature,
                    **(request.model_dump(exclude={"prompt", "temperature", "model"})),
                },
            }
            tasks.append(task)

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as batch_file:
            for task in tasks:
                batch_file.write(json.dumps(task) + "\n")
            batch_file_path = batch_file.name

        try:
            # Upload file to OpenAI
            with open(batch_file_path, "rb") as f:
                uploaded_file = await asyncio.to_thread(  # type: ignore[misc]
                    self._client.files.create, file=f, purpose="batch"
                )

            # Create batch job
            batch = await asyncio.to_thread(  # type: ignore[misc]
                self._client.batches.create,
                input_file_id=uploaded_file.id,  # type: ignore[attr-defined]
                endpoint="/v1/chat/completions",
                completion_window=completion_window,  # type: ignore[arg-type]
                metadata=metadata,
            )

            return {
                "id": batch.id,  # type: ignore[attr-defined]
                "status": batch.status,  # type: ignore[attr-defined]
                "created_at": batch.created_at,  # type: ignore[attr-defined]
                "request_counts": {
                    "total": batch.request_counts.total  # type: ignore[attr-defined]
                    if hasattr(batch, "request_counts")
                    else len(requests)
                },
            }
        finally:
            # Clean up temporary file
            Path(batch_file_path).unlink(missing_ok=True)

    async def _get_batch_status_from_api(self, job_id: str) -> Dict[str, Any]:
        """Get batch status from OpenAI API."""
        batch = await asyncio.to_thread(self._client.batches.retrieve, job_id)  # type: ignore[misc]

        result = {
            "id": batch.id,  # type: ignore[attr-defined]
            "status": batch.status,  # type: ignore[attr-defined]
            "created_at": batch.created_at,  # type: ignore[attr-defined]
            "request_counts": {
                "total": batch.request_counts.total  # type: ignore[attr-defined]
                if hasattr(batch.request_counts, "total")  # type: ignore[arg-type, attr-defined]
                else 0,
                "completed": batch.request_counts.completed  # type: ignore[attr-defined]
                if hasattr(batch.request_counts, "completed")  # type: ignore[arg-type, attr-defined]
                else 0,
                "failed": batch.request_counts.failed  # type: ignore[attr-defined]
                if hasattr(batch.request_counts, "failed")  # type: ignore[arg-type, attr-defined]
                else 0,
            },
            "errors": [],
        }

        if batch.completed_at:  # type: ignore[attr-defined]
            result["completed_at"] = batch.completed_at  # type: ignore[attr-defined]

        # Add any errors if present
        if hasattr(batch, "errors") and batch.errors:
            result["errors"] = [{"code": e.code, "message": e.message} for e in batch.errors]

        return result

    async def _retrieve_batch_results_from_api(self, job_id: str) -> List[Dict[str, Any]]:
        """Retrieve batch results from OpenAI API."""
        # Get the batch to find output file ID
        batch = await asyncio.to_thread(self._client.batches.retrieve, job_id)  # type: ignore[misc]

        if not batch.output_file_id:  # type: ignore[attr-defined]
            raise BatchException(f"Batch {job_id} has no output file", job_id=job_id, errors=[])

        # Download the results file
        file_content = await asyncio.to_thread(self._client.files.content, batch.output_file_id)  # type: ignore[misc, attr-defined]

        # Parse JSONL results
        results = []
        for line in file_content.content.decode("utf-8").strip().split("\n"):  # type: ignore[attr-defined]
            if line:
                result = json.loads(line)
                results.append(result)

        return results

    def get_cache_statistics(self):
        """Get cache performance statistics.

        Returns:
            CacheStatistics object with metrics, or empty stats if cache disabled
        """
        if self._cache:
            return self._cache.get_statistics()
        else:
            from .cache import CacheStatistics

            return CacheStatistics()

    def print_cache_statistics(self) -> None:
        """Print formatted cache statistics to console."""
        stats = self.get_cache_statistics()

        print("\n" + "=" * 50)
        print("Cache Statistics")
        print("=" * 50)
        print(f"Total requests: {stats.total_requests}")
        print(f"Cache hits: {stats.cache_hits}")
        print(f"Cache misses: {stats.cache_misses}")
        print(f"Hit rate: {stats.hit_rate*100:.1f}%")
        print(f"Tokens saved: {stats.tokens_saved}")
        print(f"Estimated cost savings: ${stats.estimated_cost_savings():.4f}")
        print("=" * 50 + "\n")
