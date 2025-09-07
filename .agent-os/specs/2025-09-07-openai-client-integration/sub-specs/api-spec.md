# API Specification

This is the API specification for the spec detailed in @.agent-os/specs/2025-09-07-openai-client-integration/spec.md

## Python Client API

### OpenAIClient Class

```python
class OpenAIClient:
    def __init__(self, config: OpenAIConfig)
    async def generate_response(self, prompt: str, **kwargs) -> ModelResponse
    async def batch_generate(self, prompts: List[str], **kwargs) -> List[ModelResponse]
    async def submit_batch_job(self, requests: List[BatchRequest]) -> BatchJob
    async def get_batch_status(self, job_id: str) -> BatchJobStatus
    async def retrieve_batch_results(self, job_id: str) -> List[ModelResponse]
```

### Core Methods

#### generate_response
**Purpose:** Generate a single response with automatic rate limiting and retry logic
**Parameters:**
- `prompt` (str): The input prompt
- `model` (str): Model identifier (default from config)
- `temperature` (float): Sampling temperature
- `max_tokens` (int): Maximum response tokens
- `timeout` (float): Request timeout in seconds

**Response:** `ModelResponse` object containing:
- `content` (str): Generated text
- `usage` (TokenUsage): Token count details
- `model` (str): Model used
- `request_id` (str): Unique request identifier
- `latency_ms` (float): Request latency

**Errors:**
- `RateLimitError`: When rate limits exceeded after retries
- `APIError`: For API-level failures
- `TimeoutError`: When request exceeds timeout

#### batch_generate
**Purpose:** Process multiple prompts concurrently with optimal throughput
**Parameters:**
- `prompts` (List[str]): List of input prompts
- `model` (str): Model identifier
- `temperature` (float): Sampling temperature
- `max_concurrent` (int): Maximum concurrent requests
- `use_batch_api` (bool): Whether to use batch API for large requests
- `progress_callback` (Callable): Optional progress reporting function

**Response:** List of `ModelResponse` objects in same order as input

**Errors:**
- `PartialFailureError`: Some requests failed (includes successful responses)
- `RateLimitError`: Rate limits preventing completion
- `BatchError`: Batch API job failure

#### submit_batch_job
**Purpose:** Submit large request batches for asynchronous processing
**Parameters:**
- `requests` (List[BatchRequest]): Structured batch requests
- `completion_window` (str): "24h" or "7d" for different pricing tiers
- `metadata` (dict): Optional job metadata

**Response:** `BatchJob` object containing:
- `job_id` (str): Unique batch job identifier
- `status` (str): Initial status ("validating")
- `created_at` (datetime): Job creation time
- `request_count` (int): Number of requests in batch

**Errors:**
- `ValidationError`: Invalid request format
- `QuotaError`: Batch quota exceeded

### Configuration Models

#### OpenAIConfig
```python
class OpenAIConfig(BaseModel):
    api_key: str
    organization_id: Optional[str]
    base_url: str = "https://api.openai.com/v1"
    default_model: str = "gpt-3.5-turbo"
    timeout: float = 30.0
    max_retries: int = 3
```

#### RateLimitConfig
```python
class RateLimitConfig(BaseModel):
    requests_per_minute: int = 500
    tokens_per_minute: int = 90000
    max_concurrent: int = 10
    adaptive_throttling: bool = True
```

#### BatchConfig
```python
class BatchConfig(BaseModel):
    auto_batch_threshold: int = 100
    polling_interval: float = 60.0
    completion_window: str = "24h"
    max_batch_size: int = 50000
```

### Response Models

#### ModelResponse
```python
class ModelResponse(BaseModel):
    content: str
    usage: TokenUsage
    model: str
    request_id: str
    latency_ms: float
    temperature: float
    finish_reason: str
    cached: bool = False
```

#### TokenUsage
```python
class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float
```

#### BatchJobStatus
```python
class BatchJobStatus(BaseModel):
    job_id: str
    status: Literal["validating", "in_progress", "completed", "failed", "expired"]
    created_at: datetime
    completed_at: Optional[datetime]
    request_count: int
    completed_count: int
    failed_count: int
    errors: List[BatchError]
```

### Integration Examples

#### Basic Usage
```python
client = OpenAIClient(config)
response = await client.generate_response(
    "What is the capital of France?",
    temperature=0.7
)
```

#### Concurrent Benchmark Evaluation
```python
prompts = load_benchmark_questions()
responses = await client.batch_generate(
    prompts,
    max_concurrent=20,
    use_batch_api=len(prompts) > 500,
    progress_callback=lambda p: print(f"Progress: {p:.1%}")
)
```

#### Cost-Optimized Batch Processing
```python
batch_job = await client.submit_batch_job(
    requests=[BatchRequest(prompt=p, temperature=0.5) for p in prompts],
    completion_window="24h"  # 50% discount
)

# Poll for completion
while True:
    status = await client.get_batch_status(batch_job.job_id)
    if status.status == "completed":
        results = await client.retrieve_batch_results(batch_job.job_id)
        break
    await asyncio.sleep(60)
```

### Error Handling

All methods implement automatic retry with exponential backoff for transient errors. Non-retryable errors are raised immediately. The client maintains internal state for rate limiting across all method calls.

### Thread Safety

The client is thread-safe and can be shared across multiple async tasks. Internal semaphores and locks ensure proper rate limiting and connection management.
