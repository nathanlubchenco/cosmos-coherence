"""OpenAI LLM client module for Cosmos Coherence."""

from .config import (
    BatchConfig,
    OpenAIConfig,
    RateLimitConfig,
    RetryConfig,
)
from .exceptions import (
    APIError,
    PartialFailureError,
    RateLimitError,
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
from .openai_client import OpenAIClient
from .token_utils import (
    TokenCounter,
    count_tokens,
    estimate_cost,
    get_model_pricing,
)

__all__ = [
    # Configuration
    "OpenAIConfig",
    "RateLimitConfig",
    "BatchConfig",
    "RetryConfig",
    # Models
    "ModelResponse",
    "TokenUsage",
    "BatchRequest",
    "BatchJob",
    "BatchJobStatus",
    "BatchError",
    # Client
    "OpenAIClient",
    # Exceptions
    "RateLimitError",
    "APIError",
    "BatchException",
    "PartialFailureError",
    # Token utilities
    "count_tokens",
    "estimate_cost",
    "get_model_pricing",
    "TokenCounter",
]
