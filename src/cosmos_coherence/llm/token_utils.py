"""Token counting and cost estimation utilities."""

import logging
from typing import Any, Dict, List, Union

import tiktoken

logger = logging.getLogger(__name__)

# Model to encoding mapping
MODEL_ENCODINGS = {
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4-32k": "cl100k_base",
    "gpt-4o": "cl100k_base",
    "gpt-4o-mini": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5-turbo-16k": "cl100k_base",
    "text-davinci-003": "p50k_base",
    "text-davinci-002": "p50k_base",
}

# Pricing per 1000 tokens (in USD)
# Note: These are example prices and should be updated regularly
MODEL_PRICING = {
    "gpt-4o-mini": {
        "input": 0.00015,
        "output": 0.0006,
    },
    "gpt-4o": {
        "input": 0.0025,
        "output": 0.01,
    },
    "gpt-4": {
        "input": 0.03,
        "output": 0.06,
    },
    "gpt-4-turbo": {
        "input": 0.01,
        "output": 0.03,
    },
    "gpt-4-32k": {
        "input": 0.06,
        "output": 0.12,
    },
    "gpt-3.5-turbo": {
        "input": 0.0005,
        "output": 0.0015,
    },
    "gpt-3.5-turbo-16k": {
        "input": 0.003,
        "output": 0.004,
    },
}

# Default pricing for unknown models
DEFAULT_PRICING = {
    "input": 0.002,
    "output": 0.002,
}


def get_encoding_for_model(model: str) -> tiktoken.Encoding:
    """Get the appropriate encoding for a model."""
    encoding_name = MODEL_ENCODINGS.get(model, "cl100k_base")
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.warning(f"Failed to get encoding {encoding_name} for model {model}: {e}")
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(
    text: Union[str, List[Dict[str, str]]],
    model: str = "gpt-3.5-turbo",
    is_messages: bool = False,
) -> int:
    """
    Count tokens in text or messages for the specified model.

    Args:
        text: String text or list of message dicts
        model: Model name for encoding selection
        is_messages: Whether input is in messages format

    Returns:
        Number of tokens
    """
    if not text:
        return 0

    encoding = get_encoding_for_model(model)

    if is_messages and isinstance(text, list):
        # Count tokens in messages format
        tokens = 0

        # Every message follows <|im_start|>{role}\n{content}<|im_end|>\n
        tokens_per_message = 4  # <|im_start|>, role, \n, <|im_end|>\n
        tokens_per_name = -1  # If name is included, it's role+name

        for message in text:
            tokens += tokens_per_message
            for key, value in message.items():
                if isinstance(value, str):
                    tokens += len(encoding.encode(value))
                if key == "name":
                    tokens += tokens_per_name

        tokens += 3  # Every reply is primed with <|im_start|>assistant<|im_sep|>
        return tokens

    elif isinstance(text, str):
        # Simple text encoding
        return len(encoding.encode(text))

    else:
        raise ValueError(f"Invalid input type: {type(text)}")


def get_model_pricing(model: str) -> Dict[str, float]:
    """
    Get pricing information for a model.

    Returns:
        Dict with 'input' and 'output' prices per 1000 tokens
    """
    # Check for exact match first
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]

    # Check for partial matches (e.g., "gpt-4-0613" matches "gpt-4")
    for model_prefix in MODEL_PRICING:
        if model.startswith(model_prefix):
            return MODEL_PRICING[model_prefix]

    # Return default pricing
    logger.warning(f"Unknown model {model}, using default pricing")
    return DEFAULT_PRICING


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str = "gpt-3.5-turbo",
    batch_api: bool = False,
) -> float:
    """
    Estimate cost for token usage.

    Args:
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        model: Model name for pricing
        batch_api: Whether using batch API (50% discount)

    Returns:
        Estimated cost in USD
    """
    pricing = get_model_pricing(model)

    # Calculate base cost
    input_cost = (prompt_tokens / 1000) * pricing["input"]
    output_cost = (completion_tokens / 1000) * pricing["output"]
    total_cost = input_cost + output_cost

    # Apply batch discount if applicable
    if batch_api:
        total_cost *= 0.5

    return total_cost


class TokenCounter:
    """Track token usage and costs across multiple requests."""

    def __init__(self):
        """Initialize the token counter."""
        self.reset()

    def reset(self):
        """Reset all counters."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self.request_count = 0
        self._model_breakdown: Dict[str, Dict[str, Any]] = {}

    def add_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "gpt-3.5-turbo",
        batch_api: bool = False,
    ):
        """
        Add token usage from a request.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            model: Model used
            batch_api: Whether batch API was used
        """
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.request_count += 1

        # Calculate cost
        cost = estimate_cost(prompt_tokens, completion_tokens, model, batch_api)
        self.total_cost += cost

        # Update model breakdown
        if model not in self._model_breakdown:
            self._model_breakdown[model] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "tokens": 0,
                "cost": 0.0,
                "requests": 0,
            }

        breakdown = self._model_breakdown[model]
        breakdown["prompt_tokens"] += prompt_tokens
        breakdown["completion_tokens"] += completion_tokens
        breakdown["tokens"] += prompt_tokens + completion_tokens
        breakdown["cost"] += cost
        breakdown["requests"] += 1

    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.total_prompt_tokens + self.total_completion_tokens

    def get_summary(self) -> Dict[str, Any]:
        """
        Get usage summary.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 4),
            "request_count": self.request_count,
            "average_tokens_per_request": (
                self.total_tokens // self.request_count if self.request_count > 0 else 0
            ),
            "average_cost_per_request": (
                round(self.total_cost / self.request_count, 4) if self.request_count > 0 else 0.0
            ),
        }

    def get_model_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """
        Get breakdown by model.

        Returns:
            Dictionary with per-model statistics
        """
        return {
            model: {
                "prompt_tokens": stats["prompt_tokens"],
                "completion_tokens": stats["completion_tokens"],
                "tokens": stats["tokens"],
                "cost": round(stats["cost"], 4),
                "requests": stats["requests"],
                "average_tokens": stats["tokens"] // stats["requests"]
                if stats["requests"] > 0
                else 0,
            }
            for model, stats in self._model_breakdown.items()
        }

    def __str__(self) -> str:
        """String representation of counter."""
        summary = self.get_summary()
        return (
            f"TokenCounter(requests={summary['request_count']}, "
            f"tokens={summary['total_tokens']}, "
            f"cost=${summary['total_cost']:.4f})"
        )

    def __repr__(self) -> str:
        """Detailed representation of counter."""
        return self.__str__()
