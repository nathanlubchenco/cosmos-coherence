"""Configuration models for OpenAI client."""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAIConfig(BaseSettings):
    """OpenAI API configuration."""

    api_key: str = Field(..., description="OpenAI API key")
    organization_id: Optional[str] = Field(None, description="OpenAI organization ID")
    base_url: str = Field("https://api.openai.com/v1", description="API base URL")
    default_model: str = Field("gpt-4o-mini", description="Default model to use")
    timeout: float = Field(30.0, gt=0, description="Request timeout in seconds")
    max_retries: int = Field(3, ge=0, description="Maximum retry attempts")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="OPENAI_",
        case_sensitive=False,
        extra="ignore",
    )


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""

    requests_per_minute: int = Field(500, gt=0, description="Maximum requests per minute")
    tokens_per_minute: int = Field(90000, gt=0, description="Maximum tokens per minute")
    max_concurrent: int = Field(10, gt=0, le=50, description="Maximum concurrent connections")
    adaptive_throttling: bool = Field(
        True, description="Enable adaptive rate limiting based on headers"
    )

    @field_validator("max_concurrent")
    @classmethod
    def validate_concurrent(cls, v):
        """Ensure concurrent connections is reasonable."""
        if v > 50:
            raise ValueError("max_concurrent should not exceed 50 for API stability")
        return v


class BatchConfig(BaseModel):
    """Batch API configuration."""

    auto_batch_threshold: int = Field(
        100, ge=1, description="Number of requests to trigger batch API"
    )
    polling_interval: float = Field(
        60.0, gt=0, description="Batch status polling interval in seconds"
    )
    completion_window: Literal["24h", "7d"] = Field(
        "24h", description="Batch completion window for pricing"
    )
    max_batch_size: int = Field(50000, gt=0, le=50000, description="Maximum requests per batch")

    @field_validator("completion_window")
    @classmethod
    def validate_completion_window(cls, v):
        """Validate completion window values."""
        if v not in ["24h", "7d"]:
            raise ValueError("completion_window must be '24h' or '7d'")
        return v

    @field_validator("max_batch_size")
    @classmethod
    def validate_batch_size(cls, v):
        """Ensure batch size doesn't exceed API limits."""
        if v > 50000:
            raise ValueError("max_batch_size cannot exceed OpenAI's limit of 50,000")
        return v


class RetryConfig(BaseModel):
    """Retry configuration for failed requests."""

    max_retries: int = Field(3, ge=0, le=10, description="Maximum retry attempts")
    initial_delay: float = Field(1.0, gt=0, description="Initial retry delay in seconds")
    max_delay: float = Field(60.0, gt=0, description="Maximum retry delay in seconds")
    exponential_base: float = Field(2.0, gt=1, description="Exponential backoff base")
    jitter: bool = Field(True, description="Add random jitter to retry delays")

    @field_validator("max_delay")
    @classmethod
    def validate_max_delay(cls, v, info):
        """Ensure max_delay is greater than initial_delay."""
        if info.data.get("initial_delay") and v < info.data["initial_delay"]:
            raise ValueError("max_delay must be greater than initial_delay")
        return v
