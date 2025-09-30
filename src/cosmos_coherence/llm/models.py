"""Model classes for OpenAI client responses and requests."""

from datetime import datetime
from typing import List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class TokenUsage(BaseModel):
    """Token usage information for a request."""

    prompt_tokens: int = Field(..., ge=0, description="Number of tokens in prompt")
    completion_tokens: int = Field(..., ge=0, description="Number of tokens in completion")
    total_tokens: int = Field(..., ge=0, description="Total tokens used")
    estimated_cost: float = Field(..., ge=0, description="Estimated cost in USD")

    @field_validator("total_tokens")
    @classmethod
    def validate_total(cls, v, info):
        """Ensure total equals sum of prompt and completion."""
        if "prompt_tokens" in info.data and "completion_tokens" in info.data:
            expected = info.data["prompt_tokens"] + info.data["completion_tokens"]
            if v != expected:
                raise ValueError(f"total_tokens must equal prompt + completion tokens ({expected})")
        return v


class ModelResponse(BaseModel):
    """Response from OpenAI model."""

    content: str = Field(..., description="Generated text content")
    usage: TokenUsage = Field(..., description="Token usage information")
    model: str = Field(..., description="Model used for generation")
    request_id: str = Field(..., description="Unique request identifier")
    latency_ms: float = Field(..., gt=0, description="Request latency in milliseconds")
    temperature: float = Field(..., ge=0, le=2, description="Temperature used")
    finish_reason: str = Field(..., description="Reason for completion")
    cached: bool = Field(False, description="Whether response was cached")
    raw_response: Optional[dict] = Field(
        None, description="Raw API response (includes logprobs if requested)"
    )


class BatchRequest(BaseModel):
    """Request for batch API processing."""

    prompt: str = Field(..., description="Input prompt")
    temperature: float = Field(..., ge=0, le=2, description="Sampling temperature")
    model: Optional[str] = Field(None, description="Model to use (overrides default)")
    max_tokens: Optional[int] = Field(None, gt=0, description="Maximum tokens to generate")
    custom_id: Optional[str] = Field(None, description="Custom identifier for request")

    def __init__(self, **data):
        """Initialize with auto-generated custom_id if not provided."""
        if "custom_id" not in data or data["custom_id"] is None:
            data["custom_id"] = f"req-{uuid4().hex[:12]}"
        super().__init__(**data)


class BatchJob(BaseModel):
    """Batch job information."""

    job_id: str = Field(..., description="Unique batch job identifier")
    status: Literal["validating", "in_progress", "completed", "failed", "expired"] = Field(
        ..., description="Current job status"
    )
    created_at: datetime = Field(..., description="Job creation timestamp")
    request_count: int = Field(..., gt=0, description="Number of requests in batch")


class BatchError(BaseModel):
    """Error information for batch processing."""

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    request_id: Optional[str] = Field(None, description="Associated request ID")


class BatchJobStatus(BaseModel):
    """Detailed batch job status."""

    job_id: str = Field(..., description="Unique batch job identifier")
    status: Literal["validating", "in_progress", "completed", "failed", "expired"] = Field(
        ..., description="Current job status"
    )
    created_at: datetime = Field(..., description="Job creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp")
    request_count: int = Field(..., ge=0, description="Total number of requests")
    completed_count: int = Field(..., ge=0, description="Number of completed requests")
    failed_count: int = Field(..., ge=0, description="Number of failed requests")
    errors: List[BatchError] = Field(default_factory=list, description="List of errors")

    @field_validator("completed_count")
    @classmethod
    def validate_completed_count(cls, v, info):
        """Ensure completed count doesn't exceed total."""
        if "request_count" in info.data and v > info.data["request_count"]:
            raise ValueError("completed_count cannot exceed request_count")
        return v

    @field_validator("failed_count")
    @classmethod
    def validate_failed_count(cls, v, info):
        """Ensure failed count doesn't exceed total."""
        if "request_count" in info.data and v > info.data["request_count"]:
            raise ValueError("failed_count cannot exceed request_count")
        return v
