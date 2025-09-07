"""Exception classes for OpenAI client."""

from typing import Any, Dict, List, Optional


class OpenAIClientError(Exception):
    """Base exception for OpenAI client errors."""

    pass


class RateLimitError(OpenAIClientError):
    """Raised when rate limits are exceeded."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class APIError(OpenAIClientError):
    """Raised for API-level failures."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class BatchError(OpenAIClientError):
    """Raised for batch API failures."""

    def __init__(
        self,
        message: str,
        job_id: Optional[str] = None,
        errors: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(message)
        self.job_id = job_id
        self.errors = errors or []


class PartialFailureError(OpenAIClientError):
    """Raised when some requests in a batch fail."""

    def __init__(
        self,
        message: str,
        successful_responses: List[Any],
        failed_indices: Dict[int, Exception],
    ):
        super().__init__(message)
        self.successful_responses = successful_responses
        self.failed_indices = failed_indices

    def get_success_count(self) -> int:
        """Get number of successful responses."""
        return len(self.successful_responses)

    def get_failure_count(self) -> int:
        """Get number of failed requests."""
        return len(self.failed_indices)

    def get_total_count(self) -> int:
        """Get total number of requests."""
        return self.get_success_count() + self.get_failure_count()


class TimeoutError(OpenAIClientError):
    """Raised when a request times out."""

    def __init__(self, message: str, timeout: float):
        super().__init__(message)
        self.timeout = timeout


class ValidationError(OpenAIClientError):
    """Raised for request validation failures."""

    pass


class QuotaError(OpenAIClientError):
    """Raised when quota limits are exceeded."""

    def __init__(self, message: str, quota_type: Optional[str] = None):
        super().__init__(message)
        self.quota_type = quota_type
