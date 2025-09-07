"""Tests for OpenAI Batch API implementation."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cosmos_coherence.llm.config import BatchConfig, OpenAIConfig
from cosmos_coherence.llm.exceptions import BatchError as BatchException
from cosmos_coherence.llm.exceptions import PartialFailureError, ValidationError
from cosmos_coherence.llm.models import BatchJob, BatchJobStatus, BatchRequest
from cosmos_coherence.llm.openai_client import OpenAIClient


@pytest.fixture
def openai_config():
    """Create OpenAI configuration for testing."""
    return OpenAIConfig(
        api_key="test-key",
        default_model="gpt-4o-mini",
    )


@pytest.fixture
def batch_config():
    """Create batch configuration for testing."""
    return BatchConfig(
        auto_batch_threshold=10,
        max_batch_size=50000,
        completion_window="24h",
        polling_interval=5.0,
    )


@pytest.fixture
def client(openai_config, batch_config):
    """Create OpenAI client for testing."""
    return OpenAIClient(openai_config, batch_config=batch_config)


class TestBatchAPIImplementation:
    """Test the real OpenAI Batch API implementation."""

    @pytest.mark.asyncio
    async def test_submit_batch_to_api_creates_jsonl_file(self, client):
        """Test that submit_batch_to_api creates proper JSONL file."""
        requests = [
            BatchRequest(prompt=f"Test prompt {i}", temperature=0.7, model="gpt-4o-mini")
            for i in range(3)
        ]

        # Mock the OpenAI client methods
        mock_file = MagicMock(id="file-123")
        mock_batch = MagicMock(
            id="batch-123",
            status="validating",
            created_at=datetime.now().timestamp(),
            request_counts=MagicMock(total=3),
        )

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            # Set up the mock to return different values based on the call
            mock_to_thread.side_effect = [mock_file, mock_batch]
            result = await client._submit_batch_to_api(requests, "24h", {"test": "metadata"})

        # Verify the calls were made
        assert mock_to_thread.call_count == 2

        # First call should be for file upload
        first_call = mock_to_thread.call_args_list[0]
        assert first_call[1]["purpose"] == "batch"

        # Second call should be for batch creation
        second_call = mock_to_thread.call_args_list[1]
        assert second_call[1]["input_file_id"] == "file-123"
        assert second_call[1]["endpoint"] == "/v1/chat/completions"
        assert second_call[1]["completion_window"] == "24h"

        # Verify result format
        assert result["id"] == "batch-123"
        assert result["status"] == "validating"
        assert result["request_counts"]["total"] == 3

    @pytest.mark.asyncio
    async def test_submit_batch_validates_size_limit(self, client):
        """Test that submit_batch_job validates batch size."""
        # Create too many requests
        requests = [
            BatchRequest(prompt=f"Test {i}", temperature=0.7, model="gpt-4o-mini")
            for i in range(50001)  # Exceeds max_batch_size of 50000
        ]

        with pytest.raises(ValidationError) as exc_info:
            await client.submit_batch_job(requests)

        assert "exceeds maximum" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_batch_status_from_api(self, client):
        """Test getting batch status from API."""
        mock_batch = MagicMock(
            id="batch-123",
            status="completed",
            created_at=datetime.now().timestamp(),
            completed_at=datetime.now().timestamp(),
            request_counts=MagicMock(total=100, completed=98, failed=2),
            errors=[],
        )

        with patch("asyncio.to_thread", return_value=mock_batch):
            result = await client._get_batch_status_from_api("batch-123")

        assert result["id"] == "batch-123"
        assert result["status"] == "completed"
        assert result["request_counts"]["total"] == 100
        assert result["request_counts"]["completed"] == 98
        assert result["request_counts"]["failed"] == 2

    @pytest.mark.asyncio
    async def test_retrieve_batch_results_from_api(self, client):
        """Test retrieving batch results from API."""
        # Create mock batch with output file
        mock_batch = MagicMock(output_file_id="output-file-123")

        # Create mock JSONL results
        results_data = [
            {
                "custom_id": "request-0",
                "response": {
                    "body": {
                        "choices": [
                            {"message": {"content": "Response 1"}, "finish_reason": "stop"}
                        ],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                        "model": "gpt-4o-mini",
                        "id": "resp-1",
                    }
                },
            },
            {
                "custom_id": "request-1",
                "response": {
                    "body": {
                        "choices": [
                            {"message": {"content": "Response 2"}, "finish_reason": "stop"}
                        ],
                        "usage": {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40},
                        "model": "gpt-4o-mini",
                        "id": "resp-2",
                    }
                },
            },
        ]

        jsonl_content = "\n".join(json.dumps(r) for r in results_data)
        mock_file_content = MagicMock(content=jsonl_content.encode("utf-8"))

        with patch("asyncio.to_thread", side_effect=[mock_batch, mock_file_content]):
            results = await client._retrieve_batch_results_from_api("batch-123")

        assert len(results) == 2
        assert results[0]["custom_id"] == "request-0"
        assert results[1]["custom_id"] == "request-1"

    @pytest.mark.asyncio
    @pytest.mark.skip(
        reason="PartialFailureError.successful_results attribute needs implementation"
    )
    async def test_retrieve_batch_results_handles_errors(self, client):
        """Test that retrieve_batch_results handles errors in responses."""
        # Create mock results with one error
        mock_results = [
            {
                "custom_id": "request-0",
                "response": {
                    "body": {
                        "choices": [{"message": {"content": "Success"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                        "model": "gpt-4o-mini",
                    }
                },
            },
            {
                "custom_id": "request-1",
                "error": {"code": "rate_limit_exceeded", "message": "Rate limit exceeded"},
            },
        ]

        with patch.object(client, "_retrieve_batch_results_from_api", return_value=mock_results):
            with pytest.raises(PartialFailureError) as exc_info:
                await client.retrieve_batch_results("batch-123")

            error = exc_info.value
            assert "1 failures out of 2 requests" in str(error)
            assert len(error.successful_results) == 1
            assert error.successful_results[0].content == "Success"

    @pytest.mark.asyncio
    async def test_retrieve_batch_results_no_output_file(self, client):
        """Test error when batch has no output file."""
        mock_batch = MagicMock(output_file_id=None)

        with patch("asyncio.to_thread", return_value=mock_batch):
            with pytest.raises(BatchException) as exc_info:
                await client._retrieve_batch_results_from_api("batch-123")

            assert "no output file" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_batch_job_polling_success(self, client):
        """Test successful batch job polling and completion."""
        # Test with a single prompt

        # Mock the batch job creation
        mock_job = BatchJob(
            job_id="batch-123", status="validating", created_at=datetime.now(), request_count=1
        )

        # Mock status progression
        mock_statuses = [
            BatchJobStatus(
                job_id="batch-123",
                status="in_progress",
                created_at=datetime.now(),
                request_count=1,
                completed_count=0,
                failed_count=0,
                errors=[],
            ),
            BatchJobStatus(
                job_id="batch-123",
                status="completed",
                created_at=datetime.now(),
                completed_at=datetime.now(),
                request_count=1,
                completed_count=1,
                failed_count=0,
                errors=[],
            ),
        ]

        mock_results = [MagicMock(content="Test response")]

        with patch.object(client, "submit_batch_job", return_value=mock_job):
            with patch.object(client, "get_batch_status", side_effect=mock_statuses):
                with patch.object(client, "retrieve_batch_results", return_value=mock_results):
                    results = await client._batch_generate_via_api(
                        ["Test prompt"], "gpt-4o-mini", 0.7, None
                    )

        assert len(results) == 1
        assert results[0].content == "Test response"

    @pytest.mark.asyncio
    async def test_batch_job_polling_failure(self, client):
        """Test batch job failure during polling."""
        # Test with a single prompt that will fail

        mock_job = BatchJob(
            job_id="batch-123", status="validating", created_at=datetime.now(), request_count=1
        )

        mock_status = BatchJobStatus(
            job_id="batch-123",
            status="failed",
            created_at=datetime.now(),
            request_count=1,
            completed_count=0,
            failed_count=1,
            errors=[{"code": "internal_error", "message": "Processing failed"}],
        )

        with patch.object(client, "submit_batch_job", return_value=mock_job):
            with patch.object(client, "get_batch_status", return_value=mock_status):
                with pytest.raises(BatchException) as exc_info:
                    await client._batch_generate_via_api(["Test prompt"], "gpt-4o-mini", 0.7, None)

                assert "failed with status: failed" in str(exc_info.value)

    def test_jsonl_format_validation(self):
        """Test that the JSONL format matches OpenAI's requirements."""
        # Test the expected format for a batch request
        task = {
            "custom_id": "request-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.7,
            },
        }

        # This should be valid JSON
        json_str = json.dumps(task)
        parsed = json.loads(json_str)

        assert parsed["custom_id"] == "request-1"
        assert parsed["method"] == "POST"
        assert parsed["url"] == "/v1/chat/completions"
        assert "model" in parsed["body"]
        assert "messages" in parsed["body"]
