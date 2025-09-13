"""Tests for SimpleQA result models."""

import json
from uuid import uuid4

import pytest
from cosmos_coherence.benchmarks.models.datasets import SimpleQAResult
from pydantic import ValidationError


class TestSimpleQAResult:
    """Test SimpleQAResult data model."""

    def test_create_result_minimal(self):
        """Test creating result with minimal fields."""
        result = SimpleQAResult(
            experiment_id=uuid4(),
            item_id=uuid4(),
            question="What is the capital of France?",
            prediction="Paris",
            ground_truth="Paris",
        )

        assert result.question == "What is the capital of France?"
        assert result.prediction == "Paris"
        assert result.ground_truth == "Paris"
        assert result.is_correct is True  # Exact match
        assert result.f1_score == 1.0
        assert result.exact_match is True

    def test_create_result_with_metrics(self):
        """Test creating result with custom metrics."""
        exp_id = uuid4()
        item_id = uuid4()

        result = SimpleQAResult(
            experiment_id=exp_id,
            item_id=item_id,
            question="Who wrote Romeo and Juliet?",
            prediction="Shakespeare",
            ground_truth="William Shakespeare",
            is_correct=False,
            f1_score=0.5,
            exact_match=False,
            response_length=1,
            ground_truth_length=2,
        )

        assert result.is_correct is False
        assert result.f1_score == 0.5
        assert result.exact_match is False
        assert result.response_length == 1
        assert result.ground_truth_length == 2

    def test_calculate_metrics(self):
        """Test automatic metric calculation."""
        result = SimpleQAResult(
            experiment_id=uuid4(),
            item_id=uuid4(),
            question="What year did World War II end?",
            prediction="1945",
            ground_truth="1945",
        )

        # Metrics should be auto-calculated
        metrics = result.calculate_metrics()
        assert "exact_match" in metrics
        assert "f1_score" in metrics
        assert metrics["exact_match"] == 1.0
        assert metrics["f1_score"] == 1.0

    def test_calculate_metrics_partial_match(self):
        """Test metric calculation with partial match."""
        result = SimpleQAResult(
            experiment_id=uuid4(),
            item_id=uuid4(),
            question="Name the planets in our solar system",
            prediction="Mercury Venus Earth Mars",
            ground_truth="Mercury Venus Earth Mars Jupiter Saturn Uranus Neptune",
        )

        result.calculate_metrics()
        assert result.exact_match is False
        assert result.f1_score > 0 and result.f1_score < 1  # Partial match
        assert result.is_correct is False  # Not exact match

    def test_validation_empty_strings(self):
        """Test validation rejects empty strings."""
        with pytest.raises(ValidationError) as exc:
            SimpleQAResult(
                experiment_id=uuid4(),
                item_id=uuid4(),
                question="",  # Empty question
                prediction="Paris",
                ground_truth="Paris",
            )
        assert "question" in str(exc.value)

    def test_validation_f1_score_range(self):
        """Test F1 score must be in [0, 1] range."""
        with pytest.raises(ValidationError) as exc:
            SimpleQAResult(
                experiment_id=uuid4(),
                item_id=uuid4(),
                question="Test question",
                prediction="answer",
                ground_truth="answer",
                f1_score=1.5,  # Invalid score
            )
        assert "f1_score" in str(exc.value)

    def test_serialize_to_dict(self):
        """Test serialization to dictionary."""
        result = SimpleQAResult(
            experiment_id=uuid4(),
            item_id=uuid4(),
            question="What is 2+2?",
            prediction="4",
            ground_truth="4",
        )

        data = result.serialize()
        assert isinstance(data, dict)
        assert data["question"] == "What is 2+2?"
        assert data["prediction"] == "4"
        assert data["ground_truth"] == "4"
        assert data["is_correct"] is True
        assert isinstance(data["experiment_id"], str)  # UUID as string
        assert isinstance(data["timestamp"], str)  # datetime as ISO string

    def test_serialize_to_json(self):
        """Test serialization to JSON string."""
        result = SimpleQAResult(
            experiment_id=uuid4(),
            item_id=uuid4(),
            question="What is the speed of light?",
            prediction="299,792,458 m/s",
            ground_truth="299,792,458 meters per second",
        )

        json_str = result.to_json()
        data = json.loads(json_str)

        assert data["question"] == "What is the speed of light?"
        assert data["prediction"] == "299,792,458 m/s"
        assert data["ground_truth"] == "299,792,458 meters per second"

    def test_from_evaluation(self):
        """Test creating result from evaluation data."""
        eval_data = {
            "question": "What is the capital of Japan?",
            "expected": "Tokyo",
            "response": "Tokyo",
            "correct": True,
            "f1_score": 1.0,
            "exact_match": True,
        }

        result = SimpleQAResult.from_evaluation(
            experiment_id=uuid4(),
            item_id=uuid4(),
            eval_data=eval_data,
        )

        assert result.question == "What is the capital of Japan?"
        assert result.prediction == "Tokyo"
        assert result.ground_truth == "Tokyo"
        assert result.is_correct is True
        assert result.f1_score == 1.0

    def test_aggregate_results(self):
        """Test aggregating multiple results."""
        results = []
        exp_id = uuid4()

        # Create mix of correct and incorrect results
        for i in range(10):
            results.append(
                SimpleQAResult(
                    experiment_id=exp_id,
                    item_id=uuid4(),
                    question=f"Question {i}",
                    prediction="answer" if i < 7 else "wrong",
                    ground_truth="answer",
                )
            )

        # Calculate aggregate metrics
        aggregated = SimpleQAResult.aggregate_metrics(results)

        assert aggregated["total_questions"] == 10
        assert aggregated["correct_answers"] == 7
        assert aggregated["accuracy"] == 0.7
        assert "average_f1_score" in aggregated
        assert aggregated["average_f1_score"] == 0.7  # 7 perfect + 3 zero scores

    def test_export_to_jsonl(self):
        """Test exporting results to JSONL format."""
        results = []
        exp_id = uuid4()

        for i in range(3):
            results.append(
                SimpleQAResult(
                    experiment_id=exp_id,
                    item_id=uuid4(),
                    question=f"Question {i}",
                    prediction=f"Answer {i}",
                    ground_truth=f"Answer {i}",
                )
            )

        # Export to JSONL
        jsonl_lines = SimpleQAResult.to_jsonl(results, include_metadata=True)
        lines = jsonl_lines.strip().split("\n")

        # First line should be metadata
        metadata = json.loads(lines[0])
        assert metadata["type"] == "experiment_metadata"
        assert metadata["benchmark"] == "SimpleQA"
        assert metadata["total_questions"] == 3
        assert metadata["accuracy"] == 1.0

        # Subsequent lines should be individual results
        for i, line in enumerate(lines[1:], 0):
            data = json.loads(line)
            assert data["question"] == f"Question {i}"
            assert data["prediction"] == f"Answer {i}"

    def test_export_without_metadata(self):
        """Test exporting without metadata."""
        results = [
            SimpleQAResult(
                experiment_id=uuid4(),
                item_id=uuid4(),
                question="Q1",
                prediction="A1",
                ground_truth="A1",
            ),
            SimpleQAResult(
                experiment_id=uuid4(),
                item_id=uuid4(),
                question="Q2",
                prediction="A2",
                ground_truth="A2",
            ),
        ]

        jsonl_lines = SimpleQAResult.to_jsonl(results, include_metadata=False)
        lines = jsonl_lines.strip().split("\n")

        # Should only have result lines, no metadata
        assert len(lines) == 2
        for i, line in enumerate(lines, 1):
            data = json.loads(line)
            assert data["question"] == f"Q{i}"
            assert "type" not in data  # No metadata marker

    def test_load_from_jsonl(self):
        """Test loading results from JSONL."""
        jsonl_lines = [
            '{"experiment_id": "123e4567-e89b-12d3-a456-426614174000", '
            '"item_id": "123e4567-e89b-12d3-a456-426614174001", '
            '"question": "Q1", "prediction": "A1", "ground_truth": "A1", '
            '"is_correct": true, "f1_score": 1.0}',
            '{"experiment_id": "123e4567-e89b-12d3-a456-426614174000", '
            '"item_id": "123e4567-e89b-12d3-a456-426614174002", '
            '"question": "Q2", "prediction": "A2", "ground_truth": "A2", '
            '"is_correct": true, "f1_score": 1.0}',
        ]
        jsonl_content = "\n".join(jsonl_lines)

        results = SimpleQAResult.from_jsonl(jsonl_content)

        assert len(results) == 2
        assert results[0].question == "Q1"
        assert results[1].question == "Q2"
        assert all(r.is_correct for r in results)

    def test_compatibility_with_base_result(self):
        """Test that SimpleQAResult is compatible with BaseResult."""
        from cosmos_coherence.benchmarks.models.base import BaseResult

        result = SimpleQAResult(
            experiment_id=uuid4(),
            item_id=uuid4(),
            question="Test question",
            prediction="Test answer",
            ground_truth="Test answer",
        )

        # Should be instance of BaseResult
        assert isinstance(result, BaseResult)

        # Should have all BaseResult fields
        assert hasattr(result, "experiment_id")
        assert hasattr(result, "item_id")
        assert hasattr(result, "prediction")
        assert hasattr(result, "ground_truth")
        assert hasattr(result, "metrics")
        assert hasattr(result, "timestamp")
        assert hasattr(result, "version")
