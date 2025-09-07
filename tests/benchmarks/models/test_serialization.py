"""Tests for serialization and validation utilities."""
import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional
from uuid import UUID, uuid4

import pytest
from cosmos_coherence.benchmarks.models import EvaluationStrategy
from cosmos_coherence.benchmarks.models.datasets import (
    FaithBenchItem,
    FEVERItem,
    FEVERLabel,
    SimpleQAItem,
)
from cosmos_coherence.benchmarks.models.experiments import (
    BenchmarkMetrics,
    ExperimentResult,
    ExperimentRun,
    ExperimentStatus,
    ExperimentTracker,
    StatisticalSummary,
)
from cosmos_coherence.benchmarks.models.serialization import (
    BenchmarkJSONDecoder,
    BenchmarkJSONEncoder,
    JSONLReader,
    JSONLWriter,
    ModelSerializer,
    SchemaVersion,
    ValidationContext,
    batch_deserialize,
    batch_serialize,
    migrate_schema,
    validate_data_integrity,
)
from cosmos_coherence.config.models import BenchmarkType, CoherenceMeasure
from pydantic import BaseModel, ValidationError


class ComplexModel(BaseModel):
    """Test model with complex field types."""

    id: UUID
    created_at: datetime
    score: Decimal
    metadata: dict[str, Any]
    tags: list[str]
    nested: Optional["NestedModel"] = None


class NestedModel(BaseModel):
    """Nested model for testing."""

    value: str
    timestamp: datetime


class TestBenchmarkJSONEncoder:
    """Test custom JSON encoder."""

    def test_encode_uuid(self):
        """Test encoding UUID fields."""
        test_id = uuid4()
        encoder = BenchmarkJSONEncoder()
        result = encoder.encode({"id": test_id})
        assert json.loads(result)["id"] == str(test_id)

    def test_encode_datetime(self):
        """Test encoding datetime fields."""
        now = datetime.now(timezone.utc)
        encoder = BenchmarkJSONEncoder()
        result = encoder.encode({"timestamp": now})
        assert json.loads(result)["timestamp"] == now.isoformat()

    def test_encode_decimal(self):
        """Test encoding Decimal fields."""
        value = Decimal("3.14159")
        encoder = BenchmarkJSONEncoder()
        result = encoder.encode({"score": value})
        assert json.loads(result)["score"] == float(value)

    def test_encode_pydantic_model(self):
        """Test encoding Pydantic models."""
        model = ComplexModel(
            id=uuid4(),
            created_at=datetime.now(timezone.utc),
            score=Decimal("95.5"),
            metadata={"key": "value"},
            tags=["test", "benchmark"],
        )
        encoder = BenchmarkJSONEncoder()
        result = encoder.encode(model)
        data = json.loads(result)
        assert data["id"] == str(model.id)
        assert "created_at" in data
        assert data["score"] == 95.5

    def test_encode_nested_models(self):
        """Test encoding nested Pydantic models."""
        nested = NestedModel(value="test", timestamp=datetime.now(timezone.utc))
        model = ComplexModel(
            id=uuid4(),
            created_at=datetime.now(timezone.utc),
            score=Decimal("100"),
            metadata={},
            tags=[],
            nested=nested,
        )
        encoder = BenchmarkJSONEncoder()
        result = encoder.encode(model)
        data = json.loads(result)
        assert data["nested"]["value"] == "test"
        assert "timestamp" in data["nested"]


class TestBenchmarkJSONDecoder:
    """Test custom JSON decoder."""

    def test_decode_uuid(self):
        """Test decoding UUID strings."""
        test_id = str(uuid4())
        decoder = BenchmarkJSONDecoder()
        data = f'{{"id": "{test_id}", "__type__": "UUID"}}'
        result = decoder.decode(data)
        assert isinstance(result["id"], UUID)
        assert str(result["id"]) == test_id

    def test_decode_datetime(self):
        """Test decoding ISO format datetime strings."""
        now = datetime.now(timezone.utc)
        decoder = BenchmarkJSONDecoder()
        data = f'{{"timestamp": "{now.isoformat()}", "__type__": "datetime"}}'
        result = decoder.decode(data)
        assert isinstance(result["timestamp"], datetime)
        assert result["timestamp"] == now

    def test_decode_decimal(self):
        """Test decoding Decimal values."""
        decoder = BenchmarkJSONDecoder()
        data = '{"score": 3.14159, "__type__": "Decimal"}'
        result = decoder.decode(data)
        assert isinstance(result["score"], Decimal)
        assert result["score"] == Decimal("3.14159")

    def test_decode_with_model_type(self):
        """Test decoding with model type hints."""
        data = {
            "id": str(uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "score": 95.5,
            "metadata": {"key": "value"},
            "tags": ["test"],
            "__type__": "ComplexModel",
        }
        decoder = BenchmarkJSONDecoder()
        json_str = json.dumps(data)
        result = decoder.decode(json_str)
        assert "__type__" in result
        assert result["__type__"] == "ComplexModel"


class TestJSONLStreaming:
    """Test JSONL streaming functionality."""

    def test_jsonl_writer(self, tmp_path):
        """Test writing objects to JSONL file."""
        file_path = tmp_path / "test.jsonl"

        with JSONLWriter(file_path) as writer:
            for i in range(10):
                item = SimpleQAItem(
                    id=str(uuid4()),
                    question=f"Question {i}",
                    best_answer=f"Answer {i}",
                    correct_answers=[f"Answer {i}"],
                    incorrect_answers=[],
                )
                writer.write(item)

        # Verify file contents
        with open(file_path, "r") as f:
            lines = f.readlines()

        assert len(lines) == 10
        for i, line in enumerate(lines):
            data = json.loads(line)
            assert f"Question {i}" in data["question"]

    def test_jsonl_reader(self, tmp_path):
        """Test reading objects from JSONL file."""
        file_path = tmp_path / "test.jsonl"

        # Write test data
        items = []
        with open(file_path, "w") as f:
            for i in range(5):
                item = {
                    "id": str(uuid4()),
                    "question": f"Question {i}",
                    "best_answer": f"Answer {i}",
                    "correct_answers": [f"Answer {i}"],
                    "incorrect_answers": [],
                }
                items.append(item)
                f.write(json.dumps(item) + "\n")

        # Read and verify
        with JSONLReader(file_path, model_class=SimpleQAItem) as reader:
            read_items = list(reader)

        assert len(read_items) == 5
        for i, item in enumerate(read_items):
            assert isinstance(item, SimpleQAItem)
            assert item.question == f"Question {i}"

    def test_jsonl_streaming_large_dataset(self, tmp_path):
        """Test streaming large datasets efficiently."""
        file_path = tmp_path / "large.jsonl"
        num_items = 1000

        # Write large dataset
        with JSONLWriter(file_path, buffer_size=100) as writer:
            for i in range(num_items):
                item = FaithBenchItem(
                    id=str(uuid4()),
                    question=f"Is claim {i} factual?",
                    claim=f"Claim {i}",
                    context=f"Context {i}" * 100,  # Large text
                    evidence_sentences=["Evidence 1", "Evidence 2"],
                )
                writer.write(item)

        # Read with streaming
        count = 0
        with JSONLReader(file_path, model_class=FaithBenchItem, batch_size=50) as reader:
            for batch in reader.read_batches():
                assert len(batch) <= 50
                count += len(batch)

        assert count == num_items

    def test_jsonl_error_handling(self, tmp_path):
        """Test error handling in JSONL operations."""
        file_path = tmp_path / "corrupted.jsonl"

        # Write mix of valid and invalid data
        with open(file_path, "w") as f:
            f.write('{"id": "123", "question": "Valid"}\n')
            f.write("invalid json\n")
            f.write('{"id": "456", "question": "Also valid"}\n')

        # Read with error handling
        valid_items = []
        errors = []

        with JSONLReader(file_path, skip_errors=True) as reader:
            reader.on_error = lambda e, line_num: errors.append((e, line_num))
            for item in reader:
                valid_items.append(item)

        assert len(valid_items) == 2
        assert len(errors) == 1
        assert errors[0][1] == 2  # Line number of invalid JSON


class TestFieldValidators:
    """Test comprehensive field validators."""

    def test_required_field_validation(self):
        """Test required field validation."""
        with pytest.raises(ValidationError) as exc_info:
            SimpleQAItem(
                question="Test",
                # Missing best_answer
                correct_answers=[],
                incorrect_answers=[],
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("best_answer",) for e in errors)

    def test_string_length_validation(self):
        """Test string length constraints."""
        with pytest.raises(ValidationError) as exc_info:
            ExperimentTracker(
                name="x" * 300,  # Exceeds max_length
                benchmark_type=BenchmarkType.FAITHBENCH,
                strategy=EvaluationStrategy.K_RESPONSE,
            )

        errors = exc_info.value.errors()
        assert any("at most 255 characters" in str(e) for e in errors)

    def test_numeric_range_validation(self):
        """Test numeric range constraints."""
        with pytest.raises(ValidationError) as exc_info:
            BenchmarkMetrics(
                experiment_id=uuid4(),
                run_id=uuid4(),
                total_items=100,
                correct_items=85,
                accuracy=1.5,  # Should be between 0 and 1
                precision=0.9,
                recall=0.8,
                f1_score=0.85,
            )

        errors = exc_info.value.errors()
        assert any("less than or equal to 1" in str(e) for e in errors)

    def test_enum_validation(self):
        """Test enum field validation."""
        with pytest.raises(ValidationError) as exc_info:
            FEVERItem(
                id="test",
                claim="Test claim",
                label="INVALID_LABEL",  # Invalid enum value
                evidence=[],
            )

        errors = exc_info.value.errors()
        # Check for either old or new Pydantic error message format
        assert any(
            "not a valid enumeration member" in str(e) or "Input should be" in str(e)
            for e in errors
        )

    def test_custom_validators(self):
        """Test custom field validators."""
        # Test cross-field validation
        with pytest.raises(ValidationError) as exc_info:
            StatisticalSummary(
                metric_name="test_metric",
                mean=50.0,
                median=45.0,
                std_dev=-5.0,  # Invalid negative std_dev
                min_value=10.0,
                max_value=90.0,
                count=100,
            )

        errors = exc_info.value.errors()
        assert any("negative" in str(e).lower() or "std_dev" in str(e) for e in errors)


class TestDataIntegrity:
    """Test data integrity checks."""

    def test_cross_field_validation(self):
        """Test cross-field validation logic."""
        # Test min/max consistency
        with pytest.raises(ValidationError) as exc_info:
            StatisticalSummary(
                metric_name="test_metric",
                mean=50.0,
                median=45.0,
                std_dev=10.0,
                min_value=60.0,  # Min > Max
                max_value=40.0,
                count=100,
            )

        errors = exc_info.value.errors()
        assert any("min_value" in str(e) and "max_value" in str(e) for e in errors)

    def test_referential_integrity(self):
        """Test referential integrity between models."""
        tracker = ExperimentTracker(
            id=uuid4(),
            name="Test",
            benchmark_type=BenchmarkType.SIMPLEQA,
            strategy=EvaluationStrategy.K_RESPONSE,
            model_name="gpt-4o-mini",
        )

        # Create run with valid tracker reference
        run = ExperimentRun(
            id=uuid4(),
            experiment_id=tracker.id,
            run_number=1,
            dataset_size=100,
            status=ExperimentStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
        )

        # Validate reference
        assert run.experiment_id == tracker.id

        # Test integrity check function
        context = ValidationContext(
            trackers={tracker.id: tracker},
            runs={run.id: run},
        )

        errors = validate_data_integrity(context)
        assert len(errors) == 0

        # Test with missing reference
        orphan_run = ExperimentRun(
            id=uuid4(),
            experiment_id=uuid4(),  # Non-existent tracker
            run_number=2,
            dataset_size=100,
            status=ExperimentStatus.COMPLETED,
            started_at=datetime.now(timezone.utc),
        )

        context.runs[orphan_run.id] = orphan_run
        errors = validate_data_integrity(context)
        assert len(errors) > 0
        assert "experiment_id" in str(errors[0])

    def test_data_consistency(self):
        """Test data consistency validation."""
        metrics = BenchmarkMetrics(
            experiment_id=uuid4(),
            run_id=uuid4(),
            total_items=100,
            correct_items=90,
            accuracy=0.9,
            precision=0.85,
            recall=0.95,
            f1_score=0.5,  # Inconsistent with precision/recall
        )

        # Calculate expected F1
        expected_f1 = 2 * (0.85 * 0.95) / (0.85 + 0.95)

        # Validate consistency
        warnings = metrics.validate_consistency()
        assert len(warnings) > 0
        assert abs(metrics.f1_score - expected_f1) > 0.01


class TestSchemaVersioning:
    """Test schema versioning and migration."""

    def test_schema_version_tracking(self):
        """Test tracking schema versions."""
        v1 = SchemaVersion(major=1, minor=0, patch=0)
        v2 = SchemaVersion(major=1, minor=1, patch=0)
        v3 = SchemaVersion(major=2, minor=0, patch=0)

        assert v1 < v2 < v3
        assert v1.is_compatible_with(v2)  # Minor version bump
        assert not v1.is_compatible_with(v3)  # Major version bump

    def test_schema_migration(self):
        """Test migrating data between schema versions."""
        # Old schema data
        old_data = {
            "schema_version": "1.0.0",
            "id": str(uuid4()),
            "name": "Old format",
            "score": 95,  # Old field name
        }

        # Define migration
        def migrate_1_to_2(data: dict) -> dict:
            """Migrate from v1 to v2 schema."""
            if "score" in data:
                data["accuracy"] = data.pop("score") / 100.0
            data["schema_version"] = "2.0.0"
            return data

        # Apply migration
        new_data = migrate_schema(
            old_data,
            from_version=SchemaVersion(1, 0, 0),
            to_version=SchemaVersion(2, 0, 0),
            migrations={
                (SchemaVersion(1, 0, 0), SchemaVersion(2, 0, 0)): migrate_1_to_2,
            },
        )

        assert "score" not in new_data
        assert new_data["accuracy"] == 0.95
        assert new_data["schema_version"] == "2.0.0"

    def test_backward_compatibility(self):
        """Test backward compatibility for reading old schemas."""
        # Test reading v1 data with v2 reader
        v1_data = {
            "id": str(uuid4()),
            "question": "Test question",
            "answer": "Test answer",  # v1 field
        }

        # v2 expects "best_answer" instead of "answer"
        def compatibility_transform(data: dict) -> dict:
            if "answer" in data and "best_answer" not in data:
                data["best_answer"] = data.pop("answer")
            return data

        transformed = compatibility_transform(v1_data)
        assert "best_answer" in transformed
        assert transformed["best_answer"] == "Test answer"


class TestBatchOperations:
    """Test batch serialization operations."""

    def test_batch_serialize(self, tmp_path):
        """Test batch serialization of models."""
        items = [
            SimpleQAItem(
                id=str(uuid4()),
                question=f"Question {i}",
                best_answer=f"Answer {i}",
                correct_answers=[f"Answer {i}"],
                incorrect_answers=[],
            )
            for i in range(100)
        ]

        output_file = tmp_path / "batch.json"

        # Batch serialize
        batch_serialize(
            items,
            output_file,
            batch_size=10,
            format="json",
        )

        # Verify output
        with open(output_file, "r") as f:
            data = json.load(f)

        assert len(data) == 100
        assert all("question" in item for item in data)

    def test_batch_deserialize(self, tmp_path):
        """Test batch deserialization of models."""
        # Create test data
        data = [
            {
                "id": str(uuid4()),
                "question": f"Question {i}",
                "best_answer": f"Answer {i}",
                "correct_answers": [f"Answer {i}"],
                "incorrect_answers": [],
            }
            for i in range(50)
        ]

        input_file = tmp_path / "batch.json"
        with open(input_file, "w") as f:
            json.dump(data, f)

        # Batch deserialize
        items = batch_deserialize(
            input_file,
            model_class=SimpleQAItem,
            batch_size=10,
        )

        items_list = list(items)
        assert len(items_list) == 50
        assert all(isinstance(item, SimpleQAItem) for item in items_list)

    def test_batch_validation(self):
        """Test batch validation of items."""
        valid_items = [
            SimpleQAItem(
                id=str(uuid4()),
                question=f"Question {i}",
                best_answer=f"Answer {i}",
                correct_answers=[f"Answer {i}"],
                incorrect_answers=[],
            )
            for i in range(5)
        ]

        invalid_items = [
            {"question": "Missing fields"},  # Invalid
            {"id": "123", "question": "Also invalid"},  # Invalid
        ]

        # Validate batch
        results = []
        errors = []

        for item in valid_items + invalid_items:
            try:
                if isinstance(item, dict):
                    validated = SimpleQAItem(**item)
                else:
                    validated = item
                results.append(validated)
            except ValidationError as e:
                errors.append(e)

        assert len(results) == 5
        assert len(errors) == 2


class TestPerformanceOptimizations:
    """Test performance optimizations for large datasets."""

    def test_lazy_loading(self, tmp_path):
        """Test lazy loading of large datasets."""
        file_path = tmp_path / "large.jsonl"

        # Create large dataset
        with JSONLWriter(file_path) as writer:
            for i in range(10000):
                item = SimpleQAItem(
                    id=str(uuid4()),
                    question=f"Q{i}",
                    best_answer=f"A{i}",
                    correct_answers=[f"A{i}"],
                    incorrect_answers=[],
                )
                writer.write(item)

        # Test lazy loading doesn't load everything into memory
        with JSONLReader(file_path, model_class=SimpleQAItem) as reader:
            # Process only first 100 items
            count = 0
            for item in reader:
                count += 1
                if count >= 100:
                    break

            assert count == 100

    def test_streaming_aggregation(self, tmp_path):
        """Test streaming aggregation without loading full dataset."""
        file_path = tmp_path / "scores.jsonl"

        # Write score data
        with JSONLWriter(file_path) as writer:
            for i in range(1000):
                result = ExperimentResult(
                    id=uuid4(),
                    experiment_id=uuid4(),
                    run_id=uuid4(),
                    dataset_item_id=str(uuid4()),
                    confidence_score=float(i % 100) / 100.0,
                    coherence_scores={
                        CoherenceMeasure.SHOGENJI: float(i % 100) / 100.0,
                    },
                )
                writer.write(result)

        # Stream and aggregate
        total_score = 0.0
        count = 0

        with JSONLReader(file_path, model_class=ExperimentResult) as reader:
            for result in reader:
                total_score += result.confidence_score or 0.0
                count += 1

        avg_score = total_score / count
        assert 0.4 < avg_score < 0.6  # Should be around 0.5

    def test_chunked_processing(self):
        """Test processing data in chunks for memory efficiency."""
        # Create large dataset in memory
        items = [
            FaithBenchItem(
                id=str(uuid4()),
                question=f"Is claim {i} factual?",
                claim=f"Claim {i}" * 100,  # Large text
                context=f"Context {i}" * 200,  # Even larger
                evidence_sentences=[f"Evidence {j}" for j in range(10)],
            )
            for i in range(100)
        ]

        # Process in chunks
        chunk_size = 10
        processed_count = 0

        for i in range(0, len(items), chunk_size):
            chunk = items[i : i + chunk_size]
            # Simulate processing
            serialized = [item.model_dump_json() for item in chunk]
            processed_count += len(serialized)

        assert processed_count == len(items)


class TestErrorMessages:
    """Test meaningful error messages."""

    def test_validation_error_messages(self):
        """Test that validation errors have clear messages."""
        try:
            SimpleQAItem(
                question="",  # Empty question
                best_answer="Answer",
                correct_answers=[],
                incorrect_answers=[],
            )
        except ValidationError as e:
            error_str = str(e)
            assert "question" in error_str
            assert (
                "at least 1 character" in error_str
                or "ensure this value" in error_str
                or "cannot be empty" in error_str
            )

    def test_type_error_messages(self):
        """Test type error messages."""
        try:
            SimpleQAItem(
                id=123,  # Should be string
                question="Test",
                best_answer="Answer",
                correct_answers=[],
                incorrect_answers=[],
            )
        except ValidationError as e:
            error_str = str(e)
            assert "id" in error_str
            assert "string" in error_str.lower()

    def test_custom_error_messages(self):
        """Test custom validation error messages."""
        with pytest.raises(ValidationError) as exc_info:
            BenchmarkMetrics(
                accuracy=-0.1,  # Negative accuracy
                precision=0.9,
                recall=0.8,
                f1_score=0.85,
            )

        errors = exc_info.value.errors()
        assert any(
            "greater than or equal to 0" in str(e) or "Accuracy must be between 0 and 1" in str(e)
            for e in errors
        )


class TestModelSerializer:
    """Test the ModelSerializer utility class."""

    def test_serialize_to_dict(self):
        """Test serializing models to dictionaries."""
        item = SimpleQAItem(
            id=str(uuid4()),
            question="What is 2+2?",
            best_answer="4",
        )

        serializer = ModelSerializer()
        data = serializer.to_dict(item)

        assert isinstance(data, dict)
        assert data["question"] == "What is 2+2?"
        assert data["best_answer"] == "4"
        assert "id" in data
        assert "created_at" in data

    def test_serialize_to_json(self):
        """Test serializing models to JSON."""
        item = FEVERItem(
            id=str(uuid4()),
            question="Is the Earth flat?",
            claim="The Earth is flat",
            label=FEVERLabel.REFUTED,
            evidence=[["Earth", 1], ["Satellite_imagery", 3]],
        )

        serializer = ModelSerializer()
        json_str = serializer.to_json(item, indent=2)

        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["claim"] == "The Earth is flat"
        assert data["label"] == "REFUTED"

    def test_deserialize_from_dict(self):
        """Test deserializing models from dictionaries."""
        data = {
            "id": str(uuid4()),
            "question": "Test question",
            "best_answer": "Test answer",
        }

        serializer = ModelSerializer()
        item = serializer.from_dict(data, model_class=SimpleQAItem)

        assert isinstance(item, SimpleQAItem)
        assert item.question == "Test question"
        assert item.best_answer == "Test answer"

    def test_deserialize_from_json(self):
        """Test deserializing models from JSON."""
        json_str = json.dumps(
            {
                "id": str(uuid4()),
                "question": "Is this claim factual?",
                "claim": "Test claim",
                "context": "Test context",
                "evidence": ["Evidence 1", "Evidence 2"],
            }
        )

        serializer = ModelSerializer()
        item = serializer.from_json(json_str, model_class=FaithBenchItem)

        assert isinstance(item, FaithBenchItem)
        assert item.claim == "Test claim"
        assert len(item.evidence) == 2
