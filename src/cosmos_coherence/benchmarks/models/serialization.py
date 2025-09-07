"""Serialization utilities for benchmark models.

This module provides JSON/JSONL serialization support with:
- Custom encoders/decoders for complex types
- Streaming support for large datasets
- Schema versioning and migration
- Data integrity validation
"""

import json
from collections.abc import Generator, Iterator
from datetime import datetime
from decimal import Decimal
from enum import Enum
from io import IOBase
from pathlib import Path
from typing import Any, Callable, Optional, Type, TypeVar, Union
from uuid import UUID

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class BenchmarkJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for benchmark models.

    Handles special types like UUID, datetime, Decimal, and Pydantic models.
    """

    def default(self, obj: Any) -> Any:
        """Encode special types to JSON-serializable format."""
        if isinstance(obj, UUID):
            return str(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, BaseModel):
            return obj.model_dump()
        elif isinstance(obj, bytes):
            return obj.decode("utf-8")
        elif hasattr(obj, "__dict__"):
            return obj.__dict__

        return super().default(obj)


class BenchmarkJSONDecoder(json.JSONDecoder):
    """Custom JSON decoder for benchmark models.

    Reconstructs special types from their JSON representations.
    """

    def __init__(self, *args, **kwargs):
        """Initialize decoder with custom object hook."""
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj: dict) -> Any:
        """Decode special types from JSON format."""
        if not isinstance(obj, dict):
            return obj

        # Check for type hints
        if "__type__" in obj:
            type_name = obj["__type__"]

            if type_name == "UUID" and "id" in obj:
                obj["id"] = UUID(obj["id"])
            elif type_name == "datetime":
                for key, value in obj.items():
                    if key != "__type__" and isinstance(value, str):
                        try:
                            obj[key] = datetime.fromisoformat(value)
                        except (ValueError, TypeError):
                            pass
            elif type_name == "Decimal":
                for key, value in obj.items():
                    if key != "__type__" and isinstance(value, (int, float)):
                        obj[key] = Decimal(str(value))

        return obj


class JSONLWriter:
    """Streaming JSONL writer for large datasets."""

    def __init__(
        self,
        file_path: Union[str, Path],
        encoder: Optional[BenchmarkJSONEncoder] = None,
        buffer_size: int = 100,
    ):
        """Initialize JSONL writer.

        Args:
            file_path: Path to output file
            encoder: Custom JSON encoder
            buffer_size: Number of items to buffer before writing
        """
        self.file_path = Path(file_path)
        self.encoder = encoder or BenchmarkJSONEncoder()
        self.buffer_size = buffer_size
        self.buffer: list[Any] = []
        self.file_handle: Optional[IOBase] = None
        self.items_written = 0

    def __enter__(self) -> "JSONLWriter":
        """Enter context manager."""
        self.file_handle = open(self.file_path, "w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and flush buffer."""
        self.flush()
        if self.file_handle:
            self.file_handle.close()

    def write(self, item: Any) -> None:
        """Write item to JSONL file."""
        self.buffer.append(item)

        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        """Flush buffer to file."""
        if not self.buffer or not self.file_handle:
            return

        for item in self.buffer:
            if isinstance(item, BaseModel):
                json_str = item.model_dump_json()
            else:
                json_str = self.encoder.encode(item)

            self.file_handle.write(json_str + "\n")
            self.items_written += 1

        self.buffer.clear()


class JSONLReader:
    """Streaming JSONL reader for large datasets."""

    def __init__(
        self,
        file_path: Union[str, Path],
        model_class: Optional[Type[T]] = None,
        decoder: Optional[BenchmarkJSONDecoder] = None,
        batch_size: int = 100,
        skip_errors: bool = False,
    ):
        """Initialize JSONL reader.

        Args:
            file_path: Path to input file
            model_class: Pydantic model class for validation
            decoder: Custom JSON decoder
            batch_size: Size of batches for batch reading
            skip_errors: Whether to skip invalid lines
        """
        self.file_path = Path(file_path)
        self.model_class = model_class
        self.decoder = decoder or BenchmarkJSONDecoder()
        self.batch_size = batch_size
        self.skip_errors = skip_errors
        self.file_handle: Optional[IOBase] = None
        self.on_error: Optional[Callable[[Exception, int], None]] = None
        self.line_number = 0

    def __enter__(self) -> "JSONLReader":
        """Enter context manager."""
        self.file_handle = open(self.file_path, "r", encoding="utf-8")
        self.line_number = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        if self.file_handle:
            self.file_handle.close()

    def __iter__(self) -> Iterator[T]:
        """Iterate over items in file."""
        if not self.file_handle:
            raise RuntimeError("Reader not opened. Use with statement.")

        for line_num, line in enumerate(self.file_handle, 1):
            self.line_number = line_num
            line_str = line.strip() if isinstance(line, str) else line.decode("utf-8").strip()

            if not line_str:
                continue

            try:
                data = self.decoder.decode(line_str)

                if self.model_class:
                    yield self.model_class(**data)
                else:
                    yield data

            except (json.JSONDecodeError, ValidationError) as e:
                if self.on_error:
                    self.on_error(e, line_num)

                if not self.skip_errors:
                    raise

    def read_batches(self) -> Generator[list[T], None, None]:
        """Read items in batches."""
        batch: list[T] = []
        item: T
        for item in self:
            batch.append(item)

            if len(batch) >= self.batch_size:
                yield batch
                batch = []

        if batch:
            yield batch


class SchemaVersion:
    """Schema version tracking."""

    def __init__(self, major: int, minor: int, patch: int):
        """Initialize schema version."""
        self.major = major
        self.minor = minor
        self.patch = patch

    def __str__(self) -> str:
        """String representation."""
        return f"{self.major}.{self.minor}.{self.patch}"

    def __lt__(self, other: "SchemaVersion") -> bool:
        """Compare versions."""
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def is_compatible_with(self, other: "SchemaVersion") -> bool:
        """Check if versions are compatible."""
        # Major version must match for compatibility
        return self.major == other.major

    @classmethod
    def from_string(cls, version_str: str) -> "SchemaVersion":
        """Create from version string."""
        parts = version_str.split(".")
        return cls(int(parts[0]), int(parts[1]), int(parts[2]))


class ValidationContext:
    """Context for data validation."""

    def __init__(self, **kwargs):
        """Initialize validation context with model collections."""
        self.trackers = kwargs.get("trackers", {})
        self.runs = kwargs.get("runs", {})
        self.results = kwargs.get("results", {})
        self.datasets = kwargs.get("datasets", {})


def validate_data_integrity(context: ValidationContext) -> list[str]:
    """Validate data integrity across models.

    Args:
        context: Validation context with model collections

    Returns:
        List of validation error messages
    """
    errors = []

    # Check referential integrity
    for run_id, run in context.runs.items():
        if run.tracker_id not in context.trackers:
            errors.append(f"Run {run_id} references non-existent tracker_id {run.tracker_id}")

    for result_id, result in context.results.items():
        if result.run_id not in context.runs:
            errors.append(f"Result {result_id} references non-existent run_id {result.run_id}")

    return errors


def migrate_schema(
    data: dict,
    from_version: SchemaVersion,
    to_version: SchemaVersion,
    migrations: dict[tuple[SchemaVersion, SchemaVersion], Callable[[dict], dict]],
) -> dict:
    """Migrate data between schema versions.

    Args:
        data: Data to migrate
        from_version: Source schema version
        to_version: Target schema version
        migrations: Migration functions mapping

    Returns:
        Migrated data
    """
    if from_version == to_version:
        return data

    # Find direct migration path
    for (from_v, to_v), migration_fn in migrations.items():
        if (
            from_v.major == from_version.major
            and from_v.minor == from_version.minor
            and from_v.patch == from_version.patch
        ):
            if (
                to_v.major == to_version.major
                and to_v.minor == to_version.minor
                and to_v.patch == to_version.patch
            ):
                return migration_fn(data)

    # If no direct path, raise error
    raise ValueError(f"No migration path from {from_version} to {to_version}")


def batch_serialize(
    items: list[BaseModel],
    output_path: Union[str, Path],
    batch_size: int = 100,
    format: str = "json",
) -> None:
    """Serialize items in batches.

    Args:
        items: Items to serialize
        output_path: Output file path
        batch_size: Batch size for processing
        format: Output format (json or jsonl)
    """
    output_path = Path(output_path)

    if format == "jsonl":
        with JSONLWriter(output_path, buffer_size=batch_size) as writer:
            for item in items:
                writer.write(item)
    else:
        # JSON array format
        data = [item.model_dump() for item in items]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=BenchmarkJSONEncoder, indent=2)


def batch_deserialize(
    input_path: Union[str, Path],
    model_class: Type[T],
    batch_size: int = 100,
) -> Generator[T, None, None]:
    """Deserialize items in batches.

    Args:
        input_path: Input file path
        model_class: Model class for validation
        batch_size: Batch size for processing

    Yields:
        Deserialized items
    """
    input_path = Path(input_path)

    if input_path.suffix == ".jsonl":
        with JSONLReader(input_path, model_class=model_class, batch_size=batch_size) as reader:
            item: T
            for item in reader:
                yield item
    else:
        # JSON array format
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f, cls=BenchmarkJSONDecoder)

        for item_data in data:
            yield model_class(**item_data)


class ModelSerializer:
    """Utility class for model serialization."""

    def __init__(
        self,
        encoder: Optional[BenchmarkJSONEncoder] = None,
        decoder: Optional[BenchmarkJSONDecoder] = None,
    ):
        """Initialize serializer."""
        self.encoder = encoder or BenchmarkJSONEncoder()
        self.decoder = decoder or BenchmarkJSONDecoder()

    def to_dict(self, model: BaseModel) -> dict:
        """Convert model to dictionary."""
        return model.model_dump()

    def to_json(self, model: BaseModel, **kwargs) -> str:
        """Convert model to JSON string."""
        return model.model_dump_json(**kwargs)

    def from_dict(self, data: dict, model_class: Type[T]) -> T:
        """Create model from dictionary."""
        return model_class(**data)

    def from_json(self, json_str: str, model_class: Type[T]) -> T:
        """Create model from JSON string."""
        data = self.decoder.decode(json_str)
        return model_class(**data)
