"""Base models for benchmark data structures.

This module provides the data models for benchmark execution and tracking.
It imports configuration enums from config.models to maintain a single source of truth.

Key distinction:
- config.models.BenchmarkConfig: Defines WHAT to run (experiment configuration)
- benchmarks.models.BenchmarkRunConfig: Tracks HOW it's running (execution state)
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

# Import shared enums from the configuration module (single source of truth)
from cosmos_coherence.config.models import (
    BenchmarkType,
    CoherenceMeasure,
)
from cosmos_coherence.config.models import (
    StrategyType as EvaluationStrategy,  # Alias for backward compatibility
)

# Type variables for generic types
T_Input = TypeVar("T_Input")
T_Output = TypeVar("T_Output")


class BenchmarkValidationError(ValueError):
    """Raised when benchmark validation fails."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        error_code: Optional[str] = None,
    ):
        """Initialize validation error."""
        super().__init__(message)
        self.field = field
        self.value = value
        self.error_code = error_code
        self.context = {"field": field, "value": value, "error_code": error_code}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": str(self),
            "field": self.field,
            "value": str(self.value) if self.value is not None else None,
            "error_code": self.error_code,
        }


class ConfigurationError(BenchmarkValidationError):
    """Raised when experiment configuration is invalid."""

    pass


class DatasetValidationError(BenchmarkValidationError):
    """Raised when dataset validation fails."""

    pass


class ValidationMixin:
    """Mixin class providing common validation patterns."""

    @staticmethod
    def validate_non_empty_string(value: str, field_name: str = "field") -> str:
        """Validate that a string is not empty."""
        if not value or not value.strip():
            raise ValueError(f"{field_name} cannot be empty")
        return value.strip()

    @staticmethod
    def validate_uuid_format(value: Union[str, UUID]) -> UUID:
        """Validate UUID format."""
        if isinstance(value, str):
            try:
                return UUID(value)
            except ValueError:
                raise ValueError(f"Invalid UUID format: {value}")
        return value

    @staticmethod
    def validate_score_range(value: float, field_name: str = "score") -> float:
        """Validate score is in 0.0-1.0 range."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field_name} must be between 0.0 and 1.0, got {value}")
        return value

    @staticmethod
    def validate_temperature_range(value: float) -> float:
        """Validate temperature is in 0.3-1.0 range."""
        if not 0.3 <= value <= 1.0:
            raise ValueError(f"Temperature must be between 0.3 and 1.0, got {value}")
        return value

    @staticmethod
    def validate_url_format(value: str) -> str:
        """Validate URL format."""
        if not value.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL format: {value}")
        return value

    @staticmethod
    def validate_iso_datetime(value: Union[str, datetime]) -> datetime:
        """Validate ISO datetime format."""
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                raise ValueError(f"Invalid ISO datetime format: {value}")
        return value


class BaseDatasetItem(BaseModel, ValidationMixin, ABC):
    """Abstract base class for all benchmark dataset items."""

    id: UUID = Field(default_factory=uuid4, description="Unique identifier")
    question: str = Field(..., description="The question or prompt")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    version: str = Field(default="1.0", description="Model version")

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        """Validate question is not empty."""
        return cls.validate_non_empty_string(v, "question")

    @abstractmethod
    def validate_content(self) -> None:
        """Validate the content of the dataset item."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseDatasetItem":
        """Create instance from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "BaseDatasetItem":
        """Create instance from JSON string."""
        return cls(**json.loads(json_str))

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }


class BaseExperiment(BaseModel, ValidationMixin, ABC):
    """Abstract base class for experiment configurations."""

    experiment_id: UUID = Field(default_factory=uuid4, description="Experiment ID")
    model_name: str = Field(..., description="Model name")
    temperature: float = Field(0.7, description="Temperature setting")
    benchmark_type: BenchmarkType = Field(..., description="Benchmark type")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Experiment timestamp")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Additional configuration")
    version: str = Field(default="1.0", description="Model version")

    @field_validator("temperature")
    @classmethod
    def validate_temp(cls, v: float) -> float:
        """Validate temperature range."""
        return cls.validate_temperature_range(v)

    @field_validator("model_name")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate model name."""
        return cls.validate_non_empty_string(v, "model_name")

    def get_config(self) -> Dict[str, Any]:
        """Get experiment configuration."""
        return {
            "experiment_id": str(self.experiment_id),
            "model_name": self.model_name,
            "temperature": self.temperature,
            "benchmark_type": self.benchmark_type.value,
            "timestamp": self.timestamp.isoformat(),
            "config": self.config or {},
        }

    @abstractmethod
    def validate_parameters(self) -> None:
        """Validate experiment parameters."""
        pass

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }


class BaseResult(BaseModel, ValidationMixin, ABC):
    """Abstract base class for experiment results."""

    experiment_id: UUID = Field(..., description="Experiment ID")
    item_id: UUID = Field(..., description="Dataset item ID")
    prediction: str = Field(..., description="Model prediction")
    ground_truth: str = Field(..., description="Ground truth answer")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Metrics")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Result timestamp")
    version: str = Field(default="1.0", description="Model version")

    @field_validator("prediction", "ground_truth")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate prediction and ground truth."""
        return cls.validate_non_empty_string(v)

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate metrics are in valid range."""
        for name, value in v.items():
            if name in ["accuracy", "f1_score", "precision", "recall", "consistency_score"]:
                cls.validate_score_range(value, name)
        return v

    @abstractmethod
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        pass

    def serialize(self) -> Dict[str, Any]:
        """Serialize result to dictionary."""
        return self.model_dump(mode="json")

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }


class BenchmarkRunConfig(BaseModel, ValidationMixin):
    """Runtime configuration for benchmark execution.

    This class tracks HOW a benchmark is being executed, including runtime parameters
    like temperature variations, coherence measures, and response generation settings.
    It complements config.models.BenchmarkConfig which defines WHAT benchmark to run.

    Key differences from config.models.BenchmarkConfig:
    - Includes runtime execution parameters (temperature_settings, k_responses)
    - Tracks evaluation strategy with coherence measures
    - Manages response generation settings (shuffle, k_responses)
    - Validates runtime constraints (temperature ranges, coherence requirements)
    """

    benchmark_type: BenchmarkType = Field(..., description="Type of benchmark")
    dataset_path: Path = Field(..., description="Path to dataset")
    sample_size: Optional[int] = Field(None, ge=1, description="Sample size")
    metrics: List[str] = Field(
        default_factory=lambda: ["accuracy", "f1_score"],
        description="Metrics to calculate",
    )
    evaluation_strategy: EvaluationStrategy = Field(
        default=EvaluationStrategy.BASELINE,
        description="Evaluation strategy (imported from config.models.StrategyType)",
    )
    temperature_settings: List[float] = Field(
        default_factory=lambda: [0.3, 0.5, 0.7, 1.0],
        description="Temperature variations",
    )
    coherence_measures: List[CoherenceMeasure] = Field(
        default_factory=list, description="Coherence measures to use"
    )
    k_responses: int = Field(default=5, ge=1, le=10, description="Number of responses")
    shuffle: bool = Field(default=True, description="Shuffle dataset")
    evaluation_params: Dict[str, Any] = Field(
        default_factory=dict, description="Additional evaluation parameters"
    )

    @field_validator("temperature_settings")
    @classmethod
    def validate_temperatures(cls, v: List[float]) -> List[float]:
        """Validate all temperatures are in range."""
        for temp in v:
            cls.validate_temperature_range(temp)
        return v

    @field_validator("metrics")
    @classmethod
    def validate_metrics_list(cls, v: List[str]) -> List[str]:
        """Validate metrics list."""
        valid_metrics = {
            "accuracy",
            "f1_score",
            "precision",
            "recall",
            "consistency_score",
            "truthfulness_score",
            "informativeness_score",
        }
        invalid = set(v) - valid_metrics
        if invalid:
            raise ValueError(f"Invalid metrics: {invalid}")
        return v

    @model_validator(mode="after")
    def validate_coherence_config(self) -> "BenchmarkRunConfig":
        """Validate coherence configuration."""
        if self.evaluation_strategy == EvaluationStrategy.COHERENCE:
            if not self.coherence_measures:
                raise ValueError("Coherence measures must be specified for coherence evaluation")
        return self

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            Path: str,
        }


class DataPoint(BaseModel, Generic[T_Input, T_Output]):
    """Generic data point for benchmark processing."""

    input: T_Input = Field(..., description="Input data")
    output: Optional[T_Output] = Field(None, description="Output data")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Processing timestamp")
    source: Optional[str] = Field(None, description="Data source")
    id: UUID = Field(default_factory=uuid4, description="Unique identifier")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataPoint":
        """Create from dictionary."""
        return cls(**data)

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            UUID: str,
            datetime: lambda v: v.isoformat(),
        }


# Serialization helper functions
def serialize_to_json(obj: BaseModel) -> str:
    """Serialize Pydantic model to JSON."""
    return obj.model_dump_json()


def serialize_to_jsonl(objects: List[BaseModel]) -> str:
    """Serialize list of Pydantic models to JSONL."""
    lines = [obj.model_dump_json() for obj in objects]
    return "\n".join(lines)


def deserialize_from_json(json_str: str, model_class: type[BaseModel]) -> BaseModel:
    """Deserialize JSON to Pydantic model."""
    return model_class(**json.loads(json_str))


def deserialize_from_jsonl(jsonl_str: str, model_class: type[BaseModel]) -> List[BaseModel]:
    """Deserialize JSONL to list of Pydantic models."""
    lines = jsonl_str.strip().split("\n")
    return [model_class(**json.loads(line)) for line in lines if line]


def stream_jsonl(file_path: Path, model_class: type[BaseModel], batch_size: int = 100):
    """Stream JSONL file in batches."""
    with open(file_path, "r") as f:
        batch = []
        for line in f:
            if line.strip():
                batch.append(model_class(**json.loads(line)))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch
