"""Pydantic configuration models for Cosmos Coherence."""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(str, Enum):
    """Logging level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelType(str, Enum):
    """OpenAI model type enumeration."""

    # GPT-5 Family (Latest - August 2025)
    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"
    GPT_5_CHAT = "gpt-5-chat"

    # GPT-4.1 Family (April 2025)
    GPT_41 = "gpt-4.1"
    GPT_41_MINI = "gpt-4.1-mini"
    GPT_41_NANO = "gpt-4.1-nano"

    # GPT-4 Family (Legacy but still supported)
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo-preview"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"

    # Reasoning Models (o-Series)
    O3 = "o3"
    O3_MINI = "o3-mini"
    O4_MINI = "o4-mini"
    O1_PREVIEW = "o1-preview"  # Legacy
    O1_MINI = "o1-mini"  # Legacy

    # Legacy
    GPT_35_TURBO = "gpt-3.5-turbo"


class BenchmarkType(str, Enum):
    """Benchmark type enumeration."""

    FAITHBENCH = "faithbench"
    SIMPLEQA = "simpleqa"
    TRUTHFULQA = "truthfulqa"
    FEVER = "fever"
    HALUEVAL = "halueval"


class StrategyType(str, Enum):
    """Evaluation strategy type enumeration."""

    BASELINE = "baseline"  # Single response, no coherence
    K_RESPONSE = "k_response"  # Multiple responses with majority voting
    COHERENCE = "coherence"  # Coherence-based selection


class CoherenceMeasure(str, Enum):
    """Coherence measure enumeration."""

    SHOGENJI = "shogenji"
    FITELSON = "fitelson"
    OLSSON = "olsson"


class BaseConfig(BaseSettings):
    """Base configuration with common settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,  # Allow both field name and alias to be used
    )

    api_key: str = Field(
        default="",
        description="OpenAI API key",
        alias="OPENAI_API_KEY",
    )
    output_dir: Path = Field(
        default=Path("outputs"),
        description="Output directory for results",
        alias="OUTPUT_DIR",
    )
    cache_dir: Path = Field(
        default=Path(".cache"),
        description="Cache directory for datasets and models",
        alias="CACHE_DIR",
    )
    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level",
        alias="LOG_LEVEL",
    )
    log_file: Optional[Path] = Field(
        default=None,
        description="Log file path",
        alias="LOG_FILE",
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
        alias="RANDOM_SEED",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum API retry attempts",
        alias="MAX_RETRIES",
    )
    timeout: int = Field(
        default=60,
        description="API timeout in seconds",
        alias="TIMEOUT",
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseConfig":
        """Create BaseConfig from dictionary without environment variable loading.

        This is used when loading from YAML where environment variables
        have already been interpolated.
        """
        # Bypass environment loading by using direct construction
        # First validate the data
        for field_name, field in cls.model_fields.items():
            if field_name in data:
                continue
            # Use field name, not alias
            actual_name = field_name
            if actual_name in data:
                continue
            # Check if it has a default
            if field.default is not None:
                data[field_name] = field.default
            elif field.default_factory is not None:
                # Skip default factories - let Pydantic handle them
                pass

        return cls(**data)

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key is not empty."""
        if not v or not v.strip():
            raise ValueError("API key cannot be empty")
        return v

    @field_validator("output_dir", "cache_dir", "log_file")
    @classmethod
    def convert_to_path(cls, v: Any) -> Optional[Path]:
        """Convert string to Path object."""
        if v is None:
            return None
        return Path(v) if not isinstance(v, Path) else v


class ModelConfig(BaseModel):
    """OpenAI model configuration."""

    model_type: ModelType = Field(
        default=ModelType.GPT_5,
        description="OpenAI model type",
    )
    temperature: float = Field(
        default=1.0,
        description="Sampling temperature (unsupported for GPT-5 and o-series, fixed at 1)",
        ge=0.0,
        le=2.0,
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens in response (not supported for reasoning models)",
        gt=0,
    )
    max_completion_tokens: Optional[int] = Field(
        default=None,
        description="Maximum completion tokens (reasoning models and GPT-5)",
        gt=0,
    )
    max_output_tokens: Optional[int] = Field(
        default=None,
        description="Maximum output tokens (GPT-5: 128K, o3: 100K, GPT-4.1: 32K)",
        gt=0,
    )
    reasoning_effort: Optional[str] = Field(
        default=None,
        description="Reasoning effort level for o-series models",
        pattern="^(low|medium|high)$",
    )
    top_p: Optional[float] = Field(
        default=None,
        description="Top-p sampling parameter",
        ge=0.0,
        le=1.0,
    )
    frequency_penalty: Optional[float] = Field(
        default=None,
        description="Frequency penalty",
        ge=-2.0,
        le=2.0,
    )
    presence_penalty: Optional[float] = Field(
        default=None,
        description="Presence penalty",
        ge=-2.0,
        le=2.0,
    )
    response_format: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Response format specification",
    )

    @model_validator(mode="after")
    def validate_model_specific_params(self) -> "ModelConfig":
        """Validate model-specific parameter constraints."""
        # Define model categories
        reasoning_models = [
            ModelType.O1_PREVIEW,
            ModelType.O1_MINI,
            ModelType.O3,
            ModelType.O3_MINI,
            ModelType.O4_MINI,
        ]

        gpt5_models = [
            ModelType.GPT_5,
            ModelType.GPT_5_MINI,
            ModelType.GPT_5_NANO,
            ModelType.GPT_5_CHAT,
        ]

        gpt41_models = [ModelType.GPT_41, ModelType.GPT_41_MINI, ModelType.GPT_41_NANO]

        legacy_gpt4_models = [
            ModelType.GPT_4,
            ModelType.GPT_4_TURBO,
            ModelType.GPT_4O,
            ModelType.GPT_4O_MINI,
            ModelType.GPT_35_TURBO,
        ]

        # Check if it's a reasoning model or GPT-5
        is_reasoning_model = self.model_type in reasoning_models
        is_gpt5_model = self.model_type in gpt5_models
        is_gpt41_model = self.model_type in gpt41_models
        is_legacy_model = self.model_type in legacy_gpt4_models

        # GPT-5 and reasoning models don't support temperature control
        if (is_reasoning_model or is_gpt5_model) and self.temperature != 1.0:
            # Temperature is already forced to 1 by validator, just ensure it's set
            pass

        # Parameter validation for different model types
        if is_reasoning_model:
            # Reasoning models use max_completion_tokens, not max_tokens
            if self.max_tokens is not None:
                raise ValueError(
                    f"{self.model_type} does not support max_tokens, "
                    "use max_completion_tokens instead"
                )

            # Reasoning models don't support these parameters
            unsupported = []
            if self.top_p is not None:
                unsupported.append("top_p")
            if self.frequency_penalty is not None:
                unsupported.append("frequency_penalty")
            if self.presence_penalty is not None:
                unsupported.append("presence_penalty")

            if unsupported:
                raise ValueError(
                    f"{self.model_type} does not support parameters: {', '.join(unsupported)}"
                )

            # Only o-series models support reasoning_effort
            if self.reasoning_effort and self.model_type not in [
                ModelType.O3,
                ModelType.O3_MINI,
                ModelType.O4_MINI,
            ]:
                raise ValueError("reasoning_effort is only supported for o3/o4 models")

        elif is_gpt5_model:
            # GPT-5 models use max_output_tokens
            if self.max_tokens is not None:
                raise ValueError(
                    f"{self.model_type} does not support max_tokens, "
                    "use max_output_tokens instead (max 128K)"
                )

            # GPT-5 doesn't support temperature, top_p, frequency_penalty, presence_penalty
            unsupported = []
            if self.top_p is not None:
                unsupported.append("top_p")
            if self.frequency_penalty is not None:
                unsupported.append("frequency_penalty")
            if self.presence_penalty is not None:
                unsupported.append("presence_penalty")

            if unsupported:
                raise ValueError(
                    f"{self.model_type} does not support parameters: {', '.join(unsupported)}"
                )

        elif is_gpt41_model:
            # GPT-4.1 models support max_tokens up to 32K output
            if self.max_output_tokens is not None and self.max_output_tokens > 32768:
                raise ValueError(f"{self.model_type} supports max 32,768 output tokens")

            # GPT-4.1 doesn't use max_completion_tokens
            if self.max_completion_tokens is not None:
                raise ValueError(f"{self.model_type} uses max_tokens, not max_completion_tokens")

        elif is_legacy_model:
            # Legacy models use max_tokens, not the newer parameters
            if self.max_completion_tokens is not None:
                raise ValueError(f"{self.model_type} uses max_tokens, not max_completion_tokens")
            if self.max_output_tokens is not None:
                raise ValueError(f"{self.model_type} uses max_tokens, not max_output_tokens")
            if self.reasoning_effort is not None:
                raise ValueError(f"{self.model_type} does not support reasoning_effort parameter")

        return self

    @field_validator("temperature", mode="before")
    @classmethod
    def set_fixed_temperature(cls, v: Any, info: Any) -> float:
        """Force temperature to 1 for GPT-5 and reasoning models."""
        if info.data and "model_type" in info.data:
            model_type = info.data["model_type"]

            # Models that don't support temperature control (fixed at 1)
            fixed_temp_models = [
                # Reasoning models
                ModelType.O1_PREVIEW,
                ModelType.O1_MINI,
                ModelType.O3,
                ModelType.O3_MINI,
                ModelType.O4_MINI,
                # GPT-5 family
                ModelType.GPT_5,
                ModelType.GPT_5_MINI,
                ModelType.GPT_5_NANO,
                ModelType.GPT_5_CHAT,
            ]

            if model_type in fixed_temp_models:
                return 1.0

        # Handle None or missing temperature
        if v is None:
            if info.data and "model_type" in info.data:
                model_type = info.data["model_type"]
                fixed_temp_models = [
                    ModelType.O1_PREVIEW,
                    ModelType.O1_MINI,
                    ModelType.O3,
                    ModelType.O3_MINI,
                    ModelType.O4_MINI,
                    ModelType.GPT_5,
                    ModelType.GPT_5_MINI,
                    ModelType.GPT_5_NANO,
                    ModelType.GPT_5_CHAT,
                ]
                if model_type in fixed_temp_models:
                    return 1.0
            return 0.7  # Default for legacy models
        return float(v)


class BenchmarkConfig(BaseModel):
    """Benchmark configuration."""

    benchmark_type: BenchmarkType = Field(
        description="Type of benchmark to run",
    )
    dataset_path: Path = Field(
        description="Path to dataset files",
    )
    sample_size: Optional[int] = Field(
        default=None,
        description="Number of samples to use (None for full dataset)",
        gt=0,
    )
    metrics: List[str] = Field(
        default_factory=lambda: ["accuracy"],
        description="Evaluation metrics to compute",
    )
    evaluation_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Benchmark-specific evaluation parameters",
    )
    shuffle: bool = Field(
        default=True,
        description="Whether to shuffle dataset before sampling",
    )

    @field_validator("dataset_path")
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        """Convert string to Path object."""
        return Path(v) if not isinstance(v, Path) else v


class StrategyConfig(BaseModel):
    """Evaluation strategy configuration."""

    strategy_type: StrategyType = Field(
        description="Type of evaluation strategy",
    )
    k_responses: Optional[int] = Field(
        default=None,
        description="Number of responses to generate (for k_response and coherence)",
        gt=0,
    )
    aggregation_method: Optional[str] = Field(
        default="majority_vote",
        description="Method for aggregating k responses",
    )
    coherence_measures: List[CoherenceMeasure] = Field(
        default_factory=list,
        description="Coherence measures to apply",
    )
    coherence_thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Thresholds for coherence measures",
    )
    temperature_variations: Optional[List[float]] = Field(
        default=None,
        description="Temperature values to explore",
    )

    @model_validator(mode="after")
    def validate_strategy_params(self) -> "StrategyConfig":
        """Validate strategy-specific parameters."""
        if self.strategy_type in [StrategyType.K_RESPONSE, StrategyType.COHERENCE]:
            if self.k_responses is None:
                raise ValueError(
                    f"{self.strategy_type} strategy requires k_responses to be specified"
                )

        if self.strategy_type == StrategyType.COHERENCE:
            if not self.coherence_measures:
                raise ValueError("Coherence strategy requires at least one coherence measure")

        if self.strategy_type == StrategyType.BASELINE:
            if self.k_responses is not None and self.k_responses > 1:
                raise ValueError("Baseline strategy should not have k_responses > 1")

        return self


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""

    name: str = Field(
        description="Experiment name",
    )
    description: Optional[str] = Field(
        default=None,
        description="Experiment description",
    )
    base: BaseConfig = Field(
        description="Base configuration settings",
    )
    model: ModelConfig = Field(
        description="Model configuration",
    )
    benchmark: BenchmarkConfig = Field(
        description="Benchmark configuration",
    )
    strategy: StrategyConfig = Field(
        description="Evaluation strategy configuration",
    )
    grid_params: Optional[Dict[str, List[Any]]] = Field(
        default=None,
        description="Parameters for grid search",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for experiment organization",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate experiment name."""
        if not v or not v.strip():
            raise ValueError("Experiment name cannot be empty")
        # Ensure name is filesystem-safe
        invalid_chars = ["<", ">", ":", '"', "/", "\\", "|", "?", "*"]
        for char in invalid_chars:
            if char in v:
                raise ValueError(f"Experiment name cannot contain '{char}'")
        return v

    def generate_grid_configs(self) -> List["ExperimentConfig"]:
        """Generate all configuration combinations for grid search."""
        if not self.grid_params:
            return [self]

        # TODO: Implement grid search expansion
        # This would create all combinations of grid parameters
        # For now, return self as a placeholder
        return [self]
