# Model Configurations for FaithBench

This document contains the exact model configurations needed to reproduce the FaithBench benchmark results.

## Supported Models (Phase 1 - OpenAI Only)

### Standard Models (Support Temperature Variation)

#### GPT-4-Turbo
```yaml
model_id: gpt-4-turbo
display_name: GPT-4 Turbo
provider: openai
supports_temperature: true
configurations:
  - temperature: 0.0
    max_tokens: 150
    top_p: 1.0
    frequency_penalty: 0
    presence_penalty: 0
    seed: 42
  - temperature: 0.3
    max_tokens: 150
    top_p: 1.0
    frequency_penalty: 0
    presence_penalty: 0
    seed: 42
  - temperature: 0.7
    max_tokens: 150
    top_p: 1.0
    frequency_penalty: 0
    presence_penalty: 0
    seed: 42
  - temperature: 1.0
    max_tokens: 150
    top_p: 1.0
    frequency_penalty: 0
    presence_penalty: 0
    seed: 42
```

#### GPT-4o
```yaml
model_id: gpt-4o
display_name: GPT-4 Optimized
provider: openai
supports_temperature: true
configurations:
  - temperature: 0.0
    max_tokens: 150
    top_p: 1.0
    frequency_penalty: 0
    presence_penalty: 0
    seed: 42
  - temperature: 0.3
    max_tokens: 150
    top_p: 1.0
    frequency_penalty: 0
    presence_penalty: 0
    seed: 42
  - temperature: 0.7
    max_tokens: 150
    top_p: 1.0
    frequency_penalty: 0
    presence_penalty: 0
    seed: 42
  - temperature: 1.0
    max_tokens: 150
    top_p: 1.0
    frequency_penalty: 0
    presence_penalty: 0
    seed: 42
```

### Reasoning Models (No Temperature Variation)

**IMPORTANT:** OpenAI's reasoning models (o1-mini, o3-mini) do not support temperature variation. They always operate at temperature=1 internally and this parameter cannot be changed.

#### o1-mini
```yaml
model_id: o1-mini
display_name: OpenAI o1 Mini (Reasoning Model)
provider: openai
supports_temperature: false
configurations:
  - max_tokens: 150
    # Temperature is not configurable for reasoning models
    # Model internally uses chain-of-thought reasoning
```

#### o3-mini
```yaml
model_id: o3-mini
display_name: OpenAI o3 Mini (Reasoning Model)
provider: openai
supports_temperature: false
configurations:
  - max_tokens: 150
    # Temperature is not configurable for reasoning models
    # Model internally uses advanced reasoning
```

## Model Availability Note

**IMPORTANT:** Model availability depends on your OpenAI API access:
- `gpt-4-turbo`: Generally available
- `gpt-4o`: Generally available
- `o1-mini`: Limited availability (check API access)
- `o3-mini`: Limited availability (may require special access)

If you don't have access to o1-mini or o3-mini, the implementation should gracefully handle this and test only the available models.

## Configuration Usage in Code

### Example Implementation
```python
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class ModelProvider(Enum):
    OPENAI = "openai"
    # Future providers (Phase 2)
    ANTHROPIC = "anthropic"  # Not implemented
    GOOGLE = "google"  # Not implemented
    TOGETHER_AI = "together_ai"  # Not implemented

@dataclass
class ModelConfig:
    """Configuration for a specific model evaluation."""
    model_id: str
    display_name: str
    provider: ModelProvider
    supports_temperature: bool
    temperature: Optional[float]  # None for reasoning models
    max_tokens: int
    top_p: Optional[float] = 1.0
    frequency_penalty: float = 0
    presence_penalty: float = 0
    seed: Optional[int] = None

class FaithBenchModelConfigs:
    """Factory for creating benchmark-compliant model configurations."""

    STANDARD_MODELS = ["gpt-4-turbo", "gpt-4o"]
    REASONING_MODELS = ["o1-mini", "o3-mini"]
    SUPPORTED_MODELS = STANDARD_MODELS + REASONING_MODELS

    @staticmethod
    def get_gpt4_turbo_configs() -> List[ModelConfig]:
        """Get all GPT-4-Turbo configurations."""
        temps = [0.0, 0.3, 0.7, 1.0]
        return [
            ModelConfig(
                model_id="gpt-4-turbo",
                display_name="GPT-4 Turbo",
                provider=ModelProvider.OPENAI,
                supports_temperature=True,
                temperature=temp,
                max_tokens=150,
                top_p=1.0,
                frequency_penalty=0,
                presence_penalty=0,
                seed=42
            )
            for temp in temps
        ]

    @staticmethod
    def get_gpt4o_configs() -> List[ModelConfig]:
        """Get all GPT-4o configurations."""
        temps = [0.0, 0.3, 0.7, 1.0]
        return [
            ModelConfig(
                model_id="gpt-4o",
                display_name="GPT-4 Optimized",
                provider=ModelProvider.OPENAI,
                supports_temperature=True,
                temperature=temp,
                max_tokens=150,
                top_p=1.0,
                frequency_penalty=0,
                presence_penalty=0,
                seed=42
            )
            for temp in temps
        ]

    @staticmethod
    def get_o1_mini_config() -> ModelConfig:
        """Get o1-mini configuration (no temperature variation)."""
        return ModelConfig(
            model_id="o1-mini",
            display_name="OpenAI o1 Mini (Reasoning)",
            provider=ModelProvider.OPENAI,
            supports_temperature=False,
            temperature=None,  # Not applicable
            max_tokens=150,
            top_p=None,  # Not configurable
            seed=None  # Not applicable
        )

    @staticmethod
    def get_o3_mini_config() -> ModelConfig:
        """Get o3-mini configuration (no temperature variation)."""
        return ModelConfig(
            model_id="o3-mini",
            display_name="OpenAI o3 Mini (Reasoning)",
            provider=ModelProvider.OPENAI,
            supports_temperature=False,
            temperature=None,  # Not applicable
            max_tokens=150,
            top_p=None,  # Not configurable
            seed=None  # Not applicable
        )

    @staticmethod
    def get_all_baseline_configs() -> Dict[str, List[ModelConfig]]:
        """Get all baseline configurations for benchmark.

        Note: Reasoning models return single config (no temperature variation).
        """
        return {
            "gpt-4-turbo": FaithBenchModelConfigs.get_gpt4_turbo_configs(),
            "gpt-4o": FaithBenchModelConfigs.get_gpt4o_configs(),
            "o1-mini": [FaithBenchModelConfigs.get_o1_mini_config()],
            "o3-mini": [FaithBenchModelConfigs.get_o3_mini_config()],
        }

    @staticmethod
    def validate_model_support(model_id: str) -> None:
        """Validate that a model is supported in Phase 1.

        Raises:
            NotImplementedError: If model is not an OpenAI model.
        """
        if model_id not in FaithBenchModelConfigs.SUPPORTED_MODELS:
            raise NotImplementedError(
                f"Model {model_id} is not supported in Phase 1. "
                f"Only OpenAI models {FaithBenchModelConfigs.SUPPORTED_MODELS} are currently supported. "
                "See roadmap for Phase 2 implementation timeline."
            )

    @staticmethod
    def supports_temperature(model_id: str) -> bool:
        """Check if a model supports temperature variation."""
        return model_id in FaithBenchModelConfigs.STANDARD_MODELS
```

## Batch Evaluation Configuration

### Recommended Batch Sizes (OpenAI Only)
```yaml
batch_sizes:
  gpt-4-turbo: 10  # Standard batch size
  gpt-4o: 10  # Standard batch size
  o1-mini: 5  # Lower due to reasoning overhead
  o3-mini: 5  # Lower due to reasoning overhead
```

### Rate Limiting (OpenAI Only)
```yaml
rate_limits:
  gpt-4-turbo:
    requests_per_minute: 500
    tokens_per_minute: 10000
  gpt-4o:
    requests_per_minute: 500
    tokens_per_minute: 10000
  o1-mini:
    requests_per_minute: 100  # Lower for reasoning models
    tokens_per_minute: 5000
  o3-mini:
    requests_per_minute: 100  # Lower for reasoning models
    tokens_per_minute: 5000
```

## Expected Results

### Performance Expectations
```yaml
expected_performance:
  gpt-4-turbo:
    balanced_accuracy: ~0.52  # Expected slightly above 50%
    supports_temperature_analysis: true
  gpt-4o:
    balanced_accuracy: ~0.52  # Similar to GPT-4-Turbo
    supports_temperature_analysis: true
  o1-mini:
    balanced_accuracy: TBD  # No published baseline
    supports_temperature_analysis: false  # Single temperature only
  o3-mini:
    balanced_accuracy: TBD  # No published baseline
    supports_temperature_analysis: false  # Single temperature only
```

## Phase 1 Validation Checklist

- [ ] Test GPT-4-Turbo with all four temperature settings (0.0, 0.3, 0.7, 1.0)
- [ ] Test GPT-4o with all four temperature settings (0.0, 0.3, 0.7, 1.0)
- [ ] Test o1-mini with single configuration (no temperature variation)
- [ ] Test o3-mini with single configuration (no temperature variation)
- [ ] Handle API access errors gracefully for limited-access models
- [ ] Validate that reasoning models don't accept temperature parameter
- [ ] NotImplementedError raised for non-OpenAI models

## Phase 2 Items (Out of Scope)

These items are planned for Phase 2 implementation:

- Temperature variation analysis for coherence measures (standard models only)
- Support for Anthropic models (Claude-3 family)
- Support for open-source models (Llama, Mistral)
- Support for Google models (Gemini)
- Integration with Shogenji, Fitelson, and Olsson coherence measures
- Cross-temperature consistency analysis (applicable only to non-reasoning models)
