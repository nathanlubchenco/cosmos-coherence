# Technical Specification

This is the technical specification for the spec detailed in @.agent-os/specs/2025-09-04-pydantic-config-system/spec.md

## Technical Requirements

### Core Configuration Models

- **BaseConfig**: Root configuration model with common settings (API keys, output paths, logging)
- **ModelConfig**: Model-specific settings with validation for each OpenAI model type
  - GPT-4/GPT-4o: temperature (0-2), max_tokens, top_p, frequency_penalty, presence_penalty
  - o1-preview/o1-mini: temperature (fixed at 1), max_completion_tokens (no max_tokens)
  - Validation for deprecated/renamed parameters across model versions
- **BenchmarkConfig**: Settings for each benchmark type (FaithBench, SimpleQA, TruthfulQA)
  - Dataset paths, evaluation metrics, sampling parameters
- **StrategyConfig**: Configuration for coherence strategies
  - K-response parameters (k value, aggregation method)
  - Coherence measure settings (Shogenji, Fitelson, Olsson thresholds)
- **ExperimentConfig**: Composition of above configs with grid search parameters

### YAML Processing

- Support for YAML 1.2 specification with safe loading
- Environment variable interpolation using `${VAR_NAME}` or `${VAR_NAME:default}` syntax
- File inclusion via `!include` tag for modular configurations
- Preserve comments and formatting for human readability

### Configuration Inheritance

- Base configuration loading from `configs/base.yaml`
- Experiment-specific overrides from `configs/experiments/*.yaml`
- Command-line overrides using dot notation (e.g., `--model.temperature=0.7`)
- Priority order: CLI args > Environment vars > Experiment config > Base config

### Validation Rules

- Type validation using Pydantic's type system
- Range validation for numeric parameters (temperature, top_p, penalties)
- Enum validation for model names and strategy types
- Custom validators for:
  - API key format and presence
  - File path existence for datasets
  - Mutually exclusive parameter combinations
  - Model-specific parameter compatibility

### Configuration CLI

- `validate` command: Check configuration without running experiments
- `generate` command: Create configuration combinations for grid search
- `show` command: Display resolved configuration with all overrides applied
- `diff` command: Compare two configurations to see differences

## External Dependencies

- **pydantic** (2.0+) - Core validation and serialization
  - **Justification:** Type-safe configuration with automatic validation and excellent error messages
- **pydantic-settings** (2.0+) - Environment variable support
  - **Justification:** Built-in support for .env files and environment variable parsing
- **PyYAML** (6.0+) - YAML parsing
  - **Justification:** Standard YAML library with safe loading capabilities
- **python-dotenv** (1.0+) - .env file support
  - **Justification:** Load environment variables from .env files for local development
- **typer** (0.9+) - CLI framework
  - **Justification:** Modern CLI framework with automatic help generation and type hints
