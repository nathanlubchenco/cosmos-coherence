# Cosmos Coherence

A research framework that advances LLM hallucination detection by applying formal philosophical coherence measures to benchmark evaluations, providing researchers with novel insights into model reliability through systematic temperature-variant response analysis.

## Overview

Cosmos Coherence unifies benchmark reproduction with novel coherence-based evaluation methods. This framework incorporates formal epistemological measures (Shogenji, Fitelson, Olsson) to analyze response consistency across temperature variations, revealing deeper insights into when and why models hallucinate.
## Status
Early work in progress development. Few features are implimented yet. See the agent-os roadmap for details of work completed and work upcoming.

## Features

- **Coherence-Based Analysis**: Implements formal philosophical coherence measures for hallucination detection
- **Temperature Variation Studies**: Systematic analysis across different temperature settings
- **Type-Safe Configuration**: Pydantic-based configuration system with comprehensive validation
- **Flexible Benchmarking**: Extensible framework for running multiple benchmark suites
- **Reproducible Research**: YAML-based experiment configuration with environment variable support

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Poetry for dependency management
- OpenAI API key (and optionally Anthropic/HuggingFace tokens)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cosmos-coherence.git
cd cosmos-coherence

# Install dependencies with Poetry
poetry install

# Set up pre-commit hooks for code quality
poetry run pre-commit install

# Copy environment configuration template
cp .env.example .env

# Edit .env with your API keys and settings
# Required: OPENAI_API_KEY, OUTPUT_DIR, LOG_LEVEL
nano .env

# Verify installation
make test
```

### Quick Start

```python
from cosmos_coherence.config import load_config

# Load configuration
config = load_config("configs/base.yaml")

# Run with custom overrides
config = load_config(
    "configs/experiments/gpt5_coherence.yaml",
    overrides={"model.temperature": 0.8}
)
```

## Development

### Running Tests

```bash
# Run all tests
make test

# Run specific test suites
make test-config
make test-models

# Run with coverage
poetry run pytest --cov

# Run validation suite
make validate
```

### Code Quality

```bash
# Format code with Black
make format

# Run linters (Ruff)
make lint

# Type checking with Mypy
poetry run mypy src/

# Run all checks
make check-all
```

### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality:
- Black (code formatting)
- Ruff (linting)
- Mypy (type checking)
- YAML/TOML validation
- Large file detection
- Trailing whitespace removal

## Project Structure

```
cosmos-coherence/
├── src/cosmos_coherence/    # Main package source code
│   ├── config/              # Configuration system
│   ├── benchmarks/          # Benchmark implementations
│   ├── models/              # Model interfaces
│   └── coherence/           # Coherence measure implementations
├── configs/                 # Configuration files
│   ├── base.yaml           # Base configuration
│   └── experiments/        # Experiment configurations
├── tests/                  # Test suites
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── config/            # Configuration tests
├── data/                  # Datasets and results
└── docs/                  # Documentation
```

## Configuration

The framework uses a flexible YAML-based configuration system with environment variable interpolation:

```yaml
# configs/base.yaml
model:
  name: gpt-5
  temperature: ${TEMPERATURE:1.0}
  max_tokens: 2048

output:
  dir: ${OUTPUT_DIR}
  format: jsonl
```

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Development setup
- Code style requirements
- Testing standards
- Pull request process
- Project structure best practices

## Research Context

This project is funded by the Cosmos Institute to investigate reducing hallucinations in Large Language Models through the application of formal theories of coherence from epistemology. The framework enables researchers to:

1. Reproduce existing hallucination benchmarks
2. Apply novel coherence-based evaluation metrics
3. Study temperature-dependent behavior patterns
4. Identify systematic failure modes in model reasoning

## License

[License information to be added]

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{cosmos_coherence2025,
  title={Cosmos Coherence: A Framework for Coherence-Based Hallucination Detection},
  author={[Nathan Lubchenco]},
  year={2025},
  url={https://github.com/nathanlubchenco/cosmos-coherence}
}
```

## Contact

For questions and (limited) support, please open an issue on GitHub or contact the research team.
