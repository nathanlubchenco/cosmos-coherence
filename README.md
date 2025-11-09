# Cosmos Coherence

A research framework that advances LLM hallucination detection by applying formal philosophical coherence measures to benchmark evaluations, providing researchers with novel insights into model reliability through systematic temperature-variant response analysis.

## Overview

Cosmos Coherence unifies benchmark reproduction with novel coherence-based evaluation methods. This framework incorporates formal epistemological measures (Shogenji, Fitelson, Olsson) to analyze response consistency across temperature variations, revealing deeper insights into when and why models hallucinate.

## Status
Early work in progress development. Few features are implimented yet. See the agent-os roadmap for details of work completed and work upcoming.

## Features

- **Coherence-Based Analysis**: Implements formal philosophical coherence measures for hallucination detection
- **Question-Variation Coherence**: Novel approach using coherence across question paraphrases to guide abstention
- **Temperature Variation Studies**: Systematic analysis across different temperature settings
- **HuggingFace Integration**: Automatic loading and caching of popular hallucination detection datasets
- **Type-Safe Configuration**: Pydantic-based configuration system with comprehensive validation
- **Flexible Benchmarking**: Extensible framework for running multiple benchmark suites
- **Reproducible Research**: YAML-based experiment configuration with environment variable support

## Getting Started

### Prerequisites

- Docker Desktop (version 20.10+) and Docker Compose (v2.0+)
- Python 3.11 or higher (for local development)
- Poetry for dependency management
- OpenAI API key (and optionally Anthropic/HuggingFace tokens)

### Quick Start with Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/cosmos-coherence.git
cd cosmos-coherence

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Start all services
./scripts/start-dev.sh

# Access services
# API: http://localhost:8000
# Dashboard: http://localhost:8050
# Database Admin: http://localhost:8080
```

### Local Installation (Alternative)

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

### Supported Benchmarks

The framework supports automatic loading of the following hallucination detection benchmarks via HuggingFace:

- **SimpleQA**: Simple factual questions for accuracy evaluation
- **FaithBench**: Faithfulness evaluation benchmark
- **TruthfulQA**: Questions testing truthful generation
- **FEVER**: Fact extraction and verification
- **HaluEval**: Hallucination evaluation across multiple tasks

See [HuggingFace Integration Guide](docs/huggingface_integration.md) for detailed usage.

## Question-Variation Coherence for SimpleQA

### Overview

The question-variation coherence feature implements a novel approach to hallucination prevention by applying coherence theory to question variations rather than answer variations. This method addresses two key limitations of traditional coherence-based hallucination detection when applied to SimpleQA:

1. **No pre-generated answer samples**: SimpleQA provides only question-answer pairs, not multiple answer samples
2. **Short answers**: SimpleQA answers are typically 1-3 words, making coherence analysis within a single answer impractical

### Core Hypothesis

**Well-formed questions based on factual premises maintain high coherence across paraphrasing, while ambiguous or ill-posed questions show low coherence, indicating higher hallucination risk.**

The system measures coherence across question paraphrases and uses this score to guide answering:
- **High coherence** → Well-formed, unambiguous question → Safe to answer confidently
- **Low coherence** → Ambiguous or ill-posed question → Abstain ("I don't know")

### How It Works

The evaluation pipeline follows these steps:

1. **Load SimpleQA question** from HuggingFace dataset (4,326 questions)
2. **Generate question variations** using OpenAI API with temperature-based sampling (default: 5 variations)
3. **Calculate coherence score** across question variations using SelfCheckGPT coherence variants
4. **Determine answer strategy** based on question type:
   - **YES/NO questions**: Coherence score directly determines answer polarity
   - **Other questions**: Coherence threshold determines whether to answer or abstain
5. **Generate answer** (if coherence threshold met for non-YES/NO questions)
6. **Grade answer** using existing SimpleQA grader
7. **Aggregate metrics** and compare to baseline

### Usage Examples

#### Basic Usage

Run coherence-guided evaluation on SimpleQA with default settings:

```bash
# Basic evaluation with 100 questions
cosmos-coherence simpleqa run-with-coherence \
    --model gpt-4o-mini \
    --sample-size 100 \
    --coherence-variant shogenji \
    --output results/coherence_shogenji.json

# View results
cat results/coherence_shogenji.json | jq '.metrics'
```

#### Threshold Tuning

Compare different coherence thresholds to find optimal precision-recall tradeoff:

```bash
# Conservative threshold (higher precision, lower recall)
cosmos-coherence simpleqa run-with-coherence \
    --model gpt-4o-mini \
    --sample-size 100 \
    --coherence-threshold 0.7 \
    --output results/threshold_0.7.json

# Balanced threshold
cosmos-coherence simpleqa run-with-coherence \
    --model gpt-4o-mini \
    --sample-size 100 \
    --coherence-threshold 0.5 \
    --output results/threshold_0.5.json

# Aggressive threshold (lower precision, higher recall)
cosmos-coherence simpleqa run-with-coherence \
    --model gpt-4o-mini \
    --sample-size 100 \
    --coherence-threshold 0.3 \
    --output results/threshold_0.3.json

# Compare results
python scripts/compare_thresholds.py results/threshold_*.json
```

#### Comparing Coherence Variants

Evaluate all three coherence measures (Shogenji, Fitelson, Olsson) on the same dataset:

```bash
# Run all three variants
for variant in shogenji fitelson olsson; do
    cosmos-coherence simpleqa run-with-coherence \
        --model gpt-4o-mini \
        --sample-size 100 \
        --coherence-variant $variant \
        --output results/variant_${variant}.json
done

# Compare variant performance
python scripts/compare_variants.py results/variant_*.json
```

#### Resuming from Checkpoint

Evaluation automatically saves checkpoints every 50 questions (configurable). Resume interrupted evaluations:

```bash
# Start evaluation
cosmos-coherence simpleqa run-with-coherence \
    --model gpt-4o-mini \
    --sample-size 500 \
    --checkpoint-interval 50 \
    --output results/large_eval.json

# If interrupted, resume from last checkpoint
cosmos-coherence simpleqa run-with-coherence \
    --resume-from results/large_eval_checkpoint.json
```

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | (required) | Target model to evaluate (e.g., gpt-4o-mini, gpt-4-turbo) |
| `--coherence-variant` | shogenji | Coherence measure: shogenji, fitelson, olsson |
| `--num-variations` | 5 | Number of question paraphrases per question |
| `--coherence-threshold` | 0.5 | Abstention threshold (0.0-1.0) |
| `--temperature-range` | 0.7,1.0,1.3 | Temperatures for variation generation |
| `--sample-size` | 4326 | Number of questions to evaluate (max: 4,326) |
| `--output` | auto | Path to save results JSON |
| `--checkpoint-interval` | 50 | Save progress every N questions (0=disabled) |
| `--resume-from` | None | Path to checkpoint file to resume from |
| `--workers` | 4 | Number of parallel workers for API calls |
| `--run-baseline` | true | Also run baseline (no coherence) for comparison |
| `--abstention-response` | "I don't know" | Custom text for abstentions |
| `--grader-model` | gpt-4o-mini | Model for AI grading |
| `--verbose` | false | Show detailed debug output |

### Output Format

Results are saved in JSON format with comprehensive metrics:

```json
{
  "config": {
    "model": "gpt-4o-mini",
    "coherence_variant": "shogenji",
    "num_variations": 5,
    "coherence_threshold": 0.5,
    "sample_size": 100
  },
  "metrics": {
    "overall_correct": 0.70,
    "overall_incorrect": 0.18,
    "overall_not_attempted": 0.12,
    "correct_given_attempted": 0.80,
    "total_questions": 100
  },
  "baseline_metrics": {
    "overall_correct": 0.75,
    "overall_incorrect": 0.25,
    "correct_given_attempted": 0.75
  },
  "coherence_stats": {
    "mean_coherence": 0.62,
    "median_coherence": 0.58,
    "abstention_rate": 0.12
  },
  "api_stats": {
    "total_calls": 1800,
    "cached_calls": 540,
    "cache_hit_rate": 0.30,
    "estimated_cost_usd": 0.15
  }
}
```

### Interpretation Guide

**Success Criteria:**
- **Precision improvement**: `correct_given_attempted` should be higher than baseline (5+ percentage points)
- **Reduced incorrect answers**: `overall_incorrect` should decrease (20%+ relative reduction)
- **Reasonable abstention rate**: `overall_not_attempted` should be 10-30% (not 0% or 100%)

**Expected Tradeoffs:**
- Lower threshold (0.3) → More answers attempted, higher recall, lower precision
- Higher threshold (0.7) → Fewer answers attempted, lower recall, higher precision
- Optimal threshold depends on use case: prioritize precision or recall?

### API Cost Analysis

**Cost Breakdown:**
- **Baseline SimpleQA**: ~2 API calls per question
  - Answer generation: 1 call
  - AI grading: 1 call
  - **Full dataset (4,326 questions)**: ~8,652 calls ≈ $0.87 with gpt-4o-mini

- **Coherence-Guided SimpleQA**: ~18 API calls per question (with N=5 variations, Shogenji variant)
  - Question variation generation: 5 calls
  - Coherence probability extraction: 11 calls (1 + 2×5)
  - Answer generation: 0-1 call (only if threshold met)
  - AI grading: 1 call
  - **Full dataset (4,326 questions)**: ~77,868 calls ≈ $5.70-6.00 with gpt-4o-mini
  - With 20-30% cache hit rate: ~$4.50-5.00

**Cost Comparison:**
- **Cost multiplier**: ~6-7x vs baseline
- **Model pricing** (approximate):
  - gpt-4o-mini: $5-6 for full evaluation (recommended)
  - gpt-4-turbo: $40-45 for full evaluation

**Cost-Benefit Analysis:**
- Reduced hallucinations justify higher cost for production use cases
- Use sample-size parameter for cost-effective experimentation
- Start with small samples (100-500 questions) before full evaluation
- Fitelson variant requires ~30% more API calls than Shogenji/Olsson

### Integration with SelfCheckGPT

This feature imports coherence calculation logic from the [SelfCheckGPT](https://github.com/potsawee/selfcheckgpt) package:

```python
from selfcheckgpt import SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson
from selfcheckgpt.modeling_coherence_api import CoherenceAPIClient
from selfcheckgpt.utils_coherence import normalize_coherence_scores
```

**Coherence Variants:**
1. **Shogenji**: Ratio-based independence measure - `C2(A,B) = P(A ∧ B) / (P(A) × P(B))`
2. **Fitelson**: Confirmation-based support measure - `s(H,E) = P(H|E) - P(H|¬E)`
3. **Olsson**: Relative overlap measure - `C1(A,B) = P(A ∧ B) / P(A ∨ B)`

See [SelfCheckGPT Integration Guide](docs/SELFCHECKGPT_INTEGRATION.md) for installation instructions.

### Documentation

- **Validation Guide**: [docs/VALIDATION_GUIDE.md](docs/VALIDATION_GUIDE.md) - Testing and validation methodology
- **Architecture Overview**: See spec in `agent-os/specs/2025-11-07-question-variation-coherence/planning/architecture-overview.md`
- **API Documentation**: See docstrings in implementation modules
- **Troubleshooting**: See [Troubleshooting Guide](docs/TROUBLESHOOTING.md)

### Research Context

This feature represents a novel application of coherence theory from epistemology to question-level hallucination detection. Traditional coherence methods (as used in SelfCheckGPT) evaluate coherence across multiple sampled answers to a fixed question. This approach inverts that paradigm by evaluating coherence of the question itself across semantically equivalent paraphrases.

**Key Innovation**: By measuring coherence at the question level, we can detect ambiguous or ill-posed questions before generating potentially hallucinated answers, enabling proactive abstention rather than reactive detection.

## Development

### Docker Development

```bash
# Start development environment
docker compose up -d

# View logs
docker compose logs -f

# Run tests in container
docker compose exec api pytest

# Stop services
docker compose down
```

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
│   ├── api/                 # FastAPI backend
│   ├── dashboard/           # Dash frontend
│   ├── benchmarks/          # Benchmark implementations
│   │   ├── simpleqa_coherence_cli.py          # CLI for coherence evaluation
│   │   └── implementations/
│   │       ├── question_variation_generator.py # Question paraphrase generation
│   │       ├── coherence_integration.py        # SelfCheckGPT integration
│   │       ├── answer_strategy.py              # Coherence-guided answering
│   │       └── simpleqa_coherence_benchmark.py # Main evaluation pipeline
│   ├── models/              # Model interfaces
│   └── coherence/           # Coherence measure implementations
├── configs/                 # Configuration files
│   ├── base.yaml           # Base configuration
│   ├── logging.yaml        # Logging configuration
│   ├── monitoring.yaml     # Monitoring configuration
│   └── experiments/        # Experiment configurations
├── tests/                  # Test suites
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── benchmarks/        # Question-variation coherence tests (31 tests)
├── scripts/               # Deployment and utility scripts
│   ├── deploy.sh         # Deployment script
│   ├── backup.sh         # Backup utility
│   ├── health-check.sh   # Health monitoring
│   ├── run_all_tests.py  # Test runner
│   └── validate_coherence_implementation.py  # Validation suite
├── docker-compose.yml     # Docker orchestration
├── Dockerfile            # Multi-stage Docker build
├── data/                 # Datasets and results
└── docs/                 # Documentation
    ├── docker-guide.md   # Docker setup guide
    ├── volume-management.md # Data persistence guide
    ├── SELFCHECKGPT_INTEGRATION.md # SelfCheckGPT setup
    ├── VALIDATION_GUIDE.md         # Testing and validation
    └── TROUBLESHOOTING.md          # Common issues and solutions
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
