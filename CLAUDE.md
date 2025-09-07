# Claude Code Assistant Notes

This document contains important notes and guidelines for AI assistants working on this codebase.

## Project Overview

Cosmos Coherence is a benchmark framework for evaluating Large Language Model (LLM) hallucination detection using philosophical coherence measures. The project implements multiple hallucination benchmarks and novel coherence-based evaluation strategies.

## Key Commands

- **Run tests:** `make test`
- **Run specific test module:** `PYTHONPATH=src python -m pytest tests/benchmarks/models/test_base.py -xvs`
- **Check linting:** `pre-commit run --all-files`
- **Build Docker:** `docker compose build`
- **Start services:** `docker compose up -d`

## Technical Debt Tracking

**IMPORTANT:** When you encounter failing tests or need to skip tests, please update the Technical Debt section in `.agent-os/product/roadmap.md`. This helps maintain visibility of issues that need future resolution.

### When to Update Technical Debt List:
- Tests are skipped with `pytest.skip()`
- Tests are marked with `@pytest.mark.xfail`
- Linting warnings are suppressed with `# noqa` or similar
- TODO comments are added to the code
- Workarounds are implemented due to external dependencies

### Current Known Issues:
- Some serialization tests have edge case failures
- Docker integration tests skip when containers aren't running
- Pydantic deprecation warnings about class-based config
- Mypy warnings about protected namespace conflicts

## Architecture Notes

### Configuration System
- **BenchmarkConfig** (in `config/models.py`): Defines WHAT benchmarks to run
- **BenchmarkRunConfig** (in `benchmarks/models/base.py`): Defines HOW to run them
- This separation is intentional to distinguish configuration from runtime parameters

### Model Hierarchy
```
BaseDatasetItem (abstract)
├── FaithBenchItem
├── SimpleQAItem
├── TruthfulQAItem
├── FEVERItem
└── HaluEvalItem

BaseExperiment (abstract)
├── ExperimentTracker
├── ExperimentRun
└── ExperimentResult

BaseResult (abstract)
├── BenchmarkMetrics
├── AggregationResult
└── StatisticalSummary
```

### Testing Strategy
- Tests are written BEFORE implementation (TDD)
- Each module has comprehensive test coverage
- Integration tests use Docker for isolation
- Use `pytest.skip()` for environment-specific issues

## Development Workflow

1. Check existing tests before modifying code
2. Run pre-commit hooks before committing
3. Update technical debt list when adding workarounds
4. Document any new patterns or architectural decisions
5. Keep test coverage above 80% for new code

## Important Files

- `.agent-os/product/roadmap.md` - Product roadmap and technical debt tracking
- `.agent-os/specs/*/tasks.md` - Task tracking for each specification
- `pyproject.toml` - Project configuration and dependencies
- `docker-compose.yml` - Service orchestration
- `.pre-commit-config.yaml` - Code quality checks

## Contact & Support

For questions or issues, please refer to the GitHub repository or contact the project maintainers.
