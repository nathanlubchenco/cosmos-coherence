# Claude Code Assistant Notes

This document contains important notes and guidelines for AI assistants working on this codebase.

## CRITICAL: No Placeholder Implementations

**NEVER create placeholder or mock implementations in production code.** If you cannot find the documentation or functionality needed to implement a feature:
1. **Stop and communicate** - Explicitly tell the user that the implementation cannot be completed
2. **Document the blocker** - Clearly explain what's missing or unavailable
3. **Suggest alternatives** - Propose workarounds or different approaches if possible
4. **Never pretend** - Do not create fake implementations that appear to work but don't actually function

Placeholder implementations are deceptive and dangerous - they give the false impression that functionality exists when it doesn't. This can lead to wasted time, broken production systems, and loss of trust.

## Project Overview

Cosmos Coherence is a benchmark framework for evaluating Large Language Model (LLM) hallucination detection using philosophical coherence measures. The project implements multiple hallucination benchmarks and novel coherence-based evaluation strategies.

## Key Commands

- **Run tests:** `make test`
- **Run specific test module:** `PYTHONPATH=src python -m pytest tests/benchmarks/models/test_base.py -xvs`
- **Check linting:** `pre-commit run --all-files`
- **Build Docker:** `docker compose build`
- **Start services:** `docker compose up -d`

## Pre-commit Checks and Linting

**IMPORTANT:** Always run pre-commit checks before finalizing any code changes to catch linting and type errors early.

### Running Pre-commit Checks

1. **Run all pre-commit hooks on all files:**
   ```bash
   pre-commit run --all-files
   ```

2. **Run specific hooks:**
   ```bash
   # Run only ruff linter
   pre-commit run ruff --all-files

   # Run only mypy type checker
   pre-commit run mypy --all-files

   # Run only ruff formatter
   pre-commit run ruff-format --all-files
   ```

3. **Auto-fix issues where possible:**
   ```bash
   # Ruff can auto-fix many issues
   ruff check --fix src/ tests/

   # Format code with ruff
   ruff format src/ tests/
   ```

### Common Issues and Fixes

- **Unused variables (F841):** Remove or use the variable
- **Type errors:** Add proper type annotations or fix type mismatches
- **Import sorting:** Ruff will auto-fix import order
- **Line length:** Keep lines under 100 characters (configured in pyproject.toml)

### Pre-commit Hook Configuration

The project uses these main hooks (configured in `.pre-commit-config.yaml`):
- **ruff:** Fast Python linter for code quality
- **ruff-format:** Python code formatter
- **mypy:** Static type checker

Always ensure all checks pass before considering a task complete.

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
