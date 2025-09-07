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

## Testing Philosophy

**CRITICAL: Tests are only valuable if they actually confirm the behavior we expect.** Running tests should be as important as running pre-commit checks.

### Before Committing Code:
1. **Always run tests** - Use `make test` or run specific test modules
2. **Never ignore failing tests** - If a test fails, you MUST either:
   - Fix the issue causing the failure
   - Fix the test if it's testing incorrect behavior
   - Mark it with `@pytest.mark.skip(reason="...")` and add to tech debt
3. **Update the tech debt list** - Any skipped or xfailed tests MUST be documented in `.agent-os/product/roadmap.md`

### Test Management Guidelines:
- **Fix what you can** - Always attempt to fix failing tests first
- **Be explicit about failures** - Use descriptive skip reasons that explain why the test is failing
- **Track everything** - Every test that's skipped or marked as xfail should have a corresponding tech debt item
- **Don't hide failures** - It's better to have a skipped test that's tracked than a passing test that doesn't actually test anything

### Example of Proper Test Handling:
```python
# Good - Explicit about why it's skipped and tracked in tech debt
@pytest.mark.skip(reason="Progress bar testing needs refactoring after CLI changes")
def test_progress_bar_display():
    ...

# Bad - Test passes but doesn't actually test anything
def test_progress_bar_display():
    assert True  # This provides no value!
```

### When It's OK to Compromise:
- Mock implementations when testing integrations with external services
- Skip tests that require specific environment setup (but document it)
- Use simplified assertions when the full behavior is too complex to test reliably
- **BUT ALWAYS** document these compromises in comments and tech debt

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
2. **Run tests after making changes** - Use `make test` to ensure nothing is broken
3. **Run pre-commit hooks before committing** - Use `pre-commit run --all-files`
4. **Fix or properly skip failing tests** - Never leave tests failing silently
5. Update technical debt list when adding workarounds or skipping tests
6. Document any new patterns or architectural decisions
7. Keep test coverage above 80% for new code

### The Golden Rule:
**If tests are failing, either fix them or explicitly skip them with a reason and add to tech debt. Never commit with unexplained test failures.**

## Important Files

- `.agent-os/product/roadmap.md` - Product roadmap and technical debt tracking
- `.agent-os/specs/*/tasks.md` - Task tracking for each specification
- `pyproject.toml` - Project configuration and dependencies
- `docker-compose.yml` - Service orchestration
- `.pre-commit-config.yaml` - Code quality checks

## Contact & Support

For questions or issues, please refer to the GitHub repository or contact the project maintainers.
