# Contributing to Cosmos Coherence

## Project Structure Best Practices

### Directory Organization

```
cosmos-coherence/
├── src/                      # Source code
│   └── cosmos_coherence/     # Main package
├── tests/                    # All tests
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── config/              # Configuration tests
├── configs/                  # Configuration files
│   ├── base.yaml            # Base configuration
│   └── experiments/         # Experiment configurations
├── data/                    # Dataset files
├── docs/                    # Documentation
└── scripts/                 # Utility scripts (if needed)
```

### Key Principles

1. **No Test Scripts at Root Level**
   - All validation and test scripts belong in `tests/`
   - Integration tests go in `tests/integration/`
   - Unit tests go in `tests/unit/` or module-specific test directories

2. **Keep Root Directory Clean**
   - Only essential files at root: README.md, pyproject.toml, Makefile, etc.
   - No one-off scripts or temporary files
   - Use appropriate subdirectories for all code

3. **Test Organization**
   - Unit tests: `tests/unit/` or `tests/<module>/`
   - Integration tests: `tests/integration/`
   - Validation scripts: `tests/integration/`
   - Test fixtures and data: `tests/fixtures/`

4. **Configuration Management**
   - All configs in `configs/` directory
   - Base configurations for inheritance
   - Example configurations in subdirectories

## Development Workflow

### Running Tests

```bash
# Run all tests
make test

# Run specific test suites
make test-config
make test-models
make test-loader

# Run validation
make validate

# Run all checks
make check-all
```

### Code Quality

```bash
# Format code
make format

# Run linters
make lint

# Clean cache files
make clean
```

## Code Style Guidelines

1. **Python Version**: 3.11+
2. **Type Hints**: Required for all public functions
3. **Docstrings**: Required for all classes and public methods
4. **Linting**: Must pass `ruff` and `mypy` checks
5. **Testing**: Maintain >80% code coverage

## Testing Requirements

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **Validation Tests**: End-to-end validation of system behavior
4. **Coverage**: Aim for >80% code coverage

## Configuration System

1. **YAML Format**: All configurations in YAML
2. **Environment Variables**: Use `${VAR}` syntax for interpolation
3. **Validation**: All configs must pass Pydantic validation
4. **Examples**: Provide example configurations for common use cases

## Pull Request Process

1. Create feature branch from `main`
2. Write tests for new functionality
3. Ensure all tests pass: `make test`
4. Run validation: `make validate`
5. Format code: `make format`
6. Check linting: `make lint`
7. Update documentation as needed
8. Submit PR with clear description

## Validation Philosophy

"Validation is the most important thing" - This project prioritizes correctness and reliability:

1. Comprehensive test coverage
2. Type safety with Pydantic
3. Integration validation scripts
4. CI/CD pipeline validation
5. Model-specific constraints and validation

## Adding New Features

1. Start with tests (TDD approach)
2. Implement with type safety
3. Add integration tests
4. Update validation scripts
5. Document in appropriate location
6. Add example configurations if applicable