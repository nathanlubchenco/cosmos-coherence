# Makefile for Cosmos Coherence project

.PHONY: help install test test-config test-models test-loader validate clean lint format check-all

# Default target
help:
	@echo "Cosmos Coherence - LLM Hallucination Detection Framework"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      - Install all dependencies"
	@echo "  make test         - Run all tests"
	@echo "  make test-config  - Run configuration tests only"
	@echo "  make test-models  - Run model tests only"
	@echo "  make test-loader  - Run loader tests only"
	@echo "  make validate     - Run comprehensive validation"
	@echo "  make lint         - Run code linters"
	@echo "  make format       - Format code with black"
	@echo "  make clean        - Remove cache and build files"
	@echo "  make check-all    - Run all checks (lint, test, validate)"

# Install dependencies
install:
	@echo "Installing dependencies..."
	poetry install

# Run all tests
test:
	@echo "Running all tests..."
	poetry run pytest tests/ -v --cov=src/cosmos_coherence --cov-report=term-missing

# Run specific test suites
test-config:
	@echo "Running configuration tests..."
	poetry run pytest tests/config/ -v

test-models:
	@echo "Running model tests..."
	poetry run pytest tests/config/test_models.py tests/config/test_latest_models.py -v

test-loader:
	@echo "Running loader tests..."
	poetry run pytest tests/config/test_loader.py -v

# Run comprehensive validation
validate:
	@echo "Running comprehensive validation..."
	@export OPENAI_API_KEY=$${OPENAI_API_KEY:-sk-test-validation} && \
	export TEST_API_KEY=sk-test-validation && \
	poetry run python tests/integration/test_config_validation.py

# Lint code
lint:
	@echo "Running linters..."
	poetry run ruff check src/ tests/
	poetry run mypy src/cosmos_coherence --ignore-missing-imports

# Format code
format:
	@echo "Formatting code..."
	poetry run black src/ tests/
	poetry run ruff check --fix src/ tests/

# Clean cache and build files
clean:
	@echo "Cleaning cache and build files..."
	rm -rf .cache/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf __pycache__/
	rm -rf src/cosmos_coherence/__pycache__/
	rm -rf src/cosmos_coherence/*/__pycache__/
	rm -rf tests/__pycache__/
	rm -rf tests/*/__pycache__/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Run all checks
check-all: lint test validate
	@echo "✓ All checks passed!"

# Quick validation for CI/CD
ci-test: install lint test validate
	@echo "✓ CI validation complete!"

# Development helpers
dev-setup: install
	@echo "Setting up development environment..."
	@echo "Creating .env file if it doesn't exist..."
	@test -f .env || echo "OPENAI_API_KEY=sk-your-key-here" > .env
	@echo "✓ Development environment ready!"

# Test individual components
test-gpt5:
	@echo "Testing GPT-5 configurations..."
	poetry run pytest tests/config/test_latest_models.py::TestGPT5Models -v

test-o3:
	@echo "Testing O3 reasoning model configurations..."
	poetry run pytest tests/config/test_latest_models.py::TestO3O4Models -v

test-cli:
	@echo "Testing CLI commands..."
	poetry run cosmos-config list-models
	poetry run cosmos-config list-benchmarks
	poetry run cosmos-config list-strategies

# Validate example configurations
validate-examples:
	@echo "Validating example configurations..."
	@for config in configs/experiments/*.yaml; do \
		echo "Validating $$config..."; \
		poetry run cosmos-config validate $$config --base configs/base.yaml || true; \
	done

# Show configuration
show-config:
	@echo "Showing quick test configuration..."
	poetry run cosmos-config show configs/experiments/quick_test.yaml --format yaml
