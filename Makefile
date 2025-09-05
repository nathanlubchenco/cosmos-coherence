# Makefile for Cosmos Coherence project

.PHONY: help install test test-config test-models test-loader validate clean lint format check-all \
        docker-build docker-build-dev docker-build-prod docker-run docker-run-dev docker-run-prod \
        docker-test docker-clean docker-shell \
        compose-up compose-down compose-build compose-logs compose-ps compose-restart \
        compose-up-prod compose-down-prod

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
	@echo ""
	@echo "Docker commands:"
	@echo "  make docker-build      - Build Docker image (production)"
	@echo "  make docker-build-dev  - Build Docker image (development)"
	@echo "  make docker-run        - Run container (production)"
	@echo "  make docker-run-dev    - Run container (development with hot-reload)"
	@echo "  make docker-test       - Run tests in Docker container"
	@echo "  make docker-shell      - Open shell in Docker container"
	@echo "  make docker-clean      - Clean up Docker resources"
	@echo ""
	@echo "Docker Compose commands:"
	@echo "  make compose-up        - Start all services (development)"
	@echo "  make compose-down      - Stop all services"
	@echo "  make compose-build     - Build all service images"
	@echo "  make compose-logs      - View service logs"
	@echo "  make compose-ps        - List running services"
	@echo "  make compose-restart   - Restart all services"
	@echo "  make compose-up-prod   - Start production services"

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

# Docker commands
# Build Docker image for production
docker-build: docker-build-prod

docker-build-prod:
	@echo "Building Docker image for production..."
	docker build --target production -t cosmos-coherence:latest -t cosmos-coherence:prod .

# Build Docker image for development
docker-build-dev:
	@echo "Building Docker image for development..."
	docker build --target development -t cosmos-coherence:dev .

# Run Docker container for production
docker-run: docker-run-prod

docker-run-prod:
	@echo "Running Docker container (production)..."
	docker run -d \
		--name cosmos-coherence-prod \
		-p 8000:8000 \
		-p 8050:8050 \
		--env-file .env \
		cosmos-coherence:latest

# Run Docker container for development with hot-reload
docker-run-dev:
	@echo "Running Docker container (development with hot-reload)..."
	docker run -it --rm \
		--name cosmos-coherence-dev \
		-p 8000:8000 \
		-p 8050:8050 \
		-v $(PWD)/src:/app/src \
		-v $(PWD)/configs:/app/configs \
		-v $(PWD)/tests:/app/tests \
		--env-file .env \
		cosmos-coherence:dev

# Run tests in Docker container
docker-test:
	@echo "Running tests in Docker container..."
	docker run --rm \
		-v $(PWD)/tests:/app/tests \
		--env-file .env.test \
		cosmos-coherence:dev \
		pytest tests/ -v --cov=src/cosmos_coherence --cov-report=term-missing

# Open shell in Docker container for debugging
docker-shell:
	@echo "Opening shell in Docker container..."
	docker run -it --rm \
		-v $(PWD):/app \
		--env-file .env \
		cosmos-coherence:dev \
		/bin/bash

# Clean up Docker resources
docker-clean:
	@echo "Cleaning up Docker resources..."
	@docker stop cosmos-coherence-prod 2>/dev/null || true
	@docker stop cosmos-coherence-dev 2>/dev/null || true
	@docker rm cosmos-coherence-prod 2>/dev/null || true
	@docker rm cosmos-coherence-dev 2>/dev/null || true
	@docker rmi cosmos-coherence:latest cosmos-coherence:prod cosmos-coherence:dev 2>/dev/null || true
	@echo "Docker cleanup complete!"

# Build and test Docker image
docker-check: docker-build-dev docker-test
	@echo "✓ Docker build and test complete!"

# Docker Compose commands
# Start all services (development mode with override)
compose-up:
	@echo "Starting all services (development)..."
	docker-compose up -d
	@echo "Services started! Access at:"
	@echo "  - API: http://localhost:8000"
	@echo "  - Dashboard: http://localhost:8050"
	@echo "  - Database Admin: http://localhost:8080"
	@echo "  - Mail UI: http://localhost:8025"

# Stop all services
compose-down:
	@echo "Stopping all services..."
	docker-compose down
	@echo "Services stopped!"

# Build all service images
compose-build:
	@echo "Building all service images..."
	docker-compose build
	@echo "Build complete!"

# View service logs
compose-logs:
	@echo "Viewing service logs (Ctrl+C to exit)..."
	docker-compose logs -f

# List running services
compose-ps:
	@echo "Running services:"
	docker-compose ps

# Restart all services
compose-restart:
	@echo "Restarting all services..."
	docker-compose restart
	@echo "Services restarted!"

# Start production services
compose-up-prod:
	@echo "Starting production services..."
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
	@echo "Production services started!"

# Stop production services
compose-down-prod:
	@echo "Stopping production services..."
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml down
	@echo "Production services stopped!"

# Clean up all compose resources
compose-clean:
	@echo "Cleaning up Docker Compose resources..."
	docker-compose down -v --remove-orphans
	@echo "Cleanup complete!"

# Run tests using Docker Compose
compose-test:
	@echo "Running tests in Docker Compose environment..."
	docker-compose run --rm api pytest tests/ -v
	@echo "Tests complete!"

# Open shell in a service container
compose-shell:
	@echo "Opening shell in API container..."
	docker-compose exec api /bin/bash

# Database operations
compose-db-migrate:
	@echo "Running database migrations..."
	docker-compose exec api python -m alembic upgrade head
	@echo "Migrations complete!"

compose-db-backup:
	@echo "Backing up database..."
	docker-compose exec db pg_dump -U cosmos cosmos_coherence > backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "Backup complete!"

# Full development environment setup
compose-dev-setup: compose-build compose-up
	@echo "Waiting for services to be healthy..."
	@sleep 10
	@echo "Development environment ready!"
