# Multi-stage Dockerfile for Cosmos Coherence
# Stage 1: Builder - Install Poetry and dependencies
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_VERSION=1.6.1
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.poetry_cache

# Install Poetry in a virtual environment to isolate it
RUN python -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install --upgrade pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

# Add Poetry to PATH
ENV PATH="${POETRY_VENV}/bin:${PATH}"

# Copy dependency files first for better caching
COPY pyproject.toml poetry.lock ./

# Configure Poetry to not create virtual environments
# Install dependencies in system Python
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root --only main

# Stage 2: Development - Include dev dependencies and mount code
FROM python:3.11-slim AS development

# Set working directory
WORKDIR /app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Poetry installation from builder
COPY --from=builder /opt/poetry-venv /opt/poetry-venv

# Add Poetry to PATH
ENV PATH="/opt/poetry-venv/bin:${PATH}"

# Copy installed dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dev dependencies for development stage
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --with dev

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY tests/ ./tests/

# Set Python environment variables for development
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ENVIRONMENT=development
ENV DEBUG=true

# Create non-root user for development
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose ports for FastAPI and Dash
EXPOSE 8000 8050

# Health check for development
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for development (can be overridden)
CMD ["python", "-m", "uvicorn", "src.cosmos_coherence.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 3: Production - Optimized final image
FROM python:3.11-slim AS production

# Set working directory
WORKDIR /app

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only production dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code only (no tests, docs)
COPY src/ ./src/
COPY configs/ ./configs/

# Set Python environment variables for production
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ENVIRONMENT=production
ENV DEBUG=false
ENV LOG_LEVEL=INFO

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports for FastAPI and Dash
EXPOSE 8000 8050

# Health check for production
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production command with gunicorn for better performance
CMD ["gunicorn", "src.cosmos_coherence.api:app", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--log-level", "info", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
