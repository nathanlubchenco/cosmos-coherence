# Technical Stack

## Core Technologies

- **Programming Language:** Python 3.11+
- **Application Framework:** FastAPI 0.104+ (for API and UI backend)
- **Data Storage:** JSON/JSONL flat files for results and experiments
- **ML Framework:** PyTorch 2.0+ (for model operations if needed)

## Data & Configuration

- **Configuration Management:** Pydantic 2.0+
- **Data Format:** JSON/JSONL for datasets
- **Import Strategy:** pip/poetry for dependency management

## Benchmarking & AI

- **LLM Client:** OpenAI Python SDK (latest)
- **Benchmark Frameworks:** Custom implementations with reference code
- **Coherence Libraries:** Custom implementations of philosophical measures
- **Data Processing:** Pandas 2.0+, NumPy 1.24+
- **Dataset Sources:** Hugging Face Datasets, GitHub repositories
- **Dataset Caching:** Local cache with automatic fetching

## Frontend & Visualization

- **Dashboard Framework:** Dash 2.14+ (Python-based interactive web apps)
- **Visualization:** Plotly (integrated with Dash)
- **Component Library:** Dash Bootstrap Components
- **Data Tables:** Dash DataTable for result exploration
- **Layout System:** Dash HTML/Core Components

## Infrastructure

- **Containerization:** Docker & Docker Compose
- **Application Hosting:** Local development / Research cluster
- **Data Hosting:** Local filesystem with JSON/JSONL files
- **Asset Hosting:** Local filesystem
- **Deployment Solution:** Docker containers with docker-compose

## Development Tools

- **Package Management:** Poetry
- **Testing Framework:** pytest
- **Code Quality:** ruff, mypy, black
- **Documentation:** Sphinx / MkDocs
- **Code Repository URL:** https://github.com/[username]/cosmos-coherence (TBD)

## Research Tools

- **Experiment Tracking:** JSON-based experiment logs
- **Paper Management:** Local PDF storage with BibTeX references
- **Implementation References:** Local cache of reference implementations
- **MCP Server:** Optional for advanced paper/data fetching (if needed)
- **Parallel Processing:** multiprocessing / asyncio
- **Progress Tracking:** tqdm
- **Logging:** structlog
