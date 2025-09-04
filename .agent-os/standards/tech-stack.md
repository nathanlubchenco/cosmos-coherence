# Tech Stack

## Context

Global tech stack defaults for Agent OS projects, overridable in project-specific `.agent-os/product/tech-stack.md`.

- Backend Framework: FastAPI
- Frontend Framework: Dash
- Language: Python 3.11
- Core Pipeline: Python standard library
- Scripting: Python standard library
- Data Storage: Flat files (JSON/YAML/CSV)
- State Management: File-based persistence
- Package Manager: pip
- Virtual Environment: Docker (preferred) / venv as fallback
- Containerization: Docker & Docker Compose
- Python Version: 3.11
- API Documentation: FastAPI automatic docs (Swagger/ReDoc)
- UI Components: Dash Bootstrap Components
- Plotting/Visualization: Plotly (integrated with Dash)
- Data Processing: pandas, numpy
- Configuration: YAML/JSON files
- Logging: Python logging module with file output
- Testing Framework: pytest
- Test Coverage: pytest-cov
- Code Quality: black, ruff, mypy
- Build Tool: GNU Make
- Build Commands: make build, make test, make run
- Development Server: Uvicorn (FastAPI)
- Deployment: Local/On-premise only
- Environment Management: Docker containers
- Secrets Management: Environment variables via .env files
- Development Environment: Docker Compose
- Production Environment: Docker containers
