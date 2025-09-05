# Technical Specification

This is the technical specification for the spec detailed in @.agent-os/specs/2025-09-05-repo-scaffolding/spec.md

## Technical Requirements

### .gitignore Configuration
- Use Python-specific gitignore template as base
- Include patterns for:
  - Python artifacts: `__pycache__/`, `*.py[cod]`, `*.so`, `.Python`
  - Virtual environments: `venv/`, `env/`, `.venv/`
  - Testing: `.pytest_cache/`, `.coverage`, `htmlcov/`, `.tox/`
  - Package files: `dist/`, `build/`, `*.egg-info/`
  - Environment files: `.env`, `.env.*` (except .env.example)
  - IDE files: `.vscode/`, `.idea/`, `*.swp`, `*.swo`
  - OS files: `.DS_Store`, `Thumbs.db`
  - Project specific: `.cache/`, `outputs/`, `data/` (if contains downloaded data)

### Pre-commit Hook Configuration
- Use pre-commit framework (https://pre-commit.com/)
- Configure hooks in `.pre-commit-config.yaml`:
  - **Black** (code formatting): v23.0+
  - **Ruff** (linting): latest version
  - **Mypy** (type checking): v1.0+
  - **Check-yaml**: Validate YAML syntax
  - **End-of-file-fixer**: Ensure files end with newline
  - **Trailing-whitespace**: Remove trailing whitespace
  - **Check-added-large-files**: Prevent large file commits (limit: 500kb)
  - **Check-merge-conflict**: Prevent committing merge conflicts
  - **Check-toml**: Validate TOML files
  - **Debug-statements**: Detect leftover debug statements

### Development Environment Files
- Create `.env.example` with all required environment variables (without values):
  ```
  OPENAI_API_KEY=
  OUTPUT_DIR=
  LOG_LEVEL=
  ```
- Create `.editorconfig` for consistent coding standards:
  ```
  root = true
  
  [*]
  charset = utf-8
  end_of_line = lf
  insert_final_newline = true
  trim_trailing_whitespace = true
  
  [*.py]
  indent_style = space
  indent_size = 4
  max_line_length = 100
  
  [*.{yml,yaml}]
  indent_style = space
  indent_size = 2
  
  [*.md]
  trim_trailing_whitespace = false
  ```

### Setup Instructions
- Add to README.md or CONTRIBUTING.md:
  1. Install pre-commit: `pip install pre-commit` or `poetry add --dev pre-commit`
  2. Install git hooks: `pre-commit install`
  3. Run on all files (optional): `pre-commit run --all-files`
  4. Copy environment template: `cp .env.example .env`
  5. Configure environment variables in `.env`

## External Dependencies

- **pre-commit** (v3.0+) - Git hook framework for identifying issues before submission
  - **Justification:** Industry standard for managing git hooks, ensures consistent code quality across all contributors
  
- **pre-commit hooks** (managed by pre-commit):
  - black, ruff, mypy (already in dev dependencies)
  - pre-commit/pre-commit-hooks (standard hooks collection)