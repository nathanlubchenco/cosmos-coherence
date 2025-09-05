# Code Style Guide

## Context

Global code style rules for Agent OS projects.

<conditional-block context-check="general-formatting">
IF this General Formatting section already read in current context:
  SKIP: Re-reading this section
  NOTE: "Using General Formatting rules already in context"
ELSE:
  READ: The following formatting rules

## General Formatting

### Indentation
- Use 4 spaces for indentation (PEP 8 standard)
- Never use tabs
- Maintain consistent indentation throughout files
- Align nested structures for readability

### Naming Conventions
- **Functions and Variables**: Use snake_case (e.g., `user_profile`, `calculate_total`)
- **Classes**: Use PascalCase (e.g., `UserProfile`, `PaymentProcessor`)
- **Constants**: Use UPPER_SNAKE_CASE (e.g., `MAX_RETRY_COUNT`, `DEFAULT_TIMEOUT`)
- **Private Methods/Variables**: Prefix with single underscore (e.g., `_internal_method`)
- **Module Files**: Use lowercase with underscores (e.g., `data_processor.py`)

### String Formatting
- Use double quotes for strings: `"Hello World"`
- Use single quotes for dictionary keys and short identifiers
- Use f-strings for string interpolation: `f"User {name} logged in"`
- Use triple quotes for docstrings and multi-line strings

### Code Comments
- Add docstrings to all modules, classes, and functions
- Use Google-style or NumPy-style docstrings consistently
- Document complex algorithms or calculations inline
- Explain the "why" behind implementation choices
- Keep comments concise and relevant
- Update comments when modifying code

### Type Hints
- Use type hints for function signatures
- Include return type annotations
- Use `from typing import` for complex types
- Example: `def process_data(items: list[str]) -> dict[str, int]:`

### Imports
- Group imports in order: standard library, third-party, local
- Sort alphabetically within each group
- Use absolute imports when possible
- One import per line (except for `typing`)
</conditional-block>

<conditional-block task-condition="python" context-check="python-style">
IF current task involves writing or updating Python code:
  IF python-style.md already in context:
    SKIP: Re-reading this file
    NOTE: "Using Python style guide already in context"
  ELSE:
    READ: The following Python-specific rules

## Python-Specific Style Rules

### Code Organization
- Keep functions small and focused (max 20-30 lines preferred)
- Use classes for stateful components
- Use modules to organize related functionality
- Follow single responsibility principle

### Data Modeling with Pydantic
- Use Pydantic models for all data structures (API requests/responses, configs, domain models)
- Define models in dedicated `models.py` or `schemas.py` files
- Use strict typing with Pydantic's validation
- Leverage Pydantic's `Field()` for validation rules and documentation
- Use `Config` class for model configuration
- Example structure:
```python
from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime

class UserRequest(BaseModel):
    """User creation request model."""

    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    email: str = Field(..., regex=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    age: Optional[int] = Field(None, ge=0, le=120)

    class Config:
        schema_extra = {
            "example": {
                "username": "johndoe",
                "email": "john@example.com",
                "age": 25
            }
        }

    @validator("username")
    def username_alphanumeric(cls, v):
        assert v.isalnum(), "Username must be alphanumeric"
        return v
```

### Configuration Management
- Use Pydantic Settings for environment variables and configuration
- Create a central `config.py` with settings classes
- Support `.env` files with `python-dotenv`
- Example:
```python
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """Application settings."""

    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    debug_mode: bool = Field(False, env="DEBUG")
    data_dir: str = Field("./data", env="DATA_DIR")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

### Error Handling
- Use specific exception types
- Always include error messages in exceptions
- Use context managers (`with` statements) for resource management
- Log errors appropriately
- Use Pydantic's `ValidationError` for data validation failures

### File Structure
```python
"""Module docstring."""

# Standard library imports
import os
import sys

# Third-party imports
import pandas as pd
from fastapi import FastAPI

# Local imports
from .utils import helper_function

# Constants
DEFAULT_TIMEOUT = 30

# Main code
class MainClass:
    """Class docstring."""
    pass
```

### Testing Conventions
- Test files named `test_*.py` or `*_test.py`
- Use pytest fixtures for setup/teardown
- Aim for descriptive test names: `test_should_return_error_when_input_invalid()`
- Group related tests in classes

### FastAPI Integration with Pydantic
- Use Pydantic models directly as FastAPI request/response models
- Separate request models from response models for clarity
- Use dependency injection for shared validation logic
- Example:
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

class ItemCreate(BaseModel):
    """Request model for creating items."""
    name: str
    price: float
    tags: List[str] = []

class ItemResponse(BaseModel):
    """Response model for items."""
    id: int
    name: str
    price: float
    tags: List[str]
    created_at: datetime

    class Config:
        orm_mode = True  # Enable ORM compatibility if needed

app = FastAPI()

@app.post("/items", response_model=ItemResponse)
async def create_item(item: ItemCreate):
    """Create a new item."""
    # Pydantic automatically validates the request
    return ItemResponse(id=1, created_at=datetime.now(), **item.dict())
```

### Makefile Commands
- Use lowercase with hyphens for targets: `make run-tests`
- Include help target with descriptions
- Common targets: `build`, `test`, `run`, `clean`, `lint`, `format`
</conditional-block>

<conditional-block task-condition="dash-frontend" context-check="dash-style">
IF current task involves writing or updating Dash frontend:
  IF dash-style.md already in context:
    SKIP: Re-reading this file
    NOTE: "Using Dash style guide already in context"
  ELSE:
    READ: The following Dash-specific rules

## Dash-Specific Style Rules

### Component Organization
- Separate layout from callbacks
- Group related callbacks together
- Use component IDs with clear naming: `"input-user-name"`
- Prefix IDs by component type: `"btn-submit"`, `"graph-revenue"`

### Layout Structure
- Define layouts in separate functions
- Use Dash Bootstrap Components for consistent styling
- Keep layout definitions readable with proper indentation
- Extract repeated components into functions

### Callback Best Practices
- Use clear Input/Output/State declarations
- Validate inputs at callback start
- Handle errors gracefully with try/except
- Return meaningful error states to UI
- Use `prevent_initial_call=True` when appropriate

### File Organization
```python
"""Dashboard module for X functionality."""

# Imports
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# Layout components
def create_header():
    """Create dashboard header."""
    pass

def create_sidebar():
    """Create sidebar navigation."""
    pass

# Main layout
def get_layout():
    """Return complete dashboard layout."""
    return dbc.Container([
        create_header(),
        create_sidebar(),
        # Main content
    ])

# Callbacks
def register_callbacks(app):
    """Register all callbacks for this dashboard."""

    @app.callback(
        Output("output-id", "children"),
        Input("input-id", "value")
    )
    def update_output(value):
        """Update output based on input."""
        return process_value(value)
```
</conditional-block>
