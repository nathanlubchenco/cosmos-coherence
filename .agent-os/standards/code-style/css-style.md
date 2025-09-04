# CSS Style Guide for Dash Applications

## Dash Bootstrap Components Styling

We use Dash Bootstrap Components (DBC) for consistent styling across the application.

### Component Styling Approach

- Use DBC theme classes for consistency
- Apply inline styles sparingly via the `style` prop
- Use className prop for Bootstrap utility classes
- Keep visual styling in Python dictionaries for reusability

### Style Organization

**Define reusable styles as Python constants:**

```python
# styles.py
CARD_STYLE = {
    "margin": "10px",
    "padding": "15px",
    "borderRadius": "8px",
    "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
}

GRAPH_CONFIG = {
    "displayModeBar": False,
    "responsive": True
}
```

**Apply styles in Dash components:**

```python
import dash_bootstrap_components as dbc

card = dbc.Card([
    dbc.CardBody([
        html.H4("Card Title", className="card-title"),
        html.P("Card content", className="card-text")
    ])
], style=CARD_STYLE, className="mb-3")
```

### Bootstrap Theme Classes

- Use Bootstrap utility classes for spacing: `mb-3`, `p-2`, `mt-4`
- Use Bootstrap color schemes: `primary`, `secondary`, `success`, `danger`
- Leverage responsive grid: `dbc.Row`, `dbc.Col` with breakpoints
- Apply consistent theming with `color` and `outline` props
