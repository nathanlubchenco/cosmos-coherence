"""Dash application for Cosmos Coherence dashboard."""
import dash
import plotly.graph_objects as go
from dash import dcc, html

# Initialize Dash app
app = dash.Dash(__name__, title="Cosmos Coherence Dashboard")

# Define the layout
app.layout = html.Div(
    [
        html.H1("Cosmos Coherence Dashboard"),
        html.P("LLM Hallucination Detection Framework"),
        html.Hr(),
        html.Div(
            [
                html.H2("Status"),
                html.P("Dashboard is running"),
            ]
        ),
        dcc.Graph(
            id="example-graph",
            figure=go.Figure(
                data=[go.Bar(x=["Model A", "Model B", "Model C"], y=[0.85, 0.92, 0.78])],
                layout=go.Layout(title="Coherence Scores (Example)"),
            ),
        ),
    ]
)

# Expose the server for gunicorn
server = app.server

if __name__ == "__main__":
    # Run the development server
    app.run_server(debug=True, host="0.0.0.0", port=8050)
