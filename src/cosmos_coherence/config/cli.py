"""CLI for configuration management."""

import json
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from .loader import ConfigLoader, load_config

app = typer.Typer()
console = Console()


@app.command()
def validate(
    config_path: Path = typer.Argument(..., help="Path to configuration file"),
    base_config: Optional[Path] = typer.Option(None, "--base", "-b", help="Base configuration file"),
):
    """Validate a configuration file without running experiments."""
    try:
        config = load_config(config_path, base_config)
        console.print(f"[green]✓[/green] Configuration is valid: {config.name}")

        # Show summary
        table = Table(title="Configuration Summary")
        table.add_column("Component", style="cyan")
        table.add_column("Value", style="yellow")

        table.add_row("Experiment Name", config.name)
        table.add_row("Model", str(config.model.model_type.value))
        table.add_row("Benchmark", str(config.benchmark.benchmark_type.value))
        table.add_row("Strategy", str(config.strategy.strategy_type.value))

        if config.model.temperature != 1.0:
            table.add_row("Temperature", str(config.model.temperature))

        if config.strategy.k_responses:
            table.add_row("K Responses", str(config.strategy.k_responses))

        console.print(table)

    except FileNotFoundError as e:
        console.print(f"[red]✗[/red] File not found: {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Validation failed: {e}")
        raise typer.Exit(1)


@app.command()
def generate(
    output_dir: Path = typer.Argument(..., help="Output directory for generated configs"),
    models: str = typer.Option("gpt-5,gpt-4.1", help="Comma-separated model types"),
    benchmarks: str = typer.Option("faithbench,simpleqa", help="Comma-separated benchmark types"),
    strategies: str = typer.Option("baseline,k_response", help="Comma-separated strategy types"),
    base_config: Optional[Path] = typer.Option(None, "--base", "-b", help="Base configuration file"),
):
    """Generate configuration combinations for grid search."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse inputs
    model_list = [m.strip() for m in models.split(",")]
    benchmark_list = [b.strip() for b in benchmarks.split(",")]
    strategy_list = [s.strip() for s in strategies.split(",")]

    # Load base config if provided
    base_config_dict = {}
    if base_config:
        loader = ConfigLoader()
        base_config_dict = loader.load_yaml(base_config)

    generated = []

    # Generate all combinations
    for model in model_list:
        for benchmark in benchmark_list:
            for strategy in strategy_list:
                # Create configuration
                config_dict = {
                    "name": f"{model}_{benchmark}_{strategy}",
                    "base": base_config_dict.get("base", {
                        "api_key": "${OPENAI_API_KEY}",
                        "output_dir": "outputs",
                    }),
                    "model": {
                        "model_type": model,
                    },
                    "benchmark": {
                        "benchmark_type": benchmark,
                        "dataset_path": f"data/{benchmark}",
                    },
                    "strategy": {
                        "strategy_type": strategy,
                    },
                }

                # Add strategy-specific parameters
                if strategy == "k_response":
                    config_dict["strategy"]["k_responses"] = 5
                elif strategy == "coherence":
                    config_dict["strategy"]["k_responses"] = 3
                    config_dict["strategy"]["coherence_measures"] = ["shogenji"]

                # Save configuration
                config_name = f"{model}_{benchmark}_{strategy}.yaml"
                config_path = output_dir / config_name

                with open(config_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False)

                generated.append(config_name)

    console.print(f"[green]✓[/green] Generated {len(generated)} configurations in {output_dir}")
    for config in generated:
        console.print(f"  • {config}")


@app.command()
def show(
    config_path: Path = typer.Argument(..., help="Path to configuration file"),
    base_config: Optional[Path] = typer.Option(None, "--base", "-b", help="Base configuration file"),
    format: str = typer.Option("yaml", "--format", "-f", help="Output format (yaml/json)"),
):
    """Display resolved configuration with all overrides applied."""
    try:
        config = load_config(config_path, base_config)

        # Convert to dictionary
        config_dict = config.model_dump(exclude_none=True)

        # Format output
        if format.lower() == "json":
            output = json.dumps(config_dict, indent=2)
            lexer = "json"
        else:
            output = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
            lexer = "yaml"

        # Display with syntax highlighting
        syntax = Syntax(output, lexer, theme="monokai", line_numbers=True)
        console.print(syntax)

    except Exception as e:
        console.print(f"[red]✗[/red] Failed to load configuration: {e}")
        raise typer.Exit(1)


@app.command()
def diff(
    config1: Path = typer.Argument(..., help="First configuration file"),
    config2: Path = typer.Argument(..., help="Second configuration file"),
    base_config: Optional[Path] = typer.Option(None, "--base", "-b", help="Base configuration file"),
):
    """Compare two configurations to see differences."""
    try:
        # Load both configurations
        c1 = load_config(config1, base_config)
        c2 = load_config(config2, base_config)

        # Convert to dictionaries
        dict1 = c1.model_dump(exclude_none=True)
        dict2 = c2.model_dump(exclude_none=True)

        # Find differences
        diffs = find_differences(dict1, dict2)

        if not diffs:
            console.print("[green]✓[/green] Configurations are identical")
        else:
            table = Table(title=f"Differences: {config1.name} vs {config2.name}")
            table.add_column("Path", style="cyan")
            table.add_column(config1.name, style="yellow")
            table.add_column(config2.name, style="green")

            for path, val1, val2 in diffs:
                table.add_row(path, str(val1), str(val2))

            console.print(table)

    except Exception as e:
        console.print(f"[red]✗[/red] Failed to compare configurations: {e}")
        raise typer.Exit(1)


def find_differences(dict1, dict2, path=""):
    """Recursively find differences between two dictionaries."""
    diffs = []

    # Check keys in dict1
    for key in dict1:
        current_path = f"{path}.{key}" if path else key

        if key not in dict2:
            diffs.append((current_path, dict1[key], "NOT SET"))
        elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            # Recursively check nested dictionaries
            diffs.extend(find_differences(dict1[key], dict2[key], current_path))
        elif dict1[key] != dict2[key]:
            diffs.append((current_path, dict1[key], dict2[key]))

    # Check keys only in dict2
    for key in dict2:
        if key not in dict1:
            current_path = f"{path}.{key}" if path else key
            diffs.append((current_path, "NOT SET", dict2[key]))

    return diffs


@app.command()
def list_models():
    """List all available model types."""
    table = Table(title="Available Models")
    table.add_column("Model", style="cyan")
    table.add_column("Category", style="yellow")
    table.add_column("Notes", style="white")

    # GPT-5 models
    table.add_row("gpt-5", "GPT-5", "Latest, 128K output, temp=1")
    table.add_row("gpt-5-mini", "GPT-5", "Smaller variant")
    table.add_row("gpt-5-nano", "GPT-5", "Lightweight")

    # GPT-4.1 models
    table.add_row("gpt-4.1", "GPT-4.1", "1M context, 32K output")
    table.add_row("gpt-4.1-mini", "GPT-4.1", "Smaller variant")

    # Reasoning models
    table.add_row("o3", "Reasoning", "100K output, temp=1")
    table.add_row("o3-mini", "Reasoning", "Smaller reasoning")
    table.add_row("o4-mini", "Reasoning", "Latest mini")

    # Legacy
    table.add_row("gpt-4", "Legacy", "Original GPT-4")
    table.add_row("gpt-4o", "Legacy", "Optimized GPT-4")

    console.print(table)


@app.command()
def list_benchmarks():
    """List all available benchmark types."""
    table = Table(title="Available Benchmarks")
    table.add_column("Benchmark", style="cyan")
    table.add_column("Description", style="yellow")

    table.add_row("faithbench", "Factual accuracy benchmark")
    table.add_row("simpleqa", "Simple question answering")
    table.add_row("truthfulqa", "Truthfulness evaluation")
    table.add_row("fever", "Fact extraction and verification")
    table.add_row("halueval", "Hallucination evaluation")

    console.print(table)


@app.command()
def list_strategies():
    """List all available evaluation strategies."""
    table = Table(title="Available Strategies")
    table.add_column("Strategy", style="cyan")
    table.add_column("Description", style="yellow")

    table.add_row("baseline", "Single response, no coherence")
    table.add_row("k_response", "Multiple responses with majority voting")
    table.add_row("coherence", "Coherence-based selection")

    console.print(table)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
