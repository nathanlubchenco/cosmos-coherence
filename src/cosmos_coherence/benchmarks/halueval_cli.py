"""CLI commands for HaluEval benchmark."""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from cosmos_coherence.benchmarks.implementations.halueval_benchmark import (
    HaluEvalBenchmark,
)
from cosmos_coherence.benchmarks.models.datasets import HaluEvalItem, HaluEvalTaskType
from cosmos_coherence.llm.config import OpenAIConfig
from cosmos_coherence.llm.openai_client import OpenAIClient

app = typer.Typer(help="HaluEval benchmark commands for hallucination detection")
console = Console()


def clear_halueval_cache() -> None:
    """Clear the HaluEval cache directory."""
    cache_dir = Path.home() / ".cache" / "cosmos_coherence" / "halueval"
    if cache_dir.exists():
        import shutil

        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)


async def run_evaluation(
    model: str,
    sample_size: Optional[int],
    task_type: Optional[str],
    temperature: float,
    use_cache: bool,
    show_progress: bool,
    verbose: bool,
) -> Dict:
    """Run HaluEval benchmark evaluation.

    Args:
        model: Model name to evaluate
        sample_size: Number of samples to evaluate
        task_type: Specific task type to evaluate
        temperature: Temperature for generation
        use_cache: Whether to use caching
        show_progress: Whether to show progress
        verbose: Whether to show detailed output

    Returns:
        Dictionary of results
    """
    # Set up cache directory
    cache_dir = Path.home() / ".cache" / "cosmos_coherence" / "halueval"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{model.replace('/', '_')}_cache.json"

    # Initialize OpenAI client
    config = OpenAIConfig(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        organization_id=None,
        base_url="https://api.openai.com/v1",
        default_model=model,
        timeout=30.0,
        max_retries=3,
    )

    client = OpenAIClient(
        openai_config=config,
        enable_cache=use_cache,
        cache_file=str(cache_file) if use_cache else None,
    )

    # Initialize benchmark with fixed random seed for reproducibility
    benchmark = HaluEvalBenchmark(
        client=client,
        hf_dataset_name="halueval",
        sample_size=sample_size,
        random_seed=42,  # Fixed seed for reproducible answer selection
    )

    # Load dataset
    if show_progress:
        console.print("[cyan]Loading HaluEval dataset...[/cyan]")

    dataset = await benchmark.load_dataset()

    # Filter by task type if specified
    if task_type:
        task_enum = HaluEvalTaskType[task_type.upper()]
        dataset = [
            item
            for item in dataset
            if isinstance(item, HaluEvalItem) and item.task_type == task_enum
        ]

    if verbose:
        console.print(f"[green]Loaded {len(dataset)} items[/green]")

    # Run evaluation
    results = []
    total = len(dataset)

    if show_progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"[cyan]Evaluating {total} samples...", total=total)

            for i, item in enumerate(dataset):
                # Get prompt with random selection
                user_prompt, is_hallucinated, system_prompt = benchmark.get_prompt_with_selection(
                    item
                )

                # Get model response
                try:
                    response = await client.generate_response(
                        user_prompt,
                        temperature=temperature,
                        max_tokens=10,
                        system_prompt=system_prompt,
                    )
                    model_answer = response.content

                    # Evaluate response
                    ground_truth = "hallucinated" if is_hallucinated else "not_hallucinated"
                    eval_result = benchmark.evaluate_response(model_answer, ground_truth, item)

                    results.append(
                        {
                            "item_id": str(item.id),
                            "task_type": item.task_type.value
                            if hasattr(item, "task_type")
                            else "unknown",
                            "is_correct": eval_result.is_correct,
                            "score": eval_result.score,
                            "prediction": eval_result.metadata.get("prediction"),
                            "expected": eval_result.metadata.get("expected"),
                        }
                    )

                    if verbose and (i < 5 or not eval_result.is_correct):
                        console.print(
                            f"[yellow]Item {i+1}:[/yellow] "
                            f"Task: {results[-1]['task_type']}, "
                            f"Prediction: {results[-1]['prediction']}, "
                            f"Expected: {results[-1]['expected']}, "
                            f"Correct: {eval_result.is_correct}"
                        )

                except Exception as e:
                    console.print(f"[red]Error evaluating item {i+1}: {e}[/red]")
                    results.append(
                        {
                            "item_id": str(item.id),
                            "task_type": item.task_type.value
                            if hasattr(item, "task_type")
                            else "unknown",
                            "is_correct": False,
                            "score": 0.0,
                            "error": str(e),
                        }
                    )

                progress.update(task, advance=1)
    else:
        for i, item in enumerate(dataset):
            # Get prompt with random selection
            user_prompt, is_hallucinated, system_prompt = benchmark.get_prompt_with_selection(item)

            # Get model response
            try:
                response = await client.generate_response(
                    user_prompt, temperature=temperature, max_tokens=10, system_prompt=system_prompt
                )
                model_answer = response.content

                # Evaluate response
                ground_truth = "hallucinated" if is_hallucinated else "not_hallucinated"
                eval_result = benchmark.evaluate_response(model_answer, ground_truth, item)

                results.append(
                    {
                        "item_id": str(item.id),
                        "task_type": item.task_type.value
                        if hasattr(item, "task_type")
                        else "unknown",
                        "is_correct": eval_result.is_correct,
                        "score": eval_result.score,
                        "prediction": eval_result.metadata.get("prediction"),
                        "expected": eval_result.metadata.get("expected"),
                    }
                )

            except Exception as e:
                results.append(
                    {
                        "item_id": str(item.id),
                        "task_type": item.task_type.value
                        if hasattr(item, "task_type")
                        else "unknown",
                        "is_correct": False,
                        "score": 0.0,
                        "error": str(e),
                    }
                )

    # Calculate metrics
    metrics = benchmark.calculate_metrics(results)

    # Save cache to disk if caching is enabled
    if use_cache and client._cache:
        try:
            client._cache.save_to_disk(cache_file)
            if verbose:
                console.print(f"[green]Cache saved to {cache_file}[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to save cache: {e}[/yellow]")

    return {
        "model": model,
        "temperature": temperature,
        "task_type": task_type,
        "total_samples": len(results),
        "correct_samples": metrics.get("correct_samples", 0),
        "metrics": metrics,
        "items": results,
    }


@app.command()
def run(
    model: str = typer.Option(
        "gpt-4",
        "--model",
        "-m",
        help="Model to evaluate (e.g., gpt-4, gpt-3.5-turbo)",
    ),
    sample_size: Optional[int] = typer.Option(
        None,
        "--sample-size",
        "-n",
        help="Number of samples to evaluate (default: all)",
    ),
    task_type: Optional[str] = typer.Option(
        None,
        "--task-type",
        "-t",
        help="Specific task type to evaluate (qa, dialogue, summarization)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to save results JSON",
    ),
    temperature: float = typer.Option(
        0.0,
        "--temperature",
        help="Temperature for generation (0.0 for deterministic)",
    ),
    show_progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Show progress bar during evaluation",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output for each item",
    ),
    use_cache: bool = typer.Option(
        True,
        "--cache/--no-cache",
        help="Enable/disable response caching (default: enabled)",
    ),
):
    """Run HaluEval benchmark evaluation for hallucination detection.

    Evaluates a model's ability to detect hallucinations across QA, dialogue,
    and summarization tasks.
    """
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[red]Error: OPENAI_API_KEY environment variable is required[/red]")
        raise typer.Exit(1)

    # Validate task type
    if task_type and task_type.lower() not in ["qa", "dialogue", "summarization"]:
        console.print(
            f"[red]Invalid task type: {task_type}. "
            "Choose from: qa, dialogue, summarization[/red]"
        )
        raise typer.Exit(1)

    # Run evaluation
    console.print(f"[cyan]Running HaluEval benchmark with {model}...[/cyan]")

    results = asyncio.run(
        run_evaluation(
            model=model,
            sample_size=sample_size,
            task_type=task_type.lower() if task_type else None,
            temperature=temperature,
            use_cache=use_cache,
            show_progress=show_progress,
            verbose=verbose,
        )
    )

    # Display results
    display_results(results)

    # Save results if output specified
    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        console.print(f"[green]Results saved to {output}[/green]")


def display_results(results: Dict) -> None:
    """Display evaluation results in a formatted table.

    Args:
        results: Dictionary of evaluation results
    """
    console.print("\n[bold cyan]HaluEval Results[/bold cyan]")
    console.print("=" * 50)

    # Basic info
    console.print(f"Model: [yellow]{results['model']}[/yellow]")
    console.print(f"Temperature: {results['temperature']}")
    if results.get("task_type"):
        console.print(f"Task Type: {results['task_type']}")
    console.print(f"Total Samples: {results['total_samples']}")
    console.print(f"Correct Samples: {results['correct_samples']}")

    # Metrics table
    if results.get("metrics"):
        console.print("\n[bold]Metrics:[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        metrics = results["metrics"]
        for key, value in metrics.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.3f}")
            else:
                table.add_row(key, str(value))

        console.print(table)


@app.command()
def compare(
    result1: Path = typer.Argument(..., help="First result file to compare"),
    result2: Path = typer.Argument(..., help="Second result file to compare"),
):
    """Compare model results from two evaluation runs.

    Loads two result JSON files and displays a comparison table.
    """
    # Load results
    if not result1.exists():
        console.print(f"[red]File not found: {result1}[/red]")
        raise typer.Exit(1)
    if not result2.exists():
        console.print(f"[red]File not found: {result2}[/red]")
        raise typer.Exit(1)

    with open(result1) as f:
        data1 = json.load(f)
    with open(result2) as f:
        data2 = json.load(f)

    # Display comparison
    console.print("\n[bold cyan]Comparison Results[/bold cyan]")
    console.print("=" * 60)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column(f"{data1['model']}", justify="right")
    table.add_column(f"{data2['model']}", justify="right")
    table.add_column("Difference", justify="right")

    # Compare metrics
    metrics1 = data1.get("metrics", {})
    metrics2 = data2.get("metrics", {})

    for key in set(metrics1.keys()) | set(metrics2.keys()):
        val1 = metrics1.get(key, 0)
        val2 = metrics2.get(key, 0)
        if isinstance(val1, float) and isinstance(val2, float):
            diff = val1 - val2
            color = "green" if diff > 0 else "red" if diff < 0 else "white"
            table.add_row(
                key,
                f"{val1:.3f}",
                f"{val2:.3f}",
                f"[{color}]{diff:+.3f}[/{color}]",
            )

    console.print(table)


@app.command()
def baseline(
    accuracy: float = typer.Option(
        ..., "--accuracy", "-a", help="Your model's accuracy to compare"
    ),
):
    """Compare your results against published baselines.

    Shows how your model's performance compares to GPT-3.5 baseline from the paper.
    """
    console.print("\n[bold cyan]Baseline Comparison[/bold cyan]")
    console.print("=" * 50)

    baseline_accuracy = 0.65  # GPT-3.5 baseline from paper

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Accuracy", justify="right")
    table.add_column("Difference", justify="right")

    table.add_row("GPT-3.5 (Paper)", f"{baseline_accuracy:.3f}", "-")
    diff = accuracy - baseline_accuracy
    color = "green" if diff > 0 else "red" if diff < 0 else "white"
    table.add_row(
        "Your Model",
        f"{accuracy:.3f}",
        f"[{color}]{diff:+.3f}[/{color}]",
    )

    console.print(table)

    if diff > 0:
        console.print(f"[green]✓ Your model outperforms the baseline by {diff:.1%}[/green]")
    elif diff < 0:
        console.print(f"[yellow]Your model underperforms the baseline by {abs(diff):.1%}[/yellow]")
    else:
        console.print("[white]Your model matches the baseline performance[/white]")


@app.command("clear-cache")
def clear_cache(
    confirm: bool = typer.Option(
        False,
        "--confirm",
        help="Confirm cache clearing without prompt",
    ),
):
    """Clear the HaluEval cache directory.

    Removes all cached API responses for HaluEval evaluations.
    """
    cache_dir = Path.home() / ".cache" / "cosmos_coherence" / "halueval"

    if not cache_dir.exists():
        console.print("[yellow]Cache directory does not exist[/yellow]")
        return

    if not confirm:
        confirm = typer.confirm(
            f"Clear cache directory {cache_dir}?",
            default=False,
        )

    if confirm:
        clear_halueval_cache()
        console.print("[green]Cache cleared successfully[/green]")
    else:
        console.print("[yellow]Cache clearing cancelled[/yellow]")


if __name__ == "__main__":
    app()
