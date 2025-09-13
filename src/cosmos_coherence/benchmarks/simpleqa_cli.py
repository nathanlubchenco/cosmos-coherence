"""CLI commands for SimpleQA benchmark."""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from cosmos_coherence.benchmarks.implementations.simpleqa_benchmark import SimpleQABenchmark
from cosmos_coherence.benchmarks.models.base import BenchmarkRunConfig, BenchmarkType
from cosmos_coherence.benchmarks.models.datasets import SimpleQAItem, SimpleQAResult
from cosmos_coherence.config.loader import load_config
from cosmos_coherence.llm.config import OpenAIConfig
from cosmos_coherence.llm.openai_client import OpenAIClient

app = typer.Typer(help="SimpleQA benchmark commands")
console = Console()


@app.command()
def run(
    model: str = typer.Option(
        "gpt-4",
        "--model",
        "-m",
        help="Model to evaluate (e.g., gpt-4, gpt-4-turbo, gpt-3.5-turbo)",
    ),
    sample_size: Optional[int] = typer.Option(
        None,
        "--sample-size",
        "-n",
        help="Number of samples to evaluate (default: all 4,326)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to save results JSON",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML configuration file",
    ),
    temperature: float = typer.Option(
        0.3,
        "--temperature",
        "-t",
        help="Temperature for generation (0.3-1.0, default: 0.3 for factual accuracy)",
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
        help="Show detailed output for each question",
    ),
    use_cache: bool = typer.Option(
        True,
        "--cache/--no-cache",
        help="Enable/disable response caching for efficiency (default: enabled)",
    ),
):
    """Run SimpleQA benchmark evaluation."""
    try:
        # Load configuration if provided
        if config:
            experiment_config = load_config(config)
            # Create benchmark config from experiment config
            benchmark_config = BenchmarkRunConfig(
                benchmark_type=BenchmarkType.SIMPLEQA,
                dataset_path=Path("simpleqa"),
                sample_size=sample_size or experiment_config.benchmark.sample_size,
                temperature_settings=[temperature],
                shuffle=False,
                use_cache=use_cache,  # Use the CLI flag value
            )
            # Use model from config if not overridden
            if not model:
                model = experiment_config.model.model_type.value
        else:
            # Create config from CLI arguments
            benchmark_config = BenchmarkRunConfig(
                benchmark_type=BenchmarkType.SIMPLEQA,
                dataset_path=Path("simpleqa"),  # Will use HuggingFace dataset
                sample_size=sample_size,
                temperature_settings=[temperature],
                shuffle=False,  # Maintain reproducibility
                use_cache=use_cache,  # Use the CLI flag value
            )

        # Validate API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            console.print(
                "[red]Error:[/red] OPENAI_API_KEY environment variable not found.\n"
                "Please set your OpenAI API key:\n"
                "  export OPENAI_API_KEY='your-api-key-here'"
            )
            raise typer.Exit(code=1)

        # Run the benchmark
        console.print("\n[bold]Running SimpleQA Benchmark[/bold]")
        console.print(f"Model: [cyan]{model}[/cyan]")
        console.print(f"Temperature: [cyan]{temperature}[/cyan]")
        if sample_size:
            console.print(f"Sample size: [cyan]{sample_size}[/cyan]")
        else:
            console.print("Sample size: [cyan]All 4,326 questions[/cyan]")
        console.print()

        # Run asynchronously
        results = asyncio.run(
            _run_benchmark(benchmark_config, api_key, model, temperature, show_progress, verbose)
        )

        # Display results
        _display_results(results)

        # Save results if output path provided
        if output:
            _save_results(results, output)
            console.print(f"\n[green]Results saved to:[/green] {output}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(code=1)


@app.command()
def validate_baseline(
    results_file: Path = typer.Argument(
        ...,
        help="Path to results JSON file from a previous run",
    ),
    show_details: bool = typer.Option(
        False,
        "--details",
        "-d",
        help="Show detailed comparison with baselines",
    ),
):
    """Compare benchmark results against published baselines."""
    try:
        # Load results
        with open(results_file, "r") as f:
            results = json.load(f)

        # Get baselines
        benchmark = SimpleQABenchmark(use_huggingface=False)
        baselines = benchmark.get_baseline_metrics()

        # Create comparison table
        table = Table(title="Baseline Comparison")
        table.add_column("Model", style="cyan")
        table.add_column("Your Score", style="green")
        table.add_column("Baseline", style="blue")
        table.add_column("Difference", style="yellow")
        table.add_column("Status", style="white")

        # Extract model name and score from results
        model_name = results.get("model", "unknown")
        exact_match_score = results.get("metrics", {}).get("exact_match_accuracy", 0.0)

        # Find matching baseline
        baseline_key = None
        baseline_score = None

        if "gpt-4" in model_name.lower():
            baseline_key = "gpt-4_accuracy"
            baseline_score = baselines.get(baseline_key, 0.82)
        elif "gpt-3.5" in model_name.lower():
            baseline_key = "gpt-3.5_accuracy"
            baseline_score = baselines.get(baseline_key, 0.68)

        if baseline_score:
            diff = exact_match_score - baseline_score
            diff_pct = diff * 100

            # Determine status
            if abs(diff_pct) <= 5:
                status = "[green]✓ Within 5%[/green]"
            else:
                status = "[red]⚠ >5% deviation[/red]"

            table.add_row(
                model_name,
                f"{exact_match_score:.1%}",
                f"{baseline_score:.1%}",
                f"{diff_pct:+.1f}%",
                status,
            )

        console.print(table)

        # Show additional metrics if available
        if show_details:
            console.print("\n[bold]Detailed Metrics:[/bold]")
            console.print(f"F1 Score: {results.get('metrics', {}).get('f1_score', 0.0):.3f}")
            console.print(f"Total Questions: {results.get('total_questions', 0)}")
            console.print(f"Correct Answers: {results.get('correct_answers', 0)}")

    except FileNotFoundError:
        console.print(f"[red]Error:[/red] Results file not found: {results_file}")
        raise typer.Exit(code=1)
    except json.JSONDecodeError:
        console.print("[red]Error:[/red] Invalid JSON in results file")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(code=1)


@app.command()
def export(
    results_file: Path = typer.Argument(
        ...,
        help="Path to results JSON file",
    ),
    output: Path = typer.Argument(
        ...,
        help="Path for JSONL export",
    ),
    include_metadata: bool = typer.Option(
        True,
        "--metadata/--no-metadata",
        help="Include metadata in export",
    ),
):
    """Export results to JSONL format."""
    try:
        # Load results
        with open(results_file, "r") as f:
            results = json.load(f)

        # Convert results to SimpleQAResult objects if needed
        from uuid import uuid4

        exp_id = uuid4()

        result_objects = []
        for item in results.get("items", []):
            if not isinstance(item, SimpleQAResult):
                # Convert dictionary to SimpleQAResult
                result = SimpleQAResult(
                    experiment_id=exp_id,
                    item_id=uuid4(),
                    question=item["question"],
                    prediction=item.get("response", item.get("prediction")),
                    ground_truth=item.get("expected", item.get("ground_truth")),
                    is_correct=item.get("correct"),
                    f1_score=item.get("f1_score"),
                    exact_match=item.get("exact_match"),
                    response_length=None,  # Optional field
                    ground_truth_length=None,  # Optional field
                )
                result_objects.append(result)
            else:
                result_objects.append(item)

        # Use SimpleQAResult's built-in JSONL export
        jsonl_content = SimpleQAResult.to_jsonl(result_objects, include_metadata=include_metadata)

        # Write to file
        with open(output, "w") as f:
            f.write(jsonl_content)

        console.print(f"[green]Exported to:[/green] {output}")
        console.print(f"Total items: {len(result_objects)}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(code=1)


async def _run_benchmark(
    config: BenchmarkRunConfig,
    api_key: str,
    model_name: str,
    temperature: float,
    show_progress: bool,
    verbose: bool,
) -> Dict:
    """Run the benchmark evaluation."""
    # Initialize components
    benchmark = SimpleQABenchmark(
        hf_dataset_name="simpleqa",
        sample_size=config.sample_size,
    )

    # Create OpenAIConfig with all required fields
    openai_config = OpenAIConfig(
        api_key=api_key,
        organization_id=None,
        base_url="https://api.openai.com/v1",
        default_model=model_name,
        timeout=30.0,
        max_retries=3,
    )

    # Initialize client with caching enabled
    # Cache file will be stored in .cache directory for efficiency
    from pathlib import Path

    cache_dir = Path.home() / ".cache" / "cosmos_coherence" / "simpleqa"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{model_name.replace('/', '_')}_cache.json"

    client = OpenAIClient(
        openai_config,
        enable_cache=config.use_cache,  # Use cache setting from config (default: True)
        cache_file=cache_file,  # Persistent cache file for efficiency
    )

    # Load dataset
    dataset = await benchmark.load_dataset()

    results: Dict = {
        "model": model_name,
        "temperature": temperature,
        "total_questions": len(dataset),
        "correct_answers": 0,
        "items": [],
        "metrics": {},
    }

    # Evaluate each item
    if show_progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Evaluating {len(dataset)} questions...", total=len(dataset))

            for item in dataset:
                if isinstance(item, SimpleQAItem):
                    result = await _evaluate_item(
                        benchmark, client, item, model_name, temperature, verbose
                    )
                    results["items"].append(result)
                    if result["correct"]:
                        results["correct_answers"] += 1
                progress.advance(task)
    else:
        for item in dataset:
            if isinstance(item, SimpleQAItem):
                result = await _evaluate_item(
                    benchmark, client, item, model_name, temperature, verbose
                )
                results["items"].append(result)
                if result["correct"]:
                    results["correct_answers"] += 1

    # Calculate aggregate metrics
    results["metrics"]["exact_match_accuracy"] = (
        results["correct_answers"] / results["total_questions"]
    )

    # Calculate average F1 score
    f1_scores = [item["f1_score"] for item in results["items"]]
    results["metrics"]["f1_score"] = sum(f1_scores) / len(f1_scores)

    return results


async def _evaluate_item(
    benchmark: SimpleQABenchmark,
    client: OpenAIClient,
    item: SimpleQAItem,
    model_name: str,
    temperature: float,
    verbose: bool,
) -> Dict:
    """Evaluate a single item."""
    # Get prompt
    prompt = benchmark.get_prompt(item)

    # Get model response
    model_response = await client.generate_response(
        prompt, model=model_name, temperature=temperature, max_tokens=50
    )
    response = model_response.content

    # Evaluate response
    eval_result = benchmark.evaluate_response(response, item.best_answer, item)

    # Show verbose output if requested
    if verbose:
        console.print(f"[dim]Q: {item.question}[/dim]")
        console.print(f"[dim]Expected: {item.best_answer}[/dim]")
        console.print(f"[dim]Got: {response}[/dim]")
        console.print(f"[dim]Correct: {eval_result.is_correct}[/dim]\n")

    return {
        "question": item.question,
        "expected": item.best_answer,
        "response": response,
        "correct": eval_result.is_correct,
        "f1_score": eval_result.metadata["f1_score"],
        "exact_match": eval_result.metadata["exact_match"],
    }


def _display_results(results: Dict) -> None:
    """Display evaluation results."""
    table = Table(title="SimpleQA Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Model", results["model"])
    table.add_row("Temperature", str(results["temperature"]))
    table.add_row("Total Questions", str(results["total_questions"]))
    table.add_row("Correct Answers", str(results["correct_answers"]))
    table.add_row("Exact Match Accuracy", f"{results['metrics']['exact_match_accuracy']:.1%}")
    table.add_row("Average F1 Score", f"{results['metrics']['f1_score']:.3f}")

    console.print(table)


def _save_results(results: Dict, output_path: Path) -> None:
    """Save results to JSON file."""
    from datetime import datetime

    results["timestamp"] = datetime.now().isoformat()

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    app()
