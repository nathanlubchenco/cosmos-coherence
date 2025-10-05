"""CLI commands for SelfCheckGPT benchmark."""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from cosmos_coherence.benchmarks.implementations.selfcheckgpt_benchmark import (
    SelfCheckGPTBenchmark,
)
from cosmos_coherence.benchmarks.models.datasets import SelfCheckGPTItem
from cosmos_coherence.harness.huggingface_loader import HuggingFaceDatasetLoader
from cosmos_coherence.llm.config import OpenAIConfig
from cosmos_coherence.llm.openai_client import OpenAIClient

app = typer.Typer(help="SelfCheckGPT benchmark commands")
console = Console()


@app.command()
def run(
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model",
        "-m",
        help="Model to evaluate (e.g., gpt-4o-mini, gpt-4, gpt-3.5-turbo)",
    ),
    sample_size: Optional[int] = typer.Option(
        None,
        "--sample-size",
        "-n",
        help="Number of passages to evaluate (default: all 238)",
    ),
    num_samples: int = typer.Option(
        5,
        "--num-samples",
        help="Number of stochastic samples at temp 1.0 (default: 5, paper uses 20)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to save results JSON (default: results_selfcheckgpt.json)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output for each passage",
    ),
    use_cache: bool = typer.Option(
        True,
        "--cache/--no-cache",
        help="Enable/disable response caching for efficiency (default: enabled)",
    ),
):
    """Run SelfCheckGPT benchmark for hallucination detection via consistency checking.

    This benchmark implements the SelfCheckGPT methodology from Manakul et al. (2023):
    https://arxiv.org/abs/2303.08896

    The benchmark generates multiple samples at different temperatures and uses
    NLI (Natural Language Inference) to detect hallucinations through consistency analysis.
    """
    try:
        # Validate API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            console.print(
                "[red]Error:[/red] OPENAI_API_KEY environment variable not found.\n"
                "Please set your OpenAI API key:\n"
                "  export OPENAI_API_KEY='your-api-key-here'"
            )
            raise typer.Exit(code=1)

        # Set up output path
        if output is None:
            output = Path("results_selfcheckgpt.json")

        # Run the benchmark
        console.print("\n[bold]Running SelfCheckGPT Benchmark[/bold]")
        console.print(f"Model: [cyan]{model}[/cyan]")
        console.print(
            f"Sampling: [cyan]1 baseline (temp 0.0) + {num_samples} samples (temp 1.0)[/cyan]"
        )
        if sample_size:
            console.print(f"Sample size: [cyan]{sample_size}[/cyan] passages")
        else:
            console.print("Sample size: [cyan]All 238 passages[/cyan]")
        console.print(f"Cache: [cyan]{'Enabled' if use_cache else 'Disabled'}[/cyan]")
        console.print()

        # Run asynchronously
        results = asyncio.run(
            run_benchmark(
                model=model,
                sample_size=sample_size,
                num_samples=num_samples,
                verbose=verbose,
                use_cache=use_cache,
            )
        )

        # Save results
        with open(output, "w") as f:
            json.dump(results, f, indent=2)

        console.print(f"\n[green]Results saved to:[/green] {output}")

        # Display summary
        display_results_summary(results)

    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted by user[/yellow]")
        raise typer.Exit(code=130)
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {str(e)}")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


async def run_benchmark(
    model: str,
    sample_size: Optional[int],
    num_samples: int,
    verbose: bool,
    use_cache: bool,
) -> Dict:
    """Run the benchmark asynchronously.

    Args:
        model: Model name to evaluate
        sample_size: Number of passages to evaluate
        num_samples: Number of stochastic samples to generate
        verbose: Show detailed output
        use_cache: Enable caching

    Returns:
        Dictionary with results
    """
    # Load dataset
    console.print("[bold]Loading dataset...[/bold]")
    loader = HuggingFaceDatasetLoader()
    items = await loader.load_dataset("selfcheckgpt", sample_size=sample_size)
    console.print(f"Loaded {len(items)} passages\n")

    # Initialize OpenAI client
    cache_dir = Path.home() / ".cache" / "cosmos_coherence" / "selfcheckgpt"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{model.replace('/', '_')}_cache.json"

    openai_config = OpenAIConfig(api_key=os.getenv("OPENAI_API_KEY"), model=model)
    client = OpenAIClient(openai_config, enable_cache=use_cache, cache_file=cache_file)

    # Initialize benchmark
    benchmark = SelfCheckGPTBenchmark(
        client=client,
        num_samples=num_samples,
        hf_dataset_name="selfcheckgpt",
        use_huggingface=True,
    )

    # Run evaluation with progress bar
    results_list = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]Evaluating {len(items)} passages...", total=len(items))

        item: SelfCheckGPTItem
        for idx, item in enumerate(items):
            if not isinstance(item, SelfCheckGPTItem):
                console.print(f"[yellow]Skipping non-SelfCheckGPT item at index {idx}[/yellow]")
                continue

            try:
                # Evaluate item with consistency checking
                result = await benchmark.evaluate_item_with_consistency(item)

                # Add to results
                results_list.append(
                    {
                        "topic": result["topic"],
                        "aggregate_score": result["aggregate_score"],
                        "num_sentences": len(result["sentences"]),
                        "sentence_scores": result["sentence_scores"],
                        "baseline": result["baseline"],
                        "num_samples": result["num_samples"],
                    }
                )

                if verbose:
                    console.print(
                        f"\n[bold]{result['topic']}[/bold] - "
                        f"Aggregate score: {result['aggregate_score']:.3f}"
                    )

                progress.update(task, advance=1)

            except Exception as e:
                console.print(f"\n[red]Error evaluating {item.topic}:[/red] {str(e)}")
                if verbose:
                    console.print_exception()
                progress.update(task, advance=1)
                continue

    # Get cache statistics
    cache_stats = benchmark.get_cache_statistics()

    # Calculate aggregate metrics
    aggregate_scores = [r["aggregate_score"] for r in results_list]
    mean_aggregate_score = (
        sum(aggregate_scores) / len(aggregate_scores) if aggregate_scores else 0.0
    )

    return {
        "model": model,
        "num_samples_per_passage": num_samples,
        "num_passages_evaluated": len(results_list),
        "mean_aggregate_score": mean_aggregate_score,
        "results": results_list,
        "cache_statistics": cache_stats,
    }


def display_results_summary(results: Dict):
    """Display results summary in a table.

    Args:
        results: Results dictionary
    """
    console.print("\n[bold]Results Summary[/bold]\n")

    # Create results table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    table.add_row("Model", results["model"])
    table.add_row("Passages Evaluated", str(results["num_passages_evaluated"]))
    table.add_row("Samples per Passage", str(results["num_samples_per_passage"]))
    table.add_row("Mean Aggregate Score", f"{results['mean_aggregate_score']:.4f}")

    console.print(table)

    # Cache statistics
    if "cache_statistics" in results:
        stats = results["cache_statistics"]
        console.print("\n[bold]Cache Statistics[/bold]\n")

        cache_table = Table(show_header=True, header_style="bold cyan")
        cache_table.add_column("Metric", style="dim")
        cache_table.add_column("Value", justify="right")

        cache_table.add_row("Total Requests", str(stats.get("total_requests", 0)))
        cache_table.add_row("Cache Hits", str(stats.get("cache_hits", 0)))
        cache_table.add_row("Cache Misses", str(stats.get("cache_misses", 0)))
        cache_table.add_row("Hit Rate", f"{stats.get('hit_rate', 0.0):.1%}")
        cache_table.add_row("Tokens Saved", str(stats.get("tokens_saved", 0)))

        console.print(cache_table)

    console.print("\n[dim]Note: Higher aggregate scores indicate potential hallucinations.[/dim]")
    console.print("[dim]Paper baseline: AUC-PR 92.50% with 20 samples (we use 5 samples).[/dim]\n")


if __name__ == "__main__":
    app()
