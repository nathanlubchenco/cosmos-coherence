"""CLI commands specific to FaithBench benchmark."""

import asyncio
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from cosmos_coherence.benchmarks.faithbench import FaithBenchBenchmark
from cosmos_coherence.benchmarks.faithbench_metrics import FaithBenchMetrics
from cosmos_coherence.benchmarks.models.datasets import FaithBenchAnnotation
from cosmos_coherence.harness.base_benchmark import BenchmarkEvaluationResult

app = typer.Typer(help="FaithBench-specific commands")
console = Console()


@app.command()
def run(
    model: str = typer.Option(
        "gpt-4-turbo",
        "--model",
        "-m",
        help="Model to evaluate (gpt-4-turbo, gpt-4o, o1-mini, o3-mini)",
    ),
    sample_size: Optional[int] = typer.Option(
        None,
        "--sample-size",
        "-n",
        help="Number of samples to evaluate (default: all)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to save results JSON",
    ),
    cache_dir: Optional[Path] = typer.Option(
        None,
        "--cache-dir",
        help="Directory for caching datasets",
    ),
    temperature: Optional[float] = typer.Option(
        None,
        "--temperature",
        "-t",
        help="Temperature for generation (not supported for o1/o3 models)",
    ),
    show_challenging: bool = typer.Option(
        False,
        "--show-challenging",
        help="Show performance on challenging samples",
    ),
    compare_baseline: bool = typer.Option(
        False,
        "--compare-baseline",
        help="Compare with paper baseline metrics",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable LLM response caching",
    ),
    show_cache_stats: bool = typer.Option(
        False,
        "--show-cache-stats",
        help="Display cache statistics after evaluation",
    ),
):
    """Run FaithBench evaluation."""
    try:
        # Validate model choice
        if model not in FaithBenchBenchmark.SUPPORTED_MODELS:
            console.print(
                f"[red]Error:[/red] Model '{model}' not supported. "
                f"Choose from: {', '.join(sorted(FaithBenchBenchmark.SUPPORTED_MODELS))}"
            )
            raise typer.Exit(code=1)

        # Validate temperature for reasoning models
        if model in FaithBenchBenchmark.NO_TEMPERATURE_MODELS and temperature is not None:
            if temperature != 0.0:
                console.print(
                    f"[red]Error:[/red] Model '{model}' doesn't support temperature variation"
                )
                raise typer.Exit(code=1)

        # Get API key from environment or prompt
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            console.print(
                "[red]Error:[/red] OPENAI_API_KEY environment variable not found.\n"
                "Please set your OpenAI API key:\n"
                "  export OPENAI_API_KEY='your-api-key-here'"
            )
            raise typer.Exit(code=1)

        # Create benchmark instance
        benchmark = FaithBenchBenchmark(cache_dir=cache_dir, api_key=api_key)

        # Configure cache settings if needed
        if no_cache:
            # Set environment variable to disable cache
            os.environ["COSMOS_DISABLE_CACHE"] = "1"

        # Prepare configuration
        config = {
            "model": model,
            "sample_size": sample_size,
            "temperature": temperature if temperature is not None else 0.0,
        }

        # Validate configuration
        benchmark.validate_config(config)

        console.print(f"[bold]Running FaithBench with {model}[/bold]")
        if sample_size:
            console.print(f"  Sample size: {sample_size}")
        console.print(f"  Temperature: {config['temperature']}")

        # Load dataset
        async def load_and_evaluate():
            dataset = await benchmark.load_dataset(sample_size=sample_size)
            console.print(f"  Loaded {len(dataset)} items")

            results = []

            # Use progress bar for evaluation
            from rich.progress import track

            for i, item in enumerate(track(dataset, description="Evaluating...")):
                # Use actual LLM evaluation
                is_consistent = await benchmark.evaluate_response_with_llm(
                    response=item.summary if hasattr(item, "summary") else "",
                    source=item.source if hasattr(item, "source") else "",
                    model=model,
                )

                # Create evaluation result based on LLM response
                # Binary classification per FaithBench binarize.py:
                # - Binary label 1 (not hallucinated): Items NOT in hallucinated_classes
                # - Binary label 0 (hallucinated): Items in hallucinated_classes
                # - classes = [Questionable, Unwanted, Unwanted_Intrinsic, Unwanted_Extrinsic]
                # So: CONSISTENT and BENIGN map to 1 (not hallucinated)
                #     QUESTIONABLE and HALLUCINATED map to 0 (hallucinated)
                is_label_consistent = (
                    item.annotation_label
                    in [FaithBenchAnnotation.CONSISTENT, FaithBenchAnnotation.BENIGN]
                    if hasattr(item, "annotation_label") and item.annotation_label
                    else True
                )
                result = BenchmarkEvaluationResult(
                    is_correct=is_consistent == is_label_consistent,
                    score=1.0 if is_consistent else 0.0,
                    original_metric_score=1.0 if is_consistent else 0.0,
                    metadata={
                        "annotation_label": item.annotation_label.value
                        if hasattr(item, "annotation_label") and item.annotation_label
                        else None,
                        "entropy_score": item.entropy_score
                        if hasattr(item, "entropy_score")
                        else None,
                        "is_challenging": (
                            item.entropy_score > 0.67
                            if hasattr(item, "entropy_score") and item.entropy_score
                            else False
                        ),
                        "llm_evaluation": "consistent" if is_consistent else "hallucinated",
                    },
                )

                results.append(result)

            return results

        results = asyncio.run(load_and_evaluate())

        # Save cache to disk if using cache file
        if (
            benchmark.openai_client
            and benchmark.openai_client._cache
            and benchmark.openai_client._cache._cache_file
        ):
            benchmark.openai_client._cache.save_to_disk(benchmark.openai_client._cache._cache_file)
            console.print(
                f"[green]✓[/green] Cache saved to {benchmark.openai_client._cache._cache_file}"
            )

        # Calculate metrics
        metrics_calculator = FaithBenchMetrics()
        metrics = metrics_calculator.calculate_aggregate_metrics(results)

        # Display cache statistics if requested
        if show_cache_stats and benchmark.openai_client:
            cache_stats = benchmark.openai_client.get_cache_statistics()
            if cache_stats:
                console.print("\n[bold]Cache Statistics[/bold]")
                cache_table = Table()
                cache_table.add_column("Metric", style="cyan")
                cache_table.add_column("Value", style="white")

                cache_table.add_row("Total Requests", str(cache_stats.total_requests))
                cache_table.add_row("Cache Hits", str(cache_stats.cache_hits))
                cache_table.add_row("Cache Misses", str(cache_stats.cache_misses))
                cache_table.add_row("Hit Rate", f"{cache_stats.hit_rate:.1%}")
                cache_table.add_row("Tokens Saved", f"{cache_stats.tokens_saved:,}")
                cache_table.add_row(
                    "Estimated Savings", f"${cache_stats.estimated_cost_savings():.4f}"
                )

                console.print(cache_table)

        # Display results
        console.print("\n[bold green]Evaluation Complete[/bold green]")

        # Main metrics table
        table = Table(title="Overall Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        key_metrics = [
            ("Overall Accuracy", metrics.get("overall_accuracy", 0)),
            ("Balanced Accuracy", metrics.get("balanced_accuracy", 0)),
            ("F1-Macro (Binary)", metrics.get("f1_macro_binary", metrics.get("f1_macro", 0))),
            ("Precision (Hallucinated)", metrics.get("precision_hallucinated", 0)),
            ("Recall (Hallucinated)", metrics.get("recall_hallucinated", 0)),
            ("F1 (Hallucinated)", metrics.get("f1_hallucinated", 0)),
        ]

        for name, value in key_metrics:
            table.add_row(name, f"{value:.3f}")

        console.print(table)

        # Challenging samples metrics if requested
        if show_challenging:
            console.print("\n[bold]Challenging Samples Performance[/bold]")
            challenging_table = Table()
            challenging_table.add_column("Metric", style="cyan")
            challenging_table.add_column("Value", style="white")

            challenging_metrics = [
                ("Challenging Accuracy", metrics.get("challenging_accuracy", 0)),
                ("Challenging Count", metrics.get("challenging_count", 0)),
                ("Non-Challenging Accuracy", metrics.get("non_challenging_accuracy", 0)),
                ("Average Entropy", metrics.get("average_entropy", 0)),
            ]

            for name, value in challenging_metrics:
                if isinstance(value, int):
                    challenging_table.add_row(name, str(value))
                else:
                    challenging_table.add_row(name, f"{value:.3f}")

            console.print(challenging_table)

        # Compare with baseline if requested
        if compare_baseline:
            console.print("\n[bold]Comparison with Paper Baseline[/bold]")
            baseline = benchmark.get_baseline_metrics()

            comparison_table = Table()
            comparison_table.add_column("Model", style="cyan")
            comparison_table.add_column("Paper Baseline", style="white")
            comparison_table.add_column("Your Result", style="white")
            comparison_table.add_column("Difference", style="yellow")

            model_baseline_key = f"{model}_accuracy"
            if model_baseline_key in baseline:
                baseline_acc = baseline[model_baseline_key]
                # Use balanced accuracy for comparison (paper likely reports balanced accuracy)
                current_acc = metrics.get("balanced_accuracy", metrics.get("overall_accuracy", 0))
                diff = current_acc - baseline_acc

                comparison_table.add_row(
                    model,
                    f"{baseline_acc:.3f}",
                    f"{current_acc:.3f}",
                    f"{diff:+.3f}",
                )

                console.print(comparison_table)

        # Save results if output path provided
        if output:
            import json

            output_data = {
                "model": model,
                "config": config,
                "metrics": metrics,
                "detailed_metrics": metrics_calculator.export_metrics(metrics),
            }

            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                json.dump(output_data, f, indent=2, default=str)

            console.print(f"\n[green]✓[/green] Results saved to {output}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def analyze(
    results_file: Path = typer.Argument(..., help="Path to FaithBench results JSON"),
    show_per_label: bool = typer.Option(
        False,
        "--per-label",
        help="Show metrics for each annotation label",
    ),
    show_entropy: bool = typer.Option(
        False,
        "--entropy",
        help="Show entropy-stratified metrics",
    ),
    export_report: Optional[Path] = typer.Option(
        None,
        "--export",
        "-e",
        help="Export detailed analysis report",
    ),
):
    """Analyze FaithBench evaluation results."""
    try:
        import json

        with open(results_file) as f:
            data = json.load(f)

        metrics = data.get("metrics", {})
        detailed = data.get("detailed_metrics", {})

        console.print("[bold]FaithBench Analysis[/bold]")
        console.print(f"Model: {data.get('model', 'Unknown')}")

        # Overall performance
        console.print("\n[bold]Overall Performance[/bold]")
        overall_table = Table()
        overall_table.add_column("Metric", style="cyan")
        overall_table.add_column("Value", style="white")

        for key in ["overall_accuracy", "overall_precision", "overall_recall", "overall_f1"]:
            if key in metrics:
                overall_table.add_row(
                    key.replace("_", " ").title(),
                    f"{metrics[key]:.3f}",
                )

        console.print(overall_table)

        # Per-label metrics if requested
        if show_per_label:
            console.print("\n[bold]Per-Label Performance[/bold]")
            label_table = Table()
            label_table.add_column("Label", style="cyan")
            label_table.add_column("Precision", style="white")
            label_table.add_column("Recall", style="white")
            label_table.add_column("F1", style="white")

            for label in ["consistent", "questionable", "benign", "hallucinated"]:
                precision = metrics.get(f"precision_{label}", 0)
                recall = metrics.get(f"recall_{label}", 0)
                f1 = metrics.get(f"f1_{label}", 0)

                label_table.add_row(
                    label.title(),
                    f"{precision:.3f}",
                    f"{recall:.3f}",
                    f"{f1:.3f}",
                )

            console.print(label_table)

        # Entropy-stratified metrics if requested
        if show_entropy:
            console.print("\n[bold]Entropy-Stratified Performance[/bold]")
            entropy_table = Table()
            entropy_table.add_column("Entropy Level", style="cyan")
            entropy_table.add_column("Accuracy", style="white")
            entropy_table.add_column("Count", style="white")

            for level in ["low", "medium", "high"]:
                acc_key = f"{level}_entropy_accuracy"
                count_key = f"{level}_entropy_count"

                if acc_key in metrics:
                    entropy_table.add_row(
                        level.title(),
                        f"{metrics.get(acc_key, 0):.3f}",
                        str(metrics.get(count_key, 0)),
                    )

            console.print(entropy_table)

        # Export detailed report if requested
        if export_report:
            report = {
                "analysis_timestamp": str(Path(results_file).stat().st_mtime),
                "model": data.get("model"),
                "config": data.get("config"),
                "summary_metrics": {
                    "accuracy": metrics.get("overall_accuracy"),
                    "precision": metrics.get("overall_precision"),
                    "recall": metrics.get("overall_recall"),
                    "f1": metrics.get("overall_f1"),
                },
                "detailed_metrics": detailed,
                "recommendations": [],
            }

            # Add recommendations based on metrics
            if metrics.get("overall_accuracy", 0) < 0.7:
                report["recommendations"].append(
                    "Consider fine-tuning or using a more capable model"
                )
            if metrics.get("recall_hallucinated", 0) < 0.5:
                report["recommendations"].append("Model has difficulty detecting hallucinations")

            export_report.parent.mkdir(parents=True, exist_ok=True)
            with open(export_report, "w") as f:
                json.dump(report, f, indent=2, default=str)

            console.print(f"\n[green]✓[/green] Report exported to {export_report}")

    except FileNotFoundError:
        console.print(f"[red]Error:[/red] Results file not found: {results_file}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)


@app.command()
def info():
    """Show information about FaithBench benchmark."""
    console.print("[bold]FaithBench - Hallucination Detection Benchmark[/bold]\n")

    console.print("[cyan]Description:[/cyan]")
    console.print(
        "FaithBench evaluates hallucination detection in summarization tasks "
        "using a 4-level annotation taxonomy and focusing on challenging samples "
        "where multiple detectors disagree.\n"
    )

    console.print("[cyan]Annotation Levels:[/cyan]")
    levels = [
        ("Consistent", "Summary is fully consistent with source"),
        ("Questionable", "Borderline case, may have minor issues"),
        ("Benign", "Contains minor hallucinations that don't affect meaning"),
        ("Hallucinated", "Contains significant factual errors"),
    ]

    for level, desc in levels:
        console.print(f"  • [bold]{level}:[/bold] {desc}")

    console.print("\n[cyan]Supported Models:[/cyan]")
    for model in sorted(FaithBenchBenchmark.SUPPORTED_MODELS):
        temp_note = (
            " (no temperature)" if model in FaithBenchBenchmark.NO_TEMPERATURE_MODELS else ""
        )
        console.print(f"  • {model}{temp_note}")

    console.print("\n[cyan]Key Metrics:[/cyan]")
    metrics_info = [
        ("Precision/Recall", "Per-label classification performance"),
        ("Entropy Score", "Measure of detector disagreement (0-1)"),
        ("Challenging Accuracy", "Performance on high-entropy samples"),
        ("F1 Score", "Harmonic mean of precision and recall"),
    ]

    for metric, desc in metrics_info:
        console.print(f"  • [bold]{metric}:[/bold] {desc}")

    console.print("\n[cyan]Paper Reference:[/cyan]")
    console.print(
        "FaithBench: A Benchmark for Hallucination Detection in Summarization "
        "with Challenging Samples (2024)"
    )


def main():
    """Main entry point for FaithBench CLI."""
    app()


if __name__ == "__main__":
    main()
