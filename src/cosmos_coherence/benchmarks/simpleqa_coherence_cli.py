"""CLI commands for SimpleQA benchmark with coherence-guided evaluation."""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.console import Console
from rich.progress import (
    Progress,
    BarColumn,
    MofNCompleteColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from cosmos_coherence.benchmarks.implementations.simpleqa_coherence_benchmark import (
    SimpleQACoherenceBenchmark,
)
from cosmos_coherence.benchmarks.implementations.simpleqa_benchmark import SimpleQABenchmark
from cosmos_coherence.benchmarks.implementations.simpleqa_grader import SimpleQAGrader
from cosmos_coherence.llm.config import OpenAIConfig
from cosmos_coherence.llm.openai_client import OpenAIClient

app = typer.Typer(help="SimpleQA coherence-guided evaluation commands")
console = Console()


@app.command()
def run_with_coherence(
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model",
        "-m",
        help="Model to evaluate (e.g., gpt-4o-mini, gpt-4-turbo)",
    ),
    coherence_variant: str = typer.Option(
        "shogenji",
        "--coherence-variant",
        help="Coherence measure: shogenji|fitelson|olsson",
    ),
    num_variations: int = typer.Option(
        5,
        "--num-variations",
        help="Number of question paraphrases to generate",
    ),
    coherence_threshold: float = typer.Option(
        0.5,
        "--coherence-threshold",
        help="Threshold for abstention decision (0.0-1.0)",
    ),
    temperature_range: str = typer.Option(
        "0.7,1.0,1.3",
        "--temperature-range",
        help="Comma-separated temperatures for variation generation",
    ),
    sample_size: Optional[int] = typer.Option(
        None,
        "--sample-size",
        "-n",
        help="Number of questions to evaluate (default: all 4,326)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to save results JSON",
    ),
    checkpoint_interval: int = typer.Option(
        50,
        "--checkpoint-interval",
        help="Save progress every N questions (0 to disable)",
    ),
    resume_from: Optional[Path] = typer.Option(
        None,
        "--resume",
        "-r",
        help="Resume from checkpoint file",
    ),
    workers: int = typer.Option(
        1,
        "--workers",
        help="Number of parallel workers (default: 1)",
    ),
    run_baseline: bool = typer.Option(
        True,
        "--run-baseline/--no-baseline",
        help="Also run baseline (no coherence) for comparison",
    ),
    abstention_response: str = typer.Option(
        "I don't know",
        "--abstention-response",
        help="Text to return when abstaining",
    ),
    grader_model: str = typer.Option(
        "gpt-4o-mini",
        "--grader-model",
        help="Model for AI grading",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output for debugging",
    ),
):
    """Run SimpleQA evaluation with coherence-guided abstention.

    This command evaluates models on the SimpleQA benchmark using question-variation
    coherence to guide answer generation. High coherence questions are answered
    confidently, while low coherence questions trigger abstention ("I don't know").

    Example:
        cosmos-coherence simpleqa run-with-coherence \\
            --model gpt-4o-mini \\
            --coherence-variant shogenji \\
            --sample-size 100 \\
            --output results.json
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

        # Parse temperature range
        try:
            temperature_list = [float(t.strip()) for t in temperature_range.split(",")]
        except ValueError:
            console.print(
                f"[red]Error:[/red] Invalid temperature range: {temperature_range}\n"
                "Expected comma-separated floats (e.g., '0.7,1.0,1.3')"
            )
            raise typer.Exit(code=1)

        # Generate default output path if not provided
        if output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = Path(f"simpleqa_coherence_results_{timestamp}.json")

        # Display configuration
        console.print("\n[bold]SimpleQA Coherence-Guided Evaluation[/bold]")
        console.print("[dim]Using Answer-Space Coherence: Coherence computed on answer sketches[/dim]")
        console.print(f"Model: [cyan]{model}[/cyan]")
        console.print(f"Coherence variant: [cyan]{coherence_variant}[/cyan]")
        console.print(f"Coherence threshold: [cyan]{coherence_threshold}[/cyan]")
        console.print(f"Number of variations: [cyan]{num_variations}[/cyan]")
        console.print(f"Temperature range: [cyan]{temperature_list}[/cyan]")
        if sample_size:
            console.print(f"Sample size: [cyan]{sample_size}[/cyan]")
        else:
            console.print("Sample size: [cyan]All 4,326 questions[/cyan]")
        console.print(f"Output: [cyan]{output}[/cyan]")
        console.print()

        # Run evaluation
        results = asyncio.run(
            _run_coherence_evaluation(
                model=model,
                api_key=api_key,
                coherence_variant=coherence_variant,
                num_variations=num_variations,
                coherence_threshold=coherence_threshold,
                temperature_list=temperature_list,
                sample_size=sample_size,
                checkpoint_interval=checkpoint_interval,
                resume_from=resume_from,
                workers=workers,
                abstention_response=abstention_response,
                grader_model=grader_model,
                verbose=verbose,
            )
        )

        # Run baseline if requested
        baseline_results = None
        if run_baseline:
            console.print("\n[bold]Running Baseline (No Coherence)[/bold]")
            baseline_results = asyncio.run(
                _run_baseline_evaluation(
                    model=model,
                    api_key=api_key,
                    sample_size=sample_size,
                    grader_model=grader_model,
                )
            )

        # Aggregate metrics
        metrics = _calculate_metrics(results)
        baseline_metrics = _calculate_metrics(baseline_results) if baseline_results else None

        # Display results
        _display_results(metrics, baseline_metrics)

        # Save results
        output_data = {
            "config": {
                "model": model,
                "coherence_variant": coherence_variant,
                "num_variations": num_variations,
                "coherence_threshold": coherence_threshold,
                "temperature_range": temperature_list,
                "sample_size": sample_size or 4326,
                "timestamp": datetime.now().isoformat(),
            },
            "metrics": metrics,
            "baseline_metrics": baseline_metrics,
            "per_question_results": results,
        }

        _save_results(output_data, output)
        console.print(f"\n[green]Results saved to:[/green] {output}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        raise typer.Exit(code=1)


async def _run_coherence_evaluation(
    model: str,
    api_key: str,
    coherence_variant: str,
    num_variations: int,
    coherence_threshold: float,
    temperature_list: List[float],
    sample_size: Optional[int],
    checkpoint_interval: int,
    resume_from: Optional[Path],
    workers: int,
    abstention_response: str,
    grader_model: str,
    verbose: bool,
) -> List[Dict]:
    """Run coherence-guided evaluation."""
    # Initialize OpenAI client
    openai_config = OpenAIConfig(
        api_key=api_key,
        organization_id=None,
        base_url="https://api.openai.com/v1",
        default_model=model,
        timeout=30.0,
        max_retries=3,
    )

    # Enable caching for efficiency
    cache_dir = Path.home() / ".cache" / "cosmos_coherence" / "simpleqa_coherence"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{model.replace('/', '_')}_coherence_cache.json"

    client = OpenAIClient(
        openai_config,
        enable_cache=True,
        cache_file=cache_file,
    )

    # Initialize benchmark
    benchmark = SimpleQACoherenceBenchmark(
        client=client,
        coherence_variant=coherence_variant,
        coherence_threshold=coherence_threshold,
        num_variations=num_variations,
        temperature_range=temperature_list,
        abstention_response=abstention_response,
        model=model,
        grader_model=grader_model,
        verbose=verbose,
        hf_dataset_name="simpleqa",
        sample_size=sample_size,
    )

    # Load dataset
    dataset = await benchmark.load_dataset()
    total_questions = len(dataset)

    console.print(f"Loaded {total_questions} questions from SimpleQA dataset")

    # Resume from checkpoint if provided
    start_idx = 0
    results = []

    if resume_from and resume_from.exists():
        console.print(f"[yellow]Resuming from checkpoint: {resume_from}[/yellow]")
        results, start_idx = benchmark.load_checkpoint(resume_from)
        start_idx += 1
        console.print(f"[green]Resuming from question {start_idx + 1}[/green]")

    # Evaluate with progress bar
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("•"),
        TextColumn("[green]{task.fields[correct]}/{task.total} correct"),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Evaluating questions",
            total=total_questions,
            correct=sum(1 for r in results if r.get("is_correct")),
            completed=start_idx,
        )

        # Advance to start position if resuming
        if start_idx > 0:
            progress.update(task, completed=start_idx)

        for idx, item in enumerate(dataset[start_idx:], start=start_idx):
            try:
                result = await benchmark.evaluate_with_coherence(item)
                results.append(result)

                # Update progress
                correct_count = sum(1 for r in results if r.get("is_correct"))
                progress.update(
                    task,
                    advance=1,
                    correct=correct_count,
                    description=f"Q{idx+1}: {item.question[:40]}..." if len(item.question) > 40 else f"Q{idx+1}: {item.question}",
                )

                # Save checkpoint periodically
                if checkpoint_interval > 0 and (idx + 1) % checkpoint_interval == 0:
                    checkpoint_path = Path(f"simpleqa_coherence_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                    benchmark.save_checkpoint(
                        results=results,
                        checkpoint_path=checkpoint_path,
                        last_index=idx,
                        config={
                            "model": model,
                            "coherence_variant": coherence_variant,
                            "coherence_threshold": coherence_threshold,
                            "num_variations": num_variations,
                        }
                    )
                    console.print(f"[dim]Checkpoint saved at question {idx + 1}[/dim]")

            except Exception as e:
                console.print(f"[red]Error on question {idx+1}: {str(e)}[/red]")
                # Add error result
                results.append({
                    "question": item.question,
                    "ground_truth": item.best_answer,
                    "coherence_score": 0.0,
                    "final_answer": f"ERROR: {str(e)}",
                    "is_correct": False,
                })

    # Save cache
    client.save_cache()
    stats = client.get_cache_statistics()
    if stats.total_requests > 0:
        console.print(
            f"[dim]Cache: {stats.cache_hits} hits, {stats.cache_misses} misses ({stats.hit_rate*100:.1f}% hit rate)[/dim]"
        )

    return results


async def _run_baseline_evaluation(
    model: str,
    api_key: str,
    sample_size: Optional[int],
    grader_model: str,
) -> List[Dict]:
    """Run baseline evaluation (no coherence)."""
    # Initialize client
    openai_config = OpenAIConfig(
        api_key=api_key,
        organization_id=None,
        base_url="https://api.openai.com/v1",
        default_model=model,
        timeout=30.0,
        max_retries=3,
    )

    cache_dir = Path.home() / ".cache" / "cosmos_coherence" / "simpleqa"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{model.replace('/', '_')}_baseline_cache.json"

    client = OpenAIClient(
        openai_config,
        enable_cache=True,
        cache_file=cache_file,
    )

    # Initialize baseline benchmark
    benchmark = SimpleQABenchmark(
        client=client,
        use_ai_grading=True,
        hf_dataset_name="simpleqa",
        sample_size=sample_size,
    )

    # Ensure grader is initialized
    if not benchmark._grader:
        benchmark._grader = SimpleQAGrader(client, grader_model=grader_model)

    # Load dataset
    dataset = await benchmark.load_dataset()

    console.print(f"Evaluating {len(dataset)} questions (baseline)")

    results = []

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Baseline evaluation", total=len(dataset))

        for item in dataset:
            try:
                # Get model response
                prompt = benchmark.get_prompt(item)
                response = await client.generate_response(
                    prompt=prompt,
                    model=model,
                    temperature=0.3,
                    max_tokens=50,
                )

                # Grade response
                grade, metadata = await benchmark._grader.grade_response(
                    question=item.question,
                    expert_answer=item.best_answer,
                    submission=response.content,
                )

                results.append({
                    "question": item.question,
                    "ground_truth": item.best_answer,
                    "final_answer": response.content,
                    "grade": grade,
                    "is_correct": grade == "CORRECT",
                })

                progress.update(task, advance=1)

            except Exception as e:
                console.print(f"[red]Baseline error: {str(e)}[/red]")
                results.append({
                    "question": item.question,
                    "is_correct": False,
                })

    client.save_cache()

    return results


def _calculate_metrics(results: Optional[List[Dict]]) -> Dict:
    """Calculate evaluation metrics."""
    if not results:
        return {}

    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct"))
    incorrect = sum(1 for r in results if r.get("grade") == "INCORRECT")
    not_attempted = sum(1 for r in results if r.get("grade") == "NOT_ATTEMPTED")

    # Calculate coherence statistics if available
    coherence_scores = [r.get("coherence_score") for r in results if "coherence_score" in r]
    coherence_stats = {}
    if coherence_scores:
        import numpy as np
        coherence_stats = {
            "mean_coherence": float(np.mean(coherence_scores)),
            "median_coherence": float(np.median(coherence_scores)),
            "std_coherence": float(np.std(coherence_scores)),
        }

    return {
        "total_questions": total,
        "correct_count": correct,
        "incorrect_count": incorrect,
        "not_attempted_count": not_attempted,
        "overall_correct": correct / total if total > 0 else 0.0,
        "overall_incorrect": incorrect / total if total > 0 else 0.0,
        "overall_not_attempted": not_attempted / total if total > 0 else 0.0,
        "correct_given_attempted": correct / (correct + incorrect) if (correct + incorrect) > 0 else 0.0,
        **coherence_stats,
    }


def _display_results(metrics: Dict, baseline_metrics: Optional[Dict] = None) -> None:
    """Display evaluation results."""
    table = Table(title="Coherence-Guided Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Coherence", style="green")

    if baseline_metrics:
        table.add_column("Baseline", style="blue")
        table.add_column("Delta", style="yellow")

    # Core metrics
    table.add_row("Total Questions", str(metrics.get("total_questions", 0)))
    table.add_row("Correct", str(metrics.get("correct_count", 0)))
    table.add_row("Incorrect", str(metrics.get("incorrect_count", 0)))
    table.add_row("Not Attempted", str(metrics.get("not_attempted_count", 0)))

    table.add_row("", "")  # Spacer

    # Percentage metrics
    if baseline_metrics:
        overall_correct_coh = metrics.get("overall_correct", 0.0)
        overall_correct_base = baseline_metrics.get("overall_correct", 0.0)
        delta_correct = overall_correct_coh - overall_correct_base

        table.add_row(
            "Overall Correct",
            f"{overall_correct_coh:.1%}",
            f"{overall_correct_base:.1%}",
            f"{delta_correct:+.1%}"
        )

        overall_incorrect_coh = metrics.get("overall_incorrect", 0.0)
        overall_incorrect_base = baseline_metrics.get("overall_incorrect", 0.0)
        delta_incorrect = overall_incorrect_coh - overall_incorrect_base

        table.add_row(
            "Overall Incorrect",
            f"{overall_incorrect_coh:.1%}",
            f"{overall_incorrect_base:.1%}",
            f"{delta_incorrect:+.1%}"
        )

        precision_coh = metrics.get("correct_given_attempted", 0.0)
        precision_base = baseline_metrics.get("correct_given_attempted", 0.0)
        delta_precision = precision_coh - precision_base

        table.add_row(
            "Precision (Correct Given Attempted)",
            f"{precision_coh:.1%}",
            f"{precision_base:.1%}",
            f"{delta_precision:+.1%}"
        )
    else:
        table.add_row("Overall Correct", f"{metrics.get('overall_correct', 0.0):.1%}")
        table.add_row("Overall Incorrect", f"{metrics.get('overall_incorrect', 0.0):.1%}")
        table.add_row("Overall Not Attempted", f"{metrics.get('overall_not_attempted', 0.0):.1%}")
        table.add_row("Precision (Correct Given Attempted)", f"{metrics.get('correct_given_attempted', 0.0):.1%}")

    # Coherence statistics
    if "mean_coherence" in metrics:
        table.add_row("", "")  # Spacer
        table.add_row("Mean Coherence", f"{metrics.get('mean_coherence', 0.0):.4f}")
        table.add_row("Median Coherence", f"{metrics.get('median_coherence', 0.0):.4f}")
        table.add_row("Std Dev Coherence", f"{metrics.get('std_coherence', 0.0):.4f}")

    console.print(table)


def _save_results(data: Dict, output_path: Path) -> None:
    """Save results to JSON file."""
    import numpy as np

    def convert_to_serializable(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_data = convert_to_serializable(data)
    with open(output_path, 'w') as f:
        json.dump(serializable_data, f, indent=2)


if __name__ == "__main__":
    app()
