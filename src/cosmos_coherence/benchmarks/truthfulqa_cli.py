"""CLI commands for TruthfulQA benchmark."""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from cosmos_coherence.benchmarks.implementations.truthfulqa_benchmark import (
    TruthfulQABenchmark,
)
from cosmos_coherence.benchmarks.models.datasets import TruthfulQAItem
from cosmos_coherence.harness.huggingface_loader import HuggingFaceDatasetLoader
from cosmos_coherence.llm.config import OpenAIConfig
from cosmos_coherence.llm.openai_client import OpenAIClient

app = typer.Typer(help="TruthfulQA benchmark commands for evaluating model truthfulness")
console = Console()


async def run_evaluation(
    model: str,
    sample_size: Optional[int],
    category: Optional[str],
    temperature: float,
    use_cache: bool,
    show_progress: bool,
    verbose: bool,
) -> Dict:
    """Run TruthfulQA benchmark evaluation.

    Args:
        model: Model name to evaluate
        sample_size: Number of samples to evaluate
        category: Specific category to evaluate
        temperature: Temperature for generation
        use_cache: Whether to use caching
        show_progress: Whether to show progress
        verbose: Whether to show detailed output

    Returns:
        Dictionary of results
    """
    # Set up cache directory
    cache_dir = Path.home() / ".cache" / "cosmos_coherence" / "truthfulqa"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{model.replace('/', '_')}_cache.json"

    # Initialize OpenAI client
    config = OpenAIConfig(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        organization_id=None,
        base_url="https://api.openai.com/v1",
        default_model=model,
        timeout=60.0,  # Longer timeout for logprobs requests
        max_retries=3,
    )

    client = OpenAIClient(
        openai_config=config,
        enable_cache=use_cache,
        cache_file=str(cache_file) if use_cache else None,
    )

    # Initialize benchmark
    benchmark = TruthfulQABenchmark(client=client, hf_dataset_name="truthfulqa")

    # Load dataset
    if show_progress:
        console.print("[cyan]Loading TruthfulQA dataset...[/cyan]")

    # For TruthfulQA, we need to load both configs and merge them
    from datasets import load_dataset as hf_load_dataset

    # Clear cache to avoid metadata corruption issues
    truthfulqa_cache = (
        Path.home() / ".cache" / "huggingface" / "datasets" / "truthfulqa___truthful_qa"
    )

    if truthfulqa_cache.exists():
        if verbose:
            console.print("[yellow]Clearing dataset cache...[/yellow]")
        import shutil

        shutil.rmtree(truthfulqa_cache, ignore_errors=True)

    if verbose:
        console.print("[cyan]Loading dataset from HuggingFace...[/cyan]")

    try:
        # Load generation config (has category, best_answer, correct/incorrect answers)
        generation_ds = hf_load_dataset(
            "truthfulqa/truthful_qa",
            "generation",
            split="validation",
            download_mode="force_redownload",
        )

        # Load multiple_choice config (has mc1_targets and mc2_targets)
        mc_ds = hf_load_dataset(
            "truthfulqa/truthful_qa",
            "multiple_choice",
            split="validation",
            download_mode="force_redownload",
        )

        # Merge by question - create lookup dict from MC config
        mc_by_question = {item["question"]: item for item in mc_ds}

        # Merge generation + MC data
        dataset_raw = []
        for gen_item in generation_ds:
            question = gen_item["question"]
            mc_item = mc_by_question.get(question)

            if mc_item:
                # Merge both configs
                merged = dict(gen_item)
                merged["mc1_targets"] = mc_item.get("mc1_targets")
                merged["mc2_targets"] = mc_item.get("mc2_targets")
                dataset_raw.append(merged)
            else:
                if verbose:
                    console.print(f"[yellow]Warning: No MC data for: {question[:50]}[/yellow]")

        if verbose:
            console.print(f"[green]Loaded {len(dataset_raw)} items with categories[/green]")

    except Exception as e:
        console.print(f"[red]Error loading dataset: {e}[/red]")
        if verbose:
            import traceback

            console.print(traceback.format_exc())
        raise

    # Convert to TruthfulQAItem objects
    dataset: List[TruthfulQAItem] = []
    loader = HuggingFaceDatasetLoader()

    for item in dataset_raw:
        try:
            # Convert the Arrow table row to a dict (streaming already returns dicts)
            item_dict = dict(item)
            converted = loader._convert_truthfulqa_item(item_dict)
            if category and converted.category.value.lower() != category.lower():
                continue
            dataset.append(converted)
        except Exception as e:
            if verbose:
                console.print(f"[yellow]Warning: Skipping invalid item: {e}[/yellow]")

    # Apply sample size
    if sample_size:
        dataset = dataset[:sample_size]

    if verbose:
        console.print(f"[green]Loaded {len(dataset)} items[/green]")

    if not dataset:
        console.print("[red]No items to evaluate![/red]")
        return {"error": "No items found"}

    # Run evaluation
    mc1_results = []
    mc2_results = []
    per_question_results = []

    total = len(dataset)

    if show_progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"[cyan]Evaluating {total} questions...", total=total)

            for i, question_item in enumerate(dataset):
                try:
                    # MC1 Evaluation (if available)
                    mc1_result = None
                    if question_item.mc1_targets:
                        mc1_logprobs = []
                        for choice_idx, choice in enumerate(question_item.mc1_targets["choices"]):
                            prompt = benchmark.format_mc_prompt(question_item.question, choice)

                            # Retry up to 3 times if logprobs are missing
                            max_retries = 3
                            for retry in range(max_retries):
                                try:
                                    response = await client.generate_response(
                                        prompt,
                                        temperature=temperature,
                                        max_tokens=1,
                                        logprobs=True,
                                    )
                                    if response.raw_response is None:
                                        raise ValueError("API returned None response")
                                    logprob = benchmark.extract_logprob(response.raw_response)
                                    mc1_logprobs.append(logprob)
                                    break
                                except ValueError as e:
                                    if retry < max_retries - 1:
                                        # Wait longer between retries
                                        await asyncio.sleep(1.0 * (retry + 1))
                                    else:
                                        raise ValueError(
                                            f"Failed to get logprobs after {max_retries} "
                                            f"attempts: {e}"
                                        )

                            # Small delay to avoid rate limiting
                            await asyncio.sleep(0.1)

                        mc1_result = benchmark.evaluate_mc1(question_item, mc1_logprobs)
                        mc1_results.append(mc1_result)

                    # MC2 Evaluation (if available)
                    mc2_result = None
                    if question_item.mc2_targets:
                        mc2_logprobs = []
                        for choice_idx, choice in enumerate(question_item.mc2_targets["choices"]):
                            prompt = benchmark.format_mc_prompt(question_item.question, choice)

                            # Retry up to 3 times if logprobs are missing
                            max_retries = 3
                            for retry in range(max_retries):
                                try:
                                    response = await client.generate_response(
                                        prompt,
                                        temperature=temperature,
                                        max_tokens=1,
                                        logprobs=True,
                                    )
                                    if response.raw_response is None:
                                        raise ValueError("API returned None response")
                                    logprob = benchmark.extract_logprob(response.raw_response)
                                    mc2_logprobs.append(logprob)
                                    break
                                except ValueError as e:
                                    if retry < max_retries - 1:
                                        # Wait longer between retries
                                        await asyncio.sleep(1.0 * (retry + 1))
                                    else:
                                        raise ValueError(
                                            f"Failed to get logprobs after {max_retries} "
                                            f"attempts: {e}"
                                        )

                            # Small delay to avoid rate limiting
                            await asyncio.sleep(0.1)

                        mc2_result = benchmark.evaluate_mc2(question_item, mc2_logprobs)
                        mc2_results.append(mc2_result)

                    # Store per-question result
                    per_question_results.append(
                        {
                            "question": question_item.question,
                            "category": question_item.category.value,
                            "mc1_correct": mc1_result["correct"] if mc1_result else None,
                            "mc1_predicted": (
                                mc1_result["predicted_choice"] if mc1_result else None
                            ),
                            "mc1_correct_choice": (
                                mc1_result["correct_choice"] if mc1_result else None
                            ),
                            "mc2_score": mc2_result["mc2_score"] if mc2_result else None,
                            "mc2_correct_probs_sum": (
                                mc2_result["correct_probs_sum"] if mc2_result else None
                            ),
                            "mc2_incorrect_probs_sum": (
                                mc2_result["incorrect_probs_sum"] if mc2_result else None
                            ),
                        }
                    )

                    # Show progress every 5 questions
                    if (i + 1) % 5 == 0:
                        mc1_acc = (
                            sum(1 for r in mc1_results if r.get("correct", False))
                            / len(mc1_results)
                            if mc1_results
                            else 0
                        )
                        mc2_avg = (
                            sum(r["mc2_score"] for r in mc2_results) / len(mc2_results)
                            if mc2_results
                            else 0
                        )
                        msg = (
                            f"[green]Processed {i + 1}/{total} - "
                            f"MC1: {mc1_acc:.1%}, MC2: {mc2_avg:.1%}[/green]"
                        )
                        console.print(msg)

                except Exception as e:
                    console.print(f"[red]Error on question {i}: {e}[/red]")
                    if verbose:
                        import traceback

                        console.print(traceback.format_exc())

                progress.update(task, advance=1)
    else:
        # Run without progress bar
        for i, question_item in enumerate(dataset):
            # Same evaluation logic as above
            try:
                if question_item.mc1_targets:
                    mc1_logprobs = []
                    for choice in question_item.mc1_targets["choices"]:
                        prompt = benchmark.format_mc_prompt(question_item.question, choice)

                        # Retry up to 3 times if logprobs are missing
                        max_retries = 3
                        for retry in range(max_retries):
                            try:
                                response = await client.generate_response(
                                    prompt, temperature=temperature, max_tokens=1, logprobs=True
                                )
                                if response.raw_response is None:
                                    raise ValueError("API returned None response")
                                logprob = benchmark.extract_logprob(response.raw_response)
                                mc1_logprobs.append(logprob)
                                break
                            except ValueError as e:
                                if retry < max_retries - 1:
                                    await asyncio.sleep(1.0 * (retry + 1))
                                else:
                                    raise ValueError(
                                        f"Failed to get logprobs after {max_retries} attempts: {e}"
                                    )

                        # Small delay to avoid rate limiting
                        await asyncio.sleep(0.1)
                    mc1_result = benchmark.evaluate_mc1(question_item, mc1_logprobs)
                    mc1_results.append(mc1_result)

                if question_item.mc2_targets:
                    mc2_logprobs = []
                    for choice in question_item.mc2_targets["choices"]:
                        prompt = benchmark.format_mc_prompt(question_item.question, choice)

                        # Retry up to 3 times if logprobs are missing
                        max_retries = 3
                        for retry in range(max_retries):
                            try:
                                response = await client.generate_response(
                                    prompt, temperature=temperature, max_tokens=1, logprobs=True
                                )
                                if response.raw_response is None:
                                    raise ValueError("API returned None response")
                                logprob = benchmark.extract_logprob(response.raw_response)
                                mc2_logprobs.append(logprob)
                                break
                            except ValueError as e:
                                if retry < max_retries - 1:
                                    await asyncio.sleep(1.0 * (retry + 1))
                                else:
                                    raise ValueError(
                                        f"Failed to get logprobs after {max_retries} attempts: {e}"
                                    )

                        # Small delay to avoid rate limiting
                        await asyncio.sleep(0.1)
                    mc2_result = benchmark.evaluate_mc2(question_item, mc2_logprobs)
                    mc2_results.append(mc2_result)
            except Exception as e:
                if verbose:
                    console.print(f"[red]Error on question {i}: {e}[/red]")

    # Calculate aggregate metrics
    metrics = benchmark.calculate_metrics_by_category(mc1_results, mc2_results, dataset)

    # Save cache to disk and show statistics
    if use_cache:
        try:
            client.save_cache()
            stats = client.get_cache_statistics()
            if show_progress or verbose:
                console.print(
                    f"[green]Cache: {stats.cache_hits} hits, "
                    f"{stats.cache_misses} misses "
                    f"({stats.hit_rate:.1%} hit rate)[/green]"
                )
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to save cache: {e}[/yellow]")

    return {
        "model": model,
        "temperature": temperature,
        "total_questions": len(dataset),
        "category_filter": category,
        "metrics": metrics,
        "per_question_results": per_question_results,
    }


@app.command()
def run(
    model: str = typer.Option(
        "gpt-4",
        "--model",
        "-m",
        help="Model to evaluate (e.g., gpt-4, gpt-4o, gpt-3.5-turbo)",
    ),
    sample_size: Optional[int] = typer.Option(
        None,
        "--sample-size",
        "-n",
        help="Number of questions to evaluate (default: all 817)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to save results JSON",
    ),
    category: Optional[str] = typer.Option(
        None,
        "--category",
        "-c",
        help="Filter by specific category (e.g., Science, Health, Misconceptions)",
    ),
    temperature: float = typer.Option(
        0.0,
        "--temperature",
        "-t",
        help="Temperature for generation (default: 0.0 for deterministic evaluation)",
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
        help="Enable/disable response caching (default: enabled)",
    ),
):
    """Run TruthfulQA benchmark evaluation.

    Evaluates model truthfulness using MC1 (single correct answer) and MC2
    (multiple true/false answers) formats. Results include overall accuracy
    and per-category breakdowns across 38 question categories.

    ⚠️  IMPORTANT: This MC evaluation is fundamentally limited when using
    OpenAI models due to Chat API constraints. Results will be significantly
    below published baselines. See docs/limitations/truthfulqa.md for details.

    Example:
        poetry run python -m cosmos_coherence.benchmarks.truthfulqa_cli run \\
            --model gpt-4 --sample-size 50 --output results.json
    """
    console.print("[bold cyan]TruthfulQA Benchmark Evaluation[/bold cyan]")
    console.print("[yellow]⚠️  WARNING: MC evaluation is limited for OpenAI models.[/yellow]")
    console.print(
        "[yellow]   Scores will be below baselines. See docs/limitations/truthfulqa.md[/yellow]"
    )
    console.print()
    console.print(f"Model: {model}")
    console.print(f"Temperature: {temperature}")
    if category:
        console.print(f"Category filter: {category}")
    if sample_size:
        console.print(f"Sample size: {sample_size}")
    console.print()

    # Run evaluation
    results = asyncio.run(
        run_evaluation(
            model=model,
            sample_size=sample_size,
            category=category,
            temperature=temperature,
            use_cache=use_cache,
            show_progress=show_progress,
            verbose=verbose,
        )
    )

    if "error" in results:
        console.print(f"[red]Error: {results['error']}[/red]")
        raise typer.Exit(1)

    # Display results
    console.print("\n[bold green]Results[/bold green]")

    # Overall metrics table
    overall = results["metrics"]["overall"]
    table = Table(title="Overall Performance")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green")

    table.add_row("MC1 Accuracy", f"{overall['mc1_accuracy']:.1%}")
    table.add_row("MC2 Score", f"{overall['mc2_score']:.1%}")
    table.add_row("Total Questions", str(overall["total_questions"]))

    console.print(table)
    console.print()

    # Category breakdown
    by_category = results["metrics"]["by_category"]
    if by_category:
        # Sort by MC2 score
        sorted_categories = sorted(
            by_category.items(), key=lambda x: x[1]["mc2_score"], reverse=True
        )

        # Top 5 categories
        console.print("[bold]Top 5 Categories (by MC2 score):[/bold]")
        top_table = Table()
        top_table.add_column("Category", style="cyan")
        top_table.add_column("MC1", style="green")
        top_table.add_column("MC2", style="green")
        top_table.add_column("Count", style="yellow")

        for cat, metrics in sorted_categories[:5]:
            top_table.add_row(
                cat,
                f"{metrics['mc1_accuracy']:.1%}",
                f"{metrics['mc2_score']:.1%}",
                str(metrics["count"]),
            )

        console.print(top_table)
        console.print()

        # Bottom 5 categories
        console.print("[bold]Bottom 5 Categories (by MC2 score):[/bold]")
        bottom_table = Table()
        bottom_table.add_column("Category", style="cyan")
        bottom_table.add_column("MC1", style="red")
        bottom_table.add_column("MC2", style="red")
        bottom_table.add_column("Count", style="yellow")

        for cat, metrics in sorted_categories[-5:]:
            bottom_table.add_row(
                cat,
                f"{metrics['mc1_accuracy']:.1%}",
                f"{metrics['mc2_score']:.1%}",
                str(metrics["count"]),
            )

        console.print(bottom_table)

    # Baseline comparison
    console.print("\n[bold]Baseline Comparison:[/bold]")
    baseline_table = Table()
    baseline_table.add_column("Model", style="cyan")
    baseline_table.add_column("MC2 Score", style="green")
    baseline_table.add_column("Difference", style="yellow")

    # Get baselines from benchmark class
    temp_benchmark = TruthfulQABenchmark()
    baselines = temp_benchmark.get_baseline_metrics()
    if "gpt-3.5-turbo_mc2" in baselines:
        diff_35 = overall["mc2_score"] - baselines["gpt-3.5-turbo_mc2"]
        baseline_table.add_row(
            "GPT-3.5-turbo (baseline)",
            f"{baselines['gpt-3.5-turbo_mc2']:.1%}",
            f"{diff_35:+.1%}",
        )
    if "gpt-4_mc2" in baselines:
        diff_4 = overall["mc2_score"] - baselines["gpt-4_mc2"]
        baseline_table.add_row(
            "GPT-4 (baseline)", f"{baselines['gpt-4_mc2']:.1%}", f"{diff_4:+.1%}"
        )
    if "gpt-4-turbo_mc2" in baselines:
        diff_4t = overall["mc2_score"] - baselines["gpt-4-turbo_mc2"]
        baseline_table.add_row(
            "GPT-4-turbo (baseline)",
            f"{baselines['gpt-4-turbo_mc2']:.1%}",
            f"{diff_4t:+.1%}",
        )
    baseline_table.add_row(f"{model} (your run)", f"{overall['mc2_score']:.1%}", "—")

    console.print(baseline_table)

    # Save results if output specified
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[green]Results saved to {output_path}[/green]")


if __name__ == "__main__":
    app()
