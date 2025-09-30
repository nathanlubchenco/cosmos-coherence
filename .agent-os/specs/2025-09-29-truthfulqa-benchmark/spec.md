# Spec Requirements Document

> Spec: TruthfulQA Benchmark Implementation
> Created: 2025-09-29

## Overview

Implement the TruthfulQA benchmark using multiple-choice evaluation tasks (MC1 and MC2) to measure model truthfulness across 817 questions spanning 38 categories. This implementation will use log-probability evaluation requiring no fine-tuned judge models, following the standard approach used by major LLM benchmarks.

## User Stories

### Research Scientist Evaluating Model Truthfulness

As a research scientist, I want to evaluate language models on their ability to provide truthful answers across diverse question categories, so that I can measure susceptibility to common misconceptions and false beliefs.

The researcher runs the TruthfulQA benchmark on GPT-4, which evaluates the model across all 817 questions using both MC1 (single correct answer) and MC2 (multiple true/false answers) formats. Results are broken down by category (e.g., misconceptions, conspiracies, stereotypes) to identify specific areas where the model struggles with truthfulness. The benchmark completes in a reasonable time using cached responses for repeated evaluations.

### Benchmark Developer Comparing Multiple Models

As a benchmark developer, I want to run TruthfulQA evaluations across multiple OpenAI models (GPT-3.5, GPT-4) with consistent methodology, so that I can compare their truthfulness scores and identify improvements.

The developer uses the CLI to run evaluations on both gpt-3.5-turbo and gpt-4, with results stored separately. The system reports MC1 accuracy (single correct answer) and MC2 normalized scores (probability distribution), matching published baseline scores. Category-level breakdowns help identify which question types each model handles better, enabling targeted analysis of model improvements.

### Developer Testing Implementation Changes

As a developer, I want to validate my TruthfulQA implementation against published baselines using a small sample, so that I can quickly verify correctness before running full evaluations.

The developer runs the benchmark with --sample-size 50 to test a subset of questions, comparing results against expected patterns. Once validated, they run the full 817-question dataset with caching enabled, ensuring subsequent runs reuse responses to avoid redundant API calls. The implementation exactly matches the evaluation methodology used by HuggingFace Open LLM Leaderboard and EleutherAI's lm-evaluation-harness.

## Spec Scope

1. **MC1 Evaluation** - Implement single correct answer multiple-choice evaluation using log-probability comparison across 4-5 answer choices per question
2. **MC2 Evaluation** - Implement multiple true/false answer evaluation with normalized probability scoring across correct and incorrect answer sets
3. **Dataset Integration** - Load TruthfulQA dataset from HuggingFace (817 questions, 38 categories) with full support for mc1_targets and mc2_targets fields
4. **Category Reporting** - Generate accuracy metrics broken down by question category (38 categories) for detailed analysis
5. **CLI Interface** - Provide command-line interface with model selection, sample size control, temperature settings, and caching support
6. **Response Caching** - Integrate with existing OpenAI client caching system to avoid redundant API calls during iterative development

## Out of Scope

- Human evaluation (requires manual annotation)
- Generation task evaluation (requires judge models - can be added in Phase 2)
- Fine-tuned model evaluation (Phase 1 focuses on OpenAI models only)
- Real-time evaluation dashboard (Phase 3 feature)
- Cross-provider model comparison (Phase 2 feature - Anthropic, Google, etc.)
- Custom judge model training or integration with open-source judges (future enhancement)

## Expected Deliverable

1. TruthfulQA benchmark runs successfully on gpt-3.5-turbo and gpt-4 models, producing MC1 accuracy and MC2 normalized scores that align within 5% of published baselines (HuggingFace Open LLM Leaderboard: GPT-3.5 ~47%, GPT-4 ~59%)
2. CLI command `poetry run python -m cosmos_coherence.benchmarks.truthfulqa_cli run --model gpt-4 --sample-size 50` executes successfully and displays results with category breakdowns
3. Results JSON file contains per-question results with category labels, model predictions, correct answers, and computed metrics for both MC1 and MC2 evaluation methods
