# Spec Requirements Document

> Spec: SimpleQA Benchmark Implementation
> Created: 2025-09-11

## Overview

Implement the SimpleQA benchmark for evaluating LLM factual accuracy using short-form question answering. This benchmark will integrate with the existing Cosmos Coherence framework to enable standardized evaluation of hallucination patterns across OpenAI models.

## User Stories

### Research Scientist Story

As a research scientist, I want to evaluate LLM performance on factual questions, so that I can quantify hallucination rates and compare model capabilities.

The scientist loads the SimpleQA dataset through the CLI, configures the evaluation to use GPT-4 or GPT-3.5, and runs the benchmark. The system processes all 4,326 questions, evaluates responses using exact match and F1 scoring, and generates a comprehensive report comparing results against published baselines (GPT-4: 82%, GPT-3.5: 68%). Results are exported in JSONL format for further analysis.

### Benchmark Developer Story

As a benchmark developer, I want to integrate SimpleQA with the existing harness, so that it follows the same patterns as other benchmarks in the framework.

The developer uses the existing HuggingFace loader to fetch the SimpleQA dataset, implements the evaluation logic following the established BaseExperiment pattern, and ensures compatibility with the CLI interface. The implementation supports sampling for quick tests and full dataset evaluation for complete benchmarks.

## Spec Scope

1. **Dataset Integration** - Load SimpleQA dataset from HuggingFace (openai/simple-evals) with automatic caching
2. **Evaluation Implementation** - Implement exact match and F1 scoring as specified in the OpenAI paper
3. **CLI Command** - Add simpleqa subcommand to run benchmark with configurable parameters
4. **Baseline Comparison** - Compare results against published GPT-4 (82%) and GPT-3.5 (68%) baselines
5. **Results Export** - Export results in JSONL format consistent with existing framework patterns

## Out of Scope

- Temperature variation testing (Phase 2)
- Coherence measure calculations (Phase 2)
- Non-OpenAI model support (Phase 2)
- Batch API functionality (being removed)
- Web dashboard visualization (Phase 3)

## Expected Deliverable

1. Functional SimpleQA benchmark that reproduces paper results within 5% margin
2. CLI interface supporting: `python -m cosmos_coherence.benchmarks.simpleqa_cli run --model gpt-4 --sample-size 100`
3. JSONL export with per-question results and aggregate metrics matching framework standards
