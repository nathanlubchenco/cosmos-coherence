# Spec Requirements Document

> Spec: SelfCheckGPT Implementation
> Created: 2025-09-29

## Overview

Implement SelfCheckGPT benchmark for consistency-based hallucination detection using temperature variation sampling. This benchmark generates multiple responses at different temperatures and evaluates consistency to detect hallucinations, directly supporting the project's coherence research goals by enabling comparison between SelfCheckGPT's consistency measures and philosophical coherence measures (Shogenji, Fitelson, Olsson).

**Research Foundation**: See `research-references.md` for complete paper details, methodology, and baseline results.

**Paper**: Manakul et al. (2023) - SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models (EMNLP 2023, arXiv:2303.08896)

**API Compatibility**: ✅ Confirmed compatible with OpenAI Chat Completions API (paper uses ChatGPT gpt-3.5-turbo)

## User Stories

### Researcher Evaluating Model Hallucination via Consistency

As a researcher, I want to evaluate an LLM's hallucination tendency using SelfCheckGPT, so that I can detect hallucinations through inconsistency patterns across multiple temperature-variant samples.

The researcher runs the SelfCheckGPT benchmark which generates one response at temperature 0.0 (baseline) and five responses at temperature 1.0 (samples). The system then evaluates consistency between the baseline and samples using Natural Language Inference (NLI). High inconsistency indicates potential hallucination. Results show per-sentence hallucination scores and aggregate metrics comparable to published baselines.

### Researcher Comparing Consistency vs Coherence Measures

As a researcher, I want to compare SelfCheckGPT consistency scores with philosophical coherence measures, so that I can determine whether coherence-based detection outperforms sampling-based methods.

The researcher runs both SelfCheckGPT evaluation and coherence-based analysis on the same dataset. The system generates multiple samples per question, calculates both SelfCheckGPT consistency scores and Shogenji/Fitelson/Olsson coherence measures, and produces comparative analysis showing correlation and predictive performance for hallucination detection.

### Developer Benchmarking Multiple Models

As a developer, I want to run SelfCheckGPT across multiple LLMs with caching support, so that I can efficiently compare hallucination tendencies across different models.

The developer configures model parameters, sample sizes, and temperature settings. The system leverages the existing caching infrastructure to avoid redundant API calls, displays progress during evaluation, and outputs standardized results compatible with the existing benchmark framework.

## Spec Scope

1. **SelfCheckGPT Benchmark Implementation** - Implement core SelfCheckGPT evaluation logic following the NLI variant methodology from the reference paper.
2. **Multi-Temperature Sampling** - Generate 1 response at temperature 0.0 and 5 responses at temperature 1.0 per question, integrated with existing OpenAI client.
   - ⚠️ **DEVIATION FROM PAPER**: Paper uses 20 samples; we use 5 for Phase 1 (cost/speed trade-off, documented in research-references.md)
3. **NLI-Based Consistency Evaluation** - Use sentence-level Natural Language Inference to evaluate consistency between baseline and sample responses.
4. **CLI Interface** - Create command-line interface for running SelfCheckGPT benchmarks with configurable parameters.
5. **Results Integration** - Output results in standardized format compatible with existing benchmark framework and visualization tools.
6. **Caching Support** - Leverage existing LLM caching infrastructure to support temperature-variant sampling efficiently.
   - **CRITICAL**: Must properly handle multi-temperature caching (temp 0.0 vs 1.0 cached separately)
   - See research-references.md "Caching Requirements" section for lessons from TruthfulQA

## Out of Scope

- BERTScore, Question-Answering, N-gram, and LLM-Prompting variants (NLI variant only for Phase 1)
- Custom dataset creation (use existing Wiki Bio hallucination dataset)
- Integration with visualization dashboard (defer to later phase)
- Comparison with coherence measures (implementation only; analysis deferred)
- GPU optimization for NLI model inference
- Batch API support (use sequential generation with caching)

## Expected Deliverable

1. Running `selfcheckgpt run --model gpt-4o-mini --sample-size 50` successfully evaluates 50 questions from Wiki Bio dataset and outputs consistency scores.
2. Results JSON file includes per-sentence consistency scores, aggregate hallucination metrics, and metadata matching existing benchmark output format.
3. System correctly generates 6 samples per question (1 at temp 0.0, 5 at temp 1.0) with proper cache handling for each temperature setting.
4. NLI model successfully loads and evaluates sentence-level consistency with scores between 0-1.
5. Benchmark completes 100 questions in under 10 minutes using GPT-4o-mini with caching enabled.
6. **Validation against paper baselines**: AUC-PR scores within 10% of paper results (target: >85% for non-factual detection)
   - Paper baseline (20 samples): 92.50 AUC-PR for non-factual
   - Our target (5 samples): >82 AUC-PR for non-factual (acceptable given reduced sample count)
7. **Caching verification**: Cache statistics show proper separation of temp 0.0 vs 1.0 responses, >80% hit rate on repeat runs
