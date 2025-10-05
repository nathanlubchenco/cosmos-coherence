# Spec Tasks

These are the tasks to be completed for the spec detailed in @.agent-os/specs/2025-09-29-selfcheckgpt-implementation/spec.md

> Created: 2025-10-05
> Status: Ready for Implementation

## Tasks

### 1. Dataset Loading and Data Models (Foundation) ✅ COMPLETE

Build the foundation for loading Wikipedia biography data and representing SelfCheckGPT items.

- [x] 1.1 Write tests for HuggingFace dataset loader (potsawee/wiki_bio_gpt3_hallucination)
- [x] 1.2 Create SelfCheckGPTItem dataclass with fields: topic, wiki_bio_text, gpt3_text, gpt3_sentences, annotation
- [x] 1.3 Implement dataset loading with proper error handling
- [x] 1.4 Add sentence tokenization using Spacy (en_core_web_sm)
- [x] 1.5 Verify all tests pass with `make test`

**Critical Notes:**
- Dataset has 238 passages total
- Each passage has human annotations for hallucinated sentences
- Sentence tokenization must match paper methodology (Spacy)

### 2. Multi-Temperature Sampling Infrastructure (Core Capability) ✅ COMPLETE

Implement the ability to generate multiple samples at different temperatures with proper caching.

- [x] 2.1 Write tests for multi-temperature generation (1 sample at temp=0.0, 5 samples at temp=1.0)
- [x] 2.2 Write tests verifying cache temperature differentiation (temp 0.0 vs 1.0 create separate cache entries)
- [x] 2.3 Implement generate_multiple_samples() method in benchmark class
- [x] 2.4 Add validation test confirming temperature is in cache key (verify openai_client.py:150 behavior)
- [x] 2.5 Add cache persistence to ~/.cache/cosmos_coherence/selfcheckgpt/
- [x] 2.6 Add cache statistics display (hit rate, entries, savings)
- [x] 2.7 Verify all tests pass with `make test`

**Critical Notes:**
- Temperature MUST be in cache key (already implemented in openai_client.py:150)
- Using 5 samples instead of paper's 20 (cost optimization)
- Cache hit rate should be >80% on second run
- Each temperature creates separate cache entries

### 3. NLI-Based Consistency Evaluation (Hallucination Detection) ✅ COMPLETE

Integrate the SelfCheckGPT library's NLI scorer to detect hallucinations.

- [x] 3.1 Write tests for NLI scorer integration (SelfCheckNLI)
- [x] 3.2 Write tests for sentence-level consistency calculation
- [x] 3.3 Install selfcheckgpt library: `pip install selfcheckgpt`
- [x] 3.4 Install DeBERTa dependencies: `pip install transformers torch`
- [x] 3.5 Integrate SelfCheckNLI scorer (uses potsawee/deberta-v3-large-mnli)
- [x] 3.6 Implement evaluate_consistency() method for per-sentence scoring
- [x] 3.7 Add aggregate scoring across all sentences
- [x] 3.8 Verify all tests pass with `make test`

**Critical Notes:**
- Use SelfCheckGPT library's NLI scorer (matches paper exactly)
- NLI model: DeBERTa-v3-large fine-tuned on MNLI
- Lower consistency scores indicate higher hallucination probability
- Return per-sentence scores for analysis

### 4. Benchmark Implementation and CLI (Integration) ✅ COMPLETE

Create the complete benchmark class and command-line interface.

- [x] 4.1 Write tests for SelfCheckGPTBenchmark class
- [x] 4.2 Create SelfCheckGPTBenchmark extending HuggingFaceEnabledBenchmark
- [x] 4.3 Implement get_prompt() method (system: summarize biography, user: topic name)
- [x] 4.4 Implement evaluate_response() method (generate samples, run NLI, score)
- [x] 4.5 Create CLI interface at src/cosmos_coherence/benchmarks/selfcheckgpt_cli.py
- [x] 4.6 Add progress display with tqdm (show current passage, cache stats)
- [x] 4.7 Add --cache/--no-cache flags to CLI
- [x] 4.8 Add results JSON output format (per-sentence scores, aggregate metrics)
- [x] 4.9 Verify all tests pass with `make test`

**Critical Notes:**
- Extend HuggingFaceEnabledBenchmark (provides dataset loading)
- CLI should match pattern from truthfulqa_cli.py
- Display cache hit rate prominently
- Save results to results.json by default

### 5. Validation Against Paper Baselines (Quality Assurance) 🔄 IN PROGRESS

Validate implementation against paper's reported results and document deviations.

- [x] 5.1 Run benchmark on small sample (2 passages) for smoke test ✅
- [x] 5.2 Verify smoke test completes without errors ✅
- [ ] 5.3 Run full dataset (238 passages) with caching enabled
- [ ] 5.4 Calculate AUC-PR scores for non-factual sentence detection
- [ ] 5.5 Verify AUC-PR >82% (target: match paper's 92.50% within 10%)
- [ ] 5.6 Verify cache hit rate >80% on second run
- [ ] 5.7 Document any deviations from paper results in results analysis
- [x] 5.8 Run pre-commit checks: `pre-commit run --all-files` ✅
- [x] 5.9 Verify all linting passes (ruff, mypy) ✅
- [x] 5.10 Verify all tests pass with `make test` ✅ (20/20 tests passing)

**Critical Notes:**
- Target AUC-PR: >82% (accounting for 5 samples vs paper's 20)
- Paper baseline: 92.50% AUC-PR with SelfCheck-NLI
- Document deviation: Using 5 samples instead of 20 (cost optimization)
- Cache behavior critical for cost control
- Compare results with TruthfulQA lessons learned

## Ordering Principles

1. **TDD First**: Write tests before implementation for each component
2. **Incremental Build**: Data Models → Sampling → Evaluation → CLI → Validation
3. **Early Cache Verification**: Test cache behavior in Task 2 (learned from TruthfulQA)
4. **Baseline Testing**: Validate against paper results early (avoid late discovery of incompatibilities)

## Critical Implementation Reminders

- **Cache Key**: Temperature MUST be in cache key (already verified in openai_client.py:150)
- **Sample Count**: Using 5 samples vs paper's 20 (documented deviation)
- **Target Metric**: AUC-PR >82% (90% of paper's 92.50%)
- **NLI Scorer**: Use SelfCheckGPT library's implementation (matches paper exactly)
- **Sentence Tokenization**: Use Spacy en_core_web_sm (matches paper methodology)
- **Cache Location**: ~/.cache/cosmos_coherence/selfcheckgpt/
- **Line Length**: MUST be <=100 characters (ruff E501 rule)

## Success Criteria

- [ ] All tests pass (`make test`)
- [ ] All pre-commit checks pass (`pre-commit run --all-files`)
- [ ] AUC-PR score >82% on full dataset
- [ ] Cache hit rate >80% on second run
- [ ] CLI provides clear progress and statistics
- [ ] Results match paper methodology (accounting for sample count deviation)
