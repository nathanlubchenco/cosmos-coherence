# Technical Specification

This is the technical specification for the spec detailed in @.agent-os/specs/2025-09-29-selfcheckgpt-implementation/spec.md

**Research Foundation**: See `research-references.md` for complete paper methodology, baseline results, and API compatibility verification.

**CRITICAL LESSONS FROM TRUTHFULQA**: This spec incorporates lessons learned from TruthfulQA debugging:
1. ✅ Verify API compatibility FIRST (SelfCheckGPT confirmed compatible with Chat API)
2. ✅ Match paper methodology exactly (using NLI variant, temperatures 0.0/1.0)
3. ✅ Document all deviations clearly (5 samples vs paper's 20)
4. ✅ Test against published baselines (AUC-PR target: >82%)
5. ✅ Explicit caching requirements (temperature in cache key already verified)

## Technical Requirements

### Core Benchmark Implementation

1. **SelfCheckGPTBenchmark Class**
   - Inherit from `HuggingFaceEnabledBenchmark` base class
   - Implement `get_prompt()` method to format Wiki Bio prompts
   - Implement `evaluate_consistency()` method for NLI-based scoring
   - Follow existing benchmark patterns from SimpleQA and HaluEval

2. **Multi-Temperature Sampling**
   - Generate 1 baseline response with `temperature=0.0` (deterministic, matches paper)
   - Generate 5 sample responses with `temperature=1.0` (stochastic, shows variation)
   - ⚠️ **DEVIATION**: Paper uses 20 samples; we use 5 for cost/speed (Phase 1 only)
   - Use existing `OpenAIClient.generate_response()` with temperature parameter
   - **VERIFIED**: Cache keys include temperature (openai_client.py:150) ✅
   - Return list of responses: `[baseline, sample1, sample2, sample3, sample4, sample5]`
   - Each temperature generates separate cache entries (temp 0.0 ≠ temp 1.0)

3. **Sentence-Level Processing**
   - Split baseline response into sentences using NLTK or spaCy sentence tokenizer
   - For each sentence in baseline, evaluate against all 5 samples
   - Return per-sentence consistency scores

4. **NLI-Based Consistency Evaluation**
   - **Paper uses**: DeBERTa-v3-large fine-tuned on MNLI (~1.5GB)
   - **Phase 1 alternative**: DeBERTa-v3-base (~500MB, faster, slightly lower performance)
   - **Recommended**: Use SelfCheckGPT library's built-in NLI scorer (matches paper exactly)
   - For each sentence:
     - Generate premise-hypothesis pairs (sentence vs each sample)
     - Calculate entailment probability P(entailment | premise, hypothesis)
     - Aggregate scores: `score = average(P_entailment across 5 samples)`
   - Consistency score range: 0.0 (inconsistent/hallucinated) to 1.0 (consistent/factual)
   - **Expected AUC-PR**: >82% for non-factual detection (paper achieves 92.50 with 20 samples)

5. **Dataset Integration**
   - Use HuggingFace dataset: `potsawee/wiki_bio_gpt3_hallucination`
   - Dataset contains 238 annotated passages with human labels
   - Each item: concept, GPT-3 generated text, sentence-level annotations
   - Load via existing `HuggingFaceDatasetLoader` infrastructure

### CLI Interface

1. **Command Structure**
   ```bash
   selfcheckgpt run \
     --model gpt-4o-mini \
     --sample-size 50 \
     --temperature-baseline 0.0 \
     --temperature-samples 1.0 \
     --num-samples 5 \
     --output results.json \
     --cache/--no-cache \
     --verbose
   ```

2. **Parameters**
   - `--model`: OpenAI model name (default: gpt-4o-mini)
   - `--sample-size`: Number of questions to evaluate (default: all 238)
   - `--temperature-baseline`: Temperature for baseline generation (default: 0.0)
   - `--temperature-samples`: Temperature for sample generation (default: 1.0)
   - `--num-samples`: Number of samples per question (default: 5)
   - `--output`: Results output file path
   - `--cache/--no-cache`: Enable/disable response caching
   - `--verbose`: Show detailed progress

3. **Progress Display**
   - Use rich Progress for visual feedback
   - Show: questions processed, current consistency score, estimated time remaining
   - Display cache hit rate statistics

### Results Format

```json
{
  "benchmark": "SelfCheckGPT",
  "model": "gpt-4o-mini",
  "timestamp": "2025-09-29T10:30:00Z",
  "configuration": {
    "temperature_baseline": 0.0,
    "temperature_samples": 1.0,
    "num_samples": 5,
    "sample_size": 50
  },
  "aggregate_metrics": {
    "mean_consistency_score": 0.742,
    "hallucination_rate": 0.258,
    "total_questions": 50,
    "total_sentences_evaluated": 437
  },
  "per_question_results": [
    {
      "question_id": 0,
      "concept": "Albert Einstein",
      "baseline_response": "Albert Einstein was a theoretical physicist...",
      "num_sentences": 8,
      "sentence_scores": [0.95, 0.87, 0.45, 0.92, 0.78, 0.88, 0.91, 0.82],
      "mean_score": 0.823,
      "min_score": 0.45,
      "hallucinated_sentences": [2],
      "samples": ["sample1...", "sample2...", "sample3...", "sample4...", "sample5..."]
    }
  ],
  "cache_statistics": {
    "cache_hits": 245,
    "cache_misses": 55,
    "hit_rate": 0.817
  }
}
```

### Caching Requirements (CRITICAL - Lessons from TruthfulQA)

**MUST IMPLEMENT** (to avoid TruthfulQA caching bugs):

1. **Cache Key Components** (already verified in openai_client.py:150):
   - ✅ Model name
   - ✅ Temperature (CRITICAL for multi-temperature sampling)
   - ✅ Prompt text
   - ✅ Max tokens
   - ✅ All generation parameters

2. **Multi-Temperature Caching Behavior**:
   - Same prompt + temp 0.0 = cached separately from temp 1.0 ✅
   - Verification test: Generate at 0.0, then 1.0 → should NOT return same response
   - Cache stats should show separate entries for each temperature

3. **Persistent Storage**:
   - Default location: `~/.cache/cosmos_coherence/selfcheckgpt/`
   - Save cache after each run: `client.save_cache()`
   - Display cache statistics: hit rate, total entries, size

4. **Cache Validation Tests** (add to test suite):
   ```python
   # Test: Temperature differentiation in cache
   response_t0 = client.generate_response(prompt, temperature=0.0)
   response_t1 = client.generate_response(prompt, temperature=1.0)
   assert cache has 2 entries (not 1)

   # Test: Cache persistence
   client.save_cache()
   new_client = OpenAIClient(..., cache_file=same_file)
   response = new_client.generate_response(prompt, temperature=0.0)
   assert cache hit = True
   ```

### Integration Points

1. **OpenAI Client Integration**
   - Use existing `OpenAIClient` from `cosmos_coherence.llm.openai_client`
   - Leverage existing rate limiting, retry logic, and caching
   - ✅ No modifications needed to client (temperature already in cache key)
   - ✅ Cache properly handles multi-temperature sampling (verified)

2. **Benchmark Harness Integration**
   - Extend `HuggingFaceEnabledBenchmark` base class
   - Implement standard methods: `get_prompt()`, `evaluate_response()`
   - Add new method: `generate_multiple_samples()` for temperature variation
   - Follow patterns from `SimpleQABenchmark` and `HaluEvalBenchmark`

3. **Dataset Loading**
   - Use existing `HuggingFaceDatasetLoader`
   - Dataset: `potsawee/wiki_bio_gpt3_hallucination` (238 annotated passages)
   - Convert to `SelfCheckGPTItem` dataclass (create new model)

### Performance Considerations

1. **API Call Optimization**
   - 6 API calls per question (1 baseline + 5 samples)
   - With caching: subsequent runs should have ~80%+ cache hit rate
   - 100 questions = 600 API calls (first run)
   - Estimated time: 8-10 minutes for 100 questions with GPT-4o-mini

2. **NLI Model Performance**
   - Load model once at initialization (lazy loading)
   - Use batch inference for sentence evaluation when possible
   - Expected: ~100-200ms per sentence evaluation
   - Model size: ~500MB download on first use

3. **Memory Management**
   - Store only necessary data in results (avoid storing all samples)
   - Optional flag to include full samples in output (for debugging)
   - Estimated memory: ~50-100MB for 100 questions

### Error Handling

1. **API Errors**
   - Reuse existing retry logic from TruthfulQA (3 retries with exponential backoff)
   - Handle rate limiting gracefully
   - Skip problematic questions and continue (log errors)

2. **NLI Model Errors**
   - Graceful fallback if model fails to load
   - Return error status in results
   - Clear error messages to user

3. **Dataset Errors**
   - Handle missing or malformed dataset items
   - Skip invalid items with warning
   - Continue processing remaining items

## External Dependencies

### New Dependencies

1. **selfcheckgpt** (optional, for reference NLI implementation)
   - Version: >=0.1.0
   - Purpose: Reference implementation of SelfCheckGPT NLI scorer
   - Justification: Provides validated NLI-based consistency evaluation matching paper methodology
   - Alternative: Implement custom NLI scorer using transformers library

2. **transformers** (already in project via torch)
   - Verify version: >=4.30.0
   - Purpose: Load pre-trained NLI models for consistency evaluation
   - Already used for other ML operations

3. **nltk** or **spacy** (choose one)
   - Purpose: Sentence tokenization for baseline response splitting
   - Justification: Accurate sentence boundary detection for per-sentence evaluation
   - Lightweight dependency (~10MB)

### Dependency Decision

Recommend using `selfcheckgpt` library for Phase 1 to ensure methodology matches paper exactly. This provides:
- Validated NLI implementation
- Tested on reference dataset
- Easier to compare results with published baselines
- Can reimplement custom scorer in Phase 2 if needed

Installation:
```bash
poetry add selfcheckgpt
poetry add nltk  # or spacy
```

## Testing Requirements

1. **Unit Tests**
   - Test sentence tokenization
   - Test consistency score calculation with mock data
   - Test multi-temperature sampling logic
   - Test cache key generation with temperature parameter

2. **Integration Tests**
   - Test end-to-end benchmark run with small sample (5 questions)
   - Test cache persistence and reloading
   - Test results format validation

3. **Validation Tests**
   - **Compare against paper baselines**: AUC-PR scores within 10% of paper (target: >82%)
     - Paper baseline (20 samples): 92.50 AUC-PR for non-factual
     - Our expected (5 samples): 82-92 AUC-PR for non-factual
   - Verify consistency scores fall within expected range (0-1)
   - Validate that higher temperature increases variance (temp 1.0 responses differ)
   - Test caching: Same prompt at temp 0.0 always returns same response (cache hit)
   - Test multi-temp: Same prompt at temp 0.0 vs 1.0 creates separate cache entries

## Implementation Phases

### Phase 1: Core Implementation (1-2 days)
1. Create `SelfCheckGPTBenchmark` class
2. Implement multi-temperature sampling
3. Integrate SelfCheckGPT NLI scorer
4. Create CLI interface
5. Add basic tests

### Phase 2: Validation & Optimization (0.5 days)
1. Run on full dataset (238 questions)
2. Compare with published baselines
3. Optimize performance if needed
4. Add comprehensive tests

### Phase 3: Documentation & Cleanup (0.5 days)
1. Update README with SelfCheckGPT usage
2. Add example configurations
3. Document findings in assessment report
4. Code review and cleanup
