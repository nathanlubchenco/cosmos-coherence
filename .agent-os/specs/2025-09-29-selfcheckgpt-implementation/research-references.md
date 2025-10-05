# SelfCheckGPT Research References

## Paper Details

**Title**: SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models

**Authors**:
- Potsawee Manakul
- Adian Liusie
- Mark J. F. Gales

**Publication**: EMNLP 2023 (Accepted)

**arXiv ID**: 2303.08896v3

**Links**:
- Paper: https://arxiv.org/abs/2303.08896
- Repository: https://github.com/potsawee/selfcheckgpt
- HTML: https://ar5iv.labs.arxiv.org/html/2303.08896

## Core Methodology

### Key Insight

"If an LLM has knowledge of a given concept, sampled responses are likely to be similar and contain consistent facts. For hallucinated facts, stochastically sampled responses are likely to diverge and contradict one another."

### Approach

1. **Generate main response** with temperature = 0.0 (deterministic)
2. **Generate N sample responses** with temperature = 1.0 (stochastic)
3. **Evaluate consistency** between main response and samples at sentence level
4. **Higher inconsistency = higher hallucination probability**

### Sampling Parameters (From Paper)

- **Main response temperature**: 0.0 (standard beam search)
- **Sample temperature**: 1.0 (stochastic sampling)
- **Number of samples (N)**: **20 samples per passage**
- **Model used in paper**: GPT-3 (text-davinci-003) and ChatGPT (gpt-3.5-turbo)

⚠️ **DEVIATION FOR PHASE 1**: We will use **5 samples** instead of 20 for:
- Cost efficiency (~4x reduction in API calls)
- Faster iteration during development
- Trade-off: May reduce AUC-PR scores by ~5-10% (acceptable for Phase 1)
- Can increase to 20 samples in Phase 2 for final validation

### Variants Investigated

1. **BERTScore**: Similarity-based using BERT embeddings
2. **Question-Answering (MQAG)**: Question generation and answering consistency
3. **N-gram**: Lexical overlap statistics
4. **NLI** (Natural Language Inference): Entailment probability ⭐ **BEST PERFORMANCE**
5. **LLM-Prompting**: Use LLM to judge consistency

**Phase 1 Focus**: NLI variant (best performance according to paper)

## NLI Variant Details

### Model

- **NLI Model**: DeBERTa-v3-large fine-tuned on MNLI dataset
- **Model size**: ~1.5GB (larger than base, which is ~500MB)
- **Alternative for Phase 1**: DeBERTa-v3-base (faster, slightly lower performance)

### Evaluation Process

For each sentence in the main response:
1. Use sentence as **premise**
2. Use corresponding sentences in samples as **hypotheses**
3. Calculate entailment probability P(entailment | premise, hypothesis)
4. Aggregate across samples: `score = average(P_entailment across N samples)`
5. Low score = inconsistent = likely hallucinated

### Sentence Tokenization

- Uses **Spacy** for sentence splitting
- Model: `en_core_web_sm`

## Dataset

**Name**: `wiki_bio_gpt3_hallucination`

**Source**:
- HuggingFace: `potsawee/wiki_bio_gpt3_hallucination`
- Direct download: Available in repository

**Size**: 238 passages with human annotations

**Structure**:
- Concept: Person's name (e.g., "Albert Einstein")
- GPT-3 generated passage: Biographical text
- Annotations: Sentence-level factuality labels
  - Factual
  - Non-Factual
  - Non-Factual* (minor inaccuracies)

**Annotation Details**:
- 2 annotators per passage
- Disagreements resolved through discussion
- Sentence-level annotations (not just passage-level)

## Baseline Results (From Paper)

### AUC-PR Scores (NLI Variant)

Performance on different factuality categories:

| Category | AUC-PR | Description |
|----------|--------|-------------|
| Non-Factual | 92.50 | Major hallucinations (clearly wrong facts) |
| Non-Factual* | 45.17 | Minor inaccuracies (dates off by 1 year, etc.) |
| Factual | 66.08 | Correct factual sentences |

**Overall Performance**:
- Best at detecting major hallucinations (92.50 AUC-PR)
- Struggles with minor inaccuracies (45.17 AUC-PR)
- Good general factual detection (66.08 AUC-PR)

### Comparison with Other Variants

From Table 1 in paper (AUC-PR scores):

| Method | Non-Factual | Non-Factual* | Factual |
|--------|-------------|--------------|---------|
| BERTScore | 90.53 | 41.08 | 64.00 |
| MQAG | 86.13 | 34.33 | 56.67 |
| N-gram | 78.13 | 32.75 | 51.42 |
| **NLI** | **92.50** | **45.17** | **66.08** |
| LLM-Prompt | 89.87 | 40.25 | 63.33 |

**Conclusion**: NLI variant performs best across all categories.

## API Compatibility

### OpenAI API Usage

✅ **CONFIRMED COMPATIBLE** with Chat Completions API

The paper explicitly uses:
- GPT-3 (text-davinci-003) for passage generation
- **ChatGPT (gpt-3.5-turbo)** for LLM-Prompt variant
- Both accessible via OpenAI API

### Cost Estimates (From Paper)

- GPT-3 (text-davinci-003): $0.020 per passage evaluation
- ChatGPT (gpt-3.5-turbo): $0.002 per passage evaluation

**For Phase 1 with GPT-4o-mini** (cheaper than both):
- Estimated: ~$0.0005 per passage with 5 samples
- 238 passages: ~$0.12 total
- 1000 passages: ~$0.50 total

### No Known Limitations

Unlike TruthfulQA:
- ✅ Uses standard text generation (not logprobs)
- ✅ Works with Chat Completions API
- ✅ No special API features required
- ✅ Temperature variation supported on all models

## Expected Implementation Results

### Success Criteria (Phase 1)

1. **AUC-PR scores within 10% of paper baselines**:
   - Non-Factual: 82-92 (target: >85)
   - Factual: 60-70 (target: >63)
   - Overall: Similar ranking of passages

2. **Performance**:
   - 238 passages with 5 samples each = 1,190 API calls
   - With 80% cache hit rate: ~240 new API calls
   - Estimated time: <10 minutes for full dataset

3. **Reproducibility**:
   - Same model, same passages → same scores (via caching)
   - Deterministic evaluation (fixed seed for NLI if stochastic)
   - Results format matches existing benchmarks

### Phase 2 Enhancements

- Increase to 20 samples (match paper exactly)
- Test other variants (BERTScore, LLM-Prompt)
- Add coherence measure comparison
- Multi-model evaluation

## Implementation Notes

### Critical Lessons from TruthfulQA

1. ✅ **Verify API compatibility FIRST** - Confirmed SelfCheckGPT uses Chat API
2. ✅ **Match paper methodology** - Use DeBERTa NLI, temperature 0.0/1.0
3. ✅ **Document deviations** - Using 5 samples instead of 20 (clearly noted)
4. ✅ **Test against baselines** - Expect AUC-PR ~85-92% for non-factual
5. ✅ **Proper caching** - Temperature MUST be in cache key (already done)

### Caching Requirements

**CRITICAL** (learned from TruthfulQA debugging):

1. **Cache key MUST include**:
   - Model name
   - Temperature (already in `openai_client.py:150`)
   - Prompt text
   - Max tokens
   - Any other generation parameters

2. **Multi-temperature handling**:
   - Temperature 0.0 responses cached separately from 1.0
   - Same prompt, different temperature = different cache entries ✅
   - Verified: Already implemented correctly in our OpenAI client

3. **Persistent storage**:
   - Save cache to disk after each run
   - Show cache statistics (hit rate)
   - Default location: `~/.cache/cosmos_coherence/selfcheckgpt/`

### Validation Approach

1. **Unit tests**: Test sentence tokenization, NLI scoring
2. **Integration tests**: Run on 5 passages, verify consistency
3. **Baseline comparison**: Run full 238 passages, compare AUC-PR
4. **Debugging**: If results don't match:
   - Check sentence tokenization (must match Spacy output)
   - Verify NLI model loading (DeBERTa-v3-large or base)
   - Inspect sample quality (temperature 1.0 should show variation)
   - Compare aggregation method (average vs min vs max)

## References

1. Manakul, P., Liusie, A., & Gales, M. J. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. *EMNLP 2023*.

2. Official Implementation: https://github.com/potsawee/selfcheckgpt

3. Dataset: https://huggingface.co/datasets/potsawee/wiki_bio_gpt3_hallucination

4. DeBERTa Model: https://huggingface.co/microsoft/deberta-v3-large
