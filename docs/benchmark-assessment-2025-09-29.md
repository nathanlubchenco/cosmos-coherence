# Comprehensive Benchmark Assessment for Cosmos Coherence
**Investigation Date:** 2025-09-29
**Purpose:** Identify potential issues in current implementations and recommend additional benchmarks

---

## Executive Summary

### Current Status

✅ **Working Implementations:**
- **SimpleQA**: AI-graded factuality evaluation (Chat API compatible)
- **HaluEval**: Binary classification hallucination detection (Chat API compatible)

❌ **Problematic Implementation:**
- **TruthfulQA**: Fundamentally incompatible with OpenAI Chat API (see `docs/limitations/truthfulqa.md`)

### Key Recommendations

🎯 **Highest Priority Addition:**
- **SelfCheckGPT**: Perfect alignment with project goals (temperature variation + consistency checking)

🌟 **Valuable Additions:**
- **HalluLens** (2025): Modern, comprehensive, dynamic test generation
- **Vectara Hallucination Leaderboard**: Summarization faithfulness

---

## Part 1: Current Implementation Analysis

### 1.1 SimpleQA ✅ WORKING

**Implementation Details:**
- **Method**: Generate answer → AI-grade as CORRECT/INCORRECT/NOT_ATTEMPTED
- **Grading Model**: GPT-4o-mini with comprehensive prompt template
- **Dataset**: basicv8vc/SimpleQA from HuggingFace
- **API Compatibility**: ✅ Works perfectly with Chat API
- **Cost**: Low (~$0.05-0.10 per 100 questions with GPT-4o-mini grading)

**Potential Issues Found:** NONE

**Assessment**: This is a solid, well-implemented benchmark. No changes needed.

**Code Quality:**
- Clear separation of benchmark logic and grading
- Follows OpenAI reference implementation
- Good test coverage
- Proper caching support

---

### 1.2 HaluEval ✅ WORKING

**Implementation Details:**
- **Method**: Binary classification (Yes=hallucinated, No=not hallucinated)
- **Tasks**: QA, Dialogue, Summarization
- **Dataset**: pminervini/HaluEval from HuggingFace (35K examples)
- **API Compatibility**: ✅ Works perfectly with Chat API
- **Evaluation**: Prompts with few-shot examples, model responds Yes/No

**Potential Issues Found:** MINOR

The implementation uses token truncation at 2033 tokens (line 65 in `halueval_benchmark.py`):
```python
if HaluEvalBenchmark.num_tokens_from_message(prompt1 + prompt2, model) > 2033:
```

**Recommendation**: This 2033 limit seems arbitrary and quite low for modern models:
- GPT-4-turbo: 128K context
- GPT-4o: 128K context
- GPT-4o-mini: 128K context

Consider:
1. Making this configurable via parameter
2. Increasing default to at least 8K-16K for modern models
3. Document why 2033 was chosen (likely from original davinci limits)

**Assessment**: Working well, minor optimization opportunity.

---

### 1.3 TruthfulQA ❌ INCOMPATIBLE (Documented)

**Implementation Details:**
- **Method**: Multiple-choice with log probability scoring
- **Issue**: Requires `log P(answer_choice | question)` which Chat API doesn't support
- **Status**: Thoroughly documented in `docs/limitations/truthfulqa.md`

**Current Results:**
- 77/100 questions wrong
- -17% below published baselines
- Fundamentally broken evaluation methodology

**Assessment**: Cannot be fixed without either:
1. Using generation-based evaluation (requires GPU for LLaMA judge)
2. Using open-source models with proper logprob support
3. Accepting the limitation and marking as "reference only"

**Recommendation**: Already documented. Move on to other benchmarks.

---

## Part 2: Modern Hallucination Benchmarks (2024-2025)

### 2.1 SelfCheckGPT 🎯 HIGHEST PRIORITY

**Why This Is Perfect for Your Project:**

Your project's core focus:
> "Temperature variation studies: Systematic analysis across different temperature settings"
> "Implements formal philosophical coherence measures (Shogenji, Fitelson, Olsson)"

SelfCheckGPT does EXACTLY this:
- Generates 1 passage at **temperature 0.0**
- Generates 5 passages at **temperature 1.0**
- Evaluates consistency across samples
- Detects hallucinations via inconsistency

**Method:**
```
1. Prompt model to generate passage (temp=0.0)
2. Generate 5 more samples (temp=1.0)
3. Check if sentences in passage #1 are supported by passages #2-6
4. High inconsistency = hallucination
```

**Principle**:
- If model has knowledge → samples are consistent
- If model is hallucinating → samples diverge and contradict

**Implementation Variants:**
- BERTScore (similarity-based)
- Question-Answering (QA-based verification)
- N-gram (overlap-based)
- **NLI (Natural Language Inference)** ← Recommended, best performance
- LLM Prompting (use LLM to check consistency)

**API Compatibility:** ✅ Perfect
- Just needs multiple generations at different temperatures
- All evaluation can use existing OpenAI models or open-source NLI models

**Dataset:**
- `wiki_bio_gpt3_hallucination` (238 annotated passages)
- Can also evaluate on custom prompts

**Installation:** `pip install selfcheckgpt`

**Integration Complexity:** LOW-MEDIUM
- ✅ Pure generation-based (no logprobs)
- ✅ Already aligned with project's temperature variation focus
- ⚠️ NLI variant needs HuggingFace model (~500MB)
- ⚠️ Need to implement consistency scoring

**Cost Estimate:**
- 6 generations per question (1 at temp=0, 5 at temp=1)
- Using GPT-4o-mini: ~$0.30-0.50 per 100 questions
- NLI model: Free (local inference)

**Recommendation:** **IMPLEMENT IMMEDIATELY**

This benchmark is uniquely aligned with your project's research goals. It directly tests the hypothesis that temperature variation and consistency can detect hallucinations.

---

### 2.2 HalluLens 🌟 HIGH VALUE

**Released:** April 2025 (Very recent!)

**What Makes It Special:**
1. **Clear Taxonomy**: Distinguishes extrinsic vs intrinsic hallucinations
2. **Dynamic Test Generation**: Prevents data leakage and memorization
3. **Comprehensive Coverage**: Multiple task types

**Tasks:**
- **PreciseWikiQA**: Short, fact-based queries
- **LongWiki**: Long-form content generation
- **NonExistentRefusal**: Tests if model admits ignorance for non-existent entities

**Key Innovation:**
Unlike static benchmarks, HalluLens dynamically generates test questions, making it harder for models to game through memorization.

**API Compatibility:** ✅ Compatible
- Generation-based evaluation
- No special API requirements

**Availability:**
- Open-source: https://github.com/facebookresearch/HalluLens
- Tested on 13 major LLMs

**Integration Complexity:** MEDIUM
- Need to implement dynamic question generation
- Need to set up evaluation framework
- May require understanding their taxonomy

**Recommendation:** **HIGH PRIORITY**

This is the most modern, comprehensive hallucination benchmark available. It represents current best practices (as of 2025) and would make your project very current.

---

### 2.3 Vectara Hallucination Leaderboard 📊 MEDIUM PRIORITY

**Focus:** Summarization faithfulness

**Method:**
1. Feed 1000 short documents to model
2. Ask model to summarize using ONLY facts from document
3. Use trained classifier to detect hallucinations in summaries

**Why It's Interesting:**
- Tests faithfulness specifically (not general factuality)
- Includes latest 2025 models (GPT-4.5, Claude 3.7, o3-mini)
- Industry-recognized leaderboard

**API Compatibility:** ✅ Compatible
- Standard generation task

**Dataset:**
- 1000 curated documents
- Available via leaderboard website

**Integration Complexity:** MEDIUM
- Need hallucination classifier (they provide one)
- Or could use their API/leaderboard directly

**Recommendation:** **CONSIDER IF FOCUSING ON SUMMARIZATION**

Good addition if your project wants to specifically test summarization tasks. Otherwise, SimpleQA and HaluEval cover general hallucination well.

---

### 2.4 Hugging Face Hallucinations Leaderboard 📊 REFERENCE

**What It Includes:**
- SelfCheckGPT
- NQ Open, TriviaQA
- TruthfulQA
- MemoTrap, IFEval
- XSum, CNN/DM summarization
- HaluEval, FaithDial

**Why It's Useful:**
- Industry-standard benchmark suite
- Normalized scores (0-1 scale)
- Can compare your results against leaderboard

**Recommendation:** **USE AS REFERENCE**

Don't reimplement all of these, but use the leaderboard to:
1. Validate your SimpleQA/HaluEval implementations
2. Compare your models' performance
3. Identify which specific weaknesses your coherence measures address

---

## Part 3: API Compatibility Assessment

### Benchmark Compatibility Matrix

| Benchmark | Chat API | Logprobs Needed | Special Requirements | Status |
|-----------|----------|-----------------|---------------------|---------|
| SimpleQA | ✅ | ❌ | AI grader | WORKING |
| HaluEval | ✅ | ❌ | None | WORKING |
| TruthfulQA (MC) | ❌ | ✅ (arbitrary text) | N/A | INCOMPATIBLE |
| TruthfulQA (Gen) | ✅ | ❌ | GPU for judge | NOT IMPLEMENTED |
| SelfCheckGPT | ✅ | ❌ | NLI model (small) | NOT IMPLEMENTED |
| HalluLens | ✅ | ❌ | Dynamic gen framework | NOT IMPLEMENTED |
| Vectara Leaderboard | ✅ | ❌ | Hallucination classifier | NOT IMPLEMENTED |
| RAGAS (RAG) | ✅ | ❌ | RAG system | NOT APPLICABLE |
| FActScore | ✅ | ❌ | Fact decomposition | NOT IMPLEMENTED |
| FELM | ✅ | ❌ | None | NOT IMPLEMENTED |

### Key Insights:

**✅ No Compatibility Issues for Generation-Based Benchmarks**

All modern hallucination benchmarks work with Chat API because they use:
- Generation tasks (what the model says)
- Classification tasks (hallucinated: yes/no)
- Consistency checking (comparing multiple outputs)

**❌ Only Legacy Multiple-Choice Benchmarks Have Issues**

The logprobs problem is specific to:
- TruthfulQA MC (legacy design for Completions API)
- Any other benchmark requiring `log P(arbitrary_text | prompt)`

**🎯 Future-Proof Strategy:**

Focus on:
1. Generation-based evaluation
2. Consistency/coherence measures
3. AI-as-a-judge approaches

Avoid:
1. Log probability scoring of arbitrary text
2. Benchmarks designed for legacy Completions API

---

## Part 4: Recommendations Prioritized by Value

### TIER 1: Implement Immediately 🚀

**1. SelfCheckGPT**

**Why:** Perfect alignment with project's core research goals
- Uses temperature variation (your key feature!)
- Measures consistency (coherence measure alignment!)
- Well-established method (EMNLP 2023)
- Easy to integrate

**Implementation Effort:** LOW-MEDIUM (1-2 days)

**Expected Value:** VERY HIGH
- Directly tests your coherence hypothesis
- Can integrate with your coherence measures
- Publishable research contribution

**Action Items:**
1. Install: `pip install selfcheckgpt`
2. Implement NLI variant (best performance)
3. Integrate with existing temperature variation system
4. Compare SelfCheckGPT consistency with your coherence measures

---

### TIER 2: High Value Additions ⭐

**2. HalluLens**

**Why:** Most modern, comprehensive benchmark (2025)
- Dynamic test generation prevents data leakage
- Clear hallucination taxonomy
- Industry recognition

**Implementation Effort:** MEDIUM (3-5 days)

**Expected Value:** HIGH
- Shows you're using cutting-edge benchmarks
- Comprehensive evaluation
- Strong for publications

**Action Items:**
1. Clone: https://github.com/facebookresearch/HalluLens
2. Study their taxonomy
3. Implement integration
4. Compare with other benchmarks

---

**3. Vectara Hallucination Leaderboard**

**Why:** Specific focus on summarization faithfulness
- Industry leaderboard (good for visibility)
- Tests a different dimension (faithfulness vs factuality)

**Implementation Effort:** MEDIUM (2-4 days)

**Expected Value:** MEDIUM-HIGH
- Adds summarization dimension
- Leaderboard comparison

**Action Items:**
1. Get dataset and classifier from Vectara
2. Implement summarization evaluation
3. Submit results to leaderboard (optional)

---

### TIER 3: Consider If Time Permits 📋

**4. FActScore** (Long-form factuality)

**Why:** Granular fact-level verification
- Good for long-form generation
- Decomposes into individual facts

**Implementation Effort:** MEDIUM-HIGH

**Expected Value:** MEDIUM

---

**5. RAGAS** (RAG-specific)

**Why:** If you add RAG capabilities
- Industry standard for RAG evaluation
- Faithfulness and relevance metrics

**Implementation Effort:** MEDIUM

**Expected Value:** MEDIUM (only if adding RAG)

---

### TIER 4: Lower Priority ⏸️

**6. MMLU / BBH** (General knowledge)

**Why:** Well-known benchmarks, but not hallucination-specific
- Good for general capability testing
- Not aligned with hallucination focus

**Recommendation:** Skip unless you want general capability baselines

---

**7. FEVER** (Fact verification)

**Why:** Complex multi-step verification
- Requires claim extraction
- Requires evidence retrieval
- Requires verification pipeline

**Recommendation:** Skip in favor of simpler hallucination benchmarks

---

## Part 5: Integration Strategy

### Phased Approach

**Phase 1: SelfCheckGPT (Week 1)**
- Implement core SelfCheckGPT evaluation
- Integrate with temperature variation system
- Compare with SimpleQA and HaluEval results
- Initial analysis of consistency vs coherence measures

**Phase 2: HalluLens (Week 2-3)**
- Implement HalluLens framework
- Dynamic test generation
- Comprehensive evaluation across all tasks
- Compare results with existing benchmarks

**Phase 3: Optional Additions (Week 4+)**
- Vectara leaderboard (if focusing on summarization)
- Additional benchmarks as needed
- Consolidate results across all benchmarks

---

## Part 6: Potential Issues to Watch For

### Common Pitfalls

**1. Inconsistent Prompt Formatting**
- Issue: Different benchmarks may expect different formats
- Solution: Document prompt templates clearly
- Test: Validate against reference implementations

**2. Caching with Temperature Variation**
- Issue: SelfCheckGPT needs multiple generations with different temps
- Current cache includes temperature in key? **VERIFY THIS**
- Solution: Ensure cache key includes both prompt AND temperature

**3. Token Limits**
- Issue: HaluEval uses 2033 token limit (very low)
- Solution: Make configurable, increase defaults
- Test: Verify truncation doesn't break evaluation

**4. Rate Limiting**
- Issue: SelfCheckGPT needs 6x more API calls
- Solution: Already have retry logic from TruthfulQA fixes
- Monitor: Track costs carefully

**5. Model Consistency**
- Issue: Using different models for evaluation vs testing
- Solution: Document which models are used where
- Example: GPT-4o for generation, GPT-4o-mini for grading

---

## Part 7: Cost Estimates

### Per-Benchmark Cost Analysis (100 questions)

| Benchmark | API Calls | Tokens/Call | Cost (GPT-4o-mini) | Cost (GPT-4) |
|-----------|-----------|-------------|-------------------|--------------|
| SimpleQA | 100 + 100 grading | ~200 | $0.05-0.10 | $0.50-1.00 |
| HaluEval | 100 | ~500 | $0.10-0.15 | $1.00-1.50 |
| SelfCheckGPT | 600 (6 per Q) | ~200 | $0.30-0.50 | $3.00-5.00 |
| HalluLens | 100-300 | ~300 | $0.15-0.45 | $1.50-4.50 |
| Vectara | 1000 (fixed) | ~500 | $1.50-2.00 | $15-20 |

**Total for Full Suite:** ~$2-4 with GPT-4o-mini, ~$20-30 with GPT-4

---

## Part 8: Technical Debt & Maintenance

### What Needs Fixing in Current Code

**1. HaluEval Token Limit**
- File: `halueval_benchmark.py:65`
- Issue: Hardcoded 2033 token limit
- Priority: LOW (works but suboptimal)
- Fix: Make configurable parameter

**2. TruthfulQA Tests**
- Issue: Tests written for buggy behavior (label=0 instead of label=1)
- Priority: HIGH (currently failing)
- Fix: Update test assertions to match corrected behavior

**3. Cache Temperature Key**
- Issue: Need to verify temperature is in cache key
- Priority: HIGH (critical for SelfCheckGPT)
- Fix: Confirmed in `openai_client.py:150` - temperature IS included ✅

---

## Part 9: Research Impact

### How These Benchmarks Support Your Research Goals

**Core Project Goal:**
> "Apply formal philosophical coherence measures to benchmark evaluations"

**How Benchmarks Align:**

**SelfCheckGPT** 🎯 PERFECT ALIGNMENT
- Measures consistency across samples (coherence!)
- Uses temperature variation (your method!)
- Can directly compare SelfCheckGPT scores with Shogenji/Fitelson/Olsson measures
- Research question: "Do coherence measures predict hallucinations better than SelfCheckGPT?"

**HalluLens** ⭐ STRONG ALIGNMENT
- Comprehensive hallucination taxonomy
- Tests diverse scenarios
- Can show coherence measures work across different hallucination types

**SimpleQA & HaluEval** ✅ GOOD BASELINES
- Well-established benchmarks
- Allow comparison with published work
- Validate your framework works on standard benchmarks

**Potential Publications:**

1. **"Coherence-Based Hallucination Detection: Comparing Philosophical Measures with Temperature Sampling"**
   - Compare Shogenji/Fitelson/Olsson with SelfCheckGPT
   - Show when coherence measures outperform consistency checking

2. **"Temperature-Variant Coherence Analysis for LLM Reliability"**
   - Systematic study of temperature effects on coherence
   - Use multiple benchmarks (SimpleQA, HaluEval, SelfCheckGPT, HalluLens)

3. **"Dynamic Hallucination Detection via Multi-Response Coherence"**
   - Novel application of philosophical coherence measures
   - Evaluated on modern benchmarks

---

## Part 10: Final Recommendations

### DO THIS IMMEDIATELY:

1. **Implement SelfCheckGPT**
   - Direct alignment with research goals
   - Low implementation effort
   - High research value
   - Can integrate with existing temperature variation system

2. **Verify Cache Includes Temperature**
   - Critical for SelfCheckGPT
   - Already implemented ✅ but should test

3. **Update TruthfulQA Tests**
   - Fix test assertions for corrected label behavior
   - Mark as "limited compatibility" in docs

### DO THIS SOON:

4. **Implement HalluLens**
   - Most modern benchmark (2025)
   - Comprehensive evaluation
   - Strong for publications

5. **Make HaluEval Token Limit Configurable**
   - Low effort
   - Better results on modern models

### CONSIDER LATER:

6. **Vectara Leaderboard** (if focusing on summarization)
7. **FActScore** (if needing long-form evaluation)
8. **RAGAS** (only if adding RAG capabilities)

### DON'T BOTHER WITH:

- MMLU/BBH (not hallucination-specific)
- FEVER (too complex for value gained)
- Additional benchmarks not aligned with coherence focus

---

## Conclusion

**Current State:**
- 2 working benchmarks (SimpleQA, HaluEval)
- 1 documented limitation (TruthfulQA)
- Strong foundation for expansion

**Recommended Next Steps:**
1. SelfCheckGPT (perfect research fit)
2. HalluLens (modern, comprehensive)
3. Fix minor issues (HaluEval token limit, TruthfulQA tests)

**Expected Outcome:**
- 4-5 solid hallucination benchmarks
- Strong research contribution (coherence measures)
- Modern, well-validated framework
- Multiple publication opportunities

**Time Estimate:**
- SelfCheckGPT: 1-2 days
- HalluLens: 3-5 days
- Minor fixes: 1 day
- Total: ~1-2 weeks for comprehensive benchmark suite

---

## Appendix A: Benchmark Comparison Matrix

| Feature | SimpleQA | HaluEval | TruthfulQA | SelfCheckGPT | HalluLens |
|---------|----------|----------|------------|--------------|-----------|
| **Type** | Factuality | Hallucination | Truthfulness | Consistency | Comprehensive |
| **Method** | AI grading | Classification | MC/Generation | Multi-sample | Dynamic |
| **Chat API** | ✅ | ✅ | ❌ (MC) ✅ (Gen) | ✅ | ✅ |
| **Temp Variation** | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Size** | ~4K | ~35K | ~817 | ~238 | Dynamic |
| **Year** | 2024 | 2023 | 2021 | 2023 | 2025 |
| **Research Value** | Good | Good | Limited | **VERY HIGH** | **HIGH** |
| **Implementation** | ✅ Done | ✅ Done | ⚠️ Limited | 🎯 TODO | 🌟 TODO |

---

*End of Comprehensive Benchmark Assessment*
