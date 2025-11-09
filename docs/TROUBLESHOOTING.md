# Troubleshooting Guide: Question-Variation Coherence

This guide addresses common issues encountered when using the question-variation coherence feature and provides solutions.

## Table of Contents

1. [High Abstention Rate](#high-abstention-rate)
2. [No Improvement Over Baseline](#no-improvement-over-baseline)
3. [API Cost Overruns](#api-cost-overruns)
4. [Poor Variation Quality](#poor-variation-quality)
5. [Import Errors](#import-errors)
6. [OpenAI API Issues](#openai-api-issues)
7. [Checkpoint Issues](#checkpoint-issues)
8. [Performance Problems](#performance-problems)
9. [Coherence Score Issues](#coherence-score-issues)
10. [YES/NO Question Detection Issues](#yesno-question-detection-issues)

---

## High Abstention Rate

### Symptom

Most or all questions are abstained (NOT_ATTEMPTED), resulting in very low recall.

```
Overall correct: 5%
Overall incorrect: 0%
Overall not attempted: 95%
```

### Diagnosis

1. **Coherence threshold too high**: All coherence scores are below threshold
2. **Coherence scores uniformly low**: Variation quality issues causing low coherence
3. **Inverted interpretation**: Using wrong coherence measure or threshold logic

### Solutions

#### Solution 1: Lower Threshold

```bash
# Try decreasing threshold from 0.5 to 0.3
cosmos-coherence simpleqa run-with-coherence \
    --model gpt-4o-mini \
    --coherence-threshold 0.3 \
    --sample-size 100 \
    --output results/lower_threshold.json
```

#### Solution 2: Check Coherence Score Distribution

```python
import json

with open("results/results.json") as f:
    results = json.load(f)

scores = [r["coherence_score"] for r in results["per_question_results"]]
print(f"Mean coherence: {sum(scores) / len(scores):.2f}")
print(f"Min coherence: {min(scores):.2f}")
print(f"Max coherence: {max(scores):.2f}")

# If all scores are < 0.5, threshold is too high
# If all scores are near 0.0 or 1.0, check variation quality
```

#### Solution 3: Verify Variation Quality

Enable verbose mode to inspect generated variations:

```bash
cosmos-coherence simpleqa run-with-coherence \
    --model gpt-4o-mini \
    --verbose \
    --sample-size 10 \
    --output results/debug.json
```

Check log output for:
- Are variations meaningfully different from original?
- Are variations preserving question meaning?
- Are variations being rejected due to similarity?

#### Solution 4: Tune Threshold Systematically

Run multiple thresholds to find optimal:

```bash
for threshold in 0.2 0.3 0.4 0.5 0.6 0.7; do
    cosmos-coherence simpleqa run-with-coherence \
        --model gpt-4o-mini \
        --coherence-threshold $threshold \
        --sample-size 100 \
        --output results/threshold_${threshold}.json
done

# Compare abstention rates
for f in results/threshold_*.json; do
    echo "$f:"
    jq '.metrics.overall_not_attempted' $f
done
```

**Expected Behavior:**
- Threshold 0.3: ~5-15% abstention (aggressive)
- Threshold 0.5: ~10-30% abstention (balanced)
- Threshold 0.7: ~30-50% abstention (conservative)

---

## No Improvement Over Baseline

### Symptom

Coherence-guided evaluation shows no improvement in precision over baseline, or worse performance.

```
Baseline precision: 75%
Coherence precision: 72%  # WORSE than baseline
```

### Diagnosis

1. **Coherence not correlated with correctness**: Approach may not work for this model/dataset
2. **Threshold poorly tuned**: Abstracting on wrong questions
3. **Variation quality poor**: Coherence signal is noisy
4. **YES/NO polarity inverted**: Wrong polarity mapping for binary questions

### Solutions

#### Solution 1: Check Coherence-Correctness Correlation

```python
import json
import numpy as np
from scipy.stats import pearsonr

with open("results/results.json") as f:
    results = json.load(f)

# Extract coherence scores and correctness
coherence = []
correctness = []
for r in results["per_question_results"]:
    if r["grade"] != "NOT_ATTEMPTED":  # Only attempted questions
        coherence.append(r["coherence_score"])
        correctness.append(1 if r["grade"] == "CORRECT" else 0)

# Calculate correlation
correlation, p_value = pearsonr(coherence, correctness)
print(f"Correlation: {correlation:.3f} (p={p_value:.4f})")

# Expected: correlation > 0.3 for approach to work
# If correlation < 0.1, approach may not be effective for this model
```

#### Solution 2: Analyze Abstention Patterns

```python
# Check if abstracting on correct vs incorrect questions
abstained = [r for r in results["per_question_results"] if r["grade"] == "NOT_ATTEMPTED"]
attempted_correct = [r for r in results["per_question_results"] if r["grade"] == "CORRECT"]
attempted_incorrect = [r for r in results["per_question_results"] if r["grade"] == "INCORRECT"]

print(f"Mean coherence (abstained): {np.mean([r['coherence_score'] for r in abstained]):.2f}")
print(f"Mean coherence (correct): {np.mean([r['coherence_score'] for r in attempted_correct]):.2f}")
print(f"Mean coherence (incorrect): {np.mean([r['coherence_score'] for r in attempted_incorrect]):.2f}")

# Expected: abstained < incorrect < correct
# If pattern reversed, threshold logic may be inverted
```

#### Solution 3: Try Different Coherence Variants

Shogenji, Fitelson, and Olsson may perform differently:

```bash
for variant in shogenji fitelson olsson; do
    cosmos-coherence simpleqa run-with-coherence \
        --model gpt-4o-mini \
        --coherence-variant $variant \
        --sample-size 100 \
        --output results/variant_${variant}.json
done

# Compare precision
for f in results/variant_*.json; do
    echo "$f:"
    jq '.metrics.correct_given_attempted' $f
done
```

#### Solution 4: Check YES/NO Polarity

If many YES/NO questions are wrong, polarity may be inverted:

```bash
# Try flipped polarity
cosmos-coherence simpleqa run-with-coherence \
    --model gpt-4o-mini \
    --high-coherence-answer-no \  # Flip: high coherence → NO
    --sample-size 100 \
    --output results/flipped_polarity.json
```

**When to Abandon Approach:**

If after trying all solutions:
- Correlation remains < 0.1
- No coherence variant shows improvement
- Precision does not improve over baseline

The approach may not be effective for this specific model or dataset. Document negative results for research value.

---

## API Cost Overruns

### Symptom

API costs significantly higher than expected or budget exceeded.

### Diagnosis

1. **Underestimated API calls**: Actual calls exceed estimated ~18 per question
2. **Cache not working**: Low cache hit rate increases costs
3. **Sample size too large**: Running full dataset before validating approach
4. **Expensive model**: Using gpt-4-turbo instead of gpt-4o-mini

### Solutions

#### Solution 1: Estimate Costs Before Running

```python
# Use built-in cost estimation
from cosmos_coherence.benchmarks.implementations.simpleqa_coherence_benchmark import SimpleQACoherenceBenchmark

benchmark = SimpleQACoherenceBenchmark(
    model="gpt-4o-mini",
    coherence_variant="shogenji",
    num_variations=5
)

# Estimate for 100 questions
estimate = benchmark.estimate_api_cost(num_questions=100)
print(f"Estimated cost for 100 questions: ${estimate['total_cost_usd']:.2f}")
print(f"Estimated calls: {estimate['total_calls']}")

# Decide whether to proceed
if estimate['total_cost_usd'] > budget_limit:
    print("Cost exceeds budget, reduce sample size")
```

#### Solution 2: Start with Small Samples

```bash
# Test with 10 questions first (~$0.15)
cosmos-coherence simpleqa run-with-coherence \
    --model gpt-4o-mini \
    --sample-size 10 \
    --output results/test_10.json

# If results promising, scale to 100 (~$1.50)
cosmos-coherence simpleqa run-with-coherence \
    --model gpt-4o-mini \
    --sample-size 100 \
    --output results/test_100.json

# Only run full dataset after validation
```

#### Solution 3: Use gpt-4o-mini

```bash
# gpt-4o-mini is ~10x cheaper than gpt-4-turbo
cosmos-coherence simpleqa run-with-coherence \
    --model gpt-4o-mini \  # NOT gpt-4-turbo
    --sample-size 100
```

**Cost Comparison:**
- gpt-4o-mini: $0.15 per 100 questions, $5-6 for full dataset
- gpt-4-turbo: $1.50 per 100 questions, $40-45 for full dataset

#### Solution 4: Check Cache Hit Rate

```bash
# Run evaluation and check cache statistics
cosmos-coherence simpleqa run-with-coherence \
    --model gpt-4o-mini \
    --sample-size 100 \
    --output results/cache_test.json

# Check cache performance
jq '.api_stats' results/cache_test.json
```

Expected cache hit rate: 20-30%

If cache hit rate < 10%:
- Caching may not be working
- Questions may be too diverse for cache benefit
- Check OpenAI client cache configuration

#### Solution 5: Reduce Number of Variations

```bash
# Reduce from 5 to 3 variations (saves ~6 API calls per question)
cosmos-coherence simpleqa run-with-coherence \
    --model gpt-4o-mini \
    --num-variations 3 \
    --sample-size 100
```

**Cost Impact:**
- 5 variations: ~18 calls/question = $5-6 for full dataset
- 3 variations: ~13 calls/question = $3-4 for full dataset

---

## Poor Variation Quality

### Symptom

Generated question variations are:
- Too similar to original (low diversity)
- Too different (semantic drift)
- Nonsensical or grammatically incorrect

### Diagnosis

1. **Temperature range too narrow**: All variations similar
2. **Temperature range too wide**: Variations drift from original meaning
3. **Prompt template unclear**: Model not understanding paraphrase task
4. **Model limitations**: Model struggles with paraphrasing

### Solutions

#### Solution 1: Adjust Temperature Range

For more diversity:

```bash
# Wider temperature range
cosmos-coherence simpleqa run-with-coherence \
    --model gpt-4o-mini \
    --temperature-range 0.5,0.8,1.0,1.3,1.6 \
    --sample-size 10 \
    --verbose
```

For more similarity:

```bash
# Narrower temperature range
cosmos-coherence simpleqa run-with-coherence \
    --model gpt-4o-mini \
    --temperature-range 0.7,0.9,1.1 \
    --sample-size 10 \
    --verbose
```

#### Solution 2: Modify Prompt Template

```python
from cosmos_coherence.benchmarks.implementations.question_variation_generator import QuestionVariationGenerator
from cosmos_coherence.llm.openai_client import OpenAIClient

client = OpenAIClient()
generator = QuestionVariationGenerator(
    client=client,
    prompt_template="Rewrite this question using different words but keeping the exact same meaning: {question}"
)

# Test on sample questions
test_questions = [
    "What is the capital of France?",
    "Can humans fly?",
    "When was Einstein born?"
]

for q in test_questions:
    variations = await generator.generate_variations(q)
    print(f"\nOriginal: {q}")
    for i, v in enumerate(variations, 1):
        print(f"  {i}. {v}")
```

#### Solution 3: Increase min_edit_distance

```python
# Reject variations too similar to original
generator = QuestionVariationGenerator(
    client=client,
    min_edit_distance=10,  # Increased from default 5
    num_variations=5
)
```

#### Solution 4: Manual Quality Check

```bash
# Run with verbose logging and manual inspection
cosmos-coherence simpleqa run-with-coherence \
    --model gpt-4o-mini \
    --sample-size 20 \
    --verbose \
    --output results/quality_check.json 2>&1 | tee variation_log.txt

# Manually review variation_log.txt for quality issues
```

**Quality Criteria:**
- **Good variation**: "What is France's capital?" (different structure, same meaning)
- **Too similar**: "What is the capital of France" (minimal change)
- **Semantic drift**: "What are the major cities in France?" (different question)
- **Nonsensical**: "What capital is France of the?" (grammatically incorrect)

---

## Import Errors

### Symptom

```
ModuleNotFoundError: No module named 'selfcheckgpt'
ImportError: cannot import name 'SelfCheckShogenji' from 'selfcheckgpt'
```

### Solutions

#### Solution 1: Install SelfCheckGPT

```bash
# With Poetry (recommended)
cd /path/to/cosmos-coherence
poetry install

# Or install selfcheckgpt separately
poetry add selfcheckgpt

# Without Poetry
pip install selfcheckgpt
```

#### Solution 2: Development Mode Installation

If developing coherence variants:

```bash
# Clone selfcheckgpt if not present
git clone https://github.com/potsawee/selfcheckgpt.git /path/to/selfcheckgpt

# Install in editable mode
cd /path/to/cosmos-coherence
poetry add -e /path/to/selfcheckgpt
```

#### Solution 3: Verify Installation

```bash
python -c "from selfcheckgpt import SelfCheckShogenji; print('Success')"
python -c "from selfcheckgpt.modeling_coherence_api import CoherenceAPIClient; print('Success')"
```

#### Solution 4: Check Python Environment

```bash
# Ensure using correct environment
poetry env info
poetry run python -c "import selfcheckgpt; print(selfcheckgpt.__file__)"
```

See [SELFCHECKGPT_INTEGRATION.md](SELFCHECKGPT_INTEGRATION.md) for detailed installation instructions.

---

## OpenAI API Issues

### Symptom

```
openai.error.AuthenticationError: Incorrect API key provided
openai.error.RateLimitError: Rate limit exceeded
openai.error.APIError: The server had an error while processing your request
```

### Solutions

#### Solution 1: API Key Not Set

```bash
# Set in environment
export OPENAI_API_KEY="your-api-key-here"

# Or in .env file
echo "OPENAI_API_KEY=your-api-key-here" >> .env

# Verify
echo $OPENAI_API_KEY
```

#### Solution 2: Rate Limit Exceeded

```bash
# Reduce parallel workers
cosmos-coherence simpleqa run-with-coherence \
    --model gpt-4o-mini \
    --workers 1 \  # Reduced from default 4
    --sample-size 100
```

Or wait and retry:

```bash
# Use exponential backoff (automatic in OpenAIClient)
# If errors persist, reduce workers or add delays
```

#### Solution 3: Insufficient Quota

Check quota at: https://platform.openai.com/account/billing

Solutions:
- Add credits to account
- Use smaller sample sizes
- Use cheaper model (gpt-4o-mini)

#### Solution 4: Server Errors

```bash
# Retry with checkpointing
cosmos-coherence simpleqa run-with-coherence \
    --model gpt-4o-mini \
    --checkpoint-interval 10 \  # Frequent checkpoints
    --sample-size 100

# If interrupted by server error, resume
cosmos-coherence simpleqa run-with-coherence \
    --resume-from results/checkpoint.json
```

---

## Checkpoint Issues

### Symptom

```
Failed to load checkpoint: JSONDecodeError
Checkpoint corrupted or missing progress information
Resume not continuing from correct position
```

### Solutions

#### Solution 1: Validate Checkpoint

```bash
# Check if valid JSON
jq '.' results/checkpoint.json

# Check structure
jq '.progress' results/checkpoint.json
```

Expected structure:
```json
{
  "config": {...},
  "progress": {
    "questions_completed": 50,
    "last_index": 49,
    "timestamp": "..."
  },
  "partial_results": [...]
}
```

#### Solution 2: Checkpoint Corruption

If checkpoint is corrupted:

```bash
# Restart from beginning
cosmos-coherence simpleqa run-with-coherence \
    --model gpt-4o-mini \
    --sample-size 100 \
    --output results/restart.json
    # Do NOT use --resume-from
```

#### Solution 3: More Frequent Checkpoints

```bash
# Save every 10 questions instead of 50
cosmos-coherence simpleqa run-with-coherence \
    --model gpt-4o-mini \
    --checkpoint-interval 10 \
    --sample-size 100
```

#### Solution 4: Manual Checkpoint Recovery

```python
import json

# Load partial checkpoint
with open("results/checkpoint.json") as f:
    checkpoint = json.load(f)

# Check progress
print(f"Questions completed: {checkpoint['progress']['questions_completed']}")
print(f"Last index: {checkpoint['progress']['last_index']}")

# Extract partial results
partial_results = checkpoint["partial_results"]
print(f"Partial results count: {len(partial_results)}")

# Save to new file and resume
with open("results/recovered_checkpoint.json", "w") as f:
    json.dump(checkpoint, f, indent=2)
```

---

## Performance Problems

### Symptom

Evaluation is very slow, taking much longer than expected.

### Diagnosis

1. **Sequential processing**: Workers set to 1
2. **API latency**: Network issues or OpenAI slowness
3. **Large variation count**: Generating many variations per question
4. **Cache misses**: Low cache hit rate increases API calls

### Solutions

#### Solution 1: Increase Parallel Workers

```bash
# Use more workers for concurrent API calls
cosmos-coherence simpleqa run-with-coherence \
    --model gpt-4o-mini \
    --workers 8 \  # Increased from default 4
    --sample-size 100
```

**Note**: Don't exceed rate limits. If you hit rate limits, reduce workers.

#### Solution 2: Monitor Progress

```bash
# Use verbose mode to see progress
cosmos-coherence simpleqa run-with-coherence \
    --model gpt-4o-mini \
    --verbose \
    --sample-size 100
```

#### Solution 3: Profile Performance

```python
import time
from cosmos_coherence.benchmarks.implementations.simpleqa_coherence_benchmark import SimpleQACoherenceBenchmark

benchmark = SimpleQACoherenceBenchmark(model="gpt-4o-mini")

# Time single question
start = time.time()
result = await benchmark.evaluate_with_coherence(test_item)
elapsed = time.time() - start

print(f"Time per question: {elapsed:.2f}s")
print(f"Estimated time for 100 questions: {elapsed * 100 / 60:.1f} minutes")
```

**Expected Performance:**
- With 4 workers: ~3-5 seconds per question
- 100 questions: ~5-8 minutes
- 4,326 questions (full dataset): ~4-6 hours

---

## Coherence Score Issues

### Symptom

All coherence scores are the same (e.g., all 0.0, all 1.0, or all 0.5)

### Diagnosis

1. **Coherence calculation failing**: All returning fallback score
2. **Normalization issue**: Scores not properly normalized
3. **Variation quality**: All variations identical or all completely different

### Solutions

#### Solution 1: Check for Errors

```bash
# Run with verbose logging
cosmos-coherence simpleqa run-with-coherence \
    --model gpt-4o-mini \
    --verbose \
    --sample-size 10 \
    --output results/debug.json 2>&1 | tee coherence_debug.txt

# Check for error messages in log
grep -i "error\|warning\|fail" coherence_debug.txt
```

#### Solution 2: Test Coherence Calculation Directly

```python
from cosmos_coherence.benchmarks.implementations.coherence_integration import CoherenceIntegration

integration = CoherenceIntegration(model="gpt-4o-mini")
shogenji = integration._init_coherence_variant("shogenji")

# Test on known example
original = "What is the capital of France?"
variations = [
    "Which city is the capital of France?",
    "What city serves as France's capital?"
]

score = integration.score_question_variations(
    shogenji, original, variations, verbose=True
)
print(f"Coherence score: {score:.2f}")

# Expected: score between 0.0 and 1.0, likely 0.6-0.9 for good paraphrases
```

#### Solution 3: Try Different Coherence Variant

```bash
# If Shogenji fails, try Olsson
cosmos-coherence simpleqa run-with-coherence \
    --model gpt-4o-mini \
    --coherence-variant olsson \
    --sample-size 10 \
    --verbose
```

#### Solution 4: Check OpenAI API Key

Coherence variants require OpenAI API access for probability extraction:

```bash
# Verify API key is set and valid
echo $OPENAI_API_KEY

# Test API access
python -c "import openai; openai.api_key='$OPENAI_API_KEY'; print(openai.Model.list())"
```

---

## YES/NO Question Detection Issues

### Symptom

Questions that should be detected as YES/NO are treated as factual questions, or vice versa.

### Diagnosis

1. **Pattern matching limitations**: Question structure not in predefined patterns
2. **Edge cases**: Questions like "What is... true or false?"
3. **Metadata missing**: SimpleQA metadata not indicating question type

### Solutions

#### Solution 1: Check Detection

```python
from cosmos_coherence.benchmarks.implementations.answer_strategy import AnswerStrategy

strategy = AnswerStrategy()

# Test questions
test_questions = [
    "Is Paris the capital of France?",           # YES/NO
    "Can humans fly?",                           # YES/NO
    "What is the capital of France?",            # Factual
    "Does water boil at 100°C?",                 # YES/NO
    "Which city is the capital of France?",      # Factual
]

for q in test_questions:
    is_yes_no = strategy.is_yes_no_question(q)
    print(f"{q} -> {'YES/NO' if is_yes_no else 'Factual'}")
```

#### Solution 2: Add Custom Patterns

If you have domain-specific YES/NO patterns:

```python
from cosmos_coherence.benchmarks.implementations.answer_strategy import AnswerStrategy

# Extend YES_NO_PATTERNS
custom_patterns = AnswerStrategy.YES_NO_PATTERNS + [
    r"^could\s",       # "Could it be..."
    r"^might\s",       # "Might this..."
    r"^may\s",         # "May we..."
]

# Create custom strategy (requires code modification)
# Or log false negatives for analysis
```

#### Solution 3: Manual Override

If you know which questions should be YES/NO:

```python
# Pre-process dataset to add metadata
dataset_with_types = []
for item in dataset:
    if is_yes_no_question_custom_logic(item["question"]):
        item["metadata"]["question_type"] = "yes_no"
    dataset_with_types.append(item)

# Modify benchmark to respect metadata (requires code change)
```

#### Solution 4: Disable YES/NO Special Handling

If YES/NO handling causes issues:

```python
# Treat all questions uniformly (threshold-based abstention for all)
benchmark = SimpleQACoherenceBenchmark(
    model="gpt-4o-mini",
    # Disable YES/NO special handling by setting threshold very high/low
    # Or modify code to skip YES/NO detection
)
```

**Future Enhancement**: Add metadata-based detection in SimpleQA dataset loading.

---

## Getting Help

If issues persist after trying these solutions:

1. **Check logs**: Review verbose output and error messages
2. **Minimal reproducible example**: Isolate the issue with smallest possible test case
3. **Open GitHub issue**: Include:
   - Error message or unexpected behavior
   - Configuration used
   - Sample data (if applicable)
   - Steps to reproduce
4. **Check documentation**: Review [API_DOCUMENTATION.md](API_DOCUMENTATION.md) and [VALIDATION_GUIDE.md](VALIDATION_GUIDE.md)

---

## Debugging Tools

### Enable Verbose Logging

```bash
cosmos-coherence simpleqa run-with-coherence \
    --verbose \
    --sample-size 10 \
    --output results/debug.json 2>&1 | tee debug_log.txt
```

### Inspect Results

```bash
# View metrics
jq '.metrics' results/results.json

# View coherence statistics
jq '.coherence_stats' results/results.json

# View per-question results
jq '.per_question_results[] | select(.grade == "INCORRECT")' results/results.json

# Find low coherence questions
jq '.per_question_results[] | select(.coherence_score < 0.3)' results/results.json
```

### Monitor API Calls

```bash
# Check API statistics
jq '.api_stats' results/results.json

# Calculate actual cost
jq '.api_stats.total_calls, .api_stats.estimated_cost_usd' results/results.json
```

---

## Common Error Messages

### "Unable to generate sufficient valid variations after max retries"

**Cause**: Variation generator couldn't produce enough diverse paraphrases

**Solution**: Increase `max_retries` or reduce `min_edit_distance`:

```python
generator = QuestionVariationGenerator(
    client=client,
    max_retries=5,           # Increased from 2
    min_edit_distance=3      # Reduced from 5
)
```

### "Unknown coherence variant: xyz"

**Cause**: Invalid variant name provided

**Solution**: Use one of: `"shogenji"`, `"fitelson"`, `"olsson"`

### "OpenAI API key not found"

**Cause**: `OPENAI_API_KEY` environment variable not set

**Solution**: Set environment variable or pass `api_key` parameter:

```bash
export OPENAI_API_KEY="your-key-here"
```

---

## See Also

- [README.md](../README.md) - Feature overview
- [API_DOCUMENTATION.md](API_DOCUMENTATION.md) - Detailed API reference
- [VALIDATION_GUIDE.md](VALIDATION_GUIDE.md) - Testing and validation
- [SELFCHECKGPT_INTEGRATION.md](SELFCHECKGPT_INTEGRATION.md) - Integration setup
