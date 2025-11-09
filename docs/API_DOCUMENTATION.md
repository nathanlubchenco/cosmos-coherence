# API Documentation: Question-Variation Coherence

This document provides detailed API documentation for the question-variation coherence implementation.

## Table of Contents

1. [QuestionVariationGenerator](#questionvariationgenerator)
2. [CoherenceIntegration](#coherenceintegration)
3. [AnswerStrategy](#answerstrategy)
4. [SimpleQACoherenceBenchmark](#simpleqacoherencebenchmark)
5. [Helper Methods](#helper-methods)

---

## QuestionVariationGenerator

**Module**: `cosmos_coherence.benchmarks.implementations.question_variation_generator`

### Class: `QuestionVariationGenerator`

Generate paraphrased versions of questions for coherence analysis using temperature-based sampling.

#### Constructor

```python
QuestionVariationGenerator(
    client: OpenAIClient,
    model: str = "gpt-4o-mini",
    temperature_range: Optional[List[float]] = None,
    num_variations: int = 5,
    prompt_template: Optional[str] = None,
    min_edit_distance: int = 5,
    max_retries: int = 2,
    verbose: bool = False
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `client` | `OpenAIClient` | (required) | OpenAI client for API calls |
| `model` | `str` | `"gpt-4o-mini"` | Model to use for variation generation |
| `temperature_range` | `List[float]` | `[0.7, 1.0, 1.3]` | List of temperatures for diversity |
| `num_variations` | `int` | `5` | Number of variations to generate per question |
| `prompt_template` | `str` | (default provided) | Custom prompt template (uses `{question}` placeholder) |
| `min_edit_distance` | `int` | `5` | Minimum character difference for variation acceptance |
| `max_retries` | `int` | `2` | Maximum retry attempts for filtered variations |
| `verbose` | `bool` | `False` | Enable verbose logging for debugging |

**Default Prompt Template:**
```
"Rephrase the following question while preserving its exact meaning: {question}"
```

#### Methods

##### `generate_variations(question: str) -> List[str]`

Generate N paraphrased versions of the question.

**Parameters:**
- `question` (str): Original question to paraphrase

**Returns:**
- `List[str]`: List of N unique, sufficiently different question variations

**Raises:**
- `ValueError`: If unable to generate sufficient valid variations after max retries

**Example:**

```python
from cosmos_coherence.llm.openai_client import OpenAIClient
from cosmos_coherence.benchmarks.implementations.question_variation_generator import QuestionVariationGenerator

client = OpenAIClient(api_key="your-api-key")
generator = QuestionVariationGenerator(
    client=client,
    model="gpt-4o-mini",
    num_variations=5,
    verbose=True
)

question = "What is the capital of France?"
variations = await generator.generate_variations(question)
# Output: [
#     "Which city serves as the capital of France?",
#     "What city is France's capital?",
#     "Name the capital city of France",
#     "France's capital is located in which city?",
#     "What is the main city of France?"
# ]
```

##### `get_statistics() -> Dict[str, int]`

Get generation statistics for the last `generate_variations()` call.

**Returns:**
- `Dict[str, int]`: Dictionary with keys `generation_attempts`, `rejections`, `final_count`

**Example:**

```python
stats = generator.get_statistics()
print(f"Generated {stats['final_count']} variations after {stats['generation_attempts']} attempts")
print(f"Rejected {stats['rejections']} variations due to insufficient difference")
```

---

## CoherenceIntegration

**Module**: `cosmos_coherence.benchmarks.implementations.coherence_integration`

### Class: `CoherenceIntegration`

Integration layer for coherence-based question variation scoring. Adapts SelfCheckGPT coherence variants for question-variation use case.

#### Constructor

```python
CoherenceIntegration(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"gpt-4o-mini"` | OpenAI model name for probability extraction |
| `api_key` | `str` | `None` | OpenAI API key (defaults to `OPENAI_API_KEY` environment variable) |

#### Methods

##### `_init_coherence_variant(variant_name: str) -> Union[SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson]`

Initialize coherence variant by name.

**Parameters:**
- `variant_name` (str): Name of coherence variant (`"shogenji"`, `"fitelson"`, or `"olsson"`)

**Returns:**
- Initialized coherence variant instance

**Raises:**
- `ValueError`: If variant_name is not recognized

**Example:**

```python
from cosmos_coherence.benchmarks.implementations.coherence_integration import CoherenceIntegration

integration = CoherenceIntegration(model="gpt-4o-mini")

# Initialize all three variants
shogenji = integration._init_coherence_variant("shogenji")
fitelson = integration._init_coherence_variant("fitelson")
olsson = integration._init_coherence_variant("olsson")
```

##### `score_question_variations(variant, original_question: str, variations: List[str], verbose: bool = False) -> float`

Score question variations using coherence measure.

**Parameters:**
- `variant`: Initialized coherence variant (from `_init_coherence_variant()`)
- `original_question` (str): Original question to evaluate
- `variations` (List[str]): List of paraphrased question variations
- `verbose` (bool): Enable verbose logging (default: False)

**Returns:**
- `float`: Coherence score in range [0.0, 1.0], where higher = more coherent

**Example:**

```python
original = "What is the capital of France?"
variations = [
    "Which city serves as the capital of France?",
    "What city is France's capital?"
]

shogenji = integration._init_coherence_variant("shogenji")
score = integration.score_question_variations(
    variant=shogenji,
    original_question=original,
    variations=variations,
    verbose=True
)
print(f"Coherence score: {score:.2f}")  # Output: Coherence score: 0.85
```

##### `get_error_rate() -> float`

Get error rate for coherence calculations.

**Returns:**
- `float`: Proportion of evaluations that encountered errors (0.0-1.0)

---

## AnswerStrategy

**Module**: `cosmos_coherence.benchmarks.implementations.answer_strategy`

### Class: `AnswerStrategy`

Strategy for using coherence scores to guide answer selection with different treatment for YES/NO questions vs other question types.

#### Constructor

```python
AnswerStrategy(
    coherence_threshold: float = 0.5,
    abstention_response: str = "I don't know",
    high_coherence_answer_yes: bool = True,
    verbose: bool = False
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `coherence_threshold` | `float` | `0.5` | Threshold for coherence score (0.0-1.0). Scores >= threshold indicate high coherence (confident), scores < threshold indicate low coherence (abstain or opposite polarity) |
| `abstention_response` | `str` | `"I don't know"` | Response to return when abstaining |
| `high_coherence_answer_yes` | `bool` | `True` | For YES/NO questions, whether high coherence → YES. Set `False` to flip polarity mapping. This mapping should be tuned empirically based on experimental results. |
| `verbose` | `bool` | `False` | Enable verbose logging for debugging |

#### Methods

##### `is_yes_no_question(question: str) -> bool`

Detect if a question is a YES/NO question based on pattern matching.

**Parameters:**
- `question` (str): Question to evaluate

**Returns:**
- `bool`: True if question matches YES/NO pattern, False otherwise

**YES/NO Patterns (case-insensitive):**
- `^is\s` - "Is Paris..."
- `^are\s` - "Are cats..."
- `^does\s` - "Does water..."
- `^do\s` - "Do birds..."
- `^can\s` - "Can humans..."
- `^will\s`, `^would\s`, `^should\s`
- `^has\s`, `^have\s`
- `^was\s`, `^were\s`
- `^did\s` - "Did Napoleon..."

**Example:**

```python
from cosmos_coherence.benchmarks.implementations.answer_strategy import AnswerStrategy

strategy = AnswerStrategy()

print(strategy.is_yes_no_question("Is Paris the capital of France?"))  # True
print(strategy.is_yes_no_question("Can humans fly?"))  # True
print(strategy.is_yes_no_question("What is the capital of France?"))  # False
```

##### `coherence_to_yes_no(coherence_score: float) -> str`

Convert coherence score to YES/NO answer for YES/NO questions.

**Parameters:**
- `coherence_score` (float): Coherence score in range [0.0, 1.0]

**Returns:**
- `str`: "YES" or "NO" based on coherence score and polarity mapping

**Logic:**
- If `high_coherence_answer_yes=True`:
  - `coherence_score >= threshold` → "YES"
  - `coherence_score < threshold` → "NO"
- If `high_coherence_answer_yes=False`:
  - `coherence_score >= threshold` → "NO"
  - `coherence_score < threshold` → "YES"

**Example:**

```python
strategy = AnswerStrategy(
    coherence_threshold=0.5,
    high_coherence_answer_yes=True
)

print(strategy.coherence_to_yes_no(0.8))  # "YES" (high coherence)
print(strategy.coherence_to_yes_no(0.3))  # "NO" (low coherence)
```

##### `apply_coherence_threshold(coherence_score: float, model_answer: str) -> str`

Apply threshold-based abstention for non-YES/NO questions.

**Parameters:**
- `coherence_score` (float): Coherence score in range [0.0, 1.0]
- `model_answer` (str): Model's answer to the question

**Returns:**
- `str`: Either `model_answer` (if high coherence) or `abstention_response` (if low coherence)

**Example:**

```python
strategy = AnswerStrategy(
    coherence_threshold=0.5,
    abstention_response="I don't know"
)

# High coherence → return model answer
print(strategy.apply_coherence_threshold(0.8, "Paris"))  # "Paris"

# Low coherence → return abstention
print(strategy.apply_coherence_threshold(0.3, "Paris"))  # "I don't know"
```

##### `apply_strategy(question: str, coherence_score: float, model_answer: Optional[str] = None) -> str`

Apply full answer strategy (detects question type and applies appropriate logic).

**Parameters:**
- `question` (str): Original question
- `coherence_score` (float): Coherence score for question variations
- `model_answer` (str, optional): Model's answer (required for non-YES/NO questions)

**Returns:**
- `str`: Final answer (YES/NO for binary questions, model_answer or abstention for others)

**Example:**

```python
strategy = AnswerStrategy(coherence_threshold=0.5)

# YES/NO question (model_answer not needed)
question1 = "Is Paris the capital of France?"
answer1 = strategy.apply_strategy(question1, coherence_score=0.8)
print(answer1)  # "YES"

# Factual question (model_answer required)
question2 = "What is the capital of France?"
answer2 = strategy.apply_strategy(question2, coherence_score=0.8, model_answer="Paris")
print(answer2)  # "Paris"

# Low coherence factual question
answer3 = strategy.apply_strategy(question2, coherence_score=0.3, model_answer="Paris")
print(answer3)  # "I don't know"
```

##### `get_statistics() -> Dict[str, int]`

Get statistics about answer strategy application.

**Returns:**
- `Dict[str, int]`: Dictionary with keys:
  - `yes_no_count`: Number of YES/NO questions processed
  - `other_question_count`: Number of other questions processed
  - `abstention_count`: Number of abstentions (excluding YES/NO)
  - `confident_answer_count`: Number of confident answers (excluding YES/NO)

---

## SimpleQACoherenceBenchmark

**Module**: `cosmos_coherence.benchmarks.implementations.simpleqa_coherence_benchmark`

### Class: `SimpleQACoherenceBenchmark`

SimpleQA benchmark with coherence-guided abstention. Orchestrates the full evaluation pipeline.

#### Constructor

```python
SimpleQACoherenceBenchmark(
    model: str,
    coherence_variant: str = "shogenji",
    coherence_threshold: float = 0.5,
    num_variations: int = 5,
    temperature_range: Optional[List[float]] = None,
    abstention_response: str = "I don't know",
    high_coherence_answer_yes: bool = True,
    verbose: bool = False,
    **kwargs
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | (required) | Target model to evaluate (e.g., "gpt-4o-mini") |
| `coherence_variant` | `str` | `"shogenji"` | Coherence measure: "shogenji", "fitelson", or "olsson" |
| `coherence_threshold` | `float` | `0.5` | Abstention threshold (0.0-1.0) |
| `num_variations` | `int` | `5` | Number of question paraphrases |
| `temperature_range` | `List[float]` | `[0.7, 1.0, 1.3]` | Temperatures for variation generation |
| `abstention_response` | `str` | `"I don't know"` | Custom text for abstentions |
| `high_coherence_answer_yes` | `bool` | `True` | YES/NO polarity mapping |
| `verbose` | `bool` | `False` | Enable verbose logging |
| `**kwargs` | | | Additional arguments passed to parent `SimpleQABenchmark` |

#### Methods

##### `evaluate_with_coherence(item: Dict) -> Dict`

Evaluate a single question with coherence-guided abstention.

**Parameters:**
- `item` (Dict): SimpleQA dataset item with keys `question`, `best_answer`, `metadata`

**Returns:**
- `Dict`: Evaluation result with structure:

```python
{
    "question": str,                    # Original question
    "ground_truth": str,                # Correct answer
    "question_variations": List[str],   # Generated paraphrases
    "coherence_score": float,           # Coherence score [0.0, 1.0]
    "threshold_met": bool,              # Whether coherence >= threshold
    "question_type": str,               # "yes_no" or "factual"
    "model_answer": str,                # Model's original answer
    "final_answer": str,                # Answer after applying strategy
    "grade": str,                       # "CORRECT", "INCORRECT", or "NOT_ATTEMPTED"
    "is_correct": bool                  # Whether answer is correct
}
```

**Example:**

```python
from cosmos_coherence.benchmarks.implementations.simpleqa_coherence_benchmark import SimpleQACoherenceBenchmark

benchmark = SimpleQACoherenceBenchmark(
    model="gpt-4o-mini",
    coherence_variant="shogenji",
    coherence_threshold=0.5,
    num_variations=5,
    verbose=True
)

item = {
    "question": "What is the capital of France?",
    "best_answer": "Paris",
    "metadata": {}
}

result = await benchmark.evaluate_with_coherence(item)
print(f"Question: {result['question']}")
print(f"Coherence: {result['coherence_score']:.2f}")
print(f"Final answer: {result['final_answer']}")
print(f"Grade: {result['grade']}")
```

##### `evaluate_full_dataset(sample_size: Optional[int] = None, checkpoint_interval: int = 50, resume_from: Optional[Path] = None) -> Dict`

Evaluate full SimpleQA dataset with checkpointing.

**Parameters:**
- `sample_size` (int, optional): Number of questions to evaluate (default: all 4,326)
- `checkpoint_interval` (int): Save progress every N questions (default: 50, set 0 to disable)
- `resume_from` (Path, optional): Path to checkpoint file to resume from

**Returns:**
- `Dict`: Comprehensive evaluation results with keys:
  - `config`: Configuration parameters
  - `metrics`: Aggregated metrics (overall_correct, overall_incorrect, etc.)
  - `baseline_metrics`: Baseline comparison (if run_baseline=True)
  - `coherence_stats`: Coherence score statistics
  - `api_stats`: API call statistics
  - `per_question_results`: List of per-question results

**Example:**

```python
benchmark = SimpleQACoherenceBenchmark(
    model="gpt-4o-mini",
    coherence_variant="shogenji"
)

results = await benchmark.evaluate_full_dataset(
    sample_size=100,
    checkpoint_interval=50
)

print(f"Overall correct: {results['metrics']['overall_correct']:.2%}")
print(f"Overall incorrect: {results['metrics']['overall_incorrect']:.2%}")
print(f"Precision (correct given attempted): {results['metrics']['correct_given_attempted']:.2%}")
print(f"Abstention rate: {results['metrics']['overall_not_attempted']:.2%}")
```

##### `save_checkpoint(results: Dict, checkpoint_path: Path)`

Save evaluation checkpoint for resumption.

**Parameters:**
- `results` (Dict): Current evaluation results
- `checkpoint_path` (Path): Path to save checkpoint JSON

##### `load_checkpoint(checkpoint_path: Path) -> Dict`

Load evaluation checkpoint.

**Parameters:**
- `checkpoint_path` (Path): Path to checkpoint JSON

**Returns:**
- `Dict`: Checkpoint data with partial results and progress information

---

## Helper Methods

### Evaluation Result Structure

The standard result structure used throughout the pipeline:

```python
{
    # Question information
    "question": str,                    # Original question text
    "ground_truth": str,                # Correct answer from dataset
    "metadata": Dict,                   # Dataset metadata

    # Question variations
    "question_variations": List[str],   # Generated paraphrases

    # Coherence scoring
    "coherence_score": float,           # Score in [0.0, 1.0]
    "threshold_met": bool,              # Whether score >= threshold
    "question_type": str,               # "yes_no" or "factual"

    # Answer generation
    "model_answer": Optional[str],      # Model's original answer
    "final_answer": str,                # Answer after strategy applied

    # Grading
    "grade": str,                       # "CORRECT", "INCORRECT", "NOT_ATTEMPTED"
    "is_correct": bool,                 # Whether answer is correct

    # API statistics (optional)
    "api_calls": int,                   # Number of API calls for this question
    "cached_calls": int                 # Number of cache hits
}
```

### Metrics Structure

Aggregated metrics returned by `evaluate_full_dataset()`:

```python
{
    # Standard SimpleQA metrics
    "total_questions": int,
    "correct_count": int,
    "incorrect_count": int,
    "not_attempted_count": int,
    "overall_correct": float,           # correct / total
    "overall_incorrect": float,         # incorrect / total
    "overall_not_attempted": float,     # not_attempted / total
    "correct_given_attempted": float,   # correct / (correct + incorrect)

    # Coherence statistics
    "mean_coherence": float,
    "median_coherence": float,
    "std_coherence": float,
    "abstention_rate": float,           # Proportion abstained

    # API statistics
    "total_calls": int,
    "cached_calls": int,
    "cache_hit_rate": float,
    "estimated_cost_usd": float
}
```

---

## Usage Patterns

### Pattern 1: Basic Evaluation

```python
from cosmos_coherence.benchmarks.implementations.simpleqa_coherence_benchmark import SimpleQACoherenceBenchmark

# Initialize benchmark
benchmark = SimpleQACoherenceBenchmark(
    model="gpt-4o-mini",
    coherence_variant="shogenji",
    coherence_threshold=0.5
)

# Run evaluation
results = await benchmark.evaluate_full_dataset(sample_size=100)

# Access metrics
print(f"Precision: {results['metrics']['correct_given_attempted']:.2%}")
```

### Pattern 2: Custom Configuration

```python
benchmark = SimpleQACoherenceBenchmark(
    model="gpt-4o-mini",
    coherence_variant="fitelson",
    coherence_threshold=0.7,                    # Conservative threshold
    num_variations=10,                          # More variations for better signal
    temperature_range=[0.5, 0.8, 1.0, 1.5],    # Custom temperatures
    abstention_response="I am not confident",   # Custom abstention
    high_coherence_answer_yes=False,           # Flipped polarity for YES/NO
    verbose=True
)
```

### Pattern 3: Checkpoint Management

```python
# Start evaluation with checkpointing
results = await benchmark.evaluate_full_dataset(
    sample_size=500,
    checkpoint_interval=50,  # Save every 50 questions
    resume_from=None         # Start from beginning
)

# If interrupted, resume from checkpoint
results = await benchmark.evaluate_full_dataset(
    resume_from=Path("results/checkpoint_250.json")
)
```

### Pattern 4: Comparing Variants

```python
variants = ["shogenji", "fitelson", "olsson"]
results = {}

for variant in variants:
    benchmark = SimpleQACoherenceBenchmark(
        model="gpt-4o-mini",
        coherence_variant=variant
    )
    results[variant] = await benchmark.evaluate_full_dataset(sample_size=100)

# Compare performance
for variant, result in results.items():
    print(f"{variant}: {result['metrics']['correct_given_attempted']:.2%}")
```

---

## Error Handling

All components implement robust error handling:

1. **API Errors**: Retry with exponential backoff (handled by OpenAIClient)
2. **Variation Generation Failures**: Retry up to `max_retries`, skip question if all fail
3. **Coherence Calculation Errors**: Return fallback score (0.5) and log error
4. **Checkpoint Corruption**: Validate JSON on load, restart if invalid
5. **Partial Evaluation**: Continue processing remaining questions if individual question fails

**Example Error Handling:**

```python
try:
    results = await benchmark.evaluate_with_coherence(item)
except Exception as e:
    logger.error(f"Evaluation failed for question: {item['question']}")
    logger.error(f"Error: {str(e)}")
    # Continue with next question
```

---

## Performance Considerations

1. **Parallelization**: Use `workers` parameter for concurrent API calls
2. **Caching**: Both variation generation and coherence calculation use caching
3. **Checkpointing**: Save progress frequently to avoid re-processing on interruption
4. **Sample Size**: Start with small samples (100-500) before full evaluation
5. **Model Selection**: Use `gpt-4o-mini` for cost efficiency (~10x cheaper than gpt-4)

---

## See Also

- [README.md](../README.md) - Feature overview and usage examples
- [VALIDATION_GUIDE.md](VALIDATION_GUIDE.md) - Testing and validation
- [SELFCHECKGPT_INTEGRATION.md](SELFCHECKGPT_INTEGRATION.md) - Setup instructions
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions
