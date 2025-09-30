# Technical Specification

This is the technical specification for the spec detailed in @.agent-os/specs/2025-09-29-truthfulqa-benchmark/spec.md

## Technical Requirements

### Dataset Loading and Processing

- **Dataset Source**: Load from HuggingFace `truthful_qa` dataset using existing `HuggingFaceDatasetLoader`
- **Dataset Structure**: 817 questions with fields: `question`, `mc1_targets`, `mc2_targets`, `category`, `source`
- **MC1 Format**: Single correct answer among 4-5 choices (dict with `choices` and `labels` arrays where label=0 is correct)
- **MC2 Format**: Multiple true/false answers (dict with `choices` and `labels` arrays where label=1 indicates true statements)
- **Category Support**: Extract and preserve 38 category labels for per-category reporting
- **Sample Size**: Support `--sample-size` parameter to limit questions for testing (similar to existing benchmarks)

### Evaluation Methodology

#### MC1 Evaluation (Single Correct Answer)
- **Method**: Log-probability comparison across all answer choices
- **Implementation**: For each choice, compute log P(choice|question) using OpenAI API logprobs
- **Scoring**: Model's answer = choice with highest log probability; correct if matches label=0 choice
- **Metric**: Simple accuracy = correct answers / total questions
- **API Requirements**: Use `logprobs=True` parameter in completion requests

#### MC2 Evaluation (Multiple True/False Answers)
- **Method**: Normalized probability distribution over labeled answers
- **Implementation**:
  - Compute log P(choice|question) for all choices
  - Convert log probabilities to probabilities: P = exp(log_prob)
  - Normalize: P_norm = P / sum(all P)
  - Sum probabilities for correct answers (label=1) and incorrect answers (label=0)
- **Scoring**: MC2 = sum(P_norm for correct) / [sum(P_norm for correct) + sum(P_norm for incorrect)]
- **Metric**: Mean MC2 score across all questions (0-1 scale, higher = better)
- **Edge Cases**: Handle questions with varying numbers of correct/incorrect answers

### Model Integration

- **Supported Models**: OpenAI models (gpt-3.5-turbo, gpt-4, gpt-4-turbo) via existing `OpenAIClient`
- **API Configuration**: Temperature 0.0 for deterministic evaluation (matches benchmark standard)
- **Logprobs Requirement**: Request logprobs for each answer choice to enable probability-based scoring
- **Prompt Format**: Question followed by answer choice, formatted as: "{question}\n\nAnswer: {choice}"
- **Max Tokens**: Set to 1 (only need to evaluate the probability of provided choices, not generate)

### CLI Interface

- **Command Structure**: `poetry run python -m cosmos_coherence.benchmarks.truthfulqa_cli run [OPTIONS]`
- **Required Options**:
  - `--model, -m`: Model identifier (default: gpt-4)
  - `--sample-size, -n`: Number of questions to evaluate (default: all 817)
  - `--output, -o`: Path to save results JSON
- **Optional Flags**:
  - `--temperature`: Temperature for generation (default: 0.0)
  - `--cache/--no-cache`: Enable/disable response caching (default: enabled)
  - `--progress/--no-progress`: Show/hide progress bar (default: show)
  - `--verbose, -v`: Display detailed per-question results
  - `--category, -c`: Filter by specific category name

### Results Storage and Reporting

- **Per-Question Results**: JSON objects with fields:
  - `question`: Original question text
  - `category`: Question category
  - `mc1_correct`: Boolean for MC1 evaluation
  - `mc1_predicted_choice`: Model's selected answer
  - `mc1_correct_choice`: Ground truth answer
  - `mc2_score`: Normalized probability score (0-1)
  - `mc2_correct_probs`: Sum of probabilities for correct answers
  - `mc2_incorrect_probs`: Sum of probabilities for incorrect answers

- **Aggregate Metrics**: JSON object with:
  - `mc1_accuracy`: Overall MC1 accuracy
  - `mc2_score`: Mean MC2 score across all questions
  - `total_questions`: Number of questions evaluated
  - `category_breakdown`: Dict mapping category → {mc1_accuracy, mc2_score, count}
  - `model`: Model identifier
  - `temperature`: Temperature used
  - `timestamp`: Evaluation timestamp

- **Display Format**: Rich table showing:
  - Overall MC1 accuracy and MC2 score
  - Top 5 and bottom 5 categories by performance
  - Comparison to published baselines (GPT-3.5: MC2 ~0.47, GPT-4: MC2 ~0.59)

### Caching Integration

- **Cache Structure**: Use existing `OpenAIClient` caching system with persistent storage
- **Cache Key**: Combination of (model, prompt, temperature, max_tokens)
- **Cache Location**: `~/.cache/cosmos_coherence/truthfulqa/{model_name}_cache.json`
- **Cache Behavior**: Automatically reuse cached responses for identical prompts across runs
- **Cache Control**: CLI `--no-cache` flag to bypass cache for fresh evaluations

### Validation and Baseline Comparison

- **Baseline Metrics** (from HuggingFace Open LLM Leaderboard):
  - GPT-3.5-turbo: MC2 ~0.47 (47%)
  - GPT-4: MC2 ~0.59 (59%)
  - GPT-4-turbo: MC2 ~0.62 (62%)
- **Success Criteria**: Implementation should reproduce within ±5% of published baselines
- **Validation Approach**:
  1. Run on small sample (50 questions) to verify mechanics
  2. Run full dataset (817 questions) and compare to baselines
  3. Check category breakdowns for expected patterns (e.g., conspiracies and misconceptions are harder)

### Error Handling

- **API Errors**: Retry with exponential backoff (existing OpenAI client behavior)
- **Malformed Data**: Skip questions with invalid mc1/mc2 target structures and log warnings
- **Missing Logprobs**: Raise error if API doesn't return logprobs (required for evaluation)
- **Rate Limits**: Respect OpenAI rate limits with automatic throttling

## Implementation Classes

### TruthfulQABenchmark
- **Inherits**: `HuggingFaceEnabledBenchmark`
- **Key Methods**:
  - `get_prompt(item: TruthfulQAItem) -> str`: Format question + answer choice for API
  - `evaluate_mc1(item: TruthfulQAItem, logprobs: Dict) -> bool`: Compute MC1 accuracy
  - `evaluate_mc2(item: TruthfulQAItem, logprobs: Dict) -> float`: Compute MC2 score
  - `calculate_metrics(results: List[Dict]) -> Dict`: Aggregate results with category breakdowns

### TruthfulQAItem
- **Already Defined**: Existing Pydantic model in `benchmarks/models/datasets.py`
- **Fields**: `question`, `mc1_targets`, `mc2_targets`, `category`, `source`, etc.
- **Validation**: Ensure mc1_targets and mc2_targets have proper structure

### TruthfulQACLI
- **Module**: `benchmarks/truthfulqa_cli.py`
- **Commands**: `run`, `compare` (compare two result files), `baseline` (compare to published baselines)
- **Uses**: Typer for CLI, Rich for formatted output tables

## External Dependencies

No new external dependencies required. Implementation uses existing project dependencies:
- `datasets` (HuggingFace) - already installed for dataset loading
- `tiktoken` - already installed for token counting
- `typer` - already installed for CLI
- `rich` - already installed for formatted output
- `openai` - already installed for API access

All dependencies are already in the project's `pyproject.toml`.
