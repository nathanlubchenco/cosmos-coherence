# Technical Specification

This is the technical specification for the spec detailed in @.agent-os/specs/2025-09-09-faithbench-implementation/spec.md

> Created: 2025-09-09
> Version: 1.0.0

## Research Foundation

### Primary References
- **Paper:** [FaithBench: A Diverse Hallucination Benchmark](https://arxiv.org/abs/2410.13210) (arXiv:2410.13210)
- **Repository:** https://github.com/vectara/FaithBench
- **Authors:** Forrest Sheng Bao et al., October 2024
- **See:** @.agent-os/specs/2025-09-09-faithbench-implementation/sub-specs/research-references.md for detailed methodology

## Technical Requirements

### Dataset Management
- Implement `FaithBenchDatasetLoader` class extending `HuggingFaceDatasetLoader`
- Load from official FaithBench repository: `data_for_release/batch_{batch_id}.json`
- Support FaithBench's 4-level annotation taxonomy:
  - Consistent: Factually accurate
  - Questionable: Gray area, potentially subjective
  - Benign: Incorrect but harmless hallucination
  - Hallucinated: Clear factual errors
- Parse dataset structure per repository format:
  ```python
  {
    "sample_id": str,
    "source": str,  # Original text (106-380 words typical)
    "summary": str,  # Generated summary
    "annotations": List[Dict],  # Human annotations with spans
    "metadata": Dict  # Summarizer and detector info
  }
  ```
- Implement entropy-based sample selection for "challenging" cases
- Support dataset caching with configurable cache directory

### Data Models
- Create `FaithBenchItem` dataclass inheriting from `BaseDatasetItem`
- Required fields matching paper structure:
  - `sample_id`: Unique identifier
  - `source`: Original text to summarize
  - `summary`: Generated summary
  - `annotation_label`: Enum of [consistent, questionable, benign, hallucinated]
  - `annotation_spans`: List of problematic text spans
  - `annotation_justification`: Human annotator explanation
  - `detector_predictions`: Dict of model predictions
  - `entropy_score`: Float measuring detector disagreement
- Implement validation for data integrity and format consistency
- Support serialization/deserialization to/from JSON format

### Evaluation Pipeline
- Extend `BaseBenchmark` class for FaithBench-specific logic
- Implement prompt templates from paper methodology:
  ```python
  # Summarization prompt
  "Summarize the following text in 2-3 sentences:\n\n{source_text}\n\nSummary:"

  # Hallucination detection prompt
  "Given the source text and summary below, determine if the summary contains hallucinations.\n\nSource: {source_text}\n\nSummary: {summary}\n\nClassification (consistent/questionable/benign/hallucinated):\nJustification:"
  ```
- Support summarization task (not Q&A) as per paper
- Handle batch processing with configurable batch sizes
- Implement checkpoint/resume functionality for long-running evaluations
- Add timeout handling for individual API calls

### Model Configurations (OpenAI Models)

#### Standard Models (Support Temperature Variation)

##### GPT-4-Turbo Configuration
```python
{
    "model": "gpt-4-turbo",
    "temperature": [0.0, 0.3, 0.7, 1.0],  # Test multiple temperatures
    "max_tokens": 150,  # For summaries
    "top_p": 1.0,
    "frequency_penalty": 0,
    "presence_penalty": 0
}
```

##### GPT-4o Configuration
```python
{
    "model": "gpt-4o",
    "temperature": [0.0, 0.3, 0.7, 1.0],  # Test multiple temperatures
    "max_tokens": 150,  # For summaries
    "top_p": 1.0,
    "frequency_penalty": 0,
    "presence_penalty": 0
}
```

#### Reasoning Models (No Temperature Variation)

##### o1-mini Configuration
```python
{
    "model": "o1-mini",
    "max_tokens": 150,  # For summaries
    # Temperature not configurable for reasoning models
    # Uses internal chain-of-thought reasoning
}
```

##### o3-mini Configuration
```python
{
    "model": "o3-mini",
    "max_tokens": 150,  # For summaries
    # Temperature not configurable for reasoning models
    # Uses advanced internal reasoning
}
```

#### Non-OpenAI Models (Phase 2)
Models not supported in Phase 1 (will raise NotImplementedError):
- Claude-3 variants (claude-3-opus, claude-3-sonnet)
- Llama-2/3 variants
- Mistral variants (mistral-7b, mixtral-8x7b)
- Gemini variants (gemini-pro)

### Metrics Implementation (Phase 1)
- Calculate FaithBench-specific metrics:
  - **Balanced Accuracy**: Primary metric (handles class imbalance)
  - **Per-Class Precision/Recall**: For 4-level taxonomy
  - **Entropy of Predictions**: Measure detector disagreement
  - **Cohen's Kappa**: Inter-annotator agreement
- Expected baselines (if available from paper):
  - GPT-4-Turbo and GPT-4o: ~50-52% balanced accuracy
  - o1-mini and o3-mini: No published baselines (new models)
- Handle temperature analysis for standard models only (not reasoning models)
- Generate metric aggregations compatible with experiment tracking
- Note: Temperature variation analysis for coherence measures moved to Phase 2

### Integration Points
- CLI commands: `benchmark run faithbench`, `benchmark evaluate faithbench`
- Configuration through existing `BenchmarkConfig` and `BenchmarkRunConfig`
- Result storage in standardized JSON format under `results/faithbench/`
- Progress tracking via existing progress bar implementation
- Error handling with detailed logging and recovery options

### Performance Criteria
- Dataset loading: < 10 seconds for full dataset
- Evaluation throughput: > 10 questions/second (excluding API latency)
- Memory usage: < 2GB for full dataset in memory
- Support parallel processing with configurable worker count
- Checkpoint saves every 100 evaluations or 5 minutes

## Implementation Resources

### Required Downloads
1. **FaithBench Repository**: Clone from https://github.com/vectara/FaithBench
2. **Dataset Files**: Located in `data_for_release/batch_{batch_id}.json`
3. **Evaluation Scripts**: Reference `scripts/how_to_load.py` and `scripts/binarize.py`

### Validation Criteria
- Reproduce paper's ~50% balanced accuracy baseline
- Match GPT-4o > GPT-3.5-Turbo performance ranking
- Support full 4-level annotation taxonomy (not just binary)
- Handle "challenging" samples with high detector disagreement
