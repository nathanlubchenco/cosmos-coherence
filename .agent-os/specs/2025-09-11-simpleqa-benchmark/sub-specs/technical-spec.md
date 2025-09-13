# Technical Specification

This is the technical specification for the spec detailed in @.agent-os/specs/2025-09-11-simpleqa-benchmark/spec.md

## Technical Requirements

### Dataset Loading and Processing
- Utilize HuggingFaceDatasetLoader class to fetch SimpleQA dataset from "openai/simple-evals" repository
- Implement SimpleQAItem data model inheriting from BaseDatasetItem with fields:
  - question: str (the factual question)
  - answer: str (ground truth answer)
  - metadata: Dict[str, Any] (optional metadata from dataset)
- Support dataset sampling through sample_size parameter for development/testing
- Implement automatic caching using existing cache directory structure (~/.cache/cosmos-coherence/)

### Evaluation Implementation
- Create SimpleQABenchmark class inheriting from BaseExperiment
- Implement two evaluation metrics as per OpenAI paper:
  - **Exact Match**: Case-insensitive string comparison after normalization (removing articles, punctuation)
  - **F1 Score**: Token-level F1 calculation treating answer as bag of words
- Evaluation flow:
  1. Load questions from dataset
  2. Query LLM with prompt template: "Answer this question concisely: {question}"
  3. Extract answer from response
  4. Calculate exact match and F1 scores
  5. Aggregate results across dataset

### CLI Integration
- Create simpleqa_cli.py module following existing pattern from faithbench_cli.py
- Implement commands:
  - `run`: Execute benchmark with parameters (model, sample_size, output_path)
  - `validate-baseline`: Compare results against published baselines
  - `export`: Export results to JSONL format
- Support configuration through YAML files using BenchmarkRunConfig

### Model Integration
- Use existing OpenAIClient class for LLM interactions
- Remove batch API functionality as per requirements
- Support models: gpt-4, gpt-4-turbo, gpt-3.5-turbo
- Implement retry logic for API failures using existing retry decorators

### Results and Metrics
- Store results using existing BaseResult pattern
- Implement SimpleQAResult class with fields:
  - question_id: str
  - question: str
  - ground_truth: str
  - model_response: str
  - exact_match: bool
  - f1_score: float
  - metadata: Dict[str, Any]
- Calculate aggregate metrics:
  - Overall exact match accuracy (%)
  - Overall F1 score
  - Per-category breakdowns if metadata available

### Baseline Comparison
- Store baseline results as constants:
  - GPT-4: 82% exact match
  - GPT-3.5: 68% exact match
- Implement comparison logic to report delta from baselines
- Flag warnings if results deviate >5% from expected baselines

### Export Format
- JSONL format with one result per line
- Include experiment metadata (model, timestamp, parameters)
- Ensure compatibility with existing experiment tracking system
- Support both individual results and aggregated summary export

## Performance Considerations

- Implement progress bars using tqdm for long-running evaluations
- Use async/concurrent processing where possible (within API rate limits)
- Cache LLM responses to avoid redundant API calls during development
- Optimize F1 calculation using set operations for efficiency

## Error Handling

- Graceful handling of API rate limits with exponential backoff
- Validation of dataset integrity before evaluation
- Clear error messages for missing dependencies or configuration issues
- Checkpoint support to resume interrupted evaluations
