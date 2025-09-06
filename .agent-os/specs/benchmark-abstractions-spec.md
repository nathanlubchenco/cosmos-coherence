# Benchmark Abstractions Specification

## Overview
Create type-safe Pydantic abstractions for the benchmark harness, providing a consistent interface for all benchmark implementations.

## Goals
1. Define core benchmark abstractions using Pydantic models
2. Create a flexible, extensible benchmark framework
3. Ensure type safety across all benchmark operations
4. Support multiple benchmark types (FaithBench, SimpleQA, TruthfulQA, etc.)

## Technical Requirements

### Core Components

#### 1. Base Models (`src/cosmos_coherence/benchmarks/base.py`)
```python
- BenchmarkConfig: Configuration for benchmark execution
- BenchmarkDataset: Abstract dataset interface
- BenchmarkSample: Individual test sample
- BenchmarkResult: Result for a single sample
- BenchmarkMetrics: Aggregate metrics
- BenchmarkRun: Complete benchmark execution
```

#### 2. Evaluation Models (`src/cosmos_coherence/benchmarks/evaluation.py`)
```python
- EvaluationStrategy: Abstract evaluation strategy
- ExactMatchEvaluator: String exact match evaluation
- FuzzyMatchEvaluator: Fuzzy string matching
- SemanticEvaluator: Semantic similarity evaluation
- MultiChoiceEvaluator: Multiple choice evaluation
```

#### 3. Response Models (`src/cosmos_coherence/benchmarks/response.py`)
```python
- ModelResponse: LLM response wrapper
- ResponseSet: Multiple responses for k-sampling
- ResponseMetadata: Timing, tokens, etc.
```

#### 4. Runner Interface (`src/cosmos_coherence/benchmarks/runner.py`)
```python
- BenchmarkRunner: Abstract runner class
- RunConfig: Execution configuration
- RunContext: Execution context and state
```

## Implementation Plan

### Task 1: Create Base Models
1. Define Pydantic base models for benchmark abstractions
2. Add validation rules and constraints
3. Write comprehensive tests

### Task 2: Implement Evaluation Framework
1. Create evaluation strategy interface
2. Implement common evaluation strategies
3. Add metrics calculation

### Task 3: Design Response Handling
1. Create response models
2. Add metadata tracking
3. Implement response validation

### Task 4: Build Runner Framework
1. Create abstract runner class
2. Implement execution flow
3. Add error handling and retries

### Task 5: Integration and Testing
1. Create example benchmark implementation
2. Write integration tests
3. Add documentation

## Data Models

### BenchmarkSample
```python
class BenchmarkSample(BaseModel):
    id: str
    question: str
    context: Optional[str] = None
    choices: Optional[List[str]] = None
    correct_answer: Union[str, List[str]]
    metadata: Dict[str, Any] = {}
```

### BenchmarkResult
```python
class BenchmarkResult(BaseModel):
    sample_id: str
    model_response: str
    is_correct: bool
    score: float
    evaluation_details: Dict[str, Any]
    response_metadata: ResponseMetadata
```

### BenchmarkMetrics
```python
class BenchmarkMetrics(BaseModel):
    accuracy: float
    total_samples: int
    correct_samples: int
    average_score: float
    per_category_metrics: Dict[str, Dict[str, float]]
    timing_stats: TimingStatistics
```

## Testing Requirements

1. **Unit Tests**:
   - Model validation tests
   - Serialization/deserialization tests
   - Edge case handling

2. **Integration Tests**:
   - End-to-end benchmark execution
   - Multiple evaluator testing
   - Error recovery testing

3. **Type Safety Tests**:
   - MyPy type checking
   - Runtime validation tests

## Success Criteria

1. ✅ All models properly typed with Pydantic
2. ✅ 100% test coverage for core models
3. ✅ MyPy passes without errors
4. ✅ Example benchmark implementation works
5. ✅ Documentation complete with examples

## Dependencies

- Pydantic v2.0+
- Python 3.11+
- NumPy for metrics calculation
- Optional: scikit-learn for advanced metrics

## File Structure

```
src/cosmos_coherence/benchmarks/
├── __init__.py
├── base.py           # Core base models
├── evaluation.py     # Evaluation strategies
├── response.py       # Response models
├── runner.py         # Runner framework
├── metrics.py        # Metrics calculation
└── utils.py          # Utility functions

tests/benchmarks/
├── __init__.py
├── test_base.py
├── test_evaluation.py
├── test_response.py
├── test_runner.py
└── test_integration.py
```

## Example Usage

```python
from cosmos_coherence.benchmarks import (
    BenchmarkRunner,
    BenchmarkConfig,
    ExactMatchEvaluator
)

# Configure benchmark
config = BenchmarkConfig(
    name="simple_qa",
    dataset_path="data/simple_qa.json",
    evaluator=ExactMatchEvaluator(),
    max_samples=100
)

# Run benchmark
runner = BenchmarkRunner(config)
results = await runner.run(model_client)

# Access metrics
print(f"Accuracy: {results.metrics.accuracy}")
```

## Timeline

- Task 1: 2 hours - Base models
- Task 2: 2 hours - Evaluation framework
- Task 3: 1 hour - Response handling
- Task 4: 2 hours - Runner framework
- Task 5: 1 hour - Integration and documentation

Total: ~8 hours (1 day)
