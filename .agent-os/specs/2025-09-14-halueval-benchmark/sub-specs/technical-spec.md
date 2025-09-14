# Technical Specification

This is the technical specification for the spec detailed in @.agent-os/specs/2025-09-14-halueval-benchmark/spec.md

> Created: 2025-09-14
> Version: 1.0.0

## Technical Requirements

### Dataset Requirements

1. **Data Loading System**
   - Load 35,000 samples from 4 JSON files:
     - QA tasks: 10,000 samples
     - Dialogue tasks: 10,000 samples
     - Summarization tasks: 10,000 samples
     - General tasks: 5,000 samples
   - Support both local file loading and HuggingFace dataset integration
   - Implement lazy loading for memory efficiency with large datasets
   - Validate JSON schema on load with descriptive error messages

2. **Binary Hallucination Detection**
   - Implement binary classification system (Yes/No responses)
   - Support evaluation of both "right" and "hallucinated" answer pairs
   - Random sampling between ground-truth and hallucinated outputs for evaluation
   - Consistent prompt formatting across all evaluation calls

3. **Task-Specific Evaluation**
   - Implement distinct evaluation prompts for each task category:
     - QA: Question-answering hallucination detection
     - Dialogue: Multi-turn conversation hallucination detection
     - Summarization: Document summary hallucination detection
     - General: General knowledge hallucination detection
   - Support task-specific context handling and validation
   - Configurable prompt templates per task type

4. **Metrics and Evaluation**
   - Calculate accuracy metrics per task type and overall
   - Implement precision, recall, and F1 scores for hallucination detection
   - Support statistical significance testing between models
   - Generate detailed evaluation reports with per-category breakdowns
   - Track evaluation time and API usage metrics

5. **Caching Integration (Mandatory)**
   - Integrate with OpenAIClient caching system for cost efficiency
   - Implement persistent cache storage in `~/.cache/cosmos_coherence/halueval/`
   - Support cache invalidation and management
   - Enable cache control via CLI flags (`--cache/--no-cache`)
   - Cache responses by prompt hash and model configuration

6. **Sampling and Development Support**
   - Support configurable sampling strategies for development/testing
   - Implement stratified sampling to maintain task distribution
   - Support subset evaluation for rapid iteration
   - Configurable sample sizes per task type

7. **Progress Tracking**
   - Implement progress bars for large dataset evaluation
   - Support resumable evaluation with checkpoint/restart capability
   - Real-time metrics display during evaluation
   - Estimated time remaining calculations

## Data Model Requirements

### HaluEvalItem Pydantic Model

```python
class HaluEvalItem(BaseDatasetItem):
    """Pydantic model for HaluEval dataset items extending BaseDatasetItem"""

    # Core fields
    task_type: Literal["qa", "dialogue", "summarization", "general"]
    knowledge: Optional[str] = None  # Background knowledge/context
    context: Optional[str] = None    # Additional context for task
    right_answer: str               # Ground truth answer
    hallucinated_answer: str        # Hallucinated response to detect
    question: str                   # Query/prompt for the task

    # Task-specific fields
    dialogue_history: Optional[List[Dict[str, str]]] = None  # For dialogue tasks

    # Metadata
    id: str                        # Unique identifier
    source: Optional[str] = None   # Source dataset information

    @validator('task_type')
    def validate_task_type(cls, v):
        """Validate task type is supported"""
        if v not in {"qa", "dialogue", "summarization", "general"}:
            raise ValueError(f"Invalid task_type: {v}")
        return v

    @validator('dialogue_history')
    def validate_dialogue_history(cls, v, values):
        """Validate dialogue_history is present for dialogue tasks"""
        if values.get('task_type') == 'dialogue' and not v:
            raise ValueError("dialogue_history required for dialogue tasks")
        return v

    @validator('context')
    def validate_context_for_summarization(cls, v, values):
        """Validate context is present for summarization tasks"""
        if values.get('task_type') == 'summarization' and not v:
            raise ValueError("context required for summarization tasks")
        return v
```

### Field Validation Requirements

1. **Required Fields by Task Type**
   - QA: `question`, `right_answer`, `hallucinated_answer`, `knowledge`
   - Dialogue: `question`, `right_answer`, `hallucinated_answer`, `dialogue_history`
   - Summarization: `question`, `right_answer`, `hallucinated_answer`, `context`
   - General: `question`, `right_answer`, `hallucinated_answer`

2. **Data Type Validation**
   - All text fields must be non-empty strings
   - `dialogue_history` must be list of dicts with "role" and "content" keys
   - `task_type` must be one of the four supported types
   - Unique `id` validation across dataset

## Evaluation Methodology

### Exact Prompt Templates from Repository

#### QA Task Prompt (from qa_evaluation_instruction.txt)
```python
qa_prompt = """I want you to act as an answer judge. Given a question and an answer, your objective is to determine if the provided answer contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.

You are trying to determine if the answer misunderstands the question context and intention.

#Question#: {question}
#Answer#: {answer}
#Your Judgement#: """
```

#### Dialogue Task Prompt (from dialogue_evaluation_instruction.txt)
```python
dialogue_prompt = """I want you to act as a response judge. Given a dialogue history and a response, your objective is to determine if the provided response contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.

You are trying to determine if the response contains hallucination about factual information:
1. Replacing a true entity with a highly similar entity
2. Replacing a true entity with a dissimilar entity
3. Replacing a true entity with a dissimilar entity of a different type

#Dialogue History#: {dialogue_history}
#Response#: {response}
#Your Judgement#: """
```

#### Summarization Task Prompt (from summarization_evaluation_instruction.txt)
```python
summarization_prompt = """I want you to act as a summary judge. Given a document and a summary, your objective is to determine if the provided summary contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.

You are trying to determine if the summary contains:
1. Information not directly inferrable from the document
2. Incorrect information
3. Contradictions with the document

#Document#: {document}
#Summary#: {summary}
#Your Judgement#: """
```

#### General Task Prompt
```python
general_prompt = """I want you to act as an answer judge. Given a question and an answer, your objective is to determine if the provided answer contains non-factual or hallucinated information based on world knowledge.

#Question#: {question}
#Answer#: {answer}
#Your Judgement#: """
```

### Evaluation Process (Following Repository Implementation)

1. **Binary Classification System**
   - For each item, randomly select either `right_answer` or `hallucinated_answer`
   - Track which type was selected for scoring
   - Present selected answer to LLM for hallucination detection
   - Expected response: "Yes" or "No" (strictly enforced)
   - Response parsing: Extract first occurrence of "Yes" or "No" in response
   - Default to "No" if neither found (conservative approach)

2. **Scoring Methodology**
   - True Positive: Correctly identifies hallucinated content as "Yes"
   - True Negative: Correctly identifies truthful content as "No"
   - False Positive: Incorrectly identifies truthful content as "Yes"
   - False Negative: Incorrectly identifies hallucinated content as "No"

3. **Metrics Calculation**
   - Accuracy: (TP + TN) / (TP + TN + FP + FN)
   - Precision: TP / (TP + FP)
   - Recall: TP / (TP + FN)
   - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
   - Per-task and overall aggregate metrics

## Repository References

### Official Implementation
- **Repository**: https://github.com/RUCAIBox/HaluEval
- **Paper**: https://arxiv.org/abs/2305.11747 (EMNLP 2023)
- **Authors**: Junyi Li, Xiaoxue Cheng, Xin Zhao, Jian-Yun Nie, Ji-Rong Wen

### Key Implementation Files
- `evaluation/evaluate.py` - Main evaluation script
- `evaluation/qa/qa_evaluation_instruction.txt` - QA prompt template
- `evaluation/dialogue/dialogue_evaluation_instruction.txt` - Dialogue prompt
- `evaluation/summarization/summarization_evaluation_instruction.txt` - Summarization prompt
- `data/` - Contains 35,000 samples across 4 JSON files

### Data Files Structure
```python
# Each JSON file contains items with this structure:
{
    "id": "unique_identifier",
    "task_type": "qa|dialogue|summarization|general",
    "question": "the question or prompt",
    "knowledge": "background knowledge (QA tasks)",
    "dialogue_history": [{"role": "...", "content": "..."}],  # Dialogue tasks
    "context": "source document (summarization tasks)",
    "right_answer": "correct/truthful answer",
    "hallucinated_answer": "answer with hallucinations"
}
```

## External Dependencies

No external dependencies needed beyond existing framework components:
- Leverage existing `OpenAIClient` for LLM interactions
- Use existing `BaseDatasetItem` and `BaseBenchmark` abstractions
- Utilize existing caching and progress tracking infrastructure
- Integrate with existing CLI and configuration systems

## Implementation Architecture

1. **HaluEvalBenchmark Class**
   - Extends `BaseBenchmark` following framework patterns
   - Implements task-specific prompt generation
   - Handles binary response parsing and validation
   - Supports all required sampling and evaluation features

2. **Data Pipeline**
   - JSON file loader with schema validation
   - HaluEvalItem model instantiation and validation
   - Task-type specific preprocessing and prompt generation
   - Binary response post-processing and metrics calculation

3. **CLI Integration**
   - Add `halueval` command to benchmark CLI
   - Support all standard benchmark flags plus HaluEval-specific options
   - Integration with existing configuration and caching systems
   - Detailed progress reporting and results export

## Performance Considerations

1. **Memory Management**
   - Lazy loading for large datasets
   - Streaming evaluation to handle 35K samples efficiently
   - Memory-efficient metrics aggregation

2. **API Efficiency**
   - Mandatory caching to reduce API costs
   - Batch processing where possible within framework constraints
   - Configurable rate limiting and retry logic

3. **Scalability**
   - Support for distributed evaluation (future enhancement)
   - Checkpoint/resume capability for long-running evaluations
   - Configurable concurrency levels
