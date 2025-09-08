# Technical Specification

This is the technical specification for the spec detailed in @.agent-os/specs/2025-09-07-huggingface-dataset-loader/spec.md

> Created: 2025-09-07
> Version: 1.0.0

## Technical Requirements

### Core Functionality
- Integrate HuggingFace dataset loading into existing `BaseBenchmark.load_dataset()` method
- Support automatic detection and loading of 5 specific datasets: FaithBench, SimpleQA, TruthfulQA, FEVER, HaluEval
- Implement dataset-specific loaders that handle the unique structure of each benchmark
- Convert HF dataset rows to corresponding Pydantic models (FaithBenchItem, SimpleQAItem, etc.)
- Fail fast on validation errors with detailed error messages showing which field failed validation

### Caching Mechanism
- Use local file-based caching in `.cache/datasets/` directory within project root
- Cache key based on dataset name and configuration (e.g., `.cache/datasets/faithbench_v1.json`)
- Serialize datasets as JSON files for human readability and debugging
- Check cache before attempting to download from HuggingFace
- No automatic cache invalidation - manual deletion required for updates

### Dataset Mapping Configuration
- Create mapping dictionary for HF dataset identifiers:
  - FaithBench: "vectara/faithbench" or appropriate identifier
  - SimpleQA: "basicv8vc/SimpleQA"
  - TruthfulQA: "truthfulqa/truthful_qa"
  - FEVER: "fever/fever"
  - HaluEval: "pminervini/HaluEval"
- Support for dataset splits (train/validation/test) where applicable
- Default to appropriate split for each benchmark (e.g., test split for evaluation)

### Error Handling Strategy
- Raise `DatasetLoadError` for network/download failures
- Raise `DatasetValidationError` for Pydantic validation failures with field details
- Raise `DatasetNotFoundError` for unsupported dataset names
- Include original exception in error chain for debugging
- Log all errors with appropriate severity levels

### Progress Indication
- Use tqdm for download progress bars when dataset size > 10MB
- Show dataset name, download size, and estimated time remaining
- Suppress progress bars in CI/test environments (detect via environment variable)

### Memory Management
- Load entire dataset into memory (confirmed all 5 datasets are < 200MB)
- For FEVER (largest at ~185k claims), implement optional subset loading for development
- Return datasets as lists of Pydantic model instances
- No streaming required for initial implementation

## Approach

### Implementation Strategy
1. **Phase 1**: Extend `BaseBenchmark` class with HuggingFace integration
2. **Phase 2**: Implement dataset-specific mapping and validation logic
3. **Phase 3**: Add caching layer with JSON serialization
4. **Phase 4**: Integrate error handling and progress indication

### Architecture Changes
- Modify `BaseBenchmark.load_dataset()` to check for HuggingFace availability
- Create `HuggingFaceDatasetMixin` for reusable HF functionality
- Add dataset registry pattern for mapping benchmark names to HF identifiers
- Implement validation pipeline using existing Pydantic models

## External Dependencies

**datasets** - Official HuggingFace datasets library for fetching benchmark data
**Justification:** Required for accessing HuggingFace dataset hub and handling various dataset formats. Already listed in tech-stack.md as primary data source.

**tqdm** - Progress bar library for download indication
**Justification:** Provides user-friendly progress indication for large dataset downloads. Lightweight and commonly used in ML projects.
