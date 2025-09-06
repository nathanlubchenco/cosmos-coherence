# Technical Specification

This is the technical specification for the spec detailed in @.agent-os/specs/2025-09-06-pydantic-benchmark-abstractions/spec.md

> Created: 2025-09-06
> Version: 1.0.0

## Technical Requirements

### Base Model Classes with Inheritance Hierarchy

- **BaseDatasetItem**: Abstract base class for all benchmark dataset items
  - Common fields: `id`, `question`, `metadata`, `created_at`
  - Abstract methods: `validate_content()`, `to_dict()`
- **BaseExperiment**: Abstract base class for experiment configurations
  - Fields: `experiment_id`, `model_name`, `temperature`, `benchmark_type`, `timestamp`
  - Methods: `get_config()`, `validate_parameters()`
- **BaseResult**: Abstract base class for experiment results
  - Fields: `experiment_id`, `item_id`, `prediction`, `ground_truth`, `metrics`
  - Methods: `calculate_metrics()`, `serialize()`

### Field Validation using Pydantic Validators

- **Temperature validation**: Custom validator ensuring range 0.3-1.0
- **Benchmark type validation**: Enum validator for supported benchmarks
- **ID format validation**: UUID4 format enforcement
- **Score validation**: Range validators for metric scores (0.0-1.0)
- **Content validation**: Non-empty string validators for questions/answers
- **Date validation**: ISO format datetime validators

### Support for 5 Benchmarks

- **FaithBench**: Models for factual accuracy evaluation
  - `FaithBenchItem`: Fields for claim, evidence, label
  - `FaithBenchResult`: Accuracy and consistency metrics
- **SimpleQA**: Models for simple question-answering
  - `SimpleQAItem`: Fields for question, answer, category
  - `SimpleQAResult`: Correctness and confidence metrics
- **TruthfulQA**: Models for truthfulness evaluation
  - `TruthfulQAItem`: Fields for question, best_answer, incorrect_answers
  - `TruthfulQAResult`: Truth score and informativeness metrics
- **FEVER**: Models for fact extraction and verification
  - `FEVERItem`: Fields for claim, evidence, label, verdict
  - `FEVERResult`: Verification accuracy metrics
- **HaluEval**: Models for hallucination detection
  - `HaluEvalItem`: Fields for dialogue, knowledge, hallucination_type
  - `HaluEvalResult`: Hallucination detection metrics

### Serialization Methods

- **JSON serialization**: `model_dump()` with custom serializers
- **JSONL support**: Batch serialization methods for streaming
- **Custom encoders**: DateTime and UUID serialization handling
- **Compression support**: Optional gzip serialization for large datasets

### Type Hints with Python 3.11+ Features

- **Generic types**: Use of `TypeVar` for flexible base classes
- **Union types**: Modern `|` syntax for optional fields
- **Literal types**: For enum-like string constraints
- **Protocol types**: For duck-typed interfaces
- **Self type**: For method chaining support

### Custom Validators for Benchmark-Specific Constraints

- **Content validators**: Minimum length requirements, format checks
- **Logic validators**: Cross-field validation (e.g., answer consistency)
- **Range validators**: Numerical bounds for scores and metrics
- **Format validators**: URL validation for evidence links
- **Enum validators**: Strict category membership validation

### Model Versioning Support

- **Schema versioning**: `__version__` field in all models
- **Migration support**: Version-aware deserialization
- **Backward compatibility**: Legacy field support with deprecation warnings
- **Forward compatibility**: Unknown field handling with warnings

### Error Classes for Validation Failures

- **ValidationError**: Base exception for all validation failures
- **BenchmarkValidationError**: Benchmark-specific validation errors
- **SerializationError**: JSON/JSONL serialization failures
- **MigrationError**: Version migration failures
- **ConfigurationError**: Experiment configuration validation errors

### Factory Methods

- **`from_dict()`**: Create models from dictionary data
- **`from_json()`**: Create models from JSON strings
- **`from_jsonl()`**: Batch creation from JSONL files
- **`from_raw_data()`**: Transform raw benchmark data to models
- **`create_experiment()`**: Factory for experiment configuration

### Temperature Variation Tracking

- **Temperature field**: Validated float in range 0.3-1.0
- **Temperature history**: List of temperature values used
- **Temperature effects**: Metrics correlation with temperature
- **Default handling**: Standard temperature values per benchmark

### Coherence Measure Integration

- **Shogenji measure**: Probabilistic coherence calculation
- **Fitelson measure**: Alternative coherence metric
- **Olsson measure**: Third coherence measurement approach
- **Coherence results**: Dedicated model for coherence scores
- **Measure comparison**: Methods to compare coherence metrics

### Evaluation Metric Models

- **Accuracy**: Binary correctness measurement
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: True positive rate calculation
- **Recall**: Sensitivity measurement
- **Consistency Score**: Answer consistency across temperatures
- **Composite metrics**: Combined metric calculations

## Approach

### Implementation Strategy

1. **Base class design**: Start with abstract base classes defining common interface
2. **Incremental development**: Implement one benchmark at a time
3. **Validation-first**: Build robust validators before complex logic
4. **Test-driven**: Write tests for each model before implementation
5. **Documentation**: Inline docstrings and external API documentation

### Architecture Patterns

- **Factory pattern**: For model creation from various data sources
- **Strategy pattern**: For different coherence measures
- **Template method**: For common validation workflows
- **Observer pattern**: For model change notifications

### Performance Considerations

- **Lazy validation**: Defer expensive validation until needed
- **Caching**: Cache validation results and computed metrics
- **Streaming**: Support for large dataset processing
- **Memory efficiency**: Use generators for batch operations

## External Dependencies

### Pydantic 2.0+

**Justification**: Pydantic 2.0+ is essential as the core dependency because:
- Provides type-safe validation framework with excellent performance
- Native support for JSON schema generation and validation
- Built-in serialization/deserialization capabilities
- Extensive customization options for validators
- Strong integration with Python type hints
- Backward compatibility features for model versioning
- Comprehensive error reporting for debugging

**Version requirement**: `pydantic>=2.0.0,<3.0.0`

### Additional Dependencies (if needed)

- **typing-extensions**: For Python 3.11+ type features on older versions
- **uuid**: For ID generation and validation (standard library)
- **datetime**: For timestamp handling (standard library)
- **json**: For serialization support (standard library)
- **gzip**: For optional compression (standard library)
