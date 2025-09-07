# API Specification

This is the API specification for the spec detailed in @.agent-os/specs/2025-09-07-benchmark-harness-framework/spec.md

> Created: 2025-09-07
> Version: 1.0.0

## Python API Interface

### BenchmarkRunner Class

```python
class BenchmarkRunner:
    def __init__(
        self,
        openai_client: OpenAIClient,
        config: BenchmarkRunConfig,
        harness_config: Optional[HarnessConfig] = None
    ):
        """Initialize the benchmark runner with configuration."""

    async def validate_reproducibility(
        self,
        benchmark: BaseBenchmark,
        tolerance: float = 0.01
    ) -> ReproducibilityReport:
        """Validate benchmark reproducibility against published baselines.
        Must pass before temperature variations are allowed."""

    async def run_benchmark(
        self,
        benchmark: BaseBenchmark,
        dataset_items: Optional[List[BaseDatasetItem]] = None,
        temperature_range: Optional[List[float]] = None,
        require_validation: bool = True
    ) -> BenchmarkResult:
        """Execute a single benchmark with optional temperature variations.
        If require_validation=True, runs reproducibility check first."""

    async def run_baseline(
        self,
        benchmark: BaseBenchmark,
        dataset_items: Optional[List[BaseDatasetItem]] = None
    ) -> BaselineResult:
        """Run benchmark with original settings (temperature=0, deterministic)."""

    async def run_multiple_benchmarks(
        self,
        benchmarks: List[BaseBenchmark],
        shared_config: Optional[Dict[str, Any]] = None
    ) -> List[BenchmarkResult]:
        """Execute multiple benchmarks in sequence or parallel."""

    def save_results(
        self,
        results: Union[BenchmarkResult, List[BenchmarkResult]],
        output_path: Path
    ) -> None:
        """Save benchmark results to JSONL file."""

    async def resume_from_checkpoint(
        self,
        checkpoint_path: Path
    ) -> BenchmarkResult:
        """Resume an interrupted benchmark run from checkpoint."""

    def compare_to_baseline(
        self,
        results: BenchmarkResult,
        baseline: BaselineResult
    ) -> ComparisonReport:
        """Compare experimental results to validated baseline."""
```

### BaseBenchmark Abstract Interface

```python
class BaseBenchmark(ABC):
    @abstractmethod
    async def load_dataset(self) -> List[BaseDatasetItem]:
        """Load and return the benchmark dataset."""

    @abstractmethod
    def get_prompt(self, item: BaseDatasetItem) -> str:
        """Format dataset item into LLM prompt using original benchmark format."""

    @abstractmethod
    def evaluate_response(
        self,
        response: str,
        ground_truth: str,
        item: BaseDatasetItem
    ) -> EvaluationResult:
        """Evaluate model response using original benchmark metrics."""

    @abstractmethod
    def get_baseline_metrics(self) -> Dict[str, float]:
        """Return published baseline metrics for reproducibility validation."""

    @abstractmethod
    def get_original_prompts(self) -> List[str]:
        """Return example prompts from original paper for format validation."""

    @abstractmethod
    def validate_config(self, config: BenchmarkConfig) -> None:
        """Validate benchmark-specific configuration."""

    @property
    @abstractmethod
    def benchmark_name(self) -> str:
        """Return the benchmark identifier."""

    @property
    @abstractmethod
    def paper_reference(self) -> str:
        """Return the original paper reference for this benchmark."""
```

### Configuration Models

```python
class HarnessConfig(BaseModel):
    """Harness-specific configuration."""
    max_concurrent_requests: int = 10
    checkpoint_interval: int = 100
    enable_progress_bar: bool = True
    save_intermediate_results: bool = True
    output_format: Literal["json", "jsonl"] = "jsonl"
    require_reproducibility_check: bool = True
    reproducibility_tolerance: float = 0.01  # 1% tolerance

class TemperatureConfig(BaseModel):
    """Temperature variation settings."""
    temperatures: List[float] = [0.0, 0.3, 0.7, 1.0]
    run_parallel: bool = True
    aggregate_results: bool = True
    enable_after_validation: bool = True  # Only run if baseline validated

class ReproducibilityConfig(BaseModel):
    """Reproducibility validation settings."""
    validate_before_experiments: bool = True
    tolerance_percentage: float = 1.0
    use_deterministic_seed: bool = True
    random_seed: int = 42
    compare_to_published: bool = True
    save_validation_report: bool = True
```

### Result Models

```python
class BenchmarkResult(BaseResult):
    """Complete benchmark execution results."""
    benchmark_name: str
    run_id: str
    started_at: datetime
    completed_at: Optional[datetime]
    dataset_size: int
    items_completed: int
    baseline_validation_passed: bool
    reproducibility_report: Optional[ReproducibilityReport]
    temperature_results: Dict[float, List[ItemResult]]
    aggregate_metrics: Dict[str, float]
    token_usage: TokenUsage
    errors: List[ErrorDetail]

class BaselineResult(BaseModel):
    """Baseline execution results for reproducibility validation."""
    benchmark_name: str
    accuracy: float
    f1_score: Optional[float]
    exact_match_rate: float
    metrics: Dict[str, float]
    published_baseline: Dict[str, float]
    deviation_percentage: float

class ReproducibilityReport(BaseModel):
    """Report on benchmark reproducibility validation."""
    benchmark_name: str
    validation_passed: bool
    our_metrics: Dict[str, float]
    published_metrics: Dict[str, float]
    deviations: Dict[str, float]
    tolerance_used: float
    sample_comparisons: List[Dict[str, Any]]
    recommendations: List[str]

class ItemResult(BaseModel):
    """Individual item execution result."""
    item_id: str
    prompt: str
    response: str
    ground_truth: str
    is_correct: bool
    evaluation_score: float
    temperature: float
    response_metadata: Dict[str, Any]
    original_metric_score: float  # Score using original paper's metric

class EvaluationResult(BaseModel):
    """Evaluation output from benchmark."""
    is_correct: bool
    score: float
    original_metric_score: float  # Using original evaluation method
    explanation: Optional[str]
    metadata: Dict[str, Any] = {}
```

## CLI Interface

### Basic Commands

```bash
# Validate reproducibility of a benchmark
python -m cosmos_coherence.harness validate-baseline \
    --benchmark faithbench \
    --output validation/faithbench_validation.json

# Run benchmark baseline (temperature=0, deterministic)
python -m cosmos_coherence.harness run-baseline \
    --benchmark simpleqa \
    --output results/simpleqa_baseline.jsonl

# Run a single benchmark with temperature variations (after validation)
python -m cosmos_coherence.harness run \
    --benchmark faithbench \
    --temperatures 0.0,0.5,1.0 \
    --require-validation \
    --output results/faithbench_run.jsonl

# Run multiple benchmarks
python -m cosmos_coherence.harness run-suite \
    --config configs/benchmark_suite.yaml \
    --validate-first \
    --output-dir results/

# Compare results to baseline
python -m cosmos_coherence.harness compare \
    --results results/faithbench_run.jsonl \
    --baseline results/faithbench_baseline.jsonl \
    --output comparisons/faithbench_comparison.json

# Resume from checkpoint
python -m cosmos_coherence.harness resume \
    --checkpoint checkpoints/run_abc123.json \
    --output results/resumed_run.jsonl

# Validate configuration
python -m cosmos_coherence.harness validate \
    --config configs/benchmark_config.yaml
```

### Configuration File Format

```yaml
# benchmark_suite.yaml
benchmarks:
  - name: faithbench
    enabled: true
    dataset_size: 1000

  - name: simpleqa
    enabled: true
    dataset_size: 500

  - name: truthfulqa
    enabled: false

harness:
  max_concurrent_requests: 20
  checkpoint_interval: 50
  enable_progress_bar: true

temperatures:
  values: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
  run_parallel: true

openai:
  model: "gpt-4o-mini"
  use_batch_api: true
  batch_threshold: 100
```

## Integration Examples

### Basic Usage

```python
from cosmos_coherence.harness import BenchmarkRunner
from cosmos_coherence.benchmarks import FaithBenchmark
from cosmos_coherence.llm import OpenAIClient

# Initialize components
client = OpenAIClient(api_key="...")
runner = BenchmarkRunner(client)

# Run benchmark
benchmark = FaithBenchmark()
results = await runner.run_benchmark(
    benchmark,
    temperature_range=[0.0, 0.5, 1.0]
)

# Save results
runner.save_results(results, Path("results/faithbench.jsonl"))
```

### Advanced Usage with Custom Benchmark

```python
class CustomBenchmark(BaseBenchmark):
    @property
    def benchmark_name(self) -> str:
        return "custom_benchmark"

    async def load_dataset(self) -> List[BaseDatasetItem]:
        # Load custom dataset
        return load_custom_data()

    def get_prompt(self, item: BaseDatasetItem) -> str:
        return f"Question: {item.question}"

    def evaluate_response(self, response: str, ground_truth: str, item: BaseDatasetItem) -> EvaluationResult:
        # Custom evaluation logic
        is_correct = response.strip().lower() == ground_truth.strip().lower()
        return EvaluationResult(
            is_correct=is_correct,
            score=1.0 if is_correct else 0.0
        )

    def validate_config(self, config: BenchmarkConfig) -> None:
        # Validate custom requirements
        pass

# Use custom benchmark
custom = CustomBenchmark()
results = await runner.run_benchmark(custom)
```

## Error Handling

All API methods should handle and report the following error conditions:

- **ConfigurationError**: Invalid configuration parameters
- **DatasetLoadError**: Failed to load benchmark dataset
- **APIError**: OpenAI API failures (handled by existing client)
- **EvaluationError**: Benchmark evaluation failures
- **CheckpointError**: Failed to save/load checkpoints
- **ValidationError**: Input validation failures

Errors should be logged with context and optionally collected in results for analysis.
