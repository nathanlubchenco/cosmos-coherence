# HuggingFace Dataset Integration Guide

## Overview

The Cosmos Coherence framework now supports automatic loading and caching of HuggingFace datasets for hallucination detection benchmarks. This integration provides seamless access to popular datasets while maintaining backward compatibility with existing custom implementations.

## Supported Datasets

The following hallucination detection datasets are currently supported:

| Dataset | HuggingFace ID | Description |
|---------|----------------|-------------|
| SimpleQA | `basicv8vc/SimpleQA` | Simple factual questions for accuracy evaluation |
| FaithBench | `vectara/faithbench` | Faithfulness evaluation benchmark |
| TruthfulQA | `truthfulqa/truthful_qa` | Questions testing truthful generation |
| FEVER | `fever/fever` | Fact extraction and verification |
| HaluEval | `pminervini/HaluEval` | Hallucination evaluation across multiple tasks |

## Quick Start

### Basic Usage

```python
from cosmos_coherence.benchmarks.implementations.simpleqa_benchmark import SimpleQABenchmark

# Create benchmark with automatic HuggingFace loading
benchmark = SimpleQABenchmark()

# Load dataset (automatically downloads and caches)
dataset = await benchmark.load_dataset()

# Process items
for item in dataset:
    prompt = benchmark.get_prompt(item)
    # ... run model and evaluate
```

### Using HuggingFaceDatasetLoader Directly

```python
from cosmos_coherence.harness.huggingface_loader import HuggingFaceDatasetLoader

# Create loader
loader = HuggingFaceDatasetLoader()

# Load a specific dataset
items = await loader.load_dataset(
    dataset_name="simpleqa",
    split="test",
    show_progress=True
)

# Items are automatically converted to Pydantic models
for item in items:
    print(f"Question: {item.question}")
    print(f"Answer: {item.best_answer}")
```

## Configuration

### Environment Variables

Configure the HuggingFace integration using environment variables:

```bash
# Cache directory for datasets
export HF_CACHE_DIR="/path/to/cache"

# Show download progress bars
export HF_SHOW_PROGRESS=true

# Force re-download (ignore cache)
export HF_FORCE_DOWNLOAD=false

# CI/Test mode (use cache only, no downloads)
export CI=true  # or during pytest runs
```

### Configuration File

Create a `.env` file in your project root:

```env
HF_CACHE_DIR=.cache/datasets
HF_SHOW_PROGRESS=true
HF_FORCE_DOWNLOAD=false
HF_MAX_RETRIES=3
HF_TIMEOUT_SECONDS=300
```

### Programmatic Configuration

```python
from cosmos_coherence.config.huggingface_config import HuggingFaceConfig
from cosmos_coherence.harness.huggingface_loader import HuggingFaceDatasetLoader

# Custom configuration
config = HuggingFaceConfig(
    cache_dir=Path("/custom/cache"),
    show_progress=True,
    force_download=False,
    max_retries=5
)

# Use with loader
loader = HuggingFaceDatasetLoader(config=config)
```

## Creating Custom Benchmarks

### Extending HuggingFaceEnabledBenchmark

```python
from cosmos_coherence.harness.base_benchmark_hf import HuggingFaceEnabledBenchmark

class MyCustomBenchmark(HuggingFaceEnabledBenchmark):
    """Custom benchmark using HuggingFace datasets."""

    def __init__(self, **kwargs):
        # Set default HuggingFace dataset
        if "hf_dataset_name" not in kwargs:
            kwargs["hf_dataset_name"] = "my_dataset"
        super().__init__(**kwargs)

    def get_prompt(self, item):
        """Format item into prompt."""
        return f"Question: {item.question}"

    def evaluate_response(self, response, ground_truth, item):
        """Evaluate model response."""
        # Custom evaluation logic
        pass

    # ... implement other required methods
```

### Using Factory Method

```python
from cosmos_coherence.benchmarks.implementations.simpleqa_benchmark import SimpleQABenchmark

# Create benchmark with custom configuration
benchmark = SimpleQABenchmark.from_huggingface(
    dataset_name="simpleqa",
    split="validation",
    cache_dir=Path("/custom/cache"),
    show_progress=True
)
```

## Cache Management

### Cache Location

By default, datasets are cached in `.cache/datasets/` relative to your project root. Each dataset split is stored as a separate JSON file:

```
.cache/datasets/
├── simpleqa_test.json
├── simpleqa_train.json
├── faithbench_test.json
└── ...
```

### Clearing Cache

```python
from cosmos_coherence.harness.huggingface_loader import HuggingFaceDatasetLoader

loader = HuggingFaceDatasetLoader()

# Clear specific dataset cache
loader.clear_cache("simpleqa")

# Clear all cached datasets
loader.clear_cache()
```

### Programmatic Cache Management

```python
# With benchmark
benchmark = SimpleQABenchmark()
benchmark.clear_hf_cache("simpleqa")

# Check cache status
info = benchmark.get_hf_dataset_info()
print(f"Cache enabled: {info['enabled']}")
print(f"Cache directory: {info['cache_dir']}")
```

## CI/Testing Integration

### Automatic CI Detection

The framework automatically detects CI environments and test runs:

- Checks `CI` environment variable
- Detects pytest execution via `PYTEST_CURRENT_TEST`
- Uses cached data only (no network requests)
- Returns empty datasets if cache is missing

### Testing with Mock Data

```python
import pytest
from unittest.mock import patch

@pytest.mark.asyncio
async def test_with_mock_data():
    with patch("cosmos_coherence.harness.huggingface_loader.HuggingFaceDatasetLoader.load_dataset") as mock:
        mock.return_value = [
            SimpleQAItem(question="Test?", best_answer="Answer")
        ]

        benchmark = SimpleQABenchmark()
        dataset = await benchmark.load_dataset()
        assert len(dataset) == 1
```

### Pre-populating Test Cache

```python
import json
from pathlib import Path

def setup_test_cache():
    cache_dir = Path(".cache/datasets")
    cache_dir.mkdir(parents=True, exist_ok=True)

    test_data = [
        {"question": "Test question", "best_answer": "Test answer"}
    ]

    with open(cache_dir / "simpleqa_test.json", "w") as f:
        json.dump(test_data, f)
```

## Dataset Converters

### Automatic Field Mapping

The converters handle various field naming conventions:

```python
# SimpleQA: accepts "answer" or "best_answer"
{"question": "...", "answer": "..."}  # ✓ Accepted
{"question": "...", "best_answer": "..."}  # ✓ Accepted

# FaithBench: uses "claim" as question
{"claim": "...", "context": "..."}  # ✓ Accepted

# FEVER: automatic label conversion
{"claim": "...", "label": "SUPPORTED"}  # ✓ Converted to enum
```

### Default Values

Missing or invalid fields are handled gracefully:

- TruthfulQA: Invalid category → defaults to "other"
- FEVER: Missing label → defaults to "NOTENOUGHINFO"
- HaluEval: Missing task_type → defaults to "general"
- FaithBench: Missing annotations → defaults to empty list

### Validation Error Handling

```python
try:
    items = loader._convert_to_pydantic(raw_data, "simpleqa")
except DatasetValidationError as e:
    print(f"Validation failed: {e}")
    print(f"Failed field: {e.field}")
    print(f"Invalid value: {e.value}")
```

## Performance Considerations

### Memory Usage

- Datasets are loaded entirely into memory by default
- Configure `max_dataset_size_mb` to limit memory usage
- Use streaming for very large datasets (future feature)

### Network Optimization

- Automatic retry with exponential backoff
- Configurable timeout (default: 300 seconds)
- Progress bars for large downloads
- Local caching eliminates redundant downloads

### Parallel Processing

```python
import asyncio

async def load_multiple_datasets():
    tasks = [
        loader.load_dataset("simpleqa"),
        loader.load_dataset("faithbench"),
        loader.load_dataset("truthfulqa")
    ]
    datasets = await asyncio.gather(*tasks)
    return datasets
```

## Troubleshooting

### Common Issues

1. **Import Error: No module named 'datasets'**
   ```bash
   pip install datasets
   # or
   poetry add datasets
   ```

2. **Network timeout during download**
   ```python
   # Increase timeout
   config = HuggingFaceConfig(timeout_seconds=600)
   ```

3. **Cache permission errors**
   ```bash
   # Ensure write permissions
   chmod -R u+w .cache/datasets/
   ```

4. **CI failures due to missing cache**
   ```python
   # Pre-populate cache or mock in tests
   # See CI/Testing Integration section
   ```

### Debug Logging

Enable detailed logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("cosmos_coherence.harness.huggingface_loader")
logger.setLevel(logging.DEBUG)
```

## Examples

### Complete Benchmark Run

```python
import asyncio
from cosmos_coherence.benchmarks.implementations.simpleqa_benchmark import SimpleQABenchmark

async def run_benchmark():
    # Initialize benchmark
    benchmark = SimpleQABenchmark(
        hf_show_progress=True,
        hf_cache_dir=Path("./my_cache")
    )

    # Load dataset
    dataset = await benchmark.load_dataset()
    print(f"Loaded {len(dataset)} items")

    # Process each item
    results = []
    for item in dataset[:10]:  # Process first 10 items
        # Generate prompt
        prompt = benchmark.get_prompt(item)

        # Run your model (placeholder)
        model_response = "Paris"  # Your model here

        # Evaluate response
        result = benchmark.evaluate_response(
            response=model_response,
            ground_truth=item.best_answer,
            item=item
        )
        results.append(result)

    # Aggregate results
    accuracy = sum(1 for r in results if r.is_correct) / len(results)
    print(f"Accuracy: {accuracy:.2%}")

# Run the benchmark
asyncio.run(run_benchmark())
```

### Custom Dataset Integration

```python
from cosmos_coherence.harness.huggingface_loader import HuggingFaceDatasetLoader

class CustomDatasetLoader(HuggingFaceDatasetLoader):
    """Extended loader for custom datasets."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add custom dataset mappings
        self.dataset_mapping["my_custom"] = "org/my_custom_dataset"

    def _convert_my_custom_item(self, item):
        """Convert custom dataset item."""
        return MyCustomItem(
            id=item.get("id"),
            question=item.get("query"),
            answer=item.get("response")
        )

    def _convert_to_pydantic(self, raw_data, dataset_name):
        if dataset_name == "my_custom":
            return [self._convert_my_custom_item(item) for item in raw_data]
        return super()._convert_to_pydantic(raw_data, dataset_name)
```

## API Reference

### HuggingFaceDatasetLoader

```python
class HuggingFaceDatasetLoader:
    async def load_dataset(
        dataset_name: str,
        split: Optional[str] = None,
        force_download: bool = False,
        show_progress: bool = False
    ) -> List[BaseDatasetItem]:
        """Load dataset with caching support."""

    def clear_cache(dataset_name: Optional[str] = None) -> None:
        """Clear cache for specific dataset or all."""
```

### HuggingFaceEnabledBenchmark

```python
class HuggingFaceEnabledBenchmark(BaseBenchmark):
    def configure_huggingface(
        dataset_name: str,
        split: str = "test",
        cache_dir: Optional[Path] = None,
        show_progress: bool = False,
        force_download: bool = False
    ) -> None:
        """Configure HuggingFace dataset parameters."""

    def get_hf_dataset_info() -> Dict[str, Any]:
        """Get information about configured dataset."""
```

### HuggingFaceConfig

```python
class HuggingFaceConfig(BaseModel):
    cache_dir: Path
    force_download: bool
    show_progress: bool
    use_cached_only: bool
    max_retries: int
    timeout_seconds: int

    @classmethod
    def from_env() -> "HuggingFaceConfig":
        """Create configuration from environment."""
```

## Contributing

To add support for new datasets:

1. Add dataset mapping in `HuggingFaceDatasetLoader.DEFAULT_DATASET_MAPPING`
2. Create Pydantic model in `benchmarks/models/datasets.py`
3. Implement converter method `_convert_[dataset]_item()`
4. Add converter to `_convert_to_pydantic()` method
5. Write tests in `tests/harness/test_converter_edge_cases.py`
6. Update this documentation

## License

This integration is part of the Cosmos Coherence project and follows the same license terms.
