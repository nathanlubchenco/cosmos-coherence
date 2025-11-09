# Configuration Templates for Question-Variation Coherence

This directory contains pre-configured YAML templates for common use cases of the question-variation coherence feature.

## Available Configurations

### 1. Conservative (`conservative.yaml`)

**Use Case**: Production systems where incorrect answers are costly

**Key Settings**:
- Threshold: 0.7 (HIGH)
- Expected abstention: 30-50%
- Expected precision: 80-85%
- Expected recall: 50-60%

**When to Use**:
- Production systems
- High-stakes applications
- Minimizing incorrect answers is critical

```bash
cosmos-coherence simpleqa run-with-coherence \
    --config configs/question_variation_coherence/conservative.yaml
```

---

### 2. Balanced (`balanced.yaml`)

**Use Case**: General research evaluation, balanced performance

**Key Settings**:
- Threshold: 0.5 (MODERATE)
- Expected abstention: 10-30%
- Expected precision: 78-82%
- Expected recall: 60-70%

**When to Use**:
- Research evaluation
- Balanced precision-recall tradeoff
- Standard benchmarking

```bash
cosmos-coherence simpleqa run-with-coherence \
    --config configs/question_variation_coherence/balanced.yaml
```

---

### 3. Aggressive (`aggressive.yaml`)

**Use Case**: Exploratory analysis, maximize coverage

**Key Settings**:
- Threshold: 0.3 (LOW)
- Expected abstention: 5-15%
- Expected precision: 75-78%
- Expected recall: 65-75%

**When to Use**:
- Exploratory research
- Maximizing answer coverage
- Understanding model capabilities

```bash
cosmos-coherence simpleqa run-with-coherence \
    --config configs/question_variation_coherence/aggressive.yaml
```

---

### 4. Experimental (`experimental.yaml`)

**Use Case**: Enhanced coherence signal quality for research

**Key Settings**:
- Threshold: 0.5 (MODERATE)
- Variations: 10 (HIGH)
- Temperature range: [0.5, 0.7, 1.0, 1.3, 1.6] (WIDE)
- Sample size: 100 (SMALL)
- Verbose: true

**When to Use**:
- Research into coherence measures
- Validating approach with better signal
- Small-scale experiments

**Note**: ~2x API cost due to more variations

```bash
cosmos-coherence simpleqa run-with-coherence \
    --config configs/question_variation_coherence/experimental.yaml
```

---

### 5. Quick Test (`quick_test.yaml`)

**Use Case**: Rapid testing and debugging

**Key Settings**:
- Sample size: 10 (VERY SMALL)
- Variations: 3 (REDUCED)
- Workers: 2 (CONSERVATIVE)
- Verbose: true

**When to Use**:
- Testing installation
- Debugging issues
- Validating configuration changes
- Rapid iteration

**Cost**: ~$0.01-0.02 per run

```bash
cosmos-coherence simpleqa run-with-coherence \
    --config configs/question_variation_coherence/quick_test.yaml
```

---

## Configuration Parameters Reference

### Coherence Parameters

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `coherence_variant` | string | shogenji, fitelson, olsson | Coherence measure to use |
| `coherence_threshold` | float | 0.0-1.0 | Abstention threshold |
| `num_variations` | int | 3-20 | Number of question paraphrases |
| `temperature_range` | list | [0.5-1.6] | Temperatures for variation generation |

### Infrastructure Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sample_size` | int or null | Number of questions (null = full dataset) |
| `checkpoint_interval` | int | Save progress every N questions |
| `workers` | int | Parallel workers for API calls |
| `verbose` | bool | Enable detailed logging |

### Answer Strategy Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `abstention_response` | string | Text for abstentions |
| `high_coherence_answer_yes` | bool | YES/NO polarity mapping |

---

## Customizing Configurations

### Method 1: Copy and Modify

```bash
# Copy existing config
cp configs/question_variation_coherence/balanced.yaml configs/question_variation_coherence/my_config.yaml

# Edit parameters
nano configs/question_variation_coherence/my_config.yaml

# Run with custom config
cosmos-coherence simpleqa run-with-coherence \
    --config configs/question_variation_coherence/my_config.yaml
```

### Method 2: Command-Line Overrides

Override specific parameters without creating new config:

```bash
cosmos-coherence simpleqa run-with-coherence \
    --config configs/question_variation_coherence/balanced.yaml \
    --coherence-threshold 0.6 \
    --num-variations 7 \
    --sample-size 200
```

### Method 3: Python API

```python
from cosmos_coherence.benchmarks.implementations.simpleqa_coherence_benchmark import SimpleQACoherenceBenchmark
import yaml

# Load config
with open("configs/question_variation_coherence/balanced.yaml") as f:
    config = yaml.safe_load(f)

# Create benchmark with custom overrides
benchmark = SimpleQACoherenceBenchmark(
    model=config["evaluation"]["model"],
    coherence_variant=config["evaluation"]["coherence_variant"],
    coherence_threshold=0.6,  # Override
    num_variations=7,          # Override
    sample_size=200           # Override
)

# Run evaluation
results = await benchmark.evaluate_full_dataset()
```

---

## Cost Estimates

### Per Configuration (Full Dataset: 4,326 questions)

| Configuration | API Calls | Cost (gpt-4o-mini) | Cost (gpt-4-turbo) |
|---------------|-----------|--------------------|--------------------|
| Conservative | ~77,000 | $4-5 | $35-40 |
| Balanced | ~77,000 | $4-5 | $35-40 |
| Aggressive | ~77,000 | $4-5 | $35-40 |
| Experimental (100 questions) | ~2,600 | $0.30-0.40 | $2.50-3.50 |
| Quick Test (10 questions) | ~130-150 | $0.01-0.02 | $0.10-0.15 |

**Note**: Costs assume 20-30% cache hit rate. Fitelson variant ~30% more expensive than Shogenji/Olsson.

---

## Threshold Tuning Guide

Run multiple thresholds to find optimal for your use case:

```bash
# Create configs for different thresholds
for threshold in 0.3 0.4 0.5 0.6 0.7; do
    # Copy base config
    cp configs/question_variation_coherence/balanced.yaml configs/question_variation_coherence/threshold_${threshold}.yaml

    # Update threshold (requires manual edit or sed)
    sed -i '' "s/coherence_threshold: 0.5/coherence_threshold: ${threshold}/" configs/question_variation_coherence/threshold_${threshold}.yaml

    # Run evaluation
    cosmos-coherence simpleqa run-with-coherence \
        --config configs/question_variation_coherence/threshold_${threshold}.yaml \
        --output results/threshold_${threshold}.json
done

# Compare results
python scripts/compare_thresholds.py results/threshold_*.json
```

---

## Comparing Coherence Variants

Run all three coherence measures on the same sample:

```bash
for variant in shogenji fitelson olsson; do
    # Copy base config
    cp configs/question_variation_coherence/balanced.yaml configs/question_variation_coherence/variant_${variant}.yaml

    # Update variant (requires manual edit)
    # Edit coherence_variant: shogenji -> variant

    # Run evaluation
    cosmos-coherence simpleqa run-with-coherence \
        --config configs/question_variation_coherence/variant_${variant}.yaml \
        --output results/variant_${variant}.json
done

# Compare results
python scripts/compare_variants.py results/variant_*.json
```

---

## Best Practices

1. **Start Small**: Always test with `quick_test.yaml` before full evaluation
2. **Validate Approach**: Use `experimental.yaml` (100 questions) to validate before full dataset
3. **Cost Management**: Monitor API costs with small samples first
4. **Threshold Tuning**: Run multiple thresholds on subset before full evaluation
5. **Checkpointing**: Always enable checkpoints for long-running evaluations
6. **Baseline Comparison**: Always run baseline for fair comparison

---

## See Also

- [README.md](../../README.md) - Feature overview and usage
- [API_DOCUMENTATION.md](../API_DOCUMENTATION.md) - Detailed API reference
- [TROUBLESHOOTING.md](../TROUBLESHOOTING.md) - Common issues and solutions
- [VALIDATION_GUIDE.md](../VALIDATION_GUIDE.md) - Testing and validation
