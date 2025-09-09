# Research References and Implementation Guide

This document contains critical references and implementation details for the FaithBench benchmark integration.

## Phase 1 Scope

**IMPORTANT:** This implementation focuses on reproducing the FaithBench paper results with OpenAI models only. Coherence integration and support for non-OpenAI providers are planned for Phase 2.

## Primary References

### FaithBench Paper
- **Title:** FaithBench: A Diverse Hallucination Benchmark for Summarization by Modern LLMs
- **Authors:** Forrest Sheng Bao et al. (16 authors total)
- **arXiv:** [2410.13210](https://arxiv.org/abs/2410.13210)
- **Published:** October 17, 2024
- **Paper URL:** https://arxiv.org/abs/2410.13210
- **HTML Version:** https://arxiv.org/html/2410.13210v1

### Official Repository
- **GitHub:** https://github.com/vectara/FaithBench
- **Organization:** Vectara
- **License:** Check repository for license details

## Key Paper Insights

### Benchmark Design Philosophy
FaithBench focuses on "challenging hallucinations" - cases where state-of-the-art hallucination detectors disagree. This makes it particularly valuable for advancing the field beyond easy-to-detect hallucinations.

### Annotation Taxonomy
The paper introduces a four-level annotation system:
1. **Consistent** - Factually accurate summaries
2. **Questionable** - Gray area, potentially subjective
3. **Benign Hallucination** - Incorrect but harmless
4. **Hallucinated** - Clear factual errors

### Model Coverage
The benchmark includes summaries from:
- 10 modern LLMs from 8 different families
- Specific models mentioned: GPT-4o, GPT-3.5-Turbo
- Models tested with various temperature settings

### Dataset Characteristics
- **Task:** Summarization (not Q&A)
- **Text Length:** 106-380 words (1st-3rd quartile)
- **Token Count:** ~137-494 tokens
- **Focus:** High-entropy samples where detectors disagree

## Implementation Requirements

### Dataset Structure
Based on repository analysis:
```json
{
  "sample_id": "unique_id",
  "source": "original text to summarize",
  "summary": "generated summary",
  "annotations": [
    {
      "label": "consistent|questionable|benign|hallucinated",
      "justification": "human explanation",
      "spans": ["specific text spans"]
    }
  ],
  "metadata": {
    "summarizer": "model_name",
    "detector_predictions": {}
  }
}
```

### Evaluation Methodology

#### Metrics to Implement
1. **Balanced Accuracy** - Primary metric (handles class imbalance)
2. **Per-Class Precision/Recall** - For detailed analysis
3. **Entropy of Predictions** - Measure disagreement between detectors
4. **Cohen's Kappa** - Inter-annotator agreement

#### Model Configurations (Phase 1 - OpenAI Only)

##### Standard Models (Support Temperature Variation)

###### GPT-4-Turbo Configuration
```python
{
    "model": "gpt-4-turbo",
    "temperature": [0.0, 0.3, 0.7, 1.0],  # Test all four temperatures
    "max_tokens": 150,  # For summaries
    "top_p": 1.0,
    "frequency_penalty": 0,
    "presence_penalty": 0
}
```

###### GPT-4o Configuration
```python
{
    "model": "gpt-4o",
    "temperature": [0.0, 0.3, 0.7, 1.0],  # Test all four temperatures
    "max_tokens": 150,  # For summaries
    "top_p": 1.0,
    "frequency_penalty": 0,
    "presence_penalty": 0
}
```

##### Reasoning Models (No Temperature Variation)

###### o1-mini Configuration
```python
{
    "model": "o1-mini",
    "max_tokens": 150,  # For summaries
    # Temperature not configurable - uses internal reasoning
}
```

###### o3-mini Configuration
```python
{
    "model": "o3-mini",
    "max_tokens": 150,  # For summaries
    # Temperature not configurable - uses advanced reasoning
}
```

**IMPORTANT:** OpenAI's reasoning models (o1-mini, o3-mini) do not support temperature variation. They operate with fixed internal reasoning processes.

##### Non-OpenAI Models (Phase 2)
The following models are NOT supported in Phase 1:
- Claude-3 variants → Will raise NotImplementedError
- Llama-2/3 variants → Will raise NotImplementedError
- Mistral variants → Will raise NotImplementedError
- Gemini variants → Will raise NotImplementedError

### Prompt Templates

#### Summarization Prompt (inferred from paper context)
```
Summarize the following text in 2-3 sentences:

{source_text}

Summary:
```

#### Hallucination Detection Prompt
```
Given the source text and summary below, determine if the summary contains hallucinations.

Source: {source_text}

Summary: {summary}

Classification (consistent/questionable/benign/hallucinated):
Justification:
```

## Critical Implementation Notes

### 1. Dataset Loading
- Data files in `data_for_release/batch_{batch_id}.json`
- Use `scripts/how_to_load.py` as reference
- Implement `scripts/binarize.py` logic for binary classification

### 2. Challenging Sample Selection
The paper's key innovation is focusing on "challenging" samples:
- Compute entropy of detector predictions
- Select samples with high entropy (disagreement)
- This ensures benchmark difficulty

### 3. Human Annotation Integration
- Preserve human annotations as ground truth
- Support span-level hallucination marking
- Track annotator justifications

### 4. Baseline Reproduction
To validate implementation:
1. Run GPT-4o and GPT-3.5-Turbo on full dataset
2. Expect ~50% balanced accuracy (as per paper)
3. GPT-4o should slightly outperform GPT-3.5-Turbo

## Resources to Download

### Essential Files
1. **Paper PDF:** Download from arXiv for detailed methodology
2. **Dataset:** Clone FaithBench repository
3. **Evaluation Scripts:** From `scripts/` directory

### Recommended Reading
1. Vectara Hallucination Leaderboard documentation
2. Related papers on hallucination detection
3. RAG evaluation methodologies

## Phase 1 Integration Checklist

- [ ] Download and study full paper from arXiv
- [ ] Clone FaithBench repository
- [ ] Analyze dataset structure in `data_for_release/`
- [ ] Review evaluation scripts in `scripts/`
- [ ] Implement data loader matching repository format
- [ ] Configure OpenAI models with correct parameters:
  - [ ] GPT-4-Turbo with temperature variations
  - [ ] GPT-4o with temperature variations
  - [ ] o1-mini without temperature (reasoning model)
  - [ ] o3-mini without temperature (reasoning model)
- [ ] Implement 4-level annotation taxonomy
- [ ] Add entropy-based sample selection
- [ ] Validate against any available baseline results
- [ ] Implement NotImplementedError for non-OpenAI models
- [ ] Document any deviations from original methodology
- [ ] Handle gracefully if o1-mini/o3-mini access unavailable

## Expected Outcomes (Phase 1)

When properly implemented, the FaithBench integration should:
1. Load the official FaithBench dataset
2. Support all annotation levels (not just binary)
3. Test four OpenAI models:
   - GPT-4-Turbo (with 4 temperature settings)
   - GPT-4o (with 4 temperature settings)
   - o1-mini (single configuration, no temperature)
   - o3-mini (single configuration, no temperature)
4. Provide detailed per-category metrics
5. Handle reasoning models correctly (no temperature parameter)
6. Gracefully handle unavailable models (o1-mini/o3-mini may require special access)
7. Raise NotImplementedError for non-OpenAI models

## Phase 2 Coherence Integration (Future Work)

The following analyses are planned for Phase 2:
1. How FaithBench's "challenging" samples relate to coherence measures
2. Whether high-entropy samples have lower coherence scores
3. Correlation between annotation taxonomy and coherence measures
4. Temperature effects on both hallucination and coherence
5. Cross-temperature consistency analysis using philosophical coherence measures
