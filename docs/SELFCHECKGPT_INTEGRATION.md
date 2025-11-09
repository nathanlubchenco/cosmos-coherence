# SelfCheckGPT Integration Guide

This document explains how to install and use the `selfcheckgpt` package with cosmos-coherence for coherence-based hallucination detection.

## Installation Modes

### Package Mode (Recommended for Most Users)

Install selfcheckgpt from PyPI using Poetry:

```bash
# Install all dependencies including selfcheckgpt from PyPI
poetry install
```

This will install selfcheckgpt version ^0.1.7 from PyPI.

### Development Mode (For Coherence Variant Development)

If you are developing coherence variants or need the latest local changes from the selfcheckgpt repository:

**Option 1: Using Poetry with local path**

```bash
# Clone selfcheckgpt repository if not already present
cd /path/to/your/workspace
git clone https://github.com/potsawee/selfcheckgpt.git
# Or use your local fork: git clone https://github.com/yourusername/selfcheckgpt.git

# In cosmos-coherence, install selfcheckgpt in editable mode
cd /path/to/cosmos-coherence
poetry add -e ../selfcheckgpt
```

**Option 2: Manual pip installation (without Poetry)**

If Poetry is not available, you can use pip directly:

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install cosmos-coherence dependencies
pip install -r requirements.txt  # If you have a requirements.txt

# Install selfcheckgpt in editable mode from local repository
pip install -e /path/to/selfcheckgpt
```

**Option 3: System-wide installation (not recommended)**

```bash
# Install selfcheckgpt in editable mode system-wide (requires sudo)
pip3 install --user -e /path/to/selfcheckgpt
```

## Verifying Installation

After installation, verify that the coherence modules can be imported:

```bash
python3 -c "from selfcheckgpt import SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson; print('Success: Coherence variants imported')"
python3 -c "from selfcheckgpt.modeling_coherence_api import CoherenceAPIClient; print('Success: CoherenceAPIClient imported')"
python3 -c "from selfcheckgpt.utils_coherence import normalize_coherence_scores; print('Success: normalize_coherence_scores imported')"
```

All three commands should print success messages.

## Usage in Cosmos Coherence

### Importing Coherence Variants

```python
from selfcheckgpt import SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson
from selfcheckgpt.modeling_coherence_api import CoherenceAPIClient
from selfcheckgpt.utils_coherence import normalize_coherence_scores

# Initialize a coherence variant
coherence_variant = SelfCheckShogenji(model="gpt-4o-mini")

# Use it for scoring (adapted for question variations)
coherence_score = coherence_variant.predict(
    sentences=[original_question],  # Question to evaluate
    sampled_passages=question_variations  # Paraphrased versions
)
```

### API Configuration

The coherence variants require an OpenAI API key:

```bash
# Set in your .env file
OPENAI_API_KEY=your-api-key-here

# Or export as environment variable
export OPENAI_API_KEY=your-api-key-here
```

## Coherence Variants Overview

Three coherence measures are available:

1. **SelfCheckShogenji** - Ratio-based independence measure
   - Formula: C2(A,B) = P(A ∧ B) / (P(A) × P(B))
   - Best for detecting violations of probabilistic independence

2. **SelfCheckFitelson** - Confirmation-based support measure
   - Formula: s(H,E) = P(H|E) - P(H|¬E)
   - Best for asymmetric confirmation relationships
   - Note: Requires more API calls due to conditional probabilities

3. **SelfCheckOlsson** - Relative overlap measure
   - Formula: C1(A,B) = P(A ∧ B) / P(A ∨ B)
   - Best for measuring agreement/overlap

## API Cost Considerations

The coherence variants use OpenAI API for probability extraction:

- **Shogenji/Olsson**: 1 + 2×N API calls per question (N = number of variations)
- **Fitelson**: 1 + 3×N API calls per question (requires conditional probabilities)

For cost efficiency:
- Use `gpt-4o-mini` as the model (default, ~10x cheaper than gpt-4)
- Enable caching (automatically enabled in CoherenceAPIClient)
- Start with small sample sizes for experimentation

## Troubleshooting

### Import Error: "No module named 'selfcheckgpt'"

Solution:
1. Verify installation: `pip list | grep selfcheckgpt` or `poetry show selfcheckgpt`
2. Check Python environment: Make sure you're using the same environment where selfcheckgpt was installed
3. Reinstall: `poetry install` or `pip install -e /path/to/selfcheckgpt`

### OpenAI API Key Not Found

Solution:
1. Ensure OPENAI_API_KEY is set in your .env file
2. Or export it: `export OPENAI_API_KEY=your-key`
3. Verify: `echo $OPENAI_API_KEY`

### Version Mismatch

If you see import errors after upgrading:
```bash
# With Poetry
poetry update selfcheckgpt

# With pip
pip install --upgrade selfcheckgpt
```

## Development Workflow

When developing coherence variants:

1. Make changes in `/path/to/selfcheckgpt` repository
2. Changes are immediately reflected in cosmos-coherence (editable install)
3. No need to reinstall after each change
4. Run tests in selfcheckgpt to verify coherence logic
5. Run integration tests in cosmos-coherence to verify usage

## Additional Resources

- SelfCheckGPT Repository: https://github.com/potsawee/selfcheckgpt
- SelfCheckGPT Documentation: See selfcheckgpt/README.md
- Coherence Theory Reference: See selfcheckgpt/agent-os/specs/2025-11-02_coherence-variants/planning/coherence-theory-reference.md
- Question Variation Coherence Spec: See agent-os/specs/2025-11-07-question-variation-coherence/spec.md
