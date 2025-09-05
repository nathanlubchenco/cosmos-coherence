# OpenAI Models Support (2025)

## Overview

This document details the OpenAI models supported by Cosmos Coherence as of September 2025, including their specific parameters and constraints.

## GPT-5 Family (Latest - August 2025)

The GPT-5 family represents OpenAI's latest generation of models with significant improvements in reasoning, reduced hallucination, and multimodal understanding.

### Models
- `gpt-5` - Full model with 400K context window, 128K max output tokens
- `gpt-5-mini` - Smaller, more cost-effective variant
- `gpt-5-nano` - Most compact version for lightweight tasks
- `gpt-5-chat` - Chat-optimized variant

### Key Constraints
- **Temperature**: Fixed at 1.0 (not configurable)
- **Max Output Tokens**: Up to 128,000 tokens
- **Unsupported Parameters**: `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`, `max_tokens`
- **Use**: `max_output_tokens` instead of `max_tokens`

### Pricing
- Input: $1.25 per million tokens
- Output: $10 per million tokens

## GPT-4.1 Family (April 2025)

Enhanced GPT-4 models with improved coding, instruction-following, and long-context understanding.

### Models
- `gpt-4.1` - 1 million token context limit, 32K output
- `gpt-4.1-mini` - Smaller variant
- `gpt-4.1-nano` - Lightweight version

### Key Features
- **Temperature**: Configurable (0-2)
- **Max Tokens**: Up to 32,768 output tokens
- **Context Window**: 1 million tokens (gpt-4.1)
- **Supports**: All traditional GPT parameters

## O-Series Reasoning Models (2025)

Advanced reasoning models that allocate more compute time for internal "thinking" before producing answers.

### Models
- `o3` - Latest reasoning model (200K context, 100K output)
- `o3-mini` - Smaller reasoning model
- `o4-mini` - Most recent mini reasoning model
- `o1-preview` - Legacy reasoning model
- `o1-mini` - Legacy mini reasoning model

### Key Features
- **Temperature**: Fixed at 1.0 (not configurable)
- **Reasoning Effort**: Configurable as "low", "medium", or "high" (o3/o4 only)
- **Max Completion Tokens**: Up to 100,000 for o3
- **Unsupported Parameters**: `temperature`, `top_p`, `frequency_penalty`, `presence_penalty`, `max_tokens`
- **Use**: `max_completion_tokens` instead of `max_tokens`

### Important Notes
- Reasoning models consume additional "thinking" tokens not visible in responses
- A simple question might use 10,000+ internal reasoning tokens
- You pay for both reasoning tokens and output tokens

## Legacy Models (Still Supported)

### GPT-4 Family
- `gpt-4` - Original GPT-4
- `gpt-4-turbo-preview` - Faster GPT-4 variant
- `gpt-4o` - Optimized GPT-4
- `gpt-4o-mini` - Smaller GPT-4o variant

### GPT-3.5
- `gpt-3.5-turbo` - Legacy model

### Features
- Full parameter support (temperature, top_p, frequency_penalty, etc.)
- Use `max_tokens` for output limits
- Temperature configurable (0-2)

## Configuration Examples

### GPT-5 Configuration
```yaml
model:
  model_type: gpt-5
  max_output_tokens: 128000  # Not max_tokens!
  # temperature: 1.0  # Fixed, cannot be changed
```

### GPT-4.1 Configuration
```yaml
model:
  model_type: gpt-4.1
  temperature: 0.7  # Configurable
  max_tokens: 32768
  top_p: 0.95
```

### O3 Reasoning Model Configuration
```yaml
model:
  model_type: o3
  max_completion_tokens: 100000  # Not max_tokens!
  reasoning_effort: high  # low, medium, or high
  # temperature: 1.0  # Fixed, cannot be changed
```

### Legacy GPT-4 Configuration
```yaml
model:
  model_type: gpt-4
  temperature: 0.7
  max_tokens: 4096
  top_p: 0.95
  frequency_penalty: 0.0
  presence_penalty: 0.0
```

## Model Selection Guide

### For Hallucination Research
1. **GPT-5**: Best overall performance, lowest hallucination rates
2. **O3/O4**: Best for complex reasoning tasks requiring step-by-step thinking
3. **GPT-4.1**: Good balance of performance and configurability
4. **GPT-4**: Baseline comparison, fully configurable

### For Temperature Studies
- **Use GPT-4.1 or legacy GPT-4**: Full temperature control (0-2)
- **Avoid GPT-5 and O-series**: Temperature fixed at 1.0

### For Cost Optimization
1. **GPT-5-nano**: Most cost-effective GPT-5 variant
2. **GPT-4.1-mini**: Good performance at lower cost
3. **O3-mini/O4-mini**: For reasoning tasks at lower cost
4. **GPT-4o-mini**: Legacy option for basic tasks

## Important Considerations

1. **Registration Required**: GPT-5 models require registration for access
2. **Rate Limits**: GPT-5 has 20K TPM, 200 RPM limits by default
3. **Token Consumption**: Reasoning models use hidden "thinking" tokens
4. **Parameter Compatibility**: Always check model-specific parameter support
5. **Deprecation**: Some models may be phased out (check OpenAI's deprecation page)
