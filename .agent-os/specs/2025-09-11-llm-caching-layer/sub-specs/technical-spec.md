# Technical Specification

This is the technical specification for the spec detailed in @.agent-os/specs/2025-09-11-llm-caching-layer/spec.md

> Created: 2025-09-11
> Version: 1.0.0

## Technical Requirements

### Cache Implementation
- Use Python's built-in `dict` for in-memory storage with O(1) lookups
- Implement deterministic hashing using `hashlib.sha256` for cache keys
- Hash should include: model name, temperature, max_tokens, top_p, messages content, and any other API parameters
- Store complete OpenAI response objects including usage statistics

### Persistence Layer
- Use JSON format for human-readable cache files
- Default cache location: `.cache/llm_responses/cache.json`
- Implement atomic writes to prevent corruption during save operations
- Support compression using gzip for large cache files
- Load cache on client initialization if file exists and caching is enabled

### OpenAI Client Integration
- Extend existing `OpenAIClient` class in `src/cosmos_coherence/llm/openai_client.py`
- Add `enable_cache` parameter to client initialization (default: True)
- Add `cache_dir` parameter for custom cache location
- Implement cache check before API calls in the `generate()` method
- Preserve all existing client functionality and error handling

### Cache Statistics
- Track metrics: total requests, cache hits, cache misses, estimated tokens saved
- Calculate estimated cost savings based on OpenAI pricing
- Print statistics on client destruction or on-demand via method call
- Format output as a simple table using string formatting

### Configuration
- Add cache configuration to `BenchmarkRunConfig` model:
  - `use_cache: bool = True`
  - `cache_persist: bool = True`
  - `cache_dir: Optional[Path] = None`
- Environment variable override: `COSMOS_DISABLE_CACHE=1` to disable globally
- CLI flag support: `--no-cache` to disable for specific runs

## Performance Criteria

- Cache lookup time: < 1ms for typical prompts
- Cache serialization: < 100ms for 1000 entries
- Memory overhead: < 100MB for 10,000 typical cached responses
- Zero performance impact when caching is disabled

## External Dependencies

No new external dependencies required - implementation uses Python standard library components:
- `hashlib` for SHA256 hashing
- `json` for serialization
- `gzip` for compression
- `pathlib` for file operations
