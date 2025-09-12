# Spec Requirements Document

> Spec: LLM Caching Layer
> Created: 2025-09-11

## Overview

Implement an in-memory caching system for LLM API calls that persists to disk, reducing API costs and improving development iteration speed. The cache will use a hash of all request parameters as the key and store responses for exact match retrieval.

## User Stories

### Developer Iteration Story

As a developer working on hallucination benchmarks, I want to cache LLM responses during development, so that I can iterate quickly without incurring API costs for repeated identical calls.

When running experiments with the same prompts and parameters multiple times during development, the system should:
1. Check the in-memory cache for an exact match (model + settings + prompt hash)
2. Return the cached response if found
3. Make the API call only if not cached
4. Persist the cache to disk after runs for future sessions
5. Display cache hit statistics to monitor effectiveness

### Researcher Reproducibility Story

As a researcher, I want cached responses to be deterministic and reproducible, so that I can ensure consistent results across experiment runs.

The caching system should guarantee that identical inputs always produce identical outputs by:
1. Using a deterministic hash of all parameters
2. Storing complete response objects
3. Optionally loading previous cache from disk at initialization
4. Maintaining cache integrity across sessions

## Spec Scope

1. **In-Memory Cache Implementation** - Fast lookup cache using parameter hashing for exact match retrieval
2. **Disk Persistence Layer** - Optional save/load of cache to/from JSON files for session persistence
3. **OpenAI Client Integration** - Transparent caching layer integrated into existing OpenAI client wrapper
4. **Cache Statistics** - Simple usage metrics printed to console showing hit rates and cost savings
5. **Configuration Options** - Runtime flags to enable/disable caching and control persistence behavior

## Out of Scope

- Multiple LLM provider support (focus on OpenAI only)
- Cache eviction policies or TTL management
- Semantic similarity matching for near-duplicate prompts
- Database-backed storage systems
- Batch API integration (to be removed as tech debt)
- Web UI for cache management

## Expected Deliverable

1. OpenAI client wrapper with transparent caching that can be tested by running the same prompt twice and seeing cache statistics
2. Cache persistence that survives process restarts when enabled via configuration
3. Console output showing cache hit/miss rates and estimated cost savings after benchmark runs
