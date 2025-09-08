# Spec Requirements Document

> Spec: Hugging Face Dataset Loader
> Created: 2025-09-07

## Overview

Implement a Hugging Face dataset loader that integrates with the existing benchmark framework to fetch, cache, and convert specific hallucination benchmark datasets. This loader will enable seamless access to FaithBench, SimpleQA, TruthfulQA, FEVER, and HaluEval datasets through a unified interface integrated with the BaseBenchmark.load_dataset() method.

## User Stories

### Researcher Running Benchmarks

As a researcher, I want to load benchmark datasets from Hugging Face automatically, so that I can run hallucination detection experiments without manual data preparation.

The researcher configures their benchmark to use a specific dataset (e.g., FaithBench) in the configuration file. When they run the benchmark, the system automatically checks for a cached version of the dataset. If not found, it downloads the dataset from Hugging Face, converts it to the appropriate Pydantic model format, caches it locally, and loads it for the benchmark run. The entire process is transparent, with progress indicators for long downloads and clear error messages if issues occur.

### Developer Testing Locally

As a developer, I want to quickly load sample datasets for testing, so that I can validate my code changes without downloading full datasets repeatedly.

The developer runs a test that requires the SimpleQA dataset. The loader first checks the local cache directory. Finding the cached dataset from a previous run, it immediately loads the data without network access. The developer can also configure a sample size to load only a subset of the data for faster iteration during development.

## Spec Scope

1. **Dataset Integration** - Support loading of FaithBench, SimpleQA, TruthfulQA, FEVER, and HaluEval from Hugging Face
2. **Caching System** - Implement local file-based caching with automatic cache hit detection
3. **Format Conversion** - Automatically convert HF dataset formats to corresponding Pydantic models
4. **Error Handling** - Fail fast with clear error messages for validation failures or missing data
5. **Progress Monitoring** - Display download progress for datasets exceeding 10MB

## Out of Scope

- Generic support for arbitrary Hugging Face datasets
- Streaming support for datasets that don't fit in memory
- Cache invalidation based on dataset versions
- Custom preprocessing or filtering during loading
- Support for private datasets requiring authentication

## Expected Deliverable

1. Ability to call benchmark.load_dataset() and automatically fetch/cache HF datasets
2. Successful validation of all loaded data against Pydantic models with immediate failure on validation errors
3. Local cache that persists between runs in .cache/datasets/ directory
