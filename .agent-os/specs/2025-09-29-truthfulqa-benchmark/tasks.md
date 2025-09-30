# Spec Tasks

These are the tasks to be completed for the spec detailed in @.agent-os/specs/2025-09-29-truthfulqa-benchmark/spec.md

> Created: 2025-09-29
> Status: Ready for Implementation

## Tasks

- [x] 1. Dataset Integration and Model Setup
  - [x] 1.1 Write tests for TruthfulQAItem validation and dataset loading
  - [x] 1.2 Implement HuggingFace dataset loader integration for truthful_qa dataset
  - [x] 1.3 Add dataset parsing for mc1_targets and mc2_targets structures
  - [x] 1.4 Implement category extraction and validation (38 categories)
  - [x] 1.5 Add sample size support for testing with subsets
  - [x] 1.6 Verify all dataset loading tests pass

- [x] 2. MC1 Evaluation Implementation
  - [x] 2.1 Write tests for MC1 log-probability evaluation logic
  - [x] 2.2 Implement prompt formatting for question + answer choice pairs
  - [x] 2.3 Add OpenAI API integration with logprobs=True parameter
  - [x] 2.4 Implement log-probability comparison across choices
  - [x] 2.5 Add MC1 accuracy calculation (highest logprob wins)
  - [x] 2.6 Verify all MC1 evaluation tests pass

- [x] 3. MC2 Evaluation Implementation
  - [x] 3.1 Write tests for MC2 normalized probability scoring
  - [x] 3.2 Implement probability normalization from logprobs
  - [x] 3.3 Add separation logic for correct (label=1) vs incorrect (label=0) answers
  - [x] 3.4 Implement MC2 score calculation (sum correct / sum all)
  - [x] 3.5 Handle edge cases (varying numbers of correct/incorrect answers)
  - [x] 3.6 Verify all MC2 evaluation tests pass

- [ ] 4. Benchmark Class and Metrics
  - [ ] 4.1 Write tests for TruthfulQABenchmark class methods
  - [ ] 4.2 Implement TruthfulQABenchmark inheriting from HuggingFaceEnabledBenchmark
  - [ ] 4.3 Add get_prompt() method for question formatting
  - [ ] 4.4 Implement evaluate_mc1() and evaluate_mc2() methods
  - [ ] 4.5 Add calculate_metrics() with category-level breakdowns
  - [ ] 4.6 Implement caching integration with OpenAIClient
  - [ ] 4.7 Verify all benchmark class tests pass

- [ ] 5. CLI Interface and Results Reporting
  - [ ] 5.1 Write tests for CLI commands and argument parsing
  - [ ] 5.2 Create truthfulqa_cli.py with Typer application
  - [ ] 5.3 Implement run command with all options (model, sample-size, temperature, cache, category)
  - [ ] 5.4 Add results display with Rich tables (overall + category breakdown)
  - [ ] 5.5 Implement compare command for comparing two result files
  - [ ] 5.6 Add baseline comparison functionality (vs published GPT-3.5/GPT-4 scores)
  - [ ] 5.7 Implement JSON results export with per-question and aggregate metrics
  - [ ] 5.8 Verify all CLI tests pass

- [ ] 6. Integration Testing and Validation
  - [ ] 6.1 Write integration tests for full benchmark pipeline
  - [ ] 6.2 Run benchmark on small sample (50 questions) and verify mechanics
  - [ ] 6.3 Validate MC1 and MC2 calculations against manual examples
  - [ ] 6.4 Test caching behavior (cache hit/miss scenarios)
  - [ ] 6.5 Verify category filtering and reporting
  - [ ] 6.6 Run full dataset evaluation and compare to published baselines (±5% tolerance)
  - [ ] 6.7 Test error handling (API errors, malformed data, missing logprobs)
  - [ ] 6.8 Verify all integration tests pass and pre-commit checks succeed
