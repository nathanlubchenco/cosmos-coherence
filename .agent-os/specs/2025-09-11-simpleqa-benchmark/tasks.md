# Spec Tasks

## Tasks

- [x] 1. Implement SimpleQA Data Model and Dataset Loading
  - [x] 1.1 Write tests for SimpleQAItem data model validation
  - [x] 1.2 Create SimpleQAItem class inheriting from BaseDatasetItem
  - [x] 1.3 Write tests for SimpleQA dataset loading from HuggingFace
  - [x] 1.4 Implement dataset loading using HuggingFaceDatasetLoader
  - [x] 1.5 Add dataset caching functionality
  - [x] 1.6 Verify all tests pass

- [x] 2. Implement Evaluation Metrics and Benchmark Logic
  - [x] 2.1 Write tests for exact match scoring function
  - [x] 2.2 Implement exact match evaluation with normalization
  - [x] 2.3 Write tests for F1 scoring function
  - [x] 2.4 Implement token-level F1 score calculation
  - [x] 2.5 Write tests for SimpleQABenchmark class
  - [x] 2.6 Create SimpleQABenchmark class inheriting from BaseExperiment
  - [x] 2.7 Implement evaluation flow with OpenAI integration
  - [x] 2.8 Verify all tests pass

- [ ] 3. Create CLI Interface and Commands
  - [ ] 3.1 Write tests for CLI command parsing
  - [ ] 3.2 Create simpleqa_cli.py module with argparse setup
  - [ ] 3.3 Implement 'run' command with model and sample size parameters
  - [ ] 3.4 Add YAML configuration support using BenchmarkRunConfig
  - [ ] 3.5 Implement progress bars with tqdm
  - [ ] 3.6 Verify all tests pass

- [ ] 4. Implement Results Storage and Export
  - [ ] 4.1 Write tests for SimpleQAResult data model
  - [ ] 4.2 Create SimpleQAResult class with required fields
  - [ ] 4.3 Write tests for JSONL export functionality
  - [ ] 4.4 Implement JSONL export with per-question and aggregate results
  - [ ] 4.5 Add experiment metadata to exports
  - [ ] 4.6 Verify all tests pass

- [ ] 5. Add Baseline Comparison and Validation
  - [ ] 5.1 Write tests for baseline comparison logic
  - [ ] 5.2 Add baseline constants (GPT-4: 82%, GPT-3.5: 68%)
  - [ ] 5.3 Implement validate-baseline command in CLI
  - [ ] 5.4 Add deviation warnings for >5% variance from baselines
  - [ ] 5.5 Run full benchmark validation against sample data
  - [ ] 5.6 Document results and any deviations
  - [ ] 5.7 Verify all tests pass
