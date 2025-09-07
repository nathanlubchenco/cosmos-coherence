# Product Roadmap

## Phase 1: Foundation & Benchmark Reproduction

**Goal:** Establish core framework and reproduce existing benchmark results
**Success Criteria:** Successfully reproduce results within 5% of published benchmarks for at least 3 major datasets

### Features

- [x] Core Pydantic configuration system - Set up type-safe config management `S`
- [x] .gitignore and any other basic repo scaffolding like precommit hooks `S`
- [x] Makefile to run tests and setup any dependecies `S` (Note: Makefile updates integrated into each feature implementation)
- [x] Dockerization to make sure everything runs in a container for easy portability `S`
- [x] Core Pydantic abstractions for the benchmark - Set up type-safe benchmark harness `S`
- [x] OpenAI client integration - Implement rate-limited API wrapper (COMPLETE: Tasks 1-4) `S`
- [x] Benchmark harness implementation - Basic execution framework (includes OpenAI integration testing) `M`
- [x] Basic CLI interface - Run benchmarks with configuration files `S`
- [ ] Hugging Face dataset loader - Fetch and cache benchmark datasets `S`
- [ ] Dataset sampling system - Enable quick validation runs `S`
- [ ] FaithBench implementation - Reproduce benchmark with reference implementation `L`
- [ ] SimpleQA implementation - Reproduce benchmark methodology `M`
- [ ] TruthfulQA implementation - Import dataset and evaluation logic `M`

### Dependencies

- Python 3.11+ environment setup
- OpenAI API access and keys
- Reference paper collection
- Hugging Face datasets access

## Phase 2: Novel Coherence Integration

**Goal:** Implement philosophical coherence measures and k-response analysis
**Success Criteria:** Successfully apply all three coherence measures with demonstrable variance in results

### Features

- [ ] K-response generation system - Generate multiple samples per query `M`
- [ ] Shogenji coherence implementation - Implement measure with tests `M`
- [ ] Coherence-based selection algorithm - Choose best response by coherence `M`
- [ ] Fitelson coherence implementation - Implement measure with tests `M`
- [ ] Olsson coherence implementation - Implement measure with tests `M`
- [ ] Temperature variance controller - Systematic temperature exploration `S`
- [ ] Cross-response coherence analysis - Measure coherence across response sets `L`

### Dependencies

- Phase 1 completion
- Mathematical validation of coherence implementations
- Statistical analysis framework

## Phase 3: Analysis & Visualization

**Goal:** Create comprehensive analysis tools and user interface
**Success Criteria:** Interactive UI deployed with full benchmark results visualization

### Features

- [ ] Results storage system - JSON/JSONL flat file experiment tracking `M`
- [ ] API integration models - FastAPI request/response models for benchmark results `M`
- [ ] Dash application - Python-based interactive dashboard `L`
- [ ] Benchmark comparison views - Side-by-side benchmark analysis `M`
- [ ] Temperature-coherence heatmaps - Visualize relationship patterns `M`
- [ ] Export functionality - Generate research-ready figures and tables `S`
- [ ] Docker containerization - Package entire system for reproducibility `M`

### Dependencies

- Phase 2 completion
- Full benchmark run data
- UI/UX design decisions

## Technical Debt & Maintenance

**Goal:** Track and address technical debt items, test failures, and maintenance tasks
**Success Criteria:** Maintain code quality, test coverage, and system stability

### Items

- [ ] Fix failing serialization tests - 7 tests failing in `test_serialization.py` with validation edge cases `S`
- [ ] Fix Docker integration test skips - Tests currently skip when containers aren't running, need proper Docker environment setup `M`
- [ ] Address Pydantic deprecation warnings - Update to use ConfigDict instead of class-based config `S`
- [ ] Fix mypy type hints warnings - Address model field naming conflicts with protected namespaces `S`
- [ ] Update test assertions - Some test assertions need updating to match actual error messages from Pydantic 2.8 `S`
- [ ] Fix CLI progress monitoring test - `test_progress_bar_display` needs refactoring after CLI changes `S`
- [ ] Fix CLI config validation test - `test_load_invalid_config` needs adjustment after CLI refactoring `S`
- [ ] Fix CLI end-to-end workflow test - `test_full_benchmark_workflow` needs update after CLI refactoring `M`
- [ ] Fix batch API error handling test - `test_retrieve_batch_results_handles_errors` PartialFailureError.successful_results attribute needs implementation `S`
- [ ] Fix CLI component initialization test - `test_initialize_components` BenchmarkRunner requires BaseBenchmark instance `S`
- [ ] Fix CLI compare command tests - `test_compare_results` and `test_compare_with_output` need proper mock handling `S`
- [ ] Fix OpenAI client batch threshold test - `test_auto_batch_threshold` AsyncMock handling needs refactoring `S`

### Notes

- **Keep this list updated:** When tests are skipped or marked as xfail, add them here for future resolution
- **Review periodically:** Check this list during sprint planning or maintenance windows
- **Document fixes:** When addressing items, document the solution for future reference
