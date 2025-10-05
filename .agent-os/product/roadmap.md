# Product Roadmap

## Benchmark Implementation Procedure

**IMPORTANT:** All benchmark implementations must follow the standardized procedure documented in @.agent-os/instructions/core/benchmark-implementation-procedure.md

Key steps for each benchmark:
1. **Research First**: Find papers (including recent versions), repositories, and implementation details
2. **Document Thoroughly**: Create research-references.md with all findings
3. **Scope Properly**: Phase 1 = OpenAI models only, exact reproduction
4. **Validate Results**: Compare against paper baselines

## Phase 1: Foundation & Benchmark Reproduction

**Goal:** Establish core framework and reproduce existing benchmark results
**Success Criteria:** Successfully reproduce results within 5% of published benchmarks for at least 3 major datasets

**Latest Assessment:** See `docs/benchmark-assessment-2025-09-29.md` for comprehensive analysis of current implementations and recommended priorities (2025-09-29)

### Features

- [x] Core Pydantic configuration system - Set up type-safe config management `S`
- [x] .gitignore and any other basic repo scaffolding like precommit hooks `S`
- [x] Makefile to run tests and setup any dependecies `S` (Note: Makefile updates integrated into each feature implementation)
- [x] Dockerization to make sure everything runs in a container for easy portability `S`
- [x] Core Pydantic abstractions for the benchmark - Set up type-safe benchmark harness `S`
- [x] OpenAI client integration - Implement rate-limited API wrapper (COMPLETE: Tasks 1-4) `S`
- [x] Benchmark harness implementation - Basic execution framework (includes OpenAI integration testing) `M`
- [x] Basic CLI interface - Run benchmarks with configuration files `S`
- [x] Hugging Face dataset loader - Fetch and cache benchmark datasets `S`
- [x] Dataset sampling system - Enable quick validation runs `S`
- [x] FaithBench implementation - Reproduce benchmark with reference implementation `L`
  - Follow @.agent-os/instructions/core/benchmark-implementation-procedure.md
- [x] LLM caching layer - if we use the same model and the same prompt we should have a cache that returns the same result rather than calling the external llm, will save time and money when iterating on other parts of the code.  keep in memory and serialize to and from disk. `M`
  - Spec created: @.agent-os/specs/2025-09-09-faithbench-implementation/
- [x] SimpleQA implementation - Reproduce benchmark methodology `M`
  - Must follow benchmark implementation procedure
  - Research paper and repository first
  - Create comprehensive specs before coding
  - ✅ COMPLETE: AI-graded factuality evaluation working perfectly
- [x] HaluEval implementation - Import dataset and evaluation logic `M`
  - Must follow benchmark implementation procedure
  - Identify exact evaluation methodology from paper
  - Match original implementation exactly
  - ✅ COMPLETE: Binary classification across QA, dialogue, summarization
  - Note: Token limit (2033) could be made configurable for modern models
- [x] TruthfulQA implementation - Import dataset and evaluation logic `M`
  - Must follow benchmark implementation procedure
  - Identify exact evaluation methodology from paper
  - Match original implementation exactly
  - ⚠️ COMPLETE WITH LIMITATIONS: MC evaluation incompatible with Chat API (see docs/limitations/truthfulqa.md)
  - Generation-based alternative possible but requires GPU for judge model
- [ ] **SelfCheckGPT implementation** - Consistency-based hallucination detection `M` **← NEXT PRIORITY**
  - 🎯 PERFECT RESEARCH ALIGNMENT: Uses temperature variation (0.0 and 1.0) to detect hallucinations
  - Method: Generate multiple samples, check consistency across samples
  - See: docs/benchmark-assessment-2025-09-29.md for detailed analysis
  - Directly supports coherence research goals
  - API compatible, low-medium complexity (1-2 days)
  - Research value: Compare SelfCheckGPT consistency with Shogenji/Fitelson/Olsson measures
- [ ] **HalluLens implementation** - Modern comprehensive hallucination benchmark `L` **← SECOND PRIORITY**
  - 🌟 LATEST 2025 BENCHMARK: Dynamic test generation, clear taxonomy
  - Extrinsic vs intrinsic hallucination evaluation
  - Repository: https://github.com/facebookresearch/HalluLens
  - See: docs/benchmark-assessment-2025-09-29.md for detailed analysis
  - Medium complexity (3-5 days)
- [ ] Fix Faithbench implementation - use the more detailed repo: https://github.com/forrestbao/faithbench as a guide `M`

### Dependencies

- Python 3.11+ environment setup
- OpenAI API access and keys
- Reference paper collection
- Hugging Face datasets access

## Phase 2: Novel Coherence Integration & Extended Model Support

**Goal:** Implement philosophical coherence measures, k-response analysis, and support for non-OpenAI providers
**Success Criteria:** Successfully apply all three coherence measures with demonstrable variance in results, and support at least 3 additional model providers

### Features

#### Coherence Measures
- [ ] K-response generation system - Generate multiple samples per query `M`
- [ ] Shogenji coherence implementation - Implement measure with tests `M`
- [ ] Coherence-based selection algorithm - Choose best response by coherence `M`
- [ ] Fitelson coherence implementation - Implement measure with tests `M`
- [ ] Olsson coherence implementation - Implement measure with tests `M`
- [ ] Temperature variance controller - Systematic temperature exploration `S`
- [ ] Cross-response coherence analysis - Measure coherence across response sets `L`
- [ ] Temperature-coherence correlation analysis - Analyze how temperature affects coherence scores `M`
- [ ] FaithBench coherence integration - Apply coherence measures to FaithBench results `M`

#### Additional Model Providers
- [ ] Anthropic client implementation - Support Claude-3 models (opus, sonnet, haiku) `M`
- [ ] Google client implementation - Support Gemini models (pro, ultra) `M`
- [ ] Together AI client implementation - Support open-source models (Llama, Mistral) `M`
- [ ] Replicate client implementation - Alternative for open-source model access `M`
- [ ] Model provider abstraction layer - Unified interface for all providers `M`
- [ ] Provider-specific rate limiting - Handle different rate limits per provider `S`
- [ ] Cost tracking per provider - Monitor API costs across providers `S`

### Dependencies

- Phase 1 completion
- Mathematical validation of coherence implementations
- Statistical analysis framework
- API keys for additional providers (Anthropic, Google, Together AI/Replicate)

## Future Benchmark Implementations

**Note:** Each benchmark below must follow the standardized procedure before implementation begins.

### Phase 1 Benchmark Status Summary
Based on comprehensive assessment (docs/benchmark-assessment-2025-09-29.md):
- ✅ **SimpleQA**: Working perfectly - AI-graded factuality
- ✅ **HaluEval**: Working well - Binary hallucination classification
- ⚠️ **TruthfulQA**: Limited by Chat API constraints (documented)
- 🎯 **SelfCheckGPT**: Next priority - Perfect research alignment
- 🌟 **HalluLens**: Second priority - Modern comprehensive benchmark

### Planned Benchmarks (Lower Priority)
- [ ] **Vectara Hallucination Leaderboard** - Summarization faithfulness
  - Industry-recognized leaderboard
  - Tests faithfulness specifically
  - Consider if focusing on summarization tasks
- [ ] **FEVER** - Fact extraction and verification
  - Complex multi-step verification process
  - Requires claim verification methodology
  - Lower priority - complexity vs value trade-off

### Benchmark Research Resources
- arXiv for latest papers
- ACL Anthology for conference versions
- GitHub for official implementations
- Papers with Code for baselines

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

#### High Priority (Blocking Progress)
- [ ] Update TruthfulQA tests - Tests written for buggy label behavior (label=0 vs label=1), need to update assertions `S`
- [ ] Verify cache temperature handling - Ensure temperature is properly included in cache key for SelfCheckGPT multi-temperature sampling `S`

#### Medium Priority (Quality Improvements)
- [ ] Make HaluEval token limit configurable - Currently hardcoded at 2033 tokens, should be 8K-16K for modern models `S`

#### Lower Priority (Test Fixes)
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
- [ ] Remove batch API implementation - Batch API has 24hr SLA making it unsuitable for interactive use, remove implementation and simplify OpenAI client `M`
- [ ] Add CLI support for config files - FaithBench CLI should accept `--config` flag to use YAML config files instead of individual args `M`
- [ ] Add unit tests for binary classification logic - Test the consistent vs inconsistent mapping for all annotation types `S`
- [ ] Document example configurations - Create `examples/` directory with documented config files for each benchmark `S`
- [ ] Add debug/verbose mode to CLI - Help users understand what's happening during benchmark runs `S`

### Notes

- **Keep this list updated:** When tests are skipped or marked as xfail, add them here for future resolution
- **Review periodically:** Check this list during sprint planning or maintenance windows
- **Document fixes:** When addressing items, document the solution for future reference
