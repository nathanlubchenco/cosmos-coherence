# Benchmark Implementation Procedure

This document outlines the standardized procedure for implementing each benchmark, based on lessons learned from FaithBench implementation.

## Phase 1: Research & Documentation (Before Any Code)

### Step 1: Paper Discovery
1. **Find the primary paper**:
   - Search arXiv for the original paper
   - Check ACL Anthology for conference versions
   - Look for multiple versions (conference, journal, extended)
   - Download and save PDF locally for reference

2. **Identify the most recent version**:
   - Conference papers often have more recent versions
   - Extended versions may have additional details
   - Check paper dates and version numbers

### Step 2: Repository Analysis
1. **Locate official repository**:
   - Check paper for GitHub links
   - Search GitHub for benchmark name
   - Verify it's the official implementation

2. **Analyze repository structure**:
   - Dataset format and location
   - Evaluation scripts
   - Model configurations
   - Example usage

3. **Download key resources**:
   - Clone repository locally
   - Download dataset samples
   - Save evaluation scripts for reference

### Step 3: Extract Implementation Details
1. **From the paper, extract**:
   - Exact models tested (with versions)
   - Evaluation methodology
   - Metrics used
   - Prompt templates
   - Expected baseline results
   - Temperature/parameter settings

2. **From the repository, extract**:
   - Dataset structure (JSON/CSV/etc.)
   - Data loading approach
   - Evaluation pipeline
   - Scoring implementation
   - Any special preprocessing

### Step 4: Model Configuration Analysis
1. **Identify model requirements**:
   - Which models were tested in paper
   - Which OpenAI models to use for Phase 1
   - Temperature support vs reasoning models
   - Required API parameters

2. **Check model availability**:
   - Verify API access for each model
   - Note limited-access models (o1, o3 series)
   - Plan fallbacks for unavailable models

## Phase 2: Specification Creation

### Step 5: Create Comprehensive Specs
Using `/create-spec` command, create:

1. **research-references.md**:
   - Paper details (title, authors, arXiv ID)
   - Repository URL
   - Key methodology points
   - Model configurations from paper
   - Expected results

2. **model-configs.md**:
   - OpenAI models for Phase 1
   - Exact parameter settings
   - Temperature variations (if applicable)
   - Reasoning model constraints
   - NotImplementedError for unsupported models

3. **technical-spec.md**:
   - Dataset loading approach
   - Evaluation pipeline
   - Metrics implementation
   - Integration points

4. **spec.md & spec-lite.md**:
   - User stories
   - Scope definition
   - Expected deliverables

### Step 6: Scope Management
1. **Phase 1 (Reproduction)**:
   - Focus on reproducing paper results
   - OpenAI models only
   - No coherence integration yet
   - Match paper's evaluation exactly

2. **Phase 2 (Enhancement)**:
   - Additional model providers
   - Coherence measure integration
   - Temperature variation analysis
   - Cross-benchmark comparisons

## Phase 3: Implementation Planning

### Step 7: Task Breakdown
Use `/create-tasks` to generate implementation tasks:

1. **Data Pipeline Tasks**:
   - Dataset loader implementation
   - Data model creation
   - Validation and preprocessing

2. **Evaluation Tasks**:
   - Benchmark class implementation
   - Prompt template creation
   - Metric calculation

3. **Integration Tasks**:
   - CLI command addition
   - Configuration integration
   - Result storage

### Step 8: Validation Planning
1. **Define success criteria**:
   - Results within X% of paper baseline
   - All metrics properly calculated
   - Graceful handling of unavailable models

2. **Create test cases**:
   - Unit tests for data loading
   - Integration tests for evaluation
   - End-to-end benchmark runs

## Phase 4: Implementation

### Step 9: Follow Research-Driven Development
1. **Implement exactly as researched**:
   - Use dataset format from repository
   - Match prompt templates from paper
   - Replicate evaluation methodology

2. **Document deviations**:
   - Any changes from paper approach
   - Reasons for modifications
   - Impact on results

### Step 10: Validation & Testing
1. **Compare against baselines**:
   - Run with paper's model configs
   - Verify metrics match expected ranges
   - Document any discrepancies

2. **Handle edge cases**:
   - Missing API access
   - Dataset loading errors
   - Model-specific constraints

## Checklist Template for Each Benchmark

```markdown
## [Benchmark Name] Implementation Checklist

### Research Phase
- [ ] Found and downloaded primary paper
- [ ] Checked for recent versions (ACL, NAACL, etc.)
- [ ] Located official GitHub repository
- [ ] Analyzed dataset format and structure
- [ ] Extracted model configurations from paper
- [ ] Identified evaluation methodology
- [ ] Downloaded dataset samples

### Documentation Phase
- [ ] Created research-references.md
- [ ] Created model-configs.md
- [ ] Updated technical-spec.md
- [ ] Defined Phase 1 scope (OpenAI only)
- [ ] Defined Phase 2 enhancements

### Implementation Phase
- [ ] Implemented dataset loader
- [ ] Created data models
- [ ] Built evaluation pipeline
- [ ] Added CLI integration
- [ ] Implemented metrics

### Validation Phase
- [ ] Tested with all specified models
- [ ] Compared results to paper baselines
- [ ] Handled unavailable models gracefully
- [ ] Documented any deviations
- [ ] All tests passing
```

## Key Principles

1. **Research First**: Always understand the paper and code before implementing
2. **Match the Paper**: Reproduce exactly what was published before enhancing
3. **Document Everything**: Keep detailed references for future maintainers
4. **Phase Appropriately**: OpenAI-only in Phase 1, enhancements in Phase 2
5. **Handle Gracefully**: Account for limited model access and API constraints
6. **Test Thoroughly**: Validate against published baselines

## Common Pitfalls to Avoid

1. **Don't assume task type**: FaithBench was summarization, not Q&A
2. **Check model constraints**: Reasoning models don't support temperature
3. **Verify dataset format**: Use actual structure from repository
4. **Match evaluation exactly**: Use paper's metrics and methodology
5. **Account for access limits**: o1/o3 models may not be available

## Next Benchmarks to Implement

Following this procedure for:
1. **SimpleQA**: Search for paper, repository, and implementation details
2. **TruthfulQA**: Find dataset structure and evaluation approach
3. **HaluEval**: Analyze hallucination detection methodology
4. **FEVER**: Understand fact verification pipeline

Each benchmark should follow this exact procedure to ensure consistency and completeness.
