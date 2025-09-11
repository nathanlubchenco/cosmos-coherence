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
   - Metrics used (and their EXACT definitions)
   - Prompt templates (check referenced papers too!)
   - Expected baseline results (all reported metrics)
   - Temperature/parameter settings
   - Binary vs multi-class task definition
   - Class definitions (what is "positive" vs "negative")

2. **From the repository, extract**:
   - Dataset structure (JSON/CSV/etc.)
   - Data loading approach
   - Evaluation pipeline
   - Scoring implementation
   - Any special preprocessing
   - **CRITICAL**: Check utility scripts (binarize.py, load.py, etc.)
   - Aggregation strategies for multiple annotations
   - Label mapping and transformation logic

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
- [ ] Checked referenced papers for methodology details
- [ ] Located official GitHub repository
- [ ] Analyzed dataset format and structure
- [ ] **Examined ALL utility scripts** (binarize.py, load.py, etc.)
- [ ] Extracted model configurations from paper
- [ ] Found EXACT prompts (never create your own!)
- [ ] Identified evaluation methodology
- [ ] Understood binary vs multi-class task definition
- [ ] Verified metric definitions (F1-macro, balanced accuracy, etc.)
- [ ] Noted aggregation strategies for multiple annotations
- [ ] Downloaded dataset samples
- [ ] Documented expected baseline results for ALL metrics

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

1. **Don't assume task type**: FaithBench was summarization evaluation, not Q&A
2. **Check model constraints**: Reasoning models don't support temperature
3. **Verify dataset format**: Use actual structure from repository
4. **Match evaluation exactly**: Use paper's metrics and methodology
5. **Account for access limits**: o1/o3 models may not be available
6. **Paper vs Code conflicts**: When paper and code disagree, code is ground truth
7. **Never create your own prompts**: Find the EXACT prompts (may be in referenced papers)
8. **Metric ambiguity**: "F1-macro" may mean different things (binary vs multi-class)
9. **Terminology confusion**: "Positive class" might mean different things
10. **Hidden preprocessing**: Check ALL utility scripts, not just main code
11. **Aggregation matters**: Multiple annotations need careful handling (worst/majority/best)
12. **Binary isn't simple**: Binary evaluation may still involve complex multi-class mapping

## Debugging When Results Don't Match

### Systematic Debugging Approach
1. **Check fundamentals first**:
   - Verify data is loaded correctly (print samples)
   - Confirm label mapping matches paper/code
   - Test with 5-10 samples manually

2. **Analyze prediction patterns**:
   - Look for systematic biases (e.g., over-predicting one class)
   - Check per-class performance breakdown
   - Compare label distributions with paper

3. **Verify intermediate values**:
   - Print raw model outputs before processing
   - Check pre vs post aggregation metrics
   - Trace through metric calculations step-by-step

4. **Common issues from FaithBench**:
   - Model predicted "consistent" 87% of the time (way too high)
   - F1 for questionable class was 0.036 (nearly random)
   - Balanced accuracy close (0.545 vs 0.577) but F1-macro far (0.244 vs 0.436)
   - This pattern suggests evaluation approach issues, not data issues

### Accept Imperfect Reproduction
- **Close enough is OK**: Within 5-10% on main metrics
- **Document differences**: Note what doesn't match and hypotheses why
- **Directional correctness**: Model rankings should match even if absolute values don't
- **Focus on learning**: Understanding WHY differences exist is valuable

## Lessons from FaithBench Implementation

### What We Learned
1. **Binary mapping was correct**: CONSISTENT+BENIGN=1, QUESTIONABLE+HALLUCINATED=0
2. **Prompt from referenced paper**: Not in main paper but in citation
3. **F1-macro ambiguity**: Averaged across 4 classes, not binary
4. **Model struggles with nuance**: Poor performance on questionable/benign categories
5. **Utility scripts are critical**: binarize.py contained key logic

### Remaining Mysteries
- Why F1-macro is still significantly lower (0.244 vs 0.436)
- Whether paper uses different evaluation for questionable/benign
- Possible unreported preprocessing or filtering

## Next Benchmarks to Implement

Following this procedure for:
1. **SimpleQA**: Search for paper, repository, and implementation details
2. **TruthfulQA**: Find dataset structure and evaluation approach
3. **HaluEval**: Analyze hallucination detection methodology
4. **FEVER**: Understand fact verification pipeline

Each benchmark should follow this exact procedure to ensure consistency and completeness.
