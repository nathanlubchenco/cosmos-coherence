# Benchmark Research Tracker

This document tracks the research status for each benchmark implementation.

## Implementation Status

### ✅ FaithBench
- **Status**: Specs complete, ready for implementation
- **Paper**: arXiv:2410.13210
- **Repository**: https://github.com/vectara/FaithBench
- **Models**: GPT-4-Turbo, GPT-4o, o1-mini, o3-mini
- **Task Type**: Summarization (not Q&A)
- **Key Learning**: Must check task type carefully

### 🔄 SimpleQA
- **Status**: Pending research
- **Next Steps**:
  1. Search for SimpleQA paper on arXiv
  2. Find official GitHub repository
  3. Analyze dataset format
  4. Identify evaluation methodology
  5. Create specs following procedure

### 🔄 TruthfulQA
- **Status**: Pending research
- **Known Info**: Popular benchmark, likely has HuggingFace dataset
- **Next Steps**:
  1. Find TruthfulQA paper
  2. Check for multiple versions
  3. Analyze evaluation approach
  4. Map to our framework
  5. Create comprehensive specs

### 🔄 HaluEval
- **Status**: Pending research
- **Next Steps**:
  1. Research hallucination evaluation methodology
  2. Find official implementation
  3. Understand metric calculations
  4. Create implementation plan

### 🔄 FEVER
- **Status**: Pending research
- **Known Info**: Fact verification, multi-step process
- **Next Steps**:
  1. Study FEVER paper and dataset
  2. Understand claim verification pipeline
  3. Map to our evaluation framework

## Research Checklist Template

For each new benchmark, complete this checklist:

```markdown
### [Benchmark Name]
- [ ] Paper found and downloaded
- [ ] Recent versions checked (conference proceedings)
- [ ] GitHub repository located
- [ ] Dataset format analyzed
- [ ] Evaluation methodology understood
- [ ] Model requirements identified
- [ ] Prompt templates extracted
- [ ] Baseline results noted
- [ ] OpenAI model mapping completed
- [ ] Specs created using /create-spec
```

## Lessons Learned

### From FaithBench Implementation:
1. **Always verify task type** - We initially assumed Q&A but it was summarization
2. **Check for recent paper versions** - Conference papers may be more recent
3. **Understand model constraints** - Reasoning models don't support temperature
4. **Document model availability** - o1/o3 models require special access
5. **Match evaluation exactly** - Use paper's metrics and methodology

### Best Practices:
1. **Download papers locally** - Keep for reference during implementation
2. **Clone repositories** - Have code available for inspection
3. **Create comprehensive specs** - Document everything before coding
4. **Test incrementally** - Validate each component separately
5. **Compare to baselines** - Always check against paper results

## Resource Links

### Paper Sources:
- arXiv: https://arxiv.org/
- ACL Anthology: https://aclanthology.org/
- Papers with Code: https://paperswithcode.com/
- Google Scholar: https://scholar.google.com/

### Dataset Sources:
- HuggingFace Datasets: https://huggingface.co/datasets
- GitHub (search for benchmark name)
- Official benchmark websites

### Implementation References:
- Our procedure: @.agent-os/instructions/core/benchmark-implementation-procedure.md
- FaithBench specs: @.agent-os/specs/2025-09-09-faithbench-implementation/
- Model configs template: @.agent-os/specs/2025-09-09-faithbench-implementation/sub-specs/model-configs.md
