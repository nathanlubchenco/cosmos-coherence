# Spec Summary (Lite)

Implement SelfCheckGPT benchmark for consistency-based hallucination detection using temperature variation (1 sample at temp 0.0, 5 samples at temp 1.0). Use NLI variant to evaluate sentence-level consistency between baseline and samples. Directly supports coherence research by enabling comparison between consistency measures and philosophical coherence measures (Shogenji/Fitelson/Olsson).

**Research Foundation**: See `research-references.md` for complete paper details (Manakul et al. 2023, EMNLP, arXiv:2303.08896).

**API Compatibility**: ✅ Verified compatible with OpenAI Chat Completions API.

**Baseline Target**: AUC-PR >82% for non-factual detection (paper achieves 92.50 with 20 samples; we use 5 for Phase 1).

**Critical Requirements**: Proper multi-temperature caching (verified: temperature in cache key), validation against paper baselines, explicit deviation documentation (5 vs 20 samples).
