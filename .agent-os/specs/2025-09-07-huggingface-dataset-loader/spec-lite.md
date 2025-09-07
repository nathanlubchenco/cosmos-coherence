# Spec Summary (Lite)

Implement a Hugging Face dataset loader integrated with BaseBenchmark.load_dataset() to automatically fetch, cache, and convert five specific hallucination benchmark datasets (FaithBench, SimpleQA, TruthfulQA, FEVER, HaluEval) to Pydantic models. The loader uses local file caching in .cache/datasets/, fails fast on validation errors, and provides progress indicators for large downloads.
