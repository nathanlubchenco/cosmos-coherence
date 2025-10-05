"""SelfCheckGPT benchmark implementation with temperature-variant consistency evaluation.

This benchmark implements the SelfCheckGPT methodology from Manakul et al. (2023):
https://arxiv.org/abs/2303.08896

The benchmark evaluates hallucinations by checking consistency across multiple
stochastic samples. Key insight: factual statements remain consistent across samples,
while hallucinated content shows variation.

Methodology:
1. Generate 1 baseline response at temperature 0.0 (deterministic)
2. Generate 5 sample responses at temperature 1.0 (stochastic)
3. Evaluate sentence-level consistency using NLI (Natural Language Inference)
4. Lower consistency = higher hallucination probability

Note: Paper uses 20 samples; we use 5 for Phase 1 (cost/speed trade-off).
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

from cosmos_coherence.benchmarks.models.base import BaseDatasetItem
from cosmos_coherence.benchmarks.models.datasets import SelfCheckGPTItem
from cosmos_coherence.harness.base_benchmark import BenchmarkEvaluationResult
from cosmos_coherence.harness.base_benchmark_hf import HuggingFaceEnabledBenchmark
from cosmos_coherence.llm.models import ModelResponse
from cosmos_coherence.llm.openai_client import OpenAIClient

try:
    import torch
    from selfcheckgpt.modeling_selfcheck import SelfCheckNLI

    SELFCHECKGPT_AVAILABLE = True
except ImportError:
    SELFCHECKGPT_AVAILABLE = False
    torch = None  # type: ignore
    SelfCheckNLI = None  # type: ignore


class SelfCheckGPTBenchmark(HuggingFaceEnabledBenchmark):
    """SelfCheckGPT benchmark for consistency-based hallucination detection.

    Uses multi-temperature sampling to detect hallucinations:
    - 1 sample at temperature 0.0 (baseline)
    - 5 samples at temperature 1.0 (stochastic variations)

    Evaluates consistency using NLI (Natural Language Inference) to determine
    whether facts in the baseline are supported by the sample responses.
    """

    def __init__(
        self,
        client: OpenAIClient,
        num_samples: int = 5,
        baseline_temperature: float = 0.0,
        sample_temperature: float = 1.0,
        cache_dir: Optional[Path] = None,
        **kwargs,
    ):
        """Initialize SelfCheckGPT benchmark.

        Args:
            client: OpenAI client for generating responses
            num_samples: Number of stochastic samples to generate (default: 5)
            baseline_temperature: Temperature for baseline generation (default: 0.0)
            sample_temperature: Temperature for samples (default: 1.0)
            cache_dir: Directory for caching responses
                (default: ~/.cache/cosmos_coherence/selfcheckgpt/)
            **kwargs: Additional arguments for parent class
        """
        # Extract use_huggingface flag if present
        use_huggingface = kwargs.pop("use_huggingface", True)

        # Default to HuggingFace dataset
        if "hf_dataset_name" not in kwargs and use_huggingface:
            kwargs["hf_dataset_name"] = "selfcheckgpt"

        super().__init__(**kwargs)

        if client is None:
            raise ValueError("SelfCheckGPT requires an OpenAI client for generation")

        self.client = client
        self.num_samples = num_samples
        self.baseline_temperature = baseline_temperature
        self.sample_temperature = sample_temperature

        # Set up cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "cosmos_coherence" / "selfcheckgpt"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # NLI scorer will be initialized lazily on first use
        self._nli_scorer = None
        self._device = None

    def _get_nli_scorer(self):
        """Lazy initialization of NLI scorer.

        Returns:
            SelfCheckNLI instance for consistency evaluation
        """
        if self._nli_scorer is None:
            if not SELFCHECKGPT_AVAILABLE:
                raise ImportError(
                    "selfcheckgpt library is not installed. "
                    "Please install it with: pip install selfcheckgpt torch"
                )

            # Determine device (GPU if available, else CPU)
            if self._device is None:
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Initialize NLI scorer
            self._nli_scorer = SelfCheckNLI(device=self._device)

        return self._nli_scorer

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple regex.

        This is a lightweight sentence splitter that works reasonably well
        for English text without requiring spacy models.

        Args:
            text: Input text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting on period, exclamation, question mark
        # followed by space or end of string
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        # Filter out empty sentences
        return [s.strip() for s in sentences if s.strip()]

    def get_prompt(self, item: BaseDatasetItem) -> str:
        """Format dataset item into prompt for biography generation.

        Args:
            item: SelfCheckGPTItem with topic (person's name)

        Returns:
            Prompt string for biography generation
        """
        if not isinstance(item, SelfCheckGPTItem):
            raise ValueError(f"Expected SelfCheckGPTItem, got {type(item)}")

        # Simple prompt: generate a short biography
        return f"Write a short biography about {item.topic}."

    async def generate_multiple_samples(
        self, prompt: str, max_tokens: int = 200
    ) -> List[ModelResponse]:
        """Generate multiple samples at different temperatures.

        This is the core multi-temperature sampling for SelfCheckGPT.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens per response

        Returns:
            List of responses: [baseline, sample1, sample2, ..., sample_n]
            where baseline is at temp 0.0 and samples are at temp 1.0
        """
        responses = []

        # Generate baseline at temperature 0.0 (deterministic)
        baseline = await self.client.generate_response(
            prompt, temperature=self.baseline_temperature, max_tokens=max_tokens
        )
        responses.append(baseline)

        # Generate N samples at temperature 1.0 (stochastic)
        for _ in range(self.num_samples):
            sample = await self.client.generate_response(
                prompt,
                temperature=self.sample_temperature,
                max_tokens=max_tokens,
            )
            responses.append(sample)

        return responses

    def evaluate_response(
        self, response: str, ground_truth: str, item: BaseDatasetItem
    ) -> BenchmarkEvaluationResult:
        """Evaluate a response using NLI-based consistency.

        Note: This method is not fully compatible with SelfCheckGPT's multi-sample
        approach. Use evaluate_item_with_consistency() for proper evaluation.

        Args:
            response: Generated response
            ground_truth: Ground truth (not used in SelfCheckGPT)
            item: Original dataset item

        Returns:
            Evaluation result with placeholder values
        """
        # SelfCheckGPT requires multiple samples for consistency evaluation
        # This single-response evaluation is a simplified placeholder
        return BenchmarkEvaluationResult(
            score=0.0,
            is_correct=False,
            original_metric_score=0.0,
            metadata={
                "message": (
                    "Use evaluate_item_with_consistency() for proper SelfCheckGPT evaluation"
                ),
                "baseline_text": response,
            },
        )

    async def evaluate_item_with_consistency(self, item: SelfCheckGPTItem) -> Dict:
        """Evaluate an item using multi-temperature consistency checking.

        This implements the full SelfCheckGPT methodology:
        1. Generate baseline at temp 0.0
        2. Generate N samples at temp 1.0
        3. Split baseline into sentences
        4. Use NLI to check consistency of each sentence against samples
        5. Calculate per-sentence and aggregate consistency scores

        Args:
            item: SelfCheckGPTItem to evaluate

        Returns:
            Dictionary with:
                - topic: Person's name
                - baseline: Baseline text at temp 0.0
                - samples: List of sample texts at temp 1.0
                - sentences: List of sentences from baseline
                - sentence_scores: Per-sentence NLI scores (higher = hallucination)
                - aggregate_score: Mean consistency score
                - num_samples: Number of samples generated
        """
        # Generate prompt
        prompt = self.get_prompt(item)

        # Generate multiple samples (1 baseline + N samples)
        responses = await self.generate_multiple_samples(prompt)

        # Extract text from responses
        baseline_text = responses[0].content
        sample_texts = [r.content for r in responses[1:]]

        # Split baseline into sentences
        sentences = self._split_into_sentences(baseline_text)

        # Get NLI scorer
        nli_scorer = self._get_nli_scorer()

        # Calculate consistency scores for each sentence
        # Higher scores indicate potential hallucination (contradiction)
        sentence_scores_raw = nli_scorer.predict(sentences=sentences, sampled_passages=sample_texts)

        # Convert to list if numpy array
        if hasattr(sentence_scores_raw, "tolist"):
            sentence_scores = sentence_scores_raw.tolist()
        else:
            sentence_scores = list(sentence_scores_raw) if sentence_scores_raw is not None else []

        # Calculate aggregate score (mean of sentence scores)
        aggregate_score = (
            float(sum(sentence_scores) / len(sentence_scores)) if sentence_scores else 0.0
        )

        return {
            "topic": item.topic,
            "baseline": baseline_text,
            "samples": sample_texts,
            "sentences": sentences,
            "sentence_scores": sentence_scores,
            "aggregate_score": aggregate_score,
            "num_samples": len(sample_texts),
        }

    def get_cache_statistics(self) -> Dict:
        """Get cache statistics from the OpenAI client.

        Returns:
            Dictionary with cache statistics
        """
        stats = self.client.get_cache_statistics()
        return {
            "total_requests": stats.total_requests,
            "cache_hits": stats.cache_hits,
            "cache_misses": stats.cache_misses,
            "hit_rate": stats.hit_rate,
            "tokens_saved": stats.tokens_saved,
        }

    def save_cache(self) -> None:
        """Save cache to disk."""
        self.client.save_cache()

    def get_baseline_metrics(self) -> Dict[str, float]:
        """Return published baseline metrics for reproducibility validation.

        From Manakul et al. (2023), Table 2: SelfCheck-NLI results.

        Returns:
            Dictionary with baseline AUC-PR scores
        """
        return {
            "auc_pr_non_factual": 0.925,  # Paper reports 92.50% with 20 samples
            "auc_pr_factual": 0.661,  # Paper reports 66.08% with 20 samples
            "target_auc_pr": 0.82,  # Our target with 5 samples (90% of paper's result)
        }

    def get_original_prompts(self) -> List[str]:
        """Return example prompts from original paper for format validation.

        Returns:
            List of example prompts
        """
        return [
            "Write a short biography about Albert Einstein.",
            "Write a short biography about Marie Curie.",
            "Write a short biography about Isaac Newton.",
        ]

    def validate_config(self, config: Dict) -> None:
        """Validate benchmark-specific configuration.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If configuration is invalid
        """
        if "num_samples" in config:
            num_samples = config["num_samples"]
            if not isinstance(num_samples, int) or num_samples < 1:
                raise ValueError(f"num_samples must be a positive integer, got {num_samples}")

        if "baseline_temperature" in config:
            temp = config["baseline_temperature"]
            if temp != 0.0:
                raise ValueError(f"baseline_temperature must be 0.0 for deterministic, got {temp}")

        if "sample_temperature" in config:
            temp = config["sample_temperature"]
            if temp != 1.0:
                raise ValueError(f"sample_temperature must be 1.0 for stochastic, got {temp}")

    @property
    def benchmark_name(self) -> str:
        """Return the benchmark identifier.

        Returns:
            Benchmark name
        """
        return "selfcheckgpt"

    @property
    def paper_reference(self) -> str:
        """Return the original paper reference for this benchmark.

        Returns:
            Paper citation
        """
        return (
            "Manakul, P., Liusie, A., & Gales, M. J. F. (2023). "
            "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection "
            "for Generative Large Language Models. "
            "EMNLP 2023. arXiv:2303.08896"
        )

    def get_evaluation_method(self) -> str:
        """Return description of the evaluation methodology.

        Returns:
            Evaluation method description
        """
        return (
            "Multi-temperature consistency checking via NLI: "
            f"Generate 1 baseline at temp {self.baseline_temperature} + "
            f"{self.num_samples} samples at temp {self.sample_temperature}, "
            "then use DeBERTa-v3-large NLI model to check sentence-level consistency. "
            "Higher inconsistency scores indicate potential hallucinations."
        )
