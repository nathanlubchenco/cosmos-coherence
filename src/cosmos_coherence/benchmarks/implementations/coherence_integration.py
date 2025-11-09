"""Coherence integration for question variation scoring.

This module adapts SelfCheckGPT coherence variants for question-variation
coherence scoring rather than answer-variation hallucination detection.

Key adaptation:
- Traditional use: predict(sentences=[answer], sampled_passages=[answer_samples])
- Question-variation use: predict(sentences=[question], sampled_passages=[question_variations])

The coherence score measures how well-formed and unambiguous the question is
across paraphrasing. High coherence suggests a well-formed question; low coherence
suggests an ambiguous or ill-posed question.
"""

import logging
from typing import Union, List, Optional
import numpy as np

# Import coherence variants from selfcheckgpt
from selfcheckgpt import SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson
from selfcheckgpt.utils_coherence import normalize_coherence_scores

logger = logging.getLogger(__name__)


class CoherenceIntegration:
    """Integration layer for coherence-based question variation scoring.

    This class provides:
    1. Initialization helpers for coherence variants
    2. Adaptation of coherence API for question-variation use case
    3. Score normalization and interpretation
    4. Error handling for numerical instability

    Example:
        >>> integration = CoherenceIntegration(model="gpt-4o-mini")
        >>> variant = integration._init_coherence_variant("shogenji")
        >>> score = integration.score_question_variations(
        ...     variant,
        ...     "What is the capital of France?",
        ...     ["Which city is France's capital?", "What city is the French capital?"]
        ... )
        >>> print(f"Question coherence: {score:.2f}")
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        """Initialize coherence integration.

        Args:
            model: OpenAI model name for probability extraction (default: gpt-4o-mini)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
        """
        self.model = model
        self.api_key = api_key

        # Error tracking for metrics
        self._error_count = 0
        self._total_evaluations = 0

    def _init_coherence_variant(
        self, variant_name: str
    ) -> Union[SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson]:
        """Initialize coherence variant by name.

        Args:
            variant_name: Name of coherence variant ("shogenji", "fitelson", or "olsson")

        Returns:
            Initialized coherence variant instance

        Raises:
            ValueError: If variant_name is not recognized

        Example:
            >>> integration = CoherenceIntegration()
            >>> shogenji = integration._init_coherence_variant("shogenji")
            >>> fitelson = integration._init_coherence_variant("fitelson")
            >>> olsson = integration._init_coherence_variant("olsson")
        """
        variant_map = {
            "shogenji": SelfCheckShogenji,
            "fitelson": SelfCheckFitelson,
            "olsson": SelfCheckOlsson,
        }

        if variant_name.lower() not in variant_map:
            valid_variants = ", ".join(variant_map.keys())
            raise ValueError(
                f"Unknown coherence variant: {variant_name}. "
                f"Valid variants are: {valid_variants}"
            )

        variant_class = variant_map[variant_name.lower()]

        # Initialize with model and optional api_key
        if self.api_key:
            return variant_class(model=self.model, api_key=self.api_key)
        else:
            return variant_class(model=self.model)

    def score_question_variations(
        self,
        coherence_variant: Union[SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson],
        original_question: str,
        question_variations: List[str],
        fallback_score: Optional[float] = None,
        verbose: bool = False,
    ) -> float:
        """Score coherence of question variations.

        Adapts the coherence API for question-variation use case:
        - sentences = [original_question] (single-element list)
        - sampled_passages = question_variations (list of paraphrases)

        The coherence score indicates how well-formed and unambiguous the question is.
        High coherence suggests the question maintains consistency across paraphrasing.
        Low coherence suggests ambiguity or ill-formed structure.

        Args:
            coherence_variant: Initialized coherence variant (Shogenji, Fitelson, or Olsson)
            original_question: Original question to evaluate
            question_variations: List of paraphrased versions of the question
            fallback_score: Score to return on error (default: None, raises error)
            verbose: Enable verbose logging for debugging

        Returns:
            Coherence score in [0, 1] range (normalized)
            - 1.0 = perfect coherence (well-formed question)
            - 0.0 = no coherence (ambiguous/ill-posed question)

        Raises:
            ValueError: If coherence calculation fails and fallback_score is None

        Example:
            >>> variant = SelfCheckShogenji(model="gpt-4o-mini")
            >>> score = integration.score_question_variations(
            ...     variant,
            ...     "What is the capital of France?",
            ...     ["Which city is France's capital?", "What city is the French capital?"]
            ... )
        """
        self._total_evaluations += 1

        try:
            # Call coherence variant's predict() method
            # Adapt for question-variation use case:
            # - sentences: [original_question] (what we're evaluating)
            # - sampled_passages: question_variations (what we're comparing against)
            coherence_scores = coherence_variant.predict(
                sentences=[original_question],
                sampled_passages=question_variations,
                verbose=verbose,
            )

            # Extract score from result
            # predict() returns np.ndarray with one score per sentence
            # We passed one sentence, so extract the first (and only) score
            if isinstance(coherence_scores, np.ndarray):
                score = float(coherence_scores[0])
            else:
                score = float(coherence_scores)

            # Ensure score is in [0, 1] range
            # The coherence variants already apply normalization, but clamp for safety
            score = np.clip(score, 0.0, 1.0)

            if verbose:
                logger.info(
                    f"Question coherence score: {score:.4f} "
                    f"(evaluated against {len(question_variations)} variations)"
                )

            return score

        except Exception as e:
            self._error_count += 1
            logger.error(
                f"Error calculating coherence score: {e}\n"
                f"Question: {original_question[:100]}...\n"
                f"Variations: {len(question_variations)}"
            )

            if fallback_score is not None:
                logger.warning(f"Using fallback score: {fallback_score}")
                return fallback_score
            else:
                # Re-raise with context
                raise ValueError(
                    f"Coherence calculation failed for question: {original_question[:100]}..."
                ) from e

    def score_answer_sketches(
        self,
        coherence_variant: Union[SelfCheckShogenji, SelfCheckFitelson, SelfCheckOlsson],
        original_answer: str,
        variation_answers: List[str],
        fallback_score: Optional[float] = None,
        verbose: bool = False,
    ) -> float:
        """Score coherence of answer sketches from question variations.

        This is the theoretically sound approach: answers have truth values,
        so probabilistic coherence measures are semantically appropriate.

        Flow:
        1. Generate question variations
        2. Generate answer sketch for original question
        3. Generate answer sketch for each variation
        4. Apply coherence to answer sketches (this method)

        High coherence in answers suggests the question is well-formed and unambiguous.
        Low coherence suggests the question variations elicit inconsistent answers.

        Args:
            coherence_variant: Initialized coherence variant (Shogenji, Fitelson, or Olsson)
            original_answer: Answer sketch for the original question
            variation_answers: List of answer sketches for question variations
            fallback_score: Score to return on error (default: None, raises error)
            verbose: Enable verbose logging for debugging

        Returns:
            Coherence score in [0, 1] range (normalized)
            - 1.0 = perfect coherence (consistent answers across variations)
            - 0.0 = no coherence (inconsistent/contradictory answers)

        Raises:
            ValueError: If coherence calculation fails and fallback_score is None

        Example:
            >>> variant = SelfCheckShogenji(model="gpt-4o-mini")
            >>> score = integration.score_answer_sketches(
            ...     variant,
            ...     "Paris is the capital of France.",
            ...     ["Paris is France's capital.", "The capital is Paris."]
            ... )
        """
        self._total_evaluations += 1

        try:
            # Call coherence variant's predict() method
            # sentences: [original_answer] (what we're evaluating)
            # sampled_passages: variation_answers (what we're comparing against)
            coherence_scores = coherence_variant.predict(
                sentences=[original_answer],
                sampled_passages=variation_answers,
                verbose=verbose,
            )

            # Extract score from result
            if isinstance(coherence_scores, np.ndarray):
                score = float(coherence_scores[0])
            else:
                score = float(coherence_scores)

            # Ensure score is in [0, 1] range
            score = np.clip(score, 0.0, 1.0)

            if verbose:
                logger.info(
                    f"Answer sketch coherence score: {score:.4f} "
                    f"(evaluated against {len(variation_answers)} answer variations)"
                )

            return score

        except Exception as e:
            self._error_count += 1
            logger.error(
                f"Error calculating coherence score: {e}\n"
                f"Answer: {original_answer[:100]}...\n"
                f"Variations: {len(variation_answers)}"
            )

            if fallback_score is not None:
                logger.warning(f"Using fallback score: {fallback_score}")
                return fallback_score
            else:
                raise ValueError(
                    f"Coherence calculation failed for answer: {original_answer[:100]}..."
                ) from e

    def normalize_scores(
        self, scores: Union[List[float], np.ndarray]
    ) -> np.ndarray:
        """Normalize coherence scores to [0, 1] range.

        Uses min-max normalization from selfcheckgpt.utils_coherence.
        Handles edge case where all scores are identical (returns 0.5).

        Args:
            scores: List or array of coherence scores

        Returns:
            Normalized scores in [0, 1] range

        Example:
            >>> scores = [0.2, 0.5, 0.8, 0.3]
            >>> normalized = integration.normalize_scores(scores)
            >>> print(normalized)  # [0.0, 0.5, 1.0, 0.17]
        """
        scores_array = np.asarray(scores)

        # Use normalize_coherence_scores from selfcheckgpt
        normalized = normalize_coherence_scores(scores_array)

        return normalized

    def get_error_rate(self) -> float:
        """Get error rate for coherence calculations.

        Returns:
            Error rate (errors / total evaluations)
        """
        if self._total_evaluations == 0:
            return 0.0
        return self._error_count / self._total_evaluations

    def get_statistics(self) -> dict:
        """Get coherence integration statistics.

        Returns:
            Dictionary with error tracking metrics
        """
        return {
            "total_evaluations": self._total_evaluations,
            "error_count": self._error_count,
            "error_rate": self.get_error_rate(),
        }
