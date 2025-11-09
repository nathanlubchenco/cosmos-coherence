"""Question variation generator for coherence-based hallucination detection."""

import asyncio
import logging
from typing import List, Optional

from cosmos_coherence.llm.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


class QuestionVariationGenerator:
    """Generate paraphrased versions of questions for coherence analysis.

    This generator creates diverse question variations using temperature-based
    sampling to support coherence-guided hallucination detection. The variations
    are used to assess question quality by measuring coherence across paraphrases.

    The core hypothesis: well-formed questions maintain high coherence across
    paraphrasing, while ambiguous or ill-posed questions show low coherence.
    """

    def __init__(
        self,
        client: OpenAIClient,
        model: str = "gpt-4o-mini",
        temperature_range: Optional[List[float]] = None,
        num_variations: int = 5,
        prompt_template: Optional[str] = None,
        min_edit_distance: int = 5,
        max_retries: int = 2,
        verbose: bool = False,
    ):
        """Initialize the question variation generator.

        Args:
            client: OpenAI client for API calls
            model: Model to use for variation generation (default: gpt-4o-mini)
            temperature_range: List of temperatures for diversity (default: [0.7, 1.0, 1.3])
            num_variations: Number of variations to generate per question (default: 5)
            prompt_template: Custom prompt template (default provided)
            min_edit_distance: Minimum character difference for variation acceptance (default: 5)
            max_retries: Maximum retry attempts for filtered variations (default: 2)
            verbose: Enable verbose logging for debugging (default: False)
        """
        self.client = client
        self.model = model
        self.temperature_range = temperature_range or [0.7, 1.0, 1.3]
        self.num_variations = num_variations
        self.min_edit_distance = min_edit_distance
        self.max_retries = max_retries
        self.verbose = verbose

        # Default prompt template preserves exact meaning
        self.prompt_template = (
            prompt_template
            or "Rephrase the following question while preserving its exact meaning: {question}"
        )

        # Statistics tracking
        self._generation_attempts = 0
        self._rejections = 0
        self._final_count = 0

    async def generate_variations(self, question: str) -> List[str]:
        """Generate N paraphrased versions of the question.

        Distributes generations across temperature range for diversity,
        validates variations are meaningful, and retries if necessary.

        Args:
            question: Original question to paraphrase

        Returns:
            List of N unique, sufficiently different question variations
        """
        # Reset statistics
        self._generation_attempts = 0
        self._rejections = 0
        self._final_count = 0

        variations = []
        retry_count = 0
        temp_index = 0

        while len(variations) < self.num_variations and retry_count <= self.max_retries:
            # Cycle through temperature range
            temperature = self.temperature_range[temp_index % len(self.temperature_range)]
            temp_index += 1

            try:
                # Generate variation
                prompt = self.prompt_template.format(question=question)
                self._generation_attempts += 1

                response = await self.client.generate_response(
                    prompt=prompt,
                    model=self.model,
                    temperature=temperature,
                )

                variation = response.content.strip()

                # Validate variation
                if self._is_valid_variation(variation, question, variations):
                    variations.append(variation)
                    if self.verbose:
                        logger.info(
                            f"Accepted variation {len(variations)}/{self.num_variations}: "
                            f"{variation[:50]}... (temp={temperature})"
                        )
                else:
                    self._rejections += 1
                    if self.verbose:
                        logger.debug(
                            f"Rejected variation (too similar or duplicate): "
                            f"{variation[:50]}... (temp={temperature})"
                        )
                    # Increment retry count only when we reject
                    if len(variations) < self.num_variations:
                        retry_count += 1

            except Exception as e:
                logger.warning(f"Error generating variation with temp={temperature}: {e}")
                # Retry on error
                retry_count += 1
                continue

        self._final_count = len(variations)

        if self.verbose:
            self._log_statistics()

        return variations

    def _is_valid_variation(
        self, variation: str, original: str, existing_variations: List[str]
    ) -> bool:
        """Validate that variation is meaningful and unique.

        Args:
            variation: Candidate variation to validate
            original: Original question
            existing_variations: Already accepted variations

        Returns:
            True if variation is valid, False otherwise
        """
        # Filter exact duplicates of original
        if variation == original:
            if self.verbose:
                logger.debug("Rejected: exact match to original")
            return False

        # Filter near-identical variations (minimum edit distance threshold)
        if self._edit_distance(variation, original) < self.min_edit_distance:
            if self.verbose:
                logger.debug(
                    f"Rejected: too similar to original "
                    f"(edit distance < {self.min_edit_distance})"
                )
            return False

        # Filter duplicates of existing variations
        for existing in existing_variations:
            if variation == existing:
                if self.verbose:
                    logger.debug("Rejected: duplicate of existing variation")
                return False

            # Also check edit distance against existing variations
            if self._edit_distance(variation, existing) < self.min_edit_distance:
                if self.verbose:
                    logger.debug(
                        f"Rejected: too similar to existing variation "
                        f"(edit distance < {self.min_edit_distance})"
                    )
                return False

        return True

    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance between two strings.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Edit distance (number of character changes needed)
        """
        # Simple character-level Levenshtein distance
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _log_statistics(self):
        """Log generation statistics for debugging."""
        logger.info(
            f"Question variation statistics: "
            f"attempts={self._generation_attempts}, "
            f"rejections={self._rejections}, "
            f"final_count={self._final_count}"
        )

    def get_statistics(self) -> dict:
        """Get generation statistics.

        Returns:
            Dictionary with generation statistics
        """
        return {
            "generation_attempts": self._generation_attempts,
            "rejections": self._rejections,
            "final_count": self._final_count,
            "rejection_rate": (
                self._rejections / self._generation_attempts
                if self._generation_attempts > 0
                else 0.0
            ),
        }
