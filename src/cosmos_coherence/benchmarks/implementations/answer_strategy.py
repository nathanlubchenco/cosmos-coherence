"""Answer strategy for coherence-guided hallucination prevention.

This module implements the key differentiating logic for question-variation coherence:
- YES/NO questions: Coherence score determines answer polarity (YES or NO)
- Other questions: Coherence score determines abstention (answer or "I don't know")

The hypothesis is that coherence across question variations predicts answer reliability:
- High coherence → Well-formed question → Confident answer
- Low coherence → Ambiguous/ill-posed question → Abstain or opposite polarity
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


class AnswerStrategy:
    """Strategy for using coherence scores to guide answer selection.

    This class implements the core innovation of question-variation coherence:
    different treatment for YES/NO questions vs other question types.

    For YES/NO questions:
        - High coherence → Answer YES (or NO, configurable)
        - Low coherence → Answer opposite polarity
        - Never abstain (always provide binary answer)

    For other questions:
        - High coherence → Return model's answer (confident)
        - Low coherence → Return abstention response ("I don't know")

    Example:
        >>> strategy = AnswerStrategy(
        ...     coherence_threshold=0.5,
        ...     abstention_response="I don't know",
        ...     high_coherence_answer_yes=True
        ... )
        >>>
        >>> # YES/NO question handling
        >>> question = "Is Paris the capital of France?"
        >>> coherence_score = 0.8
        >>> answer = strategy.apply_strategy(question, coherence_score)
        >>> print(answer)  # "YES" (high coherence)
        >>>
        >>> # Factual question handling
        >>> question = "What is the capital of France?"
        >>> model_answer = "Paris"
        >>> coherence_score = 0.3
        >>> answer = strategy.apply_strategy(question, coherence_score, model_answer)
        >>> print(answer)  # "I don't know" (low coherence, abstain)
    """

    # YES/NO question patterns (case-insensitive)
    YES_NO_PATTERNS = [
        r"^is\s",      # "Is Paris..."
        r"^are\s",     # "Are cats..."
        r"^does\s",    # "Does water..."
        r"^do\s",      # "Do birds..."
        r"^can\s",     # "Can humans..."
        r"^will\s",    # "Will the..."
        r"^would\s",   # "Would it..."
        r"^should\s",  # "Should we..."
        r"^has\s",     # "Has the..."
        r"^have\s",    # "Have they..."
        r"^was\s",     # "Was Einstein..."
        r"^were\s",    # "Were the..."
        r"^did\s",     # "Did Napoleon..."
    ]

    def __init__(
        self,
        coherence_threshold: float = 0.5,
        abstention_response: str = "I don't know",
        high_coherence_answer_yes: bool = True,
        verbose: bool = False,
    ):
        """Initialize answer strategy.

        Args:
            coherence_threshold: Threshold for coherence score (0.0-1.0, default: 0.5)
                - Scores >= threshold: high coherence (confident)
                - Scores < threshold: low coherence (abstain or opposite polarity)
            abstention_response: Response to return when abstaining (default: "I don't know")
            high_coherence_answer_yes: For YES/NO questions, whether high coherence → YES
                (default: True, set False to flip polarity mapping)
                Note: This mapping should be tuned empirically based on experimental results.
            verbose: Enable verbose logging for debugging (default: False)
        """
        self.coherence_threshold = coherence_threshold
        self.abstention_response = abstention_response
        self.high_coherence_answer_yes = high_coherence_answer_yes
        self.verbose = verbose

        # Statistics tracking
        self._yes_no_count = 0
        self._other_question_count = 0
        self._abstention_count = 0
        self._confident_answer_count = 0
        self._yes_answer_count = 0
        self._no_answer_count = 0

    def apply_strategy(
        self,
        question: str,
        coherence_score: float,
        model_answer: Optional[str] = None,
    ) -> str:
        """Apply coherence-guided answer strategy.

        This is the main entry point that orchestrates the entire strategy:
        1. Detect if question is YES/NO type
        2. If YES/NO: use coherence to determine polarity (YES or NO)
        3. If other: use coherence to determine abstention (answer or "I don't know")

        Args:
            question: The question being evaluated
            coherence_score: Coherence score from question variation analysis (0.0-1.0)
            model_answer: Model's answer (required for non-YES/NO questions, optional for YES/NO)

        Returns:
            Final answer (YES, NO, model_answer, or abstention_response)

        Example:
            >>> strategy = AnswerStrategy(coherence_threshold=0.5)
            >>>
            >>> # YES/NO question, high coherence
            >>> strategy.apply_strategy("Is Paris the capital of France?", 0.8)
            'YES'
            >>>
            >>> # YES/NO question, low coherence
            >>> strategy.apply_strategy("Is Paris the capital of France?", 0.3)
            'NO'
            >>>
            >>> # Factual question, high coherence
            >>> strategy.apply_strategy("What is the capital of France?", 0.8, "Paris")
            'Paris'
            >>>
            >>> # Factual question, low coherence
            >>> strategy.apply_strategy("What is the capital of France?", 0.3, "Paris")
            'I don't know'
        """
        # Detect question type
        is_yes_no = self._is_yes_no_question(question)

        if is_yes_no:
            # YES/NO question: coherence determines polarity
            self._yes_no_count += 1
            answer = self._coherence_to_yes_no(coherence_score, self.coherence_threshold)

            if answer == "YES":
                self._yes_answer_count += 1
            else:
                self._no_answer_count += 1

            if self.verbose:
                logger.info(
                    f"YES/NO question detected: coherence={coherence_score:.4f}, "
                    f"threshold={self.coherence_threshold}, answer={answer}"
                )

            return answer

        else:
            # Other question type: coherence determines abstention
            self._other_question_count += 1

            if model_answer is None:
                raise ValueError(
                    "model_answer is required for non-YES/NO questions. "
                    f"Question: {question[:100]}..."
                )

            answer = self._apply_coherence_threshold(
                model_answer, coherence_score, self.coherence_threshold
            )

            if answer == self.abstention_response:
                self._abstention_count += 1
            else:
                self._confident_answer_count += 1

            if self.verbose:
                abstained = answer == self.abstention_response
                logger.info(
                    f"Factual question: coherence={coherence_score:.4f}, "
                    f"threshold={self.coherence_threshold}, "
                    f"abstained={abstained}"
                )

            return answer

    def _is_yes_no_question(self, question: str) -> bool:
        """Detect if question is YES/NO type using pattern matching.

        Uses regex patterns to detect common YES/NO question structures
        (e.g., "Is", "Are", "Does", "Do", "Can", "Will", etc.).

        Args:
            question: Question text to analyze

        Returns:
            True if YES/NO question, False otherwise

        Example:
            >>> strategy = AnswerStrategy()
            >>> strategy._is_yes_no_question("Is Paris the capital of France?")
            True
            >>> strategy._is_yes_no_question("Are cats mammals?")
            True
            >>> strategy._is_yes_no_question("What is the capital of France?")
            False
            >>> strategy._is_yes_no_question("When did World War II end?")
            False
        """
        question_lower = question.strip().lower()

        # Check against all YES/NO patterns
        for pattern in self.YES_NO_PATTERNS:
            if re.match(pattern, question_lower):
                return True

        return False

    def _coherence_to_yes_no(self, coherence_score: float, threshold: float) -> str:
        """Map coherence score to YES/NO answer for binary questions.

        The mapping logic (to be empirically validated):
        - High coherence (>= threshold): Return YES (or NO if high_coherence_answer_yes=False)
        - Low coherence (< threshold): Return opposite polarity

        Note: The polarity mapping should be tuned based on experimental results.
        The current default assumes high coherence → YES, but this may need to be
        flipped depending on how coherence correlates with question validity.

        Args:
            coherence_score: Coherence score from question variation analysis (0.0-1.0)
            threshold: Coherence threshold for binary decision

        Returns:
            "YES" or "NO" based on coherence score and polarity mapping

        Example:
            >>> strategy = AnswerStrategy(high_coherence_answer_yes=True)
            >>> strategy._coherence_to_yes_no(0.8, 0.5)  # High coherence
            'YES'
            >>> strategy._coherence_to_yes_no(0.3, 0.5)  # Low coherence
            'NO'
            >>>
            >>> # Flipped polarity
            >>> strategy = AnswerStrategy(high_coherence_answer_yes=False)
            >>> strategy._coherence_to_yes_no(0.8, 0.5)  # High coherence
            'NO'
            >>> strategy._coherence_to_yes_no(0.3, 0.5)  # Low coherence
            'YES'
        """
        high_coherence = coherence_score >= threshold

        if self.high_coherence_answer_yes:
            # Default mapping: high coherence → YES
            return "YES" if high_coherence else "NO"
        else:
            # Flipped mapping: high coherence → NO
            return "NO" if high_coherence else "YES"

    def _apply_coherence_threshold(
        self, answer: str, coherence_score: float, threshold: float
    ) -> str:
        """Apply threshold-based abstention logic for non-YES/NO questions.

        The logic:
        - High coherence (>= threshold): Return model's answer (confident)
        - Low coherence (< threshold): Return abstention response (not confident)

        Args:
            answer: Model's answer to the question
            coherence_score: Coherence score from question variation analysis (0.0-1.0)
            threshold: Coherence threshold for abstention decision

        Returns:
            Original answer if coherence >= threshold, otherwise abstention response

        Example:
            >>> strategy = AnswerStrategy(
            ...     coherence_threshold=0.5,
            ...     abstention_response="I don't know"
            ... )
            >>>
            >>> # High coherence: return answer
            >>> strategy._apply_coherence_threshold("Paris", 0.8, 0.5)
            'Paris'
            >>>
            >>> # Low coherence: abstain
            >>> strategy._apply_coherence_threshold("Paris", 0.3, 0.5)
            "I don't know"
            >>>
            >>> # Boundary: threshold exactly met
            >>> strategy._apply_coherence_threshold("Paris", 0.5, 0.5)
            'Paris'
        """
        if coherence_score >= threshold:
            # High coherence: confident answer
            return answer
        else:
            # Low coherence: abstain
            return self.abstention_response

    def get_statistics(self) -> dict:
        """Get answer strategy statistics.

        Returns:
            Dictionary with strategy usage statistics

        Example:
            >>> strategy = AnswerStrategy()
            >>> # ... process some questions ...
            >>> stats = strategy.get_statistics()
            >>> print(stats)
            {
                'yes_no_count': 20,
                'other_question_count': 80,
                'abstention_count': 15,
                'confident_answer_count': 65,
                'yes_answer_count': 12,
                'no_answer_count': 8,
                'abstention_rate': 0.15,
                'yes_no_percentage': 0.20
            }
        """
        total_questions = self._yes_no_count + self._other_question_count

        return {
            "yes_no_count": self._yes_no_count,
            "other_question_count": self._other_question_count,
            "abstention_count": self._abstention_count,
            "confident_answer_count": self._confident_answer_count,
            "yes_answer_count": self._yes_answer_count,
            "no_answer_count": self._no_answer_count,
            "abstention_rate": (
                self._abstention_count / total_questions if total_questions > 0 else 0.0
            ),
            "yes_no_percentage": (
                self._yes_no_count / total_questions if total_questions > 0 else 0.0
            ),
        }
