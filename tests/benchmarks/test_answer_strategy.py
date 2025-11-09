"""Tests for answer strategy implementation.

Tests cover:
1. YES/NO question detection
2. High coherence YES/NO question handling
3. Low coherence YES/NO question handling
4. High coherence non-YES/NO question handling
5. Low coherence non-YES/NO question handling (abstention)
6. Threshold boundary behavior
"""

import pytest
from cosmos_coherence.benchmarks.implementations.answer_strategy import AnswerStrategy


class TestAnswerStrategy:
    """Test suite for AnswerStrategy class."""

    def test_yes_no_detection_positive_cases(self):
        """Test: YES/NO question detected correctly (positive cases)."""
        strategy = AnswerStrategy()

        # Test various YES/NO question patterns
        yes_no_questions = [
            "Is Paris the capital of France?",
            "Are cats mammals?",
            "Does water boil at 100 degrees Celsius?",
            "Do birds have feathers?",
            "Can humans breathe underwater?",
            "Will the sun rise tomorrow?",
            "Would it rain if the sky is cloudy?",
            "Should we exercise regularly?",
            "Has Einstein won a Nobel Prize?",
            "Have they completed the project?",
            "Was Napoleon French?",
            "Were the Romans advanced?",
            "Did Columbus discover America?",
        ]

        for question in yes_no_questions:
            assert strategy._is_yes_no_question(question) is True, (
                f"Failed to detect YES/NO question: {question}"
            )

    def test_yes_no_detection_negative_cases(self):
        """Test: YES/NO question detection excludes non-YES/NO questions."""
        strategy = AnswerStrategy()

        # Test non-YES/NO questions (factual, when, where, what, etc.)
        non_yes_no_questions = [
            "What is the capital of France?",
            "When did World War II end?",
            "Where is the Eiffel Tower located?",
            "Who invented the telephone?",
            "How does photosynthesis work?",
            "Why is the sky blue?",
            "Which country has the largest population?",
            "Name the seven continents.",
        ]

        for question in non_yes_no_questions:
            assert strategy._is_yes_no_question(question) is False, (
                f"Incorrectly detected as YES/NO question: {question}"
            )

    def test_high_coherence_yes_no_returns_yes(self):
        """Test: High coherence YES/NO question returns YES (default mapping)."""
        strategy = AnswerStrategy(
            coherence_threshold=0.5,
            high_coherence_answer_yes=True  # Default mapping
        )

        question = "Is Paris the capital of France?"
        coherence_score = 0.8  # High coherence

        answer = strategy.apply_strategy(question, coherence_score)

        assert answer == "YES", (
            f"Expected 'YES' for high coherence YES/NO question, got '{answer}'"
        )

    def test_low_coherence_yes_no_returns_no(self):
        """Test: Low coherence YES/NO question returns opposite polarity (NO)."""
        strategy = AnswerStrategy(
            coherence_threshold=0.5,
            high_coherence_answer_yes=True  # Default mapping
        )

        question = "Is Paris the capital of France?"
        coherence_score = 0.3  # Low coherence

        answer = strategy.apply_strategy(question, coherence_score)

        assert answer == "NO", (
            f"Expected 'NO' for low coherence YES/NO question, got '{answer}'"
        )

    def test_high_coherence_yes_no_flipped_polarity(self):
        """Test: Flipped polarity mapping (high coherence → NO)."""
        strategy = AnswerStrategy(
            coherence_threshold=0.5,
            high_coherence_answer_yes=False  # Flipped mapping
        )

        # High coherence should return NO with flipped mapping
        question = "Is Paris the capital of France?"
        coherence_score = 0.8  # High coherence

        answer = strategy.apply_strategy(question, coherence_score)

        assert answer == "NO", (
            f"Expected 'NO' with flipped polarity mapping, got '{answer}'"
        )

        # Low coherence should return YES with flipped mapping
        coherence_score = 0.3  # Low coherence

        answer = strategy.apply_strategy(question, coherence_score)

        assert answer == "YES", (
            f"Expected 'YES' with flipped polarity mapping, got '{answer}'"
        )

    def test_high_coherence_non_yes_no_returns_model_answer(self):
        """Test: High coherence non-YES/NO question returns model answer."""
        strategy = AnswerStrategy(coherence_threshold=0.5)

        question = "What is the capital of France?"
        model_answer = "Paris"
        coherence_score = 0.8  # High coherence

        answer = strategy.apply_strategy(question, coherence_score, model_answer)

        assert answer == "Paris", (
            f"Expected model answer 'Paris' for high coherence, got '{answer}'"
        )

    def test_low_coherence_non_yes_no_returns_abstention(self):
        """Test: Low coherence non-YES/NO question returns abstention."""
        strategy = AnswerStrategy(
            coherence_threshold=0.5,
            abstention_response="I don't know"
        )

        question = "What is the capital of France?"
        model_answer = "Paris"
        coherence_score = 0.3  # Low coherence

        answer = strategy.apply_strategy(question, coherence_score, model_answer)

        assert answer == "I don't know", (
            f"Expected abstention 'I don't know' for low coherence, got '{answer}'"
        )

    def test_threshold_boundary_behavior(self):
        """Test: Threshold boundary behavior (score exactly at threshold)."""
        strategy = AnswerStrategy(coherence_threshold=0.5)

        # Exactly at threshold should be treated as high coherence (>= threshold)
        question_yes_no = "Is Paris the capital of France?"
        coherence_score = 0.5  # Exactly at threshold

        answer_yes_no = strategy.apply_strategy(question_yes_no, coherence_score)
        assert answer_yes_no == "YES", (
            f"Expected 'YES' at threshold boundary for YES/NO, got '{answer_yes_no}'"
        )

        # For non-YES/NO questions, should return model answer
        question_factual = "What is the capital of France?"
        model_answer = "Paris"

        answer_factual = strategy.apply_strategy(
            question_factual, coherence_score, model_answer
        )
        assert answer_factual == "Paris", (
            f"Expected model answer at threshold boundary, got '{answer_factual}'"
        )

        # Just below threshold should trigger low coherence behavior
        coherence_score = 0.49999

        answer_yes_no_low = strategy.apply_strategy(question_yes_no, coherence_score)
        assert answer_yes_no_low == "NO", (
            f"Expected 'NO' just below threshold for YES/NO, got '{answer_yes_no_low}'"
        )

        answer_factual_low = strategy.apply_strategy(
            question_factual, coherence_score, model_answer
        )
        assert answer_factual_low == "I don't know", (
            f"Expected abstention just below threshold, got '{answer_factual_low}'"
        )

    def test_custom_abstention_response(self):
        """Test: Custom abstention response works correctly."""
        custom_response = "I am not confident in my answer"
        strategy = AnswerStrategy(
            coherence_threshold=0.5,
            abstention_response=custom_response
        )

        question = "What is the capital of France?"
        model_answer = "Paris"
        coherence_score = 0.3  # Low coherence

        answer = strategy.apply_strategy(question, coherence_score, model_answer)

        assert answer == custom_response, (
            f"Expected custom abstention '{custom_response}', got '{answer}'"
        )

    def test_missing_model_answer_for_non_yes_no_raises_error(self):
        """Test: Non-YES/NO question without model_answer raises clear error."""
        strategy = AnswerStrategy()

        question = "What is the capital of France?"
        coherence_score = 0.8

        with pytest.raises(ValueError) as exc_info:
            strategy.apply_strategy(question, coherence_score)

        assert "model_answer is required" in str(exc_info.value)

    def test_statistics_tracking(self):
        """Test: Statistics tracking is accurate."""
        strategy = AnswerStrategy(coherence_threshold=0.5)

        # Process a mix of questions
        # 2 YES/NO questions (1 high coherence → YES, 1 low coherence → NO)
        strategy.apply_strategy("Is Paris the capital?", 0.8)  # YES
        strategy.apply_strategy("Are cats reptiles?", 0.3)     # NO

        # 3 factual questions (1 high coherence → answer, 2 low coherence → abstain)
        strategy.apply_strategy("What is the capital?", 0.8, "Paris")  # Answer
        strategy.apply_strategy("When did it happen?", 0.3, "1989")     # Abstain
        strategy.apply_strategy("Where is it located?", 0.2, "France")  # Abstain

        stats = strategy.get_statistics()

        assert stats["yes_no_count"] == 2, f"Expected 2 YES/NO questions, got {stats['yes_no_count']}"
        assert stats["other_question_count"] == 3, f"Expected 3 other questions, got {stats['other_question_count']}"
        assert stats["yes_answer_count"] == 1, f"Expected 1 YES answer, got {stats['yes_answer_count']}"
        assert stats["no_answer_count"] == 1, f"Expected 1 NO answer, got {stats['no_answer_count']}"
        assert stats["confident_answer_count"] == 1, f"Expected 1 confident answer, got {stats['confident_answer_count']}"
        assert stats["abstention_count"] == 2, f"Expected 2 abstentions, got {stats['abstention_count']}"

        # Check calculated metrics
        assert stats["abstention_rate"] == pytest.approx(2/5), (
            f"Expected abstention rate 0.4, got {stats['abstention_rate']}"
        )
        assert stats["yes_no_percentage"] == pytest.approx(2/5), (
            f"Expected YES/NO percentage 0.4, got {stats['yes_no_percentage']}"
        )


if __name__ == "__main__":
    # Run tests directly for validation
    import sys

    test_class = TestAnswerStrategy()

    tests = [
        ("YES/NO detection (positive cases)", test_class.test_yes_no_detection_positive_cases),
        ("YES/NO detection (negative cases)", test_class.test_yes_no_detection_negative_cases),
        ("High coherence YES/NO → YES", test_class.test_high_coherence_yes_no_returns_yes),
        ("Low coherence YES/NO → NO", test_class.test_low_coherence_yes_no_returns_no),
        ("Flipped polarity mapping", test_class.test_high_coherence_yes_no_flipped_polarity),
        ("High coherence factual → answer", test_class.test_high_coherence_non_yes_no_returns_model_answer),
        ("Low coherence factual → abstention", test_class.test_low_coherence_non_yes_no_returns_abstention),
        ("Threshold boundary behavior", test_class.test_threshold_boundary_behavior),
        ("Custom abstention response", test_class.test_custom_abstention_response),
        ("Missing model_answer error", test_class.test_missing_model_answer_for_non_yes_no_raises_error),
        ("Statistics tracking", test_class.test_statistics_tracking),
    ]

    print("Running Answer Strategy Tests...\n")
    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✓ {test_name}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_name}")
            print(f"  Error: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    sys.exit(0 if failed == 0 else 1)
