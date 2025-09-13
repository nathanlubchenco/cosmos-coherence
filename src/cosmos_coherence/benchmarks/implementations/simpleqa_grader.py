"""SimpleQA AI-based grading implementation matching OpenAI reference."""

from typing import Dict, Literal, Tuple

from cosmos_coherence.llm.openai_client import OpenAIClient


class SimpleQAGrader:
    """AI-based grader for SimpleQA responses, matching OpenAI's evaluation methodology."""

    GRADING_PROMPT = """You are comparing a submitted answer to an expert answer on a question.
Here is the data:
[BEGIN DATA]
************
[Question]: {question}
************
[Expert]: {expert_answer}
************
[Submission]: {submission}
************
[END DATA]

Compare the factual content of the submitted answer with the expert answer.
Ignore any differences in style, grammar, or punctuation.

The submission may either be correct, incorrect, or not attempted.
If the submission says something like "I don't know" or "I cannot answer", then it is not attempted.

Grade the submission as follows:
- If the submission contains the same factual information as the expert answer, grade as "CORRECT"
- If the submission contradicts the expert answer or provides different factual information,
  grade it as "INCORRECT"
- If the submission indicates uncertainty or inability to answer, grade it as "NOT_ATTEMPTED"

Important grading rules:
1. Accept semantic variations (e.g., "USA" vs "United States")
2. Tolerate minor typos in proper nouns
3. Allow partial answers if the core information is accurate
4. Focus on factual accuracy, not exact wording
5. Accept answers with hedging/uncertainty if the core fact is correct

Respond with only one word: CORRECT, INCORRECT, or NOT_ATTEMPTED."""

    def __init__(self, client: OpenAIClient, grader_model: str = "gpt-4o-mini"):
        """Initialize the grader with an OpenAI client.

        Args:
            client: OpenAI client for making API calls
            grader_model: Model to use for grading (default: gpt-4o-mini for efficiency)
        """
        self.client = client
        self.grader_model = grader_model

    async def grade_response(
        self, question: str, expert_answer: str, submission: str
    ) -> Tuple[str, Dict]:
        """Grade a response using AI evaluation.

        Args:
            question: The original question
            expert_answer: The expected/correct answer
            submission: The model's response to grade

        Returns:
            Tuple of (grade, metadata)
            - grade: One of "CORRECT", "INCORRECT", or "NOT_ATTEMPTED"
            - metadata: Additional information about the grading
        """
        # Format the grading prompt
        prompt = self.GRADING_PROMPT.format(
            question=question, expert_answer=expert_answer, submission=submission
        )

        # Get grading from model
        response = await self.client.generate_response(
            prompt,
            model=self.grader_model,
            temperature=0.0,  # Deterministic grading
            max_tokens=10,  # Only need one word
        )

        # Parse the grade
        grade_text = response.content.strip().upper()

        # Validate and normalize the grade
        if "CORRECT" in grade_text and "INCORRECT" not in grade_text:
            grade = "CORRECT"
        elif "INCORRECT" in grade_text:
            grade = "INCORRECT"
        elif "NOT_ATTEMPTED" in grade_text or "NOT ATTEMPTED" in grade_text:
            grade = "NOT_ATTEMPTED"
        else:
            # Default to INCORRECT if we can't parse the grade
            grade = "INCORRECT"

        metadata = {
            "grader_model": self.grader_model,
            "raw_grade": grade_text,
            "normalized_grade": grade,
        }

        return grade, metadata

    @staticmethod
    def calculate_metrics(grades: list[Literal["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]]) -> Dict:
        """Calculate aggregate metrics from a list of grades.

        Args:
            grades: List of grades from individual questions

        Returns:
            Dictionary with metrics matching OpenAI's evaluation:
            - accuracy: Accuracy over all questions
            - accuracy_given_attempted: Accuracy excluding not attempted
            - f1_score: F1 score (for compatibility, same as accuracy here)
            - correct_percentage: Percentage of correct answers
            - incorrect_percentage: Percentage of incorrect answers
            - not_attempted_percentage: Percentage of not attempted
        """
        total = len(grades)
        if total == 0:
            return {
                "accuracy": 0.0,
                "accuracy_given_attempted": 0.0,
                "f1_score": 0.0,
                "correct_percentage": 0.0,
                "incorrect_percentage": 0.0,
                "not_attempted_percentage": 0.0,
            }

        correct = grades.count("CORRECT")
        incorrect = grades.count("INCORRECT")
        not_attempted = grades.count("NOT_ATTEMPTED")
        attempted = correct + incorrect

        return {
            "accuracy": correct / total,
            "accuracy_given_attempted": correct / attempted if attempted > 0 else 0.0,
            "f1_score": correct / total,  # For SimpleQA, F1 is same as accuracy
            "correct_percentage": (correct / total) * 100,
            "incorrect_percentage": (incorrect / total) * 100,
            "not_attempted_percentage": (not_attempted / total) * 100,
            "total_questions": total,
            "correct_count": correct,
            "incorrect_count": incorrect,
            "not_attempted_count": not_attempted,
        }
