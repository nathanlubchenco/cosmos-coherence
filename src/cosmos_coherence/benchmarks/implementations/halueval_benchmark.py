"""HaluEval benchmark implementation for hallucination detection."""

import random
from typing import Dict, List, Optional, Tuple

from cosmos_coherence.benchmarks.models.base import BaseDatasetItem
from cosmos_coherence.benchmarks.models.datasets import HaluEvalItem, HaluEvalTaskType
from cosmos_coherence.harness.base_benchmark import (
    BenchmarkEvaluationResult,
    BenchmarkMetadata,
)
from cosmos_coherence.harness.base_benchmark_hf import HuggingFaceEnabledBenchmark
from cosmos_coherence.llm.openai_client import OpenAIClient


class HaluEvalBenchmark(HuggingFaceEnabledBenchmark):
    """HaluEval benchmark for hallucination detection.

    This benchmark evaluates models' ability to detect hallucinations across
    35,000 samples in QA, dialogue, summarization, and general tasks.

    The benchmark can load data from:
    - HuggingFace: pminervini/HaluEval dataset
    - Local files (if configured)
    """

    # Exact prompts from the HaluEval repository
    QA_PROMPT = """I want you act as an answer judge. Given a question and an answer, \
your objective is to determine if the provided answer contains non-factual or \
hallucinated information. You SHOULD give your judgement based on the following \
hallucination types and the world knowledge:

1. Misunderstanding question context and intention
2. Factual contradiction with world knowledge
3. Inappropriate level of specificity
4. Inferability of the answer

You should try your best to determine if the answer contains non-factual or \
hallucinated information according to the above hallucination types. \
The answer you give MUST be "Yes" or "No".

#Question#: {question}
#Answer#: {answer}
#Your Judgement#:"""

    DIALOGUE_PROMPT = """I want you act as a response judge. Given a dialogue history \
and a response, your objective is to determine if the provided response contains \
non-factual or hallucinated information. You SHOULD give your judgement based on the \
following hallucination types and the world knowledge.

You are trying to determine if the true entity in the response is replaced with a \
highly similar entity.

You are trying to determine if the true entity in the response is replaced with a \
dissimilar entity.

You are trying to determine if the true entity in the response is replaced with a \
dissimilar entity in a different entity type.

You should try your best to determine if the response contains non-factual or \
hallucinated information according to the above hallucination types. \
The answer you give MUST be "Yes" or "No".

#Dialogue History#: {dialogue_history}
#Response#: {response}
#Your Judgement#:"""

    SUMMARIZATION_PROMPT = """I want you act as a summary judge. Given a document and a \
summary, your objective is to determine if the provided summary contains non-factual \
or hallucinated information. You SHOULD give your judgement based on the following \
hallucination types and the world knowledge.

You are trying to determine if the summary is factual but some information cannot be \
directly inferred or entailed from the document.

You are trying to determine if there exists some non-factual and incorrect information \
in the summary.

You are trying to determine if there is a factual contradiction between the summary \
and the document.

You should try your best to determine if the summary contains non-factual or \
hallucinated information according to the above hallucination types. \
The answer you give MUST be "Yes" or "No".

#Document#: {document}
#Summary#: {summary}
#Your Judgement#:"""

    # General prompt uses same format as QA for consistency
    GENERAL_PROMPT = """I want you act as an answer judge. Given a question and an answer, \
your objective is to determine if the provided answer contains non-factual or \
hallucinated information based on world knowledge. \
The answer you give MUST be "Yes" or "No".

#Question#: {question}
#Answer#: {answer}
#Your Judgement#:"""

    def __init__(self, client: Optional[OpenAIClient] = None, **kwargs):
        """Initialize HaluEval benchmark.

        Args:
            client: OpenAI client for evaluation
            **kwargs: Additional arguments for parent class
        """
        # Set benchmark name first (needed by base class)
        self._benchmark_name = "HaluEval"

        # Extract use_huggingface flag if present
        use_huggingface = kwargs.pop("use_huggingface", True)

        # Extract random seed before passing kwargs to super
        random_seed = kwargs.pop("random_seed", 42)

        # Default to HuggingFace dataset if not specified
        if "hf_dataset_name" not in kwargs and use_huggingface:
            kwargs["hf_dataset_name"] = "halueval"

        super().__init__(**kwargs)

        self.client = client

        # Set random seed for reproducible answer selection
        self.random_seed = random_seed
        self.rng = random.Random(self.random_seed)

    def get_prompt_with_selection(self, item: BaseDatasetItem) -> Tuple[str, bool, str]:
        """Get prompt with random selection of answer and hallucination flag.

        Args:
            item: HaluEvalItem with task data

        Returns:
            Tuple of (user_prompt, is_hallucinated, system_prompt)
        """
        if not isinstance(item, HaluEvalItem):
            raise ValueError(f"Expected HaluEvalItem, got {type(item)}")

        # Randomly select between right answer and hallucinated answer (using seeded RNG)
        is_hallucinated = self.rng.choice([True, False])
        selected_answer = item.hallucinated_answer if is_hallucinated else item.right_answer

        # Generate prompts based on task type with system messages from original repo
        if item.task_type == HaluEvalTaskType.QA:
            system_prompt = (
                "You are a hallucination detector. You MUST determine if the provided "
                "answer contains hallucination or not for the question based on the world "
                'knowledge. The answer you provided MUST be "Yes" or "No"'
            )
            user_prompt = self.QA_PROMPT.format(question=item.question, answer=selected_answer)
        elif item.task_type == HaluEvalTaskType.DIALOGUE:
            system_prompt = (
                "You are a response judge. You MUST determine if the provided response "
                "contains non-factual or hallucinated information. The answer you give "
                'MUST be "Yes" or "No"'
            )
            dialogue_str = "\n".join(item.dialogue_history) if item.dialogue_history else ""
            user_prompt = self.DIALOGUE_PROMPT.format(
                dialogue_history=dialogue_str, response=selected_answer
            )
        elif item.task_type == HaluEvalTaskType.SUMMARIZATION:
            system_prompt = (
                "You are a summary judge. You MUST determine if the provided summary "
                "contains non-factual or hallucinated information. The answer you give "
                'MUST be "Yes" or "No"'
            )
            user_prompt = self.SUMMARIZATION_PROMPT.format(
                document=item.document or "", summary=selected_answer
            )
        else:  # GENERAL
            system_prompt = (
                "You are a hallucination detector. You MUST determine if the provided "
                "answer contains hallucination or not for the question based on the world "
                'knowledge. The answer you provided MUST be "Yes" or "No"'
            )
            user_prompt = self.GENERAL_PROMPT.format(question=item.question, answer=selected_answer)

        return user_prompt, is_hallucinated, system_prompt

    def get_prompt(self, item: BaseDatasetItem) -> str:
        """Format dataset item into LLM prompt.

        Note: This method is for compatibility. Use get_prompt_with_selection
        for actual evaluation to get both prompt and ground truth.

        Args:
            item: Dataset item (HaluEvalItem)

        Returns:
            Formatted prompt string
        """
        user_prompt, _, _ = self.get_prompt_with_selection(item)
        return user_prompt

    def evaluate_response(
        self, response: str, ground_truth: str, item: BaseDatasetItem
    ) -> BenchmarkEvaluationResult:
        """Evaluate model response for hallucination detection.

        Args:
            response: Model's response (Yes/No)
            ground_truth: Either "hallucinated" or "not_hallucinated"
            item: Original dataset item

        Returns:
            Evaluation result
        """
        # Parse response to Yes/No
        response_lower = response.strip().lower()
        if "yes" in response_lower:
            prediction = "Yes"
        elif "no" in response_lower:
            prediction = "No"
        else:
            # Default to No if unclear
            prediction = "No"

        # Determine expected answer based on ground truth
        expected = "Yes" if ground_truth == "hallucinated" else "No"

        # Calculate if correct
        is_correct = prediction == expected
        score = 1.0 if is_correct else 0.0

        return BenchmarkEvaluationResult(
            is_correct=is_correct,
            score=score,
            original_metric_score=score,
            explanation=f"Predicted: {prediction}, Expected: {expected}",
            metadata={
                "prediction": prediction,
                "expected": expected,
                "task_type": item.task_type if isinstance(item, HaluEvalItem) else "unknown",
                "raw_response": response,
            },
        )

    def calculate_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate aggregate metrics from evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary of metrics
        """
        if not results:
            return {
                "overall_accuracy": 0.0,
                "total_samples": 0,
            }

        # Overall metrics
        correct = sum(1 for r in results if r.get("is_correct", False))
        total = len(results)

        # Task-specific metrics
        task_results = {}
        for result in results:
            task_type = result.get("task_type", "unknown")
            if task_type not in task_results:
                task_results[task_type] = {"correct": 0, "total": 0}
            task_results[task_type]["total"] += 1
            if result.get("is_correct", False):
                task_results[task_type]["correct"] += 1

        metrics = {
            "overall_accuracy": correct / total if total > 0 else 0.0,
            "total_samples": total,
            "correct_samples": correct,
        }

        # Add task-specific accuracies
        for task_type, counts in task_results.items():
            if counts["total"] > 0:
                metrics[f"{task_type}_accuracy"] = counts["correct"] / counts["total"]
                metrics[f"{task_type}_samples"] = counts["total"]

        # Calculate precision/recall for hallucination detection
        true_positives = sum(
            1 for r in results if r.get("expected") == "Yes" and r.get("prediction") == "Yes"
        )
        false_positives = sum(
            1 for r in results if r.get("expected") == "No" and r.get("prediction") == "Yes"
        )
        false_negatives = sum(
            1 for r in results if r.get("expected") == "Yes" and r.get("prediction") == "No"
        )

        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
            metrics["precision"] = precision
        else:
            metrics["precision"] = 0.0

        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
            metrics["recall"] = recall
        else:
            metrics["recall"] = 0.0

        # F1 score
        if metrics["precision"] + metrics["recall"] > 0:
            metrics["f1_score"] = (
                2
                * metrics["precision"]
                * metrics["recall"]
                / (metrics["precision"] + metrics["recall"])
            )
        else:
            metrics["f1_score"] = 0.0

        return metrics

    def get_metadata(self) -> BenchmarkMetadata:
        """Get benchmark metadata.

        Returns:
            Benchmark metadata
        """
        return BenchmarkMetadata(
            name="HaluEval",
            paper_reference="Li et al., HaluEval, EMNLP 2023",
            evaluation_method="Binary hallucination detection",
            baseline_metrics={
                "accuracy": 0.65,  # GPT-3.5 baseline from paper
                "qa_accuracy": 0.68,
                "dialogue_accuracy": 0.62,
                "summarization_accuracy": 0.72,
            },
            dataset_size=35000,
            version="1.0.0",
        )

    def get_paper_reference(self) -> str:
        """Get paper reference string.

        Returns:
            Paper reference
        """
        return (
            "HaluEval: A Large-Scale Hallucination Evaluation Benchmark "
            "for Large Language Models (Li et al., EMNLP 2023)"
        )

    # Abstract property implementations
    @property
    def benchmark_name(self) -> str:
        """Return the benchmark identifier."""
        return self._benchmark_name

    @property
    def paper_reference(self) -> str:
        """Return the original paper reference."""
        return self.get_paper_reference()

    def get_baseline_metrics(self) -> Dict[str, float]:
        """Return published baseline metrics."""
        return self.get_metadata().baseline_metrics

    def get_evaluation_method(self) -> str:
        """Return description of the evaluation methodology."""
        return "Binary hallucination detection (Yes/No)"

    def get_original_prompts(self) -> List[str]:
        """Return example prompts from original paper."""
        return [
            self.QA_PROMPT.format(
                question="What is the capital of France?", answer="Paris is the capital of France."
            ),
            self.DIALOGUE_PROMPT.format(
                dialogue_history="User: Hello\nAgent: Hi there!",
                response="How can I help you today?",
            ),
            self.SUMMARIZATION_PROMPT.format(
                document="Climate change is a global issue...",
                summary="The document discusses climate change.",
            ),
        ]

    def validate_config(self, config: Dict) -> None:
        """Validate benchmark-specific configuration."""
        # HaluEval doesn't require specific config validation
        pass
