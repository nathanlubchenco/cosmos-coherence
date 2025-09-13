"""Dataset-specific models for hallucination detection benchmarks.

This module provides concrete implementations of BaseDatasetItem for each
of the 5 supported hallucination detection benchmarks.
"""

import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from cosmos_coherence.benchmarks.models.base import (
    BaseDatasetItem,
    BaseResult,
    DatasetValidationError,
)


class SimpleQACategory(str, Enum):
    """SimpleQA question categories."""

    SCIENCE = "science"
    TECHNOLOGY = "technology"
    HISTORY = "history"
    GEOGRAPHY = "geography"
    SPORTS = "sports"
    TV_SHOWS = "tv_shows"
    VIDEO_GAMES = "video_games"
    MUSIC = "music"
    LITERATURE = "literature"
    GENERAL = "general"


class SimpleQADifficulty(str, Enum):
    """SimpleQA difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TruthfulQACategory(str, Enum):
    """TruthfulQA question categories (38 total)."""

    HEALTH = "health"
    LAW = "law"
    FINANCE = "finance"
    POLITICS = "politics"
    PSYCHOLOGY = "psychology"
    HISTORY = "history"
    SCIENCE = "science"
    MISCONCEPTIONS = "misconceptions"
    CONSPIRACY = "conspiracy"
    STEREOTYPES = "stereotypes"
    LANGUAGE = "language"
    CONFUSION = "confusion"
    FICTION = "fiction"
    MYTHS_FAIRYTALES = "myths_fairytales"
    INDEXICAL_ERROR = "indexical_error"
    DISTRACTION = "distraction"
    SUBJECTIVE = "subjective"
    ADVERTISING = "advertising"
    RELIGION = "religion"
    LOGICAL_FALSEHOOD = "logical_falsehood"
    MANDELA_EFFECT = "mandela_effect"
    NUTRITION = "nutrition"
    EDUCATION = "education"
    SOCIOLOGY = "sociology"
    ECONOMICS = "economics"
    GEOGRAPHY = "geography"
    WEATHER = "weather"
    PHILOSOPHY = "philosophy"
    ETHICS = "ethics"
    PARANORMAL = "paranormal"
    SUPERSTITIONS = "superstitions"
    STATISTICS = "statistics"
    MISQUOTATIONS = "misquotations"
    INDEXICAL_ERROR_LOCATION = "indexical_error_location"
    INDEXICAL_ERROR_IDENTITY = "indexical_error_identity"
    INDEXICAL_ERROR_OTHER = "indexical_error_other"
    PROVERBS = "proverbs"
    OTHER = "other"


class FEVERLabel(str, Enum):
    """FEVER claim verification labels."""

    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NOTENOUGHINFO = "NOTENOUGHINFO"


class HaluEvalTaskType(str, Enum):
    """HaluEval task types."""

    QA = "qa"
    DIALOGUE = "dialogue"
    SUMMARIZATION = "summarization"
    GENERAL = "general"


class FaithBenchAnnotation(str, Enum):
    """FaithBench 4-level annotation taxonomy for hallucination detection."""

    CONSISTENT = "consistent"  # Factually accurate summaries
    QUESTIONABLE = "questionable"  # Gray area, potentially subjective
    BENIGN = "benign"  # Incorrect but harmless hallucination
    HALLUCINATED = "hallucinated"  # Clear factual errors


class FaithBenchItem(BaseDatasetItem):
    """FaithBench dataset item for summarization hallucination detection.

    FaithBench is a summarization benchmark with "challenging" hallucinations
    where state-of-the-art detectors disagree. Uses a 4-level annotation taxonomy.
    """

    # Core fields from FaithBench dataset structure
    sample_id: str = Field(..., description="Unique identifier within batch")
    source: str = Field(..., description="Original text to summarize (106-380 words typical)")
    summary: str = Field(..., description="Generated summary to evaluate")

    # Annotation fields
    annotation_label: Optional[FaithBenchAnnotation] = Field(
        None, description="4-level annotation: consistent/questionable/benign/hallucinated"
    )
    annotation_spans: List[str] = Field(
        default_factory=list, description="Problematic text spans identified by annotators"
    )
    annotation_justification: Optional[str] = Field(
        None, description="Human annotator explanation for the label"
    )

    # Metadata fields
    detector_predictions: Dict[str, Any] = Field(
        default_factory=dict, description="Predictions from various hallucination detectors"
    )
    entropy_score: Optional[float] = Field(
        None, description="Entropy measuring detector disagreement (higher = more challenging)"
    )
    summarizer_model: Optional[str] = Field(None, description="Model that generated the summary")

    @field_validator("source", "summary")
    @classmethod
    def validate_required_text(cls, v: str, info) -> str:
        """Validate required text fields are not empty."""
        if not v or not v.strip():
            field_name = info.field_name
            raise ValueError(f"{field_name} cannot be empty")
        return v.strip()

    @field_validator("annotation_label")
    @classmethod
    def validate_annotation_label(
        cls, v: Optional[FaithBenchAnnotation]
    ) -> Optional[FaithBenchAnnotation]:
        """Validate annotation label is from the 4-level taxonomy."""
        if v is not None and isinstance(v, str):
            # Convert string to enum if needed
            try:
                return FaithBenchAnnotation(v.lower())
            except ValueError:
                raise ValueError(
                    f"Invalid annotation label: {v}. "
                    "Must be one of: consistent, questionable, benign, hallucinated"
                )
        return v

    @field_validator("entropy_score")
    @classmethod
    def validate_entropy(cls, v: Optional[float]) -> Optional[float]:
        """Validate entropy score is in valid range."""
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"Entropy score must be between 0 and 1, got {v}")
        return v

    def validate_content(self) -> None:
        """Validate the content of the FaithBench item."""
        if not self.source:
            raise DatasetValidationError("Source text cannot be empty")
        if not self.summary:
            raise DatasetValidationError("Summary cannot be empty")
        if not self.sample_id:
            raise DatasetValidationError("Sample ID is required")


class SimpleQAItem(BaseDatasetItem):
    """SimpleQA dataset item for straightforward Q&A evaluation.

    SimpleQA provides short, fact-seeking questions with clear correct answers
    designed for easy grading and high inter-annotator agreement.
    """

    # Question field inherited from BaseDatasetItem
    best_answer: str = Field(..., description="Reference answer")
    category: Optional[SimpleQACategory] = Field(None, description="Question category/topic")
    difficulty: Optional[SimpleQADifficulty] = Field(None, description="Question difficulty level")
    sources: Optional[List[str]] = Field(None, description="Supporting sources for the answer")
    grading_notes: Optional[str] = Field(None, description="Notes for grading the answer")

    # Additional fields that may appear in tests but are not required
    good_answers: Optional[List[str]] = Field(default=None, description="Alternative good answers")
    bad_answers: Optional[List[str]] = Field(default=None, description="Known bad answers")

    @field_validator("best_answer")
    @classmethod
    def validate_answer(cls, v: str) -> str:
        """Validate answer is not empty."""
        if not v or not v.strip():
            raise ValueError("Best answer cannot be empty")
        return v.strip()

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate sources list if provided."""
        if v is not None:
            cleaned = [s.strip() for s in v if s and s.strip()]
            if not cleaned:
                return None
            return cleaned
        return v

    def validate_content(self) -> None:
        """Validate the content of the SimpleQA item."""
        if not self.question:
            raise DatasetValidationError("Question cannot be empty")
        if not self.best_answer:
            raise DatasetValidationError("Best answer cannot be empty")


class TruthfulQAItem(BaseDatasetItem):
    """TruthfulQA dataset item for truthfulness evaluation.

    TruthfulQA contains adversarially-designed questions that target
    false beliefs and misconceptions commonly held by humans and LLMs.
    """

    # Question field inherited from BaseDatasetItem
    best_answer: str = Field(..., description="Best truthful answer")
    correct_answers: List[str] = Field(..., description="List of acceptable correct answers")
    incorrect_answers: List[str] = Field(..., description="List of common incorrect answers")
    category: TruthfulQACategory = Field(..., description="Question category")
    source: Optional[str] = Field(None, description="Source of the question")
    mc1_targets: Optional[Dict[str, Any]] = Field(
        None, description="Multiple choice format with 2 options"
    )
    mc2_targets: Optional[Dict[str, Any]] = Field(
        None, description="Multiple choice format with multiple options"
    )

    @field_validator("best_answer")
    @classmethod
    def validate_best_answer(cls, v: str) -> str:
        """Validate best answer is not empty."""
        if not v or not v.strip():
            raise ValueError("Best answer cannot be empty")
        return v.strip()

    @field_validator("correct_answers", "incorrect_answers")
    @classmethod
    def validate_answer_lists(cls, v: List[str], info) -> List[str]:
        """Validate answer lists are not empty."""
        if not v:
            field_name = info.field_name
            raise ValueError(f"{field_name} cannot be empty")
        cleaned = [a.strip() for a in v if a and a.strip()]
        if not cleaned:
            field_name = info.field_name
            raise ValueError(f"{field_name} cannot contain only empty strings")
        return cleaned

    @model_validator(mode="after")
    def validate_mc_format(self) -> "TruthfulQAItem":
        """Validate multiple choice format if provided."""
        if self.mc1_targets:
            if "choices" not in self.mc1_targets or "labels" not in self.mc1_targets:
                raise ValueError("mc1_targets must contain 'choices' and 'labels'")
            if len(self.mc1_targets["choices"]) != 2:
                raise ValueError("mc1_targets must have exactly 2 choices")

        if self.mc2_targets:
            if "choices" not in self.mc2_targets or "labels" not in self.mc2_targets:
                raise ValueError("mc2_targets must contain 'choices' and 'labels'")

        return self

    def validate_content(self) -> None:
        """Validate the content of the TruthfulQA item."""
        if not self.question:
            raise DatasetValidationError("Question cannot be empty")
        if not self.best_answer:
            raise DatasetValidationError("Best answer cannot be empty")
        if not self.correct_answers:
            raise DatasetValidationError("Correct answers list cannot be empty")
        if not self.incorrect_answers:
            raise DatasetValidationError("Incorrect answers list cannot be empty")


class FEVERItem(BaseDatasetItem):
    """FEVER dataset item for claim verification.

    FEVER (Fact Extraction and VERification) provides claims that need
    to be verified against Wikipedia evidence.
    """

    claim: str = Field(..., description="Textual claim to be verified")
    label: FEVERLabel = Field(..., description="Verification label")
    evidence: List[List[Any]] = Field(
        default_factory=list,
        description="Evidence sentences from Wikipedia [[page, line_num], ...]",
    )
    verdict: Optional[str] = Field(None, description="Final verification decision")
    wikipedia_url: Optional[str] = Field(None, description="Source Wikipedia page")
    annotation_id: Optional[str] = Field(None, description="Internal annotation ID")

    @field_validator("claim")
    @classmethod
    def validate_claim(cls, v: str) -> str:
        """Validate claim is not empty."""
        if not v or not v.strip():
            raise ValueError("Claim cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_evidence_structure(self) -> "FEVERItem":
        """Validate evidence structure based on label."""
        if self.label == FEVERLabel.NOTENOUGHINFO:
            # NOTENOUGHINFO can have empty evidence
            pass
        elif self.label in [FEVERLabel.SUPPORTED, FEVERLabel.REFUTED]:
            # SUPPORTED and REFUTED must have evidence
            if not self.evidence:
                raise ValueError(f"{self.label} claims must have evidence")
        return self

    def validate_content(self) -> None:
        """Validate the content of the FEVER item."""
        if not self.claim:
            raise DatasetValidationError("Claim cannot be empty")
        # Use parent's question field for compatibility
        self.question = self.claim


class HaluEvalItem(BaseDatasetItem):
    """HaluEval dataset item for hallucination detection.

    HaluEval provides task-specific hallucination examples across
    QA, dialogue, summarization, and general tasks.
    """

    knowledge: str = Field(..., description="Background knowledge from Wikipedia")
    task_type: HaluEvalTaskType = Field(..., description="Type of task")
    right_answer: str = Field(..., description="Ground truth response")
    hallucinated_answer: str = Field(..., description="Generated hallucinated response")

    # Task-specific fields
    dialogue_history: Optional[List[str]] = Field(
        None, description="Conversation context for dialogue tasks"
    )
    document: Optional[str] = Field(None, description="Source document for summarization tasks")
    hallucination_type: Optional[str] = Field(None, description="Type of hallucination")

    @field_validator("knowledge", "right_answer", "hallucinated_answer")
    @classmethod
    def validate_required_fields(cls, v: str, info) -> str:
        """Validate required text fields."""
        if not v or not v.strip():
            field_name = info.field_name
            raise ValueError(f"{field_name} cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_task_specific_fields(self) -> "HaluEvalItem":
        """Validate task-specific field requirements."""
        if self.task_type == HaluEvalTaskType.DIALOGUE:
            if not self.dialogue_history:
                raise ValueError("Dialogue tasks must have dialogue_history")

        elif self.task_type == HaluEvalTaskType.SUMMARIZATION:
            if not self.document:
                raise ValueError("Summarization tasks must have document field")

        return self

    def validate_content(self) -> None:
        """Validate the content of the HaluEval item."""
        if not self.knowledge:
            raise DatasetValidationError("Knowledge cannot be empty")
        if not self.right_answer:
            raise DatasetValidationError("Right answer cannot be empty")
        if not self.hallucinated_answer:
            raise DatasetValidationError("Hallucinated answer cannot be empty")


class SimpleQAResult(BaseResult):
    """Result for SimpleQA benchmark evaluation."""

    question: str = Field(..., description="The question asked")
    is_correct: Optional[bool] = Field(
        None, description="Whether the answer is correct (exact match)"
    )
    f1_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Token-level F1 score")
    exact_match: Optional[bool] = Field(
        None, description="Whether prediction exactly matches ground truth"
    )
    response_length: Optional[int] = Field(None, ge=0, description="Number of tokens in response")
    ground_truth_length: Optional[int] = Field(
        None, ge=0, description="Number of tokens in ground truth"
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        """Validate question is not empty."""
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        return v

    def model_post_init(self, __context) -> None:
        """Calculate metrics if not provided."""
        super().model_post_init(__context)
        if self.is_correct is None or self.f1_score is None:
            self.calculate_metrics()

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate evaluation metrics for the result."""
        # Normalize for comparison
        pred_normalized = self.prediction.strip().lower()
        truth_normalized = self.ground_truth.strip().lower()

        # Calculate exact match
        self.exact_match = pred_normalized == truth_normalized
        self.is_correct = self.exact_match

        # Calculate F1 score (token-level)
        pred_tokens = set(pred_normalized.split())
        truth_tokens = set(truth_normalized.split())

        self.response_length = len(pred_tokens)
        self.ground_truth_length = len(truth_tokens)

        if not pred_tokens and not truth_tokens:
            self.f1_score = 1.0
        elif not pred_tokens or not truth_tokens:
            self.f1_score = 0.0
        else:
            intersection = pred_tokens & truth_tokens
            precision = len(intersection) / len(pred_tokens)
            recall = len(intersection) / len(truth_tokens)

            if precision + recall == 0:
                self.f1_score = 0.0
            else:
                self.f1_score = 2 * (precision * recall) / (precision + recall)

        # Update metrics dictionary
        self.metrics = {
            "exact_match": float(self.exact_match),
            "f1_score": self.f1_score,
            "response_length": self.response_length,
            "ground_truth_length": self.ground_truth_length,
        }

        return self.metrics

    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_evaluation(
        cls, experiment_id: UUID, item_id: UUID, eval_data: Dict[str, Any], **kwargs
    ) -> "SimpleQAResult":
        """Create result from evaluation data.

        Args:
            experiment_id: Experiment identifier
            item_id: Dataset item identifier
            eval_data: Dictionary with evaluation data
            **kwargs: Additional fields

        Returns:
            SimpleQAResult instance
        """
        return cls(
            experiment_id=experiment_id,
            item_id=item_id,
            question=eval_data["question"],
            prediction=eval_data.get("response", eval_data.get("prediction")),
            ground_truth=eval_data.get("expected", eval_data.get("ground_truth")),
            is_correct=eval_data.get("correct"),
            f1_score=eval_data.get("f1_score"),
            exact_match=eval_data.get("exact_match"),
            **kwargs,
        )

    @classmethod
    def aggregate_metrics(cls, results: List["SimpleQAResult"]) -> Dict[str, Any]:
        """Aggregate metrics from multiple results.

        Args:
            results: List of SimpleQAResult instances

        Returns:
            Dictionary with aggregated metrics
        """
        if not results:
            return {
                "total_questions": 0,
                "correct_answers": 0,
                "accuracy": 0.0,
                "average_f1_score": 0.0,
            }

        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        avg_f1 = sum(r.f1_score for r in results) / total

        return {
            "total_questions": total,
            "correct_answers": correct,
            "accuracy": correct / total,
            "average_f1_score": avg_f1,
        }

    @classmethod
    def to_jsonl(cls, results: List["SimpleQAResult"], include_metadata: bool = True) -> str:
        """Export results to JSONL format.

        Args:
            results: List of results to export
            include_metadata: Whether to include metadata as first line

        Returns:
            JSONL string with one JSON object per line
        """
        lines = []

        if include_metadata and results:
            # Calculate aggregate metrics
            metrics = cls.aggregate_metrics(results)

            metadata = {
                "type": "experiment_metadata",
                "benchmark": "SimpleQA",
                "total_questions": metrics["total_questions"],
                "correct_answers": metrics["correct_answers"],
                "accuracy": metrics["accuracy"],
                "average_f1_score": metrics["average_f1_score"],
                "timestamp": datetime.utcnow().isoformat(),
            }
            lines.append(json.dumps(metadata))

        # Add individual results
        for result in results:
            lines.append(result.to_json())

        return "\n".join(lines)

    @classmethod
    def from_jsonl(cls, jsonl_content: str) -> List["SimpleQAResult"]:
        """Load results from JSONL format.

        Args:
            jsonl_content: JSONL string content

        Returns:
            List of SimpleQAResult instances
        """
        results = []
        for line in jsonl_content.strip().split("\n"):
            if line:
                data = json.loads(line)
                # Skip metadata lines
                if data.get("type") != "experiment_metadata":
                    results.append(cls(**data))
        return results
