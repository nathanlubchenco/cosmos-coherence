"""Dataset-specific models for hallucination detection benchmarks.

This module provides concrete implementations of BaseDatasetItem for each
of the 5 supported hallucination detection benchmarks.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator, model_validator

from cosmos_coherence.benchmarks.models.base import (
    BaseDatasetItem,
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


class FaithBenchItem(BaseDatasetItem):
    """FaithBench dataset item for factual consistency checking.

    FaithBench focuses on detecting hallucinations in LLM-generated claims
    by comparing them against source contexts and evidence.
    """

    claim: str = Field(..., description="LLM-generated claim/summary to verify")
    context: str = Field(..., description="Source passage or document")
    evidence: Optional[List[str]] = Field(default=None, description="Supporting evidence sentences")
    annotations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Human annotations marking hallucination spans"
    )
    source_dataset: Optional[str] = Field(
        None, description="Original dataset source (NLI, fact-checking, etc.)"
    )
    is_hallucinated: Optional[bool] = Field(None, description="Binary label for hallucination")

    @field_validator("claim", "context")
    @classmethod
    def validate_required_text(cls, v: str, info) -> str:
        """Validate required text fields are not empty."""
        if not v or not v.strip():
            field_name = info.field_name
            raise ValueError(f"{field_name} cannot be empty")
        return v.strip()

    @field_validator("evidence")
    @classmethod
    def validate_evidence(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate evidence list if provided."""
        if v is not None:
            cleaned = [e.strip() for e in v if e and e.strip()]
            if not cleaned:
                return None
            return cleaned
        return v

    def validate_content(self) -> None:
        """Validate the content of the FaithBench item."""
        if not self.claim:
            raise DatasetValidationError("Claim cannot be empty")
        if not self.context:
            raise DatasetValidationError("Context cannot be empty")


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
