"""SimpleQA prompt templates."""

from cosmos_coherence.benchmarks.models.base import BaseDatasetItem
from cosmos_coherence.benchmarks.models.datasets import SimpleQAItem

from .base import BasePromptTemplate, prompt_manager


class SimpleQAPromptTemplate(BasePromptTemplate):
    """Optimized prompt template for SimpleQA benchmark.

    This template encourages concise, factual answers which improves accuracy
    from ~0% to ~20% based on empirical testing.
    """

    def format(self, item: BaseDatasetItem, **kwargs) -> str:
        """Format the prompt for SimpleQA items."""
        if isinstance(item, SimpleQAItem):
            return (
                "Answer the following question with just the answer, nothing else. "
                "Give the shortest factual answer possible (typically 1-5 words).\n\n"
                f"Question: {item.question}\n"
                "Answer:"
            )

        # Fallback for other item types
        return f"Question: {item.question}\nAnswer:"

    @property
    def template_name(self) -> str:
        """Template identifier."""
        return "optimized"


class SimpleQABasicTemplate(BasePromptTemplate):
    """Basic prompt template for SimpleQA benchmark."""

    def format(self, item: BaseDatasetItem, **kwargs) -> str:
        """Format basic prompt."""
        return f"Question: {item.question}\nAnswer:"

    @property
    def template_name(self) -> str:
        """Template identifier."""
        return "basic"


class SimpleQAPaperTemplate(BasePromptTemplate):
    """Original paper prompt template for SimpleQA benchmark.

    This template exactly matches the official OpenAI SimpleQA implementation
    which sends the question directly without any formatting or instructions.
    """

    def format(self, item: BaseDatasetItem, **kwargs) -> str:
        """Format prompt exactly as in the paper - just the question."""
        if isinstance(item, SimpleQAItem):
            return item.question
        return item.question

    @property
    def template_name(self) -> str:
        """Template identifier."""
        return "paper"


# Register templates
prompt_manager.register_template(
    "simpleqa", "default", SimpleQAPaperTemplate()
)  # Default to paper format
prompt_manager.register_template("simpleqa", "paper", SimpleQAPaperTemplate())
prompt_manager.register_template("simpleqa", "optimized", SimpleQAPromptTemplate())
prompt_manager.register_template("simpleqa", "basic", SimpleQABasicTemplate())
