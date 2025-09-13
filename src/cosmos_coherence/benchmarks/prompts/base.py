"""Base prompt management for benchmarks."""

from abc import ABC, abstractmethod
from typing import Dict, Optional

from cosmos_coherence.benchmarks.models.base import BaseDatasetItem


class BasePromptTemplate(ABC):
    """Abstract base class for prompt templates."""

    @abstractmethod
    def format(self, item: BaseDatasetItem, **kwargs) -> str:
        """Format the prompt for the given item."""
        pass

    @property
    @abstractmethod
    def template_name(self) -> str:
        """Name of the template."""
        pass


class PromptManager:
    """Manager for benchmark prompts with version control."""

    def __init__(self):
        """Initialize the prompt manager."""
        self._templates: Dict[str, Dict[str, BasePromptTemplate]] = {}

    def register_template(
        self, benchmark_name: str, template_name: str, template: BasePromptTemplate
    ) -> None:
        """Register a prompt template for a benchmark."""
        if benchmark_name not in self._templates:
            self._templates[benchmark_name] = {}
        self._templates[benchmark_name][template_name] = template

    def get_template(
        self, benchmark_name: str, template_name: str = "default"
    ) -> BasePromptTemplate:
        """Get a prompt template."""
        if benchmark_name not in self._templates:
            raise KeyError(f"No templates registered for benchmark: {benchmark_name}")

        if template_name not in self._templates[benchmark_name]:
            raise KeyError(f"Template '{template_name}' not found for benchmark: {benchmark_name}")

        return self._templates[benchmark_name][template_name]

    def list_templates(self, benchmark_name: Optional[str] = None) -> Dict:
        """List all available templates."""
        if benchmark_name:
            return {benchmark_name: list(self._templates.get(benchmark_name, {}).keys())}
        return {name: list(templates.keys()) for name, templates in self._templates.items()}


# Global prompt manager instance
prompt_manager = PromptManager()
