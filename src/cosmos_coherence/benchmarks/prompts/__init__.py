"""Prompt management for benchmarks."""

from .base import BasePromptTemplate, PromptManager, prompt_manager
from .simpleqa import SimpleQAPromptTemplate

__all__ = [
    "BasePromptTemplate",
    "PromptManager",
    "prompt_manager",
    "SimpleQAPromptTemplate",
]
