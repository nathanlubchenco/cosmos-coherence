"""Tests for prompt management system."""

import pytest
from cosmos_coherence.benchmarks.models.datasets import SimpleQAItem
from cosmos_coherence.benchmarks.prompts import (
    BasePromptTemplate,
    PromptManager,
    SimpleQAPromptTemplate,
    prompt_manager,
)


class TestPromptManager:
    """Test the prompt management system."""

    @pytest.fixture
    def manager(self):
        """Create a fresh prompt manager for testing."""
        return PromptManager()

    @pytest.fixture
    def sample_template(self):
        """Create a sample prompt template."""

        class TestTemplate(BasePromptTemplate):
            def format(self, item, **kwargs):
                return f"Test: {item.question}"

            @property
            def template_name(self):
                return "test"

        return TestTemplate()

    def test_register_template(self, manager, sample_template):
        """Test registering a prompt template."""
        manager.register_template("test_benchmark", "default", sample_template)

        retrieved = manager.get_template("test_benchmark", "default")
        assert retrieved == sample_template

    def test_get_template_missing_benchmark(self, manager):
        """Test getting template for non-existent benchmark."""
        with pytest.raises(KeyError, match="No templates registered"):
            manager.get_template("nonexistent")

    def test_get_template_missing_template(self, manager, sample_template):
        """Test getting non-existent template."""
        manager.register_template("test_benchmark", "default", sample_template)

        with pytest.raises(KeyError, match="Template 'missing' not found"):
            manager.get_template("test_benchmark", "missing")

    def test_list_templates(self, manager, sample_template):
        """Test listing available templates."""
        manager.register_template("test_benchmark", "default", sample_template)
        manager.register_template("test_benchmark", "alternate", sample_template)

        # List all
        all_templates = manager.list_templates()
        assert "test_benchmark" in all_templates
        assert set(all_templates["test_benchmark"]) == {"default", "alternate"}

        # List specific benchmark
        benchmark_templates = manager.list_templates("test_benchmark")
        assert benchmark_templates == {"test_benchmark": ["default", "alternate"]}


class TestSimpleQAPromptTemplate:
    """Test SimpleQA prompt templates."""

    @pytest.fixture
    def template(self):
        """Create SimpleQA prompt template."""
        return SimpleQAPromptTemplate()

    @pytest.fixture
    def simple_item(self):
        """Create a simple SimpleQA item."""
        return SimpleQAItem(question="What is 2+2?", best_answer="4")

    def test_format_simple_item(self, template, simple_item):
        """Test formatting SimpleQA item."""
        result = template.format(simple_item)

        expected = (
            "Answer the following question with just the answer, nothing else. "
            "Give the shortest factual answer possible (typically 1-5 words).\n\n"
            "Question: What is 2+2?\n"
            "Answer:"
        )
        assert result == expected

    def test_template_name(self, template):
        """Test template name property."""
        assert template.template_name == "optimized"

    def test_global_registration(self):
        """Test that SimpleQA templates are registered globally."""
        # Templates should be auto-registered
        default_template = prompt_manager.get_template("simpleqa", "default")
        assert isinstance(default_template, SimpleQAPromptTemplate)

        optimized_template = prompt_manager.get_template("simpleqa", "optimized")
        assert isinstance(optimized_template, SimpleQAPromptTemplate)

        basic_template = prompt_manager.get_template("simpleqa", "basic")
        assert basic_template.template_name == "basic"

    def test_list_simpleqa_templates(self):
        """Test listing SimpleQA templates."""
        templates = prompt_manager.list_templates("simpleqa")
        expected_templates = {"simpleqa": ["default", "optimized", "basic"]}

        # Check that all expected templates are present
        assert "simpleqa" in templates
        for template_name in expected_templates["simpleqa"]:
            assert template_name in templates["simpleqa"]
