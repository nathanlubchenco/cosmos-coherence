"""Tests for project documentation completeness."""

from pathlib import Path

import pytest


class TestDocumentation:
    """Test that project documentation is complete and helpful."""

    def test_readme_exists(self):
        """Test that README.md exists."""
        readme_path = Path(__file__).parent.parent / "README.md"
        assert readme_path.exists(), "README.md should exist"

    def test_readme_has_setup_instructions(self):
        """Test that README.md contains development setup instructions."""
        readme_path = Path(__file__).parent.parent / "README.md"
        if not readme_path.exists():
            pytest.skip("README.md not yet created")

        content = readme_path.read_text().lower()

        # Check for essential sections
        required_sections = [
            "installation",
            "development",
            "setup",
            "getting started",
        ]

        has_setup_section = any(section in content for section in required_sections)
        assert has_setup_section, "README.md should have installation/setup instructions"

        # Check for specific setup steps
        setup_indicators = [
            "git clone",
            "poetry install",
            "pip install",
            "pre-commit",
            ".env",
        ]

        has_setup_steps = sum(1 for indicator in setup_indicators if indicator in content) >= 3
        assert has_setup_steps, "README.md should have concrete setup steps"

    def test_contributing_exists(self):
        """Test that CONTRIBUTING.md exists."""
        contributing_path = Path(__file__).parent.parent / "CONTRIBUTING.md"
        assert contributing_path.exists(), "CONTRIBUTING.md should exist"

    def test_contributing_has_precommit_instructions(self):
        """Test that CONTRIBUTING.md contains pre-commit setup instructions."""
        contributing_path = Path(__file__).parent.parent / "CONTRIBUTING.md"
        if not contributing_path.exists():
            pytest.skip("CONTRIBUTING.md not yet created")

        content = contributing_path.read_text().lower()

        # Check for pre-commit instructions
        precommit_indicators = [
            "pre-commit",
            "pre-commit install",
            "hooks",
        ]

        for indicator in precommit_indicators:
            assert (
                indicator in content
            ), f"CONTRIBUTING.md should mention '{indicator}' for pre-commit setup"

        # Check for code quality mentions
        quality_indicators = ["black", "ruff", "mypy", "formatting", "linting", "type"]

        has_quality_info = sum(1 for indicator in quality_indicators if indicator in content) >= 3
        assert has_quality_info, "CONTRIBUTING.md should discuss code quality tools"

    def test_env_example_mentioned_in_docs(self):
        """Test that .env.example is mentioned in documentation."""
        readme_path = Path(__file__).parent.parent / "README.md"
        contributing_path = Path(__file__).parent.parent / "CONTRIBUTING.md"

        env_mentioned = False

        if readme_path.exists():
            readme_content = readme_path.read_text().lower()
            if ".env" in readme_content or "environment" in readme_content:
                env_mentioned = True

        if contributing_path.exists() and not env_mentioned:
            contributing_content = contributing_path.read_text().lower()
            if ".env" in contributing_content or "environment" in contributing_content:
                env_mentioned = True

        assert env_mentioned, "Environment setup (.env) should be mentioned in documentation"

    def test_testing_instructions_present(self):
        """Test that documentation includes testing instructions."""
        readme_path = Path(__file__).parent.parent / "README.md"
        contributing_path = Path(__file__).parent.parent / "CONTRIBUTING.md"

        testing_mentioned = False

        if contributing_path.exists():
            content = contributing_path.read_text().lower()
            test_indicators = ["pytest", "make test", "poetry run test", "testing", "tests"]
            if any(indicator in content for indicator in test_indicators):
                testing_mentioned = True

        if not testing_mentioned and readme_path.exists():
            content = readme_path.read_text().lower()
            test_indicators = ["pytest", "make test", "poetry run test", "testing"]
            if any(indicator in content for indicator in test_indicators):
                testing_mentioned = True

        assert testing_mentioned, "Testing instructions should be present in documentation"

    def test_project_structure_documented(self):
        """Test that project structure is documented."""
        readme_path = Path(__file__).parent.parent / "README.md"
        contributing_path = Path(__file__).parent.parent / "CONTRIBUTING.md"

        structure_documented = False

        for doc_path in [readme_path, contributing_path]:
            if doc_path.exists():
                content = doc_path.read_text()
                # Look for directory structure indicators
                if any(
                    indicator in content
                    for indicator in [
                        "src/",
                        "tests/",
                        "configs/",
                        "Project Structure",
                        "Directory Structure",
                        "├──",
                        "│",
                    ]
                ):
                    structure_documented = True
                    break

        # This is optional but recommended
        if not structure_documented:
            pytest.skip("Project structure documentation is recommended but not required")

    def test_no_hardcoded_secrets_in_docs(self):
        """Test that no API keys or secrets are hardcoded in documentation."""
        doc_files = [
            Path(__file__).parent.parent / "README.md",
            Path(__file__).parent.parent / "CONTRIBUTING.md",
        ]

        # Patterns that might indicate hardcoded secrets
        secret_patterns = [
            "sk-",  # OpenAI API key prefix
            "api_key=",
            "token=",
            "password=",
            "secret=",
        ]

        for doc_path in doc_files:
            if doc_path.exists():
                content = doc_path.read_text().lower()
                for pattern in secret_patterns:
                    # Allow mentions in context of environment variables
                    lines = content.split("\n")
                    for line in lines:
                        if pattern in line and not any(
                            safe_context in line
                            for safe_context in [
                                ".env",
                                "environment",
                                "export",
                                "example",
                                "<your",
                                "your_",
                                "xxx",
                                "...",
                            ]
                        ):
                            assert False, f"Potential secret found in {doc_path.name}: {pattern}"

    def test_license_file_exists(self):
        """Test that a LICENSE file exists (if applicable)."""
        license_files = [
            Path(__file__).parent.parent / "LICENSE",
            Path(__file__).parent.parent / "LICENSE.txt",
            Path(__file__).parent.parent / "LICENSE.md",
        ]

        has_license = any(f.exists() for f in license_files)
        # License is optional for now
        if not has_license:
            pytest.skip("LICENSE file is recommended but not required yet")
