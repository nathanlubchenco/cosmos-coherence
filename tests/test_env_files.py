"""Tests for development environment files."""

from pathlib import Path

import pytest


class TestEnvironmentFiles:
    """Test that development environment files are properly configured."""

    def test_env_example_exists(self):
        """Test that .env.example file exists."""
        env_example_path = Path(__file__).parent.parent / ".env.example"
        assert env_example_path.exists(), ".env.example file should exist"

    def test_env_example_content(self):
        """Test that .env.example contains required variables."""
        env_example_path = Path(__file__).parent.parent / ".env.example"
        if not env_example_path.exists():
            pytest.skip(".env.example not yet created")

        content = env_example_path.read_text()
        required_vars = [
            "OPENAI_API_KEY",
            "OUTPUT_DIR",
            "LOG_LEVEL",
            "CACHE_DIR",
            "MAX_RETRIES",
            "REQUEST_TIMEOUT",
        ]

        for var in required_vars:
            assert f"{var}=" in content, f"Environment variable {var} should be in .env.example"
            # Ensure no values are provided (security best practice)
            lines = content.split("\n")
            for line in lines:
                if line.startswith(f"{var}="):
                    value_part = line.split("=", 1)[1].strip()
                    # Allow comments after the = sign but no actual values
                    if value_part and not value_part.startswith("#"):
                        assert value_part == "", f"{var} should not have a value in .env.example"

    def test_editorconfig_exists(self):
        """Test that .editorconfig file exists."""
        editorconfig_path = Path(__file__).parent.parent / ".editorconfig"
        assert editorconfig_path.exists(), ".editorconfig file should exist"

    def test_editorconfig_content(self):
        """Test that .editorconfig contains proper settings."""
        editorconfig_path = Path(__file__).parent.parent / ".editorconfig"
        if not editorconfig_path.exists():
            pytest.skip(".editorconfig not yet created")

        content = editorconfig_path.read_text()

        # Test root setting
        assert "root = true" in content, ".editorconfig should have root = true"

        # Test global settings
        assert "charset = utf-8" in content, "Should specify UTF-8 charset"
        assert "end_of_line = lf" in content, "Should use LF line endings"
        assert "insert_final_newline = true" in content, "Should insert final newline"
        assert "trim_trailing_whitespace = true" in content, "Should trim trailing whitespace"

        # Test Python-specific settings
        assert "[*.py]" in content, "Should have Python-specific settings"
        assert "indent_style = space" in content, "Python should use spaces"
        assert "indent_size = 4" in content, "Python should use 4-space indents"

        # Test YAML-specific settings
        assert (
            "[*.{yml,yaml}]" in content or "[*.yml]" in content
        ), "Should have YAML-specific settings"

        # Test Markdown settings
        assert "[*.md]" in content, "Should have Markdown-specific settings"

    def test_env_not_tracked_in_git(self):
        """Test that .env is in .gitignore but .env.example is not."""
        gitignore_path = Path(__file__).parent.parent / ".gitignore"
        assert gitignore_path.exists(), ".gitignore should exist"

        content = gitignore_path.read_text()

        # .env should be ignored
        assert ".env" in content, ".env should be in .gitignore"

        # .env.example should be tracked (either not in gitignore or explicitly included)
        lines = content.split("\n")
        env_example_ignored = False
        env_example_included = False

        for line in lines:
            if line.strip() == ".env.example":
                env_example_ignored = True
            if line.strip() == "!.env.example":
                env_example_included = True

        assert (
            not env_example_ignored or env_example_included
        ), ".env.example should be tracked in git"

    def test_python_indentation_consistency(self):
        """Test that Python indentation is consistent with project settings."""
        editorconfig_path = Path(__file__).parent.parent / ".editorconfig"
        if not editorconfig_path.exists():
            pytest.skip(".editorconfig not yet created")

        content = editorconfig_path.read_text()

        # Find Python section
        lines = content.split("\n")
        in_python_section = False
        indent_size = None
        indent_style = None

        for line in lines:
            if "[*.py]" in line:
                in_python_section = True
            elif line.startswith("[") and in_python_section:
                break
            elif in_python_section:
                if "indent_size" in line:
                    indent_size = line.split("=")[1].strip()
                elif "indent_style" in line:
                    indent_style = line.split("=")[1].strip()

        assert indent_size == "4", "Python should use 4-space indentation"
        assert indent_style == "space", "Python should use spaces, not tabs"

    def test_line_length_setting(self):
        """Test that line length is properly configured."""
        editorconfig_path = Path(__file__).parent.parent / ".editorconfig"
        if not editorconfig_path.exists():
            pytest.skip(".editorconfig not yet created")

        content = editorconfig_path.read_text()

        # Check for max_line_length in Python section
        lines = content.split("\n")
        in_python_section = False
        has_max_line_length = False

        for line in lines:
            if "[*.py]" in line:
                in_python_section = True
            elif line.startswith("[") and in_python_section:
                break
            elif in_python_section and "max_line_length" in line:
                has_max_line_length = True
                # Extract the value
                value = line.split("=")[1].strip()
                assert value in [
                    "88",
                    "100",
                    "120",
                ], f"Python max_line_length should be reasonable (88, 100, or 120), got {value}"

        assert has_max_line_length, "Python section should specify max_line_length"
