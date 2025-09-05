"""Tests for .gitignore patterns and repository hygiene."""

from pathlib import Path

import pytest


class TestGitignorePatterns:
    """Test that .gitignore properly excludes expected patterns."""

    @pytest.fixture
    def gitignore_content(self):
        """Load .gitignore content."""
        gitignore_path = Path(__file__).parent.parent / ".gitignore"
        if gitignore_path.exists():
            return gitignore_path.read_text()
        return ""

    def test_gitignore_exists(self):
        """Test that .gitignore file exists."""
        gitignore_path = Path(__file__).parent.parent / ".gitignore"
        assert gitignore_path.exists(), ".gitignore file should exist"

    def test_python_artifacts_excluded(self, gitignore_content):
        """Test that Python artifacts are excluded."""
        python_patterns = [
            "__pycache__",
            "*.py[cod]",
            "*$py.class",
            "*.so",
            ".Python",
            "build/",
            "develop-eggs/",
            "dist/",
            "downloads/",
            "eggs/",
            ".eggs/",
            "lib/",
            "lib64/",
            "parts/",
            "sdist/",
            "var/",
            "wheels/",
            "*.egg-info/",
            "*.egg",
            "MANIFEST",
        ]

        for pattern in python_patterns:
            # Clean pattern for checking (remove trailing /)
            check_pattern = pattern.rstrip("/")
            assert (
                check_pattern in gitignore_content or pattern in gitignore_content
            ), f"Python pattern {pattern} should be in .gitignore"

    def test_virtual_env_excluded(self, gitignore_content):
        """Test that virtual environments are excluded."""
        venv_patterns = [
            "venv/",
            "env/",
            "ENV/",
            ".venv",
            ".env",
        ]

        for pattern in venv_patterns:
            check_pattern = pattern.rstrip("/")
            assert (
                check_pattern in gitignore_content or pattern in gitignore_content
            ), f"Virtual env pattern {pattern} should be in .gitignore"

    def test_testing_artifacts_excluded(self, gitignore_content):
        """Test that testing artifacts are excluded."""
        test_patterns = [
            ".tox/",
            ".nox/",
            ".coverage",
            "htmlcov/",
            ".pytest_cache/",
            "nosetests.xml",
            "coverage.xml",
            "*.cover",
            ".hypothesis/",
        ]

        for pattern in test_patterns:
            check_pattern = pattern.rstrip("/")
            assert (
                check_pattern in gitignore_content or pattern in gitignore_content
            ), f"Test pattern {pattern} should be in .gitignore"

    def test_ide_files_excluded(self, gitignore_content):
        """Test that IDE files are excluded."""
        ide_patterns = [
            ".idea/",
            ".vscode/",
            "*.swp",
            "*.swo",
            "*~",
            ".project",
            ".pydevproject",
        ]

        for pattern in ide_patterns:
            check_pattern = pattern.rstrip("/")
            assert (
                check_pattern in gitignore_content or pattern in gitignore_content
            ), f"IDE pattern {pattern} should be in .gitignore"

    def test_os_files_excluded(self, gitignore_content):
        """Test that OS-specific files are excluded."""
        os_patterns = [
            ".DS_Store",
            "Thumbs.db",
            "ehthumbs.db",
            "Desktop.ini",
        ]

        for pattern in os_patterns:
            assert pattern in gitignore_content, f"OS pattern {pattern} should be in .gitignore"

    def test_project_specific_excluded(self, gitignore_content):
        """Test that project-specific directories are excluded."""
        project_patterns = [
            ".cache/",
            "outputs/",
            "data/",
        ]

        for pattern in project_patterns:
            check_pattern = pattern.rstrip("/")
            assert (
                check_pattern in gitignore_content or pattern in gitignore_content
            ), f"Project pattern {pattern} should be in .gitignore"

    def test_env_files_handled_correctly(self, gitignore_content):
        """Test that .env is excluded but .env.example is not."""
        # .env should be ignored
        assert ".env" in gitignore_content, ".env should be in .gitignore"

        # .env.example should NOT be in gitignore (we want to track it)
        assert (
            ".env.example" not in gitignore_content or "!.env.example" in gitignore_content
        ), ".env.example should be tracked (not in .gitignore or explicitly included with !)"

    def test_jupyter_artifacts_excluded(self, gitignore_content):
        """Test that Jupyter notebook artifacts are excluded."""
        jupyter_patterns = [
            ".ipynb_checkpoints",
            "*.ipynb_checkpoints",
        ]

        for pattern in jupyter_patterns:
            assert (
                pattern in gitignore_content
            ), f"Jupyter pattern {pattern} should be in .gitignore"

    def test_mypy_cache_excluded(self, gitignore_content):
        """Test that mypy cache is excluded."""
        assert ".mypy_cache" in gitignore_content, ".mypy_cache should be in .gitignore"
        assert ".dmypy.json" in gitignore_content, ".dmypy.json should be in .gitignore"
        assert "dmypy.json" in gitignore_content, "dmypy.json should be in .gitignore"

    def test_ruff_cache_excluded(self, gitignore_content):
        """Test that ruff cache is excluded."""
        assert ".ruff_cache" in gitignore_content, ".ruff_cache should be in .gitignore"

    def test_no_sensitive_patterns_missing(self, gitignore_content):
        """Test that no sensitive file patterns are missing."""
        sensitive_patterns = [
            "*.key",
            "*.pem",
            "*.p12",
            "*.pfx",
            ".env",
            "secrets/",
        ]

        for pattern in sensitive_patterns:
            check_pattern = pattern.rstrip("/")
            assert (
                check_pattern in gitignore_content or pattern in gitignore_content
            ), f"Sensitive pattern {pattern} should be in .gitignore for security"
