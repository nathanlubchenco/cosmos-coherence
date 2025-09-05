"""Tests for pre-commit configuration and hooks."""

import subprocess
from pathlib import Path

import pytest
import yaml


class TestPreCommitConfiguration:
    """Test pre-commit configuration and hook setup."""

    @pytest.fixture
    def precommit_config_path(self):
        """Get path to pre-commit config file."""
        return Path(__file__).parent.parent / ".pre-commit-config.yaml"

    @pytest.fixture
    def precommit_config(self, precommit_config_path):
        """Load pre-commit configuration."""
        if precommit_config_path.exists():
            with open(precommit_config_path) as f:
                return yaml.safe_load(f)
        return None

    def test_precommit_config_exists(self, precommit_config_path):
        """Test that .pre-commit-config.yaml exists."""
        assert precommit_config_path.exists(), ".pre-commit-config.yaml should exist"

    def test_precommit_config_valid_yaml(self, precommit_config):
        """Test that pre-commit config is valid YAML."""
        assert precommit_config is not None, "Pre-commit config should be valid YAML"
        assert "repos" in precommit_config, "Pre-commit config should have 'repos' key"

    def test_black_formatter_configured(self, precommit_config):
        """Test that Black formatter is configured."""
        assert precommit_config is not None
        repos = precommit_config.get("repos", [])

        black_found = False
        for repo in repos:
            if "black" in repo.get("repo", ""):
                black_found = True
                hooks = repo.get("hooks", [])
                assert len(hooks) > 0, "Black repo should have hooks"
                assert any(h.get("id") == "black" for h in hooks), "Black hook should be configured"
                break

        assert black_found, "Black formatter should be configured"

    def test_ruff_linter_configured(self, precommit_config):
        """Test that Ruff linter is configured."""
        assert precommit_config is not None
        repos = precommit_config.get("repos", [])

        ruff_found = False
        for repo in repos:
            if "ruff" in repo.get("repo", ""):
                ruff_found = True
                hooks = repo.get("hooks", [])
                assert len(hooks) > 0, "Ruff repo should have hooks"
                # Ruff can have both 'ruff' and 'ruff-format' hooks
                assert any(
                    h.get("id") in ["ruff", "ruff-format"] for h in hooks
                ), "Ruff hook should be configured"
                break

        assert ruff_found, "Ruff linter should be configured"

    def test_mypy_type_checker_configured(self, precommit_config):
        """Test that mypy type checker is configured."""
        assert precommit_config is not None
        repos = precommit_config.get("repos", [])

        mypy_found = False
        for repo in repos:
            if "mypy" in repo.get("repo", "") or "mirrors-mypy" in repo.get("repo", ""):
                mypy_found = True
                hooks = repo.get("hooks", [])
                assert len(hooks) > 0, "Mypy repo should have hooks"
                assert any(h.get("id") == "mypy" for h in hooks), "Mypy hook should be configured"
                break

        assert mypy_found, "Mypy type checker should be configured"

    def test_yaml_checker_configured(self, precommit_config):
        """Test that YAML validation is configured."""
        assert precommit_config is not None
        repos = precommit_config.get("repos", [])

        yaml_check_found = False
        for repo in repos:
            if "pre-commit-hooks" in repo.get("repo", ""):
                hooks = repo.get("hooks", [])
                for hook in hooks:
                    if hook.get("id") == "check-yaml":
                        yaml_check_found = True
                        break

        assert yaml_check_found, "YAML checker should be configured"

    def test_toml_checker_configured(self, precommit_config):
        """Test that TOML validation is configured."""
        assert precommit_config is not None
        repos = precommit_config.get("repos", [])

        toml_check_found = False
        for repo in repos:
            if "pre-commit-hooks" in repo.get("repo", ""):
                hooks = repo.get("hooks", [])
                for hook in hooks:
                    if hook.get("id") == "check-toml":
                        toml_check_found = True
                        break

        assert toml_check_found, "TOML checker should be configured"

    def test_file_maintenance_hooks_configured(self, precommit_config):
        """Test that file maintenance hooks are configured."""
        assert precommit_config is not None
        repos = precommit_config.get("repos", [])

        required_hooks = {
            "end-of-file-fixer": False,
            "trailing-whitespace": False,
        }

        for repo in repos:
            if "pre-commit-hooks" in repo.get("repo", ""):
                hooks = repo.get("hooks", [])
                for hook in hooks:
                    hook_id = hook.get("id")
                    if hook_id in required_hooks:
                        required_hooks[hook_id] = True

        for hook_name, found in required_hooks.items():
            assert found, f"File maintenance hook '{hook_name}' should be configured"

    def test_safety_hooks_configured(self, precommit_config):
        """Test that safety hooks are configured."""
        assert precommit_config is not None
        repos = precommit_config.get("repos", [])

        required_hooks = {
            "check-merge-conflict": False,
            "check-added-large-files": False,
        }

        for repo in repos:
            if "pre-commit-hooks" in repo.get("repo", ""):
                hooks = repo.get("hooks", [])
                for hook in hooks:
                    hook_id = hook.get("id")
                    if hook_id in required_hooks:
                        required_hooks[hook_id] = True

        for hook_name, found in required_hooks.items():
            assert found, f"Safety hook '{hook_name}' should be configured"

    def test_debug_statements_hook_configured(self, precommit_config):
        """Test that debug statement detection is configured."""
        assert precommit_config is not None
        repos = precommit_config.get("repos", [])

        debug_check_found = False
        for repo in repos:
            if "pre-commit-hooks" in repo.get("repo", ""):
                hooks = repo.get("hooks", [])
                for hook in hooks:
                    if hook.get("id") == "debug-statements":
                        debug_check_found = True
                        break

        assert debug_check_found, "Debug statements checker should be configured"

    def test_precommit_in_dev_dependencies(self):
        """Test that pre-commit is in dev dependencies."""
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        assert pyproject_path.exists(), "pyproject.toml should exist"

        content = pyproject_path.read_text()
        # Check in dev dependencies section
        assert "pre-commit" in content, "pre-commit should be in pyproject.toml"

        # More specific check - ensure it's in the dev dependencies
        import toml

        pyproject = toml.load(pyproject_path)
        dev_deps = (
            pyproject.get("tool", {})
            .get("poetry", {})
            .get("group", {})
            .get("dev", {})
            .get("dependencies", {})
        )
        assert "pre-commit" in dev_deps, "pre-commit should be in dev dependencies"

    def test_precommit_installable(self):
        """Test that pre-commit can be installed (if in environment)."""
        try:
            result = subprocess.run(
                ["poetry", "run", "pre-commit", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0, "pre-commit should be executable"
            assert "pre-commit" in result.stdout, "pre-commit version should be displayed"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("pre-commit not installed or poetry not available")
