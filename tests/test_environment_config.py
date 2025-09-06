"""Tests for environment-specific configurations."""
import os
import subprocess
from pathlib import Path

import pytest
import yaml


class TestEnvironmentConfiguration:
    """Test suite for environment-specific configurations."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Return the project root directory."""
        return Path(__file__).parent.parent

    @pytest.fixture
    def env_example_path(self) -> Path:
        """Return the path to .env.example."""
        return Path(__file__).parent.parent / ".env.example"

    @pytest.fixture
    def env_dev_path(self) -> Path:
        """Return the path to .env.development."""
        return Path(__file__).parent.parent / ".env.development"

    @pytest.fixture
    def env_prod_path(self) -> Path:
        """Return the path to .env.production."""
        return Path(__file__).parent.parent / ".env.production"

    def test_env_example_exists(self, env_example_path: Path) -> None:
        """Test that .env.example exists with all required variables."""
        assert env_example_path.exists(), ".env.example file should exist"

    def test_env_example_has_required_variables(self, env_example_path: Path) -> None:
        """Test that .env.example contains all required environment variables."""
        if not env_example_path.exists():
            pytest.skip(".env.example not yet created")

        content = env_example_path.read_text()
        required_vars = [
            "ENVIRONMENT",
            "DEBUG",
            "LOG_LEVEL",
            "DATABASE_URL",
            "REDIS_URL",
            "SECRET_KEY",
            "OPENAI_API_KEY",
            "API_HOST",
            "API_PORT",
            "DASHBOARD_HOST",
            "DASHBOARD_PORT",
        ]

        for var in required_vars:
            assert f"{var}=" in content, f".env.example should contain {var}"

    def test_development_env_exists(self, env_dev_path: Path) -> None:
        """Test that .env.development exists."""
        assert env_dev_path.exists(), ".env.development should exist for development configuration"

    def test_production_env_template_exists(self, env_prod_path: Path) -> None:
        """Test that .env.production exists."""
        assert env_prod_path.exists(), ".env.production should exist for production configuration"

    def test_docker_override_has_dev_config(self, project_root: Path) -> None:
        """Test that docker-compose.override.yml has development configurations."""
        override_path = project_root / "docker-compose.override.yml"
        if not override_path.exists():
            pytest.skip("docker-compose.override.yml not found")

        with open(override_path, "r") as f:
            config = yaml.safe_load(f)

        # Check development-specific settings
        api_env = config.get("services", {}).get("api", {}).get("environment", [])
        assert any(
            "DEBUG=true" in str(env) or "DEBUG: true" in str(env) for env in api_env
        ), "Development should have DEBUG=true"

    def test_production_compose_has_prod_config(self, project_root: Path) -> None:
        """Test that docker-compose.prod.yml has production configurations."""
        prod_path = project_root / "docker-compose.prod.yml"
        if not prod_path.exists():
            pytest.skip("docker-compose.prod.yml not found")

        with open(prod_path, "r") as f:
            config = yaml.safe_load(f)

        # Check production-specific settings
        api_config = config.get("services", {}).get("api", {})
        api_env = api_config.get("environment", {})

        # Check for production settings
        assert any(
            "DEBUG=false" in str(env) or "DEBUG: false" in str(env) for env in api_env
        ), "Production should have DEBUG=false"
        assert any(
            "ENVIRONMENT=production" in str(env) or "ENVIRONMENT: production" in str(env)
            for env in api_env
        ), "Production should have ENVIRONMENT=production"

    def test_logging_configuration_exists(self, project_root: Path) -> None:
        """Test that logging configuration files exist."""
        logging_config_path = project_root / "configs" / "logging.yaml"
        assert logging_config_path.exists(), "logging.yaml configuration should exist"

    def test_logging_has_different_levels(self, project_root: Path) -> None:
        """Test that logging configuration has different levels for environments."""
        logging_config_path = project_root / "configs" / "logging.yaml"
        if not logging_config_path.exists():
            pytest.skip("logging.yaml not found")

        with open(logging_config_path, "r") as f:
            config = yaml.safe_load(f)

        # Check for different log levels
        assert "development" in config, "Logging should have development configuration"
        assert "production" in config, "Logging should have production configuration"

        dev_level = config.get("development", {}).get("level", "")
        prod_level = config.get("production", {}).get("level", "")

        assert dev_level in ["DEBUG", "INFO"], "Development should use DEBUG or INFO level"
        assert prod_level in ["WARNING", "ERROR"], "Production should use WARNING or ERROR level"

    def test_security_headers_in_production(self, project_root: Path) -> None:
        """Test that production configuration includes security headers."""
        prod_compose = project_root / "docker-compose.prod.yml"
        if not prod_compose.exists():
            pytest.skip("docker-compose.prod.yml not found")

        with open(prod_compose, "r") as f:
            content = f.read()

        # Check for security-related configurations
        assert (
            "SSL" in content or "TLS" in content or "certbot" in content
        ), "Production should include SSL/TLS configuration"

    def test_resource_limits_in_production(self, project_root: Path) -> None:
        """Test that production has resource limits defined."""
        prod_compose = project_root / "docker-compose.prod.yml"
        if not prod_compose.exists():
            pytest.skip("docker-compose.prod.yml not found")

        with open(prod_compose, "r") as f:
            config = yaml.safe_load(f)

        # Check for resource limits
        api_config = config.get("services", {}).get("api", {})
        deploy = api_config.get("deploy", {})

        assert "resources" in deploy, "Production should define resource limits"
        assert "limits" in deploy.get("resources", {}), "Should have resource limits"
        assert "reservations" in deploy.get("resources", {}), "Should have resource reservations"

    def test_deployment_scripts_exist(self, project_root: Path) -> None:
        """Test that deployment scripts exist."""
        scripts_dir = project_root / "scripts"
        assert scripts_dir.exists(), "scripts directory should exist"

        expected_scripts = [
            "deploy.sh",
            "start-dev.sh",
            "start-prod.sh",
            "backup.sh",
            "health-check.sh",
        ]

        for script in expected_scripts:
            script_path = scripts_dir / script
            assert script_path.exists(), f"Deployment script {script} should exist"

    def test_deployment_scripts_executable(self, project_root: Path) -> None:
        """Test that deployment scripts are executable."""
        scripts_dir = project_root / "scripts"
        if not scripts_dir.exists():
            pytest.skip("scripts directory not found")

        for script_path in scripts_dir.glob("*.sh"):
            # Check if file has execute permission
            assert os.access(script_path, os.X_OK), f"{script_path.name} should be executable"

    def test_health_check_endpoints_configured(self, project_root: Path) -> None:
        """Test that health check endpoints are properly configured."""
        # Check API health endpoint
        api_init = project_root / "src" / "cosmos_coherence" / "api" / "__init__.py"
        if api_init.exists():
            content = api_init.read_text()
            assert "/health" in content, "API should have /health endpoint"

        # Check docker-compose health checks
        compose_file = project_root / "docker-compose.yml"
        if compose_file.exists():
            with open(compose_file, "r") as f:
                config = yaml.safe_load(f)

            for service in ["api", "dashboard", "db", "redis"]:
                if service in config.get("services", {}):
                    service_config = config["services"][service]
                    assert (
                        "healthcheck" in service_config
                    ), f"Service {service} should have healthcheck configured"

    def test_monitoring_configuration(self, project_root: Path) -> None:
        """Test that monitoring configuration exists."""
        monitoring_config = project_root / "configs" / "monitoring.yaml"
        assert monitoring_config.exists(), "monitoring.yaml should exist for observability"

    def test_secrets_not_in_repo(self, project_root: Path) -> None:
        """Test that actual secret files are not committed."""
        secret_files = [".env", ".env.local", "secrets.yaml", "credentials.json"]

        gitignore_path = project_root / ".gitignore"
        if gitignore_path.exists():
            gitignore_content = gitignore_path.read_text()
            for secret_file in secret_files:
                assert secret_file in gitignore_content, f"{secret_file} should be in .gitignore"

    @pytest.mark.integration
    def test_environment_switching(self, project_root: Path) -> None:
        """Test that environment can be switched between dev and prod."""
        try:
            # Test development environment
            result = subprocess.run(
                ["docker", "compose", "config"],
                cwd=project_root,
                capture_output=True,
                text=True,
                env={**os.environ, "ENVIRONMENT": "development"},
            )
            assert result.returncode == 0, "Development config should be valid"

            # Test production environment
            result = subprocess.run(
                [
                    "docker",
                    "compose",
                    "-f",
                    "docker-compose.yml",
                    "-f",
                    "docker-compose.prod.yml",
                    "config",
                ],
                cwd=project_root,
                capture_output=True,
                text=True,
                env={**os.environ, "ENVIRONMENT": "production"},
            )
            assert result.returncode == 0, "Production config should be valid"
        except FileNotFoundError:
            pytest.skip("Docker Compose not available")
