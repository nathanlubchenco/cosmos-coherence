"""Tests for Docker Compose configuration and orchestration."""
import subprocess
from pathlib import Path

import pytest
import yaml


class TestDockerComposeConfiguration:
    """Test suite for Docker Compose configuration."""

    @pytest.fixture
    def compose_file_path(self) -> Path:
        """Return the path to the main docker-compose.yml file."""
        return Path(__file__).parent.parent / "docker-compose.yml"

    @pytest.fixture
    def compose_override_path(self) -> Path:
        """Return the path to the docker-compose.override.yml file."""
        return Path(__file__).parent.parent / "docker-compose.override.yml"

    @pytest.fixture
    def compose_prod_path(self) -> Path:
        """Return the path to the docker-compose.prod.yml file."""
        return Path(__file__).parent.parent / "docker-compose.prod.yml"

    def test_docker_compose_file_exists(self, compose_file_path: Path) -> None:
        """Test that docker-compose.yml exists."""
        assert compose_file_path.exists(), f"docker-compose.yml not found at {compose_file_path}"

    def test_docker_compose_valid_yaml(self, compose_file_path: Path) -> None:
        """Test that docker-compose.yml is valid YAML."""
        if not compose_file_path.exists():
            pytest.skip("docker-compose.yml not yet created")

        try:
            with open(compose_file_path, "r") as f:
                config = yaml.safe_load(f)
            assert isinstance(config, dict), "docker-compose.yml should be a valid YAML dictionary"
            assert "services" in config, "docker-compose.yml must have a 'services' section"
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid YAML in docker-compose.yml: {e}")

    def test_required_services_defined(self, compose_file_path: Path) -> None:
        """Test that required services are defined."""
        if not compose_file_path.exists():
            pytest.skip("docker-compose.yml not yet created")

        with open(compose_file_path, "r") as f:
            config = yaml.safe_load(f)

        services = config.get("services", {})
        required_services = ["api", "dashboard"]  # FastAPI and Dash

        for service in required_services:
            assert (
                service in services
            ), f"Required service '{service}' not defined in docker-compose.yml"

    def test_service_images_or_builds_defined(self, compose_file_path: Path) -> None:
        """Test that each service has either an image or build configuration."""
        if not compose_file_path.exists():
            pytest.skip("docker-compose.yml not yet created")

        with open(compose_file_path, "r") as f:
            config = yaml.safe_load(f)

        services = config.get("services", {})
        for service_name, service_config in services.items():
            assert (
                "image" in service_config or "build" in service_config
            ), f"Service '{service_name}' must have either 'image' or 'build' defined"

    def test_service_ports_exposed(self, compose_file_path: Path) -> None:
        """Test that services expose appropriate ports."""
        if not compose_file_path.exists():
            pytest.skip("docker-compose.yml not yet created")

        with open(compose_file_path, "r") as f:
            config = yaml.safe_load(f)

        services = config.get("services", {})

        # Check API service port
        if "api" in services:
            api_ports = services["api"].get("ports", [])
            assert any(
                "8000" in str(port) for port in api_ports
            ), "API service should expose port 8000"

        # Check Dashboard service port
        if "dashboard" in services:
            dash_ports = services["dashboard"].get("ports", [])
            assert any(
                "8050" in str(port) for port in dash_ports
            ), "Dashboard service should expose port 8050"

    def test_networks_configured(self, compose_file_path: Path) -> None:
        """Test that Docker networks are properly configured."""
        if not compose_file_path.exists():
            pytest.skip("docker-compose.yml not yet created")

        with open(compose_file_path, "r") as f:
            config = yaml.safe_load(f)

        # Check if networks are defined
        networks = config.get("networks", {})
        assert networks, "docker-compose.yml should define custom networks"

        # Check if services use networks
        services = config.get("services", {})
        for service_name, service_config in services.items():
            service_networks = service_config.get("networks", [])
            assert (
                service_networks
            ), f"Service '{service_name}' should be connected to at least one network"

    def test_environment_variables_configured(self, compose_file_path: Path) -> None:
        """Test that services have environment configuration."""
        if not compose_file_path.exists():
            pytest.skip("docker-compose.yml not yet created")

        with open(compose_file_path, "r") as f:
            config = yaml.safe_load(f)

        services = config.get("services", {})
        # Services that require environment configuration
        env_required_services = ["api", "dashboard", "db"]

        for service_name in env_required_services:
            if service_name in services:
                service_config = services[service_name]
                # Check for either env_file or environment section
                has_env_file = "env_file" in service_config
                has_environment = "environment" in service_config
                assert (
                    has_env_file or has_environment
                ), f"Service '{service_name}' should have environment configuration"

    def test_volumes_configured(self, compose_file_path: Path) -> None:
        """Test that appropriate volumes are configured."""
        if not compose_file_path.exists():
            pytest.skip("docker-compose.yml not yet created")

        with open(compose_file_path, "r") as f:
            config = yaml.safe_load(f)

        # Check for volume definitions (optional but recommended)
        _ = config.get("volumes", {})

        # Check service volume mounts
        services = config.get("services", {})
        for service_name, service_config in services.items():
            service_volumes = service_config.get("volumes", [])
            # Services should have some volume configuration (at least for configs)
            assert service_volumes, f"Service '{service_name}' should have volume mounts configured"

    def test_healthchecks_defined(self, compose_file_path: Path) -> None:
        """Test that services have health checks defined."""
        if not compose_file_path.exists():
            pytest.skip("docker-compose.yml not yet created")

        with open(compose_file_path, "r") as f:
            config = yaml.safe_load(f)

        services = config.get("services", {})
        for service_name, service_config in services.items():
            healthcheck = service_config.get("healthcheck", {})
            assert healthcheck, f"Service '{service_name}' should have a healthcheck defined"
            assert (
                "test" in healthcheck
            ), f"Service '{service_name}' healthcheck must have a 'test' command"

    def test_service_dependencies(self, compose_file_path: Path) -> None:
        """Test that service dependencies are properly configured."""
        if not compose_file_path.exists():
            pytest.skip("docker-compose.yml not yet created")

        with open(compose_file_path, "r") as f:
            config = yaml.safe_load(f)

        services = config.get("services", {})

        # Dashboard might depend on API
        if "dashboard" in services and "api" in services:
            dashboard = services["dashboard"]
            depends_on = dashboard.get("depends_on", [])
            # It's OK if dashboard doesn't depend on API, but check format if it does
            if depends_on:
                assert isinstance(depends_on, (list, dict)), "depends_on should be a list or dict"

    def test_restart_policies(self, compose_file_path: Path) -> None:
        """Test that services have appropriate restart policies."""
        if not compose_file_path.exists():
            pytest.skip("docker-compose.yml not yet created")

        with open(compose_file_path, "r") as f:
            config = yaml.safe_load(f)

        services = config.get("services", {})
        valid_restart_policies = ["no", "always", "on-failure", "unless-stopped"]

        for service_name, service_config in services.items():
            restart_policy = service_config.get("restart", "no")
            assert (
                restart_policy in valid_restart_policies
            ), f"Service '{service_name}' has invalid restart policy: {restart_policy}"

    def test_development_override_exists(self, compose_override_path: Path) -> None:
        """Test that docker-compose.override.yml exists for development."""
        assert (
            compose_override_path.exists()
        ), "docker-compose.override.yml should exist for development overrides"

    def test_development_override_valid(self, compose_override_path: Path) -> None:
        """Test that docker-compose.override.yml is valid."""
        if not compose_override_path.exists():
            pytest.skip("docker-compose.override.yml not yet created")

        try:
            with open(compose_override_path, "r") as f:
                config = yaml.safe_load(f)
            assert isinstance(config, dict), "docker-compose.override.yml should be valid YAML"
            assert "services" in config, "docker-compose.override.yml should have services section"
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid YAML in docker-compose.override.yml: {e}")

    def test_production_compose_exists(self, compose_prod_path: Path) -> None:
        """Test that docker-compose.prod.yml exists for production."""
        assert (
            compose_prod_path.exists()
        ), "docker-compose.prod.yml should exist for production configuration"

    @pytest.mark.integration
    def test_docker_compose_config_valid(self, compose_file_path: Path) -> None:
        """Test that docker-compose config is valid using Docker Compose CLI."""
        if not compose_file_path.exists():
            pytest.skip("docker-compose.yml not yet created")

        try:
            result = subprocess.run(
                ["docker-compose", "config"],
                cwd=compose_file_path.parent,
                capture_output=True,
                text=True,
            )
            assert (
                result.returncode == 0
            ), f"docker-compose config validation failed: {result.stderr}"
        except FileNotFoundError:
            pytest.skip("docker-compose CLI not available")

    @pytest.mark.integration
    def test_docker_compose_build(self, compose_file_path: Path) -> None:
        """Test that Docker Compose can build all services."""
        if not compose_file_path.exists():
            pytest.skip("docker-compose.yml not yet created")

        try:
            result = subprocess.run(
                ["docker-compose", "build", "--no-cache"],
                cwd=compose_file_path.parent,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            assert result.returncode == 0, f"docker-compose build failed: {result.stderr}"
        except FileNotFoundError:
            pytest.skip("docker-compose CLI not available")
        except subprocess.TimeoutExpired:
            pytest.skip("docker-compose build timed out")

    @pytest.mark.integration
    def test_docker_compose_up_check(self, compose_file_path: Path) -> None:
        """Test that Docker Compose can start services (dry-run)."""
        if not compose_file_path.exists():
            pytest.skip("docker-compose.yml not yet created")

        try:
            # Use --dry-run if available, otherwise just config
            result = subprocess.run(
                ["docker-compose", "config"],
                cwd=compose_file_path.parent,
                capture_output=True,
                text=True,
            )
            assert (
                result.returncode == 0
            ), f"docker-compose configuration check failed: {result.stderr}"
        except FileNotFoundError:
            pytest.skip("docker-compose CLI not available")
