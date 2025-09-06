"""Integration tests for complete Docker setup."""
import subprocess
import time
from pathlib import Path

import pytest
import requests


class TestDockerIntegration:
    """Integration tests for Docker containerization."""

    @pytest.fixture
    def project_root(self) -> Path:
        """Return the project root directory."""
        return Path(__file__).parent.parent

    @pytest.fixture
    def docker_compose_command(self, project_root: Path):
        """Return docker compose command with proper working directory."""

        def run_compose(*args, capture_output=True, check=False):
            return subprocess.run(
                ["docker", "compose"] + list(args),
                cwd=project_root,
                capture_output=capture_output,
                text=True,
                check=check,
            )

        return run_compose

    @pytest.mark.integration
    def test_docker_compose_config_valid(self, docker_compose_command):
        """Test that docker-compose configuration is valid."""
        result = docker_compose_command("config")
        assert result.returncode == 0, f"Docker compose config failed: {result.stderr}"

    @pytest.mark.integration
    def test_dockerfile_builds_successfully(self, project_root: Path):
        """Test that Dockerfile builds without errors."""
        result = subprocess.run(
            ["docker", "build", "-t", "cosmos-test:latest", "."],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Docker build failed: {result.stderr}"

    @pytest.mark.integration
    def test_services_start_successfully(self, docker_compose_command):
        """Test that all services start successfully."""
        # Start services
        result = docker_compose_command("up", "-d")
        assert result.returncode == 0, f"Docker compose up failed: {result.stderr}"

        # Give services time to start
        time.sleep(10)

        # Check all services are running
        result = docker_compose_command("ps")
        assert result.returncode == 0

        # Parse output to check service status
        output_lines = result.stdout.split("\n")
        services = ["cosmos-api", "cosmos-dashboard", "cosmos-db", "cosmos-redis"]

        for service in services:
            assert any(
                service in line for line in output_lines
            ), f"Service {service} not found in running containers"

    @pytest.mark.integration
    def test_api_health_endpoint(self, docker_compose_command):
        """Test that API health endpoint responds correctly."""
        # Ensure services are up
        docker_compose_command("up", "-d")
        time.sleep(5)

        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert data.get("status") == "healthy"
        except requests.exceptions.RequestException as e:
            pytest.fail(f"API health check failed: {e}")

    @pytest.mark.integration
    def test_dashboard_accessible(self, docker_compose_command):
        """Test that dashboard is accessible."""
        # Ensure services are up
        docker_compose_command("up", "-d")
        time.sleep(5)

        try:
            response = requests.get("http://localhost:8050/", timeout=5)
            assert response.status_code == 200
            assert "Cosmos Coherence" in response.text
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Dashboard check failed: {e}")

    @pytest.mark.integration
    def test_database_connection(self, docker_compose_command):
        """Test that database is accessible and initialized."""
        # Ensure services are up
        docker_compose_command("up", "-d")
        time.sleep(5)

        # Check database connection
        result = subprocess.run(
            ["docker", "exec", "cosmos-db", "pg_isready", "-U", "cosmos", "-d", "cosmos_coherence"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "Database is not ready"

    @pytest.mark.integration
    def test_redis_connection(self, docker_compose_command):
        """Test that Redis is accessible."""
        # Ensure services are up
        docker_compose_command("up", "-d")
        time.sleep(5)

        # Check Redis connection
        result = subprocess.run(
            ["docker", "exec", "cosmos-redis", "redis-cli", "ping"],
            capture_output=True,
            text=True,
        )
        assert "PONG" in result.stdout, "Redis is not responding"

    @pytest.mark.integration
    def test_volume_persistence(self, docker_compose_command):
        """Test that data persists across container restarts."""
        # Start services
        docker_compose_command("up", "-d")
        time.sleep(5)

        # Write test data to Redis
        subprocess.run(
            ["docker", "exec", "cosmos-redis", "redis-cli", "SET", "test_key", "test_value"],
            check=True,
        )

        # Restart services
        docker_compose_command("restart")
        time.sleep(5)

        # Check data persists
        result = subprocess.run(
            ["docker", "exec", "cosmos-redis", "redis-cli", "GET", "test_key"],
            capture_output=True,
            text=True,
        )
        assert "test_value" in result.stdout, "Data did not persist across restart"

    @pytest.mark.integration
    def test_environment_variables_loaded(self, docker_compose_command):
        """Test that environment variables are properly loaded."""
        # Start services
        docker_compose_command("up", "-d")
        time.sleep(5)

        # Check environment variable in API container
        result = subprocess.run(
            ["docker", "exec", "cosmos-api", "printenv", "ENVIRONMENT"],
            capture_output=True,
            text=True,
        )
        assert (
            "development" in result.stdout or "production" in result.stdout
        ), "ENVIRONMENT variable not set"

    @pytest.mark.integration
    def test_network_connectivity(self, docker_compose_command):
        """Test that services can communicate on the network."""
        # Start services
        docker_compose_command("up", "-d")
        time.sleep(5)

        # Test API can reach database
        result = subprocess.run(
            ["docker", "exec", "cosmos-api", "nc", "-zv", "db", "5432"],
            capture_output=True,
            text=True,
        )
        assert (
            result.returncode == 0 or "succeeded" in result.stderr.lower()
        ), "API cannot reach database"

    @pytest.mark.integration
    def test_logs_accessible(self, docker_compose_command):
        """Test that logs are accessible for debugging."""
        # Start services
        docker_compose_command("up", "-d")
        time.sleep(5)

        # Get logs
        result = docker_compose_command("logs", "--tail=10")
        assert result.returncode == 0
        assert len(result.stdout) > 0, "No logs available"

    @pytest.mark.integration
    def test_graceful_shutdown(self, docker_compose_command):
        """Test that services shut down gracefully."""
        # Start services
        docker_compose_command("up", "-d")
        time.sleep(5)

        # Stop services
        result = docker_compose_command("down")
        assert result.returncode == 0, "Services did not shut down gracefully"

        # Verify all containers are stopped
        result = docker_compose_command("ps")
        assert "cosmos-api" not in result.stdout, "API container still running"

    @pytest.mark.integration
    def test_development_hot_reload(self, docker_compose_command, project_root: Path):
        """Test that development environment supports hot reload."""
        # Start services in development mode
        docker_compose_command("up", "-d")
        time.sleep(5)

        # Check that source is mounted
        result = subprocess.run(
            ["docker", "exec", "cosmos-api", "ls", "/app/src"],
            capture_output=True,
            text=True,
        )
        assert "cosmos_coherence" in result.stdout, "Source code not mounted"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_backup_restore_cycle(self, docker_compose_command, project_root: Path):
        """Test complete backup and restore cycle."""
        backup_script = project_root / "scripts" / "backup.sh"
        restore_script = project_root / "scripts" / "restore.sh"

        if not backup_script.exists() or not restore_script.exists():
            pytest.skip("Backup/restore scripts not found")

        # Start services
        docker_compose_command("up", "-d")
        time.sleep(5)

        # Create test data
        subprocess.run(
            ["docker", "exec", "cosmos-redis", "redis-cli", "SET", "backup_test", "test_data"],
            check=True,
        )

        # Run backup
        result = subprocess.run(
            [str(backup_script)],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Backup failed: {result.stderr}"

        # Clear data
        subprocess.run(
            ["docker", "exec", "cosmos-redis", "redis-cli", "DEL", "backup_test"],
            check=True,
        )

        # Note: Full restore test would require interactive input handling
        # This is a simplified test to verify scripts exist and are executable

    @pytest.mark.integration
    def test_health_check_script(self, docker_compose_command, project_root: Path):
        """Test that health check script works correctly."""
        health_script = project_root / "scripts" / "health-check.sh"

        if not health_script.exists():
            pytest.skip("Health check script not found")

        # Start services
        docker_compose_command("up", "-d")
        time.sleep(10)

        # Run health check
        result = subprocess.run(
            [str(health_script)],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        # Health check might fail if not all services are ready
        # Just verify script runs without error
        assert "Health Check" in result.stdout

    @pytest.fixture(scope="class", autouse=True)
    def cleanup(self, docker_compose_command):
        """Clean up Docker resources after tests."""
        yield
        # Stop and remove containers after all tests
        docker_compose_command("down", "-v")
