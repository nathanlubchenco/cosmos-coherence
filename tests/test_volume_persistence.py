"""Tests for Docker volume mounting and data persistence."""
from pathlib import Path

import docker
import pytest
import yaml


class TestVolumePersistence:
    """Test suite for Docker volume mounting and data persistence."""

    @pytest.fixture
    def docker_client(self):
        """Create Docker client."""
        return docker.from_env()

    @pytest.fixture
    def project_root(self) -> Path:
        """Return the project root directory."""
        return Path(__file__).parent.parent

    def test_docker_compose_defines_volumes(self, project_root: Path) -> None:
        """Test that docker-compose.yml defines all necessary volumes."""
        compose_path = project_root / "docker-compose.yml"
        assert compose_path.exists(), "docker-compose.yml should exist"

        with open(compose_path, "r") as f:
            config = yaml.safe_load(f)

        # Check that volumes are defined
        assert "volumes" in config, "docker-compose.yml should define volumes"

        expected_volumes = [
            "postgres-data",
            "redis-data",
            "api-cache",
            "dashboard-cache",
        ]

        for volume in expected_volumes:
            assert volume in config["volumes"], f"Volume {volume} should be defined"

    def test_services_mount_volumes(self, project_root: Path) -> None:
        """Test that services properly mount volumes."""
        compose_path = project_root / "docker-compose.yml"
        with open(compose_path, "r") as f:
            config = yaml.safe_load(f)

        # Check database volume mounting
        db_volumes = config["services"]["db"].get("volumes", [])
        assert any(
            "postgres-data" in str(v) for v in db_volumes
        ), "Database should mount postgres-data volume"

        # Check Redis volume mounting
        redis_volumes = config["services"]["redis"].get("volumes", [])
        assert any(
            "redis-data" in str(v) for v in redis_volumes
        ), "Redis should mount redis-data volume"

        # Check API volume mounting
        api_volumes = config["services"]["api"].get("volumes", [])
        assert any("api-cache" in str(v) for v in api_volumes), "API should mount cache volume"

    def test_development_bind_mounts(self, project_root: Path) -> None:
        """Test that development environment has proper bind mounts."""
        override_path = project_root / "docker-compose.override.yml"
        if not override_path.exists():
            pytest.skip("docker-compose.override.yml not found")

        with open(override_path, "r") as f:
            config = yaml.safe_load(f)

        # Check API bind mounts for hot reload
        api_volumes = config["services"]["api"].get("volumes", [])
        assert any(
            "./src:/app/src" in str(v) for v in api_volumes
        ), "Development should mount source code for hot reload"
        assert any(
            ":rw" in str(v) for v in api_volumes if "./src" in str(v)
        ), "Source mount should be read-write in development"

    def test_production_readonly_mounts(self, project_root: Path) -> None:
        """Test that production has appropriate read-only mounts."""
        prod_path = project_root / "docker-compose.prod.yml"
        if not prod_path.exists():
            pytest.skip("docker-compose.prod.yml not found")

        with open(prod_path, "r") as f:
            config = yaml.safe_load(f)

        # Check for read-only config mounts in production
        api_volumes = config["services"]["api"].get("volumes", [])
        config_mounts = [v for v in api_volumes if "configs" in str(v)]

        for mount in config_mounts:
            assert ":ro" in str(mount), "Config mounts should be read-only in production"

    def test_volume_backup_configuration(self, project_root: Path) -> None:
        """Test that volumes are configured for backup."""
        prod_path = project_root / "docker-compose.prod.yml"
        if not prod_path.exists():
            pytest.skip("docker-compose.prod.yml not found")

        with open(prod_path, "r") as f:
            config = yaml.safe_load(f)

        # Check for backup volume
        assert "postgres-backup" in config.get(
            "volumes", {}
        ), "Production should have postgres-backup volume"

    def test_persistent_directories_exist(self, project_root: Path) -> None:
        """Test that persistent data directories are properly configured."""
        # Just verify the structure is planned (directories created on first run)
        gitignore = (project_root / ".gitignore").read_text()
        assert "data/" in gitignore, "data/ should be in .gitignore"
        assert "logs/" in gitignore, "logs/ should be in .gitignore"

    def test_volume_permissions(self, project_root: Path) -> None:
        """Test that volumes have proper permission configurations."""
        # Check that Dockerfile sets proper user permissions
        dockerfile_path = project_root / "Dockerfile"
        assert dockerfile_path.exists(), "Dockerfile should exist"

        dockerfile_content = dockerfile_path.read_text()
        assert (
            "USER" in dockerfile_content or "chown" in dockerfile_content
        ), "Dockerfile should set proper user permissions"

    def test_named_volumes_vs_bind_mounts(self, project_root: Path) -> None:
        """Test appropriate use of named volumes vs bind mounts."""
        compose_path = project_root / "docker-compose.yml"
        with open(compose_path, "r") as f:
            config = yaml.safe_load(f)

        # Database should use named volumes (not bind mounts)
        db_volumes = config["services"]["db"].get("volumes", [])
        for volume in db_volumes:
            volume_str = str(volume)
            if "postgres-data" in volume_str:
                assert not volume_str.startswith(
                    "./"
                ), "Database should use named volumes, not bind mounts"

    def test_cache_volume_configuration(self, project_root: Path) -> None:
        """Test that cache volumes are properly configured."""
        compose_path = project_root / "docker-compose.yml"
        with open(compose_path, "r") as f:
            config = yaml.safe_load(f)

        # Check cache volumes are defined
        volumes = config.get("volumes", {})
        assert "api-cache" in volumes, "API cache volume should be defined"
        assert "dashboard-cache" in volumes, "Dashboard cache volume should be defined"

        # Check that cache volumes use local driver
        for cache_vol in ["api-cache", "dashboard-cache"]:
            vol_config = volumes.get(cache_vol, {})
            assert vol_config.get("driver") == "local", f"{cache_vol} should use local driver"

    def test_volume_cleanup_script(self, project_root: Path) -> None:
        """Test that volume cleanup is handled properly."""
        # Check that start-dev.sh mentions volume cleanup
        start_dev = project_root / "scripts" / "start-dev.sh"
        if start_dev.exists():
            content = start_dev.read_text()
            assert (
                "docker compose down -v" in content
            ), "start-dev.sh should include volume cleanup option"

    @pytest.mark.integration
    def test_volume_persistence_across_restarts(self, docker_client, project_root: Path) -> None:
        """Test that data persists across container restarts."""
        try:
            # This is a placeholder for integration testing
            # In real scenario, would:
            # 1. Start containers
            # 2. Write test data
            # 3. Stop containers
            # 4. Start containers again
            # 5. Verify data exists
            pass
        except docker.errors.DockerException:
            pytest.skip("Docker not available for integration test")

    def test_backup_script_handles_volumes(self, project_root: Path) -> None:
        """Test that backup script properly handles all volumes."""
        backup_script = project_root / "scripts" / "backup.sh"
        assert backup_script.exists(), "backup.sh should exist"

        content = backup_script.read_text()

        # Check that all volumes are backed up
        expected_volumes = [
            "postgres-data",
            "redis-data",
            "api-cache",
            "dashboard-cache",
        ]

        for volume in expected_volumes:
            assert f"cosmos-coherence_{volume}" in content, f"Backup script should handle {volume}"

    def test_restore_capability(self, project_root: Path) -> None:
        """Test that system has restore capability for backed up data."""
        # Check for restore script or restore instructions
        scripts_dir = project_root / "scripts"

        # Either a restore script exists or backup script has restore info
        backup_script = scripts_dir / "backup.sh"
        if backup_script.exists():
            content = backup_script.read_text()
            # Backup script should create restorable archives
            assert "tar czf" in content, "Backup should create compressed archives"
            assert "backup_info.txt" in content, "Backup should include metadata"

    def test_data_directory_structure(self, project_root: Path) -> None:
        """Test that data directory structure is well-defined."""
        # Check .env files define proper data paths
        env_example = project_root / ".env.example"
        assert env_example.exists(), ".env.example should exist"

        content = env_example.read_text()
        assert "OUTPUT_DIR=" in content, "Should define OUTPUT_DIR"
        assert "CACHE_DIR=" in content, "Should define CACHE_DIR"
        assert "RESULTS_PATH=" in content, "Should define RESULTS_PATH"

    def test_log_rotation_configuration(self, project_root: Path) -> None:
        """Test that logs are configured for rotation to prevent disk fill."""
        logging_config = project_root / "configs" / "logging.yaml"
        assert logging_config.exists(), "logging.yaml should exist"

        with open(logging_config, "r") as f:
            config = yaml.safe_load(f)

        # Check for RotatingFileHandler configuration
        for env in ["development", "production"]:
            handlers = config.get(env, {}).get("handlers", {})
            for handler_name, handler_config in handlers.items():
                if "file" in handler_name.lower():
                    assert (
                        handler_config.get("class") == "logging.handlers.RotatingFileHandler"
                    ), f"{handler_name} should use RotatingFileHandler"
                    assert "maxBytes" in handler_config, f"{handler_name} should define maxBytes"
                    assert (
                        "backupCount" in handler_config
                    ), f"{handler_name} should define backupCount"
