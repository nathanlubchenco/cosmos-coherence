"""Tests for configuration loader functionality."""


import pytest
import yaml
from cosmos_coherence.config.loader import ConfigLoader, load_config
from cosmos_coherence.config.models import BenchmarkType, ModelType, StrategyType


class TestConfigLoader:
    """Test configuration loader."""

    def test_env_var_interpolation(self, monkeypatch):
        """Test environment variable interpolation."""
        monkeypatch.setenv("TEST_VAR", "test_value")
        monkeypatch.setenv("TEST_PATH", "/test/path")

        text = "key: ${TEST_VAR}, path: ${TEST_PATH}"
        result = ConfigLoader.interpolate_env_vars(text)
        assert result == "key: test_value, path: /test/path"

    def test_env_var_with_default(self):
        """Test environment variable with default value."""
        text = "key: ${NONEXISTENT:default_value}"
        result = ConfigLoader.interpolate_env_vars(text)
        assert result == "key: default_value"

    def test_env_var_no_default_keeps_original(self):
        """Test that missing env var without default keeps original."""
        text = "key: ${NONEXISTENT_VAR}"
        result = ConfigLoader.interpolate_env_vars(text)
        assert result == "key: ${NONEXISTENT_VAR}"

    def test_load_yaml_file(self, tmp_path, monkeypatch):
        """Test loading YAML file with env vars."""
        monkeypatch.setenv("API_KEY", "sk-test123")

        # Create test YAML file
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
name: test_experiment
base:
  api_key: ${API_KEY}
  output_dir: outputs
model:
  model_type: gpt-5
  max_output_tokens: 1000
""")

        # Load and verify
        config = ConfigLoader.load_yaml(config_file)
        assert config["name"] == "test_experiment"
        assert config["base"]["api_key"] == "sk-test123"
        assert config["model"]["model_type"] == "gpt-5"

    def test_merge_configs(self):
        """Test deep merging of configurations."""
        base = {
            "base": {"api_key": "sk-base", "output_dir": "outputs"},
            "model": {"model_type": "gpt-4", "temperature": 0.7},
            "benchmark": {"metrics": ["accuracy"]}
        }

        override = {
            "model": {"temperature": 0.9, "max_tokens": 1000},
            "benchmark": {"metrics": ["accuracy", "f1"], "sample_size": 100}
        }

        result = ConfigLoader.merge_configs(base, override)

        # Base values preserved
        assert result["base"]["api_key"] == "sk-base"
        assert result["model"]["model_type"] == "gpt-4"

        # Override values applied
        assert result["model"]["temperature"] == 0.9
        assert result["model"]["max_tokens"] == 1000
        assert result["benchmark"]["metrics"] == ["accuracy", "f1"]
        assert result["benchmark"]["sample_size"] == 100

    def test_apply_overrides_dot_notation(self):
        """Test applying dot-notation overrides."""
        config = {
            "model": {"model_type": "gpt-4", "temperature": 0.7},
            "benchmark": {"benchmark_type": "simpleqa"}
        }

        overrides = {
            "model.temperature": 0.9,
            "model.max_tokens": 2000,
            "benchmark.sample_size": 50
        }

        result = ConfigLoader.apply_overrides(config, overrides)

        assert result["model"]["temperature"] == 0.9
        assert result["model"]["max_tokens"] == 2000
        assert result["benchmark"]["sample_size"] == 50

    def test_load_experiment_config_with_base(self, tmp_path, monkeypatch):
        """Test loading experiment config with base configuration."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

        # Create base config
        base_file = tmp_path / "base.yaml"
        base_file.write_text("""
base:
  api_key: ${OPENAI_API_KEY}
  output_dir: outputs
  log_level: INFO
model:
  model_type: gpt-4
  temperature: 0.7
""")

        # Create experiment config
        exp_file = tmp_path / "experiment.yaml"
        exp_file.write_text("""
name: test_experiment
model:
  model_type: gpt-5
  max_output_tokens: 2000
benchmark:
  benchmark_type: simpleqa
  dataset_path: data/simpleqa
strategy:
  strategy_type: baseline
""")

        # Load with base
        config = ConfigLoader.load_experiment_config(exp_file, base_file)

        # Verify merged configuration
        assert config.name == "test_experiment"
        assert config.base.api_key == "sk-test-key"
        assert config.base.log_level.value == "INFO"
        assert config.model.model_type == ModelType.GPT_5
        assert config.model.max_output_tokens == 2000
        assert config.benchmark.benchmark_type == BenchmarkType.SIMPLEQA

    def test_load_config_with_overrides(self, tmp_path, monkeypatch):
        """Test load_config convenience function with overrides."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        # Create config file
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
name: test
base:
  api_key: ${OPENAI_API_KEY}
  output_dir: outputs
model:
  model_type: gpt-4
  temperature: 0.7
  max_tokens: 1000
benchmark:
  benchmark_type: faithbench
  dataset_path: data/faithbench
strategy:
  strategy_type: baseline
""")

        # Load with overrides
        config = load_config(
            config_file,
            **{
                "model.temperature": 0.9,
                "model.max_tokens": 2000,
                "benchmark.sample_size": 100
            }
        )

        # Verify overrides applied
        assert config.model.temperature == 0.9
        assert config.model.max_tokens == 2000
        assert config.benchmark.sample_size == 100

    def test_invalid_yaml_raises_error(self, tmp_path):
        """Test that invalid YAML raises an error."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [")

        with pytest.raises(ValueError, match="Invalid YAML"):
            ConfigLoader.load_yaml(config_file)

    def test_missing_file_raises_error(self, tmp_path):
        """Test that missing file raises an error."""
        config_file = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            ConfigLoader.load_yaml(config_file)

    def test_save_config(self, tmp_path, monkeypatch):
        """Test saving configuration to YAML."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        # Create a config
        from cosmos_coherence.config.models import (
            BaseConfig,
            BenchmarkConfig,
            ExperimentConfig,
            ModelConfig,
            StrategyConfig,
        )

        config = ExperimentConfig(
            name="saved_test",
            base=BaseConfig(_env_file=None, api_key="sk-test", output_dir="outputs"),
            model=ModelConfig(model_type=ModelType.GPT_5),
            benchmark=BenchmarkConfig(
                benchmark_type=BenchmarkType.SIMPLEQA,
                dataset_path="data/simpleqa"
            ),
            strategy=StrategyConfig(strategy_type=StrategyType.BASELINE)
        )

        # Save to file
        save_path = tmp_path / "saved_config.yaml"
        ConfigLoader.save_config(config, save_path)

        # Verify file exists and can be loaded
        assert save_path.exists()
        loaded = yaml.safe_load(save_path.read_text())
        assert loaded["name"] == "saved_test"
        assert loaded["model"]["model_type"] == "gpt-5"
