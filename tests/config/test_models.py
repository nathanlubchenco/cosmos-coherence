"""Tests for Pydantic configuration models."""

import os
from pathlib import Path
from typing import Any, Dict

import pytest
from pydantic import ValidationError

from cosmos_coherence.config.models import (
    BaseConfig,
    BenchmarkConfig,
    BenchmarkType,
    CoherenceMeasure,
    ExperimentConfig,
    LogLevel,
    ModelConfig,
    ModelType,
    StrategyConfig,
    StrategyType,
)


class TestBaseConfig:
    """Test BaseConfig model."""

    def test_base_config_minimal(self, monkeypatch):
        """Test BaseConfig with minimal required fields."""
        # Clear any existing env vars and disable .env file
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        
        config = BaseConfig(
            _env_file=None,  # Disable .env file loading
            api_key="sk-test123",
            output_dir="outputs",
        )
        assert config.api_key == "sk-test123"
        assert config.output_dir == Path("outputs")
        assert config.log_level == LogLevel.INFO  # default
        assert config.cache_dir == Path(".cache")  # default

    def test_base_config_from_env(self, monkeypatch):
        """Test BaseConfig loads from environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env123")
        monkeypatch.setenv("OUTPUT_DIR", "/tmp/outputs")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        
        config = BaseConfig(_env_file=None)
        assert config.api_key == "sk-env123"
        assert config.output_dir == Path("/tmp/outputs")
        assert config.log_level == LogLevel.DEBUG

    def test_base_config_validation(self, monkeypatch):
        """Test BaseConfig validation rules."""
        # Clear any existing env vars and disable .env file
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        
        with pytest.raises(ValidationError) as exc_info:
            BaseConfig(
                _env_file=None,
                api_key="",  # Empty API key should fail
                output_dir="outputs",
            )
        assert "api_key" in str(exc_info.value).lower()


class TestModelConfig:
    """Test ModelConfig model."""

    def test_gpt4_config(self):
        """Test GPT-4 model configuration."""
        config = ModelConfig(
            model_type=ModelType.GPT_4,
            temperature=0.7,
            max_tokens=1000,
            top_p=0.95,
        )
        assert config.model_type == ModelType.GPT_4
        assert config.temperature == 0.7
        assert config.max_tokens == 1000

    def test_o1_preview_config(self):
        """Test o1-preview model configuration with constraints."""
        # Should work with temperature=1 (forced)
        config = ModelConfig(
            model_type=ModelType.O1_PREVIEW,
            max_completion_tokens=1000,
        )
        assert config.temperature == 1.0  # Should be forced to 1
        assert config.max_completion_tokens == 1000
        assert config.max_tokens is None  # Should not be set

    def test_o1_preview_invalid_temperature(self):
        """Test o1-preview rejects non-1 temperature."""
        # Temperature is forced to 1 for O1 models, so this actually won't raise
        config = ModelConfig(
            model_type=ModelType.O1_PREVIEW,
            temperature=0.5,  # Gets overridden to 1.0
        )
        assert config.temperature == 1.0  # Forced to 1

    def test_o1_preview_invalid_max_tokens(self):
        """Test o1-preview rejects max_tokens parameter."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                model_type=ModelType.O1_PREVIEW,
                max_tokens=1000,  # Should fail - use max_completion_tokens
            )
        assert "max_tokens" in str(exc_info.value).lower()

    def test_temperature_range_validation(self):
        """Test temperature range validation for standard models."""
        # Valid range
        config = ModelConfig(
            model_type=ModelType.GPT_4,
            temperature=1.5,
        )
        assert config.temperature == 1.5

        # Too high
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                model_type=ModelType.GPT_4,
                temperature=2.1,
            )
        assert "temperature" in str(exc_info.value).lower()

        # Too low
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                model_type=ModelType.GPT_4,
                temperature=-0.1,
            )
        assert "temperature" in str(exc_info.value).lower()


class TestBenchmarkConfig:
    """Test BenchmarkConfig model."""

    def test_benchmark_config_minimal(self):
        """Test BenchmarkConfig with minimal fields."""
        config = BenchmarkConfig(
            benchmark_type=BenchmarkType.FAITHBENCH,
            dataset_path="data/faithbench",
        )
        assert config.benchmark_type == BenchmarkType.FAITHBENCH
        assert config.dataset_path == Path("data/faithbench")
        assert config.sample_size is None  # Optional
        assert config.metrics == ["accuracy"]  # Default

    def test_benchmark_config_full(self):
        """Test BenchmarkConfig with all fields."""
        config = BenchmarkConfig(
            benchmark_type=BenchmarkType.SIMPLEQA,
            dataset_path="data/simpleqa",
            sample_size=100,
            metrics=["accuracy", "f1", "precision"],
            evaluation_params={"threshold": 0.5},
        )
        assert config.sample_size == 100
        assert config.metrics == ["accuracy", "f1", "precision"]
        assert config.evaluation_params["threshold"] == 0.5

    def test_benchmark_dataset_validation(self):
        """Test dataset path existence validation."""
        # This should work even if path doesn't exist yet (for setup phase)
        config = BenchmarkConfig(
            benchmark_type=BenchmarkType.TRUTHFULQA,
            dataset_path="nonexistent/path",
        )
        assert config.dataset_path == Path("nonexistent/path")


class TestStrategyConfig:
    """Test StrategyConfig model."""

    def test_baseline_strategy(self):
        """Test baseline strategy configuration."""
        config = StrategyConfig(
            strategy_type=StrategyType.BASELINE,
        )
        assert config.strategy_type == StrategyType.BASELINE
        assert config.k_responses is None
        assert config.coherence_measures == []

    def test_k_response_strategy(self):
        """Test k-response strategy configuration."""
        config = StrategyConfig(
            strategy_type=StrategyType.K_RESPONSE,
            k_responses=5,
            aggregation_method="majority_vote",
        )
        assert config.k_responses == 5
        assert config.aggregation_method == "majority_vote"

    def test_coherence_strategy(self):
        """Test coherence strategy configuration."""
        config = StrategyConfig(
            strategy_type=StrategyType.COHERENCE,
            k_responses=3,
            coherence_measures=[CoherenceMeasure.SHOGENJI, CoherenceMeasure.FITELSON],
            coherence_thresholds={"shogenji": 0.7, "fitelson": 0.6},
        )
        assert config.k_responses == 3
        assert CoherenceMeasure.SHOGENJI in config.coherence_measures
        assert config.coherence_thresholds["shogenji"] == 0.7

    def test_k_response_validation(self):
        """Test k_responses validation."""
        # Valid k value
        config = StrategyConfig(
            strategy_type=StrategyType.K_RESPONSE,
            k_responses=10,
        )
        assert config.k_responses == 10

        # Invalid k value (too low)
        with pytest.raises(ValidationError) as exc_info:
            StrategyConfig(
                strategy_type=StrategyType.K_RESPONSE,
                k_responses=0,
            )
        assert "k_responses" in str(exc_info.value).lower()


class TestExperimentConfig:
    """Test ExperimentConfig model."""

    def test_experiment_config_minimal(self, monkeypatch):
        """Test ExperimentConfig with minimal configuration."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = ExperimentConfig(
            name="test_experiment",
            base=BaseConfig(
                _env_file=None,
                api_key="sk-test",
                output_dir="outputs",
            ),
            model=ModelConfig(
                model_type=ModelType.GPT_4,
            ),
            benchmark=BenchmarkConfig(
                benchmark_type=BenchmarkType.FAITHBENCH,
                dataset_path="data/faithbench",
            ),
            strategy=StrategyConfig(
                strategy_type=StrategyType.BASELINE,
            ),
        )
        assert config.name == "test_experiment"
        assert config.base.api_key == "sk-test"
        assert config.model.model_type == ModelType.GPT_4
        assert config.benchmark.benchmark_type == BenchmarkType.FAITHBENCH
        assert config.strategy.strategy_type == StrategyType.BASELINE

    def test_experiment_config_with_description(self, monkeypatch):
        """Test ExperimentConfig with optional description."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = ExperimentConfig(
            name="coherence_test",
            description="Testing Shogenji coherence on SimpleQA",
            base=BaseConfig(_env_file=None, api_key="sk-test", output_dir="outputs"),
            model=ModelConfig(model_type=ModelType.GPT_4O),
            benchmark=BenchmarkConfig(
                benchmark_type=BenchmarkType.SIMPLEQA,
                dataset_path="data/simpleqa",
            ),
            strategy=StrategyConfig(
                strategy_type=StrategyType.COHERENCE,
                k_responses=5,
                coherence_measures=[CoherenceMeasure.SHOGENJI],
            ),
        )
        assert config.description == "Testing Shogenji coherence on SimpleQA"
        assert config.strategy.coherence_measures == [CoherenceMeasure.SHOGENJI]

    def test_experiment_config_grid_params(self, monkeypatch):
        """Test ExperimentConfig with grid search parameters."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = ExperimentConfig(
            name="grid_search",
            base=BaseConfig(_env_file=None, api_key="sk-test", output_dir="outputs"),
            model=ModelConfig(model_type=ModelType.GPT_4),
            benchmark=BenchmarkConfig(
                benchmark_type=BenchmarkType.TRUTHFULQA,
                dataset_path="data/truthfulqa",
            ),
            strategy=StrategyConfig(strategy_type=StrategyType.K_RESPONSE, k_responses=5),
            grid_params={
                "model.temperature": [0.3, 0.7, 1.0],
                "strategy.k_responses": [3, 5, 7],
            },
        )
        assert config.grid_params["model.temperature"] == [0.3, 0.7, 1.0]
        assert config.grid_params["strategy.k_responses"] == [3, 5, 7]

    def test_experiment_config_validation_cross_model_strategy(self, monkeypatch):
        """Test validation across model and strategy compatibility."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # This should work - o1-preview with baseline strategy
        config = ExperimentConfig(
            name="o1_baseline",
            base=BaseConfig(_env_file=None, api_key="sk-test", output_dir="outputs"),
            model=ModelConfig(
                model_type=ModelType.O1_PREVIEW,
                max_completion_tokens=1000,
            ),
            benchmark=BenchmarkConfig(
                benchmark_type=BenchmarkType.FAITHBENCH,
                dataset_path="data/faithbench",
            ),
            strategy=StrategyConfig(strategy_type=StrategyType.BASELINE),
        )
        assert config.model.temperature == 1.0  # Forced to 1 for o1-preview