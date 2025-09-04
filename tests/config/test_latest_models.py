"""Tests for latest OpenAI model configurations (GPT-5, o3, etc.)."""

import pytest
from pydantic import ValidationError

from cosmos_coherence.config.models import ModelConfig, ModelType


class TestGPT5Models:
    """Test GPT-5 model configurations."""

    def test_gpt5_config(self):
        """Test GPT-5 configuration."""
        config = ModelConfig(
            model_type=ModelType.GPT_5,
            max_output_tokens=128000,  # Max 128K for GPT-5
        )
        assert config.model_type == ModelType.GPT_5
        assert config.temperature == 1.0  # Fixed at 1
        assert config.max_output_tokens == 128000
        assert config.max_tokens is None

    def test_gpt5_mini_config(self):
        """Test GPT-5-mini configuration."""
        config = ModelConfig(
            model_type=ModelType.GPT_5_MINI,
            max_output_tokens=50000,
        )
        assert config.model_type == ModelType.GPT_5_MINI
        assert config.temperature == 1.0  # Fixed at 1

    def test_gpt5_nano_config(self):
        """Test GPT-5-nano configuration."""
        config = ModelConfig(
            model_type=ModelType.GPT_5_NANO,
            max_output_tokens=10000,
        )
        assert config.model_type == ModelType.GPT_5_NANO
        assert config.temperature == 1.0  # Fixed at 1

    def test_gpt5_invalid_params(self):
        """Test GPT-5 rejects unsupported parameters."""
        # Should reject max_tokens
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                model_type=ModelType.GPT_5,
                max_tokens=1000,  # Should fail
            )
        assert "max_tokens" in str(exc_info.value).lower()

        # Should reject top_p
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                model_type=ModelType.GPT_5,
                top_p=0.9,  # Should fail
            )
        assert "top_p" in str(exc_info.value).lower()

    def test_gpt5_temperature_forced(self):
        """Test GPT-5 forces temperature to 1."""
        config = ModelConfig(
            model_type=ModelType.GPT_5,
            temperature=0.5,  # Gets overridden to 1.0
        )
        assert config.temperature == 1.0


class TestGPT41Models:
    """Test GPT-4.1 model configurations."""

    def test_gpt41_config(self):
        """Test GPT-4.1 configuration."""
        config = ModelConfig(
            model_type=ModelType.GPT_41,
            temperature=0.8,
            max_tokens=32768,  # Max 32K output
        )
        assert config.model_type == ModelType.GPT_41
        assert config.temperature == 0.8  # Can be adjusted
        assert config.max_tokens == 32768

    def test_gpt41_mini_config(self):
        """Test GPT-4.1-mini configuration."""
        config = ModelConfig(
            model_type=ModelType.GPT_41_MINI,
            temperature=0.5,
            max_tokens=10000,
        )
        assert config.model_type == ModelType.GPT_41_MINI
        assert config.temperature == 0.5  # Can be adjusted

    def test_gpt41_max_output_validation(self):
        """Test GPT-4.1 max output token validation."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                model_type=ModelType.GPT_41,
                max_output_tokens=40000,  # Over 32K limit
            )
        assert "32,768" in str(exc_info.value)


class TestO3O4Models:
    """Test o3/o4 reasoning model configurations."""

    def test_o3_config(self):
        """Test o3 configuration."""
        config = ModelConfig(
            model_type=ModelType.O3,
            max_completion_tokens=100000,  # Max 100K for o3
            reasoning_effort="high",
        )
        assert config.model_type == ModelType.O3
        assert config.temperature == 1.0  # Fixed at 1
        assert config.max_completion_tokens == 100000
        assert config.reasoning_effort == "high"

    def test_o3_mini_config(self):
        """Test o3-mini configuration."""
        config = ModelConfig(
            model_type=ModelType.O3_MINI,
            max_completion_tokens=50000,
            reasoning_effort="medium",
        )
        assert config.model_type == ModelType.O3_MINI
        assert config.temperature == 1.0  # Fixed at 1
        assert config.reasoning_effort == "medium"

    def test_o4_mini_config(self):
        """Test o4-mini configuration."""
        config = ModelConfig(
            model_type=ModelType.O4_MINI,
            max_completion_tokens=65536,
            reasoning_effort="low",
        )
        assert config.model_type == ModelType.O4_MINI
        assert config.temperature == 1.0  # Fixed at 1
        assert config.reasoning_effort == "low"

    def test_o3_invalid_params(self):
        """Test o3 rejects unsupported parameters."""
        # Should reject max_tokens
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                model_type=ModelType.O3,
                max_tokens=1000,  # Should fail
            )
        assert "max_tokens" in str(exc_info.value).lower()

        # Should reject top_p
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                model_type=ModelType.O3,
                top_p=0.9,  # Should fail
            )
        assert "top_p" in str(exc_info.value).lower()

    def test_reasoning_effort_validation(self):
        """Test reasoning_effort parameter validation."""
        # Valid values
        for effort in ["low", "medium", "high"]:
            config = ModelConfig(
                model_type=ModelType.O3,
                reasoning_effort=effort,
            )
            assert config.reasoning_effort == effort

        # Invalid value
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                model_type=ModelType.O3,
                reasoning_effort="very_high",  # Invalid
            )
        # Pattern validation error

    def test_reasoning_effort_only_for_o3_o4(self):
        """Test reasoning_effort is only for o3/o4 models."""
        # Should fail for legacy o1 models
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                model_type=ModelType.O1_PREVIEW,
                reasoning_effort="high",  # Not supported
            )
        assert "reasoning_effort is only supported for o3/o4" in str(exc_info.value)


class TestLegacyModelCompatibility:
    """Test that legacy models still work correctly."""

    def test_gpt4_still_works(self):
        """Test GPT-4 configuration still works."""
        config = ModelConfig(
            model_type=ModelType.GPT_4,
            temperature=0.7,
            max_tokens=4096,
            top_p=0.95,
        )
        assert config.model_type == ModelType.GPT_4
        assert config.temperature == 0.7  # Can be adjusted
        assert config.max_tokens == 4096
        assert config.top_p == 0.95

    def test_gpt4_rejects_new_params(self):
        """Test GPT-4 rejects new parameters."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                model_type=ModelType.GPT_4,
                max_output_tokens=1000,  # New param, not for legacy
            )
        assert "max_output_tokens" in str(exc_info.value).lower()

        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(
                model_type=ModelType.GPT_4,
                reasoning_effort="high",  # New param, not for legacy
            )
        assert "reasoning_effort" in str(exc_info.value).lower()