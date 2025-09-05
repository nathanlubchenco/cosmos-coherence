"""YAML configuration loader with environment variable support."""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import ValidationError

from .models import ExperimentConfig


class ConfigLoader:
    """Configuration loader with YAML and environment variable support."""

    @staticmethod
    def interpolate_env_vars(text: str) -> str:
        """Replace ${VAR} or ${VAR:default} with environment variable values."""
        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2)

            # Get from environment or use default
            value = os.environ.get(var_name)
            if value is None:
                if default_value is not None:
                    return default_value
                else:
                    # Keep original if no env var and no default
                    return match.group(0)
            return value

        # Pattern matches ${VAR} or ${VAR:default}
        pattern = r'\$\{([^:}]+)(?::([^}]*))?\}'
        return re.sub(pattern, replacer, text)

    @classmethod
    def load_yaml(cls, file_path: Path) -> Dict[str, Any]:
        """Load YAML file with environment variable interpolation."""
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, 'r') as f:
            content = f.read()

        # Interpolate environment variables
        interpolated = cls.interpolate_env_vars(content)

        # Parse YAML
        try:
            config = yaml.safe_load(interpolated)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {file_path}: {e}")

        return config or {}

    @classmethod
    def process_includes(cls, config: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
        """Process !include tags to load additional YAML files."""
        if isinstance(config, dict):
            result = {}
            for key, value in config.items():
                if key == "!include":
                    # Load the included file
                    include_path = base_dir / value
                    included = cls.load_yaml(include_path)
                    # Merge included config
                    result.update(cls.process_includes(included, include_path.parent))
                else:
                    result[key] = cls.process_includes(value, base_dir)
            return result
        elif isinstance(config, list):
            return [cls.process_includes(item, base_dir) for item in config]
        else:
            return config

    @classmethod
    def merge_configs(cls, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configuration dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge dictionaries
                result[key] = cls.merge_configs(result[key], value)
            else:
                # Override the value
                result[key] = value

        return result

    @classmethod
    def load_experiment_config(
        cls,
        config_path: Path,
        base_config_path: Optional[Path] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> ExperimentConfig:
        """Load experiment configuration with optional base and overrides.

        Args:
            config_path: Path to main configuration file
            base_config_path: Optional path to base configuration
            overrides: Optional dictionary of override values

        Returns:
            Validated ExperimentConfig object
        """
        # Load base configuration if provided
        if base_config_path:
            base_config = cls.load_yaml(base_config_path)
            base_config = cls.process_includes(base_config, base_config_path.parent)
        else:
            base_config = {}

        # Load main configuration
        main_config = cls.load_yaml(config_path)
        main_config = cls.process_includes(main_config, config_path.parent)

        # Merge configurations
        config = cls.merge_configs(base_config, main_config)

        # Apply overrides if provided
        if overrides:
            config = cls.apply_overrides(config, overrides)

        # Create and validate the configuration object
        try:
            # Create BaseConfig using from_dict to avoid environment variable loading
            # The YAML has already been interpolated with the correct values
            if 'base' in config and isinstance(config['base'], dict):
                from cosmos_coherence.config.models import BaseConfig
                base_data = config.pop('base')
                base_config = BaseConfig.from_dict(base_data)
                config['base'] = base_config

            return ExperimentConfig(**config)
        except ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e}")

    @classmethod
    def apply_overrides(cls, config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Apply dot-notation overrides to configuration.

        Args:
            config: Base configuration dictionary
            overrides: Dictionary of overrides with dot-notation keys

        Returns:
            Configuration with overrides applied
        """
        result = config.copy()

        for key_path, value in overrides.items():
            keys = key_path.split('.')
            target = result

            # Navigate to the target location
            for key in keys[:-1]:
                if key not in target:
                    target[key] = {}
                target = target[key]

            # Set the value
            target[keys[-1]] = value

        return result

    @classmethod
    def save_config(cls, config: ExperimentConfig, file_path: Path) -> None:
        """Save configuration to YAML file.

        Args:
            config: ExperimentConfig object to save
            file_path: Path to save the configuration
        """
        # Convert to dictionary with mode='json' to serialize Path objects as strings
        config_dict = config.model_dump(exclude_none=True, mode='json')

        # Save to YAML
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def load_config(
    config_path: str | Path,
    base_path: Optional[str | Path] = None,
    **overrides: Any,
) -> ExperimentConfig:
    """Convenience function to load experiment configuration.

    Args:
        config_path: Path to configuration file
        base_path: Optional base configuration path
        **overrides: Keyword arguments for configuration overrides

    Returns:
        Loaded and validated ExperimentConfig
    """
    loader = ConfigLoader()

    config_path = Path(config_path)
    base_path = Path(base_path) if base_path else None

    return loader.load_experiment_config(config_path, base_path, overrides)
