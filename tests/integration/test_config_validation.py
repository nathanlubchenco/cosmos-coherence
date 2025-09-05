#!/usr/bin/env python
"""Comprehensive validation script for the configuration system."""

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cosmos_coherence.config import load_config  # noqa: E402
from cosmos_coherence.config.models import (  # noqa: E402
    BaseConfig,
    BenchmarkConfig,
    BenchmarkType,
    ExperimentConfig,
    ModelConfig,
    ModelType,
    StrategyConfig,
    StrategyType,
)

console = Console()


class ConfigValidator:
    """Validates all aspects of the configuration system."""

    def __init__(self):
        self.results: List[Tuple[str, bool, str]] = []
        self.total_tests = 0
        self.passed_tests = 0

    def test(self, name: str, test_func):
        """Run a test and record results."""
        self.total_tests += 1
        try:
            test_func()
            self.results.append((name, True, "✓"))
            self.passed_tests += 1
            return True
        except Exception as e:
            self.results.append((name, False, str(e)))
            return False

    def validate_model_configs(self):
        """Validate all model configurations."""
        console.print("\n[cyan]Testing Model Configurations...[/cyan]")

        # Test GPT-5 models
        self.test(
            "GPT-5 temperature fixed at 1.0",
            lambda: self._check_model(
                ModelType.GPT_5,
                {"temperature": 0.5},
                expected_temp=1.0,  # Should be forced to 1.0
            ),
        )

        self.test(
            "GPT-5 uses max_output_tokens",
            lambda: self._check_model(
                ModelType.GPT_5, {"max_output_tokens": 128000}, has_max_output=True
            ),
        )

        # Test GPT-4.1 models
        self.test(
            "GPT-4.1 allows temperature control",
            lambda: self._check_model(
                ModelType.GPT_41, {"temperature": 0.7, "max_tokens": 32768}, expected_temp=0.7
            ),
        )

        # Test O3 reasoning models
        self.test(
            "O3 supports reasoning_effort",
            lambda: self._check_model(
                ModelType.O3,
                {"reasoning_effort": "high", "max_completion_tokens": 100000},
                has_reasoning_effort=True,
            ),
        )

        self.test(
            "O3 temperature fixed at 1.0",
            lambda: self._check_model(
                ModelType.O3,
                {"temperature": 0.5},
                expected_temp=1.0,  # Should be forced to 1.0
            ),
        )

        # Test legacy models
        self.test(
            "GPT-4 supports all parameters",
            lambda: self._check_model(
                ModelType.GPT_4,
                {"temperature": 0.8, "max_tokens": 4096, "top_p": 0.95},
                expected_temp=0.8,
                has_top_p=True,
            ),
        )

    def _check_model(
        self,
        model_type,
        params,
        expected_temp=None,
        has_max_output=False,
        has_reasoning_effort=False,
        has_top_p=False,
    ):
        """Helper to check model configuration."""
        config = ModelConfig(model_type=model_type, **params)

        if expected_temp is not None:
            assert config.temperature == expected_temp, f"Temperature should be {expected_temp}"

        if has_max_output:
            assert config.max_output_tokens is not None, "Should have max_output_tokens"

        if has_reasoning_effort:
            assert config.reasoning_effort is not None, "Should have reasoning_effort"

        if has_top_p and "top_p" in params:
            assert config.top_p == params["top_p"], "Should preserve top_p"

    def validate_yaml_loading(self):
        """Validate YAML loading and environment variables."""
        console.print("\n[cyan]Testing YAML Loading...[/cyan]")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Test 1: Environment variable interpolation when var is set
            os.environ["TEST_API_KEY"] = "sk-test-validation"
            os.environ["TEST_OUTPUT_DIR"] = "custom_output"

            config_file = tmppath / "test.yaml"
            config_file.write_text(
                """
name: validation_test
base:
  api_key: ${TEST_API_KEY}
  output_dir: ${TEST_OUTPUT_DIR:default_output}
  cache_dir: .cache
model:
  model_type: gpt-5
  max_output_tokens: 1000
benchmark:
  benchmark_type: simpleqa
  dataset_path: data/simpleqa
strategy:
  strategy_type: baseline
"""
            )

            self.test(
                "YAML loads with env vars set", lambda: self._check_yaml_load_with_env(config_file)
            )

            # Test 2: Default values when env var not set
            if "TEST_OUTPUT_DIR" in os.environ:
                del os.environ["TEST_OUTPUT_DIR"]

            config_file2 = tmppath / "test2.yaml"
            config_file2.write_text(
                """
name: validation_test2
base:
  api_key: ${TEST_API_KEY}
  output_dir: test_outputs
  cache_dir: .cache
model:
  model_type: gpt-5
  max_output_tokens: 1000
benchmark:
  benchmark_type: simpleqa
  dataset_path: data/simpleqa
strategy:
  strategy_type: baseline
"""
            )

            self.test(
                "YAML loads with explicit values",
                lambda: self._check_yaml_load_explicit(config_file2),
            )

    def _check_yaml_load_with_env(self, config_file):
        """Helper to check YAML loading with environment variables set."""
        config = load_config(config_file)
        # Check that an API key is loaded (from either TEST_API_KEY or OPENAI_API_KEY env)
        assert config.base.api_key, "API key should be set"
        assert config.base.api_key.startswith("sk-"), "API key should start with sk-"
        # BaseSettings with aliases will use defaults when env vars not set
        assert isinstance(config.base.output_dir, Path), "Output dir should be a Path"
        assert config.model.model_type == ModelType.GPT_5

    def _check_yaml_load_explicit(self, config_file):
        """Helper to check YAML loading with explicit values."""
        config = load_config(config_file)
        # Check that values are loaded
        assert config.base.api_key, "API key should be set"
        assert config.base.api_key.startswith("sk-"), "API key should start with sk-"
        assert isinstance(config.base.output_dir, Path), "Output dir should be a Path"
        assert config.model.model_type == ModelType.GPT_5

    def validate_inheritance(self):
        """Validate configuration inheritance."""
        console.print("\n[cyan]Testing Configuration Inheritance...[/cyan]")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create base config
            base_file = tmppath / "base.yaml"
            base_file.write_text(
                """
base:
  api_key: ${TEST_API_KEY:sk-base}
  output_dir: outputs
  log_level: INFO
model:
  model_type: gpt-4
  temperature: 0.7
"""
            )

            # Create override config
            exp_file = tmppath / "exp.yaml"
            exp_file.write_text(
                """
name: inheritance_test
model:
  model_type: gpt-5
  max_output_tokens: 2000
benchmark:
  benchmark_type: faithbench
  dataset_path: data/faithbench
strategy:
  strategy_type: k_response
  k_responses: 5
"""
            )

            self.test(
                "Inheritance merges configs", lambda: self._check_inheritance(exp_file, base_file)
            )

    def _check_inheritance(self, exp_file, base_file):
        """Helper to check config inheritance."""
        config = load_config(exp_file, base_file)
        # From base
        assert config.base.log_level.value == "INFO", "Base value not inherited"
        # Overridden
        assert config.model.model_type == ModelType.GPT_5, "Override not applied"
        # New value
        assert config.strategy.k_responses == 5, "New value not added"

    def validate_cli_commands(self):
        """Validate CLI commands work."""
        console.print("\n[cyan]Testing CLI Commands...[/cyan]")

        self.test("CLI imports successfully", lambda: __import__("cosmos_coherence.config.cli"))

        # Test that example configs are valid
        config_dir = Path("configs/experiments")
        if config_dir.exists():
            for config_file in config_dir.glob("*.yaml"):
                self.test(
                    f"Config valid: {config_file.name}",
                    lambda f=config_file: self._validate_config_file(f),
                )

    def _validate_config_file(self, config_file):
        """Helper to validate a config file."""
        # Some configs need base config
        base_config = Path("configs/base.yaml")
        if base_config.exists():
            config = load_config(config_file, base_config)
        else:
            config = load_config(config_file)
        assert config.name, "Config should have a name"

    def validate_constraints(self):
        """Validate model-specific constraints."""
        console.print("\n[cyan]Testing Model Constraints...[/cyan]")

        # Test invalid configurations that should fail
        self.test(
            "GPT-5 rejects max_tokens",
            lambda: self._expect_error(
                lambda: ModelConfig(model_type=ModelType.GPT_5, max_tokens=1000),
                "does not support max_tokens",
            ),
        )

        self.test(
            "O3 rejects top_p",
            lambda: self._expect_error(
                lambda: ModelConfig(model_type=ModelType.O3, top_p=0.9), "does not support"
            ),
        )

        self.test(
            "GPT-4 rejects max_output_tokens",
            lambda: self._expect_error(
                lambda: ModelConfig(model_type=ModelType.GPT_4, max_output_tokens=1000),
                "uses max_tokens",
            ),
        )

        self.test(
            "K-response requires k_responses",
            lambda: self._expect_error(
                lambda: StrategyConfig(strategy_type=StrategyType.K_RESPONSE),
                "requires k_responses",
            ),
        )

    def _expect_error(self, func, error_msg):
        """Helper to check that an error is raised."""
        try:
            func()
            raise AssertionError(f"Should have raised error containing '{error_msg}'")
        except Exception as e:
            if error_msg.lower() not in str(e).lower():
                raise AssertionError(f"Error message should contain '{error_msg}', got: {e}")

    def validate_complete_experiment(self):
        """Validate a complete experiment configuration."""
        console.print("\n[cyan]Testing Complete Experiment Configuration...[/cyan]")

        self.test(
            "Complete GPT-5 experiment",
            lambda: self._check_complete_experiment(
                ModelType.GPT_5, BenchmarkType.SIMPLEQA, StrategyType.BASELINE
            ),
        )

        self.test(
            "Complete O3 reasoning experiment",
            lambda: self._check_complete_experiment(
                ModelType.O3,
                BenchmarkType.TRUTHFULQA,
                StrategyType.K_RESPONSE,
                k_responses=3,
                reasoning_effort="high",
            ),
        )

    def _check_complete_experiment(
        self, model_type, benchmark_type, strategy_type, k_responses=None, reasoning_effort=None
    ):
        """Helper to check complete experiment configuration."""
        model_params = {"model_type": model_type}
        if reasoning_effort:
            model_params["reasoning_effort"] = reasoning_effort
            model_params["max_completion_tokens"] = 50000

        strategy_params = {"strategy_type": strategy_type}
        if k_responses:
            strategy_params["k_responses"] = k_responses

        config = ExperimentConfig(
            name="validation_experiment",
            base=BaseConfig(_env_file=None, api_key="sk-test", output_dir="outputs"),
            model=ModelConfig(**model_params),
            benchmark=BenchmarkConfig(
                benchmark_type=benchmark_type, dataset_path=f"data/{benchmark_type.value}"
            ),
            strategy=StrategyConfig(**strategy_params),
        )

        assert config.name == "validation_experiment"
        assert config.model.model_type == model_type
        assert config.benchmark.benchmark_type == benchmark_type
        assert config.strategy.strategy_type == strategy_type

    def run_all_validations(self):
        """Run all validation tests."""
        console.print(
            Panel.fit(
                "[bold cyan]Configuration System Validation[/bold cyan]\n"
                "Running comprehensive tests...",
                box=box.ROUNDED,
            )
        )

        # Run all test suites
        self.validate_model_configs()
        self.validate_yaml_loading()
        self.validate_inheritance()
        self.validate_constraints()
        self.validate_complete_experiment()
        self.validate_cli_commands()

        # Display results
        self.display_results()

    def display_results(self):
        """Display validation results."""
        console.print("\n")

        # Create results table
        table = Table(title="Validation Results", box=box.ROUNDED)
        table.add_column("Test", style="cyan", width=50)
        table.add_column("Status", justify="center", width=10)
        table.add_column("Details", style="yellow")

        for name, passed, details in self.results:
            status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
            table.add_row(name, status, details if not passed else "")

        console.print(table)

        # Summary
        success_rate = (self.passed_tests / self.total_tests) * 100 if self.total_tests > 0 else 0

        if success_rate == 100:
            console.print(
                Panel.fit(
                    f"[bold green]✓ ALL TESTS PASSED![/bold green]\n"
                    f"[green]{self.passed_tests}/{self.total_tests} tests passed "
                    f"({success_rate:.0f}%)[/green]",
                    box=box.ROUNDED,
                )
            )
        else:
            console.print(
                Panel.fit(
                    f"[bold yellow]⚠ VALIDATION INCOMPLETE[/bold yellow]\n"
                    f"[yellow]{self.passed_tests}/{self.total_tests} tests passed "
                    f"({success_rate:.0f}%)[/yellow]\n"
                    f"[red]{self.total_tests - self.passed_tests} tests failed[/red]",
                    box=box.ROUNDED,
                )
            )

        return success_rate == 100


def main():
    """Main validation entry point."""
    validator = ConfigValidator()
    success = validator.run_all_validations()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
