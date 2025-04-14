"""
Tests for configuration utilities module in the ML-powered trading system.

This module contains unit tests for the various functions in the configuration
utilities module, including loading, merging, validation, and schema generation.
"""
import json
import os
import tempfile
import unittest
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock, patch

# Import the module to be tested - adjust imports based on your project structure
from ...utils.config_utils import (
    merge_configs,
    get_config,
    save_config,
    validate_all_configs,
    generate_config_schema,
    generate_example_config,
    find_config_differences,
    get_environment_overrides,
    sanitize_config,
    load_environment_config,
    create_configuration_snapshot,
    restore_configuration_snapshot
)

# Import base config classes
from ...config.base_config import BaseConfig
from ...config.system_config import SystemConfig
from ...config.strategy_config import StrategyConfig
from ...config.trading_mode_config import TradingModeConfig


# Test fixtures
class SampleEnum(Enum):
    OPTION_A = "option_a"
    OPTION_B = "option_b"


@dataclass
class NestedConfig(BaseConfig):
    nested_value: int = 10
    nested_string: str = "default"

    def validate(self) -> List[str]:
        errors = []
        if self.nested_value < 0:
            errors.append("nested_value must be non-negative")
        return errors


@dataclass
class TestConfig(BaseConfig):
    name: str = "test"
    value: int = 42
    enabled: bool = True
    mode: SampleEnum = SampleEnum.OPTION_A
    nested: NestedConfig = field(default_factory=NestedConfig)
    list_values: List[int] = field(default_factory=lambda: [1, 2, 3])

    def validate(self) -> List[str]:
        errors = []
        if self.value < 0:
            errors.append("value must be non-negative")
        if not self.name:
            errors.append("name must not be empty")

        # Also validate nested config
        nested_errors = self.nested.validate()
        if nested_errors:
            errors.extend([f"nested.{error}" for error in nested_errors])

        return errors


class ConfigUtilsTest(unittest.TestCase):
    """Test case for configuration utilities module."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config = TestConfig()
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create patchers for imported functions
        self.get_system_config_patcher = patch('...utils.config_utils.get_system_config')
        self.get_strategy_config_patcher = patch('...utils.config_utils.get_strategy_config')
        self.get_trading_mode_config_patcher = patch('...utils.config_utils.get_trading_mode_config')

        # Start patchers
        self.mock_get_system_config = self.get_system_config_patcher.start()
        self.mock_get_strategy_config = self.get_strategy_config_patcher.start()
        self.mock_get_trading_mode_config = self.get_trading_mode_config_patcher.start()

        # Set up mock return values
        self.mock_get_system_config.return_value = SystemConfig()
        self.mock_get_strategy_config.return_value = StrategyConfig()
        self.mock_get_trading_mode_config.return_value = TradingModeConfig()

    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()

        # Stop patchers
        self.get_system_config_patcher.stop()
        self.get_strategy_config_patcher.stop()
        self.get_trading_mode_config_patcher.stop()

    def test_merge_configs(self):
        """Test merging configurations."""
        # Create base config
        base_config = TestConfig(
            name="base",
            value=100,
            nested=NestedConfig(nested_value=20)
        )

        # Create override config
        override_config = TestConfig(
            value=200,
            nested=NestedConfig(nested_string="overridden")
        )

        # Merge configs
        merged_config = merge_configs(base_config, override_config)

        # Verify merged values
        self.assertEqual(merged_config.name, "base")  # Unchanged from base
        self.assertEqual(merged_config.value, 200)  # Overridden
        self.assertEqual(merged_config.nested.nested_value, 20)  # Unchanged from base
        self.assertEqual(merged_config.nested.nested_string, "overridden")  # Overridden

        # Test merging with dictionary
        override_dict = {
            "value": 300,
            "nested": {"nested_value": 30}
        }

        merged_config = merge_configs(base_config, override_dict)

        # Verify merged values
        self.assertEqual(merged_config.name, "base")  # Unchanged from base
        self.assertEqual(merged_config.value, 300)  # Overridden
        self.assertEqual(merged_config.nested.nested_value, 30)  # Overridden
        self.assertEqual(merged_config.nested.nested_string, "default")  # Unchanged from base

    def test_get_config(self):
        """Test getting configuration by type."""
        # Mock the specific config getters
        system_config = SystemConfig()
        strategy_config = StrategyConfig()
        trading_mode_config = TradingModeConfig()

        self.mock_get_system_config.return_value = system_config
        self.mock_get_strategy_config.return_value = strategy_config
        self.mock_get_trading_mode_config.return_value = trading_mode_config

        # Test getting system config
        config = get_config("system")
        self.assertIs(config, system_config)
        self.mock_get_system_config.assert_called_once_with(None)

        # Test getting strategy config
        config = get_config("strategy")
        self.assertIs(config, strategy_config)
        self.mock_get_strategy_config.assert_called_once_with(None)

        # Test getting trading_mode config
        config = get_config("trading_mode")
        self.assertIs(config, trading_mode_config)
        self.mock_get_trading_mode_config.assert_called_once_with(None)

        # Test with custom path
        custom_path = Path("/custom/path/config.yaml")
        self.mock_get_system_config.reset_mock()

        config = get_config("system", custom_path)
        self.mock_get_system_config.assert_called_once_with(custom_path)

        # Test with unknown config type
        with self.assertRaises(ValueError):
            get_config("unknown_type")

    def test_save_config(self):
        """Test saving configuration to file."""
        # Create a test config
        test_config = TestConfig(name="test_save")

        # Create temporary paths for JSON and YAML files
        json_path = Path(self.temp_dir.name) / "test_config.json"
        yaml_path = Path(self.temp_dir.name) / "test_config.yaml"

        # Mock ConfigManager.get_config_path
        with patch('...utils.config_utils._get_config_type', return_value="test"):
            with patch('...config.base_config.ConfigManager.get_config_path', return_value=json_path):
                # Test saving to JSON without explicit path
                saved_path = save_config(test_config)
                self.assertEqual(saved_path, json_path)
                self.assertTrue(json_path.exists())

                # Verify the content
                with open(json_path, "r") as f:
                    saved_data = json.load(f)
                    self.assertEqual(saved_data["name"], "test_save")

                # Test saving to YAML with explicit path
                saved_path = save_config(test_config, yaml_path)
                self.assertEqual(saved_path, yaml_path)
                self.assertTrue(yaml_path.exists())

                # Test saving with unsupported extension
                invalid_path = Path(self.temp_dir.name) / "test_config.txt"
                with self.assertRaises(ValueError):
                    save_config(test_config, invalid_path)

    def test_validate_all_configs(self):
        """Test validating all configurations."""
        # Set up mock validations
        system_config = MagicMock(spec=SystemConfig)
        system_config.validate.return_value = ["System error 1", "System error 2"]

        strategy_config = MagicMock(spec=StrategyConfig)
        strategy_config.validate.return_value = []  # No errors

        trading_mode_config = MagicMock(spec=TradingModeConfig)
        trading_mode_config.validate.return_value = ["Trading mode error"]

        self.mock_get_system_config.return_value = system_config
        self.mock_get_strategy_config.return_value = strategy_config
        self.mock_get_trading_mode_config.return_value = trading_mode_config

        # Test validation
        errors = validate_all_configs()

        # Verify errors
        self.assertIn("system", errors)
        self.assertEqual(len(errors["system"]), 2)
        self.assertNotIn("strategy", errors)  # No errors for strategy
        self.assertIn("trading_mode", errors)
        self.assertEqual(len(errors["trading_mode"]), 1)

        # Test handling exceptions during config loading
        self.mock_get_system_config.side_effect = Exception("Failed to load system config")

        errors = validate_all_configs()
        self.assertIn("system", errors)
        self.assertTrue(errors["system"][0].startswith("Error loading system config"))

    def test_generate_config_schema(self):
        """Test generating JSON schema for configuration class."""
        # Generate schema for TestConfig
        schema = generate_config_schema(TestConfig)

        # Verify schema properties
        self.assertEqual(schema["type"], "object")
        self.assertIn("properties", schema)
        self.assertIn("name", schema["properties"])
        self.assertEqual(schema["properties"]["name"]["type"], "string")
        self.assertIn("value", schema["properties"])
        self.assertEqual(schema["properties"]["value"]["type"], "integer")
        self.assertIn("enabled", schema["properties"])
        self.assertEqual(schema["properties"]["enabled"]["type"], "boolean")
        self.assertIn("mode", schema["properties"])
        self.assertEqual(schema["properties"]["mode"]["type"], "string")
        self.assertIn("enum", schema["properties"]["mode"])

        # Check nested object schema
        self.assertIn("nested", schema["properties"])
        self.assertEqual(schema["properties"]["nested"]["type"], "object")
        self.assertIn("properties", schema["properties"]["nested"])

        # Check array schema
        self.assertIn("list_values", schema["properties"])
        self.assertEqual(schema["properties"]["list_values"]["type"], "array")

    def test_generate_example_config(self):
        """Test generating example configuration."""
        # Generate example for TestConfig
        example = generate_example_config(TestConfig)

        # Verify example values
        self.assertEqual(example["name"], "test")
        self.assertEqual(example["value"], 42)
        self.assertEqual(example["enabled"], True)
        self.assertEqual(example["mode"], "option_a")
        self.assertIn("nested", example)
        self.assertEqual(example["nested"]["nested_value"], 10)

    def test_find_config_differences(self):
        """Test finding differences between configurations."""
        # Create two configs with differences
        config1 = TestConfig(
            name="config1",
            value=100,
            nested=NestedConfig(nested_value=20)
        )

        config2 = TestConfig(
            name="config2",
            value=100,  # Same value
            nested=NestedConfig(nested_value=30)  # Different nested value
        )

        # Find differences
        diff = find_config_differences(config1, config2)

        # Verify differences
        self.assertIn("different_values", diff)
        self.assertIn("name", diff["different_values"])
        self.assertEqual(diff["different_values"]["name"]["first"], "config1")
        self.assertEqual(diff["different_values"]["name"]["second"], "config2")

        self.assertIn("nested.nested_value", diff["different_values"])
        self.assertEqual(diff["different_values"]["nested.nested_value"]["first"], 20)
        self.assertEqual(diff["different_values"]["nested.nested_value"]["second"], 30)

        # No values should be only in first or second
        self.assertEqual(diff["only_in_first"], {})
        self.assertEqual(diff["only_in_second"], {})

        # Test with additional field in one config
        config1_dict = {"name": "config1", "extra_field": "only in first"}
        config2_dict = {"name": "config2"}

        # Convert to configs
        config1 = TestConfig(**config1_dict)
        config2 = TestConfig(**config2_dict)

        diff = find_config_differences(config1, config2)

        # Verify differences
        self.assertIn("only_in_first", diff)
        self.assertIn("extra_field", diff["only_in_first"])

    def test_get_environment_overrides(self):
        """Test getting configuration overrides from environment variables."""
        # Set up environment variables
        env_vars = {
            "TEST_NAME": "env_name",
            "TEST_VALUE": "123",
            "TEST_ENABLED": "false",
            "TEST_NESTED_NESTED_VALUE": "50"
        }

        with patch.dict(os.environ, env_vars):
            # Get overrides
            overrides = get_environment_overrides(TestConfig, env_prefix="TEST_")

            # Verify overrides
            self.assertIn("name", overrides)
            self.assertEqual(overrides["name"], "env_name")
            self.assertIn("value", overrides)
            self.assertEqual(overrides["value"], 123)  # Converted to int
            self.assertIn("enabled", overrides)
            self.assertEqual(overrides["enabled"], False)  # Converted to bool
            self.assertIn("nested", overrides)
            self.assertIn("nested_value", overrides["nested"])
            self.assertEqual(overrides["nested"]["nested_value"], 50)

    def test_sanitize_config(self):
        """Test sanitizing configuration with sensitive information."""
        # Create a config with sensitive data
        @dataclass
        class SensitiveConfig(BaseConfig):
            username: str = "user"
            password: str = "secret123"
            api_key: str = "abcdef123456"
            normal_value: int = 42

            def validate(self) -> List[str]:
                return []

        config = SensitiveConfig()

        # Sanitize the config
        sanitized = sanitize_config(config)

        # Verify sanitization
        self.assertEqual(sanitized["username"], "user")  # Not sensitive
        self.assertEqual(sanitized["password"], "******")  # Masked
        self.assertEqual(sanitized["api_key"], "******")  # Masked
        self.assertEqual(sanitized["normal_value"], 42)  # Not sensitive

        # Test with custom sensitive keys
        sanitized = sanitize_config(config, sensitive_keys=["username"])
        self.assertEqual(sanitized["username"], "******")  # Now masked
        self.assertEqual(sanitized["password"], "secret123")  # Not in sensitive keys

    def test_load_environment_config(self):
        """Test loading configurations for a specific environment."""
        # Mock Path.exists
        with patch('pathlib.Path.exists', return_value=True):
            # Set up specific environment config mocks
            dev_system_config = SystemConfig()
            dev_strategy_config = StrategyConfig()
            dev_trading_mode_config = TradingModeConfig()

            # Configure mocks to return different configs based on path
            def mock_get_system_config_side_effect(path=None):
                if path and "development" in str(path):
                    return dev_system_config
                return SystemConfig()

            def mock_get_strategy_config_side_effect(path=None):
                if path and "development" in str(path):
                    return dev_strategy_config
                return StrategyConfig()

            def mock_get_trading_mode_config_side_effect(path=None):
                if path and "development" in str(path):
                    return dev_trading_mode_config
                return TradingModeConfig()

            self.mock_get_system_config.side_effect = mock_get_system_config_side_effect
            self.mock_get_strategy_config.side_effect = mock_get_strategy_config_side_effect
            self.mock_get_trading_mode_config.side_effect = mock_get_trading_mode_config_side_effect

            # Mock ConfigManager.get_config_path
            with patch('...config.base_config.ConfigManager.get_config_path') as mock_get_path:
                mock_get_path.side_effect = lambda x: Path(f"/mock/path/{x}.yaml")

                # Load environment configs
                configs = load_environment_config("development")

                # Verify configs
                self.assertIn("system", configs)
                self.assertIs(configs["system"], dev_system_config)
                self.assertIn("strategy", configs)
                self.assertIs(configs["strategy"], dev_strategy_config)
                self.assertIn("trading_mode", configs)
                self.assertIs(configs["trading_mode"], dev_trading_mode_config)

    def test_create_and_restore_configuration_snapshot(self):
        """Test creating and restoring configuration snapshots."""
        # Set up configs
        system_config = SystemConfig()
        strategy_config = StrategyConfig()
        trading_mode_config = TradingModeConfig()

        self.mock_get_system_config.return_value = system_config
        self.mock_get_strategy_config.return_value = strategy_config
        self.mock_get_trading_mode_config.return_value = trading_mode_config

        # Create temporary file for snapshot
        snapshot_path = Path(self.temp_dir.name) / "snapshot.json"

        # Mock datetime to get consistent timestamp
        with patch('...utils.config_utils.datetime') as mock_datetime:
            mock_datetime.datetime.now.return_value.isoformat.return_value = "2023-01-01T12:00:00"
            mock_datetime.datetime.now.return_value.strftime.return_value = "20230101_120000"

            # Mock system_paths
            with patch.object(system_config, 'paths') as mock_paths:
                mock_paths.backup_dir = Path(self.temp_dir.name)

                # Create snapshot
                result_path = create_configuration_snapshot(snapshot_path)
                self.assertEqual(result_path, snapshot_path)
                self.assertTrue(snapshot_path.exists())

                # Verify snapshot content
                with open(snapshot_path, "r") as f:
                    snapshot = json.load(f)
                    self.assertEqual(snapshot["timestamp"], "2023-01-01T12:00:00")
                    self.assertIn("system", snapshot)
                    self.assertIn("strategy", snapshot)
                    self.assertIn("trading_mode", snapshot)

        # Test restoring snapshot
        with patch('...utils.config_utils.save_config') as mock_save_config:
            # Restore snapshot
            restore_configuration_snapshot(snapshot_path)

            # Verify save_config calls
            self.assertEqual(mock_save_config.call_count, 3)

            # Test with non-existing snapshot
            with self.assertRaises(ValueError):
                restore_configuration_snapshot(Path("/non/existing/path"))

            # Test with invalid snapshot (missing section)
            invalid_snapshot_path = Path(self.temp_dir.name) / "invalid_snapshot.json"
            with open(invalid_snapshot_path, "w") as f:
                json.dump({"system": {}, "strategy": {}}, f)  # Missing trading_mode

            with self.assertRaises(ValueError):
                restore_configuration_snapshot(invalid_snapshot_path)


if __name__ == "__main__":
    unittest.main()