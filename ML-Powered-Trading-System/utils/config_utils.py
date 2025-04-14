"""
Configuration utilities module for the ML-powered trading system.

This module provides helper functions for working with configuration objects,
including loading, merging, validation, and schema generation. It centralizes
common configuration operations to ensure consistency across the system.
"""
import json
import os
import re
import yaml
import logging
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, get_type_hints

# Import base and specific config types - adjust imports based on your project structure
from ..config.base_config import BaseConfig, ConfigManager
from ..config.system_config import SystemConfig, get_system_config
from ..config.strategy_config import StrategyConfig, get_strategy_config
from ..config.trading_mode_config import TradingModeConfig, get_trading_mode_config

# Type variable for generic config functions
T = TypeVar('T', bound=BaseConfig)

# Set up logger
logger = logging.getLogger(__name__)


def merge_configs(base_config: T, override_config: Union[T, Dict[str, Any]]) -> T:
    """
    Merge two configurations, with override_config taking precedence.

    Args:
        base_config: The base configuration to start with
        override_config: Configuration that overrides values in base_config

    Returns:
        A new configuration with merged values
    """
    # Convert override to dict if it's a config object
    if isinstance(override_config, BaseConfig):
        override_dict = asdict(override_config)
    else:
        override_dict = override_config

    # Create a deep copy of the base config to avoid modifying it
    base_dict = asdict(base_config)

    # Perform a deep merge
    merged_dict = _deep_merge(base_dict, override_dict)

    # Create a new instance of the same class as base_config
    config_class = base_config.__class__
    return config_class(**merged_dict)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.

    Args:
        base: Base dictionary
        override: Dictionary with values to override

    Returns:
        Merged dictionary
    """
    result = deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = _deep_merge(result[key], value)
        else:
            # Override or add value
            result[key] = deepcopy(value)

    return result


def get_config(config_type: str, config_path: Optional[Path] = None) -> BaseConfig:
    """
    Get a configuration by type.

    Args:
        config_type: Type of configuration to load ("system", "strategy", "trading_mode", etc.)
        config_path: Optional path to a specific configuration file

    Returns:
        The loaded configuration object

    Raises:
        ValueError: If config_type is not recognized
    """
    if config_type == "system":
        return get_system_config(config_path)
    elif config_type == "strategy":
        return get_strategy_config(config_path)
    elif config_type == "trading_mode":
        return get_trading_mode_config(config_path)
    else:
        raise ValueError(f"Unknown config type: {config_type}")


def save_config(config: BaseConfig, config_path: Optional[Path] = None) -> Path:
    """
    Save a configuration to a file.

    Args:
        config: Configuration object to save
        config_path: Optional path where to save the configuration

    Returns:
        Path where the configuration was saved
    """
    # Determine config type and default path
    config_type = _get_config_type(config)

    if config_path is None:
        config_path = ConfigManager.get_config_path(config_type)

    # Convert to dictionary
    config_dict = asdict(config)

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    # Determine file format based on extension
    extension = os.path.splitext(config_path)[1].lower()

    if extension == ".json":
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=_json_serializer)
    elif extension in (".yaml", ".yml"):
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported file extension: {extension}")

    logger.info(f"Saved {config_type} configuration to {config_path}")
    return config_path


def _get_config_type(config: BaseConfig) -> str:
    """
    Determine the type of a configuration object.

    Args:
        config: Configuration object

    Returns:
        Configuration type as string
    """
    if isinstance(config, SystemConfig):
        return "system"
    elif isinstance(config, StrategyConfig):
        return "strategy"
    elif isinstance(config, TradingModeConfig):
        return "trading_mode"
    else:
        # Try to infer from class name
        class_name = config.__class__.__name__
        if class_name.endswith("Config"):
            return re.sub(r"([A-Z])", r"_\1", class_name[:-6]).lower().lstrip("_")

        raise ValueError(f"Unknown configuration type: {config.__class__.__name__}")


def _json_serializer(obj: Any) -> Any:
    """
    Custom JSON serializer for types that aren't natively supported.

    Args:
        obj: Object to serialize

    Returns:
        JSON-serializable representation of the object
    """
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, Path):
        return str(obj)
    elif is_dataclass(obj):
        return asdict(obj)
    elif hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()

    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def validate_all_configs() -> Dict[str, List[str]]:
    """
    Validate all configurations in the system.

    Returns:
        Dictionary mapping config types to lists of validation errors
    """
    errors = {}

    try:
        system_config = get_system_config()
        system_errors = system_config.validate()
        if system_errors:
            errors["system"] = system_errors
    except Exception as e:
        errors["system"] = [f"Error loading system config: {str(e)}"]

    try:
        strategy_config = get_strategy_config()
        strategy_errors = strategy_config.validate()
        if strategy_errors:
            errors["strategy"] = strategy_errors
    except Exception as e:
        errors["strategy"] = [f"Error loading strategy config: {str(e)}"]

    try:
        trading_mode_config = get_trading_mode_config()
        trading_mode_errors = trading_mode_config.validate()
        if trading_mode_errors:
            errors["trading_mode"] = trading_mode_errors
    except Exception as e:
        errors["trading_mode"] = [f"Error loading trading mode config: {str(e)}"]

    return errors


def generate_config_schema(config_class: Type[BaseConfig]) -> Dict[str, Any]:
    """
    Generate a JSON schema for a configuration class.

    Args:
        config_class: The configuration class to generate a schema for

    Returns:
        A dictionary representing the JSON schema
    """
    schema = {
        "type": "object",
        "properties": {},
        "required": []
    }

    # Get type hints for the class
    type_hints = get_type_hints(config_class)

    # Get default values for fields
    default_instance = config_class()
    defaults = asdict(default_instance)

    for field_name, field_type in type_hints.items():
        # Skip private fields
        if field_name.startswith("_"):
            continue

        # Get field information
        field_schema = _get_field_schema(field_type, defaults.get(field_name))

        # Add to properties
        schema["properties"][field_name] = field_schema

        # Add to required fields if no default value
        if field_name not in defaults or defaults[field_name] is None:
            schema["required"].append(field_name)

    return schema


def _get_field_schema(field_type: Any, default_value: Any = None) -> Dict[str, Any]:
    """
    Generate a JSON schema for a field based on its type.

    Args:
        field_type: Type annotation of the field
        default_value: Default value of the field

    Returns:
        A dictionary representing the JSON schema for the field
    """
    import inspect
    from typing import get_origin, get_args

    # Get the origin and arguments of the type
    origin = get_origin(field_type)
    args = get_args(field_type)

    # Handle Union types
    if origin is Union:
        # Special case for Optional[X] (Union[X, None])
        if type(None) in args:
            # Remove None from the args
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                # Optional[X] -> same as X but not required
                return _get_field_schema(non_none_args[0], default_value)

        # General Union -> oneOf in JSON Schema
        return {
            "oneOf": [_get_field_schema(arg) for arg in args]
        }

    # Handle List types
    elif origin is list:
        item_type = args[0] if args else Any
        return {
            "type": "array",
            "items": _get_field_schema(item_type)
        }

    # Handle Dict types
    elif origin is dict:
        key_type = args[0] if args else Any
        value_type = args[1] if len(args) > 1 else Any

        schema = {
            "type": "object"
        }

        # If we have a specific value type, add additionalProperties
        if value_type is not Any:
            schema["additionalProperties"] = _get_field_schema(value_type)

        return schema

    # Handle Enum types
    elif inspect.isclass(field_type) and issubclass(field_type, Enum):
        return {
            "type": "string",
            "enum": [e.value for e in field_type]
        }

    # Handle basic types
    elif field_type is str:
        return {"type": "string"}
    elif field_type is int:
        return {"type": "integer"}
    elif field_type is float:
        return {"type": "number"}
    elif field_type is bool:
        return {"type": "boolean"}

    # Handle dataclasses
    elif inspect.isclass(field_type) and is_dataclass(field_type):
        return generate_config_schema(field_type)

    # Default to any type
    return {}


def generate_example_config(config_class: Type[BaseConfig]) -> Dict[str, Any]:
    """
    Generate an example configuration with default values.

    Args:
        config_class: The configuration class to generate an example for

    Returns:
        A dictionary representing an example configuration
    """
    # Create an instance with default values
    instance = config_class()

    # Convert to dictionary
    return asdict(instance)


def find_config_differences(config1: BaseConfig, config2: BaseConfig) -> Dict[str, Dict[str, Any]]:
    """
    Find differences between two configurations.

    Args:
        config1: First configuration
        config2: Second configuration

    Returns:
        Dictionary with 'only_in_first', 'only_in_second', and 'different_values' keys
    """
    dict1 = asdict(config1)
    dict2 = asdict(config2)

    return _dict_diff(dict1, dict2)


def _dict_diff(dict1: Dict[str, Any], dict2: Dict[str, Any], path: str = "") -> Dict[str, Dict[str, Any]]:
    """
    Helper function to recursively find differences between dictionaries.

    Args:
        dict1: First dictionary
        dict2: Second dictionary
        path: Current path in the nested structure

    Returns:
        Dictionary with differences
    """
    only_in_first = {}
    only_in_second = {}
    different_values = {}

    # Find keys only in dict1
    for key in dict1:
        if key not in dict2:
            if path:
                only_in_first[f"{path}.{key}"] = dict1[key]
            else:
                only_in_first[key] = dict1[key]

    # Find keys only in dict2
    for key in dict2:
        if key not in dict1:
            if path:
                only_in_second[f"{path}.{key}"] = dict2[key]
            else:
                only_in_second[key] = dict2[key]

    # Find different values for common keys
    for key in dict1:
        if key in dict2:
            current_path = f"{path}.{key}" if path else key

            # If both values are dictionaries, recurse
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                nested_diff = _dict_diff(dict1[key], dict2[key], current_path)
                only_in_first.update(nested_diff["only_in_first"])
                only_in_second.update(nested_diff["only_in_second"])
                different_values.update(nested_diff["different_values"])
            # Otherwise, compare values
            elif dict1[key] != dict2[key]:
                different_values[current_path] = {
                    "first": dict1[key],
                    "second": dict2[key]
                }

    return {
        "only_in_first": only_in_first,
        "only_in_second": only_in_second,
        "different_values": different_values
    }


def get_environment_overrides(config_class: Type[BaseConfig], env_prefix: str = "") -> Dict[str, Any]:
    """
    Get configuration overrides from environment variables.

    Args:
        config_class: The configuration class to get overrides for
        env_prefix: Prefix for environment variables

    Returns:
        Dictionary with overrides from environment variables
    """
    overrides = {}

    # Get all environment variables
    env_vars = {k: v for k, v in os.environ.items() if env_prefix and k.startswith(env_prefix)}

    # Process each environment variable
    for env_var, value in env_vars.items():
        # Remove prefix
        if env_prefix:
            config_path = env_var[len(env_prefix):].lstrip("_")
        else:
            config_path = env_var

        # Convert to nested dictionary keys
        keys = config_path.lower().split("_")

        # Build nested dictionary
        current = overrides
        for i, key in enumerate(keys):
            if i == len(keys) - 1:
                # Last key, set the value
                current[key] = _convert_env_value(value)
            else:
                # Not last key, ensure nested dictionary exists
                if key not in current:
                    current[key] = {}
                current = current[key]

    return overrides


def _convert_env_value(value: str) -> Any:
    """
    Convert environment variable string to appropriate Python type.

    Args:
        value: String value from environment variable

    Returns:
        Converted value
    """
    # Try to convert to bool
    if value.lower() in ("true", "yes", "y", "1"):
        return True
    elif value.lower() in ("false", "no", "n", "0"):
        return False

    # Try to convert to int
    try:
        return int(value)
    except ValueError:
        pass

    # Try to convert to float
    try:
        return float(value)
    except ValueError:
        pass

    # Handle lists (comma-separated values)
    if "," in value:
        return [_convert_env_value(item.strip()) for item in value.split(",")]

    # Default to string
    return value


def sanitize_config(config: BaseConfig, sensitive_keys: List[str] = None) -> Dict[str, Any]:
    """
    Create a sanitized version of a configuration object with sensitive information masked.

    Args:
        config: Configuration object to sanitize
        sensitive_keys: List of keys to mask (e.g., ["api_key", "password"])

    Returns:
        Sanitized configuration dictionary
    """
    if sensitive_keys is None:
        sensitive_keys = ["password", "secret", "key", "token", "credential"]

    # Convert to dictionary
    config_dict = asdict(config)

    # Sanitize recursively
    return _sanitize_dict(config_dict, sensitive_keys)


def _sanitize_dict(d: Dict[str, Any], sensitive_keys: List[str]) -> Dict[str, Any]:
    """
    Recursively sanitize a dictionary, masking sensitive values.

    Args:
        d: Dictionary to sanitize
        sensitive_keys: List of keys to mask

    Returns:
        Sanitized dictionary
    """
    sanitized = {}

    for key, value in d.items():
        # Check if key contains any sensitive patterns
        is_sensitive = any(pattern in key.lower() for pattern in sensitive_keys)

        if is_sensitive and value:
            # Mask sensitive value
            if isinstance(value, str):
                sanitized[key] = f"{'*' * min(6, len(value))}"
            else:
                sanitized[key] = "******"
        elif isinstance(value, dict):
            # Recursively sanitize nested dictionaries
            sanitized[key] = _sanitize_dict(value, sensitive_keys)
        elif isinstance(value, list):
            # Sanitize items in a list
            sanitized[key] = [
                _sanitize_dict(item, sensitive_keys) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            # Copy non-sensitive value as is
            sanitized[key] = value

    return sanitized


# Environment-specific configuration helpers

def load_environment_config(environment: str) -> Dict[str, BaseConfig]:
    """
    Load all configurations for a specific environment.

    Args:
        environment: Environment name (e.g., "development", "production")

    Returns:
        Dictionary mapping config types to loaded config objects
    """
    configs = {}

    # System config
    system_path = ConfigManager.get_config_path(f"system.{environment}")
    if Path(system_path).exists():
        configs["system"] = get_system_config(system_path)
    else:
        configs["system"] = get_system_config()

    # Strategy config
    strategy_path = ConfigManager.get_config_path(f"strategy.{environment}")
    if Path(strategy_path).exists():
        configs["strategy"] = get_strategy_config(strategy_path)
    else:
        configs["strategy"] = get_strategy_config()

    # Trading mode config
    trading_mode_path = ConfigManager.get_config_path(f"trading_mode.{environment}")
    if Path(trading_mode_path).exists():
        configs["trading_mode"] = get_trading_mode_config(trading_mode_path)
    else:
        configs["trading_mode"] = get_trading_mode_config()

    return configs


def create_configuration_snapshot(output_path: Optional[Path] = None) -> Path:
    """
    Create a snapshot of all current configurations.

    Args:
        output_path: Optional path where to save the snapshot

    Returns:
        Path to the created snapshot file
    """
    # Load all configurations
    system_config = get_system_config()
    strategy_config = get_strategy_config()
    trading_mode_config = get_trading_mode_config()

    # Create snapshot dictionary
    snapshot = {
        "timestamp": datetime.datetime.now().isoformat(),
        "system": asdict(system_config),
        "strategy": asdict(strategy_config),
        "trading_mode": asdict(trading_mode_config)
    }

    # Determine output path
    if output_path is None:
        system_paths = system_config.paths
        snapshots_dir = system_paths.backup_dir / "config_snapshots"
        os.makedirs(snapshots_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = snapshots_dir / f"config_snapshot_{timestamp}.json"

    # Save snapshot
    with open(output_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=_json_serializer)

    logger.info(f"Created configuration snapshot at {output_path}")
    return output_path


def restore_configuration_snapshot(snapshot_path: Path) -> None:
    """
    Restore configurations from a snapshot.

    Args:
        snapshot_path: Path to the snapshot file

    Raises:
        ValueError: If the snapshot file does not exist or has invalid format
    """
    if not snapshot_path.exists():
        raise ValueError(f"Snapshot file does not exist: {snapshot_path}")

    # Load snapshot
    with open(snapshot_path, "r") as f:
        snapshot = json.load(f)

    # Validate snapshot format
    required_keys = ["system", "strategy", "trading_mode"]
    for key in required_keys:
        if key not in snapshot:
            raise ValueError(f"Invalid snapshot format: missing '{key}' section")

    # Restore system config
    system_config = SystemConfig(**snapshot["system"])
    save_config(system_config)

    # Restore strategy config
    strategy_config = StrategyConfig(**snapshot["strategy"])
    save_config(strategy_config)

    # Restore trading mode config
    trading_mode_config = TradingModeConfig(**snapshot["trading_mode"])
    save_config(trading_mode_config)

    logger.info(f"Restored configuration from snapshot {snapshot_path}")


# Add missing imports
import datetime