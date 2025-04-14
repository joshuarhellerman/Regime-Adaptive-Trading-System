"""
Base configuration module that provides core functionality for all configs.

This module defines the base configuration classes and utilities that are used
by all other configuration modules in the system.
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast
import os
import yaml
import json


class ConfigSource(Enum):
    """Enumeration of possible configuration sources."""
    DEFAULT = "default"
    FILE = "file"
    ENV = "environment"
    COMMANDLINE = "commandline"
    DATABASE = "database"
    API = "api"
    DYNAMIC = "dynamic"


@dataclass
class ConfigMetadata:
    """Metadata about a configuration value."""
    source: ConfigSource = ConfigSource.DEFAULT
    timestamp: float = 0.0
    modified_by: str = ""


@dataclass
class BaseConfig:
    """Base class for all configuration objects."""
    # Class attribute to store defaults
    _defaults: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    # Instance attribute to store metadata about each config value
    _metadata: Dict[str, ConfigMetadata] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        """Initialize metadata for each field."""
        import time
        for field_name in self.__dataclass_fields__:
            if not field_name.startswith('_'):
                self._metadata[field_name] = ConfigMetadata(
                    timestamp=time.time()
                )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """Create a configuration object from a dictionary."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()
                        if not f.name.startswith('_')}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'BaseConfig':
        """Create a configuration object from a YAML file."""
        try:
            with open(yaml_path, 'r') as file:
                config_dict = yaml.safe_load(file)
                config = cls.from_dict(config_dict)

                # Update metadata
                for field_name in config.__dataclass_fields__:
                    if not field_name.startswith('_') and field_name in config_dict:
                        config._metadata[field_name].source = ConfigSource.FILE

                return config
        except (yaml.YAMLError, FileNotFoundError) as e:
            # Fall back to defaults if there's an error
            print(f"Warning: Unable to load config from {yaml_path}: {e}")
            return cls()

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'BaseConfig':
        """Create a configuration object from a JSON file."""
        try:
            with open(json_path, 'r') as file:
                config_dict = json.load(file)
                config = cls.from_dict(config_dict)

                # Update metadata
                for field_name in config.__dataclass_fields__:
                    if not field_name.startswith('_') and field_name in config_dict:
                        config._metadata[field_name].source = ConfigSource.FILE

                return config
        except (json.JSONDecodeError, FileNotFoundError) as e:
            # Fall back to defaults if there's an error
            print(f"Warning: Unable to load config from {json_path}: {e}")
            return cls()

    @classmethod
    def from_env(cls, prefix: str = "") -> 'BaseConfig':
        """Create a configuration object from environment variables."""
        config = cls()

        for field_name in cls.__dataclass_fields__:
            if field_name.startswith('_'):
                continue

            env_name = f"{prefix}_{field_name}".upper() if prefix else field_name.upper()

            if env_name in os.environ:
                # Get the expected type from the field
                field_type = cls.__dataclass_fields__[field_name].type

                try:
                    # Convert the environment variable to the expected type
                    if field_type == bool:
                        # Handle boolean values specially
                        value = os.environ[env_name].lower() in ('true', 'yes', '1', 'y')
                    elif field_type == int:
                        value = int(os.environ[env_name])
                    elif field_type == float:
                        value = float(os.environ[env_name])
                    elif field_type == list or field_type == List:
                        # Assume comma-separated list
                        value = os.environ[env_name].split(',')
                    elif field_type == dict or field_type == Dict:
                        # Assume JSON string
                        value = json.loads(os.environ[env_name])
                    else:
                        # Default to string
                        value = os.environ[env_name]

                    # Set the value and update metadata
                    setattr(config, field_name, value)
                    config._metadata[field_name].source = ConfigSource.ENV
                except (ValueError, json.JSONDecodeError) as e:
                    print(f"Warning: Failed to parse {env_name} as {field_type}: {e}")

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration object to a dictionary."""
        result = {}
        for field_name in self.__dataclass_fields__:
            if not field_name.startswith('_'):
                result[field_name] = getattr(self, field_name)
        return result

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save the configuration object to a YAML file."""
        with open(yaml_path, 'w') as file:
            yaml.dump(self.to_dict(), file, default_flow_style=False)

    def to_json(self, json_path: Union[str, Path]) -> None:
        """Save the configuration object to a JSON file."""
        with open(json_path, 'w') as file:
            json.dump(self.to_dict(), file, indent=2)

    def update(self, other: 'BaseConfig') -> None:
        """Update this configuration with values from another configuration."""
        for field_name in self.__dataclass_fields__:
            if not field_name.startswith('_'):
                if hasattr(other, field_name):
                    setattr(self, field_name, getattr(other, field_name))
                    # Update metadata
                    self._metadata[field_name] = other._metadata.get(
                        field_name, ConfigMetadata()
                    )

    def merge(self, *configs: 'BaseConfig') -> 'BaseConfig':
        """Create a new configuration by merging this one with others."""
        result = self.__class__(**self.to_dict())
        for config in configs:
            result.update(config)
        return result

    def get_source(self, field_name: str) -> ConfigSource:
        """Get the source of a configuration value."""
        if field_name in self._metadata:
            return self._metadata[field_name].source
        return ConfigSource.DEFAULT

    def validate(self) -> List[str]:
        """Validate the configuration. Return a list of validation errors."""
        # Base implementation does minimal validation
        errors = []

        # Check for required fields (non-Optional fields without defaults)
        for field_name, field_info in self.__dataclass_fields__.items():
            if field_name.startswith('_'):
                continue

            # Check if the field has a default value or is Optional
            has_default = field_info.default != field_info.default_factory
            is_optional = hasattr(field_info.type, "__origin__") and field_info.type.__origin__ is Union and type(
                None) in field_info.type.__args__

            if not has_default and not is_optional and getattr(self, field_name) is None:
                errors.append(f"Required field '{field_name}' is not set")

        return errors


# Type variable for configuration class type hints
T = TypeVar('T', bound=BaseConfig)


class ConfigManager:
    """
    Centralized configuration manager that handles loading, caching, and accessing
    configuration objects.
    """
    _configs: Dict[Type[BaseConfig], BaseConfig] = {}
    _config_paths: Dict[str, str] = {}

    @classmethod
    def register_config_path(cls, config_type: str, path: str) -> None:
        """Register a path for a specific configuration type."""
        cls._config_paths[config_type] = path

    @classmethod
    def get_config_path(cls, config_type: str) -> Optional[str]:
        """Get the path for a specific configuration type."""
        return cls._config_paths.get(config_type)

    @classmethod
    def load_config(cls, config_class: Type[T],
                    config_path: Optional[Union[str, Path]] = None,
                    env_prefix: str = "",
                    reload: bool = False) -> T:
        """
        Load a configuration of the specified type, potentially from a file.

        Args:
            config_class: The configuration class to load
            config_path: Path to the configuration file (YAML or JSON)
            env_prefix: Prefix for environment variables
            reload: Whether to force reload if the config is already loaded

        Returns:
            An instance of the requested configuration class
        """
        # Return cached config if available and reload is not requested
        if not reload and config_class in cls._configs:
            return cast(T, cls._configs[config_class])

        # Start with defaults
        config = config_class()

        # If a path is provided, load from file
        if config_path:
            if str(config_path).endswith('.yaml') or str(config_path).endswith('.yml'):
                file_config = config_class.from_yaml(config_path)
            elif str(config_path).endswith('.json'):
                file_config = config_class.from_json(config_path)
            else:
                raise ValueError(f"Unsupported config file format: {config_path}")

            config.update(file_config)

        # Override with environment variables if specified
        if env_prefix:
            env_config = config_class.from_env(env_prefix)
            config.update(env_config)

        # Validate the final config
        errors = config.validate()
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(errors)
            raise ValueError(error_msg)

        # Cache the config
        cls._configs[config_class] = config

        return cast(T, config)

    @classmethod
    def get_config(cls, config_class: Type[T]) -> T:
        """
        Get a loaded configuration of the specified type.

        Args:
            config_class: The configuration class to get

        Returns:
            The requested configuration

        Raises:
            KeyError: If the configuration has not been loaded
        """
        if config_class not in cls._configs:
            raise KeyError(f"Configuration of type {config_class.__name__} not loaded")

        return cast(T, cls._configs[config_class])

    @classmethod
    def reset(cls) -> None:
        """Reset the configuration manager, clearing all loaded configurations."""
        cls._configs.clear()