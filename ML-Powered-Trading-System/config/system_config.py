"""
System configuration module that defines global system settings,
paths, and environment variables.

This module provides centralized access to system-wide settings
that are used throughout the application.
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union
import os

from .base_config import BaseConfig, ConfigManager


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Environment(Enum):
    """System environment enumeration."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class SystemPaths:
    """Configuration for system paths."""
    # Base directory where the system is installed
    base_dir: Path = field(default_factory=lambda: Path(os.getcwd()))

    # Directory for data storage
    data_dir: Path = field(default_factory=lambda: Path(os.getcwd()) / "data")

    # Directory for logs
    log_dir: Path = field(default_factory=lambda: Path(os.getcwd()) / "logs")

    # Directory for configuration files
    config_dir: Path = field(default_factory=lambda: Path(os.getcwd()) / "config")

    # Directory for model storage
    model_dir: Path = field(default_factory=lambda: Path(os.getcwd()) / "models")

    # Directory for temporary files
    temp_dir: Path = field(default_factory=lambda: Path(os.getcwd()) / "temp")

    # Directory for backups
    backup_dir: Path = field(default_factory=lambda: Path(os.getcwd()) / "backups")

    def __post_init__(self):
        """Ensure all directories exist."""
        for attr_name in self.__dataclass_fields__:
            path = getattr(self, attr_name)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)


@dataclass
class SystemConfig(BaseConfig):
    """
    Global system configuration settings.

    This includes environment settings, paths, performance thresholds,
    and other system-wide parameters.
    """
    # System name and version
    system_name: str = "ML-Powered-Trading-System"
    version: str = "1.0.0"

    # Environment settings
    environment: Environment = Environment.DEVELOPMENT
    debug_mode: bool = True

    # System paths
    paths: SystemPaths = field(default_factory=SystemPaths)

    # Performance settings
    max_memory_usage_mb: int = 8192  # 8GB
    max_cpu_usage_percent: float = 80.0
    thread_pool_size: int = 8
    process_pool_size: int = 4

    # Timeouts and retry settings
    default_timeout_ms: int = 5000  # 5 seconds
    max_retries: int = 3
    retry_delay_ms: int = 1000  # 1 second

    # Scaling and capacity settings
    max_concurrent_requests: int = 100
    rate_limit_per_minute: int = 1000

    # Feature flags
    enable_telemetry: bool = True
    enable_auto_scaling: bool = False
    enable_rate_limiting: bool = True

    # Security settings
    enable_encryption: bool = True
    encryption_algorithm: str = "AES-256"

    # Monitoring settings
    health_check_interval_sec: int = 60
    metrics_reporting_interval_sec: int = 10

    def __post_init__(self):
        """Initialize the system configuration."""
        super().__post_init__()

        # Ensure paths is a SystemPaths object
        if isinstance(self.paths, dict):
            self.paths = SystemPaths(**self.paths)
        elif not isinstance(self.paths, SystemPaths):
            self.paths = SystemPaths()

        # Convert environment from string if needed
        if isinstance(self.environment, str):
            try:
                self.environment = Environment(self.environment.lower())
            except ValueError:
                print(f"Warning: Invalid environment '{self.environment}', defaulting to DEVELOPMENT")
                self.environment = Environment.DEVELOPMENT

    def is_production(self) -> bool:
        """Check if the system is running in production mode."""
        return self.environment == Environment.PRODUCTION

    def is_development(self) -> bool:
        """Check if the system is running in development mode."""
        return self.environment == Environment.DEVELOPMENT

    def validate(self) -> List[str]:
        """Validate the system configuration."""
        errors = super().validate()

        # Validate memory and CPU usage limits
        if self.max_memory_usage_mb <= 0:
            errors.append("max_memory_usage_mb must be positive")

        if not (0 < self.max_cpu_usage_percent <= 100):
            errors.append("max_cpu_usage_percent must be between 0 and 100")

        # Validate thread and process pool sizes
        if self.thread_pool_size <= 0:
            errors.append("thread_pool_size must be positive")

        if self.process_pool_size <= 0:
            errors.append("process_pool_size must be positive")

        # Validate timeouts and retries
        if self.default_timeout_ms <= 0:
            errors.append("default_timeout_ms must be positive")

        if self.max_retries < 0:
            errors.append("max_retries cannot be negative")

        if self.retry_delay_ms <= 0:
            errors.append("retry_delay_ms must be positive")

        return errors


def get_system_config(config_path: Optional[Union[str, Path]] = None) -> SystemConfig:
    """
    Get the system configuration.

    Args:
        config_path: Optional path to a configuration file. If not provided,
                    the default path from the ConfigManager will be used.

    Returns:
        The system configuration.
    """
    if config_path is None:
        config_path = ConfigManager.get_config_path("system")

    return ConfigManager.load_config(
        SystemConfig,
        config_path=config_path,
        env_prefix="TRADING_SYSTEM",
        reload=False
    )