"""
Logging configuration module that defines logging levels, formats, and destinations.

This module provides configuration for the system's logging framework, including
log levels, rotation policies, and output formats.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Union
from pathlib import Path
import os

from .base_config import BaseConfig, ConfigManager


class LogLevel(Enum):
    """Enumeration of log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(Enum):
    """Enumeration of log formats."""
    SIMPLE = "simple"
    DETAILED = "detailed"
    JSON = "json"
    STRUCTURED = "structured"
    CUSTOM = "custom"


class LogDestination(Enum):
    """Enumeration of log destinations."""
    CONSOLE = "console"
    FILE = "file"
    SYSLOG = "syslog"
    REMOTE = "remote"
    DATABASE = "database"


@dataclass
class LogHandlerConfig(BaseConfig):
    """Configuration for a log handler."""
    # Handler identification
    handler_id: str = ""
    handler_type: LogDestination = LogDestination.CONSOLE

    # Log level for this handler
    level: LogLevel = LogLevel.INFO

    # Formatting
    format_type: LogFormat = LogFormat.SIMPLE
    custom_format: str = ""

    # File-specific settings
    file_path: str = ""
    max_size_mb: int = 10
    backup_count: int = 5

    # Remote-specific settings
    remote_host: str = ""
    remote_port: int = 0
    use_ssl: bool = False

    # Database-specific settings
    database_instance_id: str = "default"
    database_table: str = "logs"

    # Filtering
    include_modules: List[str] = field(default_factory=list)
    exclude_modules: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize the log handler configuration."""
        super().__post_init__()

        # Convert handler_type from string if needed
        if isinstance(self.handler_type, str):
            try:
                self.handler_type = LogDestination(self.handler_type.lower())
            except ValueError:
                print(f"Warning: Invalid handler_type '{self.handler_type}', defaulting to CONSOLE")
                self.handler_type = LogDestination.CONSOLE

        # Convert level from string if needed
        if isinstance(self.level, str):
            try:
                self.level = LogLevel(self.level.upper())
            except ValueError:
                print(f"Warning: Invalid level '{self.level}', defaulting to INFO")
                self.level = LogLevel.INFO

        # Convert format_type from string if needed
        if isinstance(self.format_type, str):
            try:
                self.format_type = LogFormat(self.format_type.lower())
            except ValueError:
                print(f"Warning: Invalid format_type '{self.format_type}', defaulting to SIMPLE")
                self.format_type = LogFormat.SIMPLE

        # Set default formats based on format_type
        if not self.custom_format:
            if self.format_type == LogFormat.SIMPLE:
                self.custom_format = "%(asctime)s [%(levelname)s] %(message)s"
            elif self.format_type == LogFormat.DETAILED:
                self.custom_format = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s"
            elif self.format_type == LogFormat.STRUCTURED:
                self.custom_format = "%(asctime)s [%(levelname)s] [%(name)s] [%(thread)d] [%(process)d] - %(message)s"

    def validate(self) -> List[str]:
        """Validate the log handler configuration."""
        errors = super().validate()

        # Validate handler_id
        if not self.handler_id:
            errors.append("handler_id must not be empty")

        # Format validation
        if self.format_type == LogFormat.CUSTOM and not self.custom_format:
            errors.append("custom_format must not be empty when format_type is CUSTOM")

        # File-specific validation
        if self.handler_type == LogDestination.FILE:
            if not self.file_path:
                errors.append("file_path must not be empty for FILE handlers")

            if self.max_size_mb <= 0:
                errors.append("max_size_mb must be positive for FILE handlers")

            if self.backup_count < 0:
                errors.append("backup_count cannot be negative for FILE handlers")

        # Remote-specific validation
        if self.handler_type == LogDestination.REMOTE:
            if not self.remote_host:
                errors.append("remote_host must not be empty for REMOTE handlers")

            if self.remote_port <= 0 or self.remote_port > 65535:
                errors.append("remote_port must be between 1 and 65535 for REMOTE handlers")

        # Database-specific validation
        if self.handler_type == LogDestination.DATABASE:
            if not self.database_instance_id:
                errors.append("database_instance_id must not be empty for DATABASE handlers")

            if not self.database_table:
                errors.append("database_table must not be empty for DATABASE handlers")

        return errors


@dataclass
class LoggerConfig(BaseConfig):
    """Configuration for a specific logger."""
    # Logger identification
    logger_name: str = ""

    # Log level for this logger
    level: LogLevel = LogLevel.INFO

    # Propagation setting
    propagate: bool = True

    # Handler references
    handlers: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize the logger configuration."""
        super().__post_init__()

        # Convert level from string if needed
        if isinstance(self.level, str):
            try:
                self.level = LogLevel(self.level.upper())
            except ValueError:
                print(f"Warning: Invalid level '{self.level}', defaulting to INFO")
                self.level = LogLevel.INFO

    def validate(self) -> List[str]:
        """Validate the logger configuration."""
        errors = super().validate()

        # Validate logger_name
        if not self.logger_name:
            errors.append("logger_name must not be empty")

        return errors


@dataclass
class PerformanceMonitoringConfig(BaseConfig):
    """Configuration for logging performance metrics."""
    # Monitoring settings
    enable_performance_logging: bool = True
    log_slow_operations: bool = True
    slow_operation_threshold_ms: int = 100

    # Sampling settings
    enable_sampling: bool = True
    sampling_rate: float = 0.1  # Log 10% of operations

    # Detailed logging settings
    log_memory_usage: bool = True
    log_cpu_usage: bool = True
    log_disk_io: bool = True
    log_network_io: bool = True

    def validate(self) -> List[str]:
        """Validate the performance monitoring configuration."""
        errors = super().validate()

        # Validate threshold
        if self.slow_operation_threshold_ms <= 0:
            errors.append("slow_operation_threshold_ms must be positive")

        # Validate sampling rate
        if not (0 <= self.sampling_rate <= 1):
            errors.append("sampling_rate must be between 0 and 1")

        return errors


@dataclass
class LogRotationConfig(BaseConfig):
    """Configuration for log rotation."""
    # Rotation settings
    enable_rotation: bool = True
    rotation_interval: str = "daily"  # "hourly", "daily", "weekly", "monthly"

    # Size-based rotation
    max_size_mb: int = 100

    # Retention settings
    backup_count: int = 30
    compress_backups: bool = True

    def validate(self) -> List[str]:
        """Validate the log rotation configuration."""
        errors = super().validate()

        # Validate rotation interval
        valid_intervals = ["hourly", "daily", "weekly", "monthly"]
        if self.rotation_interval not in valid_intervals:
            errors.append(f"rotation_interval must be one of {valid_intervals}")

        # Validate size
        if self.max_size_mb <= 0:
            errors.append("max_size_mb must be positive")

        # Validate backup count
        if self.backup_count < 0:
            errors.append("backup_count cannot be negative")

        return errors


@dataclass
class LogAuditConfig(BaseConfig):
    """Configuration for audit logging."""
    # Audit settings
    enable_audit_logging: bool = True
    audit_logger_name: str = "audit"

    # Events to audit
    log_login_events: bool = True
    log_configuration_changes: bool = True
    log_trading_events: bool = True
    log_data_access: bool = False

    # Security settings
    use_secure_audit_log: bool = False
    tamper_detection: bool = False

    def validate(self) -> List[str]:
        """Validate the audit configuration."""
        errors = super().validate()

        # Validate audit_logger_name
        if not self.audit_logger_name:
            errors.append("audit_logger_name must not be empty")

        return errors


@dataclass
class LoggingConfig(BaseConfig):
    """
    Main logging configuration that contains all logging-related settings.

    This class serves as a container for log handlers, loggers, and other
    logging-related configurations.
    """
    # Root logger configuration
    root_level: LogLevel = LogLevel.INFO

    # Handlers configuration
    handlers: Dict[str, LogHandlerConfig] = field(default_factory=dict)

    # Loggers configuration
    loggers: Dict[str, LoggerConfig] = field(default_factory=dict)

    # Performance monitoring
    performance_monitoring: PerformanceMonitoringConfig = field(default_factory=PerformanceMonitoringConfig)

    # Log rotation
    log_rotation: LogRotationConfig = field(default_factory=LogRotationConfig)

    # Audit logging
    audit: LogAuditConfig = field(default_factory=LogAuditConfig)

    # General settings
    log_exceptions: bool = True
    capture_warnings: bool = True
    log_to_stdout: bool = True

    # Default log path
    log_dir: str = "logs"

    def __post_init__(self):
        """Initialize the logging configuration."""
        super().__post_init__()

        # Convert root_level from string if needed
        if isinstance(self.root_level, str):
            try:
                self.root_level = LogLevel(self.root_level.upper())
            except ValueError:
                print(f"Warning: Invalid root_level '{self.root_level}', defaulting to INFO")
                self.root_level = LogLevel.INFO

        # Process handlers dictionary
        self._handler_configs = {}
        for handler_id, handler_dict in self.handlers.items():
            if isinstance(handler_dict, dict):
                handler_config = LogHandlerConfig(**handler_dict)
                handler_config.handler_id = handler_id
                self._handler_configs[handler_id] = handler_config
            elif isinstance(handler_dict, LogHandlerConfig):
                self._handler_configs[handler_id] = handler_dict

        # Process loggers dictionary
        self._logger_configs = {}
        for logger_name, logger_dict in self.loggers.items():
            if isinstance(logger_dict, dict):
                logger_config = LoggerConfig(**logger_dict)
                logger_config.logger_name = logger_name
                self._logger_configs[logger_name] = logger_config
            elif isinstance(logger_dict, LoggerConfig):
                self._logger_configs[logger_name] = logger_dict

        # Ensure performance_monitoring is a PerformanceMonitoringConfig object
        if isinstance(self.performance_monitoring, dict):
            self.performance_monitoring = PerformanceMonitoringConfig(**self.performance_monitoring)

        # Ensure log_rotation is a LogRotationConfig object
        if isinstance(self.log_rotation, dict):
            self.log_rotation = LogRotationConfig(**self.log_rotation)

        # Ensure audit is a LogAuditConfig object
        if isinstance(self.audit, dict):
            self.audit = LogAuditConfig(**self.audit)

        # Create default handlers if none defined
        if not self._handler_configs:
            # Console handler
            console_handler = LogHandlerConfig(
                handler_id="console",
                handler_type=LogDestination.CONSOLE,
                level=LogLevel.INFO,
                format_type=LogFormat.SIMPLE
            )
            self._handler_configs["console"] = console_handler
            self.handlers["console"] = console_handler.to_dict()

            # File handler
            file_handler = LogHandlerConfig(
                handler_id="file",
                handler_type=LogDestination.FILE,
                level=LogLevel.DEBUG,
                format_type=LogFormat.DETAILED,
                file_path=os.path.join(self.log_dir, "trading_system.log"),
                max_size_mb=10,
                backup_count=5
            )
            self._handler_configs["file"] = file_handler
            self.handlers["file"] = file_handler.to_dict()

            # Error file handler
            error_file_handler = LogHandlerConfig(
                handler_id="error_file",
                handler_type=LogDestination.FILE,
                level=LogLevel.ERROR,
                format_type=LogFormat.DETAILED,
                file_path=os.path.join(self.log_dir, "error.log"),
                max_size_mb=10,
                backup_count=5
            )
            self._handler_configs["error_file"] = error_file_handler
            self.handlers["error_file"] = error_file_handler.to_dict()

    def get_handler_config(self, handler_id: str) -> Optional[LogHandlerConfig]:
        """
        Get a handler configuration by ID.

        Args:
            handler_id: The ID of the handler

        Returns:
            The handler configuration if found, None otherwise
        """
        return self._handler_configs.get(handler_id)

    def get_logger_config(self, logger_name: str) -> Optional[LoggerConfig]:
        """
        Get a logger configuration by name.

        Args:
            logger_name: The name of the logger

        Returns:
            The logger configuration if found, None otherwise
        """
        return self._logger_configs.get(logger_name)

    def add_handler(self, handler_config: LogHandlerConfig) -> None:
        """
        Add a handler configuration.

        Args:
            handler_config: The handler configuration to add
        """
        handler_id = handler_config.handler_id
        if not handler_id:
            raise ValueError("Handler ID cannot be empty")

        self._handler_configs[handler_id] = handler_config
        self.handlers[handler_id] = handler_config.to_dict()

    def remove_handler(self, handler_id: str) -> bool:
        """
        Remove a handler configuration.

        Args:
            handler_id: The ID of the handler to remove

        Returns:
            True if the handler was removed, False otherwise
        """
        if handler_id in self._handler_configs:
            del self._handler_configs[handler_id]
            del self.handlers[handler_id]
            return True
        return False

    def add_logger(self, logger_config: LoggerConfig) -> None:
        """
        Add a logger configuration.

        Args:
            logger_config: The logger configuration to add
        """
        logger_name = logger_config.logger_name
        if not logger_name:
            raise ValueError("Logger name cannot be empty")

        self._logger_configs[logger_name] = logger_config
        self.loggers[logger_name] = logger_config.to_dict()

    def remove_logger(self, logger_name: str) -> bool:
        """
        Remove a logger configuration.

        Args:
            logger_name: The name of the logger to remove

        Returns:
            True if the logger was removed, False otherwise
        """
        if logger_name in self._logger_configs:
            del self._logger_configs[logger_name]
            del self.loggers[logger_name]
            return True
        return False

    def get_python_logging_config(self) -> Dict:
        """
        Generate a Python logging configuration dictionary.

        Returns:
            A dictionary suitable for use with logging.config.dictConfig()
        """
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {},
            "handlers": {},
            "loggers": {},
            "root": {
                "level": self.root_level.value,
                "handlers": []
            }
        }

        # Add formatters
        for handler_id, handler in self._handler_configs.items():
            formatter_name = f"{handler_id}_formatter"
            config["formatters"][formatter_name] = {
                "format": handler.custom_format
            }

        # Add handlers
        for handler_id, handler in self._handler_configs.items():
            handler_config = {
                "level": handler.level.value,
                "formatter": f"{handler_id}_formatter",
            }

            if handler.handler_type == LogDestination.CONSOLE:
                handler_config["class"] = "logging.StreamHandler"
                handler_config["stream"] = "ext://sys.stdout"
            elif handler.handler_type == LogDestination.FILE:
                handler_config["class"] = "logging.handlers.RotatingFileHandler"
                handler_config["filename"] = handler.file_path
                handler_config["maxBytes"] = handler.max_size_mb * 1024 * 1024
                handler_config["backupCount"] = handler.backup_count
            elif handler.handler_type == LogDestination.SYSLOG:
                handler_config["class"] = "logging.handlers.SysLogHandler"
                handler_config["address"] = "/dev/log"
            elif handler.handler_type == LogDestination.REMOTE:
                handler_config["class"] = "logging.handlers.SocketHandler"
                handler_config["host"] = handler.remote_host
                handler_config["port"] = handler.remote_port
            elif handler.handler_type == LogDestination.DATABASE:
                # Database handler would typically be a custom handler
                handler_config["class"] = "utils.logger.DatabaseLogHandler"
                handler_config["database_instance_id"] = handler.database_instance_id
                handler_config["database_table"] = handler.database_table

            # Add filters if specified
            if handler.include_modules or handler.exclude_modules:
                handler_config["filters"] = [f"{handler_id}_filter"]

            config["handlers"][handler_id] = handler_config

            # Add this handler to the root logger
            config["root"]["handlers"].append(handler_id)

        # Add filters
        config["filters"] = {}
        for handler_id, handler in self._handler_configs.items():
            if handler.include_modules or handler.exclude_modules:
                filter_name = f"{handler_id}_filter"
                filter_config = {
                    "()": "utils.logger.ModuleFilter",
                    "include_modules": handler.include_modules,
                    "exclude_modules": handler.exclude_modules
                }
                config["filters"][filter_name] = filter_config

        # Add loggers
        for logger_name, logger in self._logger_configs.items():
            logger_config = {
                "level": logger.level.value,
                "propagate": logger.propagate,
                "handlers": logger.handlers
            }
            config["loggers"][logger_name] = logger_config

        return config

    def validate(self) -> List[str]:
        """Validate the logging configuration."""
        errors = super().validate()

        # Validate log directory
        if not self.log_dir:
            errors.append("log_dir must not be empty")

        # Create log directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            try:
                os.makedirs(self.log_dir, exist_ok=True)
            except Exception as e:
                errors.append(f"Failed to create log directory: {e}")

        # Validate handlers
        for handler_id, config in self._handler_configs.items():
            handler_errors = config.validate()
            for error in handler_errors:
                errors.append(f"In handler '{handler_id}': {error}")

        # Validate loggers
        for logger_name, config in self._logger_configs.items():
            logger_errors = config.validate()
            for error in logger_errors:
                errors.append(f"In logger '{logger_name}': {error}")

            # Validate that all referenced handlers exist
            for handler_id in config.handlers:
                if handler_id not in self._handler_configs:
                    errors.append(f"In logger '{logger_name}': Referenced handler '{handler_id}' does not exist")

        # Validate performance monitoring
        monitoring_errors = self.performance_monitoring.validate()
        for error in monitoring_errors:
            errors.append(f"In performance_monitoring: {error}")

        # Validate log rotation
        rotation_errors = self.log_rotation.validate()
        for error in rotation_errors:
            errors.append(f"In log_rotation: {error}")

        # Validate audit
        audit_errors = self.audit.validate()
        for error in audit_errors:
            errors.append(f"In audit: {error}")

        return errors


def get_logging_config(config_path: Optional[Union[str, Path]] = None) -> LoggingConfig:
    """
    Get the logging configuration.

    Args:
        config_path: Optional path to a configuration file. If not provided,
                    the default path from the ConfigManager will be used.

    Returns:
        The logging configuration.
    """
    if config_path is None:
        config_path = ConfigManager.get_config_path("logging")

    return ConfigManager.load_config(
        LoggingConfig,
        config_path=config_path,
        env_prefix="TRADING_LOG",
        reload=False
    )