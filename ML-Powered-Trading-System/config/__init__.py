"""
Configuration module initialization.

This module provides access to all configuration components of the trading system.
"""
from pathlib import Path
import os

from .base_config import BaseConfig, ConfigManager, ConfigSource
from .system_config import SystemConfig, SystemPaths, Environment, LogLevel, get_system_config
from .strategy_config import (
    StrategyConfig, StrategyType, TimeFrame, BaseStrategyConfig,
    BreakoutStrategyConfig, MomentumStrategyConfig, MeanReversionStrategyConfig,
    TrendFollowingStrategyConfig, VolatilityStrategyConfig, get_strategy_config
)
from .database_config import (
    DatabaseConfig, DatabaseType, DatabaseRole, ConnectionPoolConfig,
    DatabaseInstanceConfig, DataRetentionPolicy, DataRetentionConfig,
    BackupConfig as DBBackupConfig, MigrationConfig, get_database_config
)
from .logging_config import (
    LoggingConfig, LogLevel, LogFormat, LogDestination, LogHandlerConfig,
    LoggerConfig, PerformanceMonitoringConfig, LogRotationConfig,
    LogAuditConfig, get_logging_config
)
from .exchange_config import (
    ExchangeConfig, ExchangeType, AssetClass, ExchangeProvider, RateLimitConfig,
    APICredentials, ExchangeInstanceConfig, WebhookConfig, OrderRoutingConfig,
    MarketDataConfig, get_exchange_config
)
from .trading_mode_config import (
    TradingModeConfig, TradingMode, TransitionStage, PaperTradingConfig,
    LiveTradingConfig, ShadowTradingConfig, PilotTradingConfig,
    TransitionValidationConfig, AutomatedTransitionConfig, get_trading_mode_config
)
from .disaster_recovery_config import (
    DisasterRecoveryConfig, BackupStrategy, BackupStorageType, RecoveryPriority,
    RecoveryMode, BackupConfig, RecoveryConfig, HighAvailabilityConfig,
    StateJournalConfig, CircuitBreakerConfig, get_disaster_recovery_config
)


def initialize_config(config_dir: str = "config", env_prefix: str = "TRADING_SYSTEM") -> None:
    """
    Initialize the configuration system by registering config paths and loading defaults.

    Args:
        config_dir: Directory containing configuration files
        env_prefix: Prefix for environment variables
    """
    # Register configuration paths
    ConfigManager.register_config_path("system", os.path.join(config_dir, "system_config.yaml"))
    ConfigManager.register_config_path("strategy", os.path.join(config_dir, "strategy_config.yaml"))
    ConfigManager.register_config_path("database", os.path.join(config_dir, "database_config.yaml"))
    ConfigManager.register_config_path("logging", os.path.join(config_dir, "logging_config.yaml"))
    ConfigManager.register_config_path("exchange", os.path.join(config_dir, "exchange_config.yaml"))
    ConfigManager.register_config_path("trading_mode", os.path.join(config_dir, "trading_mode_config.yaml"))
    ConfigManager.register_config_path("disaster_recovery", os.path.join(config_dir, "disaster_recovery_config.yaml"))

    # Load system configuration first
    system_config = get_system_config()

    # Create configuration directory if it doesn't exist
    os.makedirs(config_dir, exist_ok=True)


def create_default_configs(output_dir: str = "config", format: str = "yaml") -> None:
    """
    Create default configuration files in the specified directory.

    Args:
        output_dir: Directory to create configuration files in
        format: Format of configuration files ("yaml" or "json")
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create system configuration
    system_config = SystemConfig()
    if format == "yaml":
        system_config.to_yaml(os.path.join(output_dir, "system_config.yaml"))
    else:
        system_config.to_json(os.path.join(output_dir, "system_config.json"))

    # Create strategy configuration
    strategy_config = StrategyConfig()
    if format == "yaml":
        strategy_config.to_yaml(os.path.join(output_dir, "strategy_config.yaml"))
    else:
        strategy_config.to_json(os.path.join(output_dir, "strategy_config.json"))

    # Create database configuration
    database_config = DatabaseConfig()
    if format == "yaml":
        database_config.to_yaml(os.path.join(output_dir, "database_config.yaml"))
    else:
        database_config.to_json(os.path.join(output_dir, "database_config.json"))

    # Create logging configuration
    logging_config = LoggingConfig()
    if format == "yaml":
        logging_config.to_yaml(os.path.join(output_dir, "logging_config.yaml"))
    else:
        logging_config.to_json(os.path.join(output_dir, "logging_config.json"))

    # Create exchange configuration
    exchange_config = ExchangeConfig()
    if format == "yaml":
        exchange_config.to_yaml(os.path.join(output_dir, "exchange_config.yaml"))
    else:
        exchange_config.to_json(os.path.join(output_dir, "exchange_config.json"))

    # Create trading mode configuration
    trading_mode_config = TradingModeConfig()
    if format == "yaml":
        trading_mode_config.to_yaml(os.path.join(output_dir, "trading_mode_config.yaml"))
    else:
        trading_mode_config.to_json(os.path.join(output_dir, "trading_mode_config.json"))

    # Create disaster recovery configuration
    disaster_recovery_config = DisasterRecoveryConfig()
    if format == "yaml":
        disaster_recovery_config.to_yaml(os.path.join(output_dir, "disaster_recovery_config.yaml"))
    else:
        disaster_recovery_config.to_json(os.path.join(output_dir, "disaster_recovery_config.json"))


# Expose key functionality at the module level
__all__ = [
    # Base config
    'BaseConfig', 'ConfigManager', 'ConfigSource',
    # System config
    'SystemConfig', 'SystemPaths', 'Environment', 'LogLevel', 'get_system_config',
    # Strategy config
    'StrategyConfig', 'StrategyType', 'TimeFrame', 'BaseStrategyConfig',
    'BreakoutStrategyConfig', 'MomentumStrategyConfig', 'MeanReversionStrategyConfig',
    'TrendFollowingStrategyConfig', 'VolatilityStrategyConfig', 'get_strategy_config',
    # Database config
    'DatabaseConfig', 'DatabaseType', 'DatabaseRole', 'ConnectionPoolConfig',
    'DatabaseInstanceConfig', 'DataRetentionPolicy', 'DataRetentionConfig',
    'DBBackupConfig', 'MigrationConfig', 'get_database_config',
    # Logging config
    'LoggingConfig', 'LogLevel', 'LogFormat', 'LogDestination', 'LogHandlerConfig',
    'LoggerConfig', 'PerformanceMonitoringConfig', 'LogRotationConfig',
    'LogAuditConfig', 'get_logging_config',
    # Exchange config
    'ExchangeConfig', 'ExchangeType', 'AssetClass', 'ExchangeProvider', 'RateLimitConfig',
    'APICredentials', 'ExchangeInstanceConfig', 'WebhookConfig', 'OrderRoutingConfig',
    'MarketDataConfig', 'get_exchange_config',
    # Trading mode config
    'TradingModeConfig', 'TradingMode', 'TransitionStage', 'PaperTradingConfig',
    'LiveTradingConfig', 'ShadowTradingConfig', 'PilotTradingConfig',
    'TransitionValidationConfig', 'AutomatedTransitionConfig', 'get_trading_mode_config',
    # Disaster recovery config
    'DisasterRecoveryConfig', 'BackupStrategy', 'BackupStorageType', 'RecoveryPriority',
    'RecoveryMode', 'BackupConfig', 'RecoveryConfig', 'HighAvailabilityConfig',
    'StateJournalConfig', 'CircuitBreakerConfig', 'get_disaster_recovery_config',
    # Utility functions
    'initialize_config', 'create_default_configs'
]