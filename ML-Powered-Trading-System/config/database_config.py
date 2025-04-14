"""
Database configuration module that defines settings for database connections
and data persistence.

This module provides configuration for various database backends, connection
pooling, and data storage policies.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union
from pathlib import Path
import os

from .base_config import BaseConfig, ConfigManager


class DatabaseType(Enum):
    """Enumeration of supported database types."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    REDIS = "redis"
    INFLUXDB = "influxdb"
    CASSANDRA = "cassandra"


class DatabaseRole(Enum):
    """Enumeration of database roles."""
    PRIMARY = "primary"
    REPLICA = "replica"
    ARCHIVE = "archive"
    ANALYTICS = "analytics"


@dataclass
class ConnectionPoolConfig(BaseConfig):
    """Configuration for database connection pooling."""
    # Pool sizing
    min_connections: int = 5
    max_connections: int = 20
    max_overflow: int = 10

    # Connection lifecycle
    connection_timeout_sec: int = 30
    idle_timeout_sec: int = 600
    max_age_sec: int = 3600

    # Retry settings
    retry_attempts: int = 3
    retry_delay_ms: int = 500

    def validate(self) -> List[str]:
        """Validate the connection pool configuration."""
        errors = super().validate()

        # Validate pool sizing
        if self.min_connections < 1:
            errors.append("min_connections must be at least 1")

        if self.max_connections < self.min_connections:
            errors.append("max_connections must be greater than or equal to min_connections")

        if self.max_overflow < 0:
            errors.append("max_overflow cannot be negative")

        # Validate connection lifecycle
        if self.connection_timeout_sec <= 0:
            errors.append("connection_timeout_sec must be positive")

        if self.idle_timeout_sec <= 0:
            errors.append("idle_timeout_sec must be positive")

        if self.max_age_sec <= 0:
            errors.append("max_age_sec must be positive")

        # Validate retry settings
        if self.retry_attempts < 0:
            errors.append("retry_attempts cannot be negative")

        if self.retry_delay_ms <= 0:
            errors.append("retry_delay_ms must be positive")

        return errors


@dataclass
class DatabaseInstanceConfig(BaseConfig):
    """Configuration for a single database instance."""
    # Instance identification
    instance_id: str = ""
    db_type: DatabaseType = DatabaseType.SQLITE
    role: DatabaseRole = DatabaseRole.PRIMARY

    # Connection details
    host: str = "localhost"
    port: int = 0  # 0 means use default port for the database type
    database_name: str = ""
    username: str = ""
    password: str = ""

    # Additional connection parameters
    connection_params: Dict[str, str] = field(default_factory=dict)

    # SSL/TLS settings
    use_ssl: bool = False
    ssl_ca_cert_path: str = ""
    ssl_client_cert_path: str = ""
    ssl_client_key_path: str = ""

    # Connection pooling
    connection_pool: ConnectionPoolConfig = field(default_factory=ConnectionPoolConfig)

    def __post_init__(self):
        """Initialize the database instance configuration."""
        super().__post_init__()

        # Convert db_type from string if needed
        if isinstance(self.db_type, str):
            try:
                self.db_type = DatabaseType(self.db_type.lower())
            except ValueError:
                print(f"Warning: Invalid db_type '{self.db_type}', defaulting to SQLITE")
                self.db_type = DatabaseType.SQLITE

        # Convert role from string if needed
        if isinstance(self.role, str):
            try:
                self.role = DatabaseRole(self.role.lower())
            except ValueError:
                print(f"Warning: Invalid role '{self.role}', defaulting to PRIMARY")
                self.role = DatabaseRole.PRIMARY

        # Set default port based on database type if not specified
        if self.port == 0:
            if self.db_type == DatabaseType.POSTGRESQL:
                self.port = 5432
            elif self.db_type == DatabaseType.MYSQL:
                self.port = 3306
            elif self.db_type == DatabaseType.MONGODB:
                self.port = 27017
            elif self.db_type == DatabaseType.REDIS:
                self.port = 6379
            elif self.db_type == DatabaseType.INFLUXDB:
                self.port = 8086
            elif self.db_type == DatabaseType.CASSANDRA:
                self.port = 9042

        # For SQLite, host and port are not applicable
        if self.db_type == DatabaseType.SQLITE:
            self.host = ""
            self.port = 0

        # Ensure connection_pool is a ConnectionPoolConfig object
        if isinstance(self.connection_pool, dict):
            self.connection_pool = ConnectionPoolConfig(**self.connection_pool)

    def get_connection_string(self) -> str:
        """
        Generate a connection string for this database instance.

        Returns:
            A connection string suitable for use with most database libraries.
        """
        if self.db_type == DatabaseType.SQLITE:
            return f"sqlite:///{self.database_name}"

        if self.db_type == DatabaseType.POSTGRESQL:
            auth = f"{self.username}:{self.password}@" if self.username else ""
            return f"postgresql://{auth}{self.host}:{self.port}/{self.database_name}"

        if self.db_type == DatabaseType.MYSQL:
            auth = f"{self.username}:{self.password}@" if self.username else ""
            return f"mysql+pymysql://{auth}{self.host}:{self.port}/{self.database_name}"

        if self.db_type == DatabaseType.MONGODB:
            auth = f"{self.username}:{self.password}@" if self.username else ""
            return f"mongodb://{auth}{self.host}:{self.port}/{self.database_name}"

        if self.db_type == DatabaseType.REDIS:
            auth = f":{self.password}@" if self.password else ""
            return f"redis://{auth}{self.host}:{self.port}/0"

        if self.db_type == DatabaseType.INFLUXDB:
            # InfluxDB typically uses HTTP API, so return base URL
            return f"http://{self.host}:{self.port}"

        if self.db_type == DatabaseType.CASSANDRA:
            # Cassandra typically uses a list of contact points
            return f"{self.host}:{self.port}"

        # Default fallback
        return f"{self.db_type.value}://{self.host}:{self.port}/{self.database_name}"

    def validate(self) -> List[str]:
        """Validate the database instance configuration."""
        errors = super().validate()

        # Validate instance_id
        if not self.instance_id:
            errors.append("instance_id must not be empty")

        # Validate database_name
        if not self.database_name:
            errors.append("database_name must not be empty")

        # For non-SQLite databases, validate host and port
        if self.db_type != DatabaseType.SQLITE:
            if not self.host:
                errors.append("host must not be empty for non-SQLite databases")

            if self.port <= 0:
                errors.append("port must be positive for non-SQLite databases")

        # For databases that require authentication, validate credentials
        auth_required_dbs = [
            DatabaseType.POSTGRESQL, DatabaseType.MYSQL,
            DatabaseType.MONGODB, DatabaseType.CASSANDRA
        ]

        if self.db_type in auth_required_dbs and not (self.username and self.password):
            errors.append(f"username and password are required for {self.db_type.value}")

        # Validate SSL settings
        if self.use_ssl:
            # If SSL is enabled, CA cert is generally required
            if not self.ssl_ca_cert_path:
                errors.append("ssl_ca_cert_path must be provided when use_ssl is enabled")

            # Validate that cert files exist if specified
            if self.ssl_ca_cert_path and not os.path.exists(self.ssl_ca_cert_path):
                errors.append(f"SSL CA certificate file not found: {self.ssl_ca_cert_path}")

            if self.ssl_client_cert_path and not os.path.exists(self.ssl_client_cert_path):
                errors.append(f"SSL client certificate file not found: {self.ssl_client_cert_path}")

            if self.ssl_client_key_path and not os.path.exists(self.ssl_client_key_path):
                errors.append(f"SSL client key file not found: {self.ssl_client_key_path}")

        # Validate connection pool
        if isinstance(self.connection_pool, ConnectionPoolConfig):
            pool_errors = self.connection_pool.validate()
            for error in pool_errors:
                errors.append(f"In connection_pool: {error}")

        return errors


class DataRetentionPolicy(Enum):
    """Enumeration of data retention policies."""
    INDEFINITE = "indefinite"  # Keep data indefinitely
    TIME_BASED = "time_based"  # Keep data for a specified time period
    SPACE_BASED = "space_based"  # Keep data until space limit is reached
    COUNT_BASED = "count_based"  # Keep a specified number of records
    HYBRID = "hybrid"  # Combination of multiple policies


@dataclass
class DataRetentionConfig(BaseConfig):
    """Configuration for data retention."""
    # Retention policy
    policy: DataRetentionPolicy = DataRetentionPolicy.INDEFINITE

    # Time-based retention (in days)
    market_data_retention_days: int = 365
    order_data_retention_days: int = 730
    trade_data_retention_days: int = 1825  # 5 years
    log_data_retention_days: int = 90

    # Space-based retention (in GB)
    max_database_size_gb: float = 100.0
    max_table_size_gb: float = 10.0

    # Count-based retention
    max_market_data_points: int = 1000000000  # 1 billion
    max_order_count: int = 1000000  # 1 million
    max_trade_count: int = 500000  # 500,000

    # Archiving settings
    enable_archiving: bool = True
    archive_before_delete: bool = True
    archive_location: str = "archive"

    def __post_init__(self):
        """Initialize the data retention configuration."""
        super().__post_init__()

        # Convert policy from string if needed
        if isinstance(self.policy, str):
            try:
                self.policy = DataRetentionPolicy(self.policy.lower())
            except ValueError:
                print(f"Warning: Invalid policy '{self.policy}', defaulting to INDEFINITE")
                self.policy = DataRetentionPolicy.INDEFINITE

    def validate(self) -> List[str]:
        """Validate the data retention configuration."""
        errors = super().validate()

        # Validate time-based retention
        if self.policy in [DataRetentionPolicy.TIME_BASED, DataRetentionPolicy.HYBRID]:
            if self.market_data_retention_days <= 0:
                errors.append("market_data_retention_days must be positive")

            if self.order_data_retention_days <= 0:
                errors.append("order_data_retention_days must be positive")

            if self.trade_data_retention_days <= 0:
                errors.append("trade_data_retention_days must be positive")

            if self.log_data_retention_days <= 0:
                errors.append("log_data_retention_days must be positive")

        # Validate space-based retention
        if self.policy in [DataRetentionPolicy.SPACE_BASED, DataRetentionPolicy.HYBRID]:
            if self.max_database_size_gb <= 0:
                errors.append("max_database_size_gb must be positive")

            if self.max_table_size_gb <= 0:
                errors.append("max_table_size_gb must be positive")

            # Validate count-based retention
            if self.policy in [DataRetentionPolicy.COUNT_BASED, DataRetentionPolicy.HYBRID]:
                if self.max_market_data_points <= 0:
                    errors.append("max_market_data_points must be positive")

                if self.max_order_count <= 0:
                    errors.append("max_order_count must be positive")

                if self.max_trade_count <= 0:
                    errors.append("max_trade_count must be positive")

            # Validate archiving settings
            if self.archive_before_delete and not self.enable_archiving:
                errors.append("archive_before_delete cannot be True when enable_archiving is False")

            if self.enable_archiving and not self.archive_location:
                errors.append("archive_location must not be empty when enable_archiving is True")

            return errors


# Add these missing classes and the singleton function to your database_config.py file

@dataclass
class BackupConfig(BaseConfig):
    """Configuration for database backup operations."""
    # Backup scheduling
    enabled: bool = True
    schedule_cron: str = "0 2 * * *"  # Daily at 2 AM

    # Backup storage
    backup_path: str = "data/backups/db"
    max_backups: int = 10
    compress_backups: bool = True
    compression_level: int = 6  # 0-9, where 9 is highest compression

    # Backup strategy
    full_backup_interval_days: int = 7
    incremental_backups: bool = True

    # Verification
    verify_after_backup: bool = True
    run_consistency_check: bool = True

    # Encryption
    encrypt_backups: bool = False
    encryption_key_path: str = ""

    def validate(self) -> List[str]:
        """Validate the backup configuration."""
        errors = super().validate()

        # Validate backup path
        if not self.backup_path:
            errors.append("backup_path must not be empty")

        # Validate max backups
        if self.max_backups <= 0:
            errors.append("max_backups must be positive")

        # Validate compression level
        if not (0 <= self.compression_level <= 9):
            errors.append("compression_level must be between 0 and 9")

        # Validate full backup interval
        if self.full_backup_interval_days <= 0:
            errors.append("full_backup_interval_days must be positive")

        # Validate encryption settings
        if self.encrypt_backups and not self.encryption_key_path:
            errors.append("encryption_key_path must be provided when encrypt_backups is enabled")

        return errors


@dataclass
class MigrationConfig(BaseConfig):
    """Configuration for database schema migrations."""
    # Migration control
    enabled: bool = True
    auto_migrate: bool = False
    migration_path: str = "migrations"

    # Version tracking
    track_in_database: bool = True
    version_table: str = "schema_versions"

    # Safety controls
    require_migration_plan: bool = True
    allow_destructive_migrations: bool = False
    backup_before_migration: bool = True

    # Validation
    validate_after_migration: bool = True
    run_tests_after_migration: bool = False
    test_timeout_seconds: int = 60

    def validate(self) -> List[str]:
        """Validate the migration configuration."""
        errors = super().validate()

        # Validate migration path
        if not self.migration_path:
            errors.append("migration_path must not be empty")

        # Validate version table
        if self.track_in_database and not self.version_table:
            errors.append("version_table must not be empty when track_in_database is enabled")

        # Validate test timeout
        if self.run_tests_after_migration and self.test_timeout_seconds <= 0:
            errors.append("test_timeout_seconds must be positive when run_tests_after_migration is enabled")

        return errors


@dataclass
class DatabaseConfig(BaseConfig):
    """Main configuration for database settings."""
    # Database instances
    instances: List[DatabaseInstanceConfig] = field(default_factory=list)

    # Default instance to use
    default_instance_id: str = ""

    # Data retention settings
    retention: DataRetentionConfig = field(default_factory=DataRetentionConfig)

    # Backup settings
    backup: BackupConfig = field(default_factory=BackupConfig)

    # Migration settings
    migration: MigrationConfig = field(default_factory=MigrationConfig)

    def __post_init__(self):
        """Initialize the database configuration."""
        super().__post_init__()

        # Ensure instances are DatabaseInstanceConfig objects
        converted_instances = []
        for instance in self.instances:
            if isinstance(instance, dict):
                converted_instances.append(DatabaseInstanceConfig(**instance))
            else:
                converted_instances.append(instance)
        self.instances = converted_instances

        # Ensure retention is a DataRetentionConfig object
        if isinstance(self.retention, dict):
            self.retention = DataRetentionConfig(**self.retention)

        # Ensure backup is a BackupConfig object
        if isinstance(self.backup, dict):
            self.backup = BackupConfig(**self.backup)

        # Ensure migration is a MigrationConfig object
        if isinstance(self.migration, dict):
            self.migration = MigrationConfig(**self.migration)

    def get_instance(self, instance_id: Optional[str] = None) -> Optional[DatabaseInstanceConfig]:
        """
        Get a database instance configuration by ID.

        Args:
            instance_id: ID of the instance to retrieve, or None for default

        Returns:
            The database instance configuration, or None if not found
        """
        # Use default instance if not specified
        if instance_id is None:
            instance_id = self.default_instance_id

        # Find the instance with matching ID
        for instance in self.instances:
            if instance.instance_id == instance_id:
                return instance

        return None

    def validate(self) -> List[str]:
        """Validate the database configuration."""
        errors = super().validate()

        # Validate instances
        if not self.instances:
            errors.append("At least one database instance must be configured")

        # Validate each instance
        for i, instance in enumerate(self.instances):
            if isinstance(instance, DatabaseInstanceConfig):
                instance_errors = instance.validate()
                for error in instance_errors:
                    errors.append(f"In instance {i} ({instance.instance_id}): {error}")

        # Validate default instance
        if self.default_instance_id and not self.get_instance(self.default_instance_id):
            errors.append(f"Default instance '{self.default_instance_id}' not found in configured instances")

        # Validate retention
        if isinstance(self.retention, DataRetentionConfig):
            retention_errors = self.retention.validate()
            for error in retention_errors:
                errors.append(f"In retention: {error}")

        # Validate backup
        if isinstance(self.backup, BackupConfig):
            backup_errors = self.backup.validate()
            for error in backup_errors:
                errors.append(f"In backup: {error}")

        # Validate migration
        if isinstance(self.migration, MigrationConfig):
            migration_errors = self.migration.validate()
            for error in migration_errors:
                errors.append(f"In migration: {error}")

        return errors


# Singleton instance for database configuration
_database_config_instance = None


def get_database_config() -> DatabaseConfig:
    """
    Get the singleton database configuration instance.

    Returns:
        The database configuration instance
    """
    global _database_config_instance

    if _database_config_instance is None:
        # Load configuration from file or default
        config_manager = ConfigManager()
        config_data = config_manager.load_config("database", {})

        # Create instance
        _database_config_instance = DatabaseConfig(**config_data)

    return _database_config_instance