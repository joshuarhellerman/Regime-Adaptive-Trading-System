"""
Disaster recovery configuration module that defines settings for backup and recovery.

This module provides configuration for system backups, recovery procedures,
and high availability settings.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Union
from pathlib import Path
import os

from .base_config import BaseConfig, ConfigManager


class BackupStrategy(Enum):
    """Enumeration of backup strategies."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    CONTINUOUS = "continuous"


class BackupStorageType(Enum):
    """Enumeration of backup storage types."""
    LOCAL = "local"
    REMOTE = "remote"
    CLOUD = "cloud"
    HYBRID = "hybrid"


class RecoveryPriority(Enum):
    """Enumeration of recovery priorities."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecoveryMode(Enum):
    """Enumeration of recovery modes."""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    HYBRID = "hybrid"


@dataclass
class BackupConfig(BaseConfig):
    """Configuration for system backups."""
    # Backup settings
    enable_backups: bool = True
    backup_strategy: BackupStrategy = BackupStrategy.INCREMENTAL

    # Schedule settings
    backup_schedule: str = "0 2 * * *"  # Cron expression: 2 AM daily
    full_backup_schedule: str = "0 2 * * 0"  # Cron expression: 2 AM on Sundays

    # Storage settings
    storage_type: BackupStorageType = BackupStorageType.LOCAL
    local_backup_dir: str = "backups"
    remote_backup_url: str = ""
    cloud_provider: str = ""  # "aws", "gcp", "azure", etc.
    cloud_bucket: str = ""

    # Retention settings
    retention_full_backups: int = 4  # Keep 4 full backups
    retention_incremental_backups: int = 14  # Keep 14 incremental backups
    retention_days: int = 30  # Keep backups for 30 days

    # Backup content
    backup_database: bool = True
    backup_configuration: bool = True
    backup_logs: bool = True
    backup_models: bool = True

    # Security settings
    encrypt_backups: bool = True
    encryption_key_path: str = ""

    # Performance settings
    compression_level: int = 6  # 0-9, where 9 is highest compression
    backup_threads: int = 2

    def __post_init__(self):
        """Initialize the backup configuration."""
        super().__post_init__()

        # Convert backup_strategy from string if needed
        if isinstance(self.backup_strategy, str):
            try:
                self.backup_strategy = BackupStrategy(self.backup_strategy.lower())
            except ValueError:
                print(f"Warning: Invalid backup_strategy '{self.backup_strategy}', defaulting to INCREMENTAL")
                self.backup_strategy = BackupStrategy.INCREMENTAL

        # Convert storage_type from string if needed
        if isinstance(self.storage_type, str):
            try:
                self.storage_type = BackupStorageType(self.storage_type.lower())
            except ValueError:
                print(f"Warning: Invalid storage_type '{self.storage_type}', defaulting to LOCAL")
                self.storage_type = BackupStorageType.LOCAL

    def validate(self) -> List[str]:
        """Validate the backup configuration."""
        errors = super().validate()

        # Only validate if backups are enabled
        if not self.enable_backups:
            return errors

        # Validate schedule settings
        import re
        cron_pattern = re.compile(
            r'^(\*|(\d+,)*\d+|(\d+(\-\d+)?)(/\d+)?) (\*|(\d+,)*\d+|(\d+(\-\d+)?)(/\d+)?) (\*|(\d+,)*\d+|(\d+(\-\d+)?)(/\d+)?) (\*|(\d+,)*\d+|(\d+(\-\d+)?)(/\d+)?) (\*|(\d+,)*\d+|(\d+(\-\d+)?)(/\d+)?)$')
        return errors


@dataclass
class RecoveryConfig(BaseConfig):
    """Configuration for system recovery procedures."""
    # Recovery settings
    enable_recovery: bool = True
    recovery_mode: RecoveryMode = RecoveryMode.MANUAL

    # Recovery priorities
    component_priorities: Dict[str, RecoveryPriority] = field(default_factory=dict)

    # Recovery timeouts
    recovery_timeout_seconds: int = 300
    component_recovery_timeout_seconds: int = 60

    # Validation settings
    verify_after_recovery: bool = True
    perform_reconciliation: bool = True

    # Notification settings
    notify_on_recovery_start: bool = True
    notify_on_recovery_complete: bool = True
    notify_on_recovery_failure: bool = True
    notification_email: List[str] = field(default_factory=list)

    # Resource limits
    max_memory_percent: int = 80
    max_cpu_percent: int = 90

    def __post_init__(self):
        """Initialize the recovery configuration."""
        super().__post_init__()

        # Convert recovery_mode from string if needed
        if isinstance(self.recovery_mode, str):
            try:
                self.recovery_mode = RecoveryMode(self.recovery_mode.lower())
            except ValueError:
                print(f"Warning: Invalid recovery_mode '{self.recovery_mode}', defaulting to MANUAL")
                self.recovery_mode = RecoveryMode.MANUAL

        # Convert component priorities from strings if needed
        for comp, priority in list(self.component_priorities.items()):
            if isinstance(priority, str):
                try:
                    self.component_priorities[comp] = RecoveryPriority(priority.lower())
                except ValueError:
                    print(f"Warning: Invalid priority '{priority}' for component '{comp}', defaulting to MEDIUM")
                    self.component_priorities[comp] = RecoveryPriority.MEDIUM

    def validate(self) -> List[str]:
        """Validate the recovery configuration."""
        errors = super().validate()

        # Only validate if recovery is enabled
        if not self.enable_recovery:
            return errors

        # Validate timeout settings
        if self.recovery_timeout_seconds <= 0:
            errors.append("recovery_timeout_seconds must be positive")

        if self.component_recovery_timeout_seconds <= 0:
            errors.append("component_recovery_timeout_seconds must be positive")

        # Validate resource limits
        if not (0 < self.max_memory_percent <= 100):
            errors.append("max_memory_percent must be between 1 and 100")

        if not (0 < self.max_cpu_percent <= 100):
            errors.append("max_cpu_percent must be between 1 and 100")

        return errors


@dataclass
class StateJournalConfig(BaseConfig):
    """Configuration for state journaling and transaction management."""
    # Journal settings
    enable_journaling: bool = True
    journal_directory: str = "data/journal"

    # Storage settings
    max_journal_size_mb: int = 1024  # 1 GB
    journal_rotation_count: int = 5

    # Performance settings
    sync_on_commit: bool = True
    batch_commits: bool = False
    max_batch_size: int = 100

    # Recovery settings
    journal_replay_batch_size: int = 1000
    max_replay_time_seconds: int = 300  # 5 minutes

    # Integrity settings
    checksum_entries: bool = True
    validate_on_recovery: bool = True

    def validate(self) -> List[str]:
        """Validate the state journal configuration."""
        errors = super().validate()

        # Only validate if journaling is enabled
        if not self.enable_journaling:
            return errors

        # Validate size settings
        if self.max_journal_size_mb <= 0:
            errors.append("max_journal_size_mb must be positive")

        if self.journal_rotation_count <= 0:
            errors.append("journal_rotation_count must be positive")

        # Validate batch settings
        if self.batch_commits and self.max_batch_size <= 0:
            errors.append("max_batch_size must be positive when batch_commits is enabled")

        # Validate replay settings
        if self.journal_replay_batch_size <= 0:
            errors.append("journal_replay_batch_size must be positive")

        if self.max_replay_time_seconds <= 0:
            errors.append("max_replay_time_seconds must be positive")

        return errors


@dataclass
class CircuitBreakerConfig(BaseConfig):
    """Configuration for circuit breaker settings."""
    # Enable/disable circuit breakers
    enable_circuit_breakers: bool = True

    # Circuit breaker types
    position_circuit_breaker: bool = True
    risk_circuit_breaker: bool = True
    execution_circuit_breaker: bool = True
    system_circuit_breaker: bool = True

    # Position circuit breaker settings
    max_position_deviation_percent: float = 5.0
    position_check_interval_seconds: int = 60

    # Risk circuit breaker settings
    max_risk_exposure_percent: float = 20.0
    risk_check_interval_seconds: int = 30

    # Execution circuit breaker settings
    max_consecutive_errors: int = 5
    error_window_seconds: int = 300

    # System circuit breaker settings
    max_cpu_usage_percent: float = 90.0
    max_memory_usage_percent: float = 85.0
    system_check_interval_seconds: int = 60

    # Recovery settings
    auto_reset_after_seconds: Dict[str, int] = field(
        default_factory=lambda: {
            "position": 300,
            "risk": 600,
            "execution": 180,
            "system": 900
        }
    )

    # Notification settings
    notify_on_breaker_trip: bool = True
    notification_email: List[str] = field(default_factory=list)

    def validate(self) -> List[str]:
        """Validate the circuit breaker configuration."""
        errors = super().validate()

        # Only validate if circuit breakers are enabled
        if not self.enable_circuit_breakers:
            return errors

        # Validate position breaker settings
        if self.position_circuit_breaker:
            if self.max_position_deviation_percent <= 0:
                errors.append("max_position_deviation_percent must be positive")

            if self.position_check_interval_seconds <= 0:
                errors.append("position_check_interval_seconds must be positive")

        # Validate risk breaker settings
        if self.risk_circuit_breaker:
            if self.max_risk_exposure_percent <= 0:
                errors.append("max_risk_exposure_percent must be positive")

            if self.risk_check_interval_seconds <= 0:
                errors.append("risk_check_interval_seconds must be positive")

        # Validate execution breaker settings
        if self.execution_circuit_breaker:
            if self.max_consecutive_errors <= 0:
                errors.append("max_consecutive_errors must be positive")

            if self.error_window_seconds <= 0:
                errors.append("error_window_seconds must be positive")

        # Validate system breaker settings
        if self.system_circuit_breaker:
            if not (0 < self.max_cpu_usage_percent <= 100):
                errors.append("max_cpu_usage_percent must be between 1 and 100")

            if not (0 < self.max_memory_usage_percent <= 100):
                errors.append("max_memory_usage_percent must be between 1 and 100")

            if self.system_check_interval_seconds <= 0:
                errors.append("system_check_interval_seconds must be positive")

        # Validate auto-reset settings
        for breaker_type, seconds in self.auto_reset_after_seconds.items():
            if seconds <= 0:
                errors.append(f"auto_reset_after_seconds[{breaker_type}] must be positive")

        return errors


@dataclass
class HighAvailabilityConfig(BaseConfig):
    """Configuration for high availability settings."""
    # Enable/disable high availability
    enable_high_availability: bool = False

    # Node configuration
    node_id: str = "node1"
    node_role: str = "primary"  # "primary" or "secondary"

    # Cluster configuration
    cluster_nodes: List[str] = field(default_factory=list)
    quorum_size: int = 2

    # Replication settings
    replication_mode: str = "synchronous"  # "synchronous" or "asynchronous"
    sync_interval_seconds: int = 5
    max_sync_delay_seconds: int = 30

    # Failover settings
    enable_automatic_failover: bool = True
    failover_timeout_seconds: int = 60
    max_failover_attempts: int = 3

    # Health check settings
    health_check_interval_seconds: int = 10
    heartbeat_timeout_seconds: int = 30

    # Connection settings
    node_endpoints: Dict[str, str] = field(default_factory=dict)
    replication_port: int = 5432
    admin_port: int = 8008

    def validate(self) -> List[str]:
        """Validate the high availability configuration."""
        errors = super().validate()

        # Only validate if high availability is enabled
        if not self.enable_high_availability:
            return errors

        # Validate node configuration
        if not self.node_id:
            errors.append("node_id must not be empty")

        if self.node_role not in ["primary", "secondary"]:
            errors.append("node_role must be 'primary' or 'secondary'")

        # Validate cluster configuration
        if not self.cluster_nodes:
            errors.append("cluster_nodes must not be empty")

        if self.quorum_size <= 0 or self.quorum_size > len(self.cluster_nodes):
            errors.append(f"quorum_size must be between 1 and {len(self.cluster_nodes) or 1}")

        # Validate replication settings
        if self.replication_mode not in ["synchronous", "asynchronous"]:
            errors.append("replication_mode must be 'synchronous' or 'asynchronous'")

        if self.sync_interval_seconds <= 0:
            errors.append("sync_interval_seconds must be positive")

        if self.max_sync_delay_seconds <= 0:
            errors.append("max_sync_delay_seconds must be positive")

        # Validate failover settings
        if self.enable_automatic_failover:
            if self.failover_timeout_seconds <= 0:
                errors.append("failover_timeout_seconds must be positive")

            if self.max_failover_attempts <= 0:
                errors.append("max_failover_attempts must be positive")

        # Validate health check settings
        if self.health_check_interval_seconds <= 0:
            errors.append("health_check_interval_seconds must be positive")

        if self.heartbeat_timeout_seconds <= 0:
            errors.append("heartbeat_timeout_seconds must be positive")

        # Validate connection settings
        if self.node_role == "primary" and not self.node_endpoints:
            errors.append("node_endpoints must not be empty for primary node")

        if self.replication_port <= 0 or self.replication_port > 65535:
            errors.append("replication_port must be between 1 and 65535")

        if self.admin_port <= 0 or self.admin_port > 65535:
            errors.append("admin_port must be between 1 and 65535")

        if self.replication_port == self.admin_port:
            errors.append("replication_port and admin_port must be different")

        return errors


@dataclass
class DisasterRecoveryConfig(BaseConfig):
    """Configuration for disaster recovery."""
    # Instance identification
    instance_id: Optional[str] = None

    # Recovery mode settings
    recovery_mode: RecoveryMode = RecoveryMode.MANUAL

    # File paths
    journal_path: str = "data/recovery/journal"
    snapshot_path: str = "data/recovery/snapshots"
    verification_path: str = "data/recovery/verification"
    temp_path: str = "data/recovery/temp"
    replication_path: str = "data/recovery/replication"
    models_path: str = "models"

    # Snapshot settings
    auto_snapshot_enabled: bool = True
    auto_snapshot_interval_minutes: int = 60
    max_snapshots_in_index: int = 100
    max_snapshots_to_keep: int = 20

    # Component settings
    component_snapshot_timeout_seconds: int = 30

    # Recovery settings
    auto_recover_on_error: bool = False
    auto_recover_on_circuit_breaker: bool = False
    auto_recover_on_health_check: bool = False
    auto_recover_on_position_mismatch: bool = False
    auto_recover_error_types: List[str] = field(default_factory=lambda: ["critical"])
    auto_recover_circuit_types: List[str] = field(default_factory=list)
    auto_recover_components: List[str] = field(default_factory=list)

    # Trigger settings
    snapshot_on_error: bool = True
    snapshot_on_circuit_breaker: bool = True
    snapshot_on_health_check: bool = True
    snapshot_on_position_mismatch: bool = True
    snapshot_on_shutdown: bool = True
    snapshot_error_types: List[str] = field(default_factory=lambda: ["critical", "error"])
    snapshot_circuit_types: List[str] = field(default_factory=lambda: ["POSITION", "RISK"])

    # Replication settings
    multi_region_enabled: bool = False
    auto_replicate: bool = True
    remote_regions: List[Dict[str, str]] = field(default_factory=list)

    # Model settings
    include_models_in_snapshot: bool = True
    include_models_in_recovery: bool = True

    # Detailed configuration objects
    backup_config: BackupConfig = field(default_factory=BackupConfig)
    recovery_config: RecoveryConfig = field(default_factory=RecoveryConfig)
    state_journal_config: StateJournalConfig = field(default_factory=StateJournalConfig)
    circuit_breaker_config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    high_availability_config: HighAvailabilityConfig = field(default_factory=HighAvailabilityConfig)

    def __post_init__(self):
        """Initialize the disaster recovery configuration."""
        super().__post_init__()

        # Convert recovery_mode from string if needed
        if isinstance(self.recovery_mode, str):
            try:
                self.recovery_mode = RecoveryMode(self.recovery_mode.lower())
            except ValueError:
                print(f"Warning: Invalid recovery_mode '{self.recovery_mode}', defaulting to MANUAL")
                self.recovery_mode = RecoveryMode.MANUAL

    def validate(self) -> List[str]:
        """Validate the disaster recovery configuration."""
        errors = super().validate()

        # Validate file paths
        for path_name in ["journal_path", "snapshot_path", "verification_path", "temp_path", "replication_path"]:
            path_value = getattr(self, path_name)
            if not path_value:
                errors.append(f"{path_name} must not be empty")

        # Validate snapshot settings
        if self.auto_snapshot_enabled:
            if self.auto_snapshot_interval_minutes <= 0:
                errors.append("auto_snapshot_interval_minutes must be positive")

            if self.max_snapshots_in_index <= 0:
                errors.append("max_snapshots_in_index must be positive")

            if self.max_snapshots_to_keep <= 0:
                errors.append("max_snapshots_to_keep must be positive")

        # Validate component settings
        if self.component_snapshot_timeout_seconds <= 0:
            errors.append("component_snapshot_timeout_seconds must be positive")

        # Validate replication settings
        if self.multi_region_enabled:
            for region in self.remote_regions:
                if "region_id" not in region:
                    errors.append("Each remote region must have a region_id")
                if "endpoint" not in region:
                    errors.append("Each remote region must have an endpoint")

        # Validate sub-configurations
        errors.extend(self.backup_config.validate())
        errors.extend(self.recovery_config.validate())
        errors.extend(self.state_journal_config.validate())
        errors.extend(self.circuit_breaker_config.validate())
        errors.extend(self.high_availability_config.validate())

        return errors


# Singleton instance access function
_disaster_recovery_config_instance = None

def get_disaster_recovery_config() -> DisasterRecoveryConfig:
    """
    Get the disaster recovery configuration singleton instance.

    Returns:
        The disaster recovery configuration instance
    """
    global _disaster_recovery_config_instance

    if _disaster_recovery_config_instance is None:
        # Load configuration from file or create default
        config_manager = ConfigManager()
        config_dict = config_manager.load_config("disaster_recovery")

        if config_dict:
            _disaster_recovery_config_instance = DisasterRecoveryConfig.from_dict(config_dict)
        else:
            _disaster_recovery_config_instance = DisasterRecoveryConfig()

    return _disaster_recovery_config_instance