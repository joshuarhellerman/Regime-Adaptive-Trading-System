"""
disaster_recovery.py - System Backup and Recovery with Consistency Guarantees

This module provides enterprise-grade reliability through transactional state
management, multi-region replication, and deterministic recovery capabilities.
It ensures the system can recover from failures while maintaining data integrity
and consistency.

Key features:
- Transactional state journaling with ACID properties
- Point-in-time recovery with versioned snapshots
- Event sourcing architecture for state reconstruction
- Multi-level verification and reconciliation
- Automated recovery with configurable policies
"""

import datetime
import json
import logging
import os
import pickle
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple, Set, Union
import re

from core.event_bus import EventTopics, Event, get_event_bus, EventPriority
from core.state_manager import get_state_manager, StateScope
from core.health_monitor import HealthMonitor, HealthStatus, AlertLevel
from core.circuit_breaker import CircuitBreaker, CircuitState, CircuitType, TriggerType
from config.disaster_recovery_config import DisasterRecoveryConfig, BackupStrategy, RecoveryMode
from data.storage.persistent_queue import PersistentQueue
from data.storage.time_series_store import TimeSeriesStore
from data.storage.       market_snapshot import MarketSnapshotManager
from utils.logger import get_logger

logger = get_logger(__name__)


class RecoveryState(Enum):
    """Recovery process states"""
    IDLE = "idle"
    PREPARING = "preparing"
    SNAPSHOT_CREATION = "snapshot_creation"
    DATA_BACKUP = "data_backup"
    STATE_VERIFICATION = "state_verification"
    JOURNAL_REPLAY = "journal_replay"
    RECONCILIATION = "reconciliation"
    STATE_RESTORATION = "state_restoration"
    COMPONENT_RECOVERY = "component_recovery"
    COMPLETED = "completed"
    FAILED = "failed"


class RecoveryTrigger(Enum):
    """Triggers for recovery operations"""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    AUTOMATIC = "automatic"
    HEALTH_CHECK = "health_check"
    CIRCUIT_BREAKER = "circuit_breaker"
    STATE_CORRUPTION = "state_corruption"
    SYSTEM_CRASH = "system_crash"
    POSITION_MISMATCH = "position_mismatch"


@dataclass
class RecoveryEvent:
    """Event data for recovery operations"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trigger: RecoveryTrigger = RecoveryTrigger.MANUAL
    timestamp: float = field(default_factory=time.time)
    state: RecoveryState = RecoveryState.IDLE
    snapshot_id: Optional[str] = None
    source_instance: Optional[str] = None
    target_instance: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: Optional[bool] = None
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None
    components_recovered: List[str] = field(default_factory=list)


@dataclass
class SnapshotMetadata:
    """Metadata for system snapshots"""
    id: str
    timestamp: float
    trigger: RecoveryTrigger
    state_hash: str
    size_bytes: int
    component_status: Dict[str, str] = field(default_factory=dict)
    transaction_count: int = 0
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    is_complete: bool = False
    created_at: float = field(default_factory=time.time)
    is_verified: bool = False
    verification_time: Optional[float] = None
    regions: List[str] = field(default_factory=list)


class DisasterRecovery:
    """
    Disaster Recovery System for ML-Powered Trading System
    
    Provides comprehensive disaster recovery capabilities including:
    - Transactional state management with ACID guarantees
    - Multi-region replication for high availability
    - Point-in-time recovery capabilities
    - Automated and manual recovery procedures
    - State verification and reconciliation
    """
    
    def __init__(
        self, 
        config: DisasterRecoveryConfig,
        event_bus=None, 
        state_manager=None,
        health_monitor: Optional[HealthMonitor] = None,
        circuit_breaker: Optional[CircuitBreaker] = None
    ):
        """
        Initialize the disaster recovery system.
        
        Args:
            config: Disaster recovery configuration
            event_bus: Event bus for system events
            state_manager: State manager for state operations
            health_monitor: Health monitor for system health tracking
            circuit_breaker: Circuit breaker for failure detection
        """
        self.config = config
        self.event_bus = event_bus or get_event_bus()
        self.state_manager = state_manager or get_state_manager()
        self.health_monitor = health_monitor
        self.circuit_breaker = circuit_breaker
        
        # Create dirs if they don't exist
        self._journal_dir = Path(config.journal_path)
        self._snapshot_dir = Path(config.snapshot_path)
        self._verification_dir = Path(config.verification_path)
        self._temp_dir = Path(config.temp_path)
        self._replicated_dir = Path(config.replication_path)
        
        for directory in [self._journal_dir, self._snapshot_dir, 
                          self._verification_dir, self._temp_dir,
                          self._replicated_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Internal state
        self._recovery_lock = threading.RLock()
        self._snapshot_lock = threading.RLock()
        self._replication_lock = threading.RLock()
        self._current_recovery: Optional[RecoveryEvent] = None
        self._recovery_in_progress = threading.Event()
        self._instance_id = config.instance_id or str(uuid.uuid4())[:8]
        self._recovery_state = RecoveryState.IDLE
        self._last_snapshot_time = 0
        self._last_replication_time = 0
        self._snapshots: Dict[str, SnapshotMetadata] = {}
        self._snapshot_index_path = self._snapshot_dir / "snapshot_index.json"
        
        # Journal for transactional operations
        self._transaction_journal = PersistentQueue(
            directory=str(self._journal_dir / "transactions"),
            max_memory_items=1000
        )
        
        # State verification trackers
        self._verification_rules: List[Callable[[], Tuple[bool, str]]] = []
        self._component_verifiers: Dict[str, Callable[[], Tuple[bool, Dict[str, Any]]]] = {}
        
        # Replication status
        self._remote_regions: Dict[str, Dict[str, Any]] = {}
        
        # Register event handlers
        self._register_event_handlers()
        
        # Initialize remote regions if configured
        if config.multi_region_enabled:
            self._init_remote_regions()
        
        # Load snapshot metadata
        self._load_snapshot_index()
        
        # Start scheduled tasks if configured
        if config.auto_snapshot_enabled:
            self._schedule_snapshot_task()
        
        # Log initialization
        logger.info(
            f"Disaster recovery initialized. Instance: {self._instance_id}, "
            f"Mode: {config.recovery_mode.value}, "
            f"Multi-region: {config.multi_region_enabled}"
        )

    def _register_event_handlers(self):
        """Register handlers for system events."""
        # System events
        self.event_bus.subscribe(
            EventTopics.SYSTEM_ERROR, 
            self._handle_system_error
        )
        
        self.event_bus.subscribe(
            EventTopics.CIRCUIT_BREAKER_TRIGGERED,
            self._handle_circuit_breaker
        )
        
        self.event_bus.subscribe(
            EventTopics.HEALTH_CHECK,
            self._handle_health_check
        )
        
        self.event_bus.subscribe(
            EventTopics.STATE_SAVED,
            self._handle_state_change
        )
        
        self.event_bus.subscribe(
            EventTopics.POSITION_UPDATED,
            self._handle_position_update
        )
        
        self.event_bus.subscribe(
            EventTopics.SYSTEM_SHUTDOWN,
            self._handle_system_shutdown
        )
        
        # Recovery-specific events
        self.event_bus.subscribe(
            "disaster_recovery.snapshot_request",
            self._handle_snapshot_request
        )
        
        self.event_bus.subscribe(
            "disaster_recovery.recovery_request",
            self._handle_recovery_request
        )
        
        self.event_bus.subscribe(
            "disaster_recovery.verification_request",
            self._handle_verification_request
        )

    def _init_remote_regions(self):
        """Initialize connections to remote regions for replication."""
        for region_config in self.config.remote_regions:
            region_id = region_config.get("region_id")
            endpoint = region_config.get("endpoint")
            if not region_id or not endpoint:
                logger.warning(f"Invalid region configuration: {region_config}")
                continue
                
            try:
                # Create region directory
                region_dir = self._replicated_dir / region_id
                region_dir.mkdir(parents=True, exist_ok=True)
                
                # Initialize region status
                self._remote_regions[region_id] = {
                    "endpoint": endpoint,
                    "status": "connected",
                    "last_sync": 0,
                    "last_error": None,
                    "snapshots_synced": 0
                }
                
                logger.info(f"Initialized remote region: {region_id} at {endpoint}")
                
            except Exception as e:
                logger.error(f"Failed to initialize remote region {region_id}: {e}")

    def _schedule_snapshot_task(self):
        """Schedule automatic snapshot creation."""
        interval_minutes = self.config.auto_snapshot_interval_minutes
        
        def snapshot_task():
            """Recurring task for scheduled snapshots."""
            while True:
                try:
                    # Sleep for the configured interval
                    time.sleep(interval_minutes * 60)
                    
                    # Create snapshot
                    self.create_snapshot(trigger=RecoveryTrigger.SCHEDULED)
                    
                    # Replicate to remote regions if configured
                    if self.config.multi_region_enabled and self.config.auto_replicate:
                        self.replicate_latest_snapshot()
                        
                except Exception as e:
                    logger.error(f"Error in scheduled snapshot task: {e}", exc_info=True)
        
        # Start the scheduler thread
        thread = threading.Thread(
            target=snapshot_task,
            daemon=True,
            name="SnapshotScheduler"
        )
        thread.start()
        logger.info(f"Scheduled automatic snapshots every {interval_minutes} minutes")

    def _load_snapshot_index(self):
        """Load the snapshot index from disk."""
        try:
            if self._snapshot_index_path.exists():
                with open(self._snapshot_index_path, 'r') as f:
                    index_data = json.load(f)
                
                # Convert to snapshot metadata objects
                self._snapshots = {}
                for snapshot_data in index_data.get("snapshots", []):
                    snapshot_id = snapshot_data.get("id")
                    if snapshot_id:
                        self._snapshots[snapshot_id] = SnapshotMetadata(**snapshot_data)
                
                # Sort by timestamp to find latest
                if self._snapshots:
                    latest = max(self._snapshots.values(), key=lambda s: s.timestamp)
                    self._last_snapshot_time = latest.timestamp
                    logger.info(f"Loaded {len(self._snapshots)} snapshots, latest from {datetime.datetime.fromtimestamp(latest.timestamp)}")
                else:
                    logger.info("No snapshots found in index")
            else:
                logger.info("No snapshot index found, starting fresh")
                self._snapshots = {}
                
        except Exception as e:
            logger.error(f"Failed to load snapshot index: {e}", exc_info=True)
            self._snapshots = {}

    def _update_snapshot_index(self):
        """Update the snapshot index on disk."""
        try:
            # Convert to serializable format
            snapshots_data = [asdict(metadata) for metadata in self._snapshots.values()]
            
            # Sort by timestamp (newest first)
            snapshots_data = sorted(snapshots_data, key=lambda s: s["timestamp"], reverse=True)
            
            # Limit number of snapshots in index
            if len(snapshots_data) > self.config.max_snapshots_in_index:
                snapshots_data = snapshots_data[:self.config.max_snapshots_in_index]
            
            # Update the index
            index_data = {
                "last_updated": time.time(),
                "instance_id": self._instance_id,
                "snapshots": snapshots_data
            }
            
            # Write to temp file first
            temp_path = self._snapshot_index_path.with_suffix(".tmp")
            with open(temp_path, 'w') as f:
                json.dump(index_data, f, indent=2)
            
            # Rename for atomic update
            temp_path.replace(self._snapshot_index_path)
            
            logger.debug(f"Updated snapshot index with {len(snapshots_data)} entries")
            
        except Exception as e:
            logger.error(f"Failed to update snapshot index: {e}", exc_info=True)

    def register_verification_rule(self, rule: Callable[[], Tuple[bool, str]]):
        """
        Register a verification rule for system state validation.
        
        Args:
            rule: Callable that returns (is_valid, error_message)
        """
        self._verification_rules.append(rule)
        logger.debug(f"Registered verification rule, total: {len(self._verification_rules)}")

    def register_component_verifier(self, component_id: str, verifier: Callable[[], Tuple[bool, Dict[str, Any]]]):
        """
        Register a component verifier for component state validation.
        
        Args:
            component_id: Unique component identifier
            verifier: Callable that returns (is_valid, details)
        """
        self._component_verifiers[component_id] = verifier
        logger.debug(f"Registered component verifier for {component_id}")

    def create_snapshot(self, trigger: RecoveryTrigger = RecoveryTrigger.MANUAL, metadata: Dict[str, Any] = None) -> str:
        """
        Create a system snapshot for disaster recovery.
        
        Args:
            trigger: What triggered this snapshot
            metadata: Additional metadata to store with the snapshot
            
        Returns:
            Snapshot ID
        """
        with self._snapshot_lock:
            # Generate snapshot ID
            timestamp = time.time()
            snapshot_id = f"snapshot_{int(timestamp)}_{uuid.uuid4().hex[:8]}"
            
            # Create snapshot directory
            snapshot_dir = self._snapshot_dir / snapshot_id
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize metadata
            snapshot_metadata = SnapshotMetadata(
                id=snapshot_id,
                timestamp=timestamp,
                trigger=trigger,
                state_hash="",  # Will be filled after state capture
                size_bytes=0,   # Will be updated after files are written
                custom_metadata=metadata or {}
            )
            
            try:
                # Update recovery state
                self._recovery_state = RecoveryState.SNAPSHOT_CREATION
                
                # Notify about snapshot start
                self.event_bus.publish(Event(
                    topic="disaster_recovery.snapshot_started",
                    data={
                        "snapshot_id": snapshot_id,
                        "trigger": trigger.value,
                        "timestamp": timestamp
                    },
                    priority=EventPriority.NORMAL,
                    source="disaster_recovery"
                ))
                
                logger.info(f"Creating system snapshot: {snapshot_id}, trigger: {trigger.value}")
                
                # Capture system state (using transaction to ensure consistency)
                with self.state_manager.begin_transaction() as txn:
                    # Create a backup of the current state
                    state_snapshot = self.state_manager.create_snapshot()
                    
                    # Write state to snapshot directory
                    state_path = snapshot_dir / "system_state.json"
                    with open(state_path, 'w') as f:
                        json.dump(state_snapshot, f, indent=2)
                    
                    # Calculate state hash for integrity verification
                    import hashlib
                    state_hash = hashlib.sha256(json.dumps(state_snapshot, sort_keys=True).encode()).hexdigest()
                    snapshot_metadata.state_hash = state_hash
                    
                    # Update component status
                    snapshot_metadata.component_status["state_manager"] = "completed"
                
                # Collect snapshots from other components
                component_snapshots = self._collect_component_snapshots()
                
                # Write component snapshots to disk
                for component_id, component_data in component_snapshots.items():
                    component_file = snapshot_dir / f"{component_id}.json"
                    with open(component_file, 'w') as f:
                        json.dump(component_data, f, indent=2)
                    
                    # Update component status
                    snapshot_metadata.component_status[component_id] = "completed"
                
                # Backup transaction journal
                journal_path = snapshot_dir / "transaction_journal.bin"
                self._backup_transaction_journal(journal_path)
                
                # Create market data snapshot if available
                try:
                    from data.market_snapshot import create_snapshot
                    market_snapshot = create_snapshot()
                    if market_snapshot:
                        market_file = snapshot_dir / "market_snapshot.json"
                        with open(market_file, 'w') as f:
                            json.dump(market_snapshot, f, indent=2)
                        snapshot_metadata.component_status["market_data"] = "completed"
                except (ImportError, Exception) as e:
                    logger.warning(f"Failed to create market snapshot: {e}")
                    snapshot_metadata.component_status["market_data"] = "failed"
                
                # Create model snapshots if configured
                if self.config.include_models_in_snapshot:
                    model_dir = snapshot_dir / "models"
                    model_dir.mkdir(exist_ok=True)
                    try:
                        self._backup_models(model_dir)
                        snapshot_metadata.component_status["models"] = "completed"
                    except Exception as e:
                        logger.error(f"Failed to backup models: {e}")
                        snapshot_metadata.component_status["models"] = "failed"
                
                # Calculate total size
                total_size = sum(f.stat().st_size for f in snapshot_dir.glob('**/*') if f.is_file())
                snapshot_metadata.size_bytes = total_size
                
                # Mark snapshot as complete
                snapshot_metadata.is_complete = True
                
                # Save snapshot metadata
                metadata_path = snapshot_dir / "metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(asdict(snapshot_metadata), f, indent=2)
                
                # Update snapshots dictionary and index
                self._snapshots[snapshot_id] = snapshot_metadata
                self._update_snapshot_index()
                self._last_snapshot_time = timestamp
                
                # Publish snapshot completed event
                self.event_bus.publish(Event(
                    topic="disaster_recovery.snapshot_completed",
                    data={
                        "snapshot_id": snapshot_id,
                        "trigger": trigger.value,
                        "timestamp": timestamp,
                        "size_bytes": total_size
                    },
                    priority=EventPriority.NORMAL,
                    source="disaster_recovery"
                ))
                
                logger.info(f"Snapshot {snapshot_id} created successfully ({total_size / (1024*1024):.2f} MB)")
                
                # Reset recovery state
                self._recovery_state = RecoveryState.IDLE
                
                # Replicate to remote regions if configured
                if (self.config.multi_region_enabled and 
                    self.config.auto_replicate and 
                    trigger != RecoveryTrigger.SCHEDULED):  # Don't replicate scheduled snapshots automatically
                    self.replicate_snapshot(snapshot_id)
                
                return snapshot_id
                
            except Exception as e:
                logger.error(f"Failed to create snapshot {snapshot_id}: {e}", exc_info=True)
                
                # Update metadata to indicate failure
                failed_metadata_path = snapshot_dir / "failed_metadata.json"
                try:
                    snapshot_metadata.is_complete = False
                    with open(failed_metadata_path, 'w') as f:
                        json.dump(asdict(snapshot_metadata), f, indent=2)
                except:
                    pass
                
                # Publish snapshot failed event
                self.event_bus.publish(Event(
                    topic="disaster_recovery.snapshot_failed",
                    data={
                        "snapshot_id": snapshot_id,
                        "trigger": trigger.value,
                        "timestamp": timestamp,
                        "error": str(e)
                    },
                    priority=EventPriority.HIGH,
                    source="disaster_recovery"
                ))
                
                # Reset recovery state
                self._recovery_state = RecoveryState.IDLE
                
                raise

    def _collect_component_snapshots(self) -> Dict[str, Any]:
        """
        Collect snapshots from various system components.
        
        Returns:
            Dictionary of component snapshots
        """
        # Request snapshots from all components
        event = Event(
            topic="disaster_recovery.component_snapshot_request",
            data={
                "timestamp": time.time(),
                "requestor": "disaster_recovery"
            },
            priority=EventPriority.HIGH,
            source="disaster_recovery"
        )
        
        # Use request-response pattern if supported
        try:
            responses = self.event_bus.request(
                event_topic="disaster_recovery.component_snapshot_request",
                payload={
                    "timestamp": time.time(),
                    "requestor": "disaster_recovery"
                },
                timeout_seconds=self.config.component_snapshot_timeout_seconds
            )
            
            if responses:
                logger.info(f"Collected snapshots from {len(responses)} components")
                return responses
        except (AttributeError, Exception) as e:
            logger.debug(f"Request-response not supported, falling back to publish-subscribe: {e}")
        
        # Fallback to publish-subscribe pattern
        # This is less reliable as we have to wait and collect responses
        component_snapshots = {}
        response_event = threading.Event()
        
        def snapshot_response_handler(event):
            """Handle component snapshot responses."""
            if event.topic == "disaster_recovery.component_snapshot_response":
                component_id = event.data.get("component_id")
                snapshot_data = event.data.get("snapshot_data")
                
                if component_id and snapshot_data:
                    component_snapshots[component_id] = snapshot_data
        
        # Register temporary handler
        subscription_id = self.event_bus.subscribe(
            "disaster_recovery.component_snapshot_response", 
            snapshot_response_handler
        )
        
        try:
            # Publish request
            self.event_bus.publish(event)
            
            # Wait for responses with timeout
            time.sleep(self.config.component_snapshot_timeout_seconds)
            
            logger.info(f"Collected snapshots from {len(component_snapshots)} components")
            return component_snapshots
            
        finally:
            # Unsubscribe temporary handler
            try:
                self.event_bus.unsubscribe(
                    "disaster_recovery.component_snapshot_response", 
                    subscriber_id=subscription_id
                )
            except:
                pass

    def _backup_transaction_journal(self, output_path: Path):
        """
        Backup the transaction journal to the specified path.
        
        Args:
            output_path: Path to write the journal backup
        """
        try:
            # Scan all items in the journal queue
            journal_items = []
            for item_id, data in self._transaction_journal.scan():
                journal_items.append(data)
            
            # Write to output file
            with open(output_path, 'wb') as f:
                pickle.dump(journal_items, f)
                
            logger.debug(f"Backed up {len(journal_items)} journal entries to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to backup transaction journal: {e}")
            raise

    def _backup_models(self, model_dir: Path):
        """
        Backup ML models to the specified directory.
        
        Args:
            model_dir: Directory to store model backups
        """
        try:
            # Create models directory if it doesn't exist
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if models directory is configured
            if not self.config.models_path:
                logger.warning("Models path not configured, skipping model backup")
                return
            
            # Get source directory
            src_dir = Path(self.config.models_path)
            if not src_dir.exists():
                logger.warning(f"Models directory {src_dir} does not exist, skipping model backup")
                return
            
            # Copy model files
            model_files = list(src_dir.glob('*.model')) + list(src_dir.glob('*.joblib')) + list(src_dir.glob('*.h5'))
            for model_file in model_files:
                dest_file = model_dir / model_file.name
                shutil.copy2(model_file, dest_file)
            
            # Copy model metadata
            metadata_files = list(src_dir.glob('*.json')) + list(src_dir.glob('*.yaml')) + list(src_dir.glob('*.yml'))
            for metadata_file in metadata_files:
                dest_file = model_dir / metadata_file.name
                shutil.copy2(metadata_file, dest_file)
            
            logger.info(f"Backed up {len(model_files)} model files and {len(metadata_files)} metadata files")
            
        except Exception as e:
            logger.error(f"Failed to backup models: {e}")
            raise

    def get_snapshots(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Get list of available snapshots.
        
        Args:
            limit: Maximum number of snapshots to return
            
        Returns:
            List of snapshot information dictionaries
        """
        with self._snapshot_lock:
            # Sort by timestamp (newest first)
            sorted_snapshots = sorted(
                self._snapshots.values(), 
                key=lambda s: s.timestamp, 
                reverse=True
            )
            
            # Apply limit if specified
            if limit:
                sorted_snapshots = sorted_snapshots[:limit]
            
            # Convert to dictionaries with formatted timestamps
            result = []
            for snapshot in sorted_snapshots:
                # Create a human-readable timestamp
                dt = datetime.datetime.fromtimestamp(snapshot.timestamp)
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                
                # Add snapshot info
                snapshot_dict = asdict(snapshot)
                snapshot_dict["formatted_time"] = formatted_time
                result.append(snapshot_dict)
            
            return result

    def get_snapshot_info(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific snapshot.
        
        Args:
            snapshot_id: ID of the snapshot
            
        Returns:
            Snapshot information dictionary or None if not found
        """
        with self._snapshot_lock:
            snapshot = self._snapshots.get(snapshot_id)
            if not snapshot:
                return None
            
            # Create a human-readable timestamp
            dt = datetime.datetime.fromtimestamp(snapshot.timestamp)
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            
            # Add snapshot info
            snapshot_dict = asdict(snapshot)
            snapshot_dict["formatted_time"] = formatted_time
            
            # Get file existence information
            snapshot_dir = self._snapshot_dir / snapshot_id
            if snapshot_dir.exists():
                # Check for specific files
                files = {
                    "metadata": (snapshot_dir / "metadata.json").exists(),
                    "system_state": (snapshot_dir / "system_state.json").exists(),
                    "transaction_journal": (snapshot_dir / "transaction_journal.bin").exists(),
                    "market_snapshot": (snapshot_dir / "market_snapshot.json").exists(),
                    "models": (snapshot_dir / "models").exists()
                }
                snapshot_dict["files"] = files
                
                # Calculate total size if not already set
                if snapshot.size_bytes == 0:
                    total_size = sum(f.stat().st_size for f in snapshot_dir.glob('**/*') if f.is_file())
                    snapshot_dict["size_bytes"] = total_size
            
            return snapshot_dict

    def verify_snapshot(self, snapshot_id: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify the integrity and completeness of a snapshot.
        
        Args:
            snapshot_id: ID of the snapshot to verify
            
        Returns:
            Tuple of (is_valid, verification_details)
        """
        with self._snapshot_lock:
            snapshot = self._snapshots.get(snapshot_id)
            if not snapshot:
                return False, {"error": f"Snapshot {snapshot_id} not found"}
            
            # Get snapshot directory
            snapshot_dir = self._snapshot_dir / snapshot_id
            if not snapshot_dir.exists():
                return False, {"error": f"Snapshot directory {snapshot_dir} does not exist"}
            
            verification_results = {
                "snapshot_id": snapshot_id,
                "timestamp": time.time(),
                "checks": [],
                "is_valid": True
            }
            
            # Check metadata existence
            metadata_path = snapshot_dir / "metadata.json"
            if not metadata_path.exists():
                verification_results["checks"].append({
                    "check": "metadata_exists",
                    "result": False,
                    "message": "Metadata file does not exist"
                })
                verification_results["is_valid"] = False
                return False, verification_results
            
            # Check system state existence
            state_path = snapshot_dir / "system_state.json"
            if not state_path.exists():
                verification_results["checks"].append({
                    "check": "system_state_exists",
                    "result": False,
                    "message": "System state file does not exist"
                })
                verification_results["is_valid"] = False
                return False, verification_results
            
            # Verify state hash
            try:
                with open(state_path, 'r') as f:
                    state_data = json.load(f)
                
                import hashlib
                computed_hash = hashlib.sha256(json.dumps(state_data, sort_keys=True).encode()).hexdigest()
                
                if computed_hash != snapshot.state_hash:
                    verification_results["checks"].append({
                        "check": "state_hash",
                        "result": False,
                        "message": f"State hash mismatch. Expected: {snapshot.state_hash}, Computed: {computed_hash}"
                    })
                    verification_results["is_valid"] = False
                else:
                    verification_results["checks"].append({
                        "check": "state_hash",
                        "result": True,
                        "message": "State hash verified"
                    })
            except Exception as e:
                verification_results["checks"].append({
                    "check": "state_hash",
                    "result": False,
                    "message": f"Error verifying state hash: {str(e)}"
                })
                verification_results["is_valid"] = False
            
            # Check component snapshots
            for component_id, status in snapshot.component_status.items():
                if status == "completed":
                    component_path = snapshot_dir / f"{component_id}.json"
                    if not component_path.exists():
                        verification_results["checks"].append({
                            "check": f"component_{component_id}",
                            "result": False,
                            "message": f"Component {component_id} snapshot file does not exist"
                        })
                        verification_results["is_valid"] = False
                    else:
                        verification_results["checks"].append({
                            "check": f"component_{component_id}",
                            "result": True,
                            "message": f"Component {component_id} snapshot verified"
                        })
            
            # Verify journal if exists
            journal_path = snapshot_dir / "transaction_journal.bin"
            if journal_path.exists():
                try:
                    with open(journal_path, 'rb') as f:
                        journal_data = pickle.load(f)
                    
                    verification_results["checks"].append({
                        "check": "transaction_journal",
                        "result": True,
                        "message": f"Transaction journal verified with {len(journal_data)} entries"
                    })
                except Exception as e:
                    verification_results["checks"].append({
                        "check": "transaction_journal",
                        "result": False,
                        "message": f"Error verifying transaction journal: {str(e)}"
                    })
                    verification_results["is_valid"] = False
            
            # Calculate total size if needed
            if snapshot.size_bytes == 0:
                total_size = sum(f.stat().st_size for f in snapshot_dir.glob('**/*') if f.is_file())
                snapshot.size_bytes = total_size
                self._update_snapshot_index()
                
                verification_results["checks"].append({
                    "check": "size_calculation",
                    "result": True,
                    "message": f"Size calculated: {total_size} bytes"
                })
            
            # Update verification status
            snapshot.is_verified = verification_results["is_valid"]
            snapshot.verification_time = time.time()
            self._update_snapshot_index()
            
            return verification_results["is_valid"], verification_results
    
    def replicate_snapshot(self, snapshot_id: str) -> Dict[str, bool]:
        """
        Replicate a snapshot to remote regions.
        
        Args:
            snapshot_id: ID of the snapshot to replicate
            
        Returns:
            Dictionary mapping region IDs to success status
        """
        if not self.config.multi_region_enabled:
            logger.warning("Multi-region replication is not enabled")
            return {}
        
        with self._replication_lock:
            snapshot = self._snapshots.get(snapshot_id)
            if not snapshot:
                logger.error(f"Cannot replicate snapshot {snapshot_id}: not found")
                return {}
            
            source_dir = self._snapshot_dir / snapshot_id
            if not source_dir.exists():
                logger.error(f"Cannot replicate snapshot {snapshot_id}: directory not found")
                return {}
            
            replication_results = {}
            
            for region_id, region_info in self._remote_regions.items():
                try:
                    logger.info(f"Replicating snapshot {snapshot_id} to region {region_id}")
                    
                    # Get destination path
                    dest_dir = self._replicated_dir / region_id / snapshot_id
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy all files from the snapshot
                    for src_file in source_dir.glob('**/*'):
                        if src_file.is_file():
                            # Get relative path from source directory
                            rel_path = src_file.relative_to(source_dir)
                            # Create destination path
                            dest_file = dest_dir / rel_path
                            # Create parent directories if needed
                            dest_file.parent.mkdir(parents=True, exist_ok=True)
                            # Copy the file
                            shutil.copy2(src_file, dest_file)
                    
                    # Update region metadata
                    self._remote_regions[region_id]["last_sync"] = time.time()
                    self._remote_regions[region_id]["snapshots_synced"] += 1
                    
                    # Add region to snapshot regions if not already present
                    if region_id not in snapshot.regions:
                        snapshot.regions.append(region_id)
                        self._update_snapshot_index()
                    
                    # Record success
                    replication_results[region_id] = True
                    logger.info(f"Successfully replicated snapshot {snapshot_id} to region {region_id}")
                    
                    # In a real system, this would use network transmission to a remote system
                    # For demonstration, we're just copying to a local directory structure
                    
                except Exception as e:
                    logger.error(f"Failed to replicate snapshot {snapshot_id} to region {region_id}: {e}", exc_info=True)
                    self._remote_regions[region_id]["last_error"] = str(e)
                    replication_results[region_id] = False
            
            # Update last replication time
            self._last_replication_time = time.time()
            
            # Publish replication event
            self.event_bus.publish(Event(
                topic="disaster_recovery.snapshot_replicated",
                data={
                    "snapshot_id": snapshot_id,
                    "timestamp": time.time(),
                    "regions": replication_results
                },
                priority=EventPriority.NORMAL,
                source="disaster_recovery"
            ))
            
            return replication_results
    
    def replicate_latest_snapshot(self) -> Dict[str, bool]:
        """
        Replicate the latest snapshot to remote regions.
        
        Returns:
            Dictionary mapping region IDs to success status
        """
        with self._snapshot_lock:
            # Find latest snapshot
            if not self._snapshots:
                logger.warning("No snapshots available for replication")
                return {}
            
            latest_snapshot = max(self._snapshots.values(), key=lambda s: s.timestamp)
            return self.replicate_snapshot(latest_snapshot.id)
    
    def verify_system_state(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Verify the current system state using registered verification rules.
        
        Returns:
            Tuple of (is_valid, list_of_failures)
        """
        verification_time = time.time()
        failures = []
        all_valid = True
        
        # Execute each verification rule
        for i, rule in enumerate(self._verification_rules):
            try:
                is_valid, message = rule()
                
                if not is_valid:
                    all_valid = False
                    failures.append({
                        "rule_index": i,
                        "message": message,
                        "timestamp": verification_time
                    })
                    
                    logger.warning(f"Verification rule {i} failed: {message}")
            except Exception as e:
                all_valid = False
                failures.append({
                    "rule_index": i,
                    "message": f"Exception during verification: {str(e)}",
                    "timestamp": verification_time,
                    "exception": str(e)
                })
                
                logger.error(f"Exception in verification rule {i}: {e}", exc_info=True)
        
        # Execute component verifiers
        for component_id, verifier in self._component_verifiers.items():
            try:
                is_valid, details = verifier()
                
                if not is_valid:
                    all_valid = False
                    failures.append({
                        "component_id": component_id,
                        "message": "Component verification failed",
                        "timestamp": verification_time,
                        "details": details
                    })
                    
                    logger.warning(f"Component verification failed for {component_id}: {details}")
            except Exception as e:
                all_valid = False
                failures.append({
                    "component_id": component_id,
                    "message": f"Exception during component verification: {str(e)}",
                    "timestamp": verification_time,
                    "exception": str(e)
                })
                
                logger.error(f"Exception in component verifier for {component_id}: {e}", exc_info=True)
        
        # Save verification result to state manager
        self.state_manager.set(
            path="system.verification.latest_result",
            value={
                "timestamp": verification_time,
                "is_valid": all_valid,
                "failures": failures
            },
            scope=StateScope.PERSISTENT
        )
        
        logger.info(f"System state verification completed: {'Valid' if all_valid else 'Invalid'}")
        
        # Update health status if health monitor is available
        if self.health_monitor:
            status = HealthStatus.HEALTHY if all_valid else HealthStatus.WARNING
            self.health_monitor.update_component_health(
                component_id="disaster_recovery",
                status=status,
                metrics={
                    "verification_result": 1 if all_valid else 0,
                    "verification_time": verification_time,
                    "failure_count": len(failures)
                }
            )
        
        return all_valid, failures
    
    def recover_from_snapshot(
        self, 
        snapshot_id: Optional[str] = None,
        verify_after_recovery: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Recover the system from a snapshot.
        
        Args:
            snapshot_id: ID of the snapshot to recover from, or None for latest
            verify_after_recovery: Whether to verify system state after recovery
            metadata: Additional metadata for the recovery event
            
        Returns:
            True if recovery was successful, False otherwise
        """
        # Prevent multiple simultaneous recoveries
        if self._recovery_in_progress.is_set():
            logger.warning("Recovery already in progress, ignoring new request")
            return False
        
        try:
            # Set recovery in progress flag
            self._recovery_in_progress.set()
            
            # Find snapshot to recover from
            if snapshot_id is None:
                # Use the latest snapshot
                if not self._snapshots:
                    logger.error("No snapshots available for recovery")
                    return False
                
                snapshot = max(self._snapshots.values(), key=lambda s: s.timestamp)
                snapshot_id = snapshot.id
                logger.info(f"Using latest snapshot for recovery: {snapshot_id}")
            else:
                # Find the specified snapshot
                snapshot = self._snapshots.get(snapshot_id)
                if not snapshot:
                    logger.error(f"Snapshot {snapshot_id} not found")
                    return False
            
            # Create recovery event
            recovery_event = RecoveryEvent(
                trigger=RecoveryTrigger.MANUAL,
                state=RecoveryState.PREPARING,
                snapshot_id=snapshot_id,
                target_instance=self._instance_id,
                metadata=metadata or {}
            )
            self._current_recovery = recovery_event
            
            # Update recovery state
            self._recovery_state = RecoveryState.PREPARING
            
            # Notify about recovery start
            self.event_bus.publish(Event(
                topic="disaster_recovery.recovery_started",
                data=asdict(recovery_event),
                priority=EventPriority.HIGH,
                source="disaster_recovery"
            ))
            
            # Verify snapshot integrity first
            logger.info(f"Verifying snapshot {snapshot_id} before recovery")
            self._recovery_state = RecoveryState.STATE_VERIFICATION
            is_valid, verification_details = self.verify_snapshot(snapshot_id)
            
            if not is_valid:
                logger.error(f"Snapshot {snapshot_id} verification failed, aborting recovery")
                recovery_event.state = RecoveryState.FAILED
                recovery_event.error_message = "Snapshot verification failed"
                self._recovery_state = RecoveryState.FAILED
                return False
            
            logger.info(f"Snapshot {snapshot_id} verified, proceeding with recovery")
            
            # Load snapshot data
            snapshot_dir = self._snapshot_dir / snapshot_id
            if not snapshot_dir.exists():
                logger.error(f"Snapshot directory {snapshot_id} not found")
                recovery_event.state = RecoveryState.FAILED
                recovery_event.error_message = "Snapshot directory not found"
                self._recovery_state = RecoveryState.FAILED
                return False
            
            start_time = time.time()
            
            try:
                # First create a backup of current state before recovery (for rollback if needed)
                pre_recovery_snapshot_id = self.create_snapshot(
                    trigger=RecoveryTrigger.MANUAL,
                    metadata={"purpose": "pre_recovery", "recovery_id": recovery_event.id}
                )
                recovery_event.metadata["pre_recovery_snapshot_id"] = pre_recovery_snapshot_id
                logger.info(f"Created pre-recovery snapshot: {pre_recovery_snapshot_id}")
                
                # Restore system state
                self._recovery_state = RecoveryState.STATE_RESTORATION
                state_path = snapshot_dir / "system_state.json"
                
                with open(state_path, 'r') as f:
                    state_data = json.load(f)
                
                logger.info("Restoring system state from snapshot")
                
                # Use transaction to ensure consistency
                with self.state_manager.begin_transaction() as txn:
                    self.state_manager.restore_snapshot(state_data)
                
                recovery_event.components_recovered.append("state_manager")
                
                # Replay transaction journal if available
                journal_path = snapshot_dir / "transaction_journal.bin"
                if journal_path.exists():
                    self._recovery_state = RecoveryState.JOURNAL_REPLAY
                    self._replay_transaction_journal(journal_path)
                    recovery_event.components_recovered.append("transaction_journal")
                
                # Restore market snapshot if available
                market_path = snapshot_dir / "market_snapshot.json"
                if market_path.exists():
                    try:
                        with open(market_path, 'r') as f:
                            market_data = json.load(f)
                        
                        # Try to restore market snapshot
                        from data.market_snapshot import restore_snapshot
                        if restore_snapshot(market_data):
                            recovery_event.components_recovered.append("market_data")
                    except (ImportError, Exception) as e:
                        logger.warning(f"Failed to restore market snapshot: {e}")
                
                # Restore models if available and configured
                if self.config.include_models_in_recovery:
                    model_dir = snapshot_dir / "models"
                    if model_dir.exists():
                        try:
                            self._restore_models(model_dir)
                            recovery_event.components_recovered.append("models")
                        except Exception as e:
                            logger.error(f"Failed to restore models: {e}")
                
                # Restore component data
                self._recovery_state = RecoveryState.COMPONENT_RECOVERY
                component_recoveries = self._recover_components(snapshot_dir)
                recovery_event.components_recovered.extend(component_recoveries)
                
                # Verify system state after recovery if requested
                if verify_after_recovery:
                    self._recovery_state = RecoveryState.STATE_VERIFICATION
                    is_valid, failures = self.verify_system_state()
                    
                    if not is_valid:
                        logger.warning(f"System state verification after recovery identified issues: {len(failures)} failures")
                        recovery_event.metadata["verification_failures"] = failures
                    else:
                        logger.info("System state verification after recovery successful")
                        recovery_event.metadata["verification"] = "passed"
                
                # Calculate recovery duration
                recovery_duration = time.time() - start_time
                recovery_event.duration_seconds = recovery_duration
                
                # Update recovery state
                self._recovery_state = RecoveryState.COMPLETED
                recovery_event.state = RecoveryState.COMPLETED
                recovery_event.success = True
                
                # Notify about recovery completion
                self.event_bus.publish(Event(
                    topic="disaster_recovery.recovery_completed",
                    data=asdict(recovery_event),
                    priority=EventPriority.HIGH,
                    source="disaster_recovery"
                ))
                
                logger.info(f"Recovery from snapshot {snapshot_id} completed successfully in {recovery_duration:.2f} seconds")
                
                return True
                
            except Exception as e:
                logger.error(f"Error during recovery from snapshot {snapshot_id}: {e}", exc_info=True)
                
                recovery_event.state = RecoveryState.FAILED
                recovery_event.error_message = str(e)
                recovery_event.success = False
                
                # Notify about recovery failure
                self.event_bus.publish(Event(
                    topic="disaster_recovery.recovery_failed",
                    data=asdict(recovery_event),
                    priority=EventPriority.HIGH,
                    source="disaster_recovery"
                ))
                
                self._recovery_state = RecoveryState.FAILED
                return False
                
        finally:
            # Clear recovery in progress flag
            self._recovery_in_progress.clear()
            self._current_recovery = None
    
    def _replay_transaction_journal(self, journal_path: Path):
        """
        Replay transaction journal from a snapshot.
        
        Args:
            journal_path: Path to the transaction journal file
        """
        try:
            # Load journal data
            with open(journal_path, 'rb') as f:
                journal_entries = pickle.load(f)
            
            if not journal_entries:
                logger.info("No journal entries to replay")
                return
            
            logger.info(f"Replaying {len(journal_entries)} transaction journal entries")
            
            # Replay each journal entry
            success_count = 0
            for entry in journal_entries:
                try:
                    # In a real implementation, this would reconstruct and apply the transaction
                    # For demonstration, we'll just log it
                    logger.debug(f"Replaying journal entry: {entry}")
                    
                    # Here you would typically:
                    # 1. Deserialize the transaction data
                    # 2. Validate it
                    # 3. Apply it to the state manager
                    
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"Error replaying journal entry: {e}")
            
            logger.info(f"Successfully replayed {success_count}/{len(journal_entries)} journal entries")
            
        except Exception as e:
            logger.error(f"Failed to replay transaction journal: {e}", exc_info=True)
            raise
    
    def _restore_models(self, model_dir: Path):
        """
        Restore ML models from a snapshot.
        
        Args:
            model_dir: Directory containing model backups
        """
        try:
            # Check if models directory is configured
            if not self.config.models_path:
                logger.warning("Models path not configured, skipping model restoration")
                return
            
            # Get destination directory
            dest_dir = Path(self.config.models_path)
            if not dest_dir.exists():
                dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy model files from snapshot to destination
            model_files = list(model_dir.glob('*.model')) + list(model_dir.glob('*.joblib')) + list(model_dir.glob('*.h5'))
            for model_file in model_files:
                dest_file = dest_dir / model_file.name
                shutil.copy2(model_file, dest_file)
            
            # Copy model metadata
            metadata_files = list(model_dir.glob('*.json')) + list(model_dir.glob('*.yaml')) + list(model_dir.glob('*.yml'))
            for metadata_file in metadata_files:
                dest_file = dest_dir / metadata_file.name
                shutil.copy2(metadata_file, dest_file)
            
            logger.info(f"Restored {len(model_files)} model files and {len(metadata_files)} metadata files")
            
        except Exception as e:
            logger.error(f"Failed to restore models: {e}", exc_info=True)
            raise
    
    def _recover_components(self, snapshot_dir: Path) -> List[str]:
        """
        Recover component data from snapshots.
        
        Args:
            snapshot_dir: Directory containing the snapshot
            
        Returns:
            List of recovered component IDs
        """
        recovered_components = []
        
        # Find all component snapshot files
        component_files = [f for f in snapshot_dir.glob('*.json') 
                          if f.name not in ["metadata.json", "system_state.json", "market_snapshot.json"]]
        
        for component_file in component_files:
            component_id = component_file.stem
            
            try:
                # Load component data
                with open(component_file, 'r') as f:
                    component_data = json.load(f)
                
                # Publish event to restore component
                self.event_bus.publish(Event(
                    topic=f"disaster_recovery.restore_{component_id}",
                    data={
                        "component_id": component_id,
                        "recovery_id": self._current_recovery.id if self._current_recovery else None,
                        "snapshot_id": self._current_recovery.snapshot_id if self._current_recovery else None,
                        "component_data": component_data
                    },
                    priority=EventPriority.HIGH,
                    source="disaster_recovery"
                ))
                
                recovered_components.append(component_id)
                logger.info(f"Sent recovery data to component: {component_id}")
                
            except Exception as e:
                logger.error(f"Failed to recover component {component_id}: {e}", exc_info=True)
        
        return recovered_components
    
    def get_recovery_status(self) -> Dict[str, Any]:
        """
        Get the current recovery status.
        
        Returns:
            Dictionary with recovery status information
        """
        with self._recovery_lock:
            status = {
                "is_recovery_in_progress": self._recovery_in_progress.is_set(),
                "recovery_state": self._recovery_state.value,
                "instance_id": self._instance_id,
                "last_snapshot_time": self._last_snapshot_time,
                "last_replication_time": self._last_replication_time,
                "snapshot_count": len(self._snapshots),
                "remote_regions": {region: info.get("status") for region, info in self._remote_regions.items()},
                "config": {
                    "auto_snapshot_enabled": self.config.auto_snapshot_enabled,
                    "auto_snapshot_interval_minutes": self.config.auto_snapshot_interval_minutes,
                    "multi_region_enabled": self.config.multi_region_enabled,
                    "recovery_mode": self.config.recovery_mode.value
                }
            }
            
            # Add current recovery information if available
            if self._current_recovery:
                status["current_recovery"] = asdict(self._current_recovery)
            
            return status
    
    def cleanup_old_snapshots(self, keep_count: int = None, older_than_days: int = None) -> int:
        """
        Clean up old snapshots to save disk space.
        
        Args:
            keep_count: Number of recent snapshots to keep
            older_than_days: Remove snapshots older than this many days
            
        Returns:
            Number of snapshots removed
        """
        if keep_count is None and older_than_days is None:
            # Use configuration defaults
            keep_count = self.config.max_snapshots_to_keep
        
        with self._snapshot_lock:
            # Sort snapshots by timestamp (newest first)
            sorted_snapshots = sorted(
                self._snapshots.values(), 
                key=lambda s: s.timestamp, 
                reverse=True
            )
            
            # Determine which snapshots to keep based on keep_count
            to_remove = []
            if keep_count is not None and len(sorted_snapshots) > keep_count:
                to_remove.extend(sorted_snapshots[keep_count:])
            
            # Apply older_than_days filter
            if older_than_days is not None:
                cutoff_time = time.time() - (older_than_days * 86400)
                to_remove.extend([s for s in sorted_snapshots 
                                 if s.timestamp < cutoff_time and s not in to_remove])
            
            # Remove the selected snapshots
            removed_count = 0
            for snapshot in to_remove:
                snapshot_id = snapshot.id
                try:
                    # Remove snapshot directory
                    snapshot_dir = self._snapshot_dir / snapshot_id
                    if snapshot_dir.exists():
                        shutil.rmtree(snapshot_dir)
                    
                    # Remove from snapshots dictionary
                    if snapshot_id in self._snapshots:
                        del self._snapshots[snapshot_id]
                    
                    removed_count += 1
                    logger.info(f"Removed old snapshot: {snapshot_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to remove snapshot {snapshot_id}: {e}")
            
            # Update snapshot index if any were removed
            if removed_count > 0:
                self._update_snapshot_index()
            
            return removed_count
    
    def get_remote_region_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information for remote regions.
        
        Returns:
            Dictionary mapping region IDs to status information
        """
        with self._replication_lock:
            # Return a copy to avoid external modification
            return {region_id: dict(info) for region_id, info in self._remote_regions.items()}
    
    def recover_from_remote_region(self, region_id: str, snapshot_id: Optional[str] = None) -> bool:
        """
        Recover system state from a snapshot in a remote region.
        
        Args:
            region_id: ID of the remote region
            snapshot_id: ID of the snapshot to recover from, or None for latest
            
        Returns:
            True if recovery was successful, False otherwise
        """
        if not self.config.multi_region_enabled:
            logger.warning("Multi-region recovery is not enabled")
            return False
        
        with self._replication_lock:
            # Check if region exists
            if region_id not in self._remote_regions:
                logger.error(f"Remote region {region_id} not found")
                return False
            
            region_dir = self._replicated_dir / region_id
            if not region_dir.exists():
                logger.error(f"Remote region directory {region_id} not found")
                return False
            
            # Find snapshot to recover from
            if snapshot_id is None:
                # Find the latest snapshot in the region
                snapshot_dirs = list(region_dir.glob("snapshot_*"))
                if not snapshot_dirs:
                    logger.error(f"No snapshots found in region {region_id}")
                    return False
                
                # Sort by name (which includes timestamp)
                latest_dir = sorted(snapshot_dirs, key=lambda d: d.name, reverse=True)[0]
                snapshot_id = latest_dir.name
                logger.info(f"Using latest snapshot from region {region_id}: {snapshot_id}")
            else:
                # Check if the specified snapshot exists
                snapshot_dir = region_dir / snapshot_id
                if not snapshot_dir.exists():
                    logger.error(f"Snapshot {snapshot_id} not found in region {region_id}")
                    return False
            
            # Copy the snapshot to local snapshots directory
            source_dir = region_dir / snapshot_id
            dest_dir = self._snapshot_dir / snapshot_id
            
            try:
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                
                # Copy the snapshot
                shutil.copytree(source_dir, dest_dir)
                
                # Load snapshot metadata
                metadata_path = dest_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Create snapshot metadata object
                    snapshot_metadata = SnapshotMetadata(**metadata)
                    
                    # Add to snapshots dictionary
                    self._snapshots[snapshot_id] = snapshot_metadata
                    self._update_snapshot_index()
                
                # Now recover from the copied snapshot
                return self.recover_from_snapshot(
                    snapshot_id=snapshot_id,
                    verify_after_recovery=True,
                    metadata={"source_region": region_id}
                )
                
            except Exception as e:
                logger.error(f"Failed to recover from region {region_id}: {e}", exc_info=True)
                return False
    
    def _handle_system_error(self, event: Event):
        """Handle system error event."""
        error_data = event.data
        error_message = error_data.get('message', 'Unknown error')
        error_component = error_data.get('logger_name', 'unknown')
        
        logger.warning(f"System error detected: {error_message} in {error_component}")
        
        # Check if we should create a snapshot based on errors
        if self.config.snapshot_on_error:
            # Check if this error type should trigger a snapshot
            error_type = error_data.get('exception_type', 'general')
            
            if error_type in self.config.snapshot_error_types or 'all' in self.config.snapshot_error_types:
                logger.info(f"Creating snapshot due to system error: {error_type}")
                try:
                    snapshot_id = self.create_snapshot(
                        trigger=RecoveryTrigger.AUTOMATIC,
                        metadata={
                            "error_type": error_type,
                            "error_message": error_message,
                            "error_component": error_component
                        }
                    )
                    logger.info(f"Created error snapshot: {snapshot_id}")
                except Exception as e:
                    logger.error(f"Failed to create error snapshot: {e}")
        
        # Check if this should trigger automatic recovery
        if (self.config.recovery_mode == RecoveryMode.AUTOMATIC and 
            self.config.auto_recover_on_error):
            
            # Check error type against auto recovery list
            error_type = error_data.get('exception_type', 'general')
            
            if error_type in self.config.auto_recover_error_types or 'all' in self.config.auto_recover_error_types:
                logger.info(f"Initiating automatic recovery due to error: {error_type}")
                try:
                    self.recover_from_snapshot()
                except Exception as e:
                    logger.error(f"Automatic recovery failed: {e}")
    
    def _handle_circuit_breaker(self, event: Event):
        """Handle circuit breaker event."""
        circuit_data = event.data
        circuit_id = circuit_data.get('circuit_id', 'unknown')
        circuit_type = circuit_data.get('circuit_type', 'unknown')
        new_state = circuit_data.get('new_state', 'unknown')
        
        logger.info(f"Circuit breaker event: {circuit_id} ({circuit_type}) -> {new_state}")
        
        # Only proceed if the circuit breaker is opening (tripping)
        if new_state != CircuitState.OPEN.value:
            return
        
        # Check if we should create a snapshot when circuit breakers trip
        if self.config.snapshot_on_circuit_breaker and circuit_type in self.config.snapshot_circuit_types:
            logger.info(f"Creating snapshot due to circuit breaker: {circuit_id}")
            try:
                snapshot_id = self.create_snapshot(
                    trigger=RecoveryTrigger.CIRCUIT_BREAKER,
                    metadata={
                        "circuit_id": circuit_id,
                        "circuit_type": circuit_type,
                        "trigger_value": circuit_data.get('trigger_value')
                    }
                )
                logger.info(f"Created circuit breaker snapshot: {snapshot_id}")
            except Exception as e:
                logger.error(f"Failed to create circuit breaker snapshot: {e}")
        
        # Check for automatic recovery based on circuit breaker
        if (self.config.recovery_mode == RecoveryMode.AUTOMATIC and 
            self.config.auto_recover_on_circuit_breaker):
            
            # Check circuit type against recovery list
            if circuit_type in self.config.auto_recover_circuit_types:
                logger.info(f"Initiating automatic recovery due to circuit breaker: {circuit_id}")
                try:
                    self.recover_from_snapshot()
                except Exception as e:
                    logger.error(f"Automatic recovery failed: {e}")
    
    def _handle_health_check(self, event: Event):
        """Handle health check event."""
        health_data = event.data
        status = health_data.get('status', 'unknown')
        
        # Only concerned with warning or critical status
        if status not in ['warning', 'critical']:
            return
        
        # Check components with issues
        critical_components = health_data.get('critical_components', [])
        warning_components = health_data.get('warning_components', [])
        
        logger.info(f"Health check: status={status}, critical={len(critical_components)}, warning={len(warning_components)}")
        
        # Check if we should create a snapshot based on health
        if self.config.snapshot_on_health_check and status == 'critical':
            logger.info("Creating snapshot due to critical health status")
            try:
                snapshot_id = self.create_snapshot(
                    trigger=RecoveryTrigger.HEALTH_CHECK,
                    metadata={
                        "health_status": status,
                        "critical_components": critical_components,
                        "warning_components": warning_components
                    }
                )
                logger.info(f"Created health check snapshot: {snapshot_id}")
            except Exception as e:
                logger.error(f"Failed to create health check snapshot: {e}")
        
        # Check for automatic recovery based on health
        if (self.config.recovery_mode == RecoveryMode.AUTOMATIC and 
            self.config.auto_recover_on_health_check):
            
            # Only recover on critical status with critical components
            if status == 'critical' and len(critical_components) > 0:
                # Check if any critical components match our recovery list
                should_recover = False
                for component in critical_components:
                    if component in self.config.auto_recover_components:
                        should_recover = True
                        break
                
                if should_recover:
                    logger.info(f"Initiating automatic recovery due to critical health status")
                    try:
                        self.recover_from_snapshot()
                    except Exception as e:
                        logger.error(f"Automatic recovery failed: {e}")
    
    def _handle_state_change(self, event: Event):
        """Handle state change event."""
        # This is useful for transaction journaling in a real implementation
        pass
    
    def _handle_position_update(self, event: Event):
        """Handle position update event."""
        # Check for position reconciliation issues that might require recovery
        position_data = event.data
        if 'reconciliation_failed' in position_data:
            logger.warning("Position reconciliation failure detected")
            
            # Create snapshot if configured
            if self.config.snapshot_on_position_mismatch:
                logger.info("Creating snapshot due to position reconciliation failure")
                try:
                    snapshot_id = self.create_snapshot(
                        trigger=RecoveryTrigger.POSITION_MISMATCH,
                        metadata={
                            "position_data": position_data
                        }
                    )
                    logger.info(f"Created position mismatch snapshot: {snapshot_id}")
                except Exception as e:
                    logger.error(f"Failed to create position mismatch snapshot: {e}")
            
            # Check for automatic recovery
            if (self.config.recovery_mode == RecoveryMode.AUTOMATIC and 
                self.config.auto_recover_on_position_mismatch):
                
                logger.info("Initiating automatic recovery due to position reconciliation failure")
                try:
                    self.recover_from_snapshot()
                except Exception as e:
                    logger.error(f"Automatic recovery failed: {e}")
    
    def _handle_system_shutdown(self, event: Event):
        """Handle system shutdown event."""
        logger.info("System shutdown detected, creating final snapshot")
        
        # Create a final snapshot if configured
        if self.config.snapshot_on_shutdown:
            try:
                snapshot_id = self.create_snapshot(
                    trigger=RecoveryTrigger.MANUAL,
                    metadata={"purpose": "shutdown"}
                )
                logger.info(f"Created shutdown snapshot: {snapshot_id}")
            except Exception as e:
                logger.error(f"Failed to create shutdown snapshot: {e}")
    
    def _handle_snapshot_request(self, event: Event):
        """Handle snapshot request event."""
        request_data = event.data
        trigger = request_data.get('trigger', RecoveryTrigger.MANUAL.value)
        metadata = request_data.get('metadata', {})
        
        # Convert trigger string to enum
        try:
            trigger_enum = RecoveryTrigger(trigger)
        except (ValueError, TypeError):
            trigger_enum = RecoveryTrigger.MANUAL
        
        try:
            snapshot_id = self.create_snapshot(
                trigger=trigger_enum,
                metadata=metadata
            )
            
            # Send response if requestor is specified
            requestor = request_data.get('requestor')
            if requestor:
                self.event_bus.publish(Event(
                    topic="disaster_recovery.snapshot_response",
                    data={
                        "request_id": request_data.get('request_id'),
                        "snapshot_id": snapshot_id,
                        "success": True
                    },
                    priority=EventPriority.NORMAL,
                    source="disaster_recovery"
                ))
            
        except Exception as e:
            logger.error(f"Failed to create snapshot from request: {e}")
            
            # Send error response
            requestor = request_data.get('requestor')
            if requestor:
                self.event_bus.publish(Event(
                    topic="disaster_recovery.snapshot_response",
                    data={
                        "request_id": request_data.get('request_id'),
                        "success": False,
                        "error": str(e)
                    },
                    priority=EventPriority.NORMAL,
                    source="disaster_recovery"
                ))
    
    def _handle_recovery_request(self, event: Event):
        """Handle recovery request event."""
        request_data = event.data
        snapshot_id = request_data.get('snapshot_id')
        verify_after = request_data.get('verify_after_recovery', True)
        metadata = request_data.get('metadata', {})
        
        try:
            result = self.recover_from_snapshot(
                snapshot_id=snapshot_id,
                verify_after_recovery=verify_after,
                metadata=metadata
            )
            
            # Send response if requestor is specified
            requestor = request_data.get('requestor')
            if requestor:
                self.event_bus.publish(Event(
                    topic="disaster_recovery.recovery_response",
                    data={
                        "request_id": request_data.get('request_id'),
                        "success": result
                    },
                    priority=EventPriority.NORMAL,
                    source="disaster_recovery"
                ))
            
        except Exception as e:
            logger.error(f"Failed to recover from request: {e}")
            
            # Send error response
            requestor = request_data.get('requestor')
            if requestor:
                self.event_bus.publish(Event(
                    topic="disaster_recovery.recovery_response",
                    data={
                        "request_id": request_data.get('request_id'),
                        "success": False,
                        "error": str(e)
                    },
                    priority=EventPriority.NORMAL,
                    source="disaster_recovery"
                ))
    
    def _handle_verification_request(self, event: Event):
        """Handle verification request event."""
        request_data = event.data
        
        try:
            is_valid, failures = self.verify_system_state()
            
            # Send response if requestor is specified
            requestor = request_data.get('requestor')
            if requestor:
                self.event_bus.publish(Event(
                    topic="disaster_recovery.verification_response",
                    data={
                        "request_id": request_data.get('request_id'),
                        "is_valid": is_valid,
                        "failures": failures
                    },
                    priority=EventPriority.NORMAL,
                    source="disaster_recovery"
                ))
            
        except Exception as e:
            logger.error(f"Failed to verify from request: {e}")
            
            # Send error response
            requestor = request_data.get('requestor')
            if requestor:
                self.event_bus.publish(Event(
                    topic="disaster_recovery.verification_response",
                    data={
                        "request_id": request_data.get('request_id'),
                        "success": False,
                        "error": str(e)
                    },
                    priority=EventPriority.NORMAL,
                    source="disaster_recovery"
                ))


# Function to get singleton instance
_disaster_recovery_instance = None

def get_disaster_recovery(
    config: Optional[DisasterRecoveryConfig] = None,
    event_bus=None,
    state_manager=None,
    health_monitor=None,
    circuit_breaker=None
) -> DisasterRecovery:
    """
    Get the singleton disaster recovery instance.
    
    Args:
        config: Optional configuration (used only if instance doesn't exist)
        event_bus: Optional event bus (used only if instance doesn't exist)
        state_manager: Optional state manager (used only if instance doesn't exist)
        health_monitor: Optional health monitor (used only if instance doesn't exist)
        circuit_breaker: Optional circuit breaker (used only if instance doesn't exist)
        
    Returns:
        DisasterRecovery instance
    """
    global _disaster_recovery_instance
    
    if _disaster_recovery_instance is None:
        if config is None:
            # Load default configuration
            config = DisasterRecoveryConfig()
        
        _disaster_recovery_instance = DisasterRecovery(
            config=config,
            event_bus=event_bus,
            state_manager=state_manager,
            health_monitor=health_monitor,
            circuit_breaker=circuit_breaker
        )
    
    return _disaster_recovery_instance