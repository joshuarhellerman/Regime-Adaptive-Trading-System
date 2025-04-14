"""
state_manager.py - Centralized State Management System

This module provides a transactional state management system with support for
atomic operations, state snapshots, and persistent storage. It serves as the
single source of truth for the system's state.
"""

import copy
import json
import logging
import threading
import time
import os
import pickle
from dataclasses import dataclass, field, asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Generic, Union, Callable, cast
from uuid import UUID, uuid4

from core.event_bus import EventTopics, create_event, get_event_bus, Event, EventPriority
from data.storage.persistent_queue import PersistentQueue
from data.storage.time_series_store import TimeSeriesStore, Resolution, Aggregation

logger = logging.getLogger(__name__)

# Type variables for generic types
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


class StateTransactionError(Exception):
    """Exception raised when a state transaction fails"""
    pass


class StateAccessError(Exception):
    """Exception raised when state access is not permitted"""
    pass


@dataclass
class StateChangeEvent:
    """Event triggered when state changes"""
    path: str
    old_value: Optional[Any]
    new_value: Any
    transaction_id: UUID


class StateScope(Enum):
    """Scope of state data"""
    EPHEMERAL = "ephemeral"  # In-memory only, not persisted
    PERSISTENT = "persistent"  # Persisted to disk
    HISTORICAL = "historical"  # Stored in time series database


@dataclass
class StateTransaction:
    """Represents a transaction for atomic state changes"""
    id: UUID = field(default_factory=uuid4)
    changes: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)  # path -> (old_value, new_value)
    timestamp: float = field(default_factory=time.time)
    committed: bool = False


@dataclass
class StateConfig:
    """Configuration for state manager"""
    persistent_queue_path: str = "data/state/journal"
    time_series_path: str = "data/state/history"
    snapshot_interval: int = 100  # Take snapshot every N transactions
    snapshot_path: str = "data/state/snapshots"
    max_snapshot_count: int = 10  # Number of snapshots to keep
    max_transaction_retry: int = 3  # Maximum retries for failed transactions


class StateManager:
    """
    Centralized state management system with transactional guarantees.
    Provides a hierarchical state structure with persistence options.
    """

    def __init__(self, config: StateConfig = StateConfig()):
        """
        Initialize the state manager.

        Args:
            config: Configuration options for the state manager
        """
        self._config = config
        self._state: Dict[str, Any] = {}
        self._scopes: Dict[str, StateScope] = {}
        self._locks: Dict[str, threading.RLock] = {}
        self._global_lock = threading.RLock()
        self._transaction_lock = threading.RLock()
        self._current_transaction: Optional[StateTransaction] = None
        self._transaction_count = 0

        # Ensure directories exist
        os.makedirs(os.path.dirname(self._config.persistent_queue_path), exist_ok=True)
        os.makedirs(os.path.dirname(self._config.time_series_path), exist_ok=True)
        os.makedirs(self._config.snapshot_path, exist_ok=True)

        # Initialize storage systems
        self._journal = PersistentQueue(self._config.persistent_queue_path)
        self._time_series = TimeSeriesStore(self._config.time_series_path)

        # Track state access patterns for optimization
        self._access_stats: Dict[str, Dict[str, int]] = {}

        # Event bus for notifications
        self._event_bus = get_event_bus()

        logger.info("StateManager initialized")

    def start(self) -> None:
        """Start the state manager and recover state if needed"""
        logger.info("Starting StateManager")
        try:
            self._recover_state()
            # Register with event bus for system shutdown to ensure clean state persistence
            self._event_bus.subscribe(EventTopics.SYSTEM_SHUTDOWN, self._handle_shutdown)
            logger.info("StateManager started successfully")
        except Exception as e:
            logger.error(f"Failed to start StateManager: {e}")
            raise

    def stop(self) -> None:
        """Stop the state manager and ensure state is persisted"""
        logger.info("Stopping StateManager")
        try:
            self._persist_state()
            logger.info("StateManager stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping StateManager: {e}")
            raise

    def get(self, path: str, default: Optional[T] = None) -> Optional[T]:
        """
        Get state value at the specified path.

        Args:
            path: Dot-separated path to the state value
            default: Default value if path doesn't exist

        Returns:
            The value at the path or default if not found
        """
        with self._get_lock(path):
            self._record_access(path, "get")
            return self._get_value(path, default)

    def set(self, path: str, value: Any, scope: Optional[StateScope] = None) -> bool:
        """
        Set state value at the specified path.

        Args:
            path: Dot-separated path to the state value
            value: Value to set
            scope: Optional scope override for this value

        Returns:
            True if successful, False otherwise
        """
        with self._get_lock(path):
            old_value = self._get_value(path)
            success = self._set_value(path, value)

            if success:
                self._record_access(path, "set")

                # Update the scope if provided
                if scope is not None:
                    self._scopes[path] = scope

                # Record historical value if appropriate
                if self._should_store_history(path):
                    self._time_series.store(path, value, time.time())

                # Emit state change event
                transaction_id = uuid4() if self._current_transaction is None else self._current_transaction.id
                event = create_event(
                    EventTopics.STATE_SAVED,
                    StateChangeEvent(path, old_value, value, transaction_id)
                )
                self._event_bus.publish(event)

            return success

    def delete(self, path: str) -> bool:
        """
        Delete state value at the specified path.

        Args:
            path: Dot-separated path to the state value

        Returns:
            True if successful, False otherwise
        """
        with self._get_lock(path):
            old_value = self._get_value(path)
            if old_value is None:
                return False

            # Use nested dictionaries to track the path
            parts = path.split('.')
            current = self._state

            # Navigate to the parent container
            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    return False
                current = current[part]

                if not isinstance(current, dict):
                    return False

            # Delete the value
            if parts[-1] in current:
                del current[parts[-1]]

                # Also remove from scopes
                if path in self._scopes:
                    del self._scopes[path]

                # Emit state change event
                transaction_id = uuid4() if self._current_transaction is None else self._current_transaction.id
                event = create_event(
                    EventTopics.STATE_SAVED,
                    StateChangeEvent(path, old_value, None, transaction_id)
                )
                self._event_bus.publish(event)

                self._record_access(path, "delete")
                return True

            return False

    def set_scope(self, path: str, scope: StateScope) -> None:
        """
        Set the scope for a state path.

        Args:
            path: Dot-separated path to the state value
            scope: Scope to set for this path
        """
        with self._global_lock:
            self._scopes[path] = scope

    def get_scope(self, path: str) -> StateScope:
        """
        Get the scope for a state path.

        Args:
            path: Dot-separated path to the state value

        Returns:
            Scope of the path, defaults to EPHEMERAL
        """
        with self._global_lock:
            # Find the most specific scope that applies
            parts = path.split('.')
            for i in range(len(parts), 0, -1):
                check_path = '.'.join(parts[:i])
                if check_path in self._scopes:
                    return self._scopes[check_path]

            # Default to ephemeral if not specified
            return StateScope.EPHEMERAL

    def begin_transaction(self) -> UUID:
        """
        Begin a new transaction for atomic state changes.

        Returns:
            Transaction ID
        """
        with self._transaction_lock:
            if self._current_transaction is not None:
                raise StateTransactionError("Transaction already in progress")

            self._current_transaction = StateTransaction()
            logger.debug(f"Transaction {self._current_transaction.id} started")
            return self._current_transaction.id

    def commit_transaction(self) -> bool:
        """
        Commit the current transaction.

        Returns:
            True if successful, False otherwise
        """
        with self._transaction_lock:
            if self._current_transaction is None:
                raise StateTransactionError("No transaction in progress")

            try:
                transaction = self._current_transaction
                self._current_transaction = None

                # Mark as committed
                transaction.committed = True
                logger.debug(f"Transaction {transaction.id} committed with {len(transaction.changes)} changes")

                # Record to journal for durability
                self._journal.enqueue(asdict(transaction))

                self._transaction_count += 1

                # Check if we need to take a snapshot
                if self._transaction_count >= self._config.snapshot_interval:
                    self._create_snapshot()
                    self._transaction_count = 0

                return True
            except Exception as e:
                logger.exception(f"Failed to commit transaction: {e}")
                return False

    def rollback_transaction(self) -> bool:
        """
        Rollback the current transaction.

        Returns:
            True if successful, False otherwise
        """
        with self._transaction_lock:
            if self._current_transaction is None:
                raise StateTransactionError("No transaction in progress")

            try:
                transaction = self._current_transaction
                self._current_transaction = None

                # Revert all changes
                for path, (old_value, _) in transaction.changes.items():
                    with self._get_lock(path):
                        self._set_value(path, old_value, record_in_transaction=False)

                logger.debug(f"Transaction {transaction.id} rolled back with {len(transaction.changes)} changes reverted")
                return True
            except Exception as e:
                logger.exception(f"Failed to rollback transaction: {e}")
                return False

    def get_history(self, path: str, start_time: float, end_time: float) -> List[Tuple[float, Any]]:
        """
        Get historical values for a state path.

        Args:
            path: Dot-separated path to the state value
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            List of (timestamp, value) tuples
        """
        return self._time_series.get(path, start_time, end_time)

    def create_snapshot(self) -> bool:
        """
        Manually create a state snapshot.

        Returns:
            True if successful, False otherwise
        """
        with self._global_lock:
            return self._create_snapshot()

    def load_snapshot(self, snapshot_id: Optional[str] = None) -> bool:
        """
        Load a state snapshot.

        Args:
            snapshot_id: Specific snapshot ID to load, or latest if None

        Returns:
            True if successful, False otherwise
        """
        with self._global_lock:
            return self._load_snapshot(snapshot_id)

    def get_snapshots(self) -> List[Dict[str, Any]]:
        """
        Get list of available snapshots.

        Returns:
            List of snapshot metadata
        """
        snapshot_dir = Path(self._config.snapshot_path)
        snapshots = []

        for file in snapshot_dir.glob("*.snapshot"):
            try:
                # Extract snapshot ID from filename
                snapshot_id = file.stem

                # Load metadata
                metadata_file = snapshot_dir / f"{snapshot_id}.meta"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                    snapshots.append({
                        'id': snapshot_id,
                        'timestamp': metadata.get('timestamp', 0),
                        'transaction_id': metadata.get('transaction_id', ''),
                        'size': os.path.getsize(file)
                    })
            except Exception as e:
                logger.error(f"Error loading snapshot metadata {file}: {e}")

        # Sort by timestamp (newest first)
        return sorted(snapshots, key=lambda x: x['timestamp'], reverse=True)

    def _get_lock(self, path: str) -> threading.RLock:
        """Get or create a lock for a path"""
        with self._global_lock:
            if path not in self._locks:
                self._locks[path] = threading.RLock()
            return self._locks[path]

    def _get_value(self, path: str, default: Optional[Any] = None) -> Any:
        """Get a value from the state tree"""
        # Handle root path
        if not path:
            return self._state

        # Split path into components
        parts = path.split('.')

        # Navigate through the state tree
        current = self._state
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]

        return current

    def _set_value(self, path: str, value: Any, record_in_transaction: bool = True) -> bool:
        """Set a value in the state tree"""
        # Make a deep copy to avoid unexpected mutations
        value_copy = copy.deepcopy(value)

        # Handle root path
        if not path:
            old_value = copy.deepcopy(self._state)
            self._state = value_copy

            # Record in current transaction if applicable
            if record_in_transaction and self._current_transaction is not None:
                self._current_transaction.changes[path] = (old_value, value_copy)

            return True

        # Split path into components
        parts = path.split('.')

        # Navigate through the state tree, creating dictionaries as needed
        current = self._state
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                # Cannot navigate further
                return False

            current = current[part]

        # Get the old value if we're recording in a transaction
        old_value = None
        if record_in_transaction and self._current_transaction is not None:
            old_value = copy.deepcopy(current.get(parts[-1]))

        # Set the value
        current[parts[-1]] = value_copy

        # Record in current transaction if applicable
        if record_in_transaction and self._current_transaction is not None:
            self._current_transaction.changes[path] = (old_value, value_copy)

        return True

    def _record_access(self, path: str, access_type: str) -> None:
        """Record state access for optimization"""
        with self._global_lock:
            if path not in self._access_stats:
                self._access_stats[path] = {
                    'get': 0,
                    'set': 0,
                    'delete': 0,
                    'last_access': 0
                }

            stats = self._access_stats[path]
            stats[access_type] = stats.get(access_type, 0) + 1
            stats['last_access'] = time.time()

    def _should_store_history(self, path: str) -> bool:
        """Check if a path should be stored in history"""
        scope = self.get_scope(path)
        return scope == StateScope.HISTORICAL

    def _create_snapshot(self) -> bool:
        """Create a snapshot of the current state"""
        try:
            # Generate snapshot ID
            snapshot_id = f"{int(time.time())}-{uuid4()}"

            # Save state to snapshot file
            snapshot_file = Path(self._config.snapshot_path) / f"{snapshot_id}.snapshot"
            with open(snapshot_file, 'wb') as f:
                pickle.dump({
                    'state': self._state,
                    'scopes': self._scopes
                }, f)

            # Save metadata
            metadata_file = Path(self._config.snapshot_path) / f"{snapshot_id}.meta"
            with open(metadata_file, 'w') as f:
                json.dump({
                    'timestamp': time.time(),
                    'transaction_id': str(uuid4()),
                    'transaction_count': self._transaction_count
                }, f)

            # Limit number of snapshots
            self._cleanup_snapshots()

            logger.info(f"Created state snapshot {snapshot_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            return False

    def _load_snapshot(self, snapshot_id: Optional[str] = None) -> bool:
        """Load a snapshot"""
        try:
            snapshot_dir = Path(self._config.snapshot_path)

            # Find the snapshot file
            if snapshot_id is not None:
                snapshot_file = snapshot_dir / f"{snapshot_id}.snapshot"
                if not snapshot_file.exists():
                    logger.error(f"Snapshot {snapshot_id} not found")
                    return False
            else:
                # Find the latest snapshot
                snapshots = self.get_snapshots()
                if not snapshots:
                    logger.error("No snapshots available")
                    return False

                snapshot_id = snapshots[0]['id']
                snapshot_file = snapshot_dir / f"{snapshot_id}.snapshot"

            # Load the snapshot
            with open(snapshot_file, 'rb') as f:
                snapshot_data = pickle.load(f)

            # Replace current state
            self._state = snapshot_data.get('state', {})
            self._scopes = snapshot_data.get('scopes', {})

            # Reset transaction counter
            self._transaction_count = 0

            logger.info(f"Loaded state snapshot {snapshot_id}")

            # Emit event
            event = create_event(
                EventTopics.STATE_LOADED,
                {'snapshot_id': snapshot_id}
            )
            self._event_bus.publish(event)

            return True
        except Exception as e:
            logger.error(f"Failed to load snapshot: {e}")
            return False

    def _cleanup_snapshots(self) -> None:
        """Remove old snapshots to stay within limits"""
        snapshots = self.get_snapshots()

        # Keep only the newest ones up to max_snapshot_count
        for snapshot in snapshots[self._config.max_snapshot_count:]:
            snapshot_id = snapshot['id']
            try:
                snapshot_file = Path(self._config.snapshot_path) / f"{snapshot_id}.snapshot"
                metadata_file = Path(self._config.snapshot_path) / f"{snapshot_id}.meta"

                if snapshot_file.exists():
                    os.remove(snapshot_file)

                if metadata_file.exists():
                    os.remove(metadata_file)

                logger.debug(f"Removed old snapshot {snapshot_id}")
            except OSError as e:
                logger.warning(f"Error removing old snapshot {snapshot_id}: {e}")

    def _recover_state(self) -> None:
        """Recover state after restart"""
        try:
            # First try to load the latest snapshot
            snapshot_loaded = self._load_snapshot()

            # Then replay any journal entries after the snapshot
            journal_entries = 0

            # Process items from the journal queue
            while True:
                entry = self._journal.dequeue()
                if entry is None:
                    break

                item_id, transaction_data = entry

                try:
                    # Convert dictionary back to StateTransaction
                    transaction_id = UUID(transaction_data.get('id', str(uuid4())))
                    timestamp = transaction_data.get('timestamp', time.time())
                    committed = transaction_data.get('committed', False)

                    # Skip uncommitted transactions
                    if not committed:
                        continue

                    # Apply changes
                    changes = transaction_data.get('changes', {})
                    for path, (_, new_value) in changes.items():
                        self._set_value(path, new_value, record_in_transaction=False)

                    journal_entries += 1

                    # Acknowledge the journal entry
                    self._journal.ack(item_id)
                except Exception as e:
                    logger.error(f"Error processing journal entry: {e}")
                    # Put the entry back in the queue
                    self._journal.nack(item_id)

            if snapshot_loaded:
                logger.info(f"Recovered state from snapshot and replayed {journal_entries} journal entries")
            elif journal_entries > 0:
                logger.info(f"Recovered state from {journal_entries} journal entries")
            else:
                logger.info("No state to recover, starting with empty state")

        except Exception as e:
            logger.error(f"Error recovering state: {e}")
            # Initialize with empty state
            self._state = {}
            self._scopes = {}

    def _persist_state(self) -> None:
        """Persist state to disk before shutdown"""
        try:
            # Create a final snapshot
            self._create_snapshot()

            # Flush time series data
            self._time_series.flush()

            logger.info("Persisted state to disk")
        except Exception as e:
            logger.error(f"Error persisting state: {e}")

    def _handle_shutdown(self, event: Event) -> None:
        """Handle system shutdown event"""
        logger.info("Handling system shutdown, persisting state")
        self._persist_state()


# Add this to the end of your state_manager.py file

# Singleton instance for state manager
_state_manager_instance = None


def get_state_manager(config: Optional[StateConfig] = None) -> StateManager:
    """
    Get the singleton state manager instance.

    Args:
        config: Optional configuration (used only if instance doesn't exist)

    Returns:
        StateManager instance
    """
    global _state_manager_instance

    if _state_manager_instance is None:
        if config is None:
            # Use default configuration
            config = StateConfig()

        _state_manager_instance = StateManager(config=config)

        # Start the state manager
        _state_manager_instance.start()

    return _state_manager_instance
