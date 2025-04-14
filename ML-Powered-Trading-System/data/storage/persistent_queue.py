"""
persistent_queue.py - Reliable Persistent Queue Implementation

This module provides a durable queue that persists items to disk, ensuring
that they survive system crashes and restarts. It's used for critical
data that must not be lost, such as state changes and transactions.
"""

import json
import logging
import os
import pickle
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterator, List, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class PersistentQueueError(Exception):
    """Base exception for persistent queue errors"""
    pass


class PersistentQueue:
    """
    A persistent queue that stores items on disk to prevent data loss.
    Provides FIFO semantics with transaction support.
    """

    def __init__(self, directory: str, max_memory_items: int = 1000):
        """
        Initialize the persistent queue.

        Args:
            directory: Directory where queue files will be stored
            max_memory_items: Maximum number of items to keep in memory
        """
        self._directory = Path(directory)
        self._max_memory_items = max_memory_items
        self._memory_queue: Deque[Tuple[str, Any]] = deque()
        self._lock = threading.RLock()
        self._next_file_id = 0
        self._current_size = 0

        # Ensure directory exists
        os.makedirs(self._directory, exist_ok=True)

        # Create specific directories for queue
        self._data_dir = self._directory / "data"
        self._tmp_dir = self._directory / "tmp"
        self._processing_dir = self._directory / "processing"

        os.makedirs(self._data_dir, exist_ok=True)
        os.makedirs(self._tmp_dir, exist_ok=True)
        os.makedirs(self._processing_dir, exist_ok=True)

        # Recover any existing queue items
        self._recover()

        logger.info(f"Persistent queue initialized at {directory}")

    def enqueue(self, item: Any) -> str:
        """
        Add an item to the queue.

        Args:
            item: The item to enqueue

        Returns:
            Item ID that can be used for acknowledgement
        """
        item_id = f"{int(time.time() * 1000)}_{self._next_file_id}"
        self._next_file_id += 1

        with self._lock:
            # Save to disk first for durability
            self._persist_item(item_id, item)

            # Then add to memory queue if not full
            if len(self._memory_queue) < self._max_memory_items:
                self._memory_queue.append((item_id, item))

            self._current_size += 1

        return item_id

    def dequeue(self) -> Optional[Tuple[str, Any]]:
        """
        Remove and return the next item from the queue.

        Returns:
            Tuple of (item_id, item) or None if the queue is empty
        """
        with self._lock:
            if self._current_size == 0:
                return None

            if self._memory_queue:
                # If we have items in memory, use those
                item_id, item = self._memory_queue.popleft()
            else:
                # Otherwise load from disk
                item_id, item = self._load_next_item()

            if item is not None:
                # Move to processing directory
                self._move_to_processing(item_id)
                self._current_size -= 1

            return (item_id, item) if item is not None else None

    def peek(self) -> Optional[Tuple[str, Any]]:
        """
        Return the next item without removing it.

        Returns:
            Tuple of (item_id, item) or None if the queue is empty
        """
        with self._lock:
            if self._current_size == 0:
                return None

            if self._memory_queue:
                # If we have items in memory, use those
                item_id, item = self._memory_queue[0]
            else:
                # Otherwise load from disk (but don't move to processing)
                item_id, item = self._load_next_item()

            return (item_id, item) if item is not None else None

    def ack(self, item_id: str) -> bool:
        """
        Acknowledge that an item has been processed and can be removed.

        Args:
            item_id: ID of the item to acknowledge

        Returns:
            True if the item was acknowledged, False otherwise
        """
        with self._lock:
            processing_file = self._processing_dir / f"{item_id}.item"
            if processing_file.exists():
                try:
                    os.remove(processing_file)
                    return True
                except OSError as e:
                    logger.error(f"Error acknowledging item {item_id}: {e}")
                    return False
            return False

    def nack(self, item_id: str) -> bool:
        """
        Negative acknowledge an item, returning it to the queue.

        Args:
            item_id: ID of the item to return to the queue

        Returns:
            True if the item was returned to the queue, False otherwise
        """
        with self._lock:
            processing_file = self._processing_dir / f"{item_id}.item"
            if processing_file.exists():
                try:
                    # Load the item
                    with open(processing_file, 'rb') as f:
                        item = pickle.load(f)

                    # Move back to data directory
                    os.rename(processing_file, self._data_dir / f"{item_id}.item")

                    # Add back to memory queue if not full
                    if len(self._memory_queue) < self._max_memory_items:
                        self._memory_queue.append((item_id, item))

                    self._current_size += 1
                    return True
                except Exception as e:
                    logger.error(f"Error returning item {item_id} to queue: {e}")
                    return False
            return False

    def size(self) -> int:
        """
        Get the current size of the queue.

        Returns:
            Number of items in the queue
        """
        with self._lock:
            return self._current_size

    def clear(self) -> int:
        """
        Clear all items from the queue.

        Returns:
            Number of items cleared
        """
        with self._lock:
            count = self._current_size

            # Clear memory queue
            self._memory_queue.clear()

            # Clear disk queue
            for directory in [self._data_dir, self._processing_dir, self._tmp_dir]:
                for file in directory.glob("*.item"):
                    try:
                        os.remove(file)
                    except OSError as e:
                        logger.warning(f"Error removing queue file {file}: {e}")

            self._current_size = 0
            return count

    def scan(self) -> Iterator[Tuple[str, Any]]:
        """
        Scan all items in the queue without removing them.

        Returns:
            Iterator yielding (item_id, item) tuples
        """
        # First yield items in memory
        for item_id, item in self._memory_queue:
            yield (item_id, item)

        # Then scan items on disk
        with self._lock:
            # Get all files in data directory
            data_files = sorted(self._data_dir.glob("*.item"))

            # Process each file
            for file in data_files:
                item_id = file.stem
                if (item_id, None) not in self._memory_queue and file.exists():
                    try:
                        with open(file, 'rb') as f:
                            item = pickle.load(f)
                        yield (item_id, item)
                    except Exception as e:
                        logger.error(f"Error reading queue item {file}: {e}")

    def _persist_item(self, item_id: str, item: Any) -> bool:
        """
        Persist an item to disk.

        Args:
            item_id: ID of the item
            item: The item to persist

        Returns:
            True if successful, False otherwise
        """
        try:
            # Write to temporary file first
            tmp_file = self._tmp_dir / f"{item_id}.item"
            with open(tmp_file, 'wb') as f:
                pickle.dump(item, f)

            # Then move to data directory (atomic operation on most file systems)
            os.rename(tmp_file, self._data_dir / f"{item_id}.item")
            return True
        except Exception as e:
            logger.error(f"Error persisting queue item {item_id}: {e}")
            return False

    def _load_next_item(self) -> Tuple[str, Any]:
        """
        Load the next item from disk.

        Returns:
            Tuple of (item_id, item)
        """
        # Get all files in data directory
        data_files = sorted(self._data_dir.glob("*.item"))

        if not data_files:
            return ("", None)

        # Get the first file
        file = data_files[0]
        item_id = file.stem

        try:
            with open(file, 'rb') as f:
                item = pickle.load(f)
            return (item_id, item)
        except Exception as e:
            logger.error(f"Error loading queue item {file}: {e}")
            # Remove corrupted file
            try:
                os.remove(file)
            except OSError:
                pass
            return ("", None)

    def _move_to_processing(self, item_id: str) -> bool:
        """
        Move an item from the data directory to the processing directory.

        Args:
            item_id: ID of the item to move

        Returns:
            True if successful, False otherwise
        """
        try:
            data_file = self._data_dir / f"{item_id}.item"
            processing_file = self._processing_dir / f"{item_id}.item"

            if data_file.exists():
                os.rename(data_file, processing_file)
                return True
            return False
        except Exception as e:
            logger.error(f"Error moving queue item {item_id} to processing: {e}")
            return False

    def _recover(self) -> None:
        """
        Recover the queue state from disk after a restart.
        """
        try:
            recovered = 0

            # First check for any items in the processing directory and move them back to data
            for file in self._processing_dir.glob("*.item"):
                try:
                    os.rename(file, self._data_dir / file.name)
                    recovered += 1
                except OSError as e:
                    logger.warning(f"Error recovering processing item {file}: {e}")

            # Remove any temporary files
            for file in self._tmp_dir.glob("*.item"):
                try:
                    os.remove(file)
                except OSError:
                    pass

            # Load items into memory queue
            data_files = sorted(self._data_dir.glob("*.item"))

            for i, file in enumerate(data_files):
                if i >= self._max_memory_items:
                    break

                item_id = file.stem
                try:
                    with open(file, 'rb') as f:
                        item = pickle.load(f)
                    self._memory_queue.append((item_id, item))
                except Exception as e:
                    logger.error(f"Error loading queue item {file} into memory: {e}")

            # Update current size
            self._current_size = len(list(self._data_dir.glob("*.item")))

            # Update next file ID
            if data_files:
                last_id = data_files[-1].stem
                try:
                    self._next_file_id = int(last_id.split('_')[-1]) + 1
                except (ValueError, IndexError):
                    self._next_file_id = len(data_files)

            if recovered > 0:
                logger.info(f"Recovered {recovered} items during queue initialization")

            logger.info(f"Persistent queue recovered with {self._current_size} items ({len(self._memory_queue)} in memory)")

        except Exception as e:
            logger.error(f"Error recovering queue state: {e}")
            # Reset to safe state
            self._memory_queue.clear()
            self._current_size = 0
            self._next_file_id = 0