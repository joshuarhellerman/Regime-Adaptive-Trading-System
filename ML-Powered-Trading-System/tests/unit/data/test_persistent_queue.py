"""
Unit tests for the PersistentQueue class.
"""

import os
import pickle
import pytest
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional, Tuple

from data.storage.persistent_queue import PersistentQueue, PersistentQueueError


class TestPersistentQueue:
    """Tests for the PersistentQueue class."""

    @pytest.fixture
    def queue_dir(self) -> str:
        """Create a temporary directory for the queue files."""
        temp_dir = tempfile.mkdtemp(prefix="pq_test_")
        yield temp_dir
        # Clean up after the test
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def queue(self, queue_dir: str) -> PersistentQueue:
        """Create a new PersistentQueue instance for testing."""
        queue = PersistentQueue(queue_dir, max_memory_items=10)
        yield queue

    def test_initialization(self, queue_dir: str) -> None:
        """Test that the queue initializes correctly."""
        queue = PersistentQueue(queue_dir)

        # Check that directories were created
        assert Path(queue_dir).exists()
        assert (Path(queue_dir) / "data").exists()
        assert (Path(queue_dir) / "tmp").exists()
        assert (Path(queue_dir) / "processing").exists()

        # Check initial state
        assert queue.size() == 0

    def test_enqueue_dequeue(self, queue: PersistentQueue) -> None:
        """Test basic enqueue and dequeue operations."""
        # Enqueue an item
        item = {"test": "data"}
        item_id = queue.enqueue(item)

        # Check size
        assert queue.size() == 1

        # Dequeue the item
        result = queue.dequeue()
        assert result is not None
        dequeued_id, dequeued_item = result

        # Verify the dequeued item
        assert dequeued_id == item_id
        assert dequeued_item == item

        # Queue should be empty now
        assert queue.size() == 0
        assert queue.dequeue() is None

    def test_enqueue_multiple_items(self, queue: PersistentQueue) -> None:
        """Test enqueueing and dequeueing multiple items in order."""
        items = [f"item-{i}" for i in range(5)]
        ids = []

        # Enqueue items
        for item in items:
            ids.append(queue.enqueue(item))

        # Check size
        assert queue.size() == len(items)

        # Dequeue items and verify order
        for i, expected_item in enumerate(items):
            result = queue.dequeue()
            assert result is not None
            dequeued_id, dequeued_item = result
            assert dequeued_id == ids[i]
            assert dequeued_item == expected_item

        # Queue should be empty now
        assert queue.size() == 0

    def test_peek(self, queue: PersistentQueue) -> None:
        """Test peeking at the next item without removing it."""
        item = "peek test"
        item_id = queue.enqueue(item)

        # Peek at the item
        result = queue.peek()
        assert result is not None
        peeked_id, peeked_item = result

        # Verify the peeked item
        assert peeked_id == item_id
        assert peeked_item == item

        # Size should still be 1
        assert queue.size() == 1

        # Dequeue should give the same item
        result = queue.dequeue()
        assert result is not None
        dequeued_id, dequeued_item = result
        assert dequeued_id == item_id
        assert dequeued_item == item

        # Now queue should be empty
        assert queue.size() == 0

    def test_acknowledgement(self, queue: PersistentQueue) -> None:
        """Test acknowledging a processed item."""
        item = "ack test"
        queue.enqueue(item)

        # Dequeue the item
        result = queue.dequeue()
        assert result is not None
        item_id, _ = result

        # Verify the item is not in the queue
        assert queue.size() == 0

        # Acknowledge the item
        assert queue.ack(item_id) is True

        # Trying to acknowledge again should fail
        assert queue.ack(item_id) is False

    def test_negative_acknowledgement(self, queue: PersistentQueue) -> None:
        """Test negatively acknowledging an item to return it to the queue."""
        item = "nack test"
        queue.enqueue(item)

        # Dequeue the item
        result = queue.dequeue()
        assert result is not None
        item_id, _ = result

        # Verify the item is not in the queue
        assert queue.size() == 0

        # Negative acknowledge the item (return it to the queue)
        assert queue.nack(item_id) is True

        # Size should be 1 again
        assert queue.size() == 1

        # Dequeue should give the same item
        result = queue.dequeue()
        assert result is not None
        dequeued_id, dequeued_item = result
        assert dequeued_item == item  # Same value
        assert dequeued_id == item_id  # Same ID

    def test_clear(self, queue: PersistentQueue) -> None:
        """Test clearing all items from the queue."""
        # Enqueue multiple items
        for i in range(5):
            queue.enqueue(f"clear-test-{i}")

        # Verify size
        assert queue.size() == 5

        # Clear the queue
        cleared = queue.clear()
        assert cleared == 5

        # Queue should be empty
        assert queue.size() == 0
        assert queue.dequeue() is None

    def test_scan(self, queue: PersistentQueue) -> None:
        """Test scanning all items in the queue without removing them."""
        items = [f"scan-test-{i}" for i in range(5)]
        ids = []

        # Enqueue items
        for item in items:
            ids.append(queue.enqueue(item))

        # Scan the queue
        scanned_items = list(queue.scan())
        assert len(scanned_items) == len(items)

        # Verify all items are present and in order
        for i, (scanned_id, scanned_item) in enumerate(scanned_items):
            assert scanned_id == ids[i]
            assert scanned_item == items[i]

        # Size should still be the same
        assert queue.size() == len(items)

    def test_persistance(self, queue_dir: str) -> None:
        """Test that items persist across queue instances."""
        # Create a queue and add items
        queue1 = PersistentQueue(queue_dir)
        items = [f"persist-test-{i}" for i in range(5)]
        ids = []

        for item in items:
            ids.append(queue1.enqueue(item))

        # Create a new queue instance pointing to the same directory
        queue2 = PersistentQueue(queue_dir)
        assert queue2.size() == len(items)

        # Dequeue items and verify
        for i, expected_item in enumerate(items):
            result = queue2.dequeue()
            assert result is not None
            dequeued_id, dequeued_item = result
            assert dequeued_id == ids[i]
            assert dequeued_item == expected_item

    def test_recovery(self, queue_dir: str) -> None:
        """Test recovery from processing state."""
        # Create a queue and process an item but don't ack it
        queue1 = PersistentQueue(queue_dir)
        item = "recovery-test"
        queue1.enqueue(item)
        queue1.dequeue()  # Item is moved to processing

        # Create a new queue instance which should recover the item
        queue2 = PersistentQueue(queue_dir)
        assert queue2.size() == 1

        # Should be able to dequeue the recovered item
        result = queue2.dequeue()
        assert result is not None
        _, dequeued_item = result
        assert dequeued_item == item

    def test_max_memory_items(self, queue_dir: str) -> None:
        """Test that items beyond max_memory_items are stored on disk."""
        max_items = 5
        queue = PersistentQueue(queue_dir, max_memory_items=max_items)

        # Enqueue more items than max_memory_items
        total_items = max_items * 2
        for i in range(total_items):
            queue.enqueue(f"memory-test-{i}")

        # All items should be dequeue-able
        for i in range(total_items):
            result = queue.dequeue()
            assert result is not None
            _, item = result
            assert item == f"memory-test-{i}"

    def test_concurrent_access(self, queue_dir: str) -> None:
        """Test concurrent access to the queue."""
        queue = PersistentQueue(queue_dir)
        num_threads = 5
        items_per_thread = 20

        def producer(thread_id: int) -> None:
            for i in range(items_per_thread):
                queue.enqueue(f"thread-{thread_id}-item-{i}")
                time.sleep(0.001)  # Small delay to increase interleaving

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=producer, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify the total number of items
        assert queue.size() == num_threads * items_per_thread

        # Dequeue all items
        dequeued = 0
        while queue.dequeue() is not None:
            dequeued += 1

        assert dequeued == num_threads * items_per_thread

    def test_invalid_ack(self, queue: PersistentQueue) -> None:
        """Test acknowledging a non-existent item."""
        assert queue.ack("non-existent-id") is False

    def test_invalid_nack(self, queue: PersistentQueue) -> None:
        """Test negatively acknowledging a non-existent item."""
        assert queue.nack("non-existent-id") is False

    def test_empty_queue_operations(self, queue: PersistentQueue) -> None:
        """Test operations on an empty queue."""
        assert queue.size() == 0
        assert queue.dequeue() is None
        assert queue.peek() is None
        assert queue.clear() == 0
        assert list(queue.scan()) == []

    def test_complex_objects(self, queue: PersistentQueue) -> None:
        """Test enqueueing and dequeueing complex Python objects."""
        class TestObject:
            def __init__(self, name: str, value: int):
                self.name = name
                self.value = value

            def __eq__(self, other):
                if not isinstance(other, TestObject):
                    return False
                return self.name == other.name and self.value == other.value

        obj = TestObject("test", 42)
        item_id = queue.enqueue(obj)

        result = queue.dequeue()
        assert result is not None
        _, dequeued_obj = result
        assert isinstance(dequeued_obj, TestObject)
        assert dequeued_obj == obj
        assert dequeued_obj.name == "test"
        assert dequeued_obj.value == 42

    def test_item_order(self, queue: PersistentQueue) -> None:
        """Test that items are dequeued in the order they were enqueued."""
        # Enqueue items with delays to ensure different timestamps
        for i in range(10):
            queue.enqueue(f"order-test-{i}")
            time.sleep(0.01)

        # Dequeue and verify order
        for i in range(10):
            result = queue.dequeue()
            assert result is not None
            _, item = result
            assert item == f"order-test-{i}"

    def test_pickle_compatibility(self, queue_dir: str, queue: PersistentQueue) -> None:
        """Test that items are correctly pickled and unpickled."""
        item = {"complex": ["data", {"with": "nesting"}]}
        item_id = queue.enqueue(item)

        # Manually verify the pickle file
        data_file = Path(queue_dir) / "data" / f"{item_id}.item"
        assert data_file.exists()

        # Read the pickled data
        with open(data_file, 'rb') as f:
            unpickled_item = pickle.load(f)

        assert unpickled_item == item

        # Dequeue should also give the correct item
        result = queue.dequeue()
        assert result is not None
        _, dequeued_item = result
        assert dequeued_item == item