"""
concurrency_utils.py - Concurrency Utilities for Trading System

This module provides utilities for managing concurrent operations in the trading system,
including thread pools, locks, synchronization primitives, and asynchronous task management.
It abstracts away the details of Python's threading and asyncio libraries and provides
a consistent interface for concurrency management throughout the system.
"""

import asyncio
import concurrent.futures
import functools
import inspect
import logging
import queue
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union, Generic, cast

logger = logging.getLogger(__name__)

# Type variables for generic type hints
T = TypeVar('T')
R = TypeVar('R')


class ThreadPriority(Enum):
    """Priority levels for thread execution"""
    CRITICAL = 0  # Highest priority, for time-sensitive operations
    HIGH = 1      # High priority operations
    NORMAL = 2    # Default priority
    LOW = 3       # Background operations
    IDLE = 4      # Only when system is idle


class ConcurrencyMode(Enum):
    """Execution mode for concurrent operations"""
    SYNC = "sync"        # Synchronous execution in the current thread
    THREAD = "thread"    # Execute in a separate thread
    PROCESS = "process"  # Execute in a separate process
    ASYNC = "async"      # Execute using asyncio


class AcquireResult(Enum):
    """Result of a lock acquisition attempt"""
    ACQUIRED = "acquired"  # Lock was acquired
    TIMEOUT = "timeout"    # Acquisition timed out
    ERROR = "error"        # Error occurred during acquisition


class ParallelResult(Generic[T]):
    """Result from a parallel execution"""
    def __init__(self, result: Optional[T] = None, error: Optional[Exception] = None):
        self.result = result
        self.error = error
        self.success = error is None
        self.execution_time: Optional[float] = None

    @property
    def is_success(self) -> bool:
        """Check if the execution was successful"""
        return self.success

    def __str__(self) -> str:
        if self.success:
            return f"Success: {self.result} (time: {self.execution_time:.6f}s)"
        return f"Error: {self.error} (time: {self.execution_time:.6f}s)"


class ThreadPoolManager:
    """
    Manages thread pools with different priorities.

    This class provides separate thread pools for different priority levels
    to ensure that high-priority tasks are not blocked by lower-priority ones.
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> 'ThreadPoolManager':
        """Get the singleton instance of the thread pool manager"""
        if cls._instance is None:
            cls._instance = ThreadPoolManager()
        return cls._instance

    def __init__(self):
        """Initialize the thread pool manager"""
        # Configure thread pools for different priority levels
        self._pools: Dict[ThreadPriority, concurrent.futures.ThreadPoolExecutor] = {}
        self._pool_configs = {
            ThreadPriority.CRITICAL: {"max_workers": 4, "thread_name_prefix": "critical"},
            ThreadPriority.HIGH: {"max_workers": 8, "thread_name_prefix": "high"},
            ThreadPriority.NORMAL: {"max_workers": 12, "thread_name_prefix": "normal"},
            ThreadPriority.LOW: {"max_workers": 6, "thread_name_prefix": "low"},
            ThreadPriority.IDLE: {"max_workers": 2, "thread_name_prefix": "idle"}
        }

        # Create thread pools
        for priority, config in self._pool_configs.items():
            self._pools[priority] = concurrent.futures.ThreadPoolExecutor(**config)

        # Thread local storage for thread metadata
        self._thread_local = threading.local()

        # Metrics
        self._metrics: Dict[ThreadPriority, Dict[str, int]] = {
            priority: {"submitted": 0, "completed": 0, "errors": 0}
            for priority in ThreadPriority
        }
        self._metrics_lock = threading.RLock()

        logger.info("ThreadPoolManager initialized")

    def submit(self,
              func: Callable[..., T],
              *args,
              priority: ThreadPriority = ThreadPriority.NORMAL,
              **kwargs) -> concurrent.futures.Future:
        """
        Submit a function for execution in a thread pool.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            priority: Execution priority
            **kwargs: Keyword arguments for the function

        Returns:
            Future representing the execution
        """
        # Update metrics
        with self._metrics_lock:
            self._metrics[priority]["submitted"] += 1

        # Wrap the function to set thread local data and update metrics
        @functools.wraps(func)
        def _wrapped_func(*args, **kwargs):
            # Set thread metadata
            self._thread_local.priority = priority
            self._thread_local.start_time = time.time()
            self._thread_local.function_name = func.__name__

            try:
                # Execute the function
                result = func(*args, **kwargs)

                # Update metrics
                with self._metrics_lock:
                    self._metrics[priority]["completed"] += 1

                return result
            except Exception as e:
                # Update metrics
                with self._metrics_lock:
                    self._metrics[priority]["errors"] += 1

                # Log the error
                logger.error(f"Error in thread pool task: {e}", exc_info=True)

                # Re-raise the exception
                raise

        # Submit to the appropriate thread pool
        return self._pools[priority].submit(_wrapped_func, *args, **kwargs)

    def submit_task(self,
                  func: Callable[..., T],
                  *args,
                  priority: ThreadPriority = ThreadPriority.NORMAL,
                  callback: Optional[Callable[[T], None]] = None,
                  error_callback: Optional[Callable[[Exception], None]] = None,
                  **kwargs) -> concurrent.futures.Future:
        """
        Submit a task with callback handling.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            priority: Execution priority
            callback: Function to call with the result
            error_callback: Function to call if an error occurs
            **kwargs: Keyword arguments for the function

        Returns:
            Future representing the execution
        """
        future = self.submit(func, *args, priority=priority, **kwargs)

        # Add callback if provided
        if callback or error_callback:
            def _callback_wrapper(future):
                try:
                    result = future.result()
                    if callback:
                        callback(result)
                except Exception as e:
                    if error_callback:
                        error_callback(e)
                    else:
                        logger.error(f"Unhandled error in thread pool task: {e}", exc_info=True)

            future.add_done_callback(_callback_wrapper)

        return future

    def map(self,
           func: Callable[[T], R],
           items: List[T],
           priority: ThreadPriority = ThreadPriority.NORMAL,
           timeout: Optional[float] = None) -> List[R]:
        """
        Apply a function to each item in a list using a thread pool.

        Args:
            func: Function to apply
            items: List of items
            priority: Execution priority
            timeout: Maximum time to wait for completion (None for no timeout)

        Returns:
            List of results
        """
        return list(self._pools[priority].map(func, items, timeout=timeout))

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the thread pools.

        Args:
            wait: Whether to wait for pending tasks to complete
        """
        for priority, pool in self._pools.items():
            logger.debug(f"Shutting down {priority.name} thread pool")
            pool.shutdown(wait=wait)

        logger.info("All thread pools shut down")

    def get_metrics(self) -> Dict[str, Dict[str, int]]:
        """
        Get thread pool metrics.

        Returns:
            Dictionary of metrics for each priority level
        """
        with self._metrics_lock:
            return {priority.name: metrics.copy() for priority, metrics in self._metrics.items()}

    def get_current_thread_info(self) -> Dict[str, Any]:
        """
        Get information about the current thread.

        Returns:
            Dictionary with thread information
        """
        info = {
            "thread_id": threading.get_ident(),
            "thread_name": threading.current_thread().name
        }

        # Add thread local data if available
        if hasattr(self._thread_local, "priority"):
            info["priority"] = self._thread_local.priority.name
        if hasattr(self._thread_local, "start_time"):
            info["start_time"] = self._thread_local.start_time
            info["elapsed_time"] = time.time() - self._thread_local.start_time
        if hasattr(self._thread_local, "function_name"):
            info["function_name"] = self._thread_local.function_name

        return info


class ProcessPoolManager:
    """
    Manages process pools for CPU-bound tasks.

    This class provides a process pool for executing CPU-intensive tasks
    that benefit from parallel execution across multiple processors.
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> 'ProcessPoolManager':
        """Get the singleton instance of the process pool manager"""
        if cls._instance is None:
            cls._instance = ProcessPoolManager()
        return cls._instance

    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize the process pool manager.

        Args:
            max_workers: Maximum number of worker processes (None for CPU count)
        """
        self._pool = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        self._metrics = {"submitted": 0, "completed": 0, "errors": 0}
        self._metrics_lock = threading.RLock()

        logger.info(f"ProcessPoolManager initialized with {max_workers or 'CPU count'} workers")

    def submit(self, func: Callable[..., T], *args, **kwargs) -> concurrent.futures.Future:
        """
        Submit a function for execution in the process pool.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Future representing the execution
        """
        # Update metrics
        with self._metrics_lock:
            self._metrics["submitted"] += 1

        # Wrap the function to update metrics
        @functools.wraps(func)
        def _wrapped_func(*args, **kwargs):
            try:
                # Execute the function
                result = func(*args, **kwargs)

                # Update metrics - note that this happens in the worker process
                # and won't affect the metrics in the main process
                return result
            except Exception as e:
                # Log the error
                logger.error(f"Error in process pool task: {e}", exc_info=True)

                # Re-raise the exception
                raise

        # Submit to the process pool
        future = self._pool.submit(_wrapped_func, *args, **kwargs)

        # Add callback to update metrics in the main process
        def _update_metrics(future):
            with self._metrics_lock:
                if future.exception() is None:
                    self._metrics["completed"] += 1
                else:
                    self._metrics["errors"] += 1

        future.add_done_callback(_update_metrics)

        return future

    def map(self, func: Callable[[T], R], items: List[T], timeout: Optional[float] = None) -> List[R]:
        """
        Apply a function to each item in a list using the process pool.

        Args:
            func: Function to apply
            items: List of items
            timeout: Maximum time to wait for completion (None for no timeout)

        Returns:
            List of results
        """
        # Update metrics
        with self._metrics_lock:
            self._metrics["submitted"] += len(items)

        # Execute the map operation
        results = list(self._pool.map(func, items, timeout=timeout))

        # Update metrics
        with self._metrics_lock:
            self._metrics["completed"] += len(items)

        return results

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the process pool.

        Args:
            wait: Whether to wait for pending tasks to complete
        """
        logger.debug("Shutting down process pool")
        self._pool.shutdown(wait=wait)
        logger.info("Process pool shut down")

    def get_metrics(self) -> Dict[str, int]:
        """
        Get process pool metrics.

        Returns:
            Dictionary of metrics
        """
        with self._metrics_lock:
            return self._metrics.copy()


class AsyncTaskManager:
    """
    Manages asynchronous tasks using asyncio.

    This class provides utilities for running and managing asyncio tasks,
    including running them from synchronous code.
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> 'AsyncTaskManager':
        """Get the singleton instance of the async task manager"""
        if cls._instance is None:
            cls._instance = AsyncTaskManager()
        return cls._instance

    def __init__(self):
        """Initialize the async task manager"""
        self._loop = None
        self._running_tasks: Set[asyncio.Task] = set()
        self._loop_thread = None
        self._running = False
        self._lock = threading.RLock()
        self._metrics = {"submitted": 0, "completed": 0, "errors": 0}
        self._metrics_lock = threading.RLock()

        logger.info("AsyncTaskManager initialized")

    def start(self) -> None:
        """Start the asyncio event loop in a separate thread"""
        with self._lock:
            if self._running:
                logger.warning("AsyncTaskManager already running")
                return

            self._running = True
            self._loop_thread = threading.Thread(
                target=self._run_event_loop,
                name="AsyncEventLoop",
                daemon=True
            )
            self._loop_thread.start()

            # Wait for event loop to start
            while self._loop is None:
                time.sleep(0.01)

            logger.info("AsyncTaskManager started")

    def _run_event_loop(self) -> None:
        """Run the asyncio event loop"""
        try:
            # Create a new event loop
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            # Run the event loop
            self._loop.run_forever()
        except Exception as e:
            logger.error(f"Error in async event loop: {e}", exc_info=True)
        finally:
            # Clean up
            if self._loop:
                self._loop.close()
            self._loop = None
            self._running = False
            logger.info("AsyncTaskManager event loop stopped")

    def stop(self) -> None:
        """Stop the asyncio event loop"""
        with self._lock:
            if not self._running:
                logger.warning("AsyncTaskManager not running")
                return

            self._running = False

            # Cancel all running tasks
            if self._loop:
                asyncio.run_coroutine_threadsafe(self._cancel_all_tasks(), self._loop)

                # Stop the event loop
                self._loop.call_soon_threadsafe(self._loop.stop)

            # Wait for the loop thread to finish
            if self._loop_thread and self._loop_thread.is_alive():
                self._loop_thread.join(timeout=5.0)

            logger.info("AsyncTaskManager stopped")

    async def _cancel_all_tasks(self) -> None:
        """Cancel all running tasks"""
        tasks = [task for task in self._running_tasks]
        for task in tasks:
            if not task.done():
                task.cancel()

        # Wait for all tasks to complete or be cancelled
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._running_tasks.clear()

    def run_coroutine(self, coro: Callable[..., T], *args, **kwargs) -> concurrent.futures.Future:
        """
        Run a coroutine function from synchronous code.

        Args:
            coro: Coroutine function to execute
            *args: Positional arguments for the coroutine
            **kwargs: Keyword arguments for the coroutine

        Returns:
            Future representing the execution
        """
        if not self._running or not self._loop:
            raise RuntimeError("AsyncTaskManager not running")

        # Create the coroutine
        if inspect.iscoroutinefunction(coro):
            coroutine = coro(*args, **kwargs)
        else:
            raise TypeError("Expected a coroutine function")

        # Update metrics
        with self._metrics_lock:
            self._metrics["submitted"] += 1

        # Wrap the coroutine to update metrics
        async def _wrapped_coro():
            try:
                result = await coroutine

                # Update metrics
                with self._metrics_lock:
                    self._metrics["completed"] += 1

                return result
            except Exception as e:
                # Update metrics
                with self._metrics_lock:
                    self._metrics["errors"] += 1

                # Log the error
                logger.error(f"Error in async task: {e}", exc_info=True)

                # Re-raise the exception
                raise

        # Submit the coroutine to the event loop
        return asyncio.run_coroutine_threadsafe(_wrapped_coro(), self._loop)

    def create_task(self, coro: Callable[..., T], *args, **kwargs) -> asyncio.Task:
        """
        Create an asyncio task.

        This method should be called from within an async context.

        Args:
            coro: Coroutine function to execute
            *args: Positional arguments for the coroutine
            **kwargs: Keyword arguments for the coroutine

        Returns:
            Asyncio Task object
        """
        if not self._running or not self._loop:
            raise RuntimeError("AsyncTaskManager not running")

        # Create the coroutine
        if inspect.iscoroutinefunction(coro):
            coroutine = coro(*args, **kwargs)
        else:
            raise TypeError("Expected a coroutine function")

        # Update metrics
        with self._metrics_lock:
            self._metrics["submitted"] += 1

        # Wrap the coroutine to update metrics
        async def _wrapped_coro():
            try:
                result = await coroutine

                # Update metrics
                with self._metrics_lock:
                    self._metrics["completed"] += 1

                return result
            except Exception as e:
                # Update metrics
                with self._metrics_lock:
                    self._metrics["errors"] += 1

                # Log the error
                logger.error(f"Error in async task: {e}", exc_info=True)

                # Re-raise the exception
                raise
            finally:
                # Remove task from running tasks
                task = asyncio.current_task()
                if task in self._running_tasks:
                    self._running_tasks.remove(task)

        # Create and schedule the task
        task = self._loop.create_task(_wrapped_coro())
        self._running_tasks.add(task)

        return task

    def get_metrics(self) -> Dict[str, int]:
        """
        Get async task metrics.

        Returns:
            Dictionary of metrics
        """
        with self._metrics_lock:
            metrics = self._metrics.copy()
            metrics["running"] = len(self._running_tasks)
            return metrics

    def is_running(self) -> bool:
        """
        Check if the async task manager is running.

        Returns:
            Whether the async task manager is running
        """
        return self._running


@contextmanager
def acquire_timeout(lock: threading.Lock, timeout: float) -> AcquireResult:
    """
    Context manager for acquiring a lock with timeout.

    Args:
        lock: Lock to acquire
        timeout: Maximum time to wait for acquisition in seconds

    Yields:
        AcquireResult indicating whether the lock was acquired
    """
    result = AcquireResult.ERROR

    try:
        # Attempt to acquire the lock
        acquired = lock.acquire(blocking=True, timeout=timeout)

        if acquired:
            result = AcquireResult.ACQUIRED
            yield result
        else:
            result = AcquireResult.TIMEOUT
            yield result
    except Exception as e:
        logger.error(f"Error acquiring lock: {e}", exc_info=True)
        result = AcquireResult.ERROR
        yield result
    finally:
        # Release the lock if it was acquired
        if result == AcquireResult.ACQUIRED:
            lock.release()


class RWLock:
    """
    Reader-writer lock implementation.

    This class provides a lock that allows multiple concurrent readers
    but only one exclusive writer.
    """

    def __init__(self):
        """Initialize the reader-writer lock"""
        self._lock = threading.RLock()
        self._readers = 0
        self._writer = False
        self._reader_event = threading.Event()
        self._writer_event = threading.Event()
        self._reader_event.set()
        self._writer_event.set()

    def acquire_read(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a read lock.

        Args:
            timeout: Maximum time to wait for acquisition in seconds

        Returns:
            Whether the lock was acquired
        """
        # Wait for any active writer to finish
        if not self._reader_event.wait(timeout):
            return False

        with self._lock:
            self._readers += 1

            # If this is the first reader, block writers
            if self._readers == 1:
                self._writer_event.clear()

        return True

    def release_read(self) -> None:
        """Release a read lock"""
        with self._lock:
            self._readers -= 1

            # If this is the last reader, allow writers
            if self._readers == 0:
                self._writer_event.set()

    def acquire_write(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a write lock.

        Args:
            timeout: Maximum time to wait for acquisition in seconds

        Returns:
            Whether the lock was acquired
        """
        # Wait for any active readers to finish
        if not self._writer_event.wait(timeout):
            return False

        with self._lock:
            # Block new readers
            self._reader_event.clear()
            self._writer = True

        return True

    def release_write(self) -> None:
        """Release a write lock"""
        with self._lock:
            self._writer = False

            # Allow readers
            self._reader_event.set()

    @contextmanager
    def read_locked(self, timeout: Optional[float] = None) -> bool:
        """
        Context manager for read lock.

        Args:
            timeout: Maximum time to wait for acquisition in seconds

        Yields:
            Whether the lock was acquired
        """
        acquired = self.acquire_read(timeout)
        try:
            yield acquired
        finally:
            if acquired:
                self.release_read()

    @contextmanager
    def write_locked(self, timeout: Optional[float] = None) -> bool:
        """
        Context manager for write lock.

        Args:
            timeout: Maximum time to wait for acquisition in seconds

        Yields:
            Whether the lock was acquired
        """
        acquired = self.acquire_write(timeout)
        try:
            yield acquired
        finally:
            if acquired:
                self.release_write()


class Barrier:
    """
    Barrier synchronization primitive.

    This class provides a synchronization point where multiple threads
    can wait until all threads have reached the barrier.
    """

    def __init__(self, parties: int, timeout: Optional[float] = None, action: Optional[Callable[[], None]] = None):
        """
        Initialize the barrier.

        Args:
            parties: Number of threads to wait for
            timeout: Maximum time to wait for all threads
            action: Function to call when the barrier is complete
        """
        self._parties = parties
        self._timeout = timeout
        self._action = action
        self._count = 0
        self._generation = 0
        self._lock = threading.RLock()
        self._event = threading.Event()
        self._broken = False

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all threads to reach the barrier.

        Args:
            timeout: Maximum time to wait (overrides the default)

        Returns:
            Whether all threads reached the barrier
        """
        if self._broken:
            raise BrokenBarrierError("Barrier is broken")

        with self._lock:
            generation = self._generation

            # Increment the count of waiting threads
            self._count += 1

            # If all threads have arrived
            if self._count == self._parties:
                # Reset the barrier
                self._count = 0
                self._generation += 1
                self._event.set()

                # Call the action if provided
                if self._action:
                    try:
                        self._action()
                    except Exception as e:
                        logger.error(f"Error in barrier action: {e}", exc_info=True)
                        self._broken = True
                        raise
            else:
                # Wait for all threads to arrive
                self._event.clear()

                # Wait with timeout
                wait_timeout = timeout if timeout is not None else self._timeout
                if not self._event.wait(wait_timeout):
                    with self._lock:
                        # If the event is still not set, the barrier has timed out
                        if generation == self._generation:
                            self._broken = True
                            return False

        return True

    def reset(self) -> None:
        """Reset the barrier to its initial state"""
        with self._lock:
            self._count = 0
            self._generation += 1
            self._event.set()
            self._broken = False

    def abort(self) -> None:
        """Mark the barrier as broken"""
        with self._lock:
            self._broken = True
            self._event.set()

    @property
    def parties(self) -> int:
        """Get the number of threads to wait for"""
        return self._parties

    @property
    def waiting(self) -> int:
        """Get the number of threads currently waiting"""
        with self._lock:
            return self._count

    @property
    def broken(self) -> bool:
        """Check if the barrier is broken"""
        return self._broken


class BrokenBarrierError(Exception):
    """Exception raised when a barrier is broken"""
    pass


class TimedSet(Generic[T]):
    """
    Set with element expiration.

    This class provides a set that automatically removes elements
    after a specified time.
    """

    def __init__(self, expiration_time: float):
        """
        Initialize the timed set.

        Args:
            expiration_time: Time in seconds after which elements expire
        """
        self._data: Dict[T, float] = {}
        self._expiration_time = expiration_time
        self._lock = threading.RLock()

        # Start cleanup thread
        self._running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="TimedSetCleanup",
            daemon=True
        )
        self._cleanup_thread.start()

    def add(self, item: T) -> None:
        """
        Add an item to the set.

        Args:
            item: Item to add
        """
        with self._lock:
            self._data[item] = time.time()

    def remove(self, item: T) -> bool:
        """
        Remove an item from the set.

        Args:
            item: Item to remove

        Returns:
            Whether the item was removed
        """
        with self._lock:
            if item in self._data:
                del self._data[item]
                return True
            return False

    def contains(self, item: T) -> bool:
        """
        Check if an item is in the set.

        Args:
            item: Item to check

        Returns:
            Whether the item is in the set
        """
        with self._lock:
            if item in self._data:
                # Check if the item has expired
                if time.time() - self._data[item] > self._expiration_time:
                    del self._data[item]
                    return False
                return True
            return False

    def clear(self) -> None:
        """Clear the set"""
        with self._lock:
            self._data.clear()

    def size(self) -> int:
        """
        Get the size of the set.

        Returns:
            Number of items in the set
        """
        with self._lock:
            return len(self._data)

    def _cleanup_loop(self) -> None:
        """Periodically clean up expired items"""
        while self._running:
            time.sleep(min(1.0, self._expiration_time / 2))

            with self._lock:
                # Current time
                now = time.time()

                # Items to remove
                expired = [
                    item for item, timestamp in self._data.items()
                    if now - timestamp > self._expiration_time
                ]

                # Remove expired items
                for item in expired:
                    del self._data[item]

    def shutdown(self) -> None:
        """Shut down the timed set"""
        self._running = False
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=1.0)


class TimedQueue(Generic[T]):
    """
    Queue with element expiration.

    This class provides a queue that automatically removes elements
    after a specified time.
    """

    def __init__(self, expiration_time: float, maxsize: int = 0):
        """
        Initialize the timed queue.

        Args:
            expiration_time: Time in seconds after which elements expire
            maxsize: Maximum size of the queue (0 for unlimited)
        """
        self._queue = queue.Queue(maxsize)
        self._timestamps: Dict[int, float] = {}
        self._expiration_time = expiration_time
        self._lock = threading.RLock()
        self._counter = 0

        # Start cleanup thread
        self._running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="TimedQueueCleanup",
            daemon=True
        )
        self._cleanup_thread.start()

    def put(self, item: T, block: bool = True, timeout: Optional[float] = None) -> None:
        """
        Put an item into the queue.

        Args:
            item: Item to add
            block: Whether to block if the queue is full
            timeout: Maximum time to wait for space in the queue
        """
        with self._lock:
            # Generate a unique ID for the item
            item_id = self._counter
            self._counter += 1

            # Store the timestamp
            self._timestamps[item_id] = time.time()

        # Put the item and its ID in the queue
        self._queue.put((item_id, item), block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[T]:
        """
        Get an item from the queue.

        Args:
            block: Whether to block if the queue is empty
            timeout: Maximum time to wait for an item

        Returns:
            An item from the queue, or None if timed out
        """
        try:
            # Get an item and its ID from the queue
            item_id, item = self._queue.get(block=block, timeout=timeout)

            with self._lock:
                # Check if the item has expired
                if item_id in self._timestamps:
                    timestamp = self._timestamps[item_id]
                    del self._timestamps[item_id]

                    if time.time() - timestamp > self._expiration_time:
                        # Item has expired, try to get another one
                        return self.get(block=block, timeout=timeout)

                    return item
                else:
                    # Item ID not found, try to get another one
                    return self.get(block=block, timeout=timeout)
        except queue.Empty:
            # Queue is empty or timed out
            return None

    def _cleanup_loop(self) -> None:
        """Periodically clean up expired items"""
        while self._running:
            time.sleep(min(1.0, self._expiration_time / 2))

            with self._lock:
                # Current time
                now = time.time()

                # Find expired items
                expired_ids = [
                    item_id for item_id, timestamp in self._timestamps.items()
                    if now - timestamp > self._expiration_time
                ]

                # Remove expired timestamps
                for item_id in expired_ids:
                    del self._timestamps[item_id]

    def shutdown(self) -> None:
        """Shut down the timed queue"""
        self._running = False
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=1.0)

    def size(self) -> int:
        """
        Get the size of the queue.

        Returns:
            Number of items in the queue
        """
        return self._queue.qsize()

    def empty(self) -> bool:
        """
        Check if the queue is empty.

        Returns:
            Whether the queue is empty
        """
        return self._queue.empty()

    def full(self) -> bool:
        """
        Check if the queue is full.

        Returns:
            Whether the queue is full
        """
        return self._queue.full()


class ResultCache(Generic[T, R]):
    """
    Cache for function results with expiration.

    This class provides a cache for function results that expire after
    a specified time, allowing for efficient reuse of computed values.
    """

    def __init__(self, expiration_time: float, max_size: int = 1000):
        """
        Initialize the result cache.

        Args:
            expiration_time: Time in seconds after which results expire
            max_size: Maximum number of results to cache
        """
        self._cache: Dict[Any, Tuple[R, float]] = {}
        self._expiration_time = expiration_time
        self._max_size = max_size
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

        # Start cleanup thread
        self._running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="ResultCacheCleanup",
            daemon=True
        )
        self._cleanup_thread.start()

    def get(self, key: Any, compute_func: Callable[[], R] = None) -> Optional[R]:
        """
        Get a result from the cache, computing it if necessary.

        Args:
            key: Cache key
            compute_func: Function to compute the result if not cached

        Returns:
            Cached or computed result, or None if not found and no compute function
        """
        with self._lock:
            # Check if the key is in the cache
            if key in self._cache:
                result, timestamp = self._cache[key]

                # Check if the result has expired
                if time.time() - timestamp > self._expiration_time:
                    # Result has expired
                    del self._cache[key]
                    self._misses += 1
                else:
                    # Result is valid
                    self._hits += 1
                    return result
            else:
                self._misses += 1

            # If a compute function is provided, compute the result
            if compute_func:
                result = compute_func()
                self.put(key, result)
                return result

            return None

    def put(self, key: Any, result: R) -> None:
        """
        Put a result in the cache.

        Args:
            key: Cache key
            result: Result to cache
        """
        with self._lock:
            # Check if the cache is full
            if len(self._cache) >= self._max_size and key not in self._cache:
                # Remove the oldest entry
                oldest_key = min(self._cache.items(), key=lambda x: x[1][1])[0]
                del self._cache[oldest_key]

            # Add the result to the cache
            self._cache[key] = (result, time.time())

    def invalidate(self, key: Any) -> None:
        """
        Invalidate a cached result.

        Args:
            key: Cache key to invalidate
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]

    def clear(self) -> None:
        """Clear the cache"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def _cleanup_loop(self) -> None:
        """Periodically clean up expired results"""
        while self._running:
            time.sleep(min(10.0, self._expiration_time / 2))

            with self._lock:
                # Current time
                now = time.time()

                # Find expired keys
                expired_keys = [
                    key for key, (_, timestamp) in self._cache.items()
                    if now - timestamp > self._expiration_time
                ]

                # Remove expired results
                for key in expired_keys:
                    del self._cache[key]

    def shutdown(self) -> None:
        """Shut down the result cache"""
        self._running = False
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=1.0)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary of cache statistics
        """
        with self._lock:
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0
            }


def run_parallel(func: Callable[..., T],
                args_list: List[Tuple],
                mode: ConcurrencyMode = ConcurrencyMode.THREAD,
                max_workers: Optional[int] = None,
                timeout: Optional[float] = None,
                priority: ThreadPriority = ThreadPriority.NORMAL) -> List[ParallelResult[T]]:
    """
    Run a function with multiple sets of arguments in parallel.

    Args:
        func: Function to execute
        args_list: List of argument tuples for each function call
        mode: Execution mode (thread, process, async)
        max_workers: Maximum number of concurrent workers
        timeout: Maximum time to wait for completion
        priority: Thread priority (only for THREAD mode)

    Returns:
        List of results for each function call
    """
    results: List[ParallelResult[T]] = []

    if mode == ConcurrencyMode.SYNC:
        # Synchronous execution
        for args in args_list:
            result = ParallelResult[T]()
            start_time = time.perf_counter()

            try:
                result.result = func(*args)
                result.success = True
            except Exception as e:
                result.error = e
                result.success = False

            result.execution_time = time.perf_counter() - start_time
            results.append(result)

    elif mode == ConcurrencyMode.THREAD:
        # Threaded execution
        thread_pool = ThreadPoolManager.get_instance()
        futures = []

        for args in args_list:
            futures.append(thread_pool.submit(func, *args, priority=priority))

        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            result = ParallelResult[T]()

            try:
                result.result = future.result()
                result.success = True
            except Exception as e:
                result.error = e
                result.success = False

            # Get execution time from future if available
            if hasattr(future, 'start_time') and hasattr(future, 'end_time'):
                result.execution_time = future.end_time - future.start_time

            results.append(result)

    elif mode == ConcurrencyMode.PROCESS:
        # Process execution
        process_pool = ProcessPoolManager.get_instance()
        futures = []

        for args in args_list:
            futures.append(process_pool.submit(func, *args))

        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            result = ParallelResult[T]()

            try:
                result.result = future.result()
                result.success = True
            except Exception as e:
                result.error = e
                result.success = False

            results.append(result)

    elif mode == ConcurrencyMode.ASYNC:
        # Async execution
        async_manager = AsyncTaskManager.get_instance()

        if not async_manager.is_running():
            async_manager.start()

        futures = []

        for args in args_list:
            # Check if func is a coroutine function
            if inspect.iscoroutinefunction(func):
                futures.append(async_manager.run_coroutine(func, *args))
            else:
                # Create a wrapper coroutine for non-coroutine functions
                async def wrapper(*wrapper_args):
                    return func(*wrapper_args)

                futures.append(async_manager.run_coroutine(wrapper, *args))

        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            result = ParallelResult[T]()

            try:
                result.result = future.result()
                result.success = True
            except Exception as e:
                result.error = e
                result.success = False

            results.append(result)

    else:
        raise ValueError(f"Unsupported concurrency mode: {mode}")

    return results


def execute_with_retry(func: Callable[..., T],
                      max_retries: int = 3,
                      retry_delay: float = 1.0,
                      backoff_factor: float = 2.0,
                      exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
                      retry_on_result: Optional[Callable[[Any], bool]] = None,
                      timeout: Optional[float] = None,
                      *args, **kwargs) -> T:
    """
    Execute a function with retry logic.

    Args:
        func: Function to execute
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Multiplicative factor for exponential backoff
        exceptions: Exception types to retry on
        retry_on_result: Function to determine if result should trigger a retry
        timeout: Maximum time for a single attempt
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function call

    Raises:
        The last exception encountered if all retries fail
    """
    last_exception = None
    current_delay = retry_delay

    for attempt in range(max_retries + 1):
        try:
            # Execute with timeout if specified
            if timeout:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(func, *args, **kwargs)
                    result = future.result(timeout=timeout)
            else:
                result = func(*args, **kwargs)

            # Check if result should trigger a retry
            if retry_on_result and retry_on_result(result):
                if attempt < max_retries:
                    logger.debug(f"Retrying due to result condition (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
                    continue
                return result

            return result

        except exceptions as e:
            last_exception = e

            if attempt < max_retries:
                logger.debug(f"Retrying due to {type(e).__name__}: {str(e)} (attempt {attempt + 1}/{max_retries + 1})")
                time.sleep(current_delay)
                current_delay *= backoff_factor
            else:
                logger.debug(f"All retry attempts failed: {str(e)}")

    if last_exception:
        raise last_exception

    # This should never be reached, but is here for type checking
    raise RuntimeError("Unexpected error in execute_with_retry")


def benchmark(func: Callable[..., T], *args, repeats: int = 1, **kwargs) -> Dict[str, Any]:
    """
    Benchmark a function's performance.

    Args:
        func: Function to benchmark
        *args: Positional arguments for the function
        repeats: Number of times to repeat the benchmark
        **kwargs: Keyword arguments for the function

    Returns:
        Dictionary with benchmark results
    """
    times = []
    results = []
    errors = []

    for _ in range(repeats):
        start_time = time.perf_counter()

        try:
            result = func(*args, **kwargs)
            times.append(time.perf_counter() - start_time)
            results.append(result)
        except Exception as e:
            times.append(time.perf_counter() - start_time)
            errors.append(e)

    if not times:
        return {
            "success": False,
            "error": "No measurements taken"
        }

    return {
        "success": len(errors) == 0,
        "mean": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
        "median": sorted(times)[len(times) // 2],
        "repeats": repeats,
        "errors": len(errors),
        "first_error": str(errors[0]) if errors else None
    }


# Decorator for retrying functions
def retry(max_retries: int = 3,
         retry_delay: float = 1.0,
         backoff_factor: float = 2.0,
         exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception):
    """
    Decorator for retrying functions.

    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Multiplicative factor for exponential backoff
        exceptions: Exception types to retry on

    Returns:
        Decorated function with retry logic
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return execute_with_retry(
                func,
                max_retries=max_retries,
                retry_delay=retry_delay,
                backoff_factor=backoff_factor,
                exceptions=exceptions,
                *args, **kwargs
            )
        return wrapper
    return decorator


# Decorator for executing functions asynchronously
def async_execution(priority: ThreadPriority = ThreadPriority.NORMAL):
    """
    Decorator for executing functions asynchronously.

    Args:
        priority: Thread priority

    Returns:
        Decorated function with asynchronous execution
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            thread_pool = ThreadPoolManager.get_instance()
            return thread_pool.submit(func, *args, priority=priority, **kwargs)
        return wrapper
    return decorator


# Decorator for timing function execution
def timed(logger_func: Optional[Callable[[str], None]] = None):
    """
    Decorator for timing function execution.

    Args:
        logger_func: Function to log the timing (defaults to print)

    Returns:
        Decorated function with timing
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time

            log_func = logger_func or print
            log_func(f"{func.__name__} executed in {elapsed:.6f} seconds")

            return result
        return wrapper
    return decorator


# Decorator for caching function results
def cached(expiration_time: float = 60.0, max_size: int = 1000):
    """
    Decorator for caching function results.

    Args:
        expiration_time: Time in seconds after which results expire
        max_size: Maximum number of results to cache

    Returns:
        Decorated function with result caching
    """
    cache = ResultCache(expiration_time, max_size)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key from the function arguments
            key = (func.__name__, args, frozenset(kwargs.items()))

            # Get from cache or compute
            return cache.get(key, lambda: func(*args, **kwargs))

        # Add cache reference to the wrapper for manual management
        wrapper.cache = cache

        return wrapper
    return decorator