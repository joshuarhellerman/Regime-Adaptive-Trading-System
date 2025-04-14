"""
scheduler.py - Deterministic Task Scheduling System

This module provides reliable task scheduling with failure handling,
precise timing control, and prioritization. It serves as the central
scheduling mechanism for both trading and non-trading operations.
"""

import logging
import threading
import time
import heapq
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from core.event_bus import EventTopics, create_event, get_event_bus, Event, EventPriority

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Priority levels for scheduled tasks"""
    CRITICAL = 0  # Must run on time, system-critical tasks
    HIGH = 1      # High priority, but can tolerate small delays
    NORMAL = 2    # Normal priority tasks
    LOW = 3       # Background tasks, run when system is not busy
    IDLE = 4      # Only run when system is idle


class TaskStatus(Enum):
    """Status of a scheduled task"""
    PENDING = "pending"      # Waiting to be executed
    RUNNING = "running"      # Currently executing
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"        # Failed with an exception
    CANCELLED = "cancelled"  # Cancelled before execution


@dataclass(order=True)
class ScheduledTask:
    """Representation of a task scheduled for execution"""
    # Priority items first for the priority queue
    scheduled_time: float = field(compare=True)
    priority: TaskPriority = field(compare=True)
    creation_time: float = field(compare=True)

    # Other attributes (not used for ordering)
    id: str = field(default_factory=lambda: str(uuid.uuid4()), compare=False)
    name: str = field(default="task", compare=False)
    callable: Callable = field(compare=False)
    args: Tuple = field(default_factory=tuple, compare=False)
    kwargs: Dict[str, Any] = field(default_factory=dict, compare=False)
    status: TaskStatus = field(default=TaskStatus.PENDING, compare=False)
    result: Any = field(default=None, compare=False)
    error: Optional[Exception] = field(default=None, compare=False)
    execution_time: Optional[float] = field(default=None, compare=False)
    completion_time: Optional[float] = field(default=None, compare=False)
    periodic: bool = field(default=False, compare=False)
    interval: Optional[float] = field(default=None, compare=False)
    max_retries: int = field(default=0, compare=False)
    retry_count: int = field(default=0, compare=False)
    retry_delay: float = field(default=1.0, compare=False)
    tags: Set[str] = field(default_factory=set, compare=False)
    timeout: Optional[float] = field(default=None, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "scheduled_time": self.scheduled_time,
            "creation_time": self.creation_time,
            "priority": self.priority.name,
            "status": self.status.name,
            "periodic": self.periodic,
            "interval": self.interval,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "tags": list(self.tags),
            "execution_time": self.execution_time,
            "completion_time": self.completion_time,
            "timeout": self.timeout
        }


class Scheduler:
    """
    A robust task scheduler with support for one-time and periodic tasks,
    precise timing, priorities, and failure handling.
    """

    def __init__(self, max_workers: int = 10):
        """
        Initialize the scheduler.

        Args:
            max_workers: Maximum number of concurrent task workers
        """
        self._task_queue = []  # Priority queue of tasks
        self._task_dict = {}   # For efficient lookup by ID
        self._periodic_tasks = set()  # Set of periodic task IDs
        self._lock = threading.RLock()
        self._running = False
        self._scheduler_thread = None
        self._workers = []
        self._max_workers = max_workers
        self._worker_semaphore = threading.Semaphore(max_workers)
        self._event_bus = get_event_bus()
        self._task_stats = {
            "scheduled": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "running": 0
        }

        logger.info(f"Scheduler initialized with {max_workers} workers")

    def start(self) -> None:
        """Start the scheduler"""
        with self._lock:
            if self._running:
                logger.warning("Scheduler is already running")
                return

            self._running = True
            self._scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                name="SchedulerThread",
                daemon=True
            )
            self._scheduler_thread.start()

            logger.info("Scheduler started")

    def stop(self) -> None:
        """Stop the scheduler"""
        with self._lock:
            if not self._running:
                logger.warning("Scheduler is not running")
                return

            self._running = False

            # Wait for scheduler thread to finish
            if self._scheduler_thread:
                self._scheduler_thread.join(timeout=5.0)

            # Wait for all workers to finish
            for worker in self._workers:
                worker.join(timeout=2.0)

            self._workers = []

            logger.info("Scheduler stopped")

    def schedule(self, func: Callable, *args,
                name: str = None,
                priority: TaskPriority = TaskPriority.NORMAL,
                delay: float = 0,
                run_at: Union[float, datetime] = None,
                periodic: bool = False,
                interval: float = None,
                max_retries: int = 0,
                retry_delay: float = 1.0,
                tags: Set[str] = None,
                timeout: float = None,
                **kwargs) -> str:
        """
        Schedule a task for execution.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            name: Task name (defaults to function name)
            priority: Task priority
            delay: Delay in seconds before execution
            run_at: Specific time to run (overrides delay)
            periodic: Whether this is a periodic task
            interval: Interval in seconds for periodic tasks
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            tags: Set of tags for categorizing tasks
            timeout: Maximum execution time in seconds
            **kwargs: Keyword arguments for the function

        Returns:
            Task ID
        """
        current_time = time.time()

        # Calculate scheduled time
        if run_at is not None:
            if isinstance(run_at, datetime):
                # Convert datetime to timestamp
                scheduled_time = run_at.timestamp()
            else:
                # Use provided timestamp
                scheduled_time = float(run_at)
        else:
            # Use delay
            scheduled_time = current_time + delay

        # Create task
        task = ScheduledTask(
            scheduled_time=scheduled_time,
            priority=priority,
            creation_time=current_time,
            name=name or func.__name__,
            callable=func,
            args=args,
            kwargs=kwargs,
            periodic=periodic,
            interval=interval,
            max_retries=max_retries,
            retry_delay=retry_delay,
            tags=tags or set(),
            timeout=timeout
        )

        with self._lock:
            # Add to queue
            heapq.heappush(self._task_queue, task)

            # Add to lookup dict
            self._task_dict[task.id] = task

            # Track periodic tasks
            if periodic:
                self._periodic_tasks.add(task.id)

            # Update stats
            self._task_stats["scheduled"] += 1

            logger.debug(f"Scheduled task: {task.name} (ID: {task.id}) at {scheduled_time}")

        # Publish event
        event = create_event(
            EventTopics.TASK_SCHEDULED,
            {
                "task_id": task.id,
                "name": task.name,
                "scheduled_time": task.scheduled_time
            }
        )
        self._event_bus.publish(event)

        # Acquire worker semaphore (blocks if no workers available)
        if not self._worker_semaphore.acquire(blocking=False):
            # No workers available, requeue the task with a small delay
            logger.debug(f"No workers available, requeuing task: {task.name} (ID: {task.id})")
            with self._lock:
                # Return to queue with a small delay
                task.status = TaskStatus.PENDING
                task.scheduled_time = time.time() + 0.1
                heapq.heappush(self._task_queue, task)
                self._task_stats["running"] -= 1
            return

        # Create worker thread
        worker = threading.Thread(
            target=self._execute_task,
            args=(task,),
            name=f"TaskWorker-{task.id}",
            daemon=True
        )

        with self._lock:
            self._workers.append(worker)

        worker.start()

    def _execute_task(self, task: ScheduledTask) -> None:
        """
        Execute a task in a worker thread.

        Args:
            task: The task to execute
        """
        try:
            logger.debug(f"Executing task: {task.name} (ID: {task.id})")

            # Execute with timeout if specified
            if task.timeout:
                result = self._execute_with_timeout(task)
            else:
                # Execute the task
                result = task.callable(*task.args, **task.kwargs)

            # Store result
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completion_time = time.time()

            with self._lock:
                self._task_stats["completed"] += 1
                self._task_stats["running"] -= 1

            logger.debug(f"Task completed: {task.name} (ID: {task.id})")

            # Publish completion event
            event = create_event(
                EventTopics.TASK_COMPLETED,
                {
                    "task_id": task.id,
                    "name": task.name,
                    "execution_time": task.completion_time - task.execution_time if task.execution_time else None
                }
            )
            self._event_bus.publish(event)

            # Handle periodic tasks
            if task.periodic and task.id in self._periodic_tasks and self._running:
                self._reschedule_periodic_task(task)

        except Exception as e:
            logger.error(f"Task execution failed: {task.name} (ID: {task.id}) - {str(e)}")

            # Store error
            task.error = e
            task.status = TaskStatus.FAILED
            task.completion_time = time.time()

            with self._lock:
                self._task_stats["failed"] += 1
                self._task_stats["running"] -= 1

            # Publish failure event
            event = create_event(
                EventTopics.TASK_FAILED,
                {
                    "task_id": task.id,
                    "name": task.name,
                    "error": str(e)
                },
                priority=EventPriority.HIGH
            )
            self._event_bus.publish(event)

            # Handle retries
            if task.retry_count < task.max_retries and self._running:
                self._retry_task(task)
            elif task.periodic and task.id in self._periodic_tasks and self._running:
                # Even if task failed, reschedule if it's periodic
                self._reschedule_periodic_task(task)

        finally:
            # Clean up worker thread reference
            with self._lock:
                if threading.current_thread() in self._workers:
                    self._workers.remove(threading.current_thread())

            # Release worker semaphore
            self._worker_semaphore.release()

    def _execute_with_timeout(self, task: ScheduledTask) -> Any:
        """
        Execute a task with timeout using a separate thread.

        Args:
            task: The task to execute

        Returns:
            Task result

        Raises:
            TimeoutError: If task execution exceeds the timeout
        """
        result = None
        error = None
        execution_finished = threading.Event()

        def _target():
            nonlocal result, error
            try:
                result = task.callable(*task.args, **task.kwargs)
            except Exception as e:
                error = e
            finally:
                execution_finished.set()

        # Start execution thread
        thread = threading.Thread(target=_target, daemon=True)
        thread.start()

        # Wait for execution with timeout
        if not execution_finished.wait(timeout=task.timeout):
            raise TimeoutError(f"Task execution timed out after {task.timeout} seconds")

        # Propagate any exception raised in the execution thread
        if error:
            raise error

        return result

    def _retry_task(self, task: ScheduledTask) -> None:
        """
        Retry a failed task.

        Args:
            task: The failed task
        """
        # Increment retry count
        task.retry_count += 1

        # Calculate retry time
        retry_time = time.time() + task.retry_delay

        logger.debug(f"Scheduling retry {task.retry_count}/{task.max_retries} for task: {task.name} (ID: {task.id}) at {retry_time}")

        with self._lock:
            # Reset task status
            task.status = TaskStatus.PENDING
            task.execution_time = None
            task.completion_time = None
            task.error = None
            task.scheduled_time = retry_time

            # Add back to queue
            heapq.heappush(self._task_queue, task)

        # Publish retry event
        event = create_event(
            EventTopics.TASK_RETRY,
            {
                "task_id": task.id,
                "name": task.name,
                "retry_count": task.retry_count,
                "max_retries": task.max_retries,
                "scheduled_time": retry_time
            }
        )
        self._event_bus.publish(event)

    def _reschedule_periodic_task(self, task: ScheduledTask) -> None:
        """
        Reschedule a periodic task.

        Args:
            task: The periodic task
        """
        # Calculate next execution time
        # If task took longer than interval, schedule immediately
        next_time = max(
            time.time(),
            task.scheduled_time + task.interval
        )

        logger.debug(f"Rescheduling periodic task: {task.name} (ID: {task.id}) at {next_time}")

        with self._lock:
            # Reset task status
            task.status = TaskStatus.PENDING
            task.execution_time = None
            task.completion_time = None
            task.error = None
            task.retry_count = 0
            task.scheduled_time = next_time

            # Add back to queue
            heapq.heappush(self._task_queue, task)
 scheduled_time,
                "priority": priority.name
            }
        )
        self._event_bus.publish(event)

        return task.id

    def schedule_at(self, timestamp: Union[float, datetime], func: Callable, *args, **kwargs) -> str:
        """
        Schedule a task at a specific time.

        Args:
            timestamp: Time to run the task (timestamp or datetime)
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Task ID
        """
        return self.schedule(func, *args, run_at=timestamp, **kwargs)

    def schedule_periodic(self, interval: float, func: Callable, *args,
                         initial_delay: float = 0, **kwargs) -> str:
        """
        Schedule a periodic task.

        Args:
            interval: Interval in seconds
            func: Function to execute
            *args: Positional arguments for the function
            initial_delay: Initial delay before first execution
            **kwargs: Keyword arguments for the function

        Returns:
            Task ID
        """
        return self.schedule(
            func, *args,
            delay=initial_delay,
            periodic=True,
            interval=interval,
            **kwargs
        )

    def cancel(self, task_id: str) -> bool:
        """
        Cancel a scheduled task.

        Args:
            task_id: ID of the task to cancel

        Returns:
            Whether the task was successfully cancelled
        """
        with self._lock:
            if task_id not in self._task_dict:
                logger.warning(f"Task not found: {task_id}")
                return False

            task = self._task_dict[task_id]

            # If task is already running or completed, it cannot be cancelled
            if task.status in [TaskStatus.RUNNING, TaskStatus.COMPLETED]:
                logger.warning(f"Cannot cancel task {task_id} with status {task.status.name}")
                return False

            # Mark as cancelled
            task.status = TaskStatus.CANCELLED

            # Remove from periodic tasks if applicable
            if task_id in self._periodic_tasks:
                self._periodic_tasks.remove(task_id)

            # Update stats
            self._task_stats["cancelled"] += 1

            logger.debug(f"Cancelled task: {task.name} (ID: {task_id})")

            # Publish event
            event = create_event(
                EventTopics.TASK_CANCELLED,
                {
                    "task_id": task_id,
                    "name": task.name
                }
            )
            self._event_bus.publish(event)

            return True

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a task.

        Args:
            task_id: ID of the task

        Returns:
            Task information dictionary or None if not found
        """
        with self._lock:
            if task_id not in self._task_dict:
                return None

            return self._task_dict[task_id].to_dict()

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """
        Get information about all tasks.

        Returns:
            List of task information dictionaries
        """
        with self._lock:
            return [task.to_dict() for task in self._task_dict.values()]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get scheduler statistics.

        Returns:
            Dictionary of scheduler statistics
        """
        with self._lock:
            stats = self._task_stats.copy()
            stats["queue_size"] = len(self._task_queue)
            stats["periodic_tasks"] = len(self._periodic_tasks)
            stats["total_tasks"] = len(self._task_dict)
            stats["available_workers"] = self._worker_semaphore._value
            stats["active_workers"] = self._max_workers - self._worker_semaphore._value
            return stats

    def _scheduler_loop(self) -> None:
        """Main scheduler loop that processes the task queue"""
        logger.debug("Scheduler loop started")

        while self._running:
            try:
                next_task = None

                with self._lock:
                    # Check if there are tasks to process
                    if self._task_queue and self._task_queue[0].scheduled_time <= time.time():
                        next_task = heapq.heappop(self._task_queue)

                if next_task:
                    # Check if the task was cancelled
                    if next_task.status == TaskStatus.CANCELLED:
                        continue

                    # Process the task
                    self._process_task(next_task)
                else:
                    # No tasks ready, sleep for a short time
                    time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
                time.sleep(1)  # Sleep to avoid tight loop in case of persistent error

    def _process_task(self, task: ScheduledTask) -> None:
        """
        Process a task by executing it or scheduling it for execution.

        Args:
            task: The task to process
        """
        # Mark task as running
        with self._lock:
            task.status = TaskStatus.RUNNING
            task.execution_time = time.time()
            self._task_stats["running"] += 1

        # Publish event
        event = create_event(
            EventTopics.TASK_STARTED,
            {
                "task_id": task.id,
                "name": task.name,
                "scheduled_time":