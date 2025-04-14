import unittest
import time
import datetime
from unittest.mock import MagicMock, patch
import threading
from concurrent.futures import ThreadPoolExecutor

from core.scheduler import Scheduler, TaskPriority, TaskStatus, ScheduledTask


class TestScheduler(unittest.TestCase):
    """Test cases for the Scheduler class"""

    def setUp(self):
        """Set up a scheduler instance for each test"""
        self.scheduler = Scheduler(max_workers=5)
        self.scheduler.start()

    def tearDown(self):
        """Clean up after each test"""
        self.scheduler.stop()

    def test_init(self):
        """Test scheduler initialization"""
        scheduler = Scheduler(max_workers=3)
        self.assertEqual(scheduler._max_workers, 3)
        self.assertFalse(scheduler._running)
        self.assertEqual(len(scheduler._task_queue), 0)
        self.assertEqual(len(scheduler._task_dict), 0)
        self.assertEqual(len(scheduler._periodic_tasks), 0)

    def test_start_stop(self):
        """Test starting and stopping the scheduler"""
        scheduler = Scheduler()
        self.assertFalse(scheduler._running)
        
        scheduler.start()
        self.assertTrue(scheduler._running)
        self.assertIsNotNone(scheduler._scheduler_thread)
        
        scheduler.stop()
        self.assertFalse(scheduler._running)

    def test_schedule_basic(self):
        """Test basic task scheduling"""
        result = []
        
        def test_task(value):
            result.append(value)
            return value
        
        task_id = self.scheduler.schedule(test_task, "test_value")
        
        # Allow task to execute
        time.sleep(0.2)
        
        # Verify task was executed
        self.assertEqual(result, ["test_value"])
        
        # Check task info
        task_info = self.scheduler.get_task(task_id)
        self.assertEqual(task_info["status"], "COMPLETED")
        self.assertEqual(task_info["name"], "test_task")

    def test_schedule_with_delay(self):
        """Test scheduling a task with delay"""
        result = []
        
        def test_task():
            result.append(time.time())
        
        start_time = time.time()
        delay = 0.5
        self.scheduler.schedule(test_task, delay=delay)
        
        # Wait for task to complete
        time.sleep(delay + 0.2)
        
        # Verify task was executed after delay
        self.assertEqual(len(result), 1)
        self.assertGreaterEqual(result[0] - start_time, delay)

    def test_schedule_at_specific_time(self):
        """Test scheduling a task at a specific time"""
        result = []
        
        def test_task():
            result.append(time.time())
        
        # Schedule for 0.5 seconds in the future
        run_at = time.time() + 0.5
        self.scheduler.schedule_at(run_at, test_task)
        
        # Wait for task to complete
        time.sleep(0.7)
        
        # Verify task was executed at the specified time
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], run_at, delta=0.1)

    def test_schedule_at_with_datetime(self):
        """Test scheduling with datetime object"""
        result = []
        
        def test_task():
            result.append(time.time())
        
        # Schedule for 0.5 seconds in the future
        run_at = datetime.datetime.now() + datetime.timedelta(seconds=0.5)
        self.scheduler.schedule_at(run_at, test_task)
        
        # Wait for task to complete
        time.sleep(0.7)
        
        # Verify task was executed
        self.assertEqual(len(result), 1)

    def test_task_priority(self):
        """Test task priority ordering"""
        execution_order = []
        
        def test_task(name):
            execution_order.append(name)
        
        # Schedule tasks with different priorities
        self.scheduler.schedule(test_task, "low", priority=TaskPriority.LOW, delay=0.1)
        self.scheduler.schedule(test_task, "normal", priority=TaskPriority.NORMAL, delay=0.1)
        self.scheduler.schedule(test_task, "high", priority=TaskPriority.HIGH, delay=0.1)
        self.scheduler.schedule(test_task, "critical", priority=TaskPriority.CRITICAL, delay=0.1)
        
        # Wait for tasks to complete
        time.sleep(0.5)
        
        # Verify execution order based on priority
        self.assertEqual(execution_order, ["critical", "high", "normal", "low"])

    def test_periodic_task(self):
        """Test periodic task execution"""
        counter = [0]
        
        def test_task():
            counter[0] += 1
        
        # Schedule a periodic task with 0.2 second interval
        self.scheduler.schedule_periodic(0.2, test_task)
        
        # Wait for multiple executions
        time.sleep(1.0)
        
        # Verify the task executed multiple times
        self.assertGreaterEqual(counter[0], 3)  # Should run at least 3 times in 1 second

    def test_cancel_task(self):
        """Test cancelling a scheduled task"""
        result = []
        
        def test_task():
            result.append(1)
        
        # Schedule task with delay
        task_id = self.scheduler.schedule(test_task, delay=0.5)
        
        # Cancel the task
        cancelled = self.scheduler.cancel(task_id)
        
        # Wait to ensure task would have run
        time.sleep(0.7)
        
        # Verify task was cancelled and not executed
        self.assertTrue(cancelled)
        self.assertEqual(result, [])
        
        # Check task info
        task_info = self.scheduler.get_task(task_id)
        self.assertEqual(task_info["status"], "CANCELLED")

    def test_cannot_cancel_running_task(self):
        """Test that running tasks cannot be cancelled"""
        def slow_task():
            time.sleep(0.5)
        
        # Schedule immediate task
        task_id = self.scheduler.schedule(slow_task)
        
        # Give task time to start
        time.sleep(0.1)
        
        # Try to cancel running task
        cancelled = self.scheduler.cancel(task_id)
        
        # Should not be able to cancel running task
        self.assertFalse(cancelled)

    def test_task_with_retries(self):
        """Test task retry functionality"""
        attempts = [0]
        
        def failing_task():
            attempts[0] += 1
            if attempts[0] < 3:
                raise ValueError("Simulated failure")
            return "success"
        
        # Schedule task with retries
        task_id = self.scheduler.schedule(
            failing_task,
            max_retries=3,
            retry_delay=0.2
        )
        
        # Wait for retries and completion
        time.sleep(1.0)
        
        # Verify retry behavior
        self.assertEqual(attempts[0], 3)
        
        # Check task info
        task_info = self.scheduler.get_task(task_id)
        self.assertEqual(task_info["status"], "COMPLETED")
        self.assertEqual(task_info["retry_count"], 2)

    def test_task_timeout(self):
        """Test task timeout functionality"""
        def slow_task():
            time.sleep(1.0)
            return "done"
        
        # Schedule task with short timeout
        task_id = self.scheduler.schedule(slow_task, timeout=0.2)
        
        # Wait for task to fail
        time.sleep(0.5)
        
        # Check task info
        task_info = self.scheduler.get_task(task_id)
        self.assertEqual(task_info["status"], "FAILED")

    def test_get_all_tasks(self):
        """Test retrieving all tasks"""
        # Schedule multiple tasks
        self.scheduler.schedule(lambda: None, name="task1")
        self.scheduler.schedule(lambda: None, name="task2")
        self.scheduler.schedule(lambda: None, name="task3")
        
        # Get all tasks
        all_tasks = self.scheduler.get_all_tasks()
        
        # Verify task count
        self.assertEqual(len(all_tasks), 3)
        
        # Verify task names
        task_names = [task["name"] for task in all_tasks]
        self.assertIn("task1", task_names)
        self.assertIn("task2", task_names)
        self.assertIn("task3", task_names)

    def test_get_stats(self):
        """Test getting scheduler statistics"""
        # Schedule some tasks
        self.scheduler.schedule(lambda: None)
        self.scheduler.schedule_periodic(1.0, lambda: None)
        
        # Get stats
        stats = self.scheduler.get_stats()
        
        # Verify stats contain expected keys
        expected_keys = [
            "scheduled", "completed", "failed", "cancelled", 
            "running", "queue_size", "periodic_tasks", 
            "total_tasks", "available_workers", "active_workers"
        ]
        for key in expected_keys:
            self.assertIn(key, stats)
        
        # Verify stats values
        self.assertEqual(stats["scheduled"], 2)
        self.assertEqual(stats["periodic_tasks"], 1)
        self.assertEqual(stats["total_tasks"], 2)

    @patch("core.event_bus.get_event_bus")
    def test_events(self, mock_get_event_bus):
        """Test event publishing"""
        # Setup mock event bus
        mock_bus = MagicMock()
        mock_get_event_bus.return_value = mock_bus
        
        # Create new scheduler with mocked event bus
        scheduler = Scheduler()
        scheduler.start()
        
        try:
            # Schedule task
            scheduler.schedule(lambda: None, name="event_test")
            
            # Wait for task to complete
            time.sleep(0.2)
            
            # Verify events were published
            self.assertTrue(mock_bus.publish.called)
            
            # Get all event topics published
            event_topics = [
                call_args[0][0].topic 
                for call_args in mock_bus.publish.call_args_list
            ]
            
            # Verify expected event topics
            self.assertIn("task.scheduled", event_topics)
            self.assertIn("task.started", event_topics)
            self.assertIn("task.completed", event_topics)
        finally:
            scheduler.stop()

    def test_task_to_dict(self):
        """Test task to_dict method"""
        # Create a task
        task = ScheduledTask(
            scheduled_time=time.time(),
            priority=TaskPriority.HIGH,
            creation_time=time.time(),
            name="test_task",
            callable=lambda: None,
            tags={"tag1", "tag2"}
        )
        
        # Convert to dict
        task_dict = task.to_dict()
        
        # Verify dict contains expected keys
        expected_keys = [
            "id", "name", "scheduled_time", "creation_time", 
            "priority", "status", "periodic", "interval", 
            "max_retries", "retry_count", "tags", 
            "execution_time", "completion_time", "timeout"
        ]
        for key in expected_keys:
            self.assertIn(key, task_dict)
        
        # Verify values
        self.assertEqual(task_dict["name"], "test_task")
        self.assertEqual(task_dict["priority"], "HIGH")
        self.assertEqual(task_dict["status"], "PENDING")
        self.assertEqual(task_dict["tags"], ["tag1", "tag2"])

    def test_concurrent_tasks(self):
        """Test concurrent task execution"""
        results = []
        lock = threading.Lock()
        
        def test_task(value):
            # Simulate workload
            time.sleep(0.2)
            with lock:
                results.append(value)
        
        # Schedule multiple concurrent tasks
        for i in range(1, 6):
            self.scheduler.schedule(test_task, i)
        
        # Wait for tasks to complete
        time.sleep(0.5)
        
        # Verify all tasks executed
        self.assertEqual(sorted(results), [1, 2, 3, 4, 5])

    def test_max_workers_limit(self):
        """Test max workers limit is enforced"""
        start_barrier = threading.Barrier(6)  # 5 tasks + 1 test thread
        results = []
        
        def blocking_task(value):
            start_barrier.wait()  # Wait for all tasks to start
            time.sleep(0.2)  # Hold the worker for some time
            results.append(value)
        
        # Create scheduler with 5 workers
        scheduler = Scheduler(max_workers=5)
        scheduler.start()
        
        try:
            # Schedule 10 tasks (more than max_workers)
            for i in range(10):
                scheduler.schedule(blocking_task, i)
            
            # Wait for first batch to start
            start_barrier.wait()
            
            # Wait for first batch to complete
            time.sleep(0.3)
            
            # Verify only 5 tasks were executed (max_workers limit)
            self.assertEqual(len(results), 5)
            
            # Wait for remaining tasks
            time.sleep(0.5)
            
            # Verify all tasks eventually completed
            self.assertEqual(len(results), 10)
        finally:
            scheduler.stop()


if __name__ == "__main__":
    unittest.main()