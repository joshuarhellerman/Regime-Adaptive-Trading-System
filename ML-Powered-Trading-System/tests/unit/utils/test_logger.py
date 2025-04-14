"""
Tests for the logger module.

This module contains tests for the logging utilities defined in utils/logger.py.
"""

import unittest
import os
import json
import time
import logging
import threading
import tempfile
import shutil
from unittest.mock import patch, MagicMock, call
from pathlib import Path

# Import the module under test
from utils.logger import (
    LoggerManager, TradingLogger, PerformanceTracker, ModuleFilter,
    StructuredFormatter, DatabaseLogHandler, log_latency,
    get_logger, track_performance, configure_logging, create_component_logger
)

# Import configuration class for testing
from config.logging_config import LoggingConfig, LogLevel, LogFormat, LogDestination


class TestLoggerManager(unittest.TestCase):
    """Test cases for the LoggerManager class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for log files
        self.temp_dir = tempfile.mkdtemp()

        # Mock the config module
        self.config_patcher = patch('utils.logger.get_logging_config')
        self.mock_get_config = self.config_patcher.start()

        # Create a mock configuration
        self.mock_config = MagicMock(spec=LoggingConfig)
        self.mock_config.root_level = LogLevel.INFO
        self.mock_config.log_dir = self.temp_dir
        self.mock_config.audit = MagicMock()
        self.mock_config.audit.enable_audit_logging = True
        self.mock_config.audit.audit_logger_name = "audit"
        self.mock_config.audit.use_secure_audit_log = False
        self.mock_config.log_rotation = MagicMock()
        self.mock_config.log_rotation.enable_rotation = True
        self.mock_config.log_rotation.rotation_interval = "daily"
        self.mock_config.log_rotation.backup_count = 7
        self.mock_config.log_rotation.max_size_mb = 10
        self.mock_config.capture_warnings = True
        self.mock_config.log_exceptions = True
        self.mock_config.get_python_logging_config.return_value = {
            'version': 1,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'standard',
                    'stream': 'ext://sys.stdout'
                },
                'file': {
                    'class': 'logging.FileHandler',
                    'level': 'DEBUG',
                    'formatter': 'standard',
                    'filename': os.path.join(self.temp_dir, 'app.log'),
                    'mode': 'a'
                }
            },
            'root': {
                'level': 'INFO',
                'handlers': ['console', 'file']
            }
        }

        self.mock_get_config.return_value = self.mock_config

        # Reset singleton state before each test
        LoggerManager._instance = None
        LoggerManager._initialized = False
        LoggerManager._loggers = {}

        # Initialize manager for testing
        self.manager = LoggerManager()

    def tearDown(self):
        """Tear down test fixtures."""
        # Stop patches
        self.config_patcher.stop()

        # Remove temp directory
        shutil.rmtree(self.temp_dir)

    def test_singleton_behavior(self):
        """Test that LoggerManager behaves as a singleton."""
        manager1 = LoggerManager()
        manager2 = LoggerManager()
        self.assertIs(manager1, manager2)

    def test_get_logger(self):
        """Test getting a logger instance."""
        logger = self.manager.get_logger("test.module")
        self.assertIsInstance(logger, TradingLogger)
        self.assertEqual(logger._name, "test.module")

        # Test that getting the same logger returns the same instance
        logger2 = self.manager.get_logger("test.module")
        self.assertIs(logger, logger2)

    def test_get_audit_logger(self):
        """Test getting the audit logger."""
        audit_logger = self.manager.get_audit_logger()
        self.assertIsInstance(audit_logger, TradingLogger)
        self.assertEqual(audit_logger._name, "audit")

    def test_start_performance_tracker(self):
        """Test starting a performance tracker."""
        tracker = self.manager.start_performance_tracker("test_context")
        self.assertIsInstance(tracker, PerformanceTracker)
        self.assertEqual(tracker._context_id, "test_context")

        # Test that the tracker is stored in the manager
        self.assertIn("test_context", self.manager._performance_trackers)
        self.assertIs(tracker, self.manager._performance_trackers["test_context"])

    def test_get_performance_tracker(self):
        """Test getting a performance tracker."""
        # Create a tracker
        tracker = self.manager.start_performance_tracker("test_context")

        # Get the tracker by ID
        retrieved_tracker = self.manager.get_performance_tracker("test_context")
        self.assertIs(tracker, retrieved_tracker)

        # Test getting a non-existent tracker
        self.assertIsNone(self.manager.get_performance_tracker("nonexistent"))

    @patch('utils.logger.logging.config.dictConfig')
    def test_update_configuration(self, mock_dict_config):
        """Test updating the logging configuration."""
        # Get a logger before update
        old_logger = self.manager.get_logger("test.module")

        # Create a new configuration
        new_config = MagicMock(spec=LoggingConfig)
        new_config.root_level = LogLevel.DEBUG
        new_config.get_python_logging_config.return_value = {
            'version': 1,
            'root': {'level': 'DEBUG'}
        }

        # Update configuration
        self.manager.update_configuration(new_config)

        # Verify that dictConfig was called with the new config
        mock_dict_config.assert_called_with(new_config.get_python_logging_config.return_value)

        # Get a logger after update and verify it's different
        new_logger = self.manager.get_logger("test.module")
        self.assertIsNot(old_logger._python_logger, new_logger._python_logger)

        # But the TradingLogger instance should be updated, not replaced
        self.assertIs(old_logger, self.manager.get_logger("test.module"))

    @patch('utils.logger.sys.excepthook')
    def test_exception_hook(self, mock_excepthook):
        """Test that the exception hook is set up."""
        # The exception hook should be set up in __init__
        # Since we're mocking sys.excepthook, we can check its args
        self.assertIsNotNone(mock_excepthook)

        # Test that the hook logs exceptions
        with patch.object(self.manager._root_logger, 'critical') as mock_critical:
            # Trigger the hook with a test exception
            exc_type = ValueError
            exc_value = ValueError("Test error")
            exc_traceback = None  # Not needed for this test

            # Get the replacement hook function
            new_hook = sys.excepthook

            # Call it directly
            new_hook(exc_type, exc_value, exc_traceback)

            # Verify that the exception was logged
            mock_critical.assert_called_once()
            self.assertEqual(mock_critical.call_args[0][0], "Uncaught exception")
            self.assertEqual(mock_critical.call_args[1]['exc_info'][0], exc_type)
            self.assertEqual(mock_critical.call_args[1]['exc_info'][1], exc_value)


class TestTradingLogger(unittest.TestCase):
    """Test cases for the TradingLogger class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the Python logger
        self.mock_python_logger = MagicMock(spec=logging.Logger)

        # Mock the manager
        self.mock_manager = MagicMock(spec=LoggerManager)

        # Create a TradingLogger instance
        self.logger = TradingLogger("test.module", self.mock_python_logger, self.mock_manager)

    def test_with_context(self):
        """Test creating a logger with context."""
        # Create a logger with context
        context_logger = self.logger.with_context(key1="value1", key2="value2")

        # Verify that a new logger was created
        self.assertIsNot(context_logger, self.logger)

        # Verify that context was added
        self.assertEqual(context_logger._context, {"key1": "value1", "key2": "value2"})

        # Verify that the original logger's context is unchanged
        self.assertEqual(self.logger._context, {})

        # Add more context
        extended_logger = context_logger.with_context(key3="value3")

        # Verify that the new context was combined with the existing context
        self.assertEqual(extended_logger._context,
                         {"key1": "value1", "key2": "value2", "key3": "value3"})

    def test_debug(self):
        """Test debug logging."""
        # Log a debug message
        self.logger.debug("Test debug message")

        # Verify that the python logger's log method was called with DEBUG level
        self.mock_python_logger.log.assert_called_with(
            logging.DEBUG, "Test debug message"
        )

    def test_info(self):
        """Test info logging."""
        # Log an info message
        self.logger.info("Test info message")

        # Verify that the python logger's log method was called with INFO level
        self.mock_python_logger.log.assert_called_with(
            logging.INFO, "Test info message"
        )

    def test_warning(self):
        """Test warning logging."""
        # Log a warning message
        self.logger.warning("Test warning message")

        # Verify that the python logger's log method was called with WARNING level
        self.mock_python_logger.log.assert_called_with(
            logging.WARNING, "Test warning message"
        )

    def test_error(self):
        """Test error logging."""
        # Log an error message
        self.logger.error("Test error message")

        # Verify that the python logger's log method was called with ERROR level
        self.mock_python_logger.log.assert_called_with(
            logging.ERROR, "Test error message"
        )

    def test_critical(self):
        """Test critical logging."""
        # Log a critical message
        self.logger.critical("Test critical message")

        # Verify that the python logger's log method was called with CRITICAL level
        self.mock_python_logger.log.assert_called_with(
            logging.CRITICAL, "Test critical message"
        )

    def test_exception(self):
        """Test exception logging."""
        # Log an exception
        try:
            raise ValueError("Test exception")
        except ValueError:
            self.logger.exception("Test exception message")

        # Verify that the python logger's log method was called with ERROR level and exc_info=True
        self.mock_python_logger.log.assert_called_once()
        args, kwargs = self.mock_python_logger.log.call_args
        self.assertEqual(args[0], logging.ERROR)
        self.assertEqual(args[1], "Test exception message")
        self.assertTrue(kwargs.get('exc_info'))

    def test_audit(self):
        """Test audit logging."""
        # Mock the audit logger
        mock_audit_logger = MagicMock(spec=TradingLogger)
        self.mock_manager.get_audit_logger.return_value = mock_audit_logger

        # Log an audit message
        self.logger.audit("Test audit message", user="testuser", operation="login")

        # Verify that the audit logger's _log method was called
        mock_audit_logger._log.assert_called_once()
        args, kwargs = mock_audit_logger._log.call_args
        self.assertEqual(args[0], logging.INFO)
        self.assertEqual(args[1], "Test audit message")

        # Verify that context was passed in extra
        self.assertIn('audit_context', kwargs.get('extra', {}))
        audit_context = kwargs['extra']['audit_context']
        self.assertEqual(audit_context['user'], "testuser")
        self.assertEqual(audit_context['operation'], "login")

    def test_metric(self):
        """Test metric logging."""
        # Log a metric
        self.logger.metric("request_latency", 150.5, "ms", "api_server")

        # Verify that the python logger's log method was called with INFO level
        self.mock_python_logger.log.assert_called_with(
            logging.INFO, "Metric request_latency: 150.5 ms"
        )

        # Test with health monitor
        mock_health_monitor = MagicMock()
        self.mock_manager.get_health_monitor.return_value = mock_health_monitor

        # Log another metric with health monitor available
        self.logger.metric("cpu_usage", 75.2, "%", "system")

        # Verify that health monitor's update_component_health was called
        mock_health_monitor.update_component_health.assert_called_with(
            component_id="system",
            metrics={"cpu_usage": 75.2}
        )

    def test_log_latency(self):
        """Test latency logging."""
        # Log a latency
        self.logger.log_latency("database_query", 50.3, True, "db_service")

        # Verify that the python logger's log method was called with INFO level
        self.mock_python_logger.log.assert_called_with(
            logging.INFO, "Operation database_query completed in 50.30 ms"
        )

        # Test with health monitor
        mock_health_monitor = MagicMock()
        self.mock_manager.get_health_monitor.return_value = mock_health_monitor

        # Log another latency with health monitor available
        self.logger.log_latency("api_request", 120.5, False, "api_service")

        # Verify that health monitor's log_operation was called
        mock_health_monitor.log_operation.assert_called_with(
            component_id="api_service",
            operation_type="api_request",
            duration_ms=120.5,
            success=False
        )

    def test_context_in_logging(self):
        """Test that context is included in log messages."""
        # Create a logger with context
        context_logger = self.logger.with_context(request_id="123", user_id="456")

        # Log a message
        context_logger.info("Test message with context")

        # Verify that context was included in extra
        self.mock_python_logger.log.assert_called_once()
        args, kwargs = self.mock_python_logger.log.call_args
        self.assertIn('extra', kwargs)
        self.assertIn('context', kwargs['extra'])
        self.assertEqual(kwargs['extra']['context'], {"request_id": "123", "user_id": "456"})

    def test_data_in_logging(self):
        """Test that data parameter is included in log messages."""
        # Log a message with data
        data = {"key1": "value1", "key2": 123}
        self.logger.info("Test message with data", data=data)

        # Verify that data was included in the message
        self.mock_python_logger.log.assert_called_once()
        args = self.mock_python_logger.log.call_args[0]
        self.assertEqual(args[0], logging.INFO)
        self.assertIn("Test message with data - Data:", args[1])
        self.assertIn('"key1": "value1"', args[1])
        self.assertIn('"key2": 123', args[1])


class TestPerformanceTracker(unittest.TestCase):
    """Test cases for the PerformanceTracker class."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the manager
        self.mock_manager = MagicMock(spec=LoggerManager)

        # Mock the logger
        self.mock_logger = MagicMock(spec=TradingLogger)
        self.mock_manager.get_logger.return_value = self.mock_logger

        # Create a PerformanceTracker instance
        self.tracker = PerformanceTracker("test_context", "test_component", self.mock_manager)

        # Track start time for comparison
        self.start_time = self.tracker._start_time

    def test_checkpoint(self):
        """Test recording checkpoints."""
        # Sleep a bit to ensure elapsed time
        time.sleep(0.01)

        # Record a checkpoint
        self.tracker.checkpoint("start_processing")

        # Verify that the checkpoint was recorded
        self.assertIn("start_processing", self.tracker._checkpoints)
        self.assertGreater(self.tracker._checkpoints["start_processing"], 0)

        # Verify that the checkpoint was logged
        self.mock_logger.debug.assert_called()

    def test_start_stop_timer(self):
        """Test starting and stopping timers."""
        # Start a timer
        self.tracker.start_timer("database_query")

        # Verify that the timer was started
        self.assertIn("database_query", self.tracker._active_timers)

        # Sleep a bit
        time.sleep(0.01)

        # Stop the timer
        duration = self.tracker.stop_timer("database_query")

        # Verify that the timer was stopped
        self.assertNotIn("database_query", self.tracker._active_timers)

        # Verify that a duration was returned
        self.assertIsNotNone(duration)
        self.assertGreater(duration, 0)

        # Verify that the duration was logged
        self.mock_logger._log.assert_called()

        # Test stopping a non-existent timer
        result = self.tracker.stop_timer("nonexistent")
        self.assertIsNone(result)
        self.mock_logger.warning.assert_called()

    def test_get_elapsed(self):
        """Test getting elapsed time."""
        # Sleep a bit
        time.sleep(0.01)

        # Get elapsed time from start
        elapsed_from_start = self.tracker.get_elapsed()

        # Verify that elapsed time is positive
        self.assertGreater(elapsed_from_start, 0)

        # Record a checkpoint
        self.tracker.checkpoint("checkpoint1")

        # Sleep a bit more
        time.sleep(0.01)

        # Get elapsed time from checkpoint
        elapsed_from_checkpoint = self.tracker.get_elapsed("checkpoint1")

        # Verify that elapsed time from checkpoint is positive but less than from start
        self.assertGreater(elapsed_from_checkpoint, 0)
        self.assertLess(elapsed_from_checkpoint, elapsed_from_start)

        # Test with a non-existent checkpoint
        self.tracker.get_elapsed("nonexistent")
        self.mock_logger.warning.assert_called()

    def test_get_checkpoint_times(self):
        """Test getting all checkpoint times."""
        # Record some checkpoints
        self.tracker.checkpoint("step1")
        time.sleep(0.01)
        self.tracker.checkpoint("step2")

        # Get checkpoint times
        checkpoint_times = self.tracker.get_checkpoint_times()

        # Verify that all checkpoints are present
        self.assertIn("step1", checkpoint_times)
        self.assertIn("step2", checkpoint_times)

        # Verify that step2 time is greater than step1 time
        self.assertGreater(checkpoint_times["step2"], checkpoint_times["step1"])

    def test_summary(self):
        """Test getting a performance summary."""
        # Record some checkpoints and timers
        self.tracker.checkpoint("start")
        self.tracker.start_timer("operation1")

        # Get summary
        summary = self.tracker.summary()

        # Verify summary structure
        self.assertEqual(summary["context_id"], "test_context")
        self.assertEqual(summary["component_id"], "test_component")
        self.assertEqual(summary["start_time"], self.start_time)
        self.assertGreater(summary["total_elapsed_ms"], 0)
        self.assertIn("start", summary["checkpoints"])
        self.assertIn("operation1", summary["active_timers"])

    def test_log_summary(self):
        """Test logging a performance summary."""
        # Record some checkpoints
        self.tracker.checkpoint("step1")
        self.tracker.checkpoint("step2")

        # Log summary
        self.tracker.log_summary()

        # Verify that summary was logged
        self.mock_logger._log.assert_called()
        log_args = self.mock_logger._log.call_args[0]
        self.assertEqual(log_args[0], logging.INFO)
        self.assertIn("Performance summary for test_context", log_args[1])
        self.assertIn("Total time", log_args[1])
        self.assertIn("Checkpoints", log_args[1])

    def test_stop(self):
        """Test stopping performance tracking."""
        # Start a timer
        self.tracker.start_timer("operation1")

        # Add tracker to manager
        self.mock_manager._performance_trackers = {"test_context": self.tracker}

        # Stop tracking
        self.tracker.stop()

        # Verify that timer was stopped
        self.assertNotIn("operation1", self.tracker._active_timers)

        # Verify that summary was logged
        self.mock_logger._log.assert_called()

        # Verify that tracker was removed from manager
        self.mock_manager._performance_trackers.pop.assert_called_with("test_context", None)


class TestModuleFilter(unittest.TestCase):
    """Test cases for the ModuleFilter class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a log record for testing
        self.record = logging.LogRecord(
            name="test.module.submodule",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )

    def test_no_patterns(self):
        """Test filtering with no patterns."""
        # Create a filter with no patterns
        filter_obj = ModuleFilter()

        # All records should be accepted
        self.assertTrue(filter_obj.filter(self.record))

    def test_include_patterns(self):
        """Test filtering with include patterns."""
        # Create a filter with include patterns
        filter_obj = ModuleFilter(include_modules=["test.*", "other.module"])

        # Record should be accepted (matches "test.*")
        self.assertTrue(filter_obj.filter(self.record))

        # Change record name to something that doesn't match
        self.record.name = "foo.bar"

        # Record should be rejected
        self.assertFalse(filter_obj.filter(self.record))

    def test_exclude_patterns(self):
        """Test filtering with exclude patterns."""
        # Create a filter with exclude patterns
        filter_obj = ModuleFilter(exclude_modules=["*.submodule", "other.module"])

        # Record should be rejected (matches "*.submodule")
        self.assertFalse(filter_obj.filter(self.record))

        # Change record name to something that doesn't match
        self.record.name = "test.module.component"

        # Record should be accepted
        self.assertTrue(filter_obj.filter(self.record))

    def test_include_and_exclude_patterns(self):
        """Test filtering with both include and exclude patterns."""
        # Create a filter with both include and exclude patterns
        filter_obj = ModuleFilter(
            include_modules=["test.*"],
            exclude_modules=["*.submodule"]
        )

        # Record should be rejected (matches include but also exclude)
        self.assertFalse(filter_obj.filter(self.record))

        # Change record name to match include but not exclude
        self.record.name = "test.module.component"

        # Record should be accepted
        self.assertTrue(filter_obj.filter(self.record))

        # Change record name to not match include
        self.record.name = "other.module"

        # Record should be rejected (doesn't match include)
        self.assertFalse(filter_obj.filter(self.record))


class TestStructuredFormatter(unittest.TestCase):
    """Test cases for the StructuredFormatter class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a formatter
        self.formatter = StructuredFormatter()

        # Create a record for testing
        self.record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        self.record.threadName = "MainThread"

    def test_format_basic(self):
        """Test basic formatting."""
        # Format the record
        formatted = self.formatter.format(self.record)

        # Parse the JSON
        log_dict = json.loads(formatted)

        # Verify basic fields
        self.assertEqual(log_dict["level"], "INFO")
        self.assertEqual(log_dict["logger"], "test.module")
        self.assertEqual(log_dict["message"], "Test message")
        self.assertEqual(log_dict["module"], "test.module")
        self.assertEqual(log_dict["line"], 42)

    def test_format_with_context(self):
        """Test formatting with context."""
        # Add context to the record
        self.record.context = {"request_id": "123", "user_id": "456"}

        # Format the record
        formatted = self.formatter.format(self.record)

        # Parse the JSON
        log_dict = json.loads(formatted)

        # Verify context is included
        self.assertIn("context", log_dict)
        self.assertEqual(log_dict["context"]["request_id"], "123")
        self.assertEqual(log_dict["context"]["user_id"], "456")

    def test_format_with_exception(self):
        """Test formatting with exception."""
        # Add exception info to the record
        try:
            raise ValueError("Test exception")
        except ValueError:
            self.record.exc_info = sys.exc_info()

        # Format the record
        formatted = self.formatter.format(self.record)

        # Parse the JSON
        log_dict = json.loads(formatted)

        # Verify exception is included
        self.assertIn("exception", log_dict)
        self.assertIn("ValueError: Test exception", log_dict["exception"])

    def test_format_with_field_filtering(self):
        """Test formatting with field filtering."""
        # Create a formatter with field filtering
        formatter = StructuredFormatter(
            include_fields=["timestamp", "level", "message", "context"],
            exclude_fields=["timestamp"]  # Exclude should override include
        )

        # Add context to the record
        self.record.context = {"request_id": "123"}

        # Format the record
        formatted = formatter.format(self.record)

        # Parse the JSON
        log_dict = json.loads(formatted)

        # Verify included fields
        self.assertIn("level", log_dict)
        self.assertIn("message", log_dict)
        self.assertIn("context", log_dict)

        # Verify excluded fields
        self.assertNotIn("timestamp", log_dict)

        # Verify fields that weren't included
        self.assertNotIn("logger", log_dict)
        self.assertNotIn("module", log_dict)
        self.assertNotIn("line", log_dict)


class TestDatabaseLogHandler(unittest.TestCase):
    """Test cases for the DatabaseLogHandler class."""

    @patch('utils.logger.sqlite3')
    def setUp(self, mock_sqlite3):
        """Set up test fixtures."""
        # Mock sqlite3 connection and cursor
        self.mock_connection = MagicMock()
        self.mock_cursor = MagicMock()
        self.mock_connection.cursor.return_value = self.mock_cursor
        mock_sqlite3.connect.return_value = self.mock_connection

        # Create a handler
        self.handler = DatabaseLogHandler(database_table="test_logs")

        # Create a record for testing
        self.record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        self.record.threadName = "MainThread"

        # Set up a formatter
        self.handler.setFormatter(logging.Formatter("%(message)s"))

    def test_emit(self):
        """Test emitting a log record."""
        # Emit the record
        self.handler.emit(self.record)

        # Verify that insert was called
        self.mock_cursor.execute.assert_called_once()

        # Get the SQL and parameters
        sql = self.mock_cursor.execute.call_args[0][0]
        params = self.mock_cursor.execute.call_args[0][1]

        # Verify SQL starts with INSERT
        self.assertTrue(sql.startswith("INSERT INTO test_logs"))

        # Verify some parameters
        self.assertEqual(params[1], "test.module")  # logger_name
        self.assertEqual(params[2], "INFO")         # level_name
        self.assertEqual(params[3], 20)             # level_no
        self.assertEqual(params[4], "Test message") # message

        # Verify commit was called
        self.mock_connection.commit.assert_called_once()

    def test_emit_with_context(self):
        """Test emitting a log record with context."""
        # Add context to the record
        self.record.context = {"request_id": "123", "user_id": "456"}

        # Emit the record
        self.handler.emit(self.record)

        # Get the parameters
        params = self.mock_cursor.execute.call_args[0][1]

        # Verify context parameter (index 11)
        context_json = params[11]
        self.assertIsNotNone(context_json)
        context_dict = json.loads(context_json)
        self.assertEqual(context_dict["request_id"], "123")
        self.assertEqual(context_dict["user_id"], "456")

    def test_emit_with_exception(self):
        """Test emitting a log record with exception info."""
        # Add exception info to the record
        try:
            raise ValueError("Test exception")
        except ValueError:
            self.record.exc_info = sys.exc_info()

        # Emit the record
        self.handler.emit(self.record)

        # Get the parameters
        params = self.mock_cursor.execute.call_args[0][1]

        # Verify exception parameter (index 12)
        exception_str = params[12]
        self.assertIsNotNone(exception_str)
        self.assertIn("ValueError: Test exception", exception_str)

    def test_close(self):
        """Test closing the handler."""
        # Close the handler
        self.handler.close()

        # Verify that the connection was closed
        self.mock_connection.close.assert_called_once()


class TestLogLatencyDecorator(unittest.TestCase):
    """Test cases for the log_latency decorator."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock logger
        self.mock_logger = MagicMock(spec=TradingLogger)

        # Patch get_logger to return the mock logger
        self.logger_patcher = patch('utils.logger.get_logger', return_value=self.mock_logger)
        self.mock_get_logger = self.logger_patcher.start()

    def tearDown(self):
        """Tear down test fixtures."""
        self.logger_patcher.stop()

    def test_successful_execution(self):
        """Test decorator with successful function execution."""
        # Create a decorated function
        @log_latency(logger_name="test.function", operation="test_operation")
        def test_function(a, b):
            time.sleep(0.01)  # Ensure measurable duration
            return a + b

        # Call the function
        result = test_function(2, 3)

        # Verify that the function was executed correctly
        self.assertEqual(result, 5)

        # Verify that latency was logged
        self.mock_logger.log.assert_called()
        log_args = self.mock_logger.log.call_args[0]
        self.assertEqual(log_args[0], logging.INFO)
        self.assertIn("test_operation completed in", log_args[1])

        # Verify that log_latency was called
        self.mock_logger.log_latency.assert_called_once()
        latency_args = self.mock_logger.log_latency.call_args[0]
        self.assertEqual(latency_args[0], "test_operation")
        self.assertGreater(latency_args[1], 0)  # duration_ms
        self.assertTrue(latency_args[2])  # success

    def test_failed_execution(self):
        """Test decorator with failed function execution."""
        # Create a decorated function that raises an exception
        @log_latency(logger_name="test.function", error_level=logging.ERROR)
        def failing_function():
            time.sleep(0.01)  # Ensure measurable duration
            raise ValueError("Test error")

        # Call the function and catch the exception
        with self.assertRaises(ValueError):
            failing_function()

        # Verify that latency was logged with error
        self.mock_logger.log.assert_called()
        log_args = self.mock_logger.log.call_args[0]
        self.assertEqual(log_args[0], logging.ERROR)
        self.assertIn("failed after", log_args[1])
        self.assertIn("Test error", log_args[1])

        # Verify that log_latency was called with success=False
        self.mock_logger.log_latency.assert_called_once()
        latency_args = self.mock_logger.log_latency.call_args[0]
        self.assertEqual(latency_args[0], "failing_function")
        self.assertGreater(latency_args[1], 0)  # duration_ms
        self.assertFalse(latency_args[2])  # success=False

    def test_with_include_args(self):
        """Test decorator with include_args=True."""
        # Create a decorated function with include_args=True
        @log_latency(include_args=True)
        def test_function(a, b, c=None):
            return a + b

        # Call the function
        test_function(1, 2, c="test")

        # Verify that args were included in the log message
        self.mock_logger.log.assert_called()
        log_args = self.mock_logger.log.call_args[0]
        self.assertIn("args=(1, 2)", log_args[1])
        self.assertIn("kwargs={'c': 'test'}", log_args[1])

    def test_with_threshold(self):
        """Test decorator with threshold_ms."""
        # Create a decorated function with high threshold
        @log_latency(threshold_ms=1000)  # 1 second threshold
        def fast_function():
            # This should complete much faster than 1 second
            return 42

        # Call the function
        fast_function()

        # Verify that latency was not logged (below threshold)
        self.mock_logger.log.assert_not_called()
        self.mock_logger.log_latency.assert_not_called()

        # Create a decorated function with low threshold
        @log_latency(threshold_ms=0.1)  # 0.1 ms threshold
        def another_function():
            time.sleep(0.01)  # Ensure we exceed the threshold
            return 42

        # Call the function
        another_function()

        # Verify that latency was logged (above threshold)
        self.mock_logger.log.assert_called()
        self.mock_logger.log_latency.assert_called()


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for the utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Patch LoggerManager
        self.manager_patcher = patch('utils.logger.LoggerManager')
        self.MockManagerClass = self.manager_patcher.start()

        # Create mock manager instance
        self.mock_manager = MagicMock()
        self.MockManagerClass.return_value = self.mock_manager

        # Mock logger for get_logger
        self.mock_logger = MagicMock(spec=TradingLogger)
        self.mock_manager.get_logger.return_value = self.mock_logger

        # Mock tracker for track_performance
        self.mock_tracker = MagicMock(spec=PerformanceTracker)
        self.mock_manager.start_performance_tracker.return_value = self.mock_tracker

    def tearDown(self):
        """Tear down test fixtures."""
        self.manager_patcher.stop()

    def test_get_logger(self):
        """Test get_logger utility function."""
        # Call get_logger with explicit name
        logger = get_logger("test.module")

        # Verify manager.get_logger was called with the right name
        self.mock_manager.get_logger.assert_called_with("test.module")

        # Verify returned logger
        self.assertIs(logger, self.mock_logger)

        # Call get_logger with no name (uses caller's module)
        with patch('utils.logger.inspect.currentframe') as mock_frame:
            # Mock the frame's globals to include __name__
            mock_frame.return_value.f_back.f_globals = {'__name__': 'caller.module'}

            logger = get_logger()

            # Verify manager.get_logger was called with caller's module name
            self.mock_manager.get_logger.assert_called_with('caller.module')

    def test_track_performance(self):
        """Test track_performance utility function."""
        # Call track_performance with explicit context_id and component_id
        tracker = track_performance("test_context", "test_component")

        # Verify manager.start_performance_tracker was called with the right args
        self.mock_manager.start_performance_tracker.assert_called_with(
            "test_context", "test_component"
        )

        # Verify returned tracker
        self.assertIs(tracker, self.mock_tracker)

        # Call track_performance with no context_id (uses caller's function)
        with patch('utils.logger.inspect.currentframe') as mock_frame:
            # Mock the frame's globals and code name
            mock_frame.return_value.f_back.f_globals = {'__name__': 'caller.module'}
            mock_frame.return_value.f_back.f_code.co_name = 'test_function'

            tracker = track_performance(component_id="test_component")

            # Verify manager.start_performance_tracker was called with derived context_id
            self.mock_manager.start_performance_tracker.assert_called_with(
                "caller.module.test_function", "test_component"
            )

    @patch('utils.logger.get_logging_config')
    def test_configure_logging(self, mock_get_config):
        """Test configure_logging utility function."""
        # Mock config
        mock_config = MagicMock(spec=LoggingConfig)
        mock_get_config.return_value = mock_config

        # Call configure_logging
        result = configure_logging()

        # Verify get_logging_config was called with no args
        mock_get_config.assert_called_with()

        # Verify manager.update_configuration was called with the config
        self.mock_manager.update_configuration.assert_called_with(mock_config)

        # Verify returned config
        self.assertEqual(result, mock_config)

        # Call configure_logging with config_path
        result = configure_logging("custom/path")

        # Verify get_logging_config was called with the path
        mock_get_config.assert_called_with("custom/path")

    @patch('utils.logger.get_logger')
    def test_create_component_logger(self, mock_get_logger):
        """Test create_component_logger utility function."""
        # Mock logger
        mock_logger = MagicMock(spec=TradingLogger)
        mock_logger.with_context.return_value = mock_logger
        mock_get_logger.return_value = mock_logger

        # Mock health monitor
        mock_health_monitor = MagicMock()
        self.mock_manager.get_health_monitor.return_value = mock_health_monitor

        # Call create_component_logger
        result = create_component_logger("test_component", "strategy", {"param": "value"})

        # Verify get_logger was called with component name
        mock_get_logger.assert_called_with("component.test_component")

        # Verify with_context was called with component info
        mock_logger.with_context.assert_called_with(
            component_id="test_component",
            component_type="strategy",
            param="value"
        )

        # Verify health_monitor.register_component was called
        mock_health_monitor.register_component.assert_called_with(
            "test_component", "strategy"
        )

        # Verify returned logger
        self.assertIs(result, mock_logger)


class TestIntegration(unittest.TestCase):
    """Integration tests for the logging module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()

        # Create a log path
        self.log_path = os.path.join(self.temp_dir, "test.log")

        # Set up a file handler pointed at our temp directory
        self.handler = logging.FileHandler(self.log_path)
        self.handler.setLevel(logging.DEBUG)
        self.handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

        # Add the handler to the root logger
        self.root_logger = logging.getLogger()
        self.root_logger.addHandler(self.handler)

        # Save original level
        self.original_level = self.root_logger.level
        self.root_logger.setLevel(logging.DEBUG)

        # Patch LoggingConfig to use our temp directory
        self.config_patcher = patch('utils.logger.get_logging_config')
        self.mock_get_config = self.config_patcher.start()

        # Create config with our handler
        mock_config = MagicMock(spec=LoggingConfig)
        mock_config.root_level = LogLevel.DEBUG
        mock_config.log_dir = self.temp_dir
        mock_config.log_rotation.enable_rotation = False
        mock_config.capture_warnings = True
        mock_config.log_exceptions = True
        mock_config.audit.enable_audit_logging = True
        mock_config.audit.audit_logger_name = "audit"

        # Configure pythong logging dict config to use our handler
        python_config = {
            'version': 1,
            'handlers': {
                'file': {
                    'class': 'logging.FileHandler',
                    'filename': self.log_path,
                    'formatter': 'standard',
                }
            },
            'formatters': {
                'standard': {
                    'format': '%(levelname)s - %(message)s'
                }
            },
            'root': {
                'handlers': ['file'],
                'level': 'DEBUG',
            }
        }
        mock_config.get_python_logging_config.return_value = python_config

        self.mock_get_config.return_value = mock_config

        # Reset LoggerManager singleton for testing
        LoggerManager._instance = None
        LoggerManager._initialized = False
        LoggerManager._loggers = {}

    def tearDown(self):
        """Tear down test fixtures."""
        # Reset root logger level
        self.root_logger.setLevel(self.original_level)

        # Remove our handler
        self.root_logger.removeHandler(self.handler)

        # Stop patches
        self.config_patcher.stop()

        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

    def test_end_to_end_logging(self):
        """End-to-end test of logging system."""
        # Initialize logging system
        configure_logging()

        # Create a logger
        logger = get_logger("test.integration")

        # Log messages at different levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

        # Create a logger with context
        context_logger = logger.with_context(request_id="123", user="testuser")
        context_logger.info("Message with context")

        # Use audit logging
        context_logger.audit("Audit message", operation="test")

        # Use performance tracking
        tracker = track_performance("test_operation")
        tracker.checkpoint("start")
        time.sleep(0.01)
        tracker.checkpoint("middle")
        time.sleep(0.01)
        tracker.checkpoint("end")
        tracker.log_summary()

        # Close the handler to ensure all data is written
        self.handler.close()

        # Read the log file
        with open(self.log_path, 'r') as f:
            log_content = f.read()

        # Verify that all messages appear in the log
        self.assertIn("DEBUG - Debug message", log_content)
        self.assertIn("INFO - Info message", log_content)
        self.assertIn("WARNING - Warning message", log_content)
        self.assertIn("ERROR - Error message", log_content)
        self.assertIn("INFO - Message with context", log_content)
        self.assertIn("INFO - Audit message", log_content)
        self.assertIn("INFO - Performance summary", log_content)


if __name__ == '__main__':
    unittest.main()