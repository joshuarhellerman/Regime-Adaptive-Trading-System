"""
Test module for the System class in system.py.

This module contains tests for the System class, which acts as
the central coordinator for the trading system.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import time
import threading

# Import the System class and other dependencies
from system import System
from core.event_bus import EventTopics, create_event, Event
from core.state_manager import StateScope

class TestSystem(unittest.TestCase):
    """Test cases for the System class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mocks for all dependencies
        self.mock_event_bus = MagicMock()
        self.mock_state_manager = MagicMock()
        self.mock_component_registry = MagicMock()
        self.mock_market_data_service = MagicMock()

        # Patch the get_* functions to return our mocks
        self.patcher1 = patch('system.get_event_bus', return_value=self.mock_event_bus)
        self.patcher2 = patch('system.get_state_manager', return_value=self.mock_state_manager)
        self.patcher3 = patch('system.get_component_registry', return_value=self.mock_component_registry)
        self.patcher4 = patch('system.get_market_data_service', return_value=self.mock_market_data_service)

        # Start all the patchers
        self.patcher1.start()
        self.patcher2.start()
        self.patcher3.start()
        self.patcher4.start()

        # Create a test config
        self.test_config = {
            "critical_components": ["market_data_service", "strategy_engine"],
            "other_config": "value"
        }

        # Create an instance of System
        self.system = System(self.test_config, mode="paper_trading")

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        # Stop all the patchers
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        self.patcher4.stop()

    def test_init(self):
        """Test the initialization of the System class."""
        # Verify that the system was initialized correctly
        self.assertEqual(self.system.config, self.test_config)
        self.assertEqual(self.system.mode, "paper_trading")
        self.assertFalse(self.system.is_running)
        self.assertIsNone(self.system.start_time)
        self.assertFalse(self.system.shutdown_requested)

        # Verify that the event bus subscriptions were set up
        self.mock_event_bus.subscribe.assert_any_call(EventTopics.SYSTEM_SHUTDOWN, self.system._handle_shutdown_request)
        self.mock_event_bus.subscribe.assert_any_call(EventTopics.COMPONENT_ERROR, self.system._handle_component_error)

        # Verify that the state manager was updated
        self.mock_state_manager.set.assert_any_call("system.mode", "paper_trading", StateScope.PERSISTENT)
        self.mock_state_manager.set.assert_any_call("system.config", self.test_config, StateScope.PERSISTENT)
        self.mock_state_manager.set.assert_any_call("system.status", "initialized", StateScope.PERSISTENT)

    def test_start(self):
        """Test the start method of the System class."""
        # Configure mocks for successful start
        self.mock_component_registry.start_all.return_value = True

        # Call the start method
        result = self.system.start()

        # Verify the result
        self.assertTrue(result)
        self.assertTrue(self.system.is_running)
        self.assertIsNotNone(self.system.start_time)
        self.assertFalse(self.system.shutdown_requested)

        # Verify that the core services were started
        self.mock_event_bus.start.assert_called_once()
        self.mock_state_manager.start.assert_called_once()

        # Verify that the market data service was initialized and registered
        self.mock_component_registry.register.assert_any_call(
            self.mock_market_data_service,
            "market_data_service",
            "data_service"
        )

        # Verify that all components were started
        self.mock_component_registry.start_all.assert_called_once()

        # Verify that the state manager was updated
        self.mock_state_manager.set.assert_any_call("system.status", "starting", StateScope.PERSISTENT)
        self.mock_state_manager.set.assert_any_call("system.status", "running", StateScope.PERSISTENT)
        self.mock_state_manager.set.assert_any_call("system.start_time", self.system.start_time, StateScope.PERSISTENT)

        # Verify that the start event was published
        self.mock_event_bus.publish.assert_called_once()
        actual_event = self.mock_event_bus.publish.call_args[0][0]
        self.assertEqual(actual_event.topic, EventTopics.SYSTEM_START)
        self.assertEqual(actual_event.data["mode"], "paper_trading")
        self.assertEqual(actual_event.data["config"], self.test_config)

    def test_start_already_running(self):
        """Test the start method when the system is already running."""
        # Set the system as running
        self.system.is_running = True

        # Call the start method
        result = self.system.start()

        # Verify the result
        self.assertTrue(result)

        # Verify that no core services were started
        self.mock_event_bus.start.assert_not_called()
        self.mock_state_manager.start.assert_not_called()
        self.mock_component_registry.start_all.assert_not_called()

    def test_start_failure(self):
        """Test the start method when an error occurs."""
        # Configure mocks for failure
        self.mock_event_bus.start.side_effect = Exception("Test exception")

        # Call the start method
        result = self.system.start()

        # Verify the result
        self.assertFalse(result)
        self.assertFalse(self.system.is_running)

        # Verify that an attempt was made to stop components
        # We can't directly check for stop() call as it might be called in a
        # different context during the error handling

        # Verify that the state manager was updated with error information
        self.mock_state_manager.set.assert_any_call("system.status", "error", StateScope.PERSISTENT)
        self.mock_state_manager.set.assert_any_call("system.error", "Test exception", StateScope.PERSISTENT)

    def test_stop(self):
        """Test the stop method of the System class."""
        # Set the system as running
        self.system.is_running = True
        self.system.start_time = time.time() - 60  # 60 seconds ago

        # Call the stop method
        result = self.system.stop()

        # Verify the result
        self.assertTrue(result)
        self.assertFalse(self.system.is_running)

        # Verify that all components were stopped
        self.mock_component_registry.stop_all.assert_called_once()

        # Verify that the core services were stopped
        self.mock_state_manager.stop.assert_called_once()
        self.mock_event_bus.stop.assert_called_once()

        # Verify that the stop event was published
        self.mock_event_bus.publish_sync.assert_called_once()
        actual_event = self.mock_event_bus.publish_sync.call_args[0][0]
        self.assertEqual(actual_event.topic, EventTopics.SYSTEM_SHUTDOWN)
        self.assertEqual(actual_event.data["mode"], "paper_trading")
        self.assertAlmostEqual(actual_event.data["uptime"], 60, delta=2)
        self.assertTrue(actual_event.data["clean_shutdown"])

        # Verify that the state manager was updated
        self.mock_state_manager.set.assert_any_call("system.status", "stopping", StateScope.PERSISTENT)
        self.mock_state_manager.set.assert_any_call("system.status", "stopped", StateScope.PERSISTENT)
        # End time and uptime are harder to verify exactly, but we can check they were set
        self.mock_state_manager.set.assert_any_call("system.end_time", self.mock_state_manager.set.call_args_list[-2][0][1], StateScope.PERSISTENT)
        self.mock_state_manager.set.assert_any_call("system.uptime", self.mock_state_manager.set.call_args_list[-1][0][1], StateScope.PERSISTENT)

    def test_stop_not_running(self):
        """Test the stop method when the system is not running."""
        # Ensure the system is not running
        self.system.is_running = False

        # Call the stop method
        result = self.system.stop()

        # Verify the result
        self.assertTrue(result)

        # Verify that no components were stopped
        self.mock_component_registry.stop_all.assert_not_called()

        # Verify that no core services were stopped
        self.mock_state_manager.stop.assert_not_called()
        self.mock_event_bus.stop.assert_not_called()

    def test_stop_failure(self):
        """Test the stop method when an error occurs."""
        # Set the system as running
        self.system.is_running = True
        self.system.start_time = time.time() - 60  # 60 seconds ago

        # Configure mocks for failure
        self.mock_component_registry.stop_all.side_effect = Exception("Test exception")

        # Call the stop method
        result = self.system.stop()

        # Verify the result
        self.assertFalse(result)
        self.assertFalse(self.system.is_running)  # System should still be marked as not running

        # Verify that the state manager was updated with error information
        self.mock_state_manager.set.assert_any_call("system.status", "error", StateScope.PERSISTENT)
        self.mock_state_manager.set.assert_any_call("system.error", "Test exception", StateScope.PERSISTENT)

    def test_restart(self):
        """Test the restart method of the System class."""
        # Mock stop and start methods
        self.system.stop = Mock(return_value=True)
        self.system.start = Mock(return_value=True)

        # Call the restart method
        result = self.system.restart()

        # Verify the result
        self.assertTrue(result)

        # Verify that stop and start were called
        self.system.stop.assert_called_once()
        self.system.start.assert_called_once()

    def test_restart_stop_failure(self):
        """Test the restart method when stop fails."""
        # Mock stop and start methods
        self.system.stop = Mock(return_value=False)
        self.system.start = Mock(return_value=True)

        # Call the restart method
        result = self.system.restart()

        # Verify the result
        self.assertFalse(result)

        # Verify that stop was called but start was not
        self.system.stop.assert_called_once()
        self.system.start.assert_not_called()

    def test_get_status(self):
        """Test the get_status method of the System class."""
        # Set up the system
        self.system.is_running = True
        self.system.start_time = time.time() - 60  # 60 seconds ago

        # Mock component registry
        self.mock_component_registry.get_component_names.return_value = ["component1", "component2"]
        self.mock_component_registry.get_component_info.side_effect = lambda name: {"name": name, "status": "running"}

        # Mock _get_system_resources
        with patch.object(self.system, '_get_system_resources', return_value={"cpu_percent": 5.0}):
            # Call the get_status method
            status = self.system.get_status()

        # Verify the status
        self.assertEqual(status["mode"], "paper_trading")
        self.assertTrue(status["is_running"])
        self.assertAlmostEqual(status["uptime"], 60, delta=2)
        self.assertEqual(len(status["components"]), 2)
        self.assertEqual(status["components"]["component1"]["name"], "component1")
        self.assertEqual(status["components"]["component2"]["name"], "component2")
        self.assertEqual(status["system_resources"]["cpu_percent"], 5.0)

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_connections')
    @patch('psutil.Process')
    def test_get_system_resources(self, mock_process, mock_net_connections, mock_disk_usage,
                                 mock_virtual_memory, mock_cpu_percent):
        """Test the _get_system_resources method of the System class."""
        # Configure mocks
        mock_cpu_percent.return_value = 10.0

        mock_memory = Mock()
        mock_memory.total = 8000000000
        mock_memory.available = 4000000000
        mock_memory.percent = 50.0
        mock_memory.used = 4000000000
        mock_virtual_memory.return_value = mock_memory

        mock_disk = Mock()
        mock_disk.total = 500000000000
        mock_disk.used = 200000000000
        mock_disk.free = 300000000000
        mock_disk.percent = 40.0
        mock_disk_usage.return_value = mock_disk

        mock_net_connections.return_value = [1, 2, 3]  # 3 connections

        mock_proc = Mock()
        mock_proc.cpu_percent.return_value = 5.0
        mock_proc.memory_percent.return_value = 2.5
        mock_proc.num_threads.return_value = 10
        mock_process.return_value = mock_proc

        # Call the _get_system_resources method
        resources = self.system._get_system_resources()

        # Verify the resources
        self.assertEqual(resources["cpu_percent"], 10.0)

        self.assertEqual(resources["memory"]["total"], 8000000000)
        self.assertEqual(resources["memory"]["available"], 4000000000)
        self.assertEqual(resources["memory"]["percent"], 50.0)
        self.assertEqual(resources["memory"]["used"], 4000000000)

        self.assertEqual(resources["disk"]["total"], 500000000000)
        self.assertEqual(resources["disk"]["used"], 200000000000)
        self.assertEqual(resources["disk"]["free"], 300000000000)
        self.assertEqual(resources["disk"]["percent"], 40.0)

        self.assertEqual(resources["network"]["connections"], 3)

        self.assertEqual(resources["process"]["cpu_percent"], 5.0)
        self.assertEqual(resources["process"]["memory_percent"], 2.5)
        self.assertEqual(resources["process"]["threads"], 10)

    def test_execute_command_start(self):
        """Test the execute_command method with the start command."""
        # Mock the start method
        self.system.start = Mock(return_value=True)

        # Call the execute_command method
        result = self.system.execute_command("start")

        # Verify the result
        self.assertTrue(result["success"])
        self.assertEqual(result["command"], "start")
        self.system.start.assert_called_once()

    def test_execute_command_stop(self):
        """Test the execute_command method with the stop command."""
        # Mock the stop method
        self.system.stop = Mock(return_value=True)

        # Call the execute_command method
        result = self.system.execute_command("stop")

        # Verify the result
        self.assertTrue(result["success"])
        self.assertEqual(result["command"], "stop")
        self.system.stop.assert_called_once()

    def test_execute_command_restart(self):
        """Test the execute_command method with the restart command."""
        # Mock the restart method
        self.system.restart = Mock(return_value=True)

        # Call the execute_command method
        result = self.system.execute_command("restart")

        # Verify the result
        self.assertTrue(result["success"])
        self.assertEqual(result["command"], "restart")
        self.system.restart.assert_called_once()

    def test_execute_command_status(self):
        """Test the execute_command method with the status command."""
        # Mock the get_status method
        status_data = {"mode": "paper_trading", "is_running": True}
        self.system.get_status = Mock(return_value=status_data)

        # Call the execute_command method
        result = self.system.execute_command("status")

        # Verify the result
        self.assertTrue(result["success"])
        self.assertEqual(result["command"], "status")
        self.assertEqual(result["status"], status_data)
        self.system.get_status.assert_called_once()

    def test_execute_command_component_command(self):
        """Test the execute_command method with the component_command command."""
        # Set up mock component
        mock_component = Mock()
        mock_component.execute_command.return_value = {"success": True, "result": "OK"}
        self.mock_component_registry.get.return_value = mock_component

        # Call the execute_command method
        result = self.system.execute_command("component_command", {
            "component": "test_component",
            "component_command": "test_cmd",
            "component_params": {"param1": "value1"}
        })

        # Verify the result
        self.assertTrue(result["success"])
        self.assertEqual(result["command"], "component_command")
        self.assertEqual(result["component_result"]["result"], "OK")

        # Verify that the component method was called
        self.mock_component_registry.get.assert_called_once_with("test_component")
        mock_component.execute_command.assert_called_once_with("test_cmd", {"param1": "value1"})

    def test_execute_command_component_command_missing_params(self):
        """Test the execute_command method with the component_command command but missing parameters."""
        # Call the execute_command method with missing component name
        result = self.system.execute_command("component_command", {})

        # Verify the result
        self.assertFalse(result["success"])
        self.assertEqual(result["command"], "component_command")
        self.assertIn("error", result)

        # Verify that the component registry was not called
        self.mock_component_registry.get.assert_not_called()

    def test_execute_command_component_command_component_not_found(self):
        """Test the execute_command method with the component_command command but component not found."""
        # Configure mock to return None (component not found)
        self.mock_component_registry.get.return_value = None

        # Call the execute_command method
        result = self.system.execute_command("component_command", {
            "component": "nonexistent_component",
            "component_command": "test_cmd"
        })

        # Verify the result
        self.assertFalse(result["success"])
        self.assertEqual(result["command"], "component_command")
        self.assertIn("error", result)

        # Verify that the component registry was called
        self.mock_component_registry.get.assert_called_once_with("nonexistent_component")

    def test_execute_command_unknown(self):
        """Test the execute_command method with an unknown command."""
        # Call the execute_command method with an unknown command
        result = self.system.execute_command("unknown_command")

        # Verify the result
        self.assertFalse(result["success"])
        self.assertEqual(result["command"], "unknown_command")
        self.assertIn("error", result)

    def test_register_component(self):
        """Test the register_component method of the System class."""
        # Mock component
        mock_component = Mock()

        # System not running
        self.system.is_running = False

        # Call the register_component method
        result = self.system.register_component("test_component", mock_component, "test_type", ["dependency1"])

        # Verify the result
        self.assertTrue(result)

        # Verify that the component was registered
        self.mock_component_registry.register.assert_called_once_with(
            mock_component, "test_component", "test_type", ["dependency1"]
        )

        # Verify that the component was not started (system not running)
        self.mock_component_registry.start.assert_not_called()

    def test_register_component_system_running(self):
        """Test the register_component method when the system is running."""
        # Mock component
        mock_component = Mock()

        # System is running
        self.system.is_running = True

        # Call the register_component method
        result = self.system.register_component("test_component", mock_component)

        # Verify the result
        self.assertTrue(result)

        # Verify that the component was registered
        self.mock_component_registry.register.assert_called_once_with(
            mock_component, "test_component", None, None
        )

        # Verify that the component was started (system is running)
        self.mock_component_registry.start.assert_called_once_with("test_component")

    def test_register_component_failure(self):
        """Test the register_component method when registration fails."""
        # Mock component
        mock_component = Mock()

        # Configure mock to raise an exception
        self.mock_component_registry.register.side_effect = Exception("Registration failed")

        # Call the register_component method
        result = self.system.register_component("test_component", mock_component)

        # Verify the result
        self.assertFalse(result)

    def test_unregister_component(self):
        """Test the unregister_component method of the System class."""
        # System not running
        self.system.is_running = False

        # Configure mock to return True (successful unregistration)
        self.mock_component_registry.unregister.return_value = True

        # Call the unregister_component method
        result = self.system.unregister_component("test_component")

        # Verify the result
        self.assertTrue(result)

        # Verify that the component was unregistered
        self.mock_component_registry.unregister.assert_called_once_with("test_component")

        # Verify that the component was not stopped (system not running)
        self.mock_component_registry.stop.assert_not_called()

    def test_unregister_component_system_running(self):
        """Test the unregister_component method when the system is running."""
        # System is running
        self.system.is_running = True

        # Configure mock to return True (successful unregistration)
        self.mock_component_registry.unregister.return_value = True

        # Call the unregister_component method
        result = self.system.unregister_component("test_component")

        # Verify the result
        self.assertTrue(result)

        # Verify that the component was first stopped
        self.mock_component_registry.stop.assert_called_once_with("test_component")

        # Verify that the component was unregistered
        self.mock_component_registry.unregister.assert_called_once_with("test_component")

    def test_unregister_component_failure(self):
        """Test the unregister_component method when unregistration fails."""
        # System not running
        self.system.is_running = False

        # Configure mock to return False (failed unregistration)
        self.mock_component_registry.unregister.return_value = False

        # Call the unregister_component method
        result = self.system.unregister_component("test_component")

        # Verify the result
        self.assertFalse(result)

        # Verify that the unregister method was called
        self.mock_component_registry.unregister.assert_called_once_with("test_component")

    def test_get_components(self):
        """Test the get_components method of the System class."""
        # Mock component names and components
        self.mock_component_registry.get_component_names.return_value = ["comp1", "comp2"]
        self.mock_component_registry.get.side_effect = lambda name: Mock(name=name)

        # Call the get_components method
        components = self.system.get_components()

        # Verify the components
        self.assertEqual(len(components), 2)
        self.assertIn("comp1", components)
        self.assertIn("comp2", components)
        self.assertEqual(components["comp1"].name, "comp1")
        self.assertEqual(components["comp2"].name, "comp2")

    def test_get_component(self):
        """Test the get_component method of the System class."""
        # Mock component
        mock_component = Mock(name="test_component")
        self.mock_component_registry.get.return_value = mock_component

        # Call the get_component method
        component = self.system.get_component("test_component")

        # Verify the component
        self.assertEqual(component, mock_component)
        self.mock_component_registry.get.assert_called_once_with("test_component")

    def test_handle_shutdown_request(self):
        """Test the _handle_shutdown_request method of the System class."""
        # Create a mock event
        mock_event = Mock(spec=Event)

        # Set up to test that stop is called via threading
        def mock_thread_target(*args, **kwargs):
            self.thread_target_called = True

        original_thread = threading.Thread
        try:
            # Replace threading.Thread with our mock
            threading.Thread = Mock()
            threading.Thread.return_value = Mock()

            # Call the _handle_shutdown_request method
            self.system._handle_shutdown_request(mock_event)

            # Verify that shutdown was requested
            self.assertTrue(self.system.shutdown_requested)

            # Verify that a thread was created to stop the system
            threading.Thread.assert_called_once()
            self.assertEqual(threading.Thread.call_args[1]["target"], self.system.stop)
            self.assertTrue(threading.Thread.call_args[1]["daemon"])
            threading.Thread.return_value.start.assert_called_once()

        finally:
            # Restore original threading.Thread
            threading.Thread = original_thread

    def test_handle_shutdown_request_already_in_progress(self):
        """Test the _handle_shutdown_request method when shutdown is already in progress."""
        # Set shutdown_requested to True
        self.system.shutdown_requested = True

        # Create a mock event
        mock_event = Mock(spec=Event)

        # Call the _handle_shutdown_request method
        self.system._handle_shutdown_request(mock_event)

        # Nothing should have happened (no exceptions)

    def test_handle_component_error(self):
        """Test the _handle_component_error method of the System class."""
        # Create a mock event with error data
        mock_event = Mock(spec=Event)
        mock_event.data = {
            "name": "test_component",
            "error": "Test error message"
        }

        # Call the _handle_component_error method
        self.system._handle_component_error(mock_event)

        # Verify that the error was recorded in the state manager
        self.mock_state_manager.set.assert_any_call(
            "system.component_errors.test_component",
            {
                "message": "Test error message",
                "timestamp": self.mock_state_manager.set.call_args[0][1]["timestamp"]
            },
            StateScope.PERSISTENT
        )

        # Verify that no shutdown was requested (non-critical component)
        self.assertFalse(self.system.shutdown_requested)

    def test_handle_component_error_critical_component(self):
        """Test the _handle_component_error method with a critical component."""
        # Create a mock event with error data for a critical component
        mock_event = Mock(spec=Event)
        mock_event.data = {
            "name": "market_data_service",  # This is in the critical_components list
            "error": "Critical error message"
        }

        # Set up to test that stop is called via threading
        original_thread = threading.Thread
        try:
            # Replace threading.Thread with our mock
            threading.Thread = Mock()
            threading.Thread.return_value = Mock()

            # Call the _handle_component_error method
            self.system._handle_component_error(mock_event)

            # Verify that the error was recorded in the state manager
            self.mock_state_manager.set.assert_any_call(
                "system.component_errors.market_data_service",
                {
                    "message": "Critical error message",
                    "timestamp": self.mock_state_manager.set.call_args[0][1]["timestamp"]
                },
                StateScope.PERSISTENT
            )

            # Verify that shutdown was requested (critical component)
            self.assertTrue(self.system.shutdown_requested)

            # Verify that a thread was created to stop the system
            threading.Thread.assert_called_once()
            self.assertEqual(threading.Thread.call_args[1]["target"], self.system.stop)
            self.assertTrue(threading.Thread.call_args[1]["daemon"])
            threading.Thread.return_value.start.assert_called_once()

        finally:
            # Restore original threading.Thread
            threading.Thread = original_thread


if __name__ == "__main__":
    unittest.main()