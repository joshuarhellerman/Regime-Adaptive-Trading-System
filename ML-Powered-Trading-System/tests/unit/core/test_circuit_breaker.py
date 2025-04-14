"""
Tests for the circuit_breaker module.
"""

import os
import sys
# Add project root to Python path - adjust the parent directory navigation as needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import unittest
from unittest.mock import Mock, patch, MagicMock
import time
import threading
from datetime import datetime

from core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitType,
    TriggerType,
    CircuitEvent,
    CircuitBreakerRegistry,
    get_circuit_breaker_registry,
    create_circuit_breaker
)
from core.event_bus import Event, EventPriority
from core.health_monitor import HealthStatus, AlertLevel


class TestCircuitBreaker(unittest.TestCase):
    """Test cases for the CircuitBreaker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.event_bus = Mock()
        self.health_monitor = Mock()

        # Create a basic config for tests
        self.config = CircuitBreakerConfig(
            name="test_circuit",
            circuit_type=CircuitType.SYSTEM,
            scope="test_scope",
            trigger_conditions={
                TriggerType.LOSS_THRESHOLD: {"threshold": -1000.0},
                TriggerType.ERROR_RATE: {"error_count": 0, "error_window": 5, "threshold": 0.6},
                TriggerType.LATENCY: {"threshold": 500.0}
            },
            recovery_time=1,  # Short time for testing
            auto_recovery=True
        )

        # Create callbacks
        self.on_open_called = False
        self.on_close_called = False

        def on_open(event):
            self.on_open_called = True

        def on_close(event):
            self.on_close_called = True

        self.on_open = on_open
        self.on_close = on_close

        # Create circuit breaker
        self.circuit_breaker = CircuitBreaker(
            config=self.config,
            health_monitor=self.health_monitor,
            event_bus=self.event_bus,
            on_open=self.on_open,
            on_close=self.on_close
        )

    def test_initial_state(self):
        """Test the initial state of the circuit breaker."""
        self.assertEqual(self.circuit_breaker.get_state(), CircuitState.CLOSED)
        self.assertTrue(self.circuit_breaker.is_closed())
        self.assertFalse(self.circuit_breaker.is_open())
        self.assertFalse(self.circuit_breaker.is_half_open())

        # Check that the event bus subscriptions were registered
        self.event_bus.subscribe.assert_called()

        # Check that the health monitor was updated
        self.health_monitor.register_component.assert_called_once()
        self.health_monitor.update_component_health.assert_called_once()

    def test_trip(self):
        """Test tripping the circuit breaker."""
        # Trip the circuit
        result = self.circuit_breaker.trip(
            trigger_type=TriggerType.LOSS_THRESHOLD,
            trigger_value=-1500.0,
            reason="Test trip"
        )

        # Check the result and state
        self.assertTrue(result)
        self.assertEqual(self.circuit_breaker.get_state(), CircuitState.OPEN)
        self.assertTrue(self.circuit_breaker.is_open())
        self.assertFalse(self.circuit_breaker.is_closed())

        # Check that callbacks were called
        self.assertTrue(self.on_open_called)
        self.assertFalse(self.on_close_called)

        # Check that events were published
        self.event_bus.publish.assert_called_once()

        # Check that health monitor was updated
        self.health_monitor.update_component_health.assert_called()

        # Trip again should return False (already open)
        result = self.circuit_breaker.trip(
            trigger_type=TriggerType.LOSS_THRESHOLD,
            trigger_value=-2000.0,
            reason="Test trip again"
        )
        self.assertFalse(result)

    def test_reset(self):
        """Test manually resetting the circuit breaker."""
        # First trip the circuit
        self.circuit_breaker.trip(
            trigger_type=TriggerType.LOSS_THRESHOLD,
            trigger_value=-1500.0,
            reason="Test trip"
        )

        # Reset the circuit
        reset_count = self.event_bus.publish.call_count
        result = self.circuit_breaker.reset(user_id="test_user", reason="Test reset")

        # Check the result and state
        self.assertTrue(result)
        self.assertEqual(self.circuit_breaker.get_state(), CircuitState.CLOSED)
        self.assertTrue(self.circuit_breaker.is_closed())

        # Check that callbacks were called
        self.assertTrue(self.on_close_called)

        # Check that events were published
        self.assertEqual(self.event_bus.publish.call_count, reset_count + 1)

        # Reset again should return False (already closed)
        result = self.circuit_breaker.reset(user_id="test_user", reason="Test reset again")
        self.assertFalse(result)

    def test_test_condition(self):
        """Test the test_condition method with different trigger types."""
        # Test LOSS_THRESHOLD
        self.assertTrue(self.circuit_breaker.test_condition(TriggerType.LOSS_THRESHOLD, -1500.0))
        self.assertFalse(self.circuit_breaker.test_condition(TriggerType.LOSS_THRESHOLD, -500.0))

        # Test ERROR_RATE - error_count starts at 0 and is incremented
        self.assertFalse(self.circuit_breaker.test_condition(TriggerType.ERROR_RATE, {"type": "error"}))
        self.assertFalse(self.circuit_breaker.test_condition(TriggerType.ERROR_RATE, {"type": "error"}))
        self.assertTrue(self.circuit_breaker.test_condition(TriggerType.ERROR_RATE, {"type": "error"}))

        # Test LATENCY
        self.assertTrue(self.circuit_breaker.test_condition(TriggerType.LATENCY, 600.0))
        self.assertFalse(self.circuit_breaker.test_condition(TriggerType.LATENCY, 400.0))

        # Test non-configured trigger type
        self.assertFalse(self.circuit_breaker.test_condition(TriggerType.VOLATILITY, 0.5))

    def test_auto_recovery(self):
        """Test automatic recovery process."""
        # Trip the circuit
        self.circuit_breaker.trip(
            trigger_type=TriggerType.LOSS_THRESHOLD,
            trigger_value=-1500.0,
            reason="Test trip"
        )

        # Wait for recovery attempt
        time.sleep(1.5)

        # Circuit should be in half-open state
        self.assertEqual(self.circuit_breaker.get_state(), CircuitState.HALF_OPEN)
        self.assertTrue(self.circuit_breaker.is_half_open())

        # Test successful recovery
        self.circuit_breaker.test_recovery(True, "Recovery test passed")

        # Circuit should be closed
        self.assertEqual(self.circuit_breaker.get_state(), CircuitState.CLOSED)
        self.assertTrue(self.circuit_breaker.is_closed())

    def test_failed_recovery(self):
        """Test failed recovery process."""
        # Trip the circuit
        self.circuit_breaker.trip(
            trigger_type=TriggerType.LOSS_THRESHOLD,
            trigger_value=-1500.0,
            reason="Test trip"
        )

        # Wait for recovery attempt
        time.sleep(1.5)

        # Circuit should be in half-open state
        self.assertEqual(self.circuit_breaker.get_state(), CircuitState.HALF_OPEN)

        # Test failed recovery
        self.circuit_breaker.test_recovery(False, "Recovery test failed")

        # Circuit should be open again
        self.assertEqual(self.circuit_breaker.get_state(), CircuitState.OPEN)
        self.assertTrue(self.circuit_breaker.is_open())

        # Check that consecutive failures is incremented
        self.assertEqual(self.circuit_breaker._consecutive_failures, 1)

    def test_max_consecutive_failures(self):
        """Test max consecutive failures logic."""
        # Set max consecutive failures to 2
        self.circuit_breaker.config.max_consecutive_failures = 2

        # Trip the circuit
        self.circuit_breaker.trip(
            trigger_type=TriggerType.LOSS_THRESHOLD,
            trigger_value=-1500.0,
            reason="Test trip"
        )

        # Wait for recovery attempt
        time.sleep(1.5)

        # Fail recovery once
        self.circuit_breaker.test_recovery(False, "First failure")
        self.assertEqual(self.circuit_breaker._consecutive_failures, 1)

        # Wait for next recovery attempt
        time.sleep(1.5)

        # Fail recovery again
        recovery_timer_before = self.circuit_breaker._recovery_timer
        self.circuit_breaker.test_recovery(False, "Second failure")
        self.assertEqual(self.circuit_breaker._consecutive_failures, 2)

        # There should be no more recovery attempts
        self.assertEqual(self.circuit_breaker._recovery_timer, None)

    def test_get_metrics(self):
        """Test getting circuit breaker metrics."""
        # Trip the circuit
        self.circuit_breaker.trip(
            trigger_type=TriggerType.LOSS_THRESHOLD,
            trigger_value=-1500.0,
            reason="Test trip"
        )

        # Get metrics
        metrics = self.circuit_breaker.get_metrics()

        # Check metrics
        self.assertEqual(metrics["circuit_id"], self.circuit_breaker.circuit_id)
        self.assertEqual(metrics["circuit_type"], self.config.circuit_type.value)
        self.assertEqual(metrics["scope"], self.config.scope)
        self.assertEqual(metrics["state"], CircuitState.OPEN.value)
        self.assertEqual(metrics["consecutive_failures"], 0)
        self.assertEqual(metrics["metrics"]["total_trips"], 1)
        self.assertTrue("current_state_since" in metrics)

    def test_get_events(self):
        """Test getting circuit breaker events."""
        # Trip the circuit
        self.circuit_breaker.trip(
            trigger_type=TriggerType.LOSS_THRESHOLD,
            trigger_value=-1500.0,
            reason="Test trip"
        )

        # Reset the circuit
        self.circuit_breaker.reset(user_id="test_user", reason="Test reset")

        # Get events
        events = self.circuit_breaker.get_events()

        # Should have 2 events
        self.assertEqual(len(events), 2)

        # Check event data
        self.assertEqual(events[0]["old_state"], CircuitState.OPEN.value)
        self.assertEqual(events[0]["new_state"], CircuitState.CLOSED.value)
        self.assertEqual(events[0]["user_id"], "test_user")

        self.assertEqual(events[1]["old_state"], CircuitState.CLOSED.value)
        self.assertEqual(events[1]["new_state"], CircuitState.OPEN.value)
        self.assertEqual(events[1]["trigger_type"], TriggerType.LOSS_THRESHOLD.value)

    def test_event_handlers(self):
        """Test event handlers for different circuit types."""
        # Create different circuit breakers for testing
        strategy_config = CircuitBreakerConfig(
            name="strategy_circuit",
            circuit_type=CircuitType.STRATEGY,
            scope="test_strategy",
            trigger_conditions={
                TriggerType.MODEL_DRIFT: {"threshold": 0.3}
            }
        )
        strategy_cb = CircuitBreaker(
            config=strategy_config,
            event_bus=self.event_bus
        )

        # Test strategy event handler
        event = Event(
            topic="strategy.signal_generated.test_strategy",
            data={"confidence": 0.2}
        )
        strategy_cb._handle_strategy_event(event)

        # The circuit should be tripped
        self.assertTrue(strategy_cb.is_open())

    def test_callbacks(self):
        """Test that callbacks are executed properly."""
        # Reset callback tracking
        self.on_open_called = False
        self.on_close_called = False

        # Trip the circuit
        self.circuit_breaker.trip(
            trigger_type=TriggerType.LOSS_THRESHOLD,
            trigger_value=-1500.0,
            reason="Test trip"
        )

        # Check that open callback was called
        self.assertTrue(self.on_open_called)

        # Reset the circuit
        self.circuit_breaker.reset(user_id="test_user", reason="Test reset")

        # Check that close callback was called
        self.assertTrue(self.on_close_called)

        # Test exception in callback
        self.on_open_called = False
        self.circuit_breaker.on_open = Mock(side_effect=Exception("Test exception"))

        # Trip should not propagate the exception
        self.circuit_breaker.reset()  # Reset first to closed state
        self.circuit_breaker.trip(
            trigger_type=TriggerType.LOSS_THRESHOLD,
            trigger_value=-1500.0,
            reason="Test trip with exception"
        )

        # The exception should be caught and logged
        self.circuit_breaker.on_open.assert_called_once()


class TestCircuitBreakerRegistry(unittest.TestCase):
    """Test cases for the CircuitBreakerRegistry class."""

    def setUp(self):
        """Set up test fixtures."""
        self.event_bus = Mock()
        self.health_monitor = Mock()

        # Create registry
        self.registry = CircuitBreakerRegistry(
            health_monitor=self.health_monitor,
            event_bus=self.event_bus
        )

        # Basic config for tests
        self.config1 = CircuitBreakerConfig(
            name="test_circuit1",
            circuit_type=CircuitType.SYSTEM,
            scope="test_scope",
            trigger_conditions={
                TriggerType.LOSS_THRESHOLD: {"threshold": -1000.0}
            }
        )

        self.config2 = CircuitBreakerConfig(
            name="test_circuit2",
            circuit_type=CircuitType.STRATEGY,
            scope="test_strategy",
            trigger_conditions={
                TriggerType.ERROR_RATE: {"error_count": 0, "error_window": 5, "threshold": 0.6}
            }
        )

    def test_register_circuit_breaker(self):
        """Test registering a circuit breaker."""
        # Register a circuit breaker
        cb = self.registry.register_circuit_breaker(self.config1)

        # Check that it was registered
        self.assertIn(cb.circuit_id, self.registry._circuit_breakers)

        # Register with same ID should return existing instance
        cb2 = self.registry.register_circuit_breaker(self.config1)
        self.assertIs(cb, cb2)

    def test_get_circuit_breaker(self):
        """Test getting a circuit breaker by ID."""
        # Register a circuit breaker
        cb = self.registry.register_circuit_breaker(self.config1)

        # Get by ID
        retrieved_cb = self.registry.get_circuit_breaker(cb.circuit_id)
        self.assertIs(cb, retrieved_cb)

        # Get non-existent ID
        self.assertIsNone(self.registry.get_circuit_breaker("non_existent"))

    def test_get_circuit_breakers(self):
        """Test getting circuit breakers by criteria."""
        # Register two circuit breakers
        cb1 = self.registry.register_circuit_breaker(self.config1)
        cb2 = self.registry.register_circuit_breaker(self.config2)

        # Get all
        all_cbs = self.registry.get_circuit_breakers()
        self.assertEqual(len(all_cbs), 2)
        self.assertIn(cb1, all_cbs)
        self.assertIn(cb2, all_cbs)

        # Get by type
        system_cbs = self.registry.get_circuit_breakers(circuit_type=CircuitType.SYSTEM)
        self.assertEqual(len(system_cbs), 1)
        self.assertIn(cb1, system_cbs)

        # Get by scope
        strategy_cbs = self.registry.get_circuit_breakers(scope="test_strategy")
        self.assertEqual(len(strategy_cbs), 1)
        self.assertIn(cb2, strategy_cbs)

    def test_get_all_circuit_breaker_states(self):
        """Test getting all circuit breaker states."""
        # Register two circuit breakers
        cb1 = self.registry.register_circuit_breaker(self.config1)
        cb2 = self.registry.register_circuit_breaker(self.config2)

        # Trip one circuit
        cb1.trip(TriggerType.LOSS_THRESHOLD, -1500.0, "Test trip")

        # Get all states
        states = self.registry.get_all_circuit_breaker_states()

        # Check states
        self.assertEqual(states[cb1.circuit_id], CircuitState.OPEN)
        self.assertEqual(states[cb2.circuit_id], CircuitState.CLOSED)

    def test_trip_all(self):
        """Test tripping all circuits of a type."""
        # Register three circuit breakers
        cb1 = self.registry.register_circuit_breaker(self.config1)  # SYSTEM

        config2 = CircuitBreakerConfig(
            name="test_circuit2",
            circuit_type=CircuitType.SYSTEM,  # Another SYSTEM
            scope="another_scope",
            trigger_conditions={}
        )
        cb2 = self.registry.register_circuit_breaker(config2)

        cb3 = self.registry.register_circuit_breaker(self.config2)  # STRATEGY

        # Trip all SYSTEM circuits
        tripped = self.registry.trip_all(
            circuit_type=CircuitType.SYSTEM,
            trigger_type=TriggerType.MANUAL,
            trigger_value="test",
            reason="Test trip all"
        )

        # Check results
        self.assertEqual(len(tripped), 2)
        self.assertIn(cb1.circuit_id, tripped)
        self.assertIn(cb2.circuit_id, tripped)
        self.assertNotIn(cb3.circuit_id, tripped)

        # Check states
        self.assertTrue(cb1.is_open())
        self.assertTrue(cb2.is_open())
        self.assertTrue(cb3.is_closed())

    def test_reset_all(self):
        """Test resetting all circuits."""
        # Register two circuit breakers and trip them
        cb1 = self.registry.register_circuit_breaker(self.config1)
        cb2 = self.registry.register_circuit_breaker(self.config2)

        cb1.trip(TriggerType.MANUAL, "test", "Trip for test")
        cb2.trip(TriggerType.MANUAL, "test", "Trip for test")

        # Reset all
        reset = self.registry.reset_all(user_id="test_user", reason="Test reset all")

        # Check results
        self.assertEqual(len(reset), 2)
        self.assertIn(cb1.circuit_id, reset)
        self.assertIn(cb2.circuit_id, reset)

        # Check states
        self.assertTrue(cb1.is_closed())
        self.assertTrue(cb2.is_closed())

    def test_handle_manual_trip(self):
        """Test handling manual trip events."""
        # Register a circuit breaker
        cb = self.registry.register_circuit_breaker(self.config1)

        # Create a manual trip event with circuit ID
        event = Event(
            topic="system.circuit_breaker.manual_trip",
            data={
                "circuit_id": cb.circuit_id,
                "trigger_type": TriggerType.MANUAL.value,
                "trigger_value": "test_event",
                "reason": "Test event trip",
                "user_id": "test_user"
            }
        )

        # Handle the event
        self.registry._handle_manual_trip(event)

        # Check that the circuit was tripped
        self.assertTrue(cb.is_open())

        # Create a manual trip event with circuit type
        cb.reset()  # Reset first

        event = Event(
            topic="system.circuit_breaker.manual_trip",
            data={
                "circuit_type": CircuitType.SYSTEM.value,
                "trigger_type": TriggerType.MANUAL.value,
                "trigger_value": "test_event",
                "reason": "Test event trip by type",
                "user_id": "test_user"
            }
        )

        # Handle the event
        self.registry._handle_manual_trip(event)

        # Check that the circuit was tripped
        self.assertTrue(cb.is_open())

    def test_handle_manual_reset(self):
        """Test handling manual reset events."""
        # Register a circuit breaker and trip it
        cb = self.registry.register_circuit_breaker(self.config1)
        cb.trip(TriggerType.MANUAL, "test", "Trip for test")

        # Create a manual reset event with circuit ID
        event = Event(
            topic="system.circuit_breaker.manual_reset",
            data={
                "circuit_id": cb.circuit_id,
                "reason": "Test event reset",
                "user_id": "test_user"
            }
        )

        # Handle the event
        self.registry._handle_manual_reset(event)

        # Check that the circuit was reset
        self.assertTrue(cb.is_closed())

        # Create a manual reset event with circuit type
        cb.trip(TriggerType.MANUAL, "test", "Trip for test")  # Trip again

        event = Event(
            topic="system.circuit_breaker.manual_reset",
            data={
                "circuit_type": CircuitType.SYSTEM.value,
                "reason": "Test event reset by type",
                "user_id": "test_user"
            }
        )

        # Handle the event
        self.registry._handle_manual_reset(event)

        # Check that the circuit was reset
        self.assertTrue(cb.is_closed())


class TestGlobalFunctions(unittest.TestCase):
    """Test cases for global functions."""

    @patch('core.circuit_breaker._circuit_breaker_registry', None)
    def test_get_circuit_breaker_registry(self):
        """Test getting the global circuit breaker registry."""
        # First call should create a new registry
        registry1 = get_circuit_breaker_registry()
        self.assertIsInstance(registry1, CircuitBreakerRegistry)

        # Second call should return the same registry
        registry2 = get_circuit_breaker_registry()
        self.assertIs(registry1, registry2)

        # Call with parameters should use existing registry
        health_monitor = Mock()
        event_bus = Mock()
        registry3 = get_circuit_breaker_registry(health_monitor, event_bus)
        self.assertIs(registry1, registry3)

    def test_create_circuit_breaker(self):
        """Test creating a circuit breaker with the global function."""
        # Mock registry
        mock_registry = Mock()

        # Patch get_circuit_breaker_registry to return mock
        with patch('core.circuit_breaker.get_circuit_breaker_registry', return_value=mock_registry):
            # Create circuit breaker
            cb = create_circuit_breaker(
                name="test_circuit",
                circuit_type=CircuitType.SYSTEM,
                scope="test_scope",
                trigger_conditions={
                    TriggerType.LOSS_THRESHOLD: {"threshold": -1000.0}
                }
            )

            # Check that registry.register_circuit_breaker was called
            mock_registry.register_circuit_breaker.assert_called_once()

            # Should return the circuit breaker
            self.assertEqual(cb, mock_registry.register_circuit_breaker.return_value)


if __name__ == '__main__':
    unittest.main()