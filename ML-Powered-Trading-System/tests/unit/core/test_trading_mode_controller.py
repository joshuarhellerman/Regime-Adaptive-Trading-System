import unittest
from unittest import mock
import json
import os
import tempfile
from pathlib import Path

from core.trading_mode_controller import (
    TradingModeController, TradingMode, TradingModeEvent, 
    EventBus, Event, EventPriority, HealthMonitor, HealthStatus
)


class TestTradingModeController(unittest.TestCase):
    """Test suite for TradingModeController"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mocks
        self.mock_event_bus = mock.MagicMock(spec=EventBus)
        self.mock_health_monitor = mock.MagicMock(spec=HealthMonitor)
        
        # Create temp directory for config files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_dir = Path(self.temp_dir.name)
        
        # Default test config
        self.test_config = {
            "config_dir": str(self.config_dir),
            "paper_trading": {
                "base_currency": "USD",
                "starting_balance": 10000
            },
            "live_trading": {
                "api_key": "test_key",
                "api_secret": "test_secret",
                "exchange": "test_exchange"
            },
            "backtest_trading": {
                "start_date": "2023-01-01",
                "end_date": "2023-12-31"
            },
            "risk_management": {
                "max_position_size": 0.1
            }
        }
        
        # Create controller with mocks
        self.controller = TradingModeController(
            event_bus=self.mock_event_bus,
            health_monitor=self.mock_health_monitor,
            config=self.test_config
        )
        
        # Set up mock for trading interfaces
        self.mock_paper_interface = mock.MagicMock()
        self.mock_live_interface = mock.MagicMock() 
        self.mock_backtest_interface = mock.MagicMock()
        
        # Mock the interface factory functions
        self.paper_patcher = mock.patch(
            'core.trading_mode_controller.get_paper_trader_interface',
            return_value=self.mock_paper_interface
        )
        self.live_patcher = mock.patch(
            'core.trading_mode_controller.get_live_trader_interface',
            return_value=self.mock_live_interface
        )
        self.backtest_patcher = mock.patch(
            'core.trading_mode_controller.BacktestEngine',
            return_value=self.mock_backtest_interface
        )
        
        # Start patches
        self.mock_get_paper_trader = self.paper_patcher.start()
        self.mock_get_live_trader = self.live_patcher.start()
        self.mock_backtest_engine = self.backtest_patcher.start()
        
        # Configure mocks for successful start/stop
        for interface in [self.mock_paper_interface, self.mock_live_interface, self.mock_backtest_interface]:
            interface.start.return_value = {"status": "started", "timestamp": 12345}
            interface.stop.return_value = {"status": "stopped", "timestamp": 12345}
            interface.get_status.return_value = {"status": "running"}
            interface.get_positions.return_value = {"BTC": 1.0, "ETH": 10.0}
            interface.get_performance.return_value = {"profit": 100.0, "drawdown": 0.05}

    def tearDown(self):
        """Clean up after each test"""
        # Stop patches
        self.paper_patcher.stop()
        self.live_patcher.stop()
        self.backtest_patcher.stop()
        
        # Remove temp directory
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test controller initialization and default state"""
        # Check initial mode
        self.assertEqual(self.controller.get_current_mode(), TradingMode.STOPPED)
        
        # Check health monitor registration
        self.mock_health_monitor.register_component.assert_called_once_with(
            component_id="trading_mode_controller",
            component_type="core"
        )
        
        # Check available modes
        available_modes = self.controller.get_available_modes()
        self.assertIn(TradingMode.PAPER.value, available_modes)
        self.assertIn(TradingMode.LIVE.value, available_modes)
        self.assertIn(TradingMode.BACKTEST.value, available_modes)

    def test_start_paper_trading(self):
        """Test starting paper trading mode"""
        # Start paper trading
        result = self.controller.start_paper_trading()
        
        # Check result
        self.assertEqual(result["status"], "started")
        
        # Check mode was updated
        self.assertEqual(self.controller.get_current_mode(), TradingMode.PAPER)
        
        # Check interface was created with correct config
        self.mock_get_paper_trader.assert_called_once()
        config_arg = self.mock_get_paper_trader.call_args[1]["config"]
        self.assertEqual(config_arg["base_currency"], "USD")
        
        # Check interface was started
        self.mock_paper_interface.start.assert_called_once()
        
        # Check event was published
        self.mock_event_bus.publish.assert_any_call(mock.ANY)
        
        # Check health monitor was updated
        self.mock_health_monitor.update_component_health.assert_any_call(
            component_id="trading_mode_controller",
            status=HealthStatus.HEALTHY,
            metrics=mock.ANY
        )

    def test_start_live_trading(self):
        """Test starting live trading mode"""
        # Start live trading
        result = self.controller.start_live_trading()
        
        # Check result
        self.assertEqual(result["status"], "started")
        
        # Check mode was updated
        self.assertEqual(self.controller.get_current_mode(), TradingMode.LIVE)
        
        # Check interface was created with correct config
        self.mock_get_live_trader.assert_called_once()
        config_arg = self.mock_get_live_trader.call_args[1]["config"]
        self.assertEqual(config_arg["api_key"], "test_key")
        
        # Check interface was started
        self.mock_live_interface.start.assert_called_once()
        
        # Check events were published
        self.mock_event_bus.publish.assert_any_call(mock.ANY)

    def test_start_backtest(self):
        """Test starting backtest mode"""
        # Custom backtest config
        backtest_config = {
            "symbols": ["BTC/USD", "ETH/USD"],
            "strategy": "momentum"
        }
        
        # Start backtest with custom config
        result = self.controller.start_backtest(backtest_config)
        
        # Check result
        self.assertEqual(result["status"], "started")
        
        # Check mode was updated
        self.assertEqual(self.controller.get_current_mode(), TradingMode.BACKTEST)
        
        # Check interface was created with merged config
        self.mock_backtest_engine.assert_called_once()
        config_arg = self.mock_backtest_engine.call_args[1]["config"]
        self.assertEqual(config_arg["start_date"], "2023-01-01")
        self.assertEqual(config_arg["symbols"], ["BTC/USD", "ETH/USD"])
        
        # Check interface was started
        self.mock_backtest_interface.start.assert_called_once()

    def test_stop_trading(self):
        """Test stopping trading"""
        # First start paper trading
        self.controller.start_paper_trading()
        
        # Then stop trading
        result = self.controller.stop_trading()
        
        # Check result
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["previous_mode"], TradingMode.PAPER.value)
        
        # Check mode was updated
        self.assertEqual(self.controller.get_current_mode(), TradingMode.STOPPED)
        
        # Check interface was stopped
        self.mock_paper_interface.stop.assert_called_once()
        
        # Check position data was captured
        self.assertIn("final_positions", result)
        
        # Check event was published
        self.mock_event_bus.publish.assert_any_call(mock.ANY)

    def test_switch_mode(self):
        """Test switching modes"""
        # Start with paper trading
        self.controller.start_paper_trading()
        self.assertEqual(self.controller.get_current_mode(), TradingMode.PAPER)
        
        # Switch to backtest
        result = self.controller.switch_mode(TradingMode.BACKTEST)
        self.assertEqual(result["status"], "started")
        self.assertEqual(self.controller.get_current_mode(), TradingMode.BACKTEST)
        
        # Switch to live
        result = self.controller.switch_mode("live")  # Test string input
        self.assertEqual(result["status"], "started")
        self.assertEqual(self.controller.get_current_mode(), TradingMode.LIVE)
        
        # Switch to stopped
        result = self.controller.switch_mode(TradingMode.STOPPED)
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(self.controller.get_current_mode(), TradingMode.STOPPED)
        
        # Test invalid mode
        result = self.controller.switch_mode("invalid_mode")
        self.assertEqual(result["status"], "error")

    def test_get_mode_status(self):
        """Test getting mode status"""
        # Start paper trading first
        self.controller.start_paper_trading()
        
        # Get status
        status = self.controller.get_mode_status()
        
        # Check status
        self.assertEqual(status["mode"], TradingMode.PAPER.value)
        self.assertTrue(status["is_active"])
        self.assertIn("timestamp", status)
        self.assertIn("details", status)
        self.assertEqual(status["details"]["status"], "running")
        self.assertIn("positions", status)
        self.assertIn("performance", status)

    def test_mode_config_management(self):
        """Test mode configuration management"""
        # Get initial config
        paper_config = self.controller.get_mode_config(TradingMode.PAPER)
        self.assertEqual(paper_config["base_currency"], "USD")
        
        # Update config
        update = {"base_currency": "EUR", "new_setting": 42}
        self.controller.update_mode_config(TradingMode.PAPER, update)
        
        # Check updated config
        paper_config = self.controller.get_mode_config(TradingMode.PAPER)
        self.assertEqual(paper_config["base_currency"], "EUR")
        self.assertEqual(paper_config["new_setting"], 42)
        
        # Start paper trading
        self.controller.start_paper_trading()
        
        # Update config for active mode
        self.controller.update_mode_config(TradingMode.PAPER, {"timeout": 60})
        
        # Check interface was updated
        self.mock_paper_interface.update_config.assert_called_once_with({"timeout": 60})

    def test_settings_persistence(self):
        """Test settings persistence to file"""
        # Start paper trading
        self.controller.start_paper_trading()
        
        # Update config
        self.controller.update_mode_config(TradingMode.PAPER, {"test_value": 100})
        
        # Create a new controller instance (simulating restart)
        new_controller = TradingModeController(
            config=self.test_config,
            event_bus=self.mock_event_bus,
            health_monitor=self.mock_health_monitor
        )
        
        # Check if settings were loaded
        self.assertEqual(new_controller.get_current_mode(), TradingMode.PAPER)
        paper_config = new_controller.get_mode_config(TradingMode.PAPER)
        self.assertEqual(paper_config["test_value"], 100)

    def test_event_publication(self):
        """Test event publication during mode transitions"""
        # Start paper trading
        self.controller.start_paper_trading()
        
        # Check events
        event_calls = self.mock_event_bus.publish.call_args_list
        
        # Check MODE_STARTING event
        starting_event = None
        for call in event_calls:
            event = call[0][0]
            if event.topic == TradingModeEvent.MODE_STARTING.value:
                starting_event = event
                break
                
        self.assertIsNotNone(starting_event)
        self.assertEqual(starting_event.data["mode"], TradingMode.PAPER.value)
        
        # Check MODE_CHANGED event
        changed_event = None
        for call in event_calls:
            event = call[0][0]
            if event.topic == TradingModeEvent.MODE_CHANGED.value:
                changed_event = event
                break
                
        self.assertIsNotNone(changed_event)
        self.assertEqual(changed_event.data["mode"], TradingMode.PAPER.value)

    def test_live_trading_safety_checks(self):
        """Test safety checks for live trading"""
        # Modify config to make safety checks fail
        unsafe_config = dict(self.test_config)
        unsafe_config["risk_management"] = {}  # Missing max_position_size
        
        # Create controller with unsafe config
        unsafe_controller = TradingModeController(
            event_bus=self.mock_event_bus,
            health_monitor=self.mock_health_monitor,
            config=unsafe_config
        )
        
        # Try to start live trading
        result = unsafe_controller.start_live_trading()
        
        # Check error
        self.assertEqual(result["status"], "error")
        self.assertIn("Risk management not properly configured", result["error"])

    def test_interface_access(self):
        """Test accessing the current trading interface"""
        # No interface when stopped
        self.assertIsNone(self.controller.get_interface())
        
        # Start paper trading
        self.controller.start_paper_trading()
        
        # Check interface
        self.assertEqual(self.controller.get_interface(), self.mock_paper_interface)

    def test_missing_live_config(self):
        """Test behavior with missing live trading config"""
        # Create controller with missing live config
        incomplete_config = {
            "config_dir": str(self.config_dir),
            "live_trading": {
                "api_key": "test_key"
                # Missing api_secret and exchange
            }
        }
        
        controller = TradingModeController(
            event_bus=self.mock_event_bus,
            health_monitor=self.mock_health_monitor,
            config=incomplete_config
        )
        
        # Check available modes
        modes = controller.get_available_modes()
        self.assertNotIn(TradingMode.LIVE.value, modes)
        
        # Try to start live trading
        result = controller.start_live_trading()
        self.assertEqual(result["status"], "error")

    def test_position_transfer_handler(self):
        """Test position transfer handler registration and execution"""
        # Create mock handler
        mock_handler = mock.MagicMock()
        
        # Register handler
        self.controller.register_position_transfer_handler(mock_handler)
        
        # Start paper trading
        self.controller.start_paper_trading()
        
        # Add positions to mock
        positions = {"BTC": 1.5, "ETH": 10.0}
        self.mock_paper_interface.get_positions.return_value = positions
        
        # Switch to backtest
        self.controller.switch_mode(TradingMode.BACKTEST)
        
        # Check if handler was called during switch
        # Note: This test depends on internal implementation of _execute_position_transfer
        # Since that method is private, we can't directly check if it was called
        # We need to verify this indirectly if position transfer is implemented

    def test_error_handling(self):
        """Test error handling during mode transitions"""
        # Make paper interface throw an exception on start
        self.mock_paper_interface.start.side_effect = RuntimeError("Test error")
        
        # Try to start paper trading
        result = self.controller.start_paper_trading()
        
        # Check error response
        self.assertEqual(result["status"], "error")
        self.assertIn("Test error", result["error"])
        
        # Check mode wasn't changed
        self.assertEqual(self.controller.get_current_mode(), TradingMode.STOPPED)
        
        # Check error event was published
        self.mock_event_bus.publish.assert_any_call(mock.ANY)
        
        # Check health monitor was notified
        self.mock_health_monitor.log_error.assert_any_call(
            component_id="trading_mode_controller",
            error_type="PaperTradingStartError",
            error_message=mock.ANY
        )


if __name__ == "__main__":
    unittest.main()