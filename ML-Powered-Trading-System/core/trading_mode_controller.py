"""
Trading Mode Controller - Manages switching between paper, backtest, and live trading modes
Coordinates system components for consistent operation across different trading environments
"""
import logging
import json
import time
import os
from typing import Dict, Optional, Union, List, Any, Callable
from enum import Enum
from pathlib import Path
import threading

from core.event_bus import EventBus, Event, EventPriority
from core.health_monitor import HealthMonitor, HealthStatus
from execution.paper_trader_interface import get_paper_trader_interface
from execution.live_trader_interface import get_live_trader_interface
from execution.backtest_engine import BacktestEngine
from models.portfolio.portfolio_manager import OnlineLearningPortfolio
from models.portfolio.allocation_manager import AllocationManager


class TradingMode(Enum):
    """Trading mode enum with expanded capabilities."""
    PAPER = "paper"         # Paper trading with real-time data
    LIVE = "live"           # Live trading with real money
    BACKTEST = "backtest"   # Historical backtesting
    SIMULATION = "simulation"  # Advanced simulation with market impact models
    STOPPED = "stopped"     # No trading active


class TradingModeEvent(Enum):
    """Events related to trading mode changes."""
    MODE_CHANGED = "trading_mode.changed"
    MODE_STARTING = "trading_mode.starting"
    MODE_STOPPING = "trading_mode.stopping"
    MODE_ERROR = "trading_mode.error"


class TradingModeController:
    """
    Enhanced controller for managing trading modes (paper/live/backtest).

    This class handles:
    1. Switching between different trading modes
    2. Ensuring only one mode is active at a time
    3. Proper initialization and shutdown of system components
    4. Mode-specific configuration management
    5. Integration with health monitoring system
    6. Event notification of mode changes
    """

    def __init__(self, event_bus: Optional[EventBus] = None,
                 health_monitor: Optional[HealthMonitor] = None,
                 config: Optional[Dict] = None):
        """
        Initialize the trading mode controller.

        Args:
            event_bus: Event bus for inter-component communication
            health_monitor: System health monitoring
            config: Configuration parameters
        """
        self.logger = logging.getLogger(__name__)
        self.event_bus = event_bus
        self.health_monitor = health_monitor
        self.config = config or {}

        # Current trading mode
        self._current_mode = TradingMode.STOPPED
        self._transition_lock = threading.RLock()
        self._mode_interfaces = {
            TradingMode.PAPER: None,
            TradingMode.LIVE: None,
            TradingMode.BACKTEST: None,
            TradingMode.SIMULATION: None
        }

        # Mode configurations
        self._config_dir = Path(self.config.get("config_dir", "./config"))
        self._config_dir.mkdir(exist_ok=True)

        # Mode-specific configuration store
        self._mode_configs = {}

        # Position and state management
        self._position_transfer_handlers = []

        # Load mode settings
        self._settings_path = self._config_dir / "trading_mode_settings.json"
        self._load_settings()

        # Register with health monitor if available
        if self.health_monitor:
            self.health_monitor.register_component(
                component_id="trading_mode_controller",
                component_type="core"
            )

        # Subscribe to relevant events
        if self.event_bus:
            self._subscribe_to_events()

    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant system events."""
        # Add event subscriptions as needed
        pass

    def _load_settings(self) -> None:
        """Load trading mode settings from file."""
        if self._settings_path.exists():
            try:
                with open(self._settings_path, 'r') as f:
                    settings = json.load(f)

                # Set mode from settings (but don't activate it yet)
                mode_str = settings.get("current_mode", "STOPPED")
                try:
                    self._current_mode = TradingMode[mode_str]
                except (KeyError, ValueError):
                    self._current_mode = TradingMode.STOPPED

                # Load mode-specific configs
                self._mode_configs = settings.get("mode_configs", {})

                self.logger.info(f"Loaded trading mode settings: {self._current_mode}")

                # Log health metric
                if self.health_monitor:
                    self.health_monitor.update_component_health(
                        component_id="trading_mode_controller",
                        metrics={"loaded_settings": 1, "current_mode": float(self._current_mode.value != "stopped")}
                    )

            except Exception as e:
                self.logger.error(f"Failed to load trading mode settings: {str(e)}")
                self._current_mode = TradingMode.STOPPED

                # Log health metric for error
                if self.health_monitor:
                    self.health_monitor.log_error(
                        component_id="trading_mode_controller",
                        error_type="SettingsLoadError",
                        error_message=str(e)
                    )
        else:
            # Create default settings
            self._current_mode = TradingMode.STOPPED
            self._save_settings()

    def _save_settings(self) -> None:
        """Save trading mode settings to file."""
        try:
            settings = {
                "current_mode": self._current_mode.name,
                "last_updated": time.time(),
                "mode_configs": self._mode_configs
            }

            with open(self._settings_path, 'w') as f:
                json.dump(settings, f, indent=2)

            self.logger.info(f"Saved trading mode settings: {self._current_mode}")

            # Log health metric
            if self.health_monitor:
                self.health_monitor.update_component_health(
                    component_id="trading_mode_controller",
                    metrics={"saved_settings": 1}
                )

        except Exception as e:
            self.logger.error(f"Failed to save trading mode settings: {str(e)}")

            # Log health metric for error
            if self.health_monitor:
                self.health_monitor.log_error(
                    component_id="trading_mode_controller",
                    error_type="SettingsSaveError",
                    error_message=str(e)
                )

    def get_current_mode(self) -> TradingMode:
        """Get the current trading mode."""
        return self._current_mode

    def get_mode_status(self) -> Dict:
        """
        Get comprehensive status of the current trading mode.

        Returns:
            Dict containing mode status details
        """
        with self._transition_lock:
            status = {
                "mode": self._current_mode.value,
                "is_active": self._current_mode != TradingMode.STOPPED,
                "timestamp": time.time(),
                "details": {}
            }

            # Get mode-specific status
            if self._current_mode != TradingMode.STOPPED:
                interface = self._mode_interfaces.get(self._current_mode)
                if interface:
                    interface_status = interface.get_status()
                    status["details"] = interface_status

                    # Add position information if available
                    if hasattr(interface, 'get_positions'):
                        try:
                            positions = interface.get_positions()
                            status["positions"] = positions
                        except Exception as e:
                            self.logger.warning(f"Failed to get positions: {str(e)}")

                    # Add performance metrics if available
                    if hasattr(interface, 'get_performance'):
                        try:
                            performance = interface.get_performance()
                            status["performance"] = performance
                        except Exception as e:
                            self.logger.warning(f"Failed to get performance metrics: {str(e)}")

            return status

    def get_available_modes(self) -> List[str]:
        """
        Get list of available trading modes.

        Returns:
            List of mode names
        """
        # Filter modes based on system configuration/capabilities
        all_modes = [mode.value for mode in TradingMode if mode != TradingMode.STOPPED]

        # Live trading might require additional verification
        if not self._is_live_trading_configured():
            all_modes.remove(TradingMode.LIVE.value)

        return all_modes

    def _is_live_trading_configured(self) -> bool:
        """Check if live trading is properly configured."""
        # Check for required configurations (API keys, etc.)
        live_config = self.config.get("live_trading", {})
        required_fields = ["api_key", "api_secret", "exchange"]

        return all(field in live_config for field in required_fields)

    def get_mode_config(self, mode: Union[str, TradingMode]) -> Dict:
        """
        Get configuration for specified trading mode.

        Args:
            mode: Trading mode to get config for

        Returns:
            Mode-specific configuration
        """
        if isinstance(mode, str):
            try:
                mode = TradingMode(mode.lower())
            except ValueError:
                self.logger.error(f"Invalid trading mode: {mode}")
                return {}

        # Get mode-specific config with defaults
        mode_str = mode.name.lower()
        base_config = self.config.get(f"{mode_str}_trading", {})
        saved_config = self._mode_configs.get(mode_str, {})

        # Merge configs with saved config taking precedence
        return {**base_config, **saved_config}

    def update_mode_config(self, mode: Union[str, TradingMode], config_updates: Dict) -> bool:
        """
        Update configuration for specified trading mode.

        Args:
            mode: Trading mode to update config for
            config_updates: Configuration updates to apply

        Returns:
            True if update successful, False otherwise
        """
        if isinstance(mode, str):
            try:
                mode = TradingMode(mode.lower())
            except ValueError:
                self.logger.error(f"Invalid trading mode: {mode}")
                return False

        mode_str = mode.name.lower()

        # Initialize if not exists
        if mode_str not in self._mode_configs:
            self._mode_configs[mode_str] = {}

        # Update config
        self._mode_configs[mode_str].update(config_updates)

        # Save settings
        self._save_settings()

        # Update active interface if this is the current mode
        if self._current_mode == mode and self._mode_interfaces[mode]:
            try:
                self._mode_interfaces[mode].update_config(config_updates)
                return True
            except Exception as e:
                self.logger.error(f"Failed to update active {mode} configuration: {str(e)}")
                return False

        return True

    def start_paper_trading(self) -> Dict:
        """
        Start paper trading mode.

        Returns:
            Status information
        """
        with self._transition_lock:
            # Publish event about mode transition starting
            self._publish_mode_event(TradingModeEvent.MODE_STARTING, TradingMode.PAPER)

            # Track start time for performance monitoring
            start_time = time.time()

            if self._current_mode != TradingMode.STOPPED:
                # Stop other modes first
                self.stop_trading()

            try:
                # Get configuration for paper trading
                paper_config = self.get_mode_config(TradingMode.PAPER)

                # Initialize paper trading interface if needed
                if not self._mode_interfaces[TradingMode.PAPER]:
                    self._mode_interfaces[TradingMode.PAPER] = get_paper_trader_interface(
                        config=paper_config,
                        event_bus=self.event_bus
                    )

                # Start paper trading
                interface = self._mode_interfaces[TradingMode.PAPER]
                result = interface.start()

                if result.get("status") == "started":
                    self._current_mode = TradingMode.PAPER
                    self._save_settings()

                    # Publish event about successful mode change
                    self._publish_mode_event(TradingModeEvent.MODE_CHANGED, TradingMode.PAPER)

                    # Track health metric
                    if self.health_monitor:
                        self.health_monitor.update_component_health(
                            component_id="trading_mode_controller",
                            status=HealthStatus.HEALTHY,
                            metrics={
                                "mode_transition_ms": (time.time() - start_time) * 1000,
                                "current_mode": 1  # 1 = active
                            }
                        )

                return result
            except Exception as e:
                self.logger.error(f"Failed to start paper trading: {str(e)}")

                # Publish error event
                self._publish_mode_event(
                    TradingModeEvent.MODE_ERROR,
                    TradingMode.PAPER,
                    {"error": str(e)}
                )

                # Track health metric for error
                if self.health_monitor:
                    self.health_monitor.log_error(
                        component_id="trading_mode_controller",
                        error_type="PaperTradingStartError",
                        error_message=str(e)
                    )

                return {
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time()
                }

    def start_live_trading(self) -> Dict:
        """
        Start live trading mode with additional safety checks.

        Returns:
            Status information dictionary
        """
        with self._transition_lock:
            # First check if live trading is properly configured
            if not self._is_live_trading_configured():
                error_msg = "Live trading is not properly configured. Check API credentials."
                self.logger.error(error_msg)
                return {
                    "status": "error",
                    "error": error_msg,
                    "timestamp": time.time()
                }

            # Publish event about mode transition starting
            self._publish_mode_event(TradingModeEvent.MODE_STARTING, TradingMode.LIVE)

            # Track start time for performance monitoring
            start_time = time.time()

            if self._current_mode != TradingMode.STOPPED:
                # Stop other modes first
                self.stop_trading()

            try:
                # Get configuration for live trading
                live_config = self.get_mode_config(TradingMode.LIVE)

                # Initialize live trading interface if needed
                if not self._mode_interfaces[TradingMode.LIVE]:
                    self._mode_interfaces[TradingMode.LIVE] = get_live_trader_interface(
                        config=live_config,
                        event_bus=self.event_bus
                    )

                # Perform additional safety checks before starting
                self._perform_live_trading_safety_checks()

                # Start live trading
                interface = self._mode_interfaces[TradingMode.LIVE]
                result = interface.start()

                if result.get("status") == "started":
                    self._current_mode = TradingMode.LIVE
                    self._save_settings()

                    # Publish event about successful mode change
                    self._publish_mode_event(TradingModeEvent.MODE_CHANGED, TradingMode.LIVE)

                    # Track health metric
                    if self.health_monitor:
                        self.health_monitor.update_component_health(
                            component_id="trading_mode_controller",
                            status=HealthStatus.HEALTHY,
                            metrics={
                                "mode_transition_ms": (time.time() - start_time) * 1000,
                                "current_mode": 2  # 2 = live (higher priority)
                            }
                        )

                return result
            except Exception as e:
                self.logger.error(f"Failed to start live trading: {str(e)}")

                # Publish error event
                self._publish_mode_event(
                    TradingModeEvent.MODE_ERROR,
                    TradingMode.LIVE,
                    {"error": str(e)}
                )

                # Track health metric for error
                if self.health_monitor:
                    self.health_monitor.log_error(
                        component_id="trading_mode_controller",
                        error_type="LiveTradingStartError",
                        error_message=str(e)
                    )

                return {
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time()
                }

    def _perform_live_trading_safety_checks(self) -> None:
        """
        Perform safety checks before allowing live trading.
        Raises exceptions if checks fail.
        """
        # 1. Verify system stability
        if self.health_monitor:
            system_health = self.health_monitor.get_system_health()
            if system_health.get("status") == "critical":
                raise RuntimeError("Cannot start live trading while system health is critical")

        # 2. Check risk limits configuration
        risk_config = self.config.get("risk_management", {})
        if not risk_config.get("max_position_size"):
            raise ValueError("Risk management not properly configured: missing max_position_size")

        # 3. Verify exchange connectivity
        # This would typically check connection to exchange API
        live_config = self.get_mode_config(TradingMode.LIVE)
        exchange = live_config.get("exchange")
        if not exchange:
            raise ValueError("Exchange not specified in live trading configuration")

        # Additional checks can be added here

    def start_backtest(self, backtest_config: Dict = None) -> Dict:
        """
        Start backtest mode.

        Args:
            backtest_config: Configuration for the backtest run

        Returns:
            Status information
        """
        with self._transition_lock:
            # Merge provided config with saved config
            config = self.get_mode_config(TradingMode.BACKTEST)
            if backtest_config:
                config.update(backtest_config)
                # Save updated config
                self.update_mode_config(TradingMode.BACKTEST, backtest_config)

            # Publish event about mode transition starting
            self._publish_mode_event(TradingModeEvent.MODE_STARTING, TradingMode.BACKTEST)

            # Track start time for performance monitoring
            start_time = time.time()

            if self._current_mode != TradingMode.STOPPED:
                # Stop other modes first
                self.stop_trading()

            try:
                # Initialize backtest engine if needed
                if not self._mode_interfaces[TradingMode.BACKTEST]:
                    self._mode_interfaces[TradingMode.BACKTEST] = BacktestEngine(
                        config=config,
                        event_bus=self.event_bus
                    )

                # Start backtest
                interface = self._mode_interfaces[TradingMode.BACKTEST]
                result = interface.start()

                if result.get("status") == "started":
                    self._current_mode = TradingMode.BACKTEST
                    self._save_settings()

                    # Publish event about successful mode change
                    self._publish_mode_event(TradingModeEvent.MODE_CHANGED, TradingMode.BACKTEST)

                    # Track health metric
                    if self.health_monitor:
                        self.health_monitor.update_component_health(
                            component_id="trading_mode_controller",
                            status=HealthStatus.HEALTHY,
                            metrics={
                                "mode_transition_ms": (time.time() - start_time) * 1000,
                                "current_mode": 3  # 3 = backtest
                            }
                        )

                return result
            except Exception as e:
                self.logger.error(f"Failed to start backtest: {str(e)}")

                # Publish error event
                self._publish_mode_event(
                    TradingModeEvent.MODE_ERROR,
                    TradingMode.BACKTEST,
                    {"error": str(e)}
                )

                # Track health metric for error
                if self.health_monitor:
                    self.health_monitor.log_error(
                        component_id="trading_mode_controller",
                        error_type="BacktestStartError",
                        error_message=str(e)
                    )

                return {
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time()
                }

    def stop_trading(self) -> Dict:
        """
        Stop all trading modes with graceful shutdown.

        Returns:
            Status information
        """
        with self._transition_lock:
            old_mode = self._current_mode

            # No need to stop if already stopped
            if old_mode == TradingMode.STOPPED:
                return {
                    "status": "already_stopped",
                    "timestamp": time.time()
                }

            # Publish event about mode stopping
            self._publish_mode_event(TradingModeEvent.MODE_STOPPING, old_mode)

            result = {
                "status": "stopped",
                "previous_mode": old_mode.value,
                "timestamp": time.time()
            }

            try:
                # Stop current mode
                interface = self._mode_interfaces.get(old_mode)
                if interface:
                    # Capture positions before stopping if available
                    if hasattr(interface, 'get_positions'):
                        try:
                            positions = interface.get_positions()
                            result["final_positions"] = positions
                        except Exception:
                            pass

                    # Capture performance metrics if available
                    if hasattr(interface, 'get_performance'):
                        try:
                            performance = interface.get_performance()
                            result["performance"] = performance
                        except Exception:
                            pass

                    # Stop the interface
                    stop_result = interface.stop()
                    result.update({
                        "details": stop_result
                    })

                # Update mode
                self._current_mode = TradingMode.STOPPED
                self._save_settings()

                # Publish event about successful mode change
                self._publish_mode_event(TradingModeEvent.MODE_CHANGED, TradingMode.STOPPED,
                                       {"previous_mode": old_mode.value})

                # Track health metric
                if self.health_monitor:
                    self.health_monitor.update_component_health(
                        component_id="trading_mode_controller",
                        metrics={
                            "current_mode": 0,  # 0 = stopped
                            "stop_trading": 1
                        }
                    )

                return result
            except Exception as e:
                self.logger.error(f"Failed to stop trading: {str(e)}")

                # Publish error event
                self._publish_mode_event(
                    TradingModeEvent.MODE_ERROR,
                    TradingMode.STOPPED,
                    {"error": str(e), "previous_mode": old_mode.value}
                )

                # Track health metric for error
                if self.health_monitor:
                    self.health_monitor.log_error(
                        component_id="trading_mode_controller",
                        error_type="StopTradingError",
                        error_message=str(e)
                    )

                return {
                    "status": "error",
                    "error": str(e),
                    "timestamp": time.time()
                }

    def switch_mode(self, mode: Union[str, TradingMode], mode_config: Dict = None) -> Dict:
        """
        Switch to the specified trading mode with optional configuration.

        Args:
            mode: The trading mode to switch to
            mode_config: Optional configuration for the new mode

        Returns:
            Status information
        """
        if isinstance(mode, str):
            try:
                mode = TradingMode(mode.lower())
            except ValueError:
                return {
                    "status": "error",
                    "error": f"Invalid trading mode: {mode}",
                    "timestamp": time.time()
                }

        # Update configuration if provided
        if mode_config and mode != TradingMode.STOPPED:
            self.update_mode_config(mode, mode_config)

        # Handle each mode type
        if mode == TradingMode.PAPER:
            return self.start_paper_trading()
        elif mode == TradingMode.LIVE:
            return self.start_live_trading()
        elif mode == TradingMode.BACKTEST:
            return self.start_backtest()
        elif mode == TradingMode.SIMULATION:
            # Add simulation support when implemented
            return {
                "status": "error",
                "error": "Simulation mode not yet implemented",
                "timestamp": time.time()
            }
        else:  # STOPPED or unknown
            return self.stop_trading()

    def set_mode(self, mode: str) -> bool:
        """
        Set the trading mode to paper, live, or backtest.
        Compatibility method for the system debugger.

        Args:
            mode: Trading mode to set (paper, live, backtest)

        Returns:
            True if mode was set successfully, False otherwise
        """
        result = self.switch_mode(mode)
        return result.get("status") not in ["error", "stopped"]

    def register_position_transfer_handler(self, handler: Callable) -> None:
        """
        Register a handler for position transfers between modes.

        Args:
            handler: Callback function that will be called during mode transitions
        """
        self._position_transfer_handlers.append(handler)

    def _execute_position_transfer(self, from_mode: TradingMode, to_mode: TradingMode,
                                 positions: Dict) -> None:
        """
        Execute position transfer between trading modes.

        Args:
            from_mode: Source trading mode
            to_mode: Destination trading mode
            positions: Position data to transfer
        """
        for handler in self._position_transfer_handlers:
            try:
                handler(from_mode, to_mode, positions)
            except Exception as e:
                self.logger.error(f"Error in position transfer handler: {str(e)}")

    def _publish_mode_event(self, event_type: TradingModeEvent, mode: TradingMode,
                          extra_data: Dict = None) -> None:
        """
        Publish a trading mode event to the event bus.

        Args:
            event_type: Type of trading mode event
            mode: The trading mode related to the event
            extra_data: Additional data to include
        """
        if not self.event_bus:
            return

        data = {
            "mode": mode.value,
            "timestamp": time.time()
        }

        if extra_data:
            data.update(extra_data)

        event = Event(
            topic=event_type.value,
            data=data,
            priority=EventPriority.MEDIUM
        )

        self.event_bus.publish(event)

    def get_interface(self) -> Any:
        """
        Get the current trading interface for direct interaction.

        Returns:
            The active trading interface or None if not active
        """
        return self._mode_interfaces.get(self._current_mode)