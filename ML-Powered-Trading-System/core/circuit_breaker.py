"""
circuit_breaker.py - Trading Circuit Breaker System

This module provides circuit breaker functionality for trading systems,
with support for multiple trigger conditions, automated recovery,
and integration with the event bus and health monitoring.
"""

import logging
import threading
import time
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
import datetime

from core.event_bus import EventTopics, Event, get_event_bus, create_event, EventPriority
from core.health_monitor import HealthMonitor, HealthStatus, AlertLevel, AlertCategory

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """States for the circuit breaker"""
    CLOSED = "closed"       # Normal operation, circuit allows trading
    OPEN = "open"           # Circuit is tripped, trading is halted
    HALF_OPEN = "half_open" # Recovery mode, limited operation allowed


class CircuitType(Enum):
    """Types of circuit breakers"""
    SYSTEM = "system"        # System-level circuit (affects all trading)
    STRATEGY = "strategy"    # Strategy-level circuit
    MARKET = "market"        # Market-wide circuit
    SYMBOL = "symbol"        # Symbol-specific circuit
    MODEL = "model"          # Model-specific circuit
    EXECUTION = "execution"  # Execution-related circuit


class TriggerType(Enum):
    """Types of circuit breaker triggers"""
    LOSS_THRESHOLD = "loss_threshold"      # P&L loss threshold
    DRAWDOWN = "drawdown"                  # Maximum drawdown
    VOLATILITY = "volatility"              # Market volatility
    ERROR_RATE = "error_rate"              # Error rate threshold
    LATENCY = "latency"                    # Latency threshold
    MANUAL = "manual"                      # Manual triggering
    API_FAILURE = "api_failure"            # API connectivity issues
    DATA_QUALITY = "data_quality"          # Data quality issues
    LIQUIDITY = "liquidity"                # Liquidity issues
    MODEL_DRIFT = "model_drift"            # Model drift threshold
    CUSTOM = "custom"                      # Custom trigger condition


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker"""
    name: str
    circuit_type: CircuitType
    scope: str  # Strategy ID, symbol, etc.
    trigger_conditions: Dict[TriggerType, Dict[str, Any]]
    recovery_time: int = 300  # Time in seconds before trying recovery
    auto_recovery: bool = True  # Whether to automatically recover
    half_open_threshold: float = 0.5  # Threshold for half-open state
    max_consecutive_failures: int = 3  # Max failures before permanent open
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class CircuitEvent:
    """Event recorded when a circuit breaker state changes"""
    circuit_id: str
    old_state: CircuitState
    new_state: CircuitState
    trigger_type: Optional[TriggerType]
    trigger_value: Any
    timestamp: float
    user_id: Optional[str] = None
    notes: str = ""
    recovery_time: Optional[int] = None


class CircuitBreaker:
    """
    Circuit breaker for trading system protection with configurable triggers,
    automated recovery, and comprehensive event logging.
    """

    def __init__(self,
                 config: CircuitBreakerConfig,
                 health_monitor: Optional[HealthMonitor] = None,
                 event_bus=None,
                 on_open: Optional[Callable[[CircuitEvent], None]] = None,
                 on_close: Optional[Callable[[CircuitEvent], None]] = None):
        """
        Initialize a circuit breaker.

        Args:
            config: Circuit breaker configuration
            health_monitor: Optional health monitor instance
            event_bus: Optional event bus instance
            on_open: Callback function when circuit opens
            on_close: Callback function when circuit closes
        """
        self.config = config
        self.health_monitor = health_monitor
        self.event_bus = event_bus or get_event_bus()
        self.on_open = on_open
        self.on_close = on_close

        # Generate circuit ID
        self.circuit_id = f"{config.circuit_type.value}_{config.scope}_{config.name}"

        # State management
        self._state = CircuitState.CLOSED
        self._state_change_time = time.time()
        self._consecutive_failures = 0
        self._lock = threading.RLock()
        self._recovery_timer = None

        # Metrics tracking
        self._metrics = {
            "total_trips": 0,
            "total_recovery_attempts": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "current_state": self._state.value,
            "time_in_current_state_sec": 0,
            "trip_history": [],
            "last_trip_reason": None
        }

        # Event history
        self._event_history = []
        self._max_history_size = 100

        # Register with event bus if it has the subscribe method
        if self.event_bus and hasattr(self.event_bus, 'subscribe'):
            self._register_event_handlers()
        else:
            logger.warning(f"Event bus is missing 'subscribe' method, skipping event handler registration")

        # Register with health monitor
        if self.health_monitor:
            self.health_monitor.register_component(
                f"circuit_breaker_{self.circuit_id}",
                "circuit_breaker"
            )

            # Initialize health status
            self.health_monitor.update_component_health(
                f"circuit_breaker_{self.circuit_id}",
                status=HealthStatus.HEALTHY,
                metrics={
                    "state": self._state.value,
                    "consecutive_failures": self._consecutive_failures,
                    "circuit_type": config.circuit_type.value,
                    "scope": config.scope
                }
            )

        logger.info(f"Circuit breaker {self.circuit_id} initialized in {self._state.value} state")

    def _register_event_handlers(self):
        """Register event handlers with the event bus"""
        # Register for relevant events based on circuit type
        if self.config.circuit_type == CircuitType.SYSTEM:
            self.event_bus.subscribe(
                EventTopics.SYSTEM_ERROR,
                self._handle_system_error
            )
        elif self.config.circuit_type == CircuitType.STRATEGY:
            self.event_bus.subscribe(
                f"{EventTopics.STRATEGY_STARTED}.{self.config.scope}",
                self._handle_strategy_event
            )
            self.event_bus.subscribe(
                f"{EventTopics.STRATEGY_STOPPED}.{self.config.scope}",
                self._handle_strategy_event
            )
            self.event_bus.subscribe(
                f"{EventTopics.SIGNAL_GENERATED}.{self.config.scope}",
                self._handle_strategy_event
            )
        elif self.config.circuit_type == CircuitType.MARKET:
            self.event_bus.subscribe(
                EventTopics.MARKET_DATA,
                self._handle_market_event
            )
        elif self.config.circuit_type == CircuitType.SYMBOL:
            self.event_bus.subscribe(
                f"{EventTopics.MARKET_DATA}.{self.config.scope}",
                self._handle_symbol_event
            )
            self.event_bus.subscribe(
                f"{EventTopics.PRICE_UPDATE}.{self.config.scope}",
                self._handle_symbol_event
            )
        elif self.config.circuit_type == CircuitType.MODEL:
            self.event_bus.subscribe(
                EventTopics.MODEL_PREDICTION,
                self._handle_model_event
            )
        elif self.config.circuit_type == CircuitType.EXECUTION:
            self.event_bus.subscribe(
                EventTopics.ORDER_SUBMITTED,
                self._handle_execution_event
            )
            self.event_bus.subscribe(
                EventTopics.ORDER_FILLED,
                self._handle_execution_event
            )
            self.event_bus.subscribe(
                EventTopics.ORDER_REJECTED,
                self._handle_execution_event
            )

    def get_state(self) -> CircuitState:
        """Get the current state of the circuit breaker"""
        with self._lock:
            return self._state

    def is_closed(self) -> bool:
        """Check if the circuit breaker is closed (allowing trading)"""
        with self._lock:
            return self._state == CircuitState.CLOSED

    def is_open(self) -> bool:
        """Check if the circuit breaker is open (blocking trading)"""
        with self._lock:
            return self._state == CircuitState.OPEN

    def is_half_open(self) -> bool:
        """Check if the circuit breaker is half-open (limited trading)"""
        with self._lock:
            return self._state == CircuitState.HALF_OPEN

    def trip(self,
             trigger_type: TriggerType,
             trigger_value: Any,
             reason: str = "",
             user_id: Optional[str] = None) -> bool:
        """
        Trip the circuit breaker to open state.

        Args:
            trigger_type: Type of trigger that caused the trip
            trigger_value: Value that caused the trip
            reason: Additional reason for the trip
            user_id: ID of user if manually triggered

        Returns:
            True if the circuit was tripped, False if already open
        """
        with self._lock:
            # If already open, no action needed
            if self._state == CircuitState.OPEN:
                return False

            old_state = self._state
            self._state = CircuitState.OPEN
            self._state_change_time = time.time()
            self._metrics["total_trips"] += 1
            self._metrics["current_state"] = self._state.value
            self._metrics["last_trip_reason"] = f"{trigger_type.value}: {trigger_value}"

            # Create circuit event
            event = CircuitEvent(
                circuit_id=self.circuit_id,
                old_state=old_state,
                new_state=self._state,
                trigger_type=trigger_type,
                trigger_value=trigger_value,
                timestamp=self._state_change_time,
                user_id=user_id,
                notes=reason,
                recovery_time=self.config.recovery_time if self.config.auto_recovery else None
            )

            # Add to history
            self._add_to_history(event)

            # Call callback if provided
            if self.on_open:
                try:
                    self.on_open(event)
                except Exception as e:
                    logger.error(f"Error in on_open callback: {str(e)}")

            # Publish event
            self._publish_circuit_event(event)

            # Schedule recovery if auto-recovery is enabled
            if self.config.auto_recovery:
                self._schedule_recovery()

            # Update health status
            if self.health_monitor:
                alert_level = AlertLevel.CRITICAL if trigger_type in [
                    TriggerType.LOSS_THRESHOLD, TriggerType.DRAWDOWN, TriggerType.API_FAILURE
                ] else AlertLevel.WARNING

                self.health_monitor.update_component_health(
                    f"circuit_breaker_{self.circuit_id}",
                    status=HealthStatus.CRITICAL,
                    metrics={
                        "state": self._state.value,
                        "trigger_type": trigger_type.value,
                        "trigger_value": str(trigger_value),
                        "tripped_at": self._state_change_time,
                        "recovery_time": self.config.recovery_time if self.config.auto_recovery else None
                    }
                )

            logger.warning(f"Circuit breaker {self.circuit_id} tripped to OPEN state: {trigger_type.value}")

            return True

    def reset(self, user_id: Optional[str] = None, reason: str = "") -> bool:
        """
        Manually reset the circuit breaker to closed state.

        Args:
            user_id: ID of user who reset the circuit
            reason: Reason for manual reset

        Returns:
            True if the circuit was reset, False if already closed
        """
        with self._lock:
            # If already closed, no action needed
            if self._state == CircuitState.CLOSED:
                return False

            old_state = self._state
            self._state = CircuitState.CLOSED
            self._state_change_time = time.time()
            self._consecutive_failures = 0
            self._metrics["current_state"] = self._state.value

            # Cancel any pending recovery timer
            if self._recovery_timer:
                self._recovery_timer.cancel()
                self._recovery_timer = None

            # Create circuit event
            event = CircuitEvent(
                circuit_id=self.circuit_id,
                old_state=old_state,
                new_state=self._state,
                trigger_type=TriggerType.MANUAL,
                trigger_value="manual_reset",
                timestamp=self._state_change_time,
                user_id=user_id,
                notes=reason
            )

            # Add to history
            self._add_to_history(event)

            # Call callback if provided
            if self.on_close:
                try:
                    self.on_close(event)
                except Exception as e:
                    logger.error(f"Error in on_close callback: {str(e)}")

            # Publish event
            self._publish_circuit_event(event)

            # Update health status
            if self.health_monitor:
                self.health_monitor.update_component_health(
                    f"circuit_breaker_{self.circuit_id}",
                    status=HealthStatus.HEALTHY,
                    metrics={
                        "state": self._state.value,
                        "reset_reason": reason,
                        "reset_at": self._state_change_time,
                        "user_id": user_id
                    }
                )

            logger.info(f"Circuit breaker {self.circuit_id} manually reset to CLOSED state")

            return True

    def test_condition(self,
                      trigger_type: TriggerType,
                      value: Any) -> bool:
        """
        Test if a value would trip the circuit breaker.

        Args:
            trigger_type: Type of trigger to test
            value: Value to test against trigger condition

        Returns:
            True if the condition would trip the circuit breaker
        """
        with self._lock:
            # Check if this trigger type is configured
            if trigger_type not in self.config.trigger_conditions:
                return False

            # Get trigger configuration
            trigger_config = self.config.trigger_conditions[trigger_type]

            # Handle different trigger types
            if trigger_type == TriggerType.LOSS_THRESHOLD:
                return self._test_loss_threshold(value, trigger_config)
            elif trigger_type == TriggerType.DRAWDOWN:
                return self._test_drawdown(value, trigger_config)
            elif trigger_type == TriggerType.VOLATILITY:
                return self._test_volatility(value, trigger_config)
            elif trigger_type == TriggerType.ERROR_RATE:
                return self._test_error_rate(value, trigger_config)
            elif trigger_type == TriggerType.LATENCY:
                return self._test_latency(value, trigger_config)
            elif trigger_type == TriggerType.API_FAILURE:
                return self._test_api_failure(value, trigger_config)
            elif trigger_type == TriggerType.DATA_QUALITY:
                return self._test_data_quality(value, trigger_config)
            elif trigger_type == TriggerType.LIQUIDITY:
                return self._test_liquidity(value, trigger_config)
            elif trigger_type == TriggerType.MODEL_DRIFT:
                return self._test_model_drift(value, trigger_config)
            elif trigger_type == TriggerType.CUSTOM:
                return self._test_custom(value, trigger_config)
            else:
                return False

    def _schedule_recovery(self):
        """Schedule automatic recovery attempt"""
        # Cancel any existing timer
        if self._recovery_timer:
            self._recovery_timer.cancel()

        # Schedule new timer
        self._recovery_timer = threading.Timer(
            self.config.recovery_time,
            self._attempt_recovery
        )
        self._recovery_timer.daemon = True
        self._recovery_timer.start()

        logger.info(f"Circuit breaker {self.circuit_id} scheduled for recovery in {self.config.recovery_time} seconds")

    def _attempt_recovery(self):
        """Attempt to recover the circuit breaker"""
        with self._lock:
            # Only try recovery if still open
            if self._state != CircuitState.OPEN:
                return

            self._metrics["total_recovery_attempts"] += 1

            # Move to half-open state for testing
            old_state = self._state
            self._state = CircuitState.HALF_OPEN
            self._state_change_time = time.time()
            self._metrics["current_state"] = self._state.value

            # Create circuit event
            event = CircuitEvent(
                circuit_id=self.circuit_id,
                old_state=old_state,
                new_state=self._state,
                trigger_type=None,
                trigger_value="auto_recovery",
                timestamp=self._state_change_time,
                notes="Automatic recovery attempt"
            )

            # Add to history
            self._add_to_history(event)

            # Publish event
            self._publish_circuit_event(event)

            # Update health status
            if self.health_monitor:
                self.health_monitor.update_component_health(
                    f"circuit_breaker_{self.circuit_id}",
                    status=HealthStatus.WARNING,
                    metrics={
                        "state": self._state.value,
                        "recovery_attempt": self._metrics["total_recovery_attempts"],
                        "half_open_at": self._state_change_time
                    }
                )

            logger.info(f"Circuit breaker {self.circuit_id} entering HALF-OPEN state for recovery testing")

    def test_recovery(self, success: bool, reason: str = ""):
        """
        Test if recovery was successful.

        Args:
            success: Whether the recovery test was successful
            reason: Reason for success/failure
        """
        with self._lock:
            # Only applicable in half-open state
            if self._state != CircuitState.HALF_OPEN:
                return

            if success:
                # Recovery succeeded, close the circuit
                old_state = self._state
                self._state = CircuitState.CLOSED
                self._state_change_time = time.time()
                self._consecutive_failures = 0
                self._metrics["current_state"] = self._state.value
                self._metrics["successful_recoveries"] += 1

                # Create circuit event
                event = CircuitEvent(
                    circuit_id=self.circuit_id,
                    old_state=old_state,
                    new_state=self._state,
                    trigger_type=None,
                    trigger_value="recovery_success",
                    timestamp=self._state_change_time,
                    notes=reason or "Recovery test successful"
                )

                # Add to history
                self._add_to_history(event)

                # Call callback if provided
                if self.on_close:
                    try:
                        self.on_close(event)
                    except Exception as e:
                        logger.error(f"Error in on_close callback: {str(e)}")

                # Publish event
                self._publish_circuit_event(event)

                # Update health status
                if self.health_monitor:
                    self.health_monitor.update_component_health(
                        f"circuit_breaker_{self.circuit_id}",
                        status=HealthStatus.HEALTHY,
                        metrics={
                            "state": self._state.value,
                            "recovery_success": True,
                            "recovered_at": self._state_change_time,
                            "reason": reason
                        }
                    )

                logger.info(f"Circuit breaker {self.circuit_id} successfully recovered to CLOSED state")
            else:
                # Recovery failed, re-open the circuit
                old_state = self._state
                self._state = CircuitState.OPEN
                self._state_change_time = time.time()
                self._consecutive_failures += 1
                self._metrics["current_state"] = self._state.value
                self._metrics["failed_recoveries"] += 1

                # Create circuit event
                event = CircuitEvent(
                    circuit_id=self.circuit_id,
                    old_state=old_state,
                    new_state=self._state,
                    trigger_type=None,
                    trigger_value="recovery_failure",
                    timestamp=self._state_change_time,
                    notes=reason or "Recovery test failed",
                    recovery_time=self.config.recovery_time if self.config.auto_recovery and
                                   self._consecutive_failures < self.config.max_consecutive_failures else None
                )

                # Add to history
                self._add_to_history(event)

                # Publish event
                self._publish_circuit_event(event)

                # Update health status
                if self.health_monitor:
                    self.health_monitor.update_component_health(
                        f"circuit_breaker_{self.circuit_id}",
                        status=HealthStatus.CRITICAL,
                        metrics={
                            "state": self._state.value,
                            "recovery_success": False,
                            "consecutive_failures": self._consecutive_failures,
                            "failed_at": self._state_change_time,
                            "reason": reason
                        }
                    )

                # Schedule recovery if not exceeded max consecutive failures
                if self.config.auto_recovery and self._consecutive_failures < self.config.max_consecutive_failures:
                    self._schedule_recovery()
                    logger.warning(f"Circuit breaker {self.circuit_id} recovery failed, scheduling another attempt")
                else:
                    # Cancel and clear the recovery timer when max failures is reached
                    if self._recovery_timer:
                        self._recovery_timer.cancel()
                        self._recovery_timer = None
                    logger.error(f"Circuit breaker {self.circuit_id} recovery failed {self._consecutive_failures} times, remaining OPEN")

    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        with self._lock:
            # Update time in current state
            self._metrics["time_in_current_state_sec"] = time.time() - self._state_change_time

            return {
                "circuit_id": self.circuit_id,
                "circuit_type": self.config.circuit_type.value,
                "scope": self.config.scope,
                "state": self._state.value,
                "consecutive_failures": self._consecutive_failures,
                "metrics": self._metrics,
                "recovery_time": self.config.recovery_time if self.config.auto_recovery else None,
                "auto_recovery": self.config.auto_recovery,
                "max_consecutive_failures": self.config.max_consecutive_failures,
                "current_state_since": datetime.datetime.fromtimestamp(self._state_change_time).isoformat()
            }

    def get_events(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent circuit breaker events.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of event dictionaries
        """
        with self._lock:
            events = []
            for event in reversed(self._event_history[-limit:]):
                events.append({
                    "circuit_id": event.circuit_id,
                    "old_state": event.old_state.value,
                    "new_state": event.new_state.value,
                    "trigger_type": event.trigger_type.value if event.trigger_type else None,
                    "trigger_value": event.trigger_value,
                    "timestamp": event.timestamp,
                    "datetime": datetime.datetime.fromtimestamp(event.timestamp).isoformat(),
                    "user_id": event.user_id,
                    "notes": event.notes,
                    "recovery_time": event.recovery_time
                })
            return events

    def _add_to_history(self, event: CircuitEvent):
        """Add an event to the history"""
        self._event_history.append(event)

        # Limit history size
        if len(self._event_history) > self._max_history_size:
            self._event_history = self._event_history[-self._max_history_size:]

        # Add to metrics history
        self._metrics["trip_history"].append({
            "timestamp": event.timestamp,
            "old_state": event.old_state.value,
            "new_state": event.new_state.value,
            "trigger_type": event.trigger_type.value if event.trigger_type else None,
            "trigger_value": str(event.trigger_value)
        })

        # Limit metrics history size
        if len(self._metrics["trip_history"]) > 10:
            self._metrics["trip_history"] = self._metrics["trip_history"][-10:]

    def _publish_circuit_event(self, event: CircuitEvent):
        """Publish a circuit breaker event to the event bus"""
        # Convert event to dictionary for publishing
        event_data = {
            "circuit_id": event.circuit_id,
            "circuit_type": self.config.circuit_type.value,
            "scope": self.config.scope,
            "old_state": event.old_state.value,
            "new_state": event.new_state.value,
            "trigger_type": event.trigger_type.value if event.trigger_type else None,
            "trigger_value": event.trigger_value,
            "timestamp": event.timestamp,
            "user_id": event.user_id,
            "notes": event.notes,
            "recovery_time": event.recovery_time
        }

        # Only publish if event_bus exists and has publish method
        if self.event_bus and hasattr(self.event_bus, 'publish'):
            # Create event for event bus
            bus_event = Event(
                topic="system.circuit_breaker",
                data=event_data,
                priority=EventPriority.HIGH if event.new_state == CircuitState.OPEN else EventPriority.NORMAL,
                source="circuit_breaker"
            )

            # Publish event
            self.event_bus.publish(bus_event)
        else:
            logger.warning(f"Event bus is missing or doesn't have 'publish' method, skipping event publication")

    # Event handlers for different circuit types
    def _handle_system_error(self, event: Event):
        """Handle system error event"""
        if self._state != CircuitState.CLOSED:
            return

        # Check if error rate trigger is configured
        if TriggerType.ERROR_RATE in self.config.trigger_conditions:
            # Increment error count and check threshold
            error_data = {
                "type": event.data.get("error_type", "unknown"),
                "component": event.data.get("component", "unknown"),
                "error": event.data.get("error", "")
            }

            if self.test_condition(TriggerType.ERROR_RATE, error_data):
                self.trip(
                    TriggerType.ERROR_RATE,
                    error_data,
                    f"System error threshold reached: {error_data['type']}"
                )

    def _handle_strategy_event(self, event: Event):
        """Handle strategy events"""
        if self._state != CircuitState.CLOSED and self._state != CircuitState.HALF_OPEN:
            return

        # Check for strategy-specific triggers
        # Modified to be more flexible with the event topic format
        if 'signal_generated' in event.topic.lower() or event.topic.startswith(EventTopics.SIGNAL_GENERATED):
            # Check for model drift
            if TriggerType.MODEL_DRIFT in self.config.trigger_conditions and 'confidence' in event.data:
                confidence = event.data.get('confidence', 1.0)
                if self.test_condition(TriggerType.MODEL_DRIFT, confidence):
                    self.trip(
                        TriggerType.MODEL_DRIFT,
                        confidence,
                        f"Strategy signal confidence below threshold: {confidence}"
                    )

        # If in half-open state, use this as an opportunity to test recovery
        if self._state == CircuitState.HALF_OPEN:
            # Strategy events can be used to confirm successful recovery
            if event.topic.startswith(EventTopics.STRATEGY_STARTED):
                self.test_recovery(True, "Strategy started successfully")

    def _handle_market_event(self, event: Event):
        """Handle market data events"""
        if self._state != CircuitState.CLOSED and self._state != CircuitState.HALF_OPEN:
            return

        # Check for market-specific triggers
        if TriggerType.VOLATILITY in self.config.trigger_conditions and 'volatility' in event.data:
            volatility = event.data.get('volatility', 0.0)
            if self.test_condition(TriggerType.VOLATILITY, volatility):
                self.trip(
                    TriggerType.VOLATILITY,
                    volatility,
                    f"Market volatility above threshold: {volatility}"
                )

        # Check for data quality issues
        if TriggerType.DATA_QUALITY in self.config.trigger_conditions:
            data_quality = event.data.get('data_quality', {})
            if data_quality and self.test_condition(TriggerType.DATA_QUALITY, data_quality):
                self.trip(
                    TriggerType.DATA_QUALITY,
                    data_quality,
                    f"Market data quality below threshold"
                )

    def _handle_symbol_event(self, event: Event):
        """Handle symbol-specific events"""
        if self._state != CircuitState.CLOSED and self._state != CircuitState.HALF_OPEN:
            return

        # Check for symbol-specific triggers
        if TriggerType.VOLATILITY in self.config.trigger_conditions and 'volatility' in event.data:
            volatility = event.data.get('volatility', 0.0)
            if self.test_condition(TriggerType.VOLATILITY, volatility):
                self.trip(
                    TriggerType.VOLATILITY,
                    volatility,
                    f"Symbol {self.config.scope} volatility above threshold: {volatility}"
                )

        # Check for liquidity issues
        if TriggerType.LIQUIDITY in self.config.trigger_conditions and 'liquidity' in event.data:
            liquidity = event.data.get('liquidity', 1.0)
            if self.test_condition(TriggerType.LIQUIDITY, liquidity):
                self.trip(
                    TriggerType.LIQUIDITY,
                    liquidity,
                    f"Symbol {self.config.scope} liquidity below threshold: {liquidity}"
                )

    def _handle_model_event(self, event: Event):
        """Handle model prediction events"""
        if self._state != CircuitState.CLOSED and self._state != CircuitState.HALF_OPEN:
            return

        # Check model-specific triggers
        if event.data.get('model_id') and event.data.get('model_type'):
            model_id = event.data.get('model_id')
            model_scope = event.data.get('model_type')

            # Only process events for this model
            if model_scope != self.config.scope:
                return

            # Check for model drift
            if TriggerType.MODEL_DRIFT in self.config.trigger_conditions and 'drift' in event.data:
                drift = event.data.get('drift', 0.0)
                if self.test_condition(TriggerType.MODEL_DRIFT, drift):
                    self.trip(
                        TriggerType.MODEL_DRIFT,
                        drift,
                        f"Model {model_id} drift above threshold: {drift}"
                    )
    def _handle_execution_event(self, event: Event):
        """Handle execution events"""
        if self._state != CircuitState.CLOSED and self._state != CircuitState.HALF_OPEN:
            return

        # Check execution-specific triggers
        if event.topic == EventTopics.ORDER_REJECTED:
            # Check for API failures
            if TriggerType.API_FAILURE in self.config.trigger_conditions:
                # Track rejection rate
                error_data = {
                    "type": "order_rejected",
                    "order_id": event.data.get("order_id", "unknown"),
                    "reason": event.data.get("reason", "")
                }

                if self.test_condition(TriggerType.API_FAILURE, error_data):
                    self.trip(
                        TriggerType.API_FAILURE,
                        error_data,
                        f"Order rejection rate above threshold: {error_data['reason']}"
                    )

        # If in half-open state, use successful order executions to confirm recovery
        if self._state == CircuitState.HALF_OPEN and event.topic == EventTopics.ORDER_FILLED:
            # Order fill can be used to confirm successful recovery
            self.test_recovery(True, "Order executed successfully during recovery test")

    # Trigger condition test methods
    def _test_loss_threshold(self, value: float, config: Dict[str, Any]) -> bool:
        """Test loss threshold condition"""
        threshold = config.get("threshold", 0.0)
        return value <= threshold  # Negative values indicate loss

    def _test_drawdown(self, value: float, config: Dict[str, Any]) -> bool:
        """Test drawdown condition"""
        threshold = config.get("threshold", 0.0)
        return value >= threshold

    def _test_volatility(self, value: float, config: Dict[str, Any]) -> bool:
        """Test volatility condition"""
        threshold = config.get("threshold", 0.0)
        return value >= threshold

    def _test_error_rate(self, value: Any, config: Dict[str, Any]) -> bool:
        """Test error rate condition"""
        error_count = config.get("error_count", 0) + 1
        error_window = config.get("error_window", 10)
        threshold = config.get("threshold", 0.5)

        # Update error count
        config["error_count"] = error_count

        # Check if count exceeds threshold within window
        return error_count >= threshold * error_window

    def _test_latency(self, value: float, config: Dict[str, Any]) -> bool:
        """Test latency condition"""
        threshold = config.get("threshold", 1000.0)  # ms
        return value >= threshold

    def _test_api_failure(self, value: Any, config: Dict[str, Any]) -> bool:
        """Test API failure condition"""
        failure_count = config.get("failure_count", 0) + 1
        failure_window = config.get("failure_window", 10)
        threshold = config.get("threshold", 0.3)

        # Update failure count
        config["failure_count"] = failure_count

        # Check if count exceeds threshold within window
        return failure_count >= threshold * failure_window

    def _test_data_quality(self, value: Any, config: Dict[str, Any]) -> bool:
        """Test data quality condition"""
        quality_score = value.get("quality_score", 1.0)
        threshold = config.get("threshold", 0.7)
        return quality_score < threshold

    def _test_liquidity(self, value: float, config: Dict[str, Any]) -> bool:
        """Test liquidity condition"""
        threshold = config.get("threshold", 0.3)
        return value < threshold

    def _test_model_drift(self, value: float, config: Dict[str, Any]) -> bool:
        """Test model drift condition"""
        threshold = config.get("threshold", 0.3)
        # FIX: Model drift is typically below a threshold, not above
        return value < threshold

    def _test_custom(self, value: Any, config: Dict[str, Any]) -> bool:
        """Test custom condition"""
        # Custom logic can be implemented by subclasses
        custom_threshold = config.get("threshold")
        if custom_threshold is not None:
            if isinstance(value, (int, float)) and isinstance(custom_threshold, (int, float)):
                return value >= custom_threshold

        # If not implemented or no threshold, return False
        return False


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers"""

    def __init__(self,
                 health_monitor: Optional[HealthMonitor] = None,
                 event_bus=None):
        """
        Initialize the circuit breaker registry.

        Args:
            health_monitor: Optional health monitor instance
            event_bus: Optional event bus instance
        """
        self.health_monitor = health_monitor
        self.event_bus = event_bus or get_event_bus()
        self._circuit_breakers = {}
        self._lock = threading.RLock()

        # Register event handlers only if event_bus has subscribe method
        if self.event_bus and hasattr(self.event_bus, 'subscribe'):
            # Register event handler for manual trips
            self.event_bus.subscribe(
                "system.circuit_breaker.manual_trip",
                self._handle_manual_trip
            )

            # Register event handler for manual resets
            self.event_bus.subscribe(
                "system.circuit_breaker.manual_reset",
                self._handle_manual_reset
            )

            logger.info("Circuit breaker registry initialized with event bus")
        else:
            logger.warning("Circuit breaker registry initialized without event bus or event bus missing subscribe method")

    def register_circuit_breaker(self,
                                config: CircuitBreakerConfig,
                                on_open: Optional[Callable[[CircuitEvent], None]] = None,
                                on_close: Optional[Callable[[CircuitEvent], None]] = None) -> CircuitBreaker:
        """
        Register a new circuit breaker.

        Args:
            config: Circuit breaker configuration
            on_open: Callback function when circuit opens
            on_close: Callback function when circuit closes

        Returns:
            The created circuit breaker instance
        """
        with self._lock:
            # Generate circuit ID
            circuit_id = f"{config.circuit_type.value}_{config.scope}_{config.name}"

            # Check if already registered
            if circuit_id in self._circuit_breakers:
                logger.warning(f"Circuit breaker {circuit_id} already registered")
                return self._circuit_breakers[circuit_id]

            # Create new circuit breaker
            circuit_breaker = CircuitBreaker(
                config=config,
                health_monitor=self.health_monitor,
                event_bus=self.event_bus,
                on_open=on_open,
                on_close=on_close
            )

            # Register it
            self._circuit_breakers[circuit_id] = circuit_breaker

            logger.info(f"Registered circuit breaker {circuit_id}")

            return circuit_breaker

    def get_circuit_breaker(self, circuit_id: str) -> Optional[CircuitBreaker]:
        """
        Get a circuit breaker by ID.

        Args:
            circuit_id: Circuit breaker ID

        Returns:
            Circuit breaker instance or None if not found
        """
        with self._lock:
            return self._circuit_breakers.get(circuit_id)

    def get_circuit_breakers(self,
                            circuit_type: Optional[CircuitType] = None,
                            scope: Optional[str] = None) -> List[CircuitBreaker]:
        """
        Get circuit breakers matching criteria.

        Args:
            circuit_type: Optional filter by circuit type
            scope: Optional filter by scope

        Returns:
            List of matching circuit breaker instances
        """
        with self._lock:
            result = []

            for cb in self._circuit_breakers.values():
                # Apply filters
                if circuit_type and cb.config.circuit_type != circuit_type:
                    continue

                if scope and cb.config.scope != scope:
                    continue

                result.append(cb)

            return result

    def get_all_circuit_breaker_states(self) -> Dict[str, CircuitState]:
        """
        Get current states of all circuit breakers.

        Returns:
            Dictionary mapping circuit IDs to states
        """
        with self._lock:
            return {
                circuit_id: cb.get_state()
                for circuit_id, cb in self._circuit_breakers.items()
            }

    def trip_all(self,
                circuit_type: CircuitType,
                trigger_type: TriggerType,
                trigger_value: Any,
                reason: str = "",
                user_id: Optional[str] = None) -> List[str]:
        """
        Trip all circuit breakers of a specific type.

        Args:
            circuit_type: Type of circuits to trip
            trigger_type: Type of trigger
            trigger_value: Trigger value
            reason: Reason for the trip
            user_id: ID of user if manually triggered

        Returns:
            List of tripped circuit IDs
        """
        with self._lock:
            tripped = []

            for cb in self.get_circuit_breakers(circuit_type=circuit_type):
                if cb.trip(trigger_type, trigger_value, reason, user_id):
                    tripped.append(cb.circuit_id)

            return tripped

    def reset_all(self,
                 circuit_type: Optional[CircuitType] = None,
                 scope: Optional[str] = None,
                 user_id: Optional[str] = None,
                 reason: str = "") -> List[str]:
        """
        Reset all circuit breakers matching criteria.

        Args:
            circuit_type: Optional filter by circuit type
            scope: Optional filter by scope
            user_id: ID of user who reset the circuits
            reason: Reason for manual reset

        Returns:
            List of reset circuit IDs
        """
        with self._lock:
            reset = []

            for cb in self.get_circuit_breakers(circuit_type=circuit_type, scope=scope):
                if cb.reset(user_id, reason):
                    reset.append(cb.circuit_id)

            return reset

    def _handle_manual_trip(self, event: Event):
        """Handle manual trip event"""
        if not event.data:
            return

        circuit_id = event.data.get("circuit_id")
        circuit_type = event.data.get("circuit_type")
        scope = event.data.get("scope")
        trigger_type = event.data.get("trigger_type", TriggerType.MANUAL.value)
        trigger_value = event.data.get("trigger_value", "manual")
        reason = event.data.get("reason", "Manual trip")
        user_id = event.data.get("user_id")

        # Validate trigger type
        try:
            trigger_type_enum = TriggerType(trigger_type)
        except ValueError:
            trigger_type_enum = TriggerType.MANUAL

        # Trip specific circuit if ID provided
        if circuit_id:
            cb = self.get_circuit_breaker(circuit_id)
            if cb:
                cb.trip(trigger_type_enum, trigger_value, reason, user_id)

        # Trip circuits by type and scope
        elif circuit_type:
            try:
                circuit_type_enum = CircuitType(circuit_type)
                self.trip_all(circuit_type_enum, trigger_type_enum, trigger_value, reason, user_id)
            except ValueError:
                logger.error(f"Invalid circuit type: {circuit_type}")

    def _handle_manual_reset(self, event: Event):
        """Handle manual reset event"""
        if not event.data:
            return

        circuit_id = event.data.get("circuit_id")
        circuit_type = event.data.get("circuit_type")
        scope = event.data.get("scope")
        reason = event.data.get("reason", "Manual reset")
        user_id = event.data.get("user_id")

        # Reset specific circuit if ID provided
        if circuit_id:
            cb = self.get_circuit_breaker(circuit_id)
            if cb:
                cb.reset(user_id, reason)

        # Reset circuits by type and scope
        else:
            circuit_type_enum = None
            if circuit_type:
                try:
                    circuit_type_enum = CircuitType(circuit_type)
                except ValueError:
                    logger.error(f"Invalid circuit type: {circuit_type}")
                    return

            self.reset_all(circuit_type_enum, scope, user_id, reason)


# Global registry
_circuit_breaker_registry = None

def get_circuit_breaker_registry(health_monitor: Optional[HealthMonitor] = None,
                                event_bus=None) -> CircuitBreakerRegistry:
    """
    Get the global circuit breaker registry.

    Args:
        health_monitor: Optional health monitor instance
        event_bus: Optional event bus instance

    Returns:
        The circuit breaker registry instance
    """
    global _circuit_breaker_registry
    if _circuit_breaker_registry is None:
        _circuit_breaker_registry = CircuitBreakerRegistry(health_monitor, event_bus)
    return _circuit_breaker_registry

def create_circuit_breaker(name: str,
                         circuit_type: CircuitType,
                         scope: str,
                         trigger_conditions: Dict[TriggerType, Dict[str, Any]],
                         recovery_time: int = 300,
                         auto_recovery: bool = True,
                         health_monitor: Optional[HealthMonitor] = None,
                         event_bus=None,
                         on_open: Optional[Callable[[CircuitEvent], None]] = None,
                         on_close: Optional[Callable[[CircuitEvent], None]] = None) -> CircuitBreaker:
    """
    Create and register a new circuit breaker.

    Args:
        name: Circuit breaker name
        circuit_type: Type of circuit breaker
        scope: Scope identifier
        trigger_conditions: Dictionary of trigger conditions
        recovery_time: Time in seconds before trying recovery
        auto_recovery: Whether to automatically recover
        health_monitor: Optional health monitor instance
        event_bus: Optional event bus instance
        on_open: Callback function when circuit opens
        on_close: Callback function when circuit closes

    Returns:
        The created circuit breaker instance
    """
    config = CircuitBreakerConfig(
        name=name,
        circuit_type=circuit_type,
        scope=scope,
        trigger_conditions=trigger_conditions,
        recovery_time=recovery_time,
        auto_recovery=auto_recovery
    )

    registry = get_circuit_breaker_registry(health_monitor, event_bus)
    return registry.register_circuit_breaker(config, on_open, on_close)