"""
event_bus.py - Unified Event Management System

This module implements a centralized event bus with prioritization, backpressure
handling, and asynchronous event processing. It serves as the communication
backbone of the entire trading system.
"""

import threading
import queue
import time
import logging
import uuid
import os
import pickle
import re
import json
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Set, Tuple, Pattern, Union
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache

# Try to import faster serialization libraries
try:
    import msgpack
    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False

try:
    import ujson
    UJSON_AVAILABLE = True
except ImportError:
    UJSON_AVAILABLE = False

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Priority levels for event processing"""
    CRITICAL = 0  # Immediate processing needed (e.g., system shutdown, circuit breaker)
    HIGH = 1      # Urgent processing (e.g., order execution, risk alerts)
    NORMAL = 2    # Standard events (e.g., regular position updates)
    LOW = 3       # Non-urgent events (e.g., UI updates, periodic stats)
    BACKGROUND = 4  # Processing when system load permits (e.g., analytics)


class EventTopics:
    """Standard event topics used throughout the system"""
    # System lifecycle events
    SYSTEM_START = "system.started"
    SYSTEM_SHUTDOWN = "system.stopped"
    SYSTEM_ERROR = "system.error"
    COMPONENT_REGISTERED = "system.component.registered"
    COMPONENT_STARTED = "system.component.started"
    COMPONENT_STOPPED = "system.component.stopped"
    COMPONENT_ERROR = "system.component.error"
    SYSTEM_HEARTBEAT = "system.heartbeat"
    SYSTEM_CONFIG_UPDATE = "system.config_update"

    # Market data events
    MARKET_DATA = "market.data"
    PRICE_UPDATE = "market.price_update"
    TICK_DATA = "market.tick"
    BAR_DATA = "market.bar"
    MARKET_DEPTH = "market.depth"
    ORDERBOOK_UPDATE = "market.orderbook"
    TRADE_UPDATE = "market.trade"
    VOLUME_PROFILE = "market.volume_profile"

    # Order and execution events
    ORDER_CREATED = "order.created"
    ORDER_SUBMITTED = "order.submitted"
    ORDER_FILLED = "order.filled"
    ORDER_PARTIAL_FILL = "order.partial_fill"
    ORDER_CANCELED = "order.canceled"
    ORDER_REJECTED = "order.rejected"
    ORDER_EXPIRED = "order.expired"
    ORDER_UPDATE = "order.update"

    # Strategy events
    SIGNAL_GENERATED = "strategy.signal"
    STRATEGY_STARTED = "strategy.started"
    STRATEGY_STOPPED = "strategy.stopped"

    # Risk and portfolio events
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"
    POSITION_UPDATED = "position.updated"
    RISK_LIMIT_BREACH = "risk.limit_breach"
    RISK_ALERT = "risk.alert"
    MARGIN_CALL = "risk.margin_call"
    PORTFOLIO_UPDATED = "portfolio.updated"
    DRAWDOWN_ALERT = "risk.drawdown_alert"
    EXPOSURE_UPDATE = "risk.exposure_update"

    # System health events
    HEALTH_CHECK = "system.health_check"
    CIRCUIT_BREAKER_TRIGGERED = "system.circuit_breaker"
    LATENCY_MEASUREMENT = "performance.latency"
    RESOURCE_USAGE = "performance.resource_usage"
    THROUGHPUT_MEASUREMENT = "performance.throughput"

    # Trading mode events
    MODE_CHANGED = "system.mode_changed"

    # Regime events
    REGIME_DETECTED = "regime.detected"
    REGIME_CHANGED = "regime.changed"
    REGIME_PROBABILITY_UPDATE = "regime.probability_update"

    # Model events
    MODEL_TRAINED = "model.trained"
    MODEL_DEPLOYED = "model.deployed"
    MODEL_PREDICTION = "model.prediction"
    MODEL_UPDATED = "analytics.model_updated"

    # Feature events
    FEATURE_CALCULATED = "feature.calculated"
    FEATURE_IMPORTANCE = "feature.importance"

    # Backtest events
    BACKTEST_STARTED = "backtest.started"
    BACKTEST_COMPLETED = "backtest.completed"
    BACKTEST_RESULT = "backtest.result"

    # Persistence events
    STATE_SAVED = "persistence.state_saved"
    STATE_LOADED = "persistence.state_loaded"

    # User events
    USER_COMMAND = "user.command"
    USER_NOTIFICATION = "user.notification"
    USER_ALERT = "user.alert"
    USER_PREFERENCE_UPDATE = "user.preference_update"

    # Connectivity events
    BROKER_STATUS = "connectivity.broker_status"
    MARKET_DATA_STATUS = "connectivity.market_data_status"
    API_RATE_LIMIT = "connectivity.api_rate_limit"
    CONNECTIVITY_CHANGE = "connectivity.status_change"

    # Analytics events
    METRICS_UPDATE = "analytics.metrics_update"
    SIGNAL_GENERATED = "analytics.signal_generated"


@dataclass
class Event:
    """Event data structure for passing messages between components"""
    topic: str
    data: Any
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None
    priority: EventPriority = EventPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Add processing metrics for monitoring
    __created_at: float = field(default_factory=time.time, init=False)
    __delivery_metrics: Dict[str, Any] = field(default_factory=dict, init=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        return {
            "event_id": self.event_id,
            "topic": self.topic,
            "data": self.data,
            "timestamp": self.timestamp,
            "source": self.source,
            "priority": self.priority.name,
            "metadata": self.metadata
        }

    def record_delivery_attempt(self, subscriber_id: str):
        """Record a delivery attempt for monitoring"""
        current_time = time.time()
        self.__delivery_metrics[subscriber_id] = {
            "delivery_time": current_time,
            "queue_time": current_time - self.__created_at
        }

    def record_delivery_complete(self, subscriber_id: str):
        """Record completed delivery for monitoring"""
        if subscriber_id in self.__delivery_metrics:
            self.__delivery_metrics[subscriber_id]["processing_time"] = (
                time.time() - self.__delivery_metrics[subscriber_id]["delivery_time"]
            )

    def get_delivery_metrics(self) -> Dict[str, Any]:
        """Get delivery metrics for monitoring"""
        return {
            "created_at": self.__created_at,
            "age": time.time() - self.__created_at,
            "deliveries": self.__delivery_metrics
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary"""
        priority_str = data.pop("priority")
        data["priority"] = EventPriority[priority_str]
        return cls(**data)


class EventBusMetrics:
    """Metrics collection for EventBus monitoring"""
    def __init__(self, window_size: int = 1000):
        self.published_count: Dict[str, int] = {}
        self.delivered_count: Dict[str, int] = {}
        self.dropped_count: Dict[str, int] = {}
        self.error_count: Dict[str, int] = {}
        self.queue_sizes: Dict[EventPriority, int] = {p: 0 for p in EventPriority}
        self.queue_latencies: Dict[EventPriority, List[float]] = {p: [] for p in EventPriority}
        self.processing_times: Dict[str, List[float]] = {}
        self.max_processing_times: Dict[str, float] = {}
        self.avg_processing_times: Dict[str, float] = {}
        self.p95_processing_times: Dict[str, float] = {}
        self.subscriber_counts: Dict[str, int] = {}
        self.last_reset_time = time.time()
        self.window_size = window_size
        self._lock = threading.RLock()

    def record_publish(self, topic: str):
        """Record a published event"""
        with self._lock:
            self.published_count[topic] = self.published_count.get(topic, 0) + 1

    def record_delivery(self, topic: str, processing_time: float):
        """Record a delivered event with processing time"""
        with self._lock:
            self.delivered_count[topic] = self.delivered_count.get(topic, 0) + 1

            if topic not in self.processing_times:
                self.processing_times[topic] = []

            times = self.processing_times[topic]
            times.append(processing_time)

            # Keep only the latest window_size measurements
            if len(times) > self.window_size:
                times.pop(0)

            # Update max processing time
            current_max = self.max_processing_times.get(topic, 0)
            if processing_time > current_max:
                self.max_processing_times[topic] = processing_time

            # Update average processing time
            self.avg_processing_times[topic] = sum(times) / len(times)

            # Update p95 processing time if we have enough samples
            if len(times) >= 20:
                sorted_times = sorted(times)
                idx = int(len(sorted_times) * 0.95)
                self.p95_processing_times[topic] = sorted_times[idx]

    def record_drop(self, topic: str):
        """Record a dropped event"""
        with self._lock:
            self.dropped_count[topic] = self.dropped_count.get(topic, 0) + 1

    def record_error(self, topic: str):
        """Record an error during event processing"""
        with self._lock:
            self.error_count[topic] = self.error_count.get(topic, 0) + 1

    def update_queue_size(self, priority: EventPriority, size: int):
        """Update queue size for a priority level"""
        with self._lock:
            self.queue_sizes[priority] = size

    def record_queue_latency(self, priority: EventPriority, latency: float):
        """Record queue processing latency"""
        with self._lock:
            latencies = self.queue_latencies[priority]
            latencies.append(latency)

            # Keep only the latest window_size measurements
            if len(latencies) > self.window_size:
                latencies.pop(0)

    def update_subscriber_count(self, topic: str, count: int):
        """Update subscriber count for a topic"""
        with self._lock:
            self.subscriber_counts[topic] = count

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        with self._lock:
            # Calculate queue latency statistics
            queue_latency_stats = {}
            for priority, latencies in self.queue_latencies.items():
                if latencies:
                    queue_latency_stats[priority.name] = {
                        "avg": sum(latencies) / len(latencies),
                        "max": max(latencies),
                        "p95": sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) >= 20 else None
                    }
                else:
                    queue_latency_stats[priority.name] = {"avg": None, "max": None, "p95": None}

            return {
                "published_events": dict(self.published_count),
                "delivered_events": dict(self.delivered_count),
                "dropped_events": dict(self.dropped_count),
                "error_count": dict(self.error_count),
                "queue_sizes": {p.name: size for p, size in self.queue_sizes.items()},
                "queue_latencies": queue_latency_stats,
                "max_processing_times": dict(self.max_processing_times),
                "avg_processing_times": dict(self.avg_processing_times),
                "p95_processing_times": dict(self.p95_processing_times),
                "subscriber_counts": dict(self.subscriber_counts),
                "metrics_age_seconds": time.time() - self.last_reset_time
            }

    def reset(self):
        """Reset metrics"""
        with self._lock:
            self.published_count = {}
            self.delivered_count = {}
            self.dropped_count = {}
            self.error_count = {}
            self.processing_times = {}
            self.max_processing_times = {}
            self.avg_processing_times = {}
            self.p95_processing_times = {}
            self.last_reset_time = time.time()

            # Keep queue sizes and latencies, just clear the history
            for priority in EventPriority:
                self.queue_latencies[priority] = []


class PatternSubscription:
    """Pattern-based subscription for topic matching"""
    def __init__(self, pattern: str, callback: Callable[[Event], None], handle_errors: bool = False, subscriber_id: str = None):
        self.pattern_str = pattern
        self.subscriber_id = subscriber_id or f"pattern_{uuid.uuid4().hex[:8]}"
        # Convert glob patterns to regex (e.g., "market.*" -> "market\\..*")
        if pattern == "*":
            # Special case for wildcard
            regex_pattern = ".*"
        elif pattern.endswith(".*"):
            # Prefix wildcard (e.g., "market.*")
            prefix = pattern[:-2]
            regex_pattern = f"^{re.escape(prefix)}\\..*$"
        elif pattern.startswith("*."):
            # Suffix wildcard (e.g., "*.tick")
            suffix = pattern[1:]
            regex_pattern = f"^.*\\{suffix}$"
        else:
            # Exact match or custom pattern
            regex_pattern = f"^{re.escape(pattern)}$"

        self.pattern = re.compile(regex_pattern)
        self.callback = callback
        self.handle_errors = handle_errors


class EventBus:
    """
    Enhanced Event Bus with priority-based messaging, pattern matching, and comprehensive metrics.
    Implements the publish-subscribe pattern for component communication.
    """

    def __init__(
        self,
        persistence_dir: Optional[str] = None,
        max_queue_size: int = 10000,
        metrics_enabled: bool = True,
        metrics_interval: float = 60.0,  # How often to log metrics in seconds
        metrics_window_size: int = 1000,  # Number of measurements to keep for statistics
        serialization_format: str = 'auto'  # 'auto', 'pickle', 'msgpack', or 'ujson'
    ):
        """
        Initialize the event bus.

        Args:
            persistence_dir: Directory for persisting events, if None, persistence is disabled
            max_queue_size: Maximum size for event queues (0 for unlimited)
            metrics_enabled: Whether to collect and report metrics
            metrics_interval: Interval in seconds for metrics reporting
            metrics_window_size: Number of measurements to keep for statistical calculations
            serialization_format: Format to use for event serialization ('auto', 'pickle', 'msgpack', 'ujson')
        """
        self._subscribers: Dict[str, List[Tuple[Callable, bool, str]]] = {}
        self._pattern_subscribers: List[PatternSubscription] = []
        self._priority_queues: Dict[EventPriority, queue.PriorityQueue] = {
            priority: queue.PriorityQueue(maxsize=max_queue_size if max_queue_size > 0 else 0)
            for priority in EventPriority
        }
        self._running = False
        self._processors: List[threading.Thread] = []
        self._lock = threading.RLock()
        self._pattern_lock = threading.RLock()  # Separate lock for pattern matching
        self._stats_lock = threading.RLock()    # Separate lock for statistics
        self._logger = logger
        self._error_handlers: Dict[str, List[Tuple[Callable, str]]] = {}
        self._persistence_dir = Path(persistence_dir) if persistence_dir else None
        self._persistent_topics: Set[str] = set()
        self._backpressure_callbacks: List[Tuple[Callable[[float], None], str]] = []
        self._metrics_enabled = metrics_enabled
        self._metrics = EventBusMetrics(window_size=metrics_window_size) if metrics_enabled else None
        self._metrics_thread = None
        self._metrics_interval = metrics_interval
        self._dropped_events_count = 0
        self._lifecycle_callbacks: Dict[str, List[Callable]] = {
            "startup": [],
            "shutdown": [],
            "error": []
        }
        self._startup_complete = threading.Event()
        self._shutdown_complete = threading.Event()

        # Choose serialization format
        self._serialization_format = self._select_serialization_format(serialization_format)

        # Topic match cache
        self._match_cache_size = 1000  # Size of LRU cache for topic matching

        # Create persistence directory if needed
        if self._persistence_dir:
            self._persistence_dir.mkdir(parents=True, exist_ok=True)

    def _select_serialization_format(self, format_name: str) -> str:
        """Select the best serialization format based on availability and preference"""
        if format_name == 'auto':
            if MSGPACK_AVAILABLE:
                chosen_format = 'msgpack'
            elif UJSON_AVAILABLE:
                chosen_format = 'ujson'
            else:
                chosen_format = 'pickle'
        elif format_name == 'msgpack' and not MSGPACK_AVAILABLE:
            self._logger.warning("msgpack requested but not available, falling back to pickle")
            chosen_format = 'pickle'
        elif format_name == 'ujson' and not UJSON_AVAILABLE:
            self._logger.warning("ujson requested but not available, falling back to pickle")
            chosen_format = 'pickle'
        else:
            chosen_format = format_name

        self._logger.info(f"Using {chosen_format} for event serialization")
        return chosen_format

    def start(self, num_workers: int = 5):
        """
        Start the event processing threads.

        Args:
            num_workers: Number of worker threads per priority level
        """
        with self._lock:
            if self._running:
                return

            self._running = True
            self._startup_complete.clear()
            self._shutdown_complete.clear()

            # Create worker threads for each priority level
            for priority in EventPriority:
                for i in range(num_workers):
                    processor = threading.Thread(
                        target=self._process_events,
                        args=(priority, f"worker-{priority.name}-{i}"),
                        daemon=True,
                        name=f"EventBus-{priority.name}-{i}"
                    )
                    processor.start()
                    self._processors.append(processor)

            # Start metrics reporting thread if enabled
            if self._metrics_enabled:
                self._metrics_thread = threading.Thread(
                    target=self._report_metrics,
                    daemon=True,
                    name="EventBus-Metrics"
                )
                self._metrics_thread.start()

            self._logger.info(f"Event bus started with {num_workers} workers per priority level")

            # Create and publish startup event
            startup_event = create_system_event(
                EventTopics.SYSTEM_START,
                {"workers_per_priority": num_workers}
            )

            # Signal startup complete
            self._startup_complete.set()

            # Notify lifecycle callbacks
            self._logger.debug(f"Executing {len(self._lifecycle_callbacks['startup'])} startup callbacks...")
            for callback_tuple in self._lifecycle_callbacks["startup"]:  # Iterate over tuples
                try:
                    # Unpack the tuple
                    cb_func, cb_id = callback_tuple
                    self._logger.debug(f"Executing startup lifecycle callback with ID: {cb_id}")
                    cb_func()  # Call the actual function (the first element)
                    self._logger.debug(f"Finished startup lifecycle callback with ID: {cb_id}")
                except Exception as e:
                    # Log the specific callback ID that failed
                    cb_id_str = callback_tuple[1] if isinstance(callback_tuple, tuple) and len(
                        callback_tuple) > 1 else "unknown"
                    self._logger.error(f"Error in startup callback (ID: {cb_id_str}): {e}",
                                       exc_info=True)  # Log traceback

            # Publish the startup event
            self.publish_sync(startup_event)

    def stop(self, timeout: float = 5.0):
        """
        Stop the event processing threads.

        Args:
            timeout: Timeout for joining threads in seconds
        """
        with self._lock:
            if not self._running:
                return

            self._running = False
            self._shutdown_complete.clear()

            # Create and publish shutdown event
            shutdown_event = create_system_event(
                EventTopics.SYSTEM_SHUTDOWN,
                {"shutdown_time": time.time()}
            )
            self.publish_sync(shutdown_event)

            # Notify lifecycle callbacks
            self._logger.debug(f"Executing {len(self._lifecycle_callbacks['shutdown'])} shutdown callbacks...")
            for callback_tuple in self._lifecycle_callbacks["shutdown"]:  # Iterate over tuples
                try:
                    # Unpack the tuple
                    cb_func, cb_id = callback_tuple
                    self._logger.debug(f"Executing shutdown lifecycle callback with ID: {cb_id}")
                    cb_func()  # Call the actual function (the first element)
                    self._logger.debug(f"Finished shutdown lifecycle callback with ID: {cb_id}")
                except Exception as e:
                    cb_id_str = callback_tuple[1] if isinstance(callback_tuple, tuple) and len(
                        callback_tuple) > 1 else "unknown"
                    self._logger.error(f"Error in shutdown callback (ID: {cb_id_str}): {e}",
                                       exc_info=True)  # Log traceback

            # Add sentinel events to unblock threads
            for priority in EventPriority:
                for _ in range(len(self._processors)):
                    try:
                        self._priority_queues[priority].put((0, None), block=False)
                    except queue.Full:
                        # Queue is full, we'll timeout anyway
                        pass

            # Join all threads with timeout
            start_time = time.time()
            for thread in self._processors:
                remaining = timeout - (time.time() - start_time)
                if remaining > 0:
                    thread.join(remaining)

            # Try to join metrics thread if it exists
            if self._metrics_thread and self._metrics_thread.is_alive():
                remaining = timeout - (time.time() - start_time)
                if remaining > 0:
                    self._metrics_thread.join(remaining)

            self._processors = []
            self._metrics_thread = None
            self._logger.info("Event bus stopped")

            # Signal shutdown complete
            self._shutdown_complete.set()

    def wait_for_startup(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the event bus to complete startup.

        Args:
            timeout: Maximum time to wait in seconds, or None to wait indefinitely

        Returns:
            True if startup completed, False if timed out
        """
        return self._startup_complete.wait(timeout)

    def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the event bus to complete shutdown.

        Args:
            timeout: Maximum time to wait in seconds, or None to wait indefinitely

        Returns:
            True if shutdown completed, False if timed out
        """
        return self._shutdown_complete.wait(timeout)

    def register_lifecycle_callback(self, event_type: str, callback: Callable[[], None]) -> str:
        """
        Register a callback for lifecycle events.

        Args:
            event_type: The lifecycle event ("startup", "shutdown", "error")
            callback: Function to call with no arguments

        Returns:
            Callback ID for unregistration
        """
        if event_type not in self._lifecycle_callbacks:
            raise ValueError(f"Invalid lifecycle event type: {event_type}")

        callback_id = str(uuid.uuid4())

        with self._lock:
            self._lifecycle_callbacks[event_type].append((callback, callback_id))

        return callback_id

    def unregister_lifecycle_callback(self, event_type: str, callback_id: str) -> bool:
        """
        Unregister a lifecycle callback.

        Args:
            event_type: The lifecycle event type
            callback_id: The callback ID to unregister

        Returns:
            True if successfully unregistered, False otherwise
        """
        if event_type not in self._lifecycle_callbacks:
            return False

        with self._lock:
            original_count = len(self._lifecycle_callbacks[event_type])

            self._lifecycle_callbacks[event_type] = [
                (callback, cb_id) for callback, cb_id in self._lifecycle_callbacks[event_type]
                if cb_id != callback_id
            ]

            return len(self._lifecycle_callbacks[event_type]) < original_count

    def _process_events(self, priority: EventPriority, worker_id: str):
        """
        Process events from a specific priority queue.

        Args:
            priority: The priority queue to process
            worker_id: Identifier for this worker thread
        """
        queue_obj = self._priority_queues[priority]

        while self._running:
            try:
                # Get event from queue with timeout to allow checking _running
                try:
                    queue_time, event = queue_obj.get(timeout=0.1)
                except queue.Empty:
                    continue

                # None is a sentinel value to exit the thread
                if event is None:
                    break

                # Record queue latency for metrics
                if self._metrics_enabled:
                    queue_latency = time.time() - queue_time
                    with self._stats_lock:
                        self._metrics.record_queue_latency(priority, queue_latency)

                # Process the event
                self._deliver_event(event)

                # Mark task as done
                queue_obj.task_done()

            except Exception as e:
                self._logger.error(f"Error in event processor for {priority.name}: {e}")

    def _deliver_event(self, event: Event):
        """
        Deliver an event to all subscribers.

        Args:
            event: The event to deliver
        """
        # Check if we have subscribers for this topic
        direct_subscribers = []
        pattern_subscribers = []

        with self._lock:
            # Check for direct topic matches
            if event.topic in self._subscribers:
                direct_subscribers = self._subscribers[event.topic].copy()

        # Check for pattern subscribers
        with self._pattern_lock:
            for sub in self._pattern_subscribers:
                if sub.pattern.match(event.topic):
                    pattern_subscribers.append((sub.callback, sub.handle_errors, sub.subscriber_id))

        # If no subscribers, log and return
        if not direct_subscribers and not pattern_subscribers:
            self._logger.debug(f"No subscribers for event: {event.topic}")
            return

        # Track timing statistics
        start_time = time.perf_counter()
        delivery_count = 0

        # Deliver to direct subscribers
        for callback, handle_errors, subscriber_id in direct_subscribers:
            try:
                # Record delivery attempt for metrics
                if self._metrics_enabled:
                    event.record_delivery_attempt(subscriber_id)

                # Call the subscriber
                callback(event)
                delivery_count += 1

                # Record delivery completion for metrics
                if self._metrics_enabled:
                    event.record_delivery_complete(subscriber_id)

            except Exception as e:
                self._logger.error(f"Error in subscriber callback for {event.topic}: {e}")
                self._handle_subscriber_error(event.topic, e, event)

                # If the subscriber wants to handle errors, send an error event
                if handle_errors:
                    error_event = Event(
                        topic=f"{event.topic}.error",
                        data={
                            "error": str(e),
                            "original_event": event.to_dict(),
                            "error_type": type(e).__name__
                        },
                        source="event_bus",
                        priority=EventPriority.HIGH
                    )
                    self.publish(error_event)

                # Record error in metrics
                if self._metrics_enabled:
                    with self._stats_lock:
                        self._metrics.record_error(event.topic)

        # Deliver to pattern subscribers
        for callback, handle_errors, subscriber_id in pattern_subscribers:
            try:
                # Record delivery attempt for metrics
                if self._metrics_enabled:
                    event.record_delivery_attempt(subscriber_id)

                # Call the subscriber
                callback(event)
                delivery_count += 1

                # Record delivery completion for metrics
                if self._metrics_enabled:
                    event.record_delivery_complete(subscriber_id)

            except Exception as e:
                self._logger.error(f"Error in pattern subscriber callback for {event.topic}: {e}")
                self._handle_subscriber_error(event.topic, e, event)

                # If the subscriber wants to handle errors, send an error event
                if handle_errors:
                    error_event = Event(
                        topic=f"{event.topic}.error",
                        data={
                            "error": str(e),
                            "original_event": event.to_dict(),
                            "error_type": type(e).__name__
                        },
                        source="event_bus",
                        priority=EventPriority.HIGH
                    )
                    self.publish(error_event)

                # Record error in metrics
                if self._metrics_enabled:
                    with self._stats_lock:
                        self._metrics.record_error(event.topic)

        # Update timing statistics
        elapsed = time.perf_counter() - start_time
        if delivery_count > 0 and self._metrics_enabled:
            with self._stats_lock:
                self._metrics.record_delivery(event.topic, elapsed / delivery_count)

    def _handle_subscriber_error(self, topic: str, exception: Exception, original_event: Event):
        """
        Handle an error in a subscriber callback.

        Args:
            topic: The topic of the event
            exception: The exception that occurred
            original_event: The original event
        """
        # Call any registered error handlers for this topic
        handlers = []

        with self._lock:
            # Direct topic handlers
            if topic in self._error_handlers:
                handlers.extend(self._error_handlers[topic])

            # Check for wildcard error handlers
            for error_topic, topic_handlers in self._error_handlers.items():
                if self._topic_matches(topic, error_topic) and error_topic != topic:
                    handlers.extend(topic_handlers)

        for handler, _ in handlers:
            try:
                handler(topic, exception, original_event)
            except Exception as e:
                self._logger.error(f"Error in error handler for {topic}: {e}")

    def _handle_backpressure(self, pressure_level: float):
        """
        Handle backpressure by notifying registered callbacks.

        Args:
            pressure_level: The current pressure level (0.0-1.0)
        """
        for callback, _ in self._backpressure_callbacks:
            try:
                callback(pressure_level)
            except Exception as e:
                self._logger.error(f"Error in backpressure callback: {e}")

    def _persist_event(self, event: Event):
        """
        Persist an event to disk.

        Args:
            event: The event to persist
        """
        if not self._persistence_dir:
            return

        topic_dir = self._persistence_dir / event.topic
        topic_dir.mkdir(parents=True, exist_ok=True)

        # Use timestamp and event_id to create a unique filename
        timestamp_str = f"{event.timestamp:.6f}".replace('.', '_')
        filename = f"{timestamp_str}_{event.event_id}.event"
        filepath = topic_dir / filename

        try:
            with filepath.open('wb') as f:
                if self._serialization_format == 'msgpack' and MSGPACK_AVAILABLE:
                    # Convert event to dictionary first
                    event_dict = event.to_dict()
                    msgpack.dump(event_dict, f)
                elif self._serialization_format == 'ujson' and UJSON_AVAILABLE:
                    # Convert to JSON-compatible dictionary
                    event_dict = event.to_dict()
                    # Convert non-JSON-serializable objects
                    serialized = json.dumps(event_dict, default=lambda o: str(o))
                    f.write(serialized.encode('utf-8'))
                else:
                    # Default to pickle
                    pickle.dump(event, f)
        except Exception as e:
            self._logger.error(f"Error persisting event {event.topic}: {e}")

    def _load_event(self, filepath: Path) -> Event:
        """
        Load an event from disk.

        Args:
            filepath: Path to the event file

        Returns:
            The loaded event
        """
        with filepath.open('rb') as f:
            try:
                if self._serialization_format == 'msgpack' and MSGPACK_AVAILABLE:
                    event_dict = msgpack.load(f)
                    return Event.from_dict(event_dict)
                elif self._serialization_format == 'ujson' and UJSON_AVAILABLE:
                    event_dict = json.loads(f.read().decode('utf-8'))
                    return Event.from_dict(event_dict)
                else:
                    # Default to pickle
                    return pickle.load(f)
            except Exception as e:
                self._logger.error(f"Error loading event from {filepath}: {e}")
                raise

    def subscribe(self, topic: str, callback: Callable[[Event], None], handle_errors: bool = False,
                  subscriber_id: Optional[str] = None) -> str:
        """
        Subscribe to a topic.

        Args:
            topic: The topic to subscribe to
            callback: The callback function to be called when an event is published to the topic
            handle_errors: Whether the callback should receive error events related to the topic
            subscriber_id: Optional identifier for the subscriber (for metrics)

        Returns:
            Subscription ID for unsubscribing
        """
        if subscriber_id is None:
            subscriber_id = f"sub_{uuid.uuid4().hex[:8]}"

        with self._lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = []

            self._subscribers[topic].append((callback, handle_errors, subscriber_id))

            # Update metrics
            if self._metrics_enabled:
                with self._stats_lock:
                    count = len(self._subscribers[topic])
                    self._metrics.update_subscriber_count(topic, count)

            self._logger.debug(f"Subscribed to topic: {topic} (id: {subscriber_id})")

        return subscriber_id

    def subscribe_pattern(self, pattern: str, callback: Callable[[Event], None], handle_errors: bool = False,
                          subscriber_id: Optional[str] = None) -> str:
        """
        Subscribe to topics matching a pattern.

        Args:
            pattern: The pattern to match topics against (e.g., 'market.*', '*.tick', '*')
            callback: The callback function to be called when an event is published to matching topics
            handle_errors: Whether the callback should receive error events
            subscriber_id: Optional identifier for the subscriber (for metrics)

        Returns:
            Subscription ID for unsubscribing
        """
        if subscriber_id is None:
            subscriber_id = f"pattern_sub_{uuid.uuid4().hex[:8]}"

        with self._pattern_lock:
            subscription = PatternSubscription(pattern, callback, handle_errors, subscriber_id)
            self._pattern_subscribers.append(subscription)
            self._logger.debug(f"Subscribed to pattern: {pattern} (id: {subscriber_id})")

        return subscriber_id

    def unsubscribe(self, topic: str, callback: Optional[Callable] = None, subscriber_id: Optional[str] = None) -> bool:
        """
        Unsubscribe from a topic.

        Args:
            topic: The topic to unsubscribe from
            callback: The callback function to unsubscribe (if None, use subscriber_id)
            subscriber_id: The subscriber ID to unsubscribe (if None, use callback)

        Returns:
            True if successfully unsubscribed, False otherwise
        """
        if callback is None and subscriber_id is None:
            self._logger.warning("Must provide either callback or subscriber_id to unsubscribe")
            return False

        with self._lock:
            if topic in self._subscribers:
                original_count = len(self._subscribers[topic])

                if callback is not None:
                    self._subscribers[topic] = [
                        (cb, handle_errs, sub_id) for cb, handle_errs, sub_id in self._subscribers[topic]
                        if cb != callback
                    ]
                else:  # Use subscriber_id
                    self._subscribers[topic] = [
                        (cb, handle_errs, sub_id) for cb, handle_errs, sub_id in self._subscribers[topic]
                        if sub_id != subscriber_id
                    ]

                if not self._subscribers[topic]:
                    del self._subscribers[topic]

                # Update metrics
                if self._metrics_enabled and topic in self._subscribers:
                    with self._stats_lock:
                        count = len(self._subscribers[topic])
                        self._metrics.update_subscriber_count(topic, count)

                # Return success if we actually removed something
                return len(self._subscribers.get(topic, [])) < original_count

        return False

    def unsubscribe_pattern(self, pattern: Optional[str] = None, callback: Optional[Callable] = None,
                            subscriber_id: Optional[str] = None) -> bool:
        """
        Unsubscribe from a pattern.

        Args:
            pattern: The pattern to unsubscribe from (if None, use callback or subscriber_id)
            callback: The callback function to unsubscribe (if None, use pattern or subscriber_id)
            subscriber_id: The subscriber ID to unsubscribe (if None, use pattern or callback)

        Returns:
            True if successfully unsubscribed, False otherwise
        """
        if pattern is None and callback is None and subscriber_id is None:
            self._logger.warning("Must provide at least one of pattern, callback, or subscriber_id to unsubscribe")
            return False

        with self._pattern_lock:
            original_count = len(self._pattern_subscribers)

            if pattern is not None and callback is not None:
                self._pattern_subscribers = [
                    sub for sub in self._pattern_subscribers
                    if not (sub.pattern_str == pattern and sub.callback == callback)
                ]
            elif pattern is not None:
                self._pattern_subscribers = [
                    sub for sub in self._pattern_subscribers
                    if sub.pattern_str != pattern
                ]
            elif callback is not None:
                self._pattern_subscribers = [
                    sub for sub in self._pattern_subscribers
                    if sub.callback != callback
                ]
            elif subscriber_id is not None:
                self._pattern_subscribers = [
                    sub for sub in self._pattern_subscribers
                    if sub.subscriber_id != subscriber_id
                ]

            return len(self._pattern_subscribers) < original_count

    def register_error_handler(self, topic: str, handler: Callable[[str, Exception, Event], None]) -> str:
        """
        Register an error handler for a topic.

        Args:
            topic: The topic to handle errors for
            handler: The handler function that takes (topic, exception, original_event)

        Returns:
            Handler ID for unregistration
        """
        handler_id = str(uuid.uuid4())

        with self._lock:
            if topic not in self._error_handlers:
                self._error_handlers[topic] = []

            self._error_handlers[topic].append((handler, handler_id))
            self._logger.debug(f"Registered error handler for topic: {topic} (ID: {handler_id})")

        return handler_id

    def unregister_error_handler(self, topic: str, handler_id: str) -> bool:
        """
        Unregister an error handler.

        Args:
            topic: The topic the handler is registered for
            handler_id: The handler ID to unregister

        Returns:
            True if successfully unregistered, False otherwise
        """
        with self._lock:
            if topic in self._error_handlers:
                original_count = len(self._error_handlers[topic])

                self._error_handlers[topic] = [
                    (handler, h_id) for handler, h_id in self._error_handlers[topic]
                    if h_id != handler_id
                ]

                if not self._error_handlers[topic]:
                    del self._error_handlers[topic]

                return len(self._error_handlers.get(topic, [])) < original_count

        return False

    def register_backpressure_callback(self, callback: Callable[[float], None]) -> str:
        """
        Register a callback that will be called when backpressure is applied.

        Args:
            callback: Function to call with the current pressure level (0.0-1.0)

        Returns:
            Callback ID for unregistration
        """
        callback_id = str(uuid.uuid4())

        with self._lock:
            self._backpressure_callbacks.append((callback, callback_id))
            self._logger.debug(f"Registered backpressure callback (ID: {callback_id})")

        return callback_id

    def unregister_backpressure_callback(self, callback_id: str) -> bool:
        """
        Unregister a backpressure callback.

        Args:
            callback_id: The callback ID to unregister

        Returns:
            True if successfully unregistered, False otherwise
        """
        with self._lock:
            original_count = len(self._backpressure_callbacks)

            self._backpressure_callbacks = [
                (callback, cb_id) for callback, cb_id in self._backpressure_callbacks
                if cb_id != callback_id
            ]

            return len(self._backpressure_callbacks) < original_count

    def enable_persistence(self, topic: str):
        """
        Enable persistence for a topic.

        Args:
            topic: The topic to enable persistence for
        """
        if not self._persistence_dir:
            self._logger.warning(f"Cannot enable persistence for topic {topic} - no persistence directory set")
            return

        with self._lock:
            self._persistent_topics.add(topic)
            topic_dir = self._persistence_dir / topic
            topic_dir.mkdir(parents=True, exist_ok=True)
            self._logger.debug(f"Enabled persistence for topic: {topic}")

    def disable_persistence(self, topic: str):
        """
        Disable persistence for a topic.

        Args:
            topic: The topic to disable persistence for
        """
        with self._lock:
            if topic in self._persistent_topics:
                self._persistent_topics.remove(topic)
                self._logger.debug(f"Disabled persistence for topic: {topic}")

    def set_serialization_format(self, format_name: str):
        """
        Set the serialization format for persistence.

        Args:
            format_name: The format to use ('pickle', 'msgpack', 'ujson')
        """
        if format_name not in ['pickle', 'msgpack', 'ujson']:
            raise ValueError(f"Unsupported serialization format: {format_name}")

        if format_name == 'msgpack' and not MSGPACK_AVAILABLE:
            self._logger.warning("msgpack requested but not available, falling back to pickle")
            format_name = 'pickle'

        if format_name == 'ujson' and not UJSON_AVAILABLE:
            self._logger.warning("ujson requested but not available, falling back to pickle")
            format_name = 'pickle'

        self._serialization_format = format_name
        self._logger.info(f"Set serialization format to: {format_name}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get event statistics.

        Returns:
            Dictionary of event statistics
        """
        if not self._metrics_enabled:
            return {
                "metrics_enabled": False,
                "dropped_events_count": self._dropped_events_count,
                "persistent_topics": list(self._persistent_topics),
                "subscriber_topics": list(self._subscribers.keys()),
                "pattern_subscribers": len(self._pattern_subscribers)
            }

        with self._stats_lock:
            metrics = self._metrics.get_metrics()

        metrics.update({
            "metrics_enabled": True,
            "dropped_events_count": self._dropped_events_count,
            "persistent_topics": list(self._persistent_topics),
            "subscriber_topics": list(self._subscribers.keys()),
            "pattern_subscribers": len(self._pattern_subscribers)
        })

        return metrics

    def reset_stats(self):
        """Reset all statistics"""
        if self._metrics_enabled:
            with self._stats_lock:
                self._metrics.reset()

        self._dropped_events_count = 0
        self._logger.debug("Event bus statistics reset")

    def export_state(self) -> Dict[str, Any]:
        """
        Export the event bus state for disaster recovery.

        Returns:
            A dictionary containing the event bus state
        """
        with self._lock:
            state = {
                "persistent_topics": list(self._persistent_topics),
                "subscriber_topics": list(self._subscribers.keys()),
                "pattern_subscribers": [sub.pattern_str for sub in self._pattern_subscribers],
                "serialization_format": self._serialization_format,
                "metrics": self.get_stats() if self._metrics_enabled else None,
                "timestamp": time.time()
            }
            return state

    def import_state(self, state: Dict[str, Any]):
        """
        Import the event bus state for disaster recovery.

        Args:
            state: The state dictionary
        """
        with self._lock:
            self._persistent_topics = set(state.get("persistent_topics", []))

            if "serialization_format" in state:
                self.set_serialization_format(state["serialization_format"])

            self._logger.info(f"Imported event bus state with {len(self._persistent_topics)} persistent topics")

    def replay_persisted_events(self, topics: Optional[List[str]] = None,
                                since: Optional[float] = None,
                                until: Optional[float] = None,
                                batch_size: int = 100) -> int:
        """
        Replay persisted events, optionally filtered by topic and time.

        Args:
            topics: List of topics to replay, if None, replay all
            since: Timestamp to replay events from, if None, replay all
            until: Timestamp to replay events until, if None, replay all
            batch_size: Number of events to process in a batch

        Returns:
            Number of events replayed
        """
        if not self._persistence_dir:
            self._logger.warning("No persistence directory set, cannot replay events")
            return 0

        count = 0

        # Find all topic directories
        topic_dirs = [d for d in self._persistence_dir.iterdir() if d.is_dir()]

        for topic_dir in topic_dirs:
            topic_name = topic_dir.name

            if topics and topic_name not in topics:
                continue

            # Get all event files sorted by name (which includes timestamp)
            event_files = sorted([f for f in topic_dir.iterdir() if f.name.endswith('.event')])

            # Process in batches
            batch = []
            for filepath in event_files:
                try:
                    event = self._load_event(filepath)

                    # Apply time filters
                    if since and event.timestamp < since:
                        continue
                    if until and event.timestamp > until:
                        continue

                    batch.append(event)

                    # Process batch when it reaches the size limit
                    if len(batch) >= batch_size:
                        self.publish_batch(batch)
                        count += len(batch)
                        batch = []

                except Exception as e:
                    self._logger.error(f"Error loading event from {filepath}: {e}")

            # Process any remaining events in the last batch
            if batch:
                self.publish_batch(batch)
                count += len(batch)

        self._logger.info(f"Replayed {count} persisted events")
        return count

    def _report_metrics(self):
        """Periodically report metrics to the logger"""
        while self._running:
            time.sleep(self._metrics_interval)

            try:
                if not self._metrics_enabled:
                    continue

                with self._stats_lock:
                    metrics = self._metrics.get_metrics()

                # Log summary metrics
                self._logger.info(
                    f"Event bus metrics: "
                    f"{sum(metrics['published_events'].values())} published, "
                    f"{sum(metrics['delivered_events'].values())} delivered, "
                    f"{sum(metrics['dropped_events'].values())} dropped, "
                    f"{sum(metrics['error_count'].values())} errors"
                )

                # Log detailed metrics at debug level
                self._logger.debug(f"Detailed event bus metrics: {metrics}")

                # Create and publish metrics event
                metrics_event = Event(
                    topic=EventTopics.METRICS_UPDATE,
                    data=metrics,
                    source="event_bus",
                    priority=EventPriority.LOW
                )

                # Publish metrics event with a short timeout
                self.publish(metrics_event, timeout=0.1)

            except Exception as e:
                self._logger.error(f"Error reporting metrics: {e}")

    @lru_cache(maxsize=1000)
    def _topic_matches(self, event_topic: str, subscription_topic: str) -> bool:
        """
        Check if an event topic matches a subscription topic pattern.
        Results are cached for performance.

        Args:
            event_topic: The topic of the event
            subscription_topic: The subscription topic pattern

        Returns:
            True if the event topic matches the subscription pattern
        """
        # Direct match
        if subscription_topic == event_topic:
            return True

        # Wildcard match
        if subscription_topic == "*":
            return True

        # Prefix wildcard match (e.g., "market.*" matches "market.tick")
        if subscription_topic.endswith(".*"):
            prefix = subscription_topic[:-2]
            return event_topic.startswith(prefix) and "." in event_topic[len(prefix):]

        # Suffix wildcard match (e.g., "*.tick" matches "market.tick")
        if subscription_topic.startswith("*."):
            suffix = subscription_topic[1:]
            return event_topic.endswith(suffix) and "." in event_topic[:-len(suffix)]

        return False

    def prune_persisted_events(self, older_than: float, topics: Optional[List[str]] = None) -> int:
        """
        Delete old persisted events.

        Args:
            older_than: Timestamp to delete events older than
            topics: List of topics to prune, if None, prune all

        Returns:
            Number of events pruned
        """
        if not self._persistence_dir:
            self._logger.warning("No persistence directory set, cannot prune events")
            return 0

        count = 0

        # Find all topic directories
        topic_dirs = [d for d in self._persistence_dir.iterdir() if d.is_dir()]

        for topic_dir in topic_dirs:
            topic_name = topic_dir.name

            if topics and topic_name not in topics:
                continue

            # Get all event files
            event_files = list(topic_dir.iterdir())

            for filepath in event_files:
                if not filepath.name.endswith('.event'):
                    continue

                try:
                    # Extract timestamp from filename
                    # Filename format is timestamp_eventid.event
                    # where timestamp has _ instead of .
                    timestamp_str = filepath.stem.split('_')[0]
                    timestamp = float(timestamp_str.replace('_', '.'))

                    if timestamp < older_than:
                        filepath.unlink()
                        count += 1
                except Exception as e:
                    self._logger.error(f"Error pruning event {filepath}: {e}")

        self._logger.info(f"Pruned {count} persisted events")
        return count

    def publish_sync(self, event: Event) -> bool:
        """
        Publish an event and process it synchronously, blocking until all handlers complete.

        Args:
            event: The event to publish

        Returns:
            True if the event was published and delivered, False otherwise
        """
        try:
            # Persist event if necessary
            if event.topic in self._persistent_topics and self._persistence_dir:
                self._persist_event(event)

            # Process the event immediately
            self._deliver_event(event)

            # Record metrics
            if self._metrics_enabled:
                with self._stats_lock:
                    self._metrics.record_publish(event.topic)

            return True

        except Exception as e:
            self._logger.error(f"Error in synchronous publish of event {event.topic}: {e}")
            return False

    def publish_batch(self, events: List[Event], block: bool = True, timeout: Optional[float] = None) -> int:
        """
        Publish multiple events at once.

        Args:
            events: List of events to publish
            block: Whether to block if queues are full
            timeout: Maximum time to block per event (None for indefinite)

        Returns:
            Number of events successfully published
        """
        if not events:
            return 0

        successful = 0
        for event in events:
            if self.publish(event, block, timeout):
                successful += 1

        return successful

    def publish(self, event: Event, block: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Publish an event to subscribers.

        Args:
            event: The event to publish
            block: Whether to block if the queue is full
            timeout: Maximum time to block (None for indefinite)

        Returns:
            True if the event was published, False if it was dropped due to a full queue
        """
        if not self._running:
            self._logger.warning("Attempted to publish event when event bus is not running")
            return False

        start_time = time.time()

        try:
            # Critical events bypass the queue and are processed immediately
            if event.priority == EventPriority.CRITICAL:
                self._deliver_event(event)

                # Record metrics
                if self._metrics_enabled:
                    with self._stats_lock:
                        self._metrics.record_publish(event.topic)
                return True

            # Persist event if necessary
            if event.topic in self._persistent_topics and self._persistence_dir:
                self._persist_event(event)

            # Use a more robust approach to queue insertion with backpressure
            queue_obj = self._priority_queues[event.priority]
            try:
                # Record queue size for metrics
                if self._metrics_enabled:
                    with self._stats_lock:
                        self._metrics.update_queue_size(event.priority, queue_obj.qsize())

                # Add to appropriate priority queue with timestamp for FIFO within same priority
                queue_obj.put((time.time(), event), block=block, timeout=timeout)

                # Record metrics
                if self._metrics_enabled:
                    with self._stats_lock:
                        self._metrics.record_publish(event.topic)

                self._logger.debug(f"Published event: {event.topic} (priority: {event.priority.name})")
                return True

            except queue.Full:
                self._dropped_events_count += 1

                # Apply backpressure
                pressure_level = 1.0  # Maximum pressure
                self._handle_backpressure(pressure_level)

                # Record dropped event in metrics
                if self._metrics_enabled:
                    with self._stats_lock:
                        self._metrics.record_drop(event.topic)

                self._logger.warning(f"Dropped event {event.topic} due to full queue (priority: {event.priority.name})")
                return False

        except Exception as e:
            self._logger.error(f"Error publishing event {event.topic}: {e}")

            # Notify error callbacks
            for callback, _ in self._lifecycle_callbacks.get("error", []):
                try:
                    callback()
                except Exception as e:
                    self._logger.error(f"Error in lifecycle error callback: {e}")

            return False


# Singleton instance
_event_bus_instance: Optional[EventBus] = None


def get_event_bus(
        persistence_dir: Optional[str] = None,
        max_queue_size: int = 10000,
        metrics_enabled: bool = True
) -> EventBus:
    """
    Get the singleton EventBus instance.

    Args:
        persistence_dir: Directory for persisting events (only used if instance doesn't exist yet)
        max_queue_size: Maximum size for priority queues
        metrics_enabled: Whether to enable metrics collection

    Returns:
        The singleton EventBus instance
    """
    global _event_bus_instance
    if _event_bus_instance is None:
        _event_bus_instance = EventBus(
            persistence_dir=persistence_dir,
            max_queue_size=max_queue_size,
            metrics_enabled=metrics_enabled
        )
    return _event_bus_instance


# Event creation helpers
def create_event(
        topic: str,
        data: Any,
        priority: EventPriority = EventPriority.NORMAL,
        source: Optional[str] = None,
        **metadata
) -> Event:
    """
    Helper function to create a new event.

    Args:
        topic: The topic of the event
        data: The event data
        priority: The event priority
        source: The source of the event
        metadata: Additional metadata for the event

    Returns:
        The created event
    """
    return Event(
        topic=topic,
        data=data,
        priority=priority,
        source=source,
        metadata=metadata
    )


def create_market_data_event(symbol: str, data: Any, source: str, **metadata) -> Event:
    """
    Helper function to create a market data event.

    Args:
        symbol: The trading symbol
        data: The market data
        source: The source of the data
        metadata: Additional metadata for the event

    Returns:
        The created market data event
    """
    return Event(
        topic=f"{EventTopics.MARKET_DATA}.{symbol}",
        data=data,
        priority=EventPriority.HIGH,  # Market data usually needs quick processing
        source=source,
        metadata=metadata
    )


def create_order_event(order_id: str, event_type: str, data: Any, **metadata) -> Event:
    """
    Helper function to create an order-related event.

    Args:
        order_id: The ID of the order
        event_type: The type of order event (from EventTopics)
        data: The event data
        metadata: Additional metadata for the event

    Returns:
        The created order event
    """
    return Event(
        topic=f"{event_type}.{order_id}",
        data=data,
        priority=EventPriority.HIGH,
        source="execution_service",
        metadata=metadata
    )


def create_system_event(event_type: str, data: Any, **metadata) -> Event:
    """
    Helper function to create a system event.

    Args:
        event_type: The type of system event (from EventTopics)
        data: The event data
        metadata: Additional metadata for the event

    Returns:
        The created system event
    """
    return Event(
        topic=event_type,
        data=data,
        priority=EventPriority.HIGH,
        source="system",
        metadata=metadata
    )


def create_strategy_event(strategy_id: str, event_type: str, data: Any, **metadata) -> Event:
    """
    Helper function to create a strategy-related event.

    Args:
        strategy_id: The ID of the strategy
        event_type: The type of strategy event (from EventTopics)
        data: The event data
        metadata: Additional metadata for the event

    Returns:
        The created strategy event
    """
    return Event(
        topic=f"{event_type}.{strategy_id}",
        data=data,
        priority=EventPriority.NORMAL,
        source=f"strategy.{strategy_id}",
        metadata=metadata
    )


def create_regime_event(regime_type: str, data: Any, **metadata) -> Event:
    """
    Helper function to create a regime-related event.

    Args:
        regime_type: The type of regime event (from EventTopics)
        data: The event data
        metadata: Additional metadata for the event

    Returns:
        The created regime event
    """
    return Event(
        topic=regime_type,
        data=data,
        priority=EventPriority.HIGH,
        source="regime_classifier",
        metadata=metadata
    )


def create_signal_event(strategy_id: str, data: Any, **metadata) -> Event:
    """
    Helper function to create a signal event.

    Args:
        strategy_id: The ID of the strategy
        data: The signal data
        metadata: Additional metadata for the event

    Returns:
        The created signal event
    """
    return Event(
        topic=f"{EventTopics.SIGNAL_GENERATED}.{strategy_id}",
        data=data,
        priority=EventPriority.HIGH,
        source=f"strategy.{strategy_id}",
        metadata=metadata
    )