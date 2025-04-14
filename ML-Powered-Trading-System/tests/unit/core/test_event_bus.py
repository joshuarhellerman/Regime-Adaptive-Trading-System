import pytest
import time
import threading
import queue
import tempfile
import os
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch
import uuid

import sys
sys.path.append('.')  # Adjust as needed for your project structure

from event_bus import (
    EventBus, Event, EventPriority, EventTopics, PatternSubscription,
    create_event, create_market_data_event, create_order_event,
    create_system_event, create_strategy_event, create_regime_event,
    create_signal_event, get_event_bus
)

# Set up logging for tests
logging.basicConfig(level=logging.DEBUG)


class TestEvent:
    """Tests for the Event data structure"""

    def test_event_creation(self):
        """Test basic event creation"""
        event = Event(
            topic="test.topic",
            data={"key": "value"},
            source="test_source",
            priority=EventPriority.HIGH
        )

        assert event.topic == "test.topic"
        assert event.data == {"key": "value"}
        assert event.source == "test_source"
        assert event.priority == EventPriority.HIGH
        assert isinstance(event.event_id, str)
        assert isinstance(event.timestamp, float)
        assert isinstance(event.metadata, dict)

    def test_event_to_dict(self):
        """Test conversion of event to dictionary"""
        event = Event(
            topic="test.topic",
            data={"key": "value"},
            event_id="test-id",
            timestamp=1234567890.0,
            source="test_source",
            priority=EventPriority.HIGH,
            metadata={"meta": "data"}
        )

        event_dict = event.to_dict()
        assert event_dict["topic"] == "test.topic"
        assert event_dict["data"] == {"key": "value"}
        assert event_dict["event_id"] == "test-id"
        assert event_dict["timestamp"] == 1234567890.0
        assert event_dict["source"] == "test_source"
        assert event_dict["priority"] == "HIGH"
        assert event_dict["metadata"] == {"meta": "data"}

    def test_event_from_dict(self):
        """Test creation of event from dictionary"""
        event_dict = {
            "topic": "test.topic",
            "data": {"key": "value"},
            "event_id": "test-id",
            "timestamp": 1234567890.0,
            "source": "test_source",
            "priority": "HIGH",
            "metadata": {"meta": "data"}
        }

        event = Event.from_dict(event_dict)
        assert event.topic == "test.topic"
        assert event.data == {"key": "value"}
        assert event.event_id == "test-id"
        assert event.timestamp == 1234567890.0
        assert event.source == "test_source"
        assert event.priority == EventPriority.HIGH
        assert event.metadata == {"meta": "data"}

    def test_delivery_metrics(self):
        """Test recording and retrieving delivery metrics"""
        event = Event(
            topic="test.topic",
            data={"key": "value"}
        )
        
        subscriber_id = "test_subscriber"
        event.record_delivery_attempt(subscriber_id)
        
        metrics = event.get_delivery_metrics()
        assert "created_at" in metrics
        assert "age" in metrics
        assert "deliveries" in metrics
        assert subscriber_id in metrics["deliveries"]
        assert "delivery_time" in metrics["deliveries"][subscriber_id]
        assert "queue_time" in metrics["deliveries"][subscriber_id]
        
        # Test recording delivery completion
        event.record_delivery_complete(subscriber_id)
        metrics = event.get_delivery_metrics()
        assert "processing_time" in metrics["deliveries"][subscriber_id]


class TestPatternSubscription:
    """Tests for PatternSubscription class"""

    def test_exact_pattern_match(self):
        """Test exact pattern matching"""
        callback = MagicMock()
        subscription = PatternSubscription("test.topic", callback)
        
        assert subscription.pattern.match("test.topic")
        assert not subscription.pattern.match("other.topic")
        
    def test_wildcard_pattern_match(self):
        """Test wildcard pattern matching"""
        callback = MagicMock()
        all_subscription = PatternSubscription("*", callback)
        prefix_subscription = PatternSubscription("market.*", callback)
        suffix_subscription = PatternSubscription("*.update", callback)
        
        # Test all wildcard
        assert all_subscription.pattern.match("any.topic")
        assert all_subscription.pattern.match("other.topic")
        
        # Test prefix wildcard
        assert prefix_subscription.pattern.match("market.data")
        assert prefix_subscription.pattern.match("market.price")
        assert not prefix_subscription.pattern.match("other.data")
        
        # Test suffix wildcard
        assert suffix_subscription.pattern.match("price.update")
        assert suffix_subscription.pattern.match("order.update")
        assert not suffix_subscription.pattern.match("price.data")


class TestEventBusBasics:
    """Basic tests for EventBus functionality"""

    @pytest.fixture
    def event_bus(self):
        """Create a fresh event bus for each test"""
        bus = EventBus()
        yield bus
        # Stop the event bus after each test
        if bus._running:
            bus.stop()

    def test_event_bus_creation(self, event_bus):
        """Test event bus initialization"""
        assert not event_bus._running
        assert len(event_bus._subscribers) == 0
        assert len(event_bus._pattern_subscribers) == 0

    def test_start_stop(self, event_bus):
        """Test starting and stopping the event bus"""
        assert not event_bus._running
        
        # Start the event bus
        event_bus.start(num_workers=2)
        assert event_bus._running
        assert len(event_bus._processors) > 0
        
        # Stop the event bus
        event_bus.stop()
        assert not event_bus._running
        assert len(event_bus._processors) == 0

    def test_subscribe_unsubscribe(self, event_bus):
        """Test subscribing and unsubscribing from topics"""
        callback = MagicMock()
        
        # Subscribe
        subscriber_id = event_bus.subscribe("test.topic", callback)
        assert "test.topic" in event_bus._subscribers
        assert len(event_bus._subscribers["test.topic"]) == 1
        
        # Unsubscribe by callback
        success = event_bus.unsubscribe("test.topic", callback=callback)
        assert success
        assert "test.topic" not in event_bus._subscribers
        
        # Subscribe again and unsubscribe by ID
        subscriber_id = event_bus.subscribe("test.topic", callback)
        success = event_bus.unsubscribe("test.topic", subscriber_id=subscriber_id)
        assert success
        assert "test.topic" not in event_bus._subscribers

    def test_subscribe_pattern_unsubscribe_pattern(self, event_bus):
        """Test subscribing and unsubscribing from patterns"""
        callback = MagicMock()
        
        # Subscribe to pattern
        subscriber_id = event_bus.subscribe_pattern("test.*", callback)
        assert len(event_bus._pattern_subscribers) == 1
        
        # Unsubscribe by ID
        success = event_bus.unsubscribe_pattern(subscriber_id=subscriber_id)
        assert success
        assert len(event_bus._pattern_subscribers) == 0
        
        # Subscribe again and unsubscribe by pattern
        event_bus.subscribe_pattern("test.*", callback)
        success = event_bus.unsubscribe_pattern(pattern="test.*")
        assert success
        assert len(event_bus._pattern_subscribers) == 0

    def test_enable_disable_persistence(self, event_bus):
        """Test enabling and disabling persistence for topics"""
        with tempfile.TemporaryDirectory() as tmpdir:
            event_bus._persistence_dir = Path(tmpdir)
            
            # Enable persistence
            event_bus.enable_persistence("test.topic")
            assert "test.topic" in event_bus._persistent_topics
            assert (Path(tmpdir) / "test.topic").exists()
            
            # Disable persistence
            event_bus.disable_persistence("test.topic")
            assert "test.topic" not in event_bus._persistent_topics


class TestEventPublishingAndDelivery:
    """Tests for event publishing and delivery"""

    @pytest.fixture
    def event_bus(self):
        """Create a fresh event bus for each test"""
        bus = EventBus()
        bus.start(num_workers=2)
        yield bus
        # Stop the event bus after each test
        if bus._running:
            bus.stop()

    def test_publish_and_receive(self, event_bus):
        """Test basic publishing and receiving of events"""
        received_events = []
        event_received = threading.Event()
        
        def callback(event):
            received_events.append(event)
            event_received.set()
        
        # Subscribe to the test topic
        event_bus.subscribe("test.topic", callback)
        
        # Create and publish an event
        test_event = Event(topic="test.topic", data={"key": "value"})
        success = event_bus.publish(test_event)
        assert success
        
        # Wait for event to be processed
        assert event_received.wait(timeout=2.0)
        
        # Check received event
        assert len(received_events) == 1
        assert received_events[0].topic == "test.topic"
        assert received_events[0].data == {"key": "value"}

    def test_publish_sync(self, event_bus):
        """Test synchronous publishing"""
        received_events = []
        
        def callback(event):
            received_events.append(event)
        
        # Subscribe to the test topic
        event_bus.subscribe("test.topic", callback)
        
        # Create and publish an event synchronously
        test_event = Event(topic="test.topic", data={"key": "value"})
        success = event_bus.publish_sync(test_event)
        assert success
        
        # Event should be processed immediately
        assert len(received_events) == 1
        assert received_events[0].topic == "test.topic"
        assert received_events[0].data == {"key": "value"}

    def test_publish_batch(self, event_bus):
        """Test batch publishing"""
        received_events = []
        events_received = threading.Event()
        expected_count = 3
        
        def callback(event):
            received_events.append(event)
            if len(received_events) >= expected_count:
                events_received.set()
        
        # Subscribe to the test topics
        event_bus.subscribe("test.topic1", callback)
        event_bus.subscribe("test.topic2", callback)
        event_bus.subscribe("test.topic3", callback)
        
        # Create and publish a batch of events
        events = [
            Event(topic="test.topic1", data={"key": "value1"}),
            Event(topic="test.topic2", data={"key": "value2"}),
            Event(topic="test.topic3", data={"key": "value3"})
        ]
        
        count = event_bus.publish_batch(events)
        assert count == expected_count
        
        # Wait for events to be processed
        assert events_received.wait(timeout=2.0)
        
        # Check received events
        assert len(received_events) == expected_count
        topics = [event.topic for event in received_events]
        assert "test.topic1" in topics
        assert "test.topic2" in topics
        assert "test.topic3" in topics

    def test_pattern_subscription(self, event_bus):
        """Test pattern-based subscriptions"""
        received_events = []
        event_received = threading.Event()
        
        def callback(event):
            received_events.append(event)
            event_received.set()
        
        # Subscribe to a pattern
        event_bus.subscribe_pattern("test.*", callback)
        
        # Create and publish an event matching the pattern
        test_event = Event(topic="test.specific", data={"key": "value"})
        success = event_bus.publish(test_event)
        assert success
        
        # Wait for event to be processed
        assert event_received.wait(timeout=2.0)
        
        # Check received event
        assert len(received_events) == 1
        assert received_events[0].topic == "test.specific"
        assert received_events[0].data == {"key": "value"}

    def test_priority_processing(self, event_bus):
        """Test that high priority events are processed before lower priority ones"""
        received_events = []
        all_received = threading.Event()
        expected_count = 2
        
        def callback(event):
            received_events.append(event)
            if len(received_events) >= expected_count:
                all_received.set()
        
        # Subscribe to both topics
        event_bus.subscribe("test.high", callback)
        event_bus.subscribe("test.low", callback)
        
        # Create a very busy situation by adding many low priority events first
        for _ in range(50):
            low_event = Event(
                topic="test.low", 
                data={"priority": "low"}, 
                priority=EventPriority.LOW
            )
            event_bus.publish(low_event, block=False)
        
        # Then add one high priority event which should be processed first
        high_event = Event(
            topic="test.high", 
            data={"priority": "high"}, 
            priority=EventPriority.HIGH
        )
        event_bus.publish(high_event)
        
        # Wait for at least the high priority and one low priority event
        assert all_received.wait(timeout=2.0)
        
        # The high priority event should be processed first
        assert len(received_events) >= 2
        assert received_events[0].topic == "test.high"
        assert received_events[0].data["priority"] == "high"

    def test_error_handling(self, event_bus):
        """Test error handling in subscribers"""
        error_handled = threading.Event()
        
        def failing_callback(event):
            raise ValueError("Test error")
        
        def error_handler(topic, exception, original_event):
            assert topic == "test.error"
            assert isinstance(exception, ValueError)
            assert exception.args[0] == "Test error"
            assert original_event.topic == "test.error"
            error_handled.set()
        
        # Register callbacks
        event_bus.subscribe("test.error", failing_callback)
        event_bus.register_error_handler("test.error", error_handler)
        
        # Publish event that will cause an error
        test_event = Event(topic="test.error", data={})
        event_bus.publish(test_event)
        
        # Error handler should be called
        assert error_handled.wait(timeout=2.0)


class TestPersistenceAndSerialization:
    """Tests for event persistence and serialization"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for persistence"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_event_persistence(self, temp_dir):
        """Test persisting events to disk"""
        bus = EventBus(persistence_dir=temp_dir)
        bus.start(num_workers=1)
        
        try:
            # Enable persistence for a topic
            bus.enable_persistence("test.persist")
            
            # Create and publish events
            for i in range(5):
                event = Event(
                    topic="test.persist",
                    data={"index": i},
                    event_id=f"test-{i}"
                )
                bus.publish(event)
            
            # Allow time for events to be processed and persisted
            time.sleep(0.5)
            
            # Check that files were created
            persist_dir = Path(temp_dir) / "test.persist"
            assert persist_dir.exists()
            event_files = list(persist_dir.glob("*.event"))
            assert len(event_files) == 5
            
        finally:
            bus.stop()

    def test_replay_persisted_events(self, temp_dir):
        """Test replaying persisted events"""
        # First, create and persist some events
        bus1 = EventBus(persistence_dir=temp_dir)
        bus1.start(num_workers=1)
        
        try:
            bus1.enable_persistence("test.replay")
            
            # Create and publish events
            for i in range(3):
                event = Event(
                    topic="test.replay",
                    data={"index": i},
                    event_id=f"test-{i}"
                )
                bus1.publish_sync(event)  # Use sync to ensure immediate persistence
        finally:
            bus1.stop()
        
        # Now create a new bus and replay events
        bus2 = EventBus(persistence_dir=temp_dir)
        bus2.start(num_workers=1)
        
        try:
            received_events = []
            events_received = threading.Event()
            
            def callback(event):
                received_events.append(event)
                if len(received_events) >= 3:
                    events_received.set()
            
            # Subscribe to the replay topic
            bus2.subscribe("test.replay", callback)
            
            # Replay persisted events
            count = bus2.replay_persisted_events(topics=["test.replay"])
            assert count == 3
            
            # Wait for events to be processed
            assert events_received.wait(timeout=2.0)
            
            # Check received events
            assert len(received_events) == 3
            indices = [event.data["index"] for event in received_events]
            assert set(indices) == {0, 1, 2}
            
        finally:
            bus2.stop()

    def test_prune_persisted_events(self, temp_dir):
        """Test pruning old persisted events"""
        bus = EventBus(persistence_dir=temp_dir)
        bus.start(num_workers=1)
        
        try:
            bus.enable_persistence("test.prune")
            
            # Create and publish older events
            older_events = []
            for i in range(3):
                # Create events with timestamps in the past
                older_time = time.time() - 3600  # 1 hour ago
                event = Event(
                    topic="test.prune",
                    data={"age": "old", "index": i},
                    event_id=f"old-{i}",
                    timestamp=older_time
                )
                older_events.append(event)
            
            # Create and publish newer events
            newer_events = []
            for i in range(2):
                event = Event(
                    topic="test.prune",
                    data={"age": "new", "index": i},
                    event_id=f"new-{i}"
                )
                newer_events.append(event)
            
            # Publish all events
            for event in older_events + newer_events:
                bus.publish_sync(event)
            
            # Check that all events were persisted
            prune_dir = Path(temp_dir) / "test.prune"
            event_files = list(prune_dir.glob("*.event"))
            assert len(event_files) == 5
            
            # Prune older events
            cutoff_time = time.time() - 1800  # 30 minutes ago
            count = bus.prune_persisted_events(older_than=cutoff_time)
            assert count == 3
            
            # Check that only newer events remain
            event_files = list(prune_dir.glob("*.event"))
            assert len(event_files) == 2
            
        finally:
            bus.stop()


class TestMetrics:
    """Tests for event bus metrics"""

    @pytest.fixture
    def event_bus(self):
        """Create a fresh event bus with metrics enabled"""
        bus = EventBus(metrics_enabled=True, metrics_interval=0.1)
        bus.start(num_workers=2)
        yield bus
        # Stop the event bus after each test
        if bus._running:
            bus.stop()

    def test_basic_metrics_collection(self, event_bus):
        """Test basic metrics collection during event processing"""
        received = threading.Event()
        
        def callback(event):
            received.set()
        
        # Subscribe to a topic
        event_bus.subscribe("test.metrics", callback)
        
        # Publish an event
        event = Event(topic="test.metrics", data={})
        event_bus.publish(event)
        
        # Wait for event to be processed
        assert received.wait(timeout=1.0)
        
        # Get metrics
        metrics = event_bus.get_stats()
        
        # Check basic metrics
        assert metrics["metrics_enabled"] is True
        assert "published_events" in metrics
        assert "test.metrics" in metrics["published_events"]
        assert metrics["published_events"]["test.metrics"] == 1
        
        assert "delivered_events" in metrics
        assert "test.metrics" in metrics["delivered_events"]
        assert metrics["delivered_events"]["test.metrics"] == 1
        
        assert "dropped_events" in metrics
        assert "subscriber_topics" in metrics
        assert "test.metrics" in metrics["subscriber_topics"]

    def test_processing_time_metrics(self, event_bus):
        """Test processing time metrics collection"""
        def slow_callback(event):
            time.sleep(0.1)  # Simulate processing time
        
        # Subscribe to a topic with a slow callback
        event_bus.subscribe("test.slow", slow_callback)
        
        # Publish multiple events to get reliable metrics
        for _ in range(3):
            event = Event(topic="test.slow", data={})
            event_bus.publish_sync(event)
        
        # Get metrics
        metrics = event_bus.get_stats()
        
        # Check processing time metrics
        assert "avg_processing_times" in metrics
        assert "test.slow" in metrics["avg_processing_times"]
        assert metrics["avg_processing_times"]["test.slow"] > 0
        
        assert "max_processing_times" in metrics
        assert "test.slow" in metrics["max_processing_times"]
        assert metrics["max_processing_times"]["test.slow"] > 0

    def test_reset_metrics(self, event_bus):
        """Test resetting metrics"""
        # Publish some events
        for _ in range(3):
            event = Event(topic="test.reset", data={})
            event_bus.publish_sync(event)
        
        # Verify metrics are collected
        metrics_before = event_bus.get_stats()
        assert "published_events" in metrics_before
        assert "test.reset" in metrics_before["published_events"]
        assert metrics_before["published_events"]["test.reset"] == 3
        
        # Reset metrics
        event_bus.reset_stats()
        
        # Verify metrics are reset
        metrics_after = event_bus.get_stats()
        assert "published_events" in metrics_after
        assert "test.reset" not in metrics_after["published_events"]


class TestEventHelpers:
    """Tests for event helper functions"""

    def test_create_event(self):
        """Test create_event helper function"""
        event = create_event(
            topic="test.helper",
            data={"key": "value"},
            priority=EventPriority.HIGH,
            source="test_source",
            custom_meta="test_meta"
        )
        
        assert event.topic == "test.helper"
        assert event.data == {"key": "value"}
        assert event.priority == EventPriority.HIGH
        assert event.source == "test_source"
        assert event.metadata["custom_meta"] == "test_meta"

    def test_create_market_data_event(self):
        """Test create_market_data_event helper function"""
        event = create_market_data_event(
            symbol="AAPL",
            data={"price": 150.0},
            source="market_feed",
            exchange="NASDAQ"
        )
        
        assert event.topic == f"{EventTopics.MARKET_DATA}.AAPL"
        assert event.data == {"price": 150.0}
        assert event.priority == EventPriority.HIGH
        assert event.source == "market_feed"
        assert event.metadata["exchange"] == "NASDAQ"

    def test_create_order_event(self):
        """Test create_order_event helper function"""
        event = create_order_event(
            order_id="order123",
            event_type=EventTopics.ORDER_FILLED,
            data={"quantity": 100},
            client_id="client001"
        )
        
        assert event.topic == f"{EventTopics.ORDER_FILLED}.order123"
        assert event.data == {"quantity": 100}
        assert event.priority == EventPriority.HIGH
        assert event.source == "execution_service"
        assert event.metadata["client_id"] == "client001"

    def test_create_system_event(self):
        """Test create_system_event helper function"""
        event = create_system_event(
            event_type=EventTopics.SYSTEM_START,
            data={"version": "1.0.0"},
            environment="production"
        )
        
        assert event.topic == EventTopics.SYSTEM_START
        assert event.data == {"version": "1.0.0"}
        assert event.priority == EventPriority.HIGH
        assert event.source == "system"
        assert event.metadata["environment"] == "production"

    def test_create_strategy_event(self):
        """Test create_strategy_event helper function"""
        event = create_strategy_event(
            strategy_id="strat001",
            event_type=EventTopics.STRATEGY_STARTED,
            data={"params": {"key": "value"}},
            user="trader1"
        )
        
        assert event.topic == f"{EventTopics.STRATEGY_STARTED}.strat001"
        assert event.data == {"params": {"key": "value"}}
        assert event.priority == EventPriority.NORMAL
        assert event.source == "strategy.strat001"
        assert event.metadata["user"] == "trader1"

    def test_create_regime_event(self):
        """Test create_regime_event helper function"""
        event = create_regime_event(
            regime_type=EventTopics.REGIME_CHANGED,
            data={"new_regime": "bullish"},
            confidence=0.85
        )
        
        assert event.topic == EventTopics.REGIME_CHANGED
        assert event.data == {"new_regime": "bullish"}
        assert event.priority == EventPriority.HIGH
        assert event.source == "regime_classifier"
        assert event.metadata["confidence"] == 0.85

    def test_create_signal_event(self):
        """Test create_signal_event helper function"""
        event = create_signal_event(
            strategy_id="strat001",
            data={"signal": "BUY", "symbol": "AAPL"},
            strength=0.75
        )
        
        assert event.topic == f"{EventTopics.SIGNAL_GENERATED}.strat001"
        assert event.data == {"signal": "BUY", "symbol": "AAPL"}
        assert event.priority == EventPriority.HIGH
        assert event.source == "strategy.strat001"
        assert event.metadata["strength"] == 0.75


class TestSingleton:
    """Tests for the singleton pattern implementation"""

    def test_get_event_bus_singleton(self):
        """Test that get_event_bus returns the same instance"""
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        
        assert bus1 is bus2
        
        # Parameters should only apply to the first call
        bus3 = get_event_bus(max_queue_size=1000)
        assert bus1 is bus3


class TestLifecycleCallbacks:
    """Tests for lifecycle callbacks"""

    def test_register_lifecycle_callbacks(self):
        """Test registering and triggering lifecycle callbacks"""
        bus = EventBus()
        
        startup_called = threading.Event()
        shutdown_called = threading.Event()
        
        def on_startup():
            startup_called.set()
            
        def on_shutdown():
            shutdown_called.set()
            
        # Register callbacks
        bus.register_lifecycle_callback("startup", on_startup)
        bus.register_lifecycle_callback("shutdown", on_shutdown)
        
        # Start the bus - should trigger startup callback
        bus.start(num_workers=1)
        assert startup_called.wait(timeout=1.0)
        
        # Stop the bus - should trigger shutdown callback
        bus.stop()
        assert shutdown_called.wait(timeout=1.0)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])