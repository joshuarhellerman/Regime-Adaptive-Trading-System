"""
Unit tests for the ReplayService class.

This module contains tests for verifying the functionality of the ReplayService,
which is responsible for replaying historical market data for backtesting purposes.
"""

import unittest
from unittest.mock import Mock, patch, call, ANY
import threading
from datetime import datetime, timedelta
import time
from typing import Dict, List, Any

from data.replay_service import ReplayService, ReplayMode
from event_bus import EventBus
from data.fetchers.historical_repository import HistoricalRepository
from data.processors.data_normalizer import DataNormalizer
from data.market_data_service import MarketDataService


class TestReplayService(unittest.TestCase):
    """Test cases for the ReplayService class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mock dependencies
        self.event_bus = Mock(spec=EventBus)
        self.historical_repository = Mock(spec=HistoricalRepository)
        self.data_normalizer = Mock(spec=DataNormalizer)
        self.market_data_service = Mock(spec=MarketDataService)

        # Initialize the ReplayService with mocks
        self.replay_service = ReplayService(
            event_bus=self.event_bus,
            historical_repository=self.historical_repository,
            data_normalizer=self.data_normalizer,
            market_data_service=self.market_data_service
        )

        # Sample time range for tests
        self.start_time = datetime(2023, 1, 1)
        self.end_time = datetime(2023, 1, 2)

        # Sample symbols
        self.symbols = ["AAPL", "MSFT"]

        # Sample data types
        self.data_types = ["bars", "trades"]

    def test_initialization(self):
        """Test that ReplayService initializes correctly with the provided dependencies."""
        self.assertEqual(self.replay_service.event_bus, self.event_bus)
        self.assertEqual(self.replay_service.historical_repository, self.historical_repository)
        self.assertEqual(self.replay_service.data_normalizer, self.data_normalizer)
        self.assertEqual(self.replay_service.market_data_service, self.market_data_service)

        # Check initial state
        self.assertFalse(self.replay_service._is_replaying)
        self.assertEqual(self.replay_service._replay_speed_multiplier, 1.0)
        self.assertEqual(self.replay_service._replay_mode, ReplayMode.REAL_TIME)

    def test_load_data(self):
        """Test loading historical data from the repository."""
        # Mock data to be returned from historical repository
        aapl_bars = [
            {"timestamp": self.start_time, "open": 150.0, "high": 152.0, "low": 149.0, "close": 151.0, "volume": 1000},
            {"timestamp": self.start_time + timedelta(minutes=1), "open": 151.0, "high": 153.0, "low": 150.0, "close": 152.0, "volume": 1200},
        ]

        msft_bars = [
            {"timestamp": self.start_time, "open": 250.0, "high": 252.0, "low": 249.0, "close": 251.0, "volume": 800},
            {"timestamp": self.start_time + timedelta(minutes=1), "open": 251.0, "high": 253.0, "low": 250.0, "close": 252.0, "volume": 900},
        ]

        # Configure the mock repository
        self.historical_repository.get_historical_data.side_effect = [
            aapl_bars,  # AAPL bars
            [],         # AAPL trades (empty for this test)
            msft_bars,  # MSFT bars
            [],         # MSFT trades (empty for this test)
        ]

        # Call the load_data method
        historical_data = self.replay_service.load_data(
            symbols=self.symbols,
            start_time=self.start_time,
            end_time=self.end_time,
            data_types=self.data_types,
            resolution="1m"
        )

        # Verify the historical repository was called correctly
        expected_calls = [
            call(symbol="AAPL", data_type="bars", start_time=self.start_time, end_time=self.end_time, resolution="1m"),
            call(symbol="AAPL", data_type="trades", start_time=self.start_time, end_time=self.end_time, resolution="1m"),
            call(symbol="MSFT", data_type="bars", start_time=self.start_time, end_time=self.end_time, resolution="1m"),
            call(symbol="MSFT", data_type="trades", start_time=self.start_time, end_time=self.end_time, resolution="1m"),
        ]
        self.historical_repository.get_historical_data.assert_has_calls(expected_calls)

        # Verify data normalizer was called for each non-empty dataset
        self.data_normalizer.normalize.assert_has_calls([
            call(aapl_bars, "bars"),
            call(msft_bars, "bars"),
        ])

        # Verify the returned data structure
        self.assertIn("AAPL", historical_data)
        self.assertIn("MSFT", historical_data)
        self.assertIn("bars", historical_data["AAPL"])
        self.assertIn("trades", historical_data["AAPL"])
        self.assertIn("bars", historical_data["MSFT"])
        self.assertIn("trades", historical_data["MSFT"])

    def test_load_data_handles_exception(self):
        """Test that load_data handles exceptions from the repository."""
        # Configure the mock repository to raise an exception
        self.historical_repository.get_historical_data.side_effect = Exception("Database connection error")

        # Call the load_data method
        historical_data = self.replay_service.load_data(
            symbols=["AAPL"],
            start_time=self.start_time,
            end_time=self.end_time,
            data_types=["bars"],
            resolution="1m"
        )

        # Verify the returned data structure contains empty lists for failed requests
        self.assertIn("AAPL", historical_data)
        self.assertIn("bars", historical_data["AAPL"])
        self.assertEqual(historical_data["AAPL"]["bars"], [])

    def test_configure_replay(self):
        """Test configuring replay parameters."""
        # Configure the replay
        custom_start = datetime(2023, 1, 1, 9, 30)
        custom_end = datetime(2023, 1, 1, 16, 0)

        self.replay_service.configure_replay(
            mode=ReplayMode.ACCELERATED,
            speed_multiplier=5.0,
            start_time=custom_start,
            end_time=custom_end
        )

        # Verify the configuration was applied
        self.assertEqual(self.replay_service._replay_mode, ReplayMode.ACCELERATED)
        self.assertEqual(self.replay_service._replay_speed_multiplier, 5.0)
        self.assertEqual(self.replay_service._start_time, custom_start)
        self.assertEqual(self.replay_service._end_time, custom_end)

    def test_configure_replay_minimum_speed(self):
        """Test that replay speed has a minimum value."""
        # Try to set speed below minimum
        self.replay_service.configure_replay(speed_multiplier=0.05)

        # Verify it was set to the minimum
        self.assertEqual(self.replay_service._replay_speed_multiplier, 0.1)

    @patch('threading.Thread')
    def test_start_replay_with_preloaded_data(self, mock_thread):
        """Test starting replay with preloaded historical data."""
        # Prepare test data
        historical_data = {
            "AAPL": {
                "bars": [
                    {"timestamp": self.start_time, "open": 150.0, "high": 152.0, "low": 149.0, "close": 151.0, "volume": 1000},
                    {"timestamp": self.start_time + timedelta(minutes=1), "open": 151.0, "high": 153.0, "low": 150.0, "close": 152.0, "volume": 1200},
                ]
            }
        }

        # Register a mock callback
        mock_callback = Mock()
        self.replay_service.register_callback("started", mock_callback)

        # Start replay
        self.replay_service.start_replay(historical_data=historical_data)

        # Verify thread was started
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()

        # Verify state changes
        self.assertTrue(self.replay_service._is_replaying)
        self.assertFalse(self.replay_service._pause_replay.is_set())
        self.assertFalse(self.replay_service._stop_replay.is_set())

        # Verify callback was called
        mock_callback.assert_called_once()

    def test_start_replay_without_data_raises_error(self):
        """Test that starting replay without data raises an error."""
        with self.assertRaises(ValueError):
            self.replay_service.start_replay()

    @patch.object(ReplayService, 'load_data')
    @patch('threading.Thread')
    def test_start_replay_loading_data(self, mock_thread, mock_load_data):
        """Test starting replay by loading data first."""
        # Mock the load_data method to return sample data
        mock_data = {
            "AAPL": {
                "bars": [
                    {"timestamp": self.start_time, "open": 150.0, "high": 152.0, "low": 149.0, "close": 151.0, "volume": 1000},
                ]
            }
        }
        mock_load_data.return_value = mock_data

        # Start replay
        self.replay_service.start_replay(
            symbols=["AAPL"],
            start_time=self.start_time,
            end_time=self.end_time,
            data_types=["bars"],
            resolution="1m"
        )

        # Verify load_data was called with the right parameters
        mock_load_data.assert_called_once_with(
            symbols=["AAPL"],
            start_time=self.start_time,
            end_time=self.end_time,
            data_types=["bars"],
            resolution="1m"
        )

        # Verify thread was started
        mock_thread.assert_called_once()

    def test_start_replay_no_events(self):
        """Test that replay doesn't start if there are no events."""
        # Mock empty historical data
        empty_data = {"AAPL": {"bars": []}}

        # Try to start replay
        self.replay_service.start_replay(historical_data=empty_data)

        # Verify replay didn't start
        self.assertFalse(self.replay_service._is_replaying)

    def test_pause_resume_replay(self):
        """Test pausing and resuming replay."""
        # Setup replay in progress
        self.replay_service._is_replaying = True
        self.replay_service._pause_replay.clear()

        # Register callbacks
        pause_callback = Mock()
        resume_callback = Mock()
        self.replay_service.register_callback("paused", pause_callback)
        self.replay_service.register_callback("resumed", resume_callback)

        # Test pause
        self.replay_service.pause_replay()
        self.assertTrue(self.replay_service._pause_replay.is_set())
        pause_callback.assert_called_once()

        # Test resume
        self.replay_service.resume_replay()
        self.assertFalse(self.replay_service._pause_replay.is_set())
        resume_callback.assert_called_once()

    def test_stop_replay(self):
        """Test stopping replay."""
        # Setup mock thread
        mock_thread = Mock()
        self.replay_service._replay_thread = mock_thread
        mock_thread.is_alive.return_value = True

        # Setup replay in progress
        self.replay_service._is_replaying = True
        self.replay_service._stop_replay.clear()

        # Register callback
        stop_callback = Mock()
        self.replay_service.register_callback("stopped", stop_callback)

        # Stop replay
        self.replay_service.stop_replay()

        # Verify state changes
        self.assertTrue(self.replay_service._stop_replay.is_set())
        self.assertFalse(self.replay_service._is_replaying)

        # Verify thread was joined
        mock_thread.join.assert_called_once_with(timeout=5.0)

        # Verify callback was called
        stop_callback.assert_called_once()

    def test_step_forward(self):
        """Test step forward functionality in STEP mode."""
        # Setup replay in STEP mode
        self.replay_service._replay_mode = ReplayMode.STEP
        self.replay_service._is_replaying = True
        self.replay_service._pause_replay.set()  # Paused

        # Step forward
        with patch('time.sleep') as mock_sleep:
            self.replay_service.step_forward()

            # Verify pause was temporarily cleared
            mock_sleep.assert_called_once_with(0.1)
            self.assertTrue(self.replay_service._pause_replay.is_set())

    def test_step_forward_wrong_mode(self):
        """Test step forward does nothing in non-STEP mode."""
        # Setup replay in REAL_TIME mode
        self.replay_service._replay_mode = ReplayMode.REAL_TIME
        self.replay_service._is_replaying = True
        self.replay_service._pause_replay.set()  # Paused

        # Step forward
        with patch('time.sleep') as mock_sleep:
            self.replay_service.step_forward()

            # Verify nothing happened
            mock_sleep.assert_not_called()
            self.assertTrue(self.replay_service._pause_replay.is_set())

    def test_get_replay_status(self):
        """Test getting replay status."""
        # Setup some replay state
        self.replay_service._is_replaying = True
        self.replay_service._pause_replay.set()
        self.replay_service._replay_mode = ReplayMode.ACCELERATED
        self.replay_service._replay_speed_multiplier = 2.5
        self.replay_service._current_replay_time = self.start_time

        # Get status
        status = self.replay_service.get_replay_status()

        # Verify returned status
        self.assertEqual(status["is_replaying"], True)
        self.assertEqual(status["is_paused"], True)
        self.assertEqual(status["current_time"], self.start_time)
        self.assertEqual(status["mode"], "accelerated")
        self.assertEqual(status["speed_multiplier"], 2.5)

    def test_prepare_replay_events(self):
        """Test preparing and sorting replay events."""
        # Create sample historical data with out-of-order timestamps
        time1 = self.start_time
        time2 = self.start_time + timedelta(minutes=1)
        time3 = self.start_time + timedelta(minutes=2)

        historical_data = {
            "AAPL": {
                "bars": [
                    {"timestamp": time2, "open": 151.0, "high": 153.0, "low": 150.0, "close": 152.0},
                    {"timestamp": time1, "open": 150.0, "high": 152.0, "low": 149.0, "close": 151.0},
                ],
                "trades": [
                    {"timestamp": time3, "price": 152.5, "size": 100},
                ],
            },
            "MSFT": {
                "bars": [
                    {"timestamp": time1, "open": 250.0, "high": 252.0, "low": 249.0, "close": 251.0},
                ],
            },
        }

        # Prepare events
        events = self.replay_service._prepare_replay_events(historical_data)

        # Verify events are sorted by timestamp
        self.assertEqual(len(events), 4)
        self.assertEqual(events[0]["timestamp"], time1)  # AAPL bar at time1
        self.assertEqual(events[1]["timestamp"], time1)  # MSFT bar at time1
        self.assertEqual(events[2]["timestamp"], time2)  # AAPL bar at time2
        self.assertEqual(events[3]["timestamp"], time3)  # AAPL trade at time3

    def test_prepare_replay_events_missing_timestamp(self):
        """Test handling data points with missing timestamps."""
        historical_data = {
            "AAPL": {
                "bars": [
                    {"open": 150.0, "high": 152.0, "low": 149.0, "close": 151.0},  # Missing timestamp
                    {"timestamp": self.start_time, "open": 151.0, "high": 153.0, "low": 150.0, "close": 152.0},
                ],
            },
        }

        # Prepare events
        events = self.replay_service._prepare_replay_events(historical_data)

        # Verify only the valid event was included
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["timestamp"], self.start_time)

    def test_publish_event(self):
        """Test publishing events to the event bus."""
        # Setup current replay time
        test_time = self.start_time
        self.replay_service._current_replay_time = test_time

        # Test different event types
        test_cases = [
            {
                "event": {
                    "symbol": "AAPL",
                    "data_type": "bars",
                    "data": {"open": 150, "high": 155, "low": 149, "close": 153, "volume": 1000}
                },
                "expected_type": "market_data.bar"
            },
            {
                "event": {
                    "symbol": "MSFT",
                    "data_type": "trades",
                    "data": {"price": 250, "size": 100}
                },
                "expected_type": "market_data.trade"
            },
            {
                "event": {
                    "symbol": "GOOG",
                    "data_type": "quotes",
                    "data": {"bid": 2800, "ask": 2802}
                },
                "expected_type": "market_data.quote"
            },
            {
                "event": {
                    "symbol": "AMZN",
                    "data_type": "book",
                    "data": {"bids": [[3400, 10], [3399, 15]], "asks": [[3401, 5], [3402, 20]]}
                },
                "expected_type": "market_data.orderbook"
            },
            {
                "event": {
                    "symbol": "IBM",
                    "data_type": "custom_data",
                    "data": {"value": 123}
                },
                "expected_type": "market_data.custom_data"
            },
        ]

        # Test each case
        for test_case in test_cases:
            # Reset mock
            self.event_bus.publish.reset_mock()

            # Call method
            self.replay_service._publish_event(test_case["event"])

            # Verify event bus was called correctly
            self.event_bus.publish.assert_called_once()
            call_args = self.event_bus.publish.call_args[1]

            self.assertEqual(call_args["event_type"], test_case["expected_type"])
            self.assertEqual(call_args["data"]["symbol"], test_case["event"]["symbol"])

            # Verify replay metadata was added
            self.assertTrue(call_args["data"]["data"]["__replay"])
            self.assertEqual(call_args["data"]["data"]["__replay_time"], test_time)
            self.assertEqual(call_args["data"]["timestamp"], test_time)

    @patch('time.sleep')
    @patch('time.time')
    def test_replay_data_real_time(self, mock_time, mock_sleep):
        """Test replay data in REAL_TIME mode."""
        # Setup mock time
        mock_time.return_value = 1000.0  # Start at time 1000

        # Setup replay mode
        self.replay_service._replay_mode = ReplayMode.REAL_TIME
        self.replay_service._replay_speed_multiplier = 1.0

        # Setup test events
        time1 = datetime(2023, 1, 1, 9, 30)
        time2 = datetime(2023, 1, 1, 9, 31)  # 1 minute later

        events = [
            {"symbol": "AAPL", "data_type": "bars", "data": {"open": 150}, "timestamp": time1},
            {"symbol": "AAPL", "data_type": "bars", "data": {"open": 151}, "timestamp": time2},
        ]

        # Register a mock callback
        mock_callback = Mock()
        self.replay_service.register_callback("event_replayed", mock_callback)
        mock_completed = Mock()
        self.replay_service.register_callback("completed", mock_completed)

        # Mock _publish_event
        with patch.object(self.replay_service, '_publish_event') as mock_publish:
            # Start the replay
            self.replay_service._is_replaying = True
            self.replay_service._replay_data(events)

            # Verify publish was called for each event
            self.assertEqual(mock_publish.call_count, 2)

            # Verify sleep was called between events
            mock_sleep.assert_called_once_with(60.0)  # 1 minute delay

            # Verify callbacks were called
            self.assertEqual(mock_callback.call_count, 2)
            mock_completed.assert_called_once()

    @patch('time.sleep')
    @patch('time.time')
    def test_replay_data_accelerated(self, mock_time, mock_sleep):
        """Test replay data in ACCELERATED mode."""
        # Setup mock time to increment with each call
        mock_time.side_effect = [1000.0, 1005.0, 1010.0]  # Simulates 5 seconds passing each time

        # Setup replay mode
        self.replay_service._replay_mode = ReplayMode.ACCELERATED
        self.replay_service._replay_speed_multiplier = 2.0  # 2x speed

        # Setup test events
        time1 = datetime(2023, 1, 1, 9, 30)
        time2 = datetime(2023, 1, 1, 9, 32)  # 2 minutes = 120 seconds later

        events = [
            {"symbol": "AAPL", "data_type": "bars", "data": {"open": 150}, "timestamp": time1},
            {"symbol": "AAPL", "data_type": "bars", "data": {"open": 151}, "timestamp": time2},
        ]

        # Mock _publish_event
        with patch.object(self.replay_service, '_publish_event'):
            # Start the replay
            self.replay_service._is_replaying = True
            self.replay_service._replay_data(events)

            # At 2x speed, the 120 second gap should become 60 seconds
            # But 5 seconds already passed in real time, so sleep should be 55 seconds
            mock_sleep.assert_called_once_with(55.0)

    @patch('time.sleep')
    def test_replay_data_step_mode(self, mock_sleep):
        """Test replay data in STEP mode."""
        # Setup replay mode
        self.replay_service._replay_mode = ReplayMode.STEP

        # Setup test events
        time1 = datetime(2023, 1, 1, 9, 30)
        time2 = datetime(2023, 1, 1, 9, 31)

        events = [
            {"symbol": "AAPL", "data_type": "bars", "data": {"open": 150}, "timestamp": time1},
            {"symbol": "AAPL", "data_type": "bars", "data": {"open": 151}, "timestamp": time2},
        ]

        # Mock _publish_event
        with patch.object(self.replay_service, '_publish_event') as mock_publish:
            # Setup a thread to simulate stepping forward
            def step_after_pause():
                # Wait for pause to be set
                while not self.replay_service._pause_replay.is_set():
                    time.sleep(0.01)

                # Step forward
                self.replay_service._pause_replay.clear()

                # Set stop after processing second event
                time.sleep(0.2)
                self.replay_service._stop_replay.set()

            threading.Thread(target=step_after_pause, daemon=True).start()

            # Start the replay
            self.replay_service._is_replaying = True
            self.replay_service._replay_data(events)

            # Verify pause was set after first event
            self.assertTrue(mock_publish.call_count >= 1)

    @patch('time.sleep')
    def test_replay_data_as_fast_as_possible(self, mock_sleep):
        """Test replay data in AS_FAST_AS_POSSIBLE mode."""
        # Setup replay mode
        self.replay_service._replay_mode = ReplayMode.AS_FAST_AS_POSSIBLE

        # Setup test events
        time1 = datetime(2023, 1, 1, 9, 30)
        time2 = datetime(2023, 1, 1, 9, 31)

        events = [
            {"symbol": "AAPL", "data_type": "bars", "data": {"open": 150}, "timestamp": time1},
            {"symbol": "AAPL", "data_type": "bars", "data": {"open": 151}, "timestamp": time2},
        ]

        # Mock _publish_event
        with patch.object(self.replay_service, '_publish_event'):
            # Start the replay
            self.replay_service._is_replaying = True
            self.replay_service._replay_data(events)

            # Verify no sleep was called
            mock_sleep.assert_not_called()

    def test_replay_data_empty_events(self):
        """Test replay data with empty events list."""
        # Mock callbacks
        completed_callback = Mock()
        self.replay_service.register_callback("completed", completed_callback)

        # Start replay with empty events
        self.replay_service._is_replaying = True
        self.replay_service._replay_data([])

        # Verify replay was marked as not running
        self.assertFalse(self.replay_service._is_replaying)

        # Verify completed callback was not called
        completed_callback.assert_not_called()

    @patch('time.sleep')
    def test_replay_data_handles_exception(self, mock_sleep):
        """Test that replay data handles exceptions gracefully."""
        # Setup test events
        time1 = datetime(2023, 1, 1, 9, 30)
        events = [{"symbol": "AAPL", "data_type": "bars", "data": {"open": 150}, "timestamp": time1}]

        # Mock _publish_event to raise an exception
        with patch.object(self.replay_service, '_publish_event', side_effect=Exception("Test error")):
            # Start replay
            self.replay_service._is_replaying = True
            self.replay_service._replay_data(events)

            # Verify replay was marked as not running
            self.assertFalse(self.replay_service._is_replaying)

    def test_register_callback(self):
        """Test registering callbacks for different event types."""
        # Create mock callbacks
        callbacks = {
            "started": Mock(),
            "paused": Mock(),
            "resumed": Mock(),
            "stopped": Mock(),
            "completed": Mock(),
            "event_replayed": Mock(),
            "unknown": Mock(),
        }

        # Register each callback
        for event_type, callback in callbacks.items():
            self.replay_service.register_callback(event_type, callback)

        # Verify callbacks were added to the appropriate lists (except unknown)
        self.assertIn(callbacks["started"], self.replay_service._on_replay_started_callbacks)
        self.assertIn(callbacks["paused"], self.replay_service._on_replay_paused_callbacks)
        self.assertIn(callbacks["resumed"], self.replay_service._on_replay_resumed_callbacks)
        self.assertIn(callbacks["stopped"], self.replay_service._on_replay_stopped_callbacks)
        self.assertIn(callbacks["completed"], self.replay_service._on_replay_completed_callbacks)
        self.assertIn(callbacks["event_replayed"], self.replay_service._on_event_replayed_callbacks)

        # Unknown event types should log a warning but not crash


if __name__ == '__main__':
    unittest.main()