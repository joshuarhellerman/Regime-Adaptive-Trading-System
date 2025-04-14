"""
Replay Service for ML-Powered Trading System

This module provides functionality to replay historical market data for backtesting
trading strategies. It simulates the behavior of real-time market data services
by replaying historical data in chronological order at customizable speeds.

Key responsibilities:
1. Load historical data from the historical repository
2. Replay data events at configurable speeds (real-time, accelerated, or step-by-step)
3. Interface with the event bus to publish market data events
4. Support for data preprocessing and normalization during replay
5. Maintain simulation clock and state during backtesting
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Callable, Any
import logging
import threading
from enum import Enum

from event_bus import EventBus
from data.fetchers.historical_repository import HistoricalRepository
from data.processors.data_normalizer import DataNormalizer
from data.market_data_service import MarketDataService


class ReplayMode(Enum):
    """Enumeration of available replay modes."""
    REAL_TIME = "real_time"  # Replay at the same pace as real market
    ACCELERATED = "accelerated"  # Replay at a faster pace (with speed multiplier)
    STEP = "step"  # Replay one event at a time (for detailed analysis)
    AS_FAST_AS_POSSIBLE = "as_fast_as_possible"  # Replay as fast as system can process


class ReplayService:
    """
    Service for replaying historical market data for backtesting purposes.

    This service loads historical data and replays it through the system's event bus,
    simulating real-time market conditions at configurable speeds to facilitate
    backtesting of trading strategies.
    """

    def __init__(
        self,
        event_bus: EventBus,
        historical_repository: HistoricalRepository,
        data_normalizer: Optional[DataNormalizer] = None,
        market_data_service: Optional[MarketDataService] = None
    ):
        """
        Initialize the ReplayService.

        Args:
            event_bus: The system event bus for publishing replayed market data events
            historical_repository: Repository for accessing historical market data
            data_normalizer: Optional normalizer for preprocessing historical data
            market_data_service: Optional reference to the market data service for coordination
        """
        self.event_bus = event_bus
        self.historical_repository = historical_repository
        self.data_normalizer = data_normalizer
        self.market_data_service = market_data_service

        self.logger = logging.getLogger(__name__)

        # Replay state
        self._is_replaying = False
        self._pause_replay = threading.Event()
        self._stop_replay = threading.Event()
        self._replay_thread = None
        self._current_replay_time = None
        self._replay_speed_multiplier = 1.0
        self._replay_mode = ReplayMode.REAL_TIME

        # Callbacks
        self._on_replay_started_callbacks = []
        self._on_replay_paused_callbacks = []
        self._on_replay_resumed_callbacks = []
        self._on_replay_stopped_callbacks = []
        self._on_replay_completed_callbacks = []
        self._on_event_replayed_callbacks = []

    def load_data(
        self,
        symbols: List[str],
        start_time: datetime,
        end_time: datetime,
        data_types: Optional[List[str]] = None,
        resolution: str = "1m"
    ) -> Dict[str, Any]:
        """
        Load historical data for the specified symbols and time range.

        Args:
            symbols: List of market symbols to load data for
            start_time: Start time for the historical data range
            end_time: End time for the historical data range
            data_types: Optional list of data types to load (e.g., ["trades", "quotes", "bars"])
            resolution: The resolution of data to load (e.g., "1m" for 1-minute bars)

        Returns:
            Dictionary containing loaded historical data organized by symbol and type
        """
        self.logger.info(f"Loading historical data for {len(symbols)} symbols from {start_time} to {end_time}")

        if data_types is None:
            data_types = ["bars"]  # Default to OHLCV bars if not specified

        historical_data = {}

        for symbol in symbols:
            historical_data[symbol] = {}
            for data_type in data_types:
                try:
                    data = self.historical_repository.get_historical_data(
                        symbol=symbol,
                        data_type=data_type,
                        start_time=start_time,
                        end_time=end_time,
                        resolution=resolution
                    )

                    if self.data_normalizer and data:
                        data = self.data_normalizer.normalize(data, data_type)

                    historical_data[symbol][data_type] = data
                    self.logger.debug(f"Loaded {len(data) if data else 0} {data_type} records for {symbol}")
                except Exception as e:
                    self.logger.error(f"Failed to load {data_type} data for {symbol}: {str(e)}")
                    historical_data[symbol][data_type] = []

        return historical_data

    def configure_replay(
        self,
        mode: ReplayMode = ReplayMode.REAL_TIME,
        speed_multiplier: float = 1.0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ):
        """
        Configure the replay parameters.

        Args:
            mode: The replay mode (real-time, accelerated, step, or as fast as possible)
            speed_multiplier: Speed multiplier for accelerated replay (e.g., 2.0 = twice as fast)
            start_time: Optional custom start time (defaults to earliest time in loaded data)
            end_time: Optional custom end time (defaults to latest time in loaded data)
        """
        self._replay_mode = mode
        self._replay_speed_multiplier = max(0.1, speed_multiplier)  # Minimum 0.1x speed

        if start_time:
            self._start_time = start_time

        if end_time:
            self._end_time = end_time

        self.logger.info(f"Replay configured: mode={mode.value}, speed={self._replay_speed_multiplier}x")

    def start_replay(
        self,
        historical_data: Optional[Dict[str, Any]] = None,
        symbols: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        data_types: Optional[List[str]] = None,
        resolution: str = "1m"
    ):
        """
        Start replaying historical data.

        Args:
            historical_data: Optional pre-loaded historical data to replay
            symbols: List of market symbols to replay (if historical_data not provided)
            start_time: Start time for replay (if historical_data not provided)
            end_time: End time for replay (if historical_data not provided)
            data_types: Types of data to replay (if historical_data not provided)
            resolution: Data resolution (if historical_data not provided)
        """
        if self._is_replaying:
            self.logger.warning("Replay is already in progress. Stop the current replay first.")
            return

        try:
            # Load data if not provided
            if not historical_data:
                if not symbols or not start_time or not end_time:
                    raise ValueError("Must provide symbols, start_time, and end_time if historical_data not provided")

                historical_data = self.load_data(
                    symbols=symbols,
                    start_time=start_time,
                    end_time=end_time,
                    data_types=data_types,
                    resolution=resolution
                )

            # Process and prepare the data for replay
            replay_events = self._prepare_replay_events(historical_data)

            if not replay_events:
                self.logger.warning("No events to replay. Replay not started.")
                return

            # Set replay state
            self._is_replaying = True
            self._pause_replay.clear()
            self._stop_replay.clear()

            # Start replay in a separate thread
            self._replay_thread = threading.Thread(
                target=self._replay_data,
                args=(replay_events,),
                daemon=True
            )
            self._replay_thread.start()

            # Notify listeners
            for callback in self._on_replay_started_callbacks:
                callback()

            self.logger.info(f"Started replay with {len(replay_events)} events")

        except Exception as e:
            self._is_replaying = False
            self.logger.error(f"Failed to start replay: {str(e)}")
            raise

    def pause_replay(self):
        """Pause an ongoing replay."""
        if not self._is_replaying:
            self.logger.warning("No replay in progress to pause.")
            return

        self._pause_replay.set()
        self.logger.info("Replay paused")

        # Notify listeners
        for callback in self._on_replay_paused_callbacks:
            callback()

    def resume_replay(self):
        """Resume a paused replay."""
        if not self._is_replaying:
            self.logger.warning("No replay in progress to resume.")
            return

        if not self._pause_replay.is_set():
            self.logger.warning("Replay is not paused.")
            return

        self._pause_replay.clear()
        self.logger.info("Replay resumed")

        # Notify listeners
        for callback in self._on_replay_resumed_callbacks:
            callback()

    def stop_replay(self):
        """Stop the ongoing replay completely."""
        if not self._is_replaying:
            self.logger.warning("No replay in progress to stop.")
            return

        self._stop_replay.set()
        if self._pause_replay.is_set():
            self._pause_replay.clear()

        if self._replay_thread and self._replay_thread.is_alive():
            self._replay_thread.join(timeout=5.0)

        self._is_replaying = False
        self.logger.info("Replay stopped")

        # Notify listeners
        for callback in self._on_replay_stopped_callbacks:
            callback()

    def step_forward(self):
        """
        Advance the replay by one step (when in STEP mode).

        This method is only relevant when the replay mode is set to STEP.
        It will process and publish the next event in the sequence.
        """
        if self._replay_mode != ReplayMode.STEP:
            self.logger.warning("Step forward only works in STEP replay mode.")
            return

        if not self._is_replaying:
            self.logger.warning("No replay in progress to step forward.")
            return

        # Signal the replay thread to process the next event
        self._pause_replay.clear()
        time.sleep(0.1)  # Brief pause to allow event processing
        self._pause_replay.set()

    def get_replay_status(self) -> Dict[str, Any]:
        """
        Get the current status of the replay.

        Returns:
            Dictionary with current replay status information
        """
        return {
            "is_replaying": self._is_replaying,
            "is_paused": self._pause_replay.is_set() if self._is_replaying else False,
            "current_time": self._current_replay_time,
            "mode": self._replay_mode.value if self._replay_mode else None,
            "speed_multiplier": self._replay_speed_multiplier
        }

    def register_callback(self, event_type: str, callback: Callable):
        """
        Register a callback function for replay events.

        Args:
            event_type: Type of event to register for (e.g., "started", "paused", "event_replayed")
            callback: Function to call when the event occurs
        """
        if event_type == "started":
            self._on_replay_started_callbacks.append(callback)
        elif event_type == "paused":
            self._on_replay_paused_callbacks.append(callback)
        elif event_type == "resumed":
            self._on_replay_resumed_callbacks.append(callback)
        elif event_type == "stopped":
            self._on_replay_stopped_callbacks.append(callback)
        elif event_type == "completed":
            self._on_replay_completed_callbacks.append(callback)
        elif event_type == "event_replayed":
            self._on_event_replayed_callbacks.append(callback)
        else:
            self.logger.warning(f"Unknown event type: {event_type}")

    def _prepare_replay_events(self, historical_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process and prepare historical data into a chronologically ordered list of events.

        Args:
            historical_data: Dictionary of historical data by symbol and type

        Returns:
            List of events ordered by timestamp
        """
        all_events = []

        # Process each symbol and data type
        for symbol, symbol_data in historical_data.items():
            for data_type, data_list in symbol_data.items():
                for data_point in data_list:
                    # Ensure all data points have a timestamp
                    if "timestamp" not in data_point:
                        self.logger.warning(f"Data point missing timestamp: {data_point}")
                        continue

                    event = {
                        "symbol": symbol,
                        "data_type": data_type,
                        "data": data_point,
                        "timestamp": data_point["timestamp"]
                    }
                    all_events.append(event)

        # Sort all events by timestamp
        all_events.sort(key=lambda event: event["timestamp"])

        return all_events

    def _replay_data(self, events: List[Dict[str, Any]]):
        """
        Main replay loop that publishes events according to the configured mode and speed.

        Args:
            events: Chronologically ordered list of events to replay
        """
        if not events:
            self.logger.warning("No events to replay.")
            self._is_replaying = False
            return

        # Initialize replay time to the first event
        self._current_replay_time = events[0]["timestamp"]
        prev_event_time = self._current_replay_time

        # Track real-world start time for calculating delays
        real_start_time = time.time()
        simulation_start_time = self._current_replay_time.timestamp()

        try:
            for i, event in enumerate(events):
                # Check if replay should stop
                if self._stop_replay.is_set():
                    break

                # Handle pause (if set)
                while self._pause_replay.is_set() and not self._stop_replay.is_set():
                    time.sleep(0.1)

                # Skip if stopped during pause
                if self._stop_replay.is_set():
                    break

                # Update current replay time
                event_time = event["timestamp"]
                self._current_replay_time = event_time

                # Calculate and apply appropriate delay based on replay mode
                if self._replay_mode == ReplayMode.REAL_TIME:
                    # Calculate the time difference between events in the data
                    if i > 0:
                        time_diff = (event_time - prev_event_time).total_seconds()
                        delay = time_diff / self._replay_speed_multiplier
                        if delay > 0:
                            time.sleep(delay)

                elif self._replay_mode == ReplayMode.ACCELERATED:
                    # Calculate how much simulated time has passed and how much real time should have passed
                    sim_time_elapsed = event_time.timestamp() - simulation_start_time
                    target_real_elapsed = sim_time_elapsed / self._replay_speed_multiplier
                    actual_elapsed = time.time() - real_start_time

                    # Sleep if we're ahead of schedule
                    if actual_elapsed < target_real_elapsed:
                        time.sleep(target_real_elapsed - actual_elapsed)

                elif self._replay_mode == ReplayMode.STEP:
                    # In step mode, pause after each event
                    self._pause_replay.set()

                # No delay for AS_FAST_AS_POSSIBLE mode

                # Publish the event to the event bus
                self._publish_event(event)

                # Update previous event time for next iteration
                prev_event_time = event_time

                # Notify event replayed callbacks
                for callback in self._on_event_replayed_callbacks:
                    callback(event)

                # Optional progress logging for long replays
                if i % 1000 == 0:
                    progress = f"{i}/{len(events)} events ({i/len(events)*100:.1f}%)"
                    self.logger.debug(f"Replay progress: {progress}")

            # Replay completed
            self.logger.info("Replay completed successfully")

            # Notify completion listeners
            for callback in self._on_replay_completed_callbacks:
                callback()

        except Exception as e:
            self.logger.error(f"Error during replay: {str(e)}")

        finally:
            self._is_replaying = False

    def _publish_event(self, event: Dict[str, Any]):
        """
        Publish a replay event to the event bus.

        Args:
            event: The event data to publish
        """
        # Determine the appropriate event type based on the data
        symbol = event["symbol"]
        data_type = event["data_type"]
        data = event["data"]

        if data_type == "bars":
            event_type = "market_data.bar"
        elif data_type == "trades":
            event_type = "market_data.trade"
        elif data_type == "quotes":
            event_type = "market_data.quote"
        elif data_type == "book":
            event_type = "market_data.orderbook"
        else:
            event_type = f"market_data.{data_type}"

        # Add metadata to indicate this is replayed data (for strategies/components that need to know)
        data["__replay"] = True
        data["__replay_time"] = self._current_replay_time

        # Publish to event bus
        self.event_bus.publish(
            event_type=event_type,
            data={
                "symbol": symbol,
                "data": data,
                "timestamp": self._current_replay_time
            }
        )