"""
Tick Aggregator for ML-Powered Trading System.

This module provides high-performance tick data aggregation and transformation
for generating bar data and microstructure-based features critical for
high-frequency trading strategies and market regime detection.
"""

import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
from functools import lru_cache
import numba
from collections import defaultdict

# Import system dependencies
from utils.logger import get_logger
from data.processors.data_integrity import DataValidator
from data.market_data_service import MarketDataService


class TickAggregator:
    """
    High-performance tick data aggregator for feature engineering.

    This class provides:
    1. Efficient conversion from tick data to time-based bars (OHLCV)
    2. Volume-based bar generation (volume bars, dollar bars, tick bars)
    3. Microstructure feature extraction (order imbalance, trade flow, etc.)
    4. Real-time aggregation for streaming data processing
    """

    def __init__(self,
                 market_data_service: Optional[MarketDataService] = None,
                 data_validator: Optional[DataValidator] = None,
                 use_numba: bool = True,
                 cache_size: int = 100,
                 logging_level: int = logging.INFO):
        """
        Initialize the tick aggregator.

        Args:
            market_data_service: Optional MarketDataService for data access
            data_validator: Optional DataValidator for data validation
            use_numba: Whether to use Numba for performance optimization
            cache_size: Size of the LRU cache for aggregation results
            logging_level: Logging level
        """
        # Set up logging
        self.logger = get_logger("tick_aggregator")
        self.logger.setLevel(logging_level)

        # Initialize dependencies
        self.market_data_service = market_data_service
        self.data_validator = data_validator

        # Configuration
        self.use_numba = use_numba
        self.cache_size = cache_size

        # State
        self.tick_buffers = defaultdict(list)
        self.last_bar_time = {}
        self.aggregation_stats = defaultdict(dict)

        # Performance tracking
        self.performance_stats = {
            'aggregate_time': [],
            'process_time': [],
            'tick_count': []
        }

        self.logger.info("TickAggregator initialized")

    def aggregate_to_bars(self,
                         ticks: pd.DataFrame,
                         timeframe: str = '1m',
                         symbol: Optional[str] = None,
                         method: str = 'time',
                         threshold: Optional[int] = None) -> pd.DataFrame:
        """
        Aggregate tick data into OHLCV bars.

        Args:
            ticks: DataFrame containing tick data
            timeframe: Target timeframe for bars (e.g., '1m', '5m', '1h')
            symbol: Optional symbol identifier
            method: Aggregation method ('time', 'volume', 'dollar', 'tick')
            threshold: Threshold for volume/dollar/tick-based aggregation

        Returns:
            DataFrame with aggregated OHLCV bars
        """
        start_time = time.time()

        if ticks.empty:
            self.logger.warning("Empty tick data provided")
            return pd.DataFrame()

        # Validate tick data
        if self.data_validator and not self.data_validator.validate_tick_data(ticks):
            self.logger.error("Tick data validation failed")
            return pd.DataFrame()

        # Ensure timestamp is in index
        if not isinstance(ticks.index, pd.DatetimeIndex):
            if 'timestamp' in ticks.columns:
                ticks = ticks.set_index('timestamp')
            else:
                self.logger.error("Tick data must have timestamp column or index")
                return pd.DataFrame()

        # Sort by timestamp
        ticks = ticks.sort_index()

        # Get required columns with fallbacks
        price_col = 'price' if 'price' in ticks.columns else 'last'
        volume_col = 'volume' if 'volume' in ticks.columns else 'size'

        # Validate required columns exist
        if price_col not in ticks.columns:
            self.logger.error(f"Price column ({price_col}) not found in tick data")
            return pd.DataFrame()

        # Choose aggregation method
        if method == 'time':
            bars = self._time_based_aggregation(ticks, timeframe, price_col, volume_col)
        elif method == 'volume':
            threshold = threshold or 100000  # Default volume threshold
            bars = self._volume_based_aggregation(ticks, threshold, price_col, volume_col)
        elif method == 'dollar':
            threshold = threshold or 1000000  # Default dollar threshold
            bars = self._dollar_based_aggregation(ticks, threshold, price_col, volume_col)
        elif method == 'tick':
            threshold = threshold or 1000  # Default tick threshold
            bars = self._tick_based_aggregation(ticks, threshold, price_col, volume_col)
        else:
            self.logger.error(f"Unknown aggregation method: {method}")
            return pd.DataFrame()

        # Track performance
        elapsed = time.time() - start_time

        if symbol:
            self.aggregation_stats[symbol] = {
                'last_aggregation': datetime.now(),
                'tick_count': len(ticks),
                'bar_count': len(bars),
                'duration': elapsed,
                'timeframe': timeframe,
                'method': method
            }

        self.performance_stats['aggregate_time'].append(elapsed)
        self.performance_stats['tick_count'].append(len(ticks))

        if len(self.performance_stats['aggregate_time']) > 100:
            self.performance_stats['aggregate_time'].pop(0)
            self.performance_stats['tick_count'].pop(0)

        self.logger.info(f"Aggregated {len(ticks)} ticks into {len(bars)} {timeframe} bars "
                       f"using {method} method in {elapsed:.3f}s")

        return bars

    def _time_based_aggregation(self,
                               ticks: pd.DataFrame,
                               timeframe: str,
                               price_col: str,
                               volume_col: str) -> pd.DataFrame:
        """
        Perform time-based aggregation of ticks into OHLCV bars.

        Args:
            ticks: DataFrame with tick data
            timeframe: Target timeframe
            price_col: Price column name
            volume_col: Volume column name

        Returns:
            DataFrame with OHLCV bars
        """
        # Extract period from timeframe
        freq = self._timeframe_to_freq(timeframe)

        # Ensure we have volume data
        if volume_col not in ticks.columns:
            ticks[volume_col] = 1  # Default to 1 for counting trades

        # Basic OHLCV resampling
        ohlc = ticks[price_col].resample(freq).ohlc()
        volume = ticks[volume_col].resample(freq).sum()

        # Combine into one dataframe
        bars = pd.concat([ohlc, volume], axis=1)

        # Rename columns to standard format
        bars.columns = ['open', 'high', 'low', 'close', 'volume']

        # Drop rows with NaN values (periods with no ticks)
        bars = bars.dropna()

        return bars

    def _volume_based_aggregation(self,
                                 ticks: pd.DataFrame,
                                 threshold: int,
                                 price_col: str,
                                 volume_col: str) -> pd.DataFrame:
        """
        Perform volume-based aggregation of ticks into OHLCV bars.

        Args:
            ticks: DataFrame with tick data
            threshold: Volume threshold for each bar
            price_col: Price column name
            volume_col: Volume column name

        Returns:
            DataFrame with OHLCV bars
        """
        if volume_col not in ticks.columns:
            self.logger.warning(f"Volume column ({volume_col}) not found, using tick count")
            ticks[volume_col] = 1

        # Use optimized implementation if available
        if self.use_numba:
            try:
                return self._volume_based_aggregation_numba(
                    ticks[price_col].values,
                    ticks[volume_col].values,
                    ticks.index.values,
                    threshold
                )
            except Exception as e:
                self.logger.error(f"Numba aggregation failed: {str(e)}")
                # Fall back to pandas implementation

        # Create empty results
        bars = []

        # Variables to track current bar
        current_volume = 0
        bar_open = None
        bar_high = float('-inf')
        bar_low = float('inf')
        bar_start_time = None

        # Iterate through ticks
        for idx, row in ticks.iterrows():
            price = row[price_col]
            volume = row[volume_col] if volume_col in ticks.columns else 1

            # Start a new bar if this is the first tick
            if bar_open is None:
                bar_open = price
                bar_high = price
                bar_low = price
                bar_start_time = idx
                current_volume = 0

            # Update high and low
            bar_high = max(bar_high, price)
            bar_low = min(bar_low, price)
            current_volume += volume

            # If we've reached the volume threshold, close the bar
            if current_volume >= threshold:
                bars.append({
                    'timestamp': bar_start_time,
                    'open': bar_open,
                    'high': bar_high,
                    'low': bar_low,
                    'close': price,
                    'volume': current_volume
                })

                # Reset for next bar
                bar_open = None

        # Add the last incomplete bar if there is one
        if bar_open is not None:
            bars.append({
                'timestamp': bar_start_time,
                'open': bar_open,
                'high': bar_high,
                'low': bar_low,
                'close': price,
                'volume': current_volume
            })

        # Convert to DataFrame
        if not bars:
            return pd.DataFrame()

        result = pd.DataFrame(bars)
        if 'timestamp' in result.columns:
            result = result.set_index('timestamp')

        return result

    def _dollar_based_aggregation(self,
                                 ticks: pd.DataFrame,
                                 threshold: int,
                                 price_col: str,
                                 volume_col: str) -> pd.DataFrame:
        """
        Perform dollar-based aggregation of ticks into OHLCV bars.

        Args:
            ticks: DataFrame with tick data
            threshold: Dollar volume threshold for each bar
            price_col: Price column name
            volume_col: Volume column name

        Returns:
            DataFrame with OHLCV bars
        """
        if volume_col not in ticks.columns:
            self.logger.warning(f"Volume column ({volume_col}) not found, using tick count")
            ticks[volume_col] = 1

        # Create dollar volume column
        ticks['dollar_volume'] = ticks[price_col] * ticks[volume_col]

        # Use the volume-based aggregation with dollar volume
        return self._volume_based_aggregation(
            ticks,
            threshold,
            price_col,
            'dollar_volume'
        )

    def _tick_based_aggregation(self,
                               ticks: pd.DataFrame,
                               threshold: int,
                               price_col: str,
                               volume_col: str) -> pd.DataFrame:
        """
        Perform tick-based aggregation of ticks into OHLCV bars.

        Args:
            ticks: DataFrame with tick data
            threshold: Tick count threshold for each bar
            price_col: Price column name
            volume_col: Volume column name

        Returns:
            DataFrame with OHLCV bars
        """
        # Create a new DataFrame with a constant tick count column
        tick_data = ticks.copy()
        tick_data['tick_count'] = 1

        # Use the volume-based aggregation with tick count
        return self._volume_based_aggregation(
            tick_data,
            threshold,
            price_col,
            'tick_count'
        )

    @lru_cache(maxsize=10)
    def _timeframe_to_freq(self, timeframe: str) -> str:
        """Convert a timeframe string to a pandas frequency string."""
        timeframe = timeframe.lower()

        # Handle common formats
        if timeframe.endswith('s'):
            return timeframe
        elif timeframe.endswith('m'):
            return timeframe
        elif timeframe.endswith('h'):
            return timeframe
        elif timeframe.endswith('d'):
            return timeframe
        elif timeframe.endswith('w'):
            return timeframe

        # Handle numeric formats
        if timeframe.isdigit():
            return f"{timeframe}min"

        # Handle combined formats
        for unit in ['s', 'm', 'h', 'd', 'w']:
            if unit in timeframe:
                value, _ = timeframe.split(unit, 1)
                if value.isdigit():
                    if unit == 'm':
                        return f"{value}min"
                    elif unit == 'h':
                        return f"{value}h"
                    elif unit == 'd':
                        return f"{value}d"
                    elif unit == 'w':
                        return f"{value}w"
                    else:
                        return f"{value}{unit}"

        # Default for unknown formats
        self.logger.warning(f"Unknown timeframe format: {timeframe}, using 1m")
        return '1min'

    def process_streaming_tick(self,
                              tick: Dict[str, Any],
                              symbol: str,
                              timeframe: str = '1m') -> Optional[Dict[str, Any]]:
        """
        Process a single streaming tick and update bar data if complete.

        Args:
            tick: Dictionary containing tick data
            symbol: Symbol identifier
            timeframe: Timeframe for aggregation

        Returns:
            Dictionary with bar data if a bar was completed, None otherwise
        """
        start_time = time.time()

        # Validate tick data
        if 'price' not in tick and 'last' not in tick:
            self.logger.error("Tick data must contain price or last field")
            return None

        # Ensure timestamp exists
        if 'timestamp' not in tick:
            tick['timestamp'] = datetime.now()

        # Convert timestamp to datetime if it's a string
        if isinstance(tick['timestamp'], str):
            tick['timestamp'] = pd.Timestamp(tick['timestamp'])

        # Get price and volume values
        price = tick.get('price', tick.get('last'))
        volume = tick.get('volume', tick.get('size', 1))

        # Get buffer key
        key = f"{symbol}_{timeframe}"

        # Initialize bar time if needed
        if key not in self.last_bar_time:
            freq = self._timeframe_to_freq(timeframe)
            self.last_bar_time[key] = self._round_time(tick['timestamp'], freq)

        # Add tick to buffer
        self.tick_buffers[key].append({
            'timestamp': tick['timestamp'],
            'price': price,
            'volume': volume
        })

        # Check if we need to generate a new bar
        current_bar_time = self._round_time(tick['timestamp'], self._timeframe_to_freq(timeframe))

        if current_bar_time > self.last_bar_time[key]:
            # Generate bar from buffer
            buffer_df = pd.DataFrame(self.tick_buffers[key])
            buffer_df = buffer_df.set_index('timestamp')

            # Only include ticks from the previous bar period
            prev_bar_ticks = buffer_df[buffer_df.index < current_bar_time]

            if not prev_bar_ticks.empty:
                # Generate bar
                bar_data = {
                    'timestamp': self.last_bar_time[key],
                    'open': prev_bar_ticks['price'].iloc[0],
                    'high': prev_bar_ticks['price'].max(),
                    'low': prev_bar_ticks['price'].min(),
                    'close': prev_bar_ticks['price'].iloc[-1],
                    'volume': prev_bar_ticks['volume'].sum()
                }

                # Update buffer - keep only ticks for current bar
                self.tick_buffers[key] = [
                    tick for tick in self.tick_buffers[key]
                    if tick['timestamp'] >= current_bar_time
                ]

                # Update last bar time
                self.last_bar_time[key] = current_bar_time

                # Track performance
                elapsed = time.time() - start_time
                self.performance_stats['process_time'].append(elapsed)

                if len(self.performance_stats['process_time']) > 100:
                    self.performance_stats['process_time'].pop(0)

                return bar_data

        # No new bar yet
        return None

    def _round_time(self, timestamp: pd.Timestamp, freq: str) -> pd.Timestamp:
        """Round timestamp to the given frequency."""
        return timestamp.floor(freq)

    def augment_with_tick_features(self,
                                  bars: pd.DataFrame,
                                  ticks: pd.DataFrame) -> pd.DataFrame:
        """
        Augment OHLCV bar data with features derived from tick data.

        Args:
            bars: DataFrame with OHLCV bar data
            ticks: DataFrame with tick data

        Returns:
            DataFrame with additional tick-based features
        """
        if bars.empty or ticks.empty:
            return bars

        # Ensure we have DatetimeIndex
        if not isinstance(bars.index, pd.DatetimeIndex):
            self.logger.error("Bar data must have DatetimeIndex")
            return bars

        if not isinstance(ticks.index, pd.DatetimeIndex):
            if 'timestamp' in ticks.columns:
                ticks = ticks.set_index('timestamp')
            else:
                self.logger.error("Tick data must have timestamp column or index")
                return bars

        # Make a copy to avoid modifying input
        result = bars.copy()

        # Add basic tick-based features
        for idx, row in bars.iterrows():
            # Get ticks for this bar period
            if idx < bars.index[-1]:
                next_time = bars.index[bars.index.get_loc(idx) + 1]
                period_ticks = ticks[(ticks.index >= idx) & (ticks.index < next_time)]
            else:
                # For the last bar, just use ticks from its start time
                period_ticks = ticks[ticks.index >= idx]

            if not period_ticks.empty:
                # Calculate features
                result.loc[idx, 'tick_count'] = len(period_ticks)

                # Use price column if available, otherwise try 'last'
                if 'price' in period_ticks.columns:
                    price_col = 'price'
                elif 'last' in period_ticks.columns:
                    price_col = 'last'
                else:
                    continue

                # Price features
                result.loc[idx, 'price_std'] = period_ticks[price_col].std()
                result.loc[idx, 'price_range'] = period_ticks[price_col].max() - period_ticks[price_col].min()

                # Time-based features
                if len(period_ticks) > 1:
                    time_diffs = period_ticks.index.to_series().diff().dropna()
                    result.loc[idx, 'avg_tick_interval'] = time_diffs.mean().total_seconds()
                    result.loc[idx, 'max_tick_interval'] = time_diffs.max().total_seconds()

                # Trade direction features
                if len(period_ticks) > 1:
                    price_changes = period_ticks[price_col].diff().dropna()
                    up_ticks = (price_changes > 0).sum()
                    down_ticks = (price_changes < 0).sum()
                    unchanged = (price_changes == 0).sum()

                    result.loc[idx, 'up_tick_ratio'] = up_ticks / len(price_changes) if len(price_changes) > 0 else 0
                    result.loc[idx, 'down_tick_ratio'] = down_ticks / len(price_changes) if len(price_changes) > 0 else 0

                    # Calculate microstructure-based metrics if we have bid/ask data
                    if all(col in period_ticks.columns for col in ['bid', 'ask']):
                        mid_prices = (period_ticks['bid'] + period_ticks['ask']) / 2
                        spreads = period_ticks['ask'] - period_ticks['bid']

                        result.loc[idx, 'avg_spread'] = spreads.mean()
                        result.loc[idx, 'max_spread'] = spreads.max()

                        # Order imbalance using bid/ask volumes if available
                        if all(col in period_ticks.columns for col in ['bid_size', 'ask_size']):
                            bid_vol = period_ticks['bid_size'].sum()
                            ask_vol = period_ticks['ask_size'].sum()

                            total_vol = bid_vol + ask_vol
                            if total_vol > 0:
                                result.loc[idx, 'order_imbalance'] = (bid_vol - ask_vol) / total_vol

        return result

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the tick aggregator.

        Returns:
            Dictionary with performance statistics
        """
        stats = {
            'avg_aggregate_time': np.mean(self.performance_stats['aggregate_time']) if self.performance_stats['aggregate_time'] else 0,
            'avg_process_time': np.mean(self.performance_stats['process_time']) if self.performance_stats['process_time'] else 0,
            'avg_ticks_per_agg': np.mean(self.performance_stats['tick_count']) if self.performance_stats['tick_count'] else 0,
            'buffer_sizes': {k: len(v) for k, v in self.tick_buffers.items()},
            'aggregation_stats': dict(self.aggregation_stats)
        }

        return stats

    def clear_buffers(self, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> None:
        """
        Clear tick buffers for specific symbol/timeframe or all if not specified.

        Args:
            symbol: Optional symbol filter
            timeframe: Optional timeframe filter
        """
        if symbol and timeframe:
            key = f"{symbol}_{timeframe}"
            if key in self.tick_buffers:
                self.tick_buffers[key] = []
                if key in self.last_bar_time:
                    del self.last_bar_time[key]

                self.logger.info(f"Cleared all buffers for timeframe {timeframe}")
        else:
            # Clear all buffers
            self.tick_buffers.clear()
            self.last_bar_time.clear()
            self.logger.info("Cleared all tick buffers")

    @staticmethod
    def _volume_based_aggregation_numba(prices, volumes, timestamps, threshold):
        """
        Numba-optimized implementation of volume-based aggregation.

        Args:
            prices: Array of price values
            volumes: Array of volume values
            timestamps: Array of timestamp values (as np.datetime64)
            threshold: Volume threshold for each bar

        Returns:
            DataFrame with OHLCV bars
        """
        # This would be implemented with Numba in a production system
        # Here's a simplified version without Numba dependency
        n = len(prices)
        if n == 0:
            return pd.DataFrame()

        # Create empty results
        bars = []

        # Variables to track current bar
        current_volume = 0
        bar_open = None
        bar_high = float('-inf')
        bar_low = float('inf')
        bar_start_time = None

        # Iterate through ticks
        for i in range(n):
            price = prices[i]
            volume = volumes[i]
            timestamp = timestamps[i]

            # Start a new bar if this is the first tick
            if bar_open is None:
                bar_open = price
                bar_high = price
                bar_low = price
                bar_start_time = timestamp
                current_volume = 0

            # Update high and low
            bar_high = max(bar_high, price)
            bar_low = min(bar_low, price)
            current_volume += volume

            # If we've reached the volume threshold, close the bar
            if current_volume >= threshold:
                bars.append({
                    'timestamp': bar_start_time,
                    'open': bar_open,
                    'high': bar_high,
                    'low': bar_low,
                    'close': price,
                    'volume': current_volume
                })

                # Reset for next bar
                bar_open = None
                bar_high = float('-inf')
                bar_low = float('inf')
                bar_start_time = None
                current_volume = 0

        # Add the last incomplete bar if there is one
        if bar_open is not None:
            bars.append({
                'timestamp': bar_start_time,
                'open': bar_open,
                'high': bar_high,
                'low': bar_low,
                'close': prices[-1],
                'volume': current_volume
            })

        # Convert to DataFrame
        if not bars:
            return pd.DataFrame()

        result = pd.DataFrame(bars)
        if 'timestamp' in result.columns:
            result = result.set_index('timestamp')

        return result

    def get_microstructure_features(self,
                                   ticks: pd.DataFrame,
                                   bars: Optional[pd.DataFrame] = None,
                                   include_basic: bool = True,
                                   include_advanced: bool = False) -> pd.DataFrame:
        """
        Extract microstructure features from tick data.

        Args:
            ticks: DataFrame with tick data
            bars: Optional DataFrame with OHLCV bars for alignment
            include_basic: Whether to include basic microstructure features
            include_advanced: Whether to include advanced microstructure features

        Returns:
            DataFrame with microstructure features
        """
        if ticks.empty:
            return pd.DataFrame()

        # Ensure timestamp index
        if not isinstance(ticks.index, pd.DatetimeIndex):
            if 'timestamp' in ticks.columns:
                ticks = ticks.set_index('timestamp')
            else:
                self.logger.error("Tick data must have timestamp column or index")
                return pd.DataFrame()

        # Sort by timestamp
        ticks = ticks.sort_index()

        # Create result dataframe
        if bars is not None:
            result = bars.copy()
        else:
            # Generate bars at 1-minute frequency if not provided
            result = self.aggregate_to_bars(ticks, '1m')

        if result.empty:
            return result

        # Add basic features
        if include_basic:
            # Add features for each bar
            for i, bar_time in enumerate(result.index):
                # Get next bar time
                if i < len(result) - 1:
                    next_bar_time = result.index[i + 1]
                else:
                    next_bar_time = bar_time + pd.Timedelta('1d')  # Far in the future

                # Get ticks for this bar
                bar_ticks = ticks[(ticks.index >= bar_time) & (ticks.index < next_bar_time)]

                if len(bar_ticks) > 0:
                    # Add basic tick count
                    result.loc[bar_time, 'tick_count'] = len(bar_ticks)

                    # Add trade direction metrics
                    if len(bar_ticks) > 1:
                        # Determine price column
                        price_col = None
                        for col in ['price', 'last', 'trade_price']:
                            if col in bar_ticks.columns:
                                price_col = col
                                break

                        if price_col:
                            # Calculate price changes
                            price_changes = bar_ticks[price_col].diff().dropna()

                            # Count direction changes
                            up_ticks = (price_changes > 0).sum()
                            down_ticks = (price_changes < 0).sum()

                            # Calculate ratios
                            total_changes = len(price_changes)
                            if total_changes > 0:
                                result.loc[bar_time, 'up_ratio'] = up_ticks / total_changes
                                result.loc[bar_time, 'down_ratio'] = down_ticks / total_changes

                                # Add trend strength
                                result.loc[bar_time, 'trend_strength'] = abs(up_ticks - down_ticks) / total_changes

                            # Add volatility metrics
                            result.loc[bar_time, 'tick_price_std'] = bar_ticks[price_col].std()
                            result.loc[bar_time, 'tick_price_range'] = bar_ticks[price_col].max() - bar_ticks[price_col].min()

                    # Add time-based metrics
                    if len(bar_ticks) > 1:
                        # Calculate time between ticks
                        tick_times = bar_ticks.index.to_series()
                        time_diffs = tick_times.diff().dropna()

                        # Convert to seconds
                        time_diffs_sec = time_diffs.dt.total_seconds()

                        result.loc[bar_time, 'avg_tick_interval'] = time_diffs_sec.mean()
                        result.loc[bar_time, 'max_tick_interval'] = time_diffs_sec.max()
                        result.loc[bar_time, 'min_tick_interval'] = time_diffs_sec.min()
                        result.loc[bar_time, 'tick_interval_std'] = time_diffs_sec.std()

        # Add advanced features
        if include_advanced:
            # Check if we have necessary bid/ask data
            has_bid_ask = all(col in ticks.columns for col in ['bid', 'ask'])
            has_sizes = all(col in ticks.columns for col in ['bid_size', 'ask_size'])

            if has_bid_ask:
                # Process each bar
                for i, bar_time in enumerate(result.index):
                    # Get next bar time
                    if i < len(result) - 1:
                        next_bar_time = result.index[i + 1]
                    else:
                        next_bar_time = bar_time + pd.Timedelta('1d')  # Far in the future

                    # Get ticks for this bar
                    bar_ticks = ticks[(ticks.index >= bar_time) & (ticks.index < next_bar_time)]

                    if len(bar_ticks) > 0:
                        # Calculate spread metrics
                        spreads = bar_ticks['ask'] - bar_ticks['bid']
                        result.loc[bar_time, 'avg_spread'] = spreads.mean()
                        result.loc[bar_time, 'max_spread'] = spreads.max()
                        result.loc[bar_time, 'min_spread'] = spreads.min()
                        result.loc[bar_time, 'spread_std'] = spreads.std()

                        # Calculate mid-price metrics
                        mid_prices = (bar_ticks['bid'] + bar_ticks['ask']) / 2

                        if len(mid_prices) > 1:
                            mid_price_changes = mid_prices.diff().dropna()
                            result.loc[bar_time, 'mid_price_volatility'] = mid_price_changes.std()

                        # Add order imbalance metrics if we have sizes
                        if has_sizes:
                            bid_volume = bar_ticks['bid_size'].sum()
                            ask_volume = bar_ticks['ask_size'].sum()

                            total_volume = bid_volume + ask_volume
                            if total_volume > 0:
                                result.loc[bar_time, 'order_imbalance'] = (bid_volume - ask_volume) / total_volume

                                # Add pressure metrics
                                if 'trade_direction' in bar_ticks.columns:
                                    # Count buys and sells
                                    buys = (bar_ticks['trade_direction'] > 0).sum()
                                    sells = (bar_ticks['trade_direction'] < 0).sum()

                                    if buys + sells > 0:
                                        result.loc[bar_time, 'buy_pressure'] = buys / (buys + sells)
                                        result.loc[bar_time, 'sell_pressure'] = sells / (buys + sells)
                                elif 'side' in bar_ticks.columns:
                                    # Count buys and sells
                                    buys = (bar_ticks['side'] == 'buy').sum()
                                    sells = (bar_ticks['side'] == 'sell').sum()

                                    if buys + sells > 0:
                                        result.loc[bar_time, 'buy_pressure'] = buys / (buys + sells)
                                        result.loc[bar_time, 'sell_pressure'] = sells / (buys + sells)

        return resultCleared buffer for {key}")
        elif symbol:
            # Clear all buffers for this symbol
            for key in list(self.tick_buffers.keys()):
                if key.startswith(f"{symbol}_"):
                    self.tick_buffers[key] = []
                    if key in self.last_bar_time:
                        del self.last_bar_time[key]

            self.logger.info(f"Cleared all buffers for {symbol}")
        elif timeframe:
            # Clear all buffers for this timeframe
            for key in list(self.tick_buffers.keys()):
                if key.endswith(f"_{timeframe}"):
                    self.tick_buffers[key] = []
                    if key in self.last_bar_time:
                        del self.last_bar_time[key]

            self.logger.info(f"