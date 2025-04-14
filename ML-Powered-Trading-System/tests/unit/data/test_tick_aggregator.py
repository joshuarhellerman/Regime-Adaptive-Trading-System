import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
from collections import defaultdict

# Import the class to be tested
from data.processors.tick_aggregator import TickAggregator
from data.market_data_service import MarketDataService
from data.processors.data_integrity import DataValidator


class TestTickAggregator(unittest.TestCase):
    """Test suite for the TickAggregator class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock dependencies
        self.mock_market_data_service = Mock(spec=MarketDataService)
        self.mock_data_validator = Mock(spec=DataValidator)
        self.mock_data_validator.validate_tick_data.return_value = True

        # Create the aggregator instance
        self.aggregator = TickAggregator(
            market_data_service=self.mock_market_data_service,
            data_validator=self.mock_data_validator,
            use_numba=False,  # Disable Numba for consistent testing
            cache_size=10
        )

        # Create sample tick data
        self.create_sample_data()

    def create_sample_data(self):
        """Create sample tick data for testing."""
        # Create a time series of ticks over 5 minutes
        now = datetime.now()
        timestamps = [now + timedelta(seconds=i) for i in range(300)]  # 5 minutes of data
        
        # Create price series with some realistic fluctuations
        base_price = 100.0
        np.random.seed(42)  # For reproducibility
        price_changes = np.random.normal(0, 0.01, len(timestamps))
        prices = base_price + np.cumsum(price_changes)
        
        # Create volume data
        volumes = np.random.randint(1, 100, len(timestamps))
        
        # Create dataframe
        self.tick_data = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': volumes
        })
        self.tick_data.set_index('timestamp', inplace=True)

        # Create bid/ask data for microstructure tests
        spreads = np.random.uniform(0.01, 0.05, len(timestamps))
        self.tick_data['bid'] = prices - spreads / 2
        self.tick_data['ask'] = prices + spreads / 2
        self.tick_data['bid_size'] = np.random.randint(1, 50, len(timestamps))
        self.tick_data['ask_size'] = np.random.randint(1, 50, len(timestamps))

    def test_init(self):
        """Test the initialization of TickAggregator."""
        self.assertIsInstance(self.aggregator.tick_buffers, defaultdict)
        self.assertEqual(self.aggregator.cache_size, 10)
        self.assertFalse(self.aggregator.use_numba)
        self.assertEqual(self.aggregator.market_data_service, self.mock_market_data_service)
        self.assertEqual(self.aggregator.data_validator, self.mock_data_validator)

    def test_aggregate_to_bars_time_based(self):
        """Test time-based aggregation."""
        # Test 1-minute bars
        bars = self.aggregator.aggregate_to_bars(
            self.tick_data,
            timeframe='1m',
            method='time'
        )
        
        # We should have around 5 bars for 5 minutes of data
        self.assertGreaterEqual(len(bars), 4)
        self.assertLessEqual(len(bars), 6)
        
        # Check that bars have the expected columns
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        self.assertListEqual(list(bars.columns), expected_columns)
        
        # Check that bars are sorted by time
        self.assertTrue(bars.index.is_monotonic_increasing)
        
        # Test that high >= open >= low for each bar
        self.assertTrue(all(bars['high'] >= bars['open']))
        self.assertTrue(all(bars['high'] >= bars['close']))
        self.assertTrue(all(bars['open'] >= bars['low']))
        self.assertTrue(all(bars['close'] >= bars['low']))

    def test_aggregate_to_bars_volume_based(self):
        """Test volume-based aggregation."""
        volume_threshold = 500
        
        bars = self.aggregator.aggregate_to_bars(
            self.tick_data,
            method='volume',
            threshold=volume_threshold
        )
        
        # Check that each bar has at least the threshold volume
        # (except possibly the last one)
        if len(bars) > 1:
            self.assertTrue(all(bars['volume'].iloc[:-1] >= volume_threshold))
        
        # Check expected columns
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        self.assertListEqual(list(bars.columns), expected_columns)

    def test_aggregate_to_bars_dollar_based(self):
        """Test dollar-based aggregation."""
        dollar_threshold = 50000
        
        bars = self.aggregator.aggregate_to_bars(
            self.tick_data,
            method='dollar',
            threshold=dollar_threshold
        )
        
        # Dollar bars should be created
        self.assertGreater(len(bars), 0)
        
        # Check expected columns
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        self.assertListEqual(list(bars.columns), expected_columns)

    def test_aggregate_to_bars_tick_based(self):
        """Test tick-based aggregation."""
        tick_threshold = 50
        
        bars = self.aggregator.aggregate_to_bars(
            self.tick_data,
            method='tick',
            threshold=tick_threshold
        )
        
        # With 300 ticks and a threshold of 50, we should have around 6 bars
        self.assertGreaterEqual(len(bars), 5)
        self.assertLessEqual(len(bars), 7)
        
        # Check expected columns
        expected_columns = ['open', 'high', 'low', 'close', 'volume']
        self.assertListEqual(list(bars.columns), expected_columns)

    def test_aggregate_to_bars_empty_data(self):
        """Test aggregation with empty data."""
        empty_df = pd.DataFrame()
        bars = self.aggregator.aggregate_to_bars(empty_df, timeframe='1m')
        
        self.assertTrue(bars.empty)

    def test_aggregate_to_bars_with_validation_failure(self):
        """Test aggregation when validation fails."""
        # Make validator return False
        self.mock_data_validator.validate_tick_data.return_value = False
        
        bars = self.aggregator.aggregate_to_bars(self.tick_data, timeframe='1m')
        
        # Should return empty dataframe when validation fails
        self.assertTrue(bars.empty)
        
        # Reset validator
        self.mock_data_validator.validate_tick_data.return_value = True

    def test_timeframe_to_freq(self):
        """Test conversion of timeframe strings to pandas frequency strings."""
        # Test various formats
        self.assertEqual(self.aggregator._timeframe_to_freq('1m'), '1min')
        self.assertEqual(self.aggregator._timeframe_to_freq('5m'), '5min')
        self.assertEqual(self.aggregator._timeframe_to_freq('1h'), '1h')
        self.assertEqual(self.aggregator._timeframe_to_freq('1d'), '1d')
        self.assertEqual(self.aggregator._timeframe_to_freq('1w'), '1w')
        
        # Test numeric format
        self.assertEqual(self.aggregator._timeframe_to_freq('60'), '60min')
        
        # Test unknown format
        self.assertEqual(self.aggregator._timeframe_to_freq('unknown'), '1min')

    def test_process_streaming_tick(self):
        """Test processing a single streaming tick."""
        symbol = 'AAPL'
        timeframe = '1m'
        
        # Process a tick
        tick = {
            'timestamp': datetime.now(),
            'price': 150.0,
            'volume': 100
        }
        
        # First tick should not generate a bar
        result = self.aggregator.process_streaming_tick(tick, symbol, timeframe)
        self.assertIsNone(result)
        
        # Check that tick was added to buffer
        key = f"{symbol}_{timeframe}"
        self.assertEqual(len(self.aggregator.tick_buffers[key]), 1)
        
        # Add more ticks from later time periods
        future_time = tick['timestamp'] + timedelta(minutes=2)  # Next minute
        
        # Add ticks for the next minute
        for i in range(5):
            new_tick = {
                'timestamp': future_time + timedelta(seconds=i),
                'price': 151.0 + i * 0.1,
                'volume': 10
            }
            result = self.aggregator.process_streaming_tick(new_tick, symbol, timeframe)
        
        # Should have generated a bar
        self.assertIsNotNone(result)
        
        # Check bar structure
        self.assertIn('timestamp', result)
        self.assertIn('open', result)
        self.assertIn('high', result)
        self.assertIn('low', result)
        self.assertIn('close', result)
        self.assertIn('volume', result)

    def test_augment_with_tick_features(self):
        """Test augmenting bar data with tick features."""
        # Generate time bars
        bars = self.aggregator.aggregate_to_bars(
            self.tick_data,
            timeframe='1m',
            method='time'
        )
        
        # Augment with features
        augmented_bars = self.aggregator.augment_with_tick_features(bars, self.tick_data)
        
        # Should have added features
        self.assertGreater(len(augmented_bars.columns), len(bars.columns))
        
        # Check that we have some of the expected new columns
        new_columns = ['tick_count', 'price_std', 'price_range']
        for col in new_columns:
            self.assertIn(col, augmented_bars.columns)

    def test_get_microstructure_features_basic(self):
        """Test extraction of basic microstructure features."""
        features = self.aggregator.get_microstructure_features(
            self.tick_data,
            include_basic=True,
            include_advanced=False
        )
        
        # Should have created some bars with features
        self.assertGreater(len(features), 0)
        
        # Basic features should be present
        basic_features = ['tick_count', 'tick_price_std', 'tick_price_range']
        for feature in basic_features:
            self.assertIn(feature, features.columns)

    def test_get_microstructure_features_advanced(self):
        """Test extraction of advanced microstructure features."""
        features = self.aggregator.get_microstructure_features(
            self.tick_data,
            include_basic=True,
            include_advanced=True
        )
        
        # Should have created some bars with features
        self.assertGreater(len(features), 0)
        
        # Advanced features should be present
        advanced_features = ['avg_spread', 'max_spread', 'min_spread', 'order_imbalance']
        for feature in advanced_features:
            self.assertIn(feature, features.columns)

    def test_get_performance_stats(self):
        """Test getting performance statistics."""
        # First aggregate some data to generate stats
        self.aggregator.aggregate_to_bars(self.tick_data, timeframe='1m')
        
        # Get stats
        stats = self.aggregator.get_performance_stats()
        
        # Should have stats
        self.assertIn('avg_aggregate_time', stats)
        self.assertIn('buffer_sizes', stats)
        self.assertIn('aggregation_stats', stats)

    def test_clear_buffers(self):
        """Test clearing tick buffers."""
        symbol = 'AAPL'
        timeframe = '1m'
        
        # Add some ticks to buffer
        tick = {
            'timestamp': datetime.now(),
            'price': 150.0,
            'volume': 100
        }
        self.aggregator.process_streaming_tick(tick, symbol, timeframe)
        
        # Verify buffer is not empty
        key = f"{symbol}_{timeframe}"
        self.assertGreater(len(self.aggregator.tick_buffers[key]), 0)
        
        # Clear buffers for this symbol and timeframe
        self.aggregator.clear_buffers(symbol, timeframe)
        
        # Buffer should now be empty
        self.assertEqual(len(self.aggregator.tick_buffers[key]), 0)
        
        # Add tick to different symbol
        self.aggregator.process_streaming_tick(tick, 'MSFT', timeframe)
        
        # Clear all buffers
        self.aggregator.clear_buffers()
        
        # All buffers should be empty
        self.assertEqual(len(self.aggregator.tick_buffers), 0)

    def test_handle_invalid_tick(self):
        """Test handling an invalid tick in streaming processing."""
        # Tick without price
        invalid_tick = {
            'timestamp': datetime.now(),
            'volume': 100
        }
        
        # Should not crash and return None
        result = self.aggregator.process_streaming_tick(invalid_tick, 'AAPL', '1m')
        self.assertIsNone(result)

    @patch('time.time')
    def test_performance_tracking(self, mock_time):
        """Test that performance tracking is working."""
        # Mock time.time() to return predictable values
        mock_time.side_effect = [10.0, 11.0]  # 1 second elapsed
        
        # Aggregate data
        self.aggregator.aggregate_to_bars(self.tick_data, timeframe='1m')
        
        # Check performance stats
        stats = self.aggregator.get_performance_stats()
        
        # Time tracked should be close to our mocked values
        self.assertEqual(stats['avg_aggregate_time'], 1.0)

    def test_edge_cases(self):
        """Test edge cases."""
        # Test with no volume column
        no_volume_data = self.tick_data.drop(columns=['volume'])
        
        # Should still work and use default volume of 1
        bars = self.aggregator.aggregate_to_bars(no_volume_data, timeframe='1m')
        self.assertGreater(len(bars), 0)
        
        # Test with timestamp in columns rather than index
        tick_data_reset = self.tick_data.reset_index()
        bars = self.aggregator.aggregate_to_bars(tick_data_reset, timeframe='1m')
        self.assertGreater(len(bars), 0)


if __name__ == '__main__':
    unittest.main()