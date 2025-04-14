"""
Unit tests for the market_data_service module.

This module contains tests for MarketDataService class functionality including:
- Initialization and configuration
- Data retrieval from different sources
- Subscription management
- Caching
- Data source selection logic
"""

import unittest
from unittest import mock
import pandas as pd
import numpy as np
from datetime import datetime
import threading
import time
import json

# Import module under test
from data.market_data_service import (
    MarketDataService,
    DataSource,
    DataTimeframe,
    MarketDataSubscription,
    get_market_data_service
)


class TestMarketDataService(unittest.TestCase):
    """Test cases for MarketDataService"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create mock fetchers
        self.mock_exchange_fetcher = mock.MagicMock()
        self.mock_historical_fetcher = mock.MagicMock()
        self.mock_alternative_fetcher = mock.MagicMock()

        # Set up sample data
        self.sample_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [98.0, 99.0, 100.0],
            'close': [103.0, 104.0, 105.0],
            'volume': [1000, 1100, 1200]
        }, index=pd.date_range(start='2023-01-01', periods=3, freq='D'))

        # Create test config
        self.test_config = {
            'use_cache': True,
            'cache_ttl': 60,
            'max_cache_items': 100,
            'default_throttle': 0.5,
            'update_interval': 0.1
        }

        # Create service with mocks
        self.service = MarketDataService(self.test_config)
        self.service.exchange_fetcher = self.mock_exchange_fetcher
        self.service.historical_fetcher = self.mock_historical_fetcher
        self.service.alternative_fetcher = self.mock_alternative_fetcher

        # Mock event bus
        self.mock_event_bus = mock.MagicMock()
        self.service._event_bus = self.mock_event_bus

    def tearDown(self):
        """Clean up after each test method"""
        if self.service._running:
            self.service.stop()

    def test_initialization(self):
        """Test initialization with config"""
        service = MarketDataService(self.test_config)

        self.assertEqual(service.use_cache, True)
        self.assertEqual(service.cache_ttl, 60)
        self.assertEqual(service.max_cache_items, 100)
        self.assertEqual(service.default_throttle, 0.5)
        self.assertEqual(service.update_interval, 0.1)
        self.assertFalse(service._running)

    def test_start_and_stop(self):
        """Test starting and stopping the service"""
        # Start service
        self.service.start()
        self.assertTrue(self.service._running)
        self.assertIsNotNone(self.service._update_thread)
        self.assertTrue(self.service._update_thread.is_alive())

        # Check event published
        self.mock_event_bus.publish.assert_called_once()

        # Stop service
        self.service.stop()
        self.assertFalse(self.service._running)
        time.sleep(0.2)  # Give thread time to stop
        self.assertFalse(self.service._update_thread.is_alive())

    def test_get_data_from_exchange(self):
        """Test getting data from exchange source"""
        symbol = "BTC/USD"
        timeframe = DataTimeframe.D1

        # Setup mock
        self.mock_exchange_fetcher.fetch_recent_data.return_value = self.sample_data

        # Call method
        result = self.service.get_data(
            symbol=symbol,
            timeframe=timeframe,
            source=DataSource.EXCHANGE
        )

        # Verify
        self.mock_exchange_fetcher.fetch_recent_data.assert_called_once_with(
            symbol=symbol,
            timeframe=timeframe.value,
            bars=100
        )
        pd.testing.assert_frame_equal(result, self.sample_data)

    def test_get_data_from_historical(self):
        """Test getting data from historical source"""
        symbol = "BTC/USD"
        timeframe = DataTimeframe.D1
        start_date = "2023-01-01"
        end_date = "2023-01-03"

        # Setup mock
        self.mock_historical_fetcher.fetch_data.return_value = self.sample_data

        # Call method
        result = self.service.get_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            source=DataSource.HISTORICAL
        )

        # Verify
        self.mock_historical_fetcher.fetch_data.assert_called_once_with(
            symbol=symbol,
            timeframe=timeframe.value,
            start_date=start_date,
            end_date=end_date,
            source='auto'
        )
        pd.testing.assert_frame_equal(result, self.sample_data)

    def test_get_data_from_alternative(self):
        """Test getting data from alternative source"""
        symbol = "BTC/USD"
        timeframe = DataTimeframe.D1
        start_date = "2023-01-01"
        end_date = "2023-01-03"

        # Setup mock
        self.mock_alternative_fetcher.fetch_data.return_value = self.sample_data

        # Call method
        result = self.service.get_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            source=DataSource.ALTERNATIVE
        )

        # Verify
        self.mock_alternative_fetcher.fetch_data.assert_called_once_with(
            symbol=symbol,
            timeframe=timeframe.value,
            start_date=start_date,
            end_date=end_date
        )
        pd.testing.assert_frame_equal(result, self.sample_data)

    def test_get_data_auto_source_current_data(self):
        """Test auto source selection for current data"""
        symbol = "BTC/USD"
        timeframe = DataTimeframe.D1

        # Setup mocks
        self.mock_exchange_fetcher.fetch_recent_data.return_value = self.sample_data
        self.mock_historical_fetcher.fetch_data.return_value = pd.DataFrame()

        # Call method
        result = self.service.get_data(
            symbol=symbol,
            timeframe=timeframe,
            source=DataSource.AUTO
        )

        # Verify exchange was used
        self.mock_exchange_fetcher.fetch_recent_data.assert_called_once()
        pd.testing.assert_frame_equal(result, self.sample_data)

    def test_get_data_auto_source_historical_data(self):
        """Test auto source selection for historical data"""
        symbol = "BTC/USD"
        timeframe = DataTimeframe.D1
        start_date = "2020-01-01"
        end_date = "2020-01-10"  # Past date

        # Setup mocks
        self.mock_historical_fetcher.fetch_data.return_value = self.sample_data

        # Call method
        result = self.service.get_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            source=DataSource.AUTO
        )

        # Verify historical was used
        self.mock_historical_fetcher.fetch_data.assert_called_once()
        pd.testing.assert_frame_equal(result, self.sample_data)

    def test_get_data_combined_source(self):
        """Test combined data source"""
        symbol = "BTC/USD"
        timeframe = DataTimeframe.D1

        # Setup different data for each source
        exchange_data = pd.DataFrame({
            'open': [110.0, 111.0],
            'high': [115.0, 116.0],
            'low': [108.0, 109.0],
            'close': [113.0, 114.0],
            'volume': [1300, 1400]
        }, index=pd.date_range(start='2023-01-04', periods=2, freq='D'))

        historical_data = self.sample_data

        alternative_data = pd.DataFrame({
            'open': [120.0],
            'high': [125.0],
            'low': [118.0],
            'close': [123.0],
            'volume': [1500]
        }, index=pd.date_range(start='2023-01-06', periods=1, freq='D'))

        # Setup mocks
        self.mock_exchange_fetcher.fetch_recent_data.return_value = exchange_data
        self.mock_historical_fetcher.fetch_data.return_value = historical_data
        self.mock_alternative_fetcher.fetch_data.return_value = alternative_data

        # Call method
        result = self.service.get_data(
            symbol=symbol,
            timeframe=timeframe,
            source=DataSource.COMBINED
        )

        # Verify all sources were used
        self.mock_exchange_fetcher.fetch_recent_data.assert_called_once()
        self.mock_historical_fetcher.fetch_data.assert_called_once()
        self.mock_alternative_fetcher.fetch_data.assert_called_once()

        # Combined data should have entries from all sources (6 days)
        self.assertEqual(len(result), 6)

    def test_data_cache(self):
        """Test data caching functionality"""
        symbol = "BTC/USD"
        timeframe = DataTimeframe.D1

        # Setup mock
        self.mock_exchange_fetcher.fetch_recent_data.return_value = self.sample_data

        # First call should use the fetcher
        result1 = self.service.get_data(
            symbol=symbol,
            timeframe=timeframe,
            source=DataSource.EXCHANGE,
            cache=True
        )

        # Second call should use cache
        self.mock_exchange_fetcher.fetch_recent_data.reset_mock()
        result2 = self.service.get_data(
            symbol=symbol,
            timeframe=timeframe,
            source=DataSource.EXCHANGE,
            cache=True
        )

        # Verify
        self.mock_exchange_fetcher.fetch_recent_data.assert_not_called()
        pd.testing.assert_frame_equal(result1, result2)

        # Test cache bypass
        self.mock_exchange_fetcher.fetch_recent_data.return_value = self.sample_data
        self.service.get_data(
            symbol=symbol,
            timeframe=timeframe,
            source=DataSource.EXCHANGE,
            cache=False
        )
        self.mock_exchange_fetcher.fetch_recent_data.assert_called_once()

        # Test cache clearing
        self.service.clear_cache()
        self.assertEqual(len(self.service._cache), 0)

    def test_subscribe_unsubscribe(self):
        """Test subscription management"""
        symbol = "BTC/USD"
        timeframe = DataTimeframe.D1
        callback = mock.MagicMock()

        # Setup
        self.mock_exchange_fetcher.fetch_recent_data.return_value = self.sample_data

        # Subscribe
        subscription_id = self.service.subscribe(
            symbol=symbol,
            timeframe=timeframe,
            callback=callback,
            source=DataSource.EXCHANGE
        )

        # Verify subscription created
        self.assertIn(subscription_id, self.service._subscriptions)
        subscription = self.service._subscriptions[subscription_id]
        self.assertEqual(subscription.symbol, symbol)
        self.assertEqual(subscription.timeframe, timeframe.value)

        # Verify callback was called with initial data
        callback.assert_called_once()
        pd.testing.assert_frame_equal(
            callback.call_args[0][0],  # First positional arg
            self.sample_data
        )

        # Unsubscribe
        result = self.service.unsubscribe(subscription_id)
        self.assertTrue(result)
        self.assertNotIn(subscription_id, self.service._subscriptions)

        # Test unsubscribe nonexistent subscription
        result = self.service.unsubscribe("nonexistent-id")
        self.assertFalse(result)

    def test_get_subscription(self):
        """Test getting subscription information"""
        symbol = "BTC/USD"
        timeframe = DataTimeframe.D1
        callback = mock.MagicMock()

        # Create subscription
        self.mock_exchange_fetcher.fetch_recent_data.return_value = self.sample_data
        subscription_id = self.service.subscribe(
            symbol=symbol,
            timeframe=timeframe,
            callback=callback
        )

        # Get subscription info
        info = self.service.get_subscription(subscription_id)

        # Verify
        self.assertEqual(info['id'], subscription_id)
        self.assertEqual(info['symbol'], symbol)
        self.assertEqual(info['timeframe'], timeframe.value)
        self.assertTrue(info['active'])

        # Test nonexistent subscription
        info = self.service.get_subscription("nonexistent-id")
        self.assertIsNone(info)

        # Cleanup
        self.service.unsubscribe(subscription_id)

    def test_get_all_subscriptions(self):
        """Test getting all subscriptions"""
        # Create multiple subscriptions
        self.mock_exchange_fetcher.fetch_recent_data.return_value = self.sample_data

        sub_id1 = self.service.subscribe(
            symbol="BTC/USD",
            timeframe=DataTimeframe.D1,
            callback=mock.MagicMock()
        )

        sub_id2 = self.service.subscribe(
            symbol="ETH/USD",
            timeframe=DataTimeframe.H1,
            callback=mock.MagicMock()
        )

        # Get all subscriptions
        subscriptions = self.service.get_all_subscriptions()

        # Verify
        self.assertEqual(len(subscriptions), 2)
        subscription_ids = [sub['id'] for sub in subscriptions]
        self.assertIn(sub_id1, subscription_ids)
        self.assertIn(sub_id2, subscription_ids)

        # Cleanup
        self.service.unsubscribe(sub_id1)
        self.service.unsubscribe(sub_id2)

    def test_update_subscription(self):
        """Test subscription update mechanism"""
        symbol = "BTC/USD"
        timeframe = DataTimeframe.D1
        callback = mock.MagicMock()

        # Create subscription with custom throttle
        options = {'throttle': 0.1}
        self.mock_exchange_fetcher.fetch_recent_data.return_value = self.sample_data

        subscription_id = self.service.subscribe(
            symbol=symbol,
            timeframe=timeframe,
            callback=callback,
            options=options
        )

        # First update happens immediately (in subscribe)
        callback.reset_mock()

        # Get subscription object
        subscription = self.service._subscriptions[subscription_id]

        # Force update
        self.service._update_subscription(subscription)

        # Should not update due to throttle
        self.assertEqual(callback.call_count, 0)

        # Wait for throttle to expire
        time.sleep(0.2)

        # Update again
        self.service._update_subscription(subscription)

        # Should update now
        self.assertEqual(callback.call_count, 1)

        # Cleanup
        self.service.unsubscribe(subscription_id)

    def test_get_available_symbols(self):
        """Test getting available symbols"""
        # Setup mocks
        self.mock_historical_fetcher.get_available_symbols.return_value = ["BTC/USD", "ETH/USD"]
        self.mock_exchange_fetcher.get_supported_exchanges.return_value = ["binance", "coinbase"]
        self.mock_exchange_fetcher.get_supported_symbols.side_effect = [
            ["BTC/USD", "LTC/USD"],  # binance
            ["ETH/USD", "XRP/USD"]   # coinbase
        ]
        self.mock_alternative_fetcher.get_available_symbols.return_value = ["BTC/USD", "AAPL"]

        # Get symbols from all sources
        symbols = self.service.get_available_symbols(DataSource.AUTO)

        # Verify
        expected = ["AAPL", "BTC/USD", "ETH/USD", "LTC/USD", "XRP/USD"]
        self.assertEqual(sorted(symbols), expected)

        # Test specific source
        symbols = self.service.get_available_symbols(DataSource.HISTORICAL)
        self.assertEqual(sorted(symbols), ["BTC/USD", "ETH/USD"])

    def test_get_available_timeframes(self):
        """Test getting available timeframes"""
        symbol = "BTC/USD"

        # Setup mocks
        self.mock_historical_fetcher.get_available_timeframes.return_value = ["1h", "4h", "1d"]
        self.mock_alternative_fetcher.get_available_timeframes.return_value = ["1d", "1w"]

        # Get timeframes from all sources
        timeframes = self.service.get_available_timeframes(symbol, DataSource.AUTO)

        # All standard timeframes plus those from historical and alternative
        self.assertIn("1m", timeframes)  # From standard enum
        self.assertIn("1h", timeframes)  # From historical
        self.assertIn("1w", timeframes)  # From alternative

    def test_save_data(self):
        """Test saving market data"""
        symbol = "BTC/USD"
        timeframe = DataTimeframe.D1

        # Setup mocks
        self.mock_historical_fetcher.save_to_csv.return_value = True
        self.mock_historical_fetcher.save_to_db.return_value = True

        # Save data
        result = self.service.save_data(
            data=self.sample_data,
            symbol=symbol,
            timeframe=timeframe,
            append=True
        )

        # Verify
        self.assertTrue(result)
        self.mock_historical_fetcher.save_to_csv.assert_called_once_with(
            self.sample_data, symbol, timeframe.value, True
        )
        self.mock_historical_fetcher.save_to_db.assert_called_once_with(
            self.sample_data, symbol, timeframe.value
        )

        # Test empty data
        result = self.service.save_data(
            data=pd.DataFrame(),
            symbol=symbol,
            timeframe=timeframe
        )
        self.assertFalse(result)

    def test_get_status(self):
        """Test getting service status"""
        # Setup
        self.mock_exchange_fetcher.get_supported_exchanges.return_value = ["binance", "coinbase"]
        self.service._running = True

        # Get status
        status = self.service.get_status()

        # Verify
        self.assertTrue(status['running'])
        self.assertEqual(status['cache_size'], 0)
        self.assertEqual(status['subscriptions'], 0)
        self.assertEqual(status['supported_exchanges'], ["binance", "coinbase"])
        self.assertTrue(status['alternative_data_available'])  # Mock is not None

    def test_market_data_event_handling(self):
        """Test handling market data events"""
        # Create a subscription
        symbol = "BTC/USD"
        timeframe = "1m"
        callback = mock.MagicMock()

        self.mock_exchange_fetcher.fetch_recent_data.return_value = pd.DataFrame()
        subscription_id = self.service.subscribe(
            symbol=symbol,
            timeframe=timeframe,
            callback=callback
        )

        callback.reset_mock()

        # Create event
        event_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'data': self.sample_data
        }
        event = mock.MagicMock()
        event.data = event_data

        # Handle event
        self.service._handle_market_data_event(event)

        # Verify callback was called
        callback.assert_called_once()
        pd.testing.assert_frame_equal(
            callback.call_args[0][0],
            self.sample_data
        )

        # Cleanup
        self.service.unsubscribe(subscription_id)

    def test_get_market_depth(self):
        """Test getting market depth data"""
        symbol = "BTC/USD"
        mock_depth = {
            'bids': [[19000, 1.5], [18990, 2.1]],
            'asks': [[19010, 1.2], [19020, 3.0]]
        }

        # Setup mock
        self.mock_exchange_fetcher.fetch_order_book.return_value = mock_depth

        # Get market depth
        result = self.service.get_market_depth(symbol)

        # Verify
        self.assertEqual(result, mock_depth)
        self.mock_exchange_fetcher.fetch_order_book.assert_called_once_with(symbol)

    def test_get_ticker(self):
        """Test getting ticker data"""
        symbol = "BTC/USD"
        mock_ticker = {
            'last': 19000,
            'bid': 18990,
            'ask': 19010,
            'volume': 1000
        }

        # Setup mock
        self.mock_exchange_fetcher.fetch_ticker.return_value = mock_ticker

        # Get ticker
        result = self.service.get_ticker(symbol)

        # Verify
        self.assertEqual(result, mock_ticker)
        self.mock_exchange_fetcher.fetch_ticker.assert_called_once_with(symbol)

    def test_get_recent_trades(self):
        """Test getting recent trades"""
        symbol = "BTC/USD"
        limit = 50
        mock_trades = pd.DataFrame({
            'price': [19000, 19005, 19010],
            'amount': [0.1, 0.2, 0.15],
            'side': ['buy', 'sell', 'buy']
        })

        # Setup mock
        self.mock_exchange_fetcher.fetch_trades.return_value = mock_trades

        # Get recent trades
        result = self.service.get_recent_trades(symbol, limit)

        # Verify
        pd.testing.assert_frame_equal(result, mock_trades)
        self.mock_exchange_fetcher.fetch_trades.assert_called_once_with(symbol, limit=limit)

    def test_singleton_pattern(self):
        """Test the singleton pattern implementation"""
        # Get first instance
        service1 = get_market_data_service(self.test_config)

        # Get second instance with different config
        # Should return same instance and ignore new config
        service2 = get_market_data_service({'use_cache': False})

        # Verify
        self.assertIs(service1, service2)
        self.assertEqual(service2.use_cache, True)  # Should keep original config


if __name__ == '__main__':
    unittest.main()