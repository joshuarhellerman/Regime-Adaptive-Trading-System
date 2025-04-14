import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data.fetchers.exchange_connector import (
    ExchangeConnector, 
    MarketType, 
    DataType, 
    DataFrequency
)
from data.processors.data_normalizer import DataNormalizer
from data.processors.data_integrity import DataIntegrityValidator
from execution.exchange.connectivity_manager import ConnectivityManager
from core.event_bus import EventBus

class TestExchangeConnector(unittest.TestCase):
    """Test cases for the ExchangeConnector class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mocks for dependencies
        self.mock_connectivity_manager = MagicMock(spec=ConnectivityManager)
        self.mock_data_normalizer = MagicMock(spec=DataNormalizer)
        self.mock_event_bus = MagicMock(spec=EventBus)
        self.mock_event_bus.publish = AsyncMock()
        
        # Create a mock for DataIntegrityValidator
        self.mock_validator = MagicMock(spec=DataIntegrityValidator)
        
        # Basic configuration for testing
        self.config = {
            'cache_ttl': 30,
            'default_market': MarketType.EQUITY,
            'default_provider': {
                'equity': 'test_provider',
                'crypto': 'crypto_provider',
            },
            'markets': {
                'equity': {
                    'enabled': True,
                    'providers': {
                        'test_provider': {
                            'api_key': 'test_key',
                        }
                    },
                    'default_provider': 'test_provider'
                },
                'crypto': {
                    'enabled': True,
                    'providers': {
                        'crypto_provider': {
                            'api_key': 'crypto_key',
                        }
                    }
                }
            }
        }
        
        # Patch the _init_market_adapters method to avoid actual adapter initialization
        with patch.object(ExchangeConnector, '_init_market_adapters'):
            self.connector = ExchangeConnector(
                self.mock_connectivity_manager,
                self.mock_data_normalizer,
                self.mock_event_bus,
                self.config
            )
        
        # Mock the market adapters that would be created by _init_market_adapters
        self.mock_equity_adapter = AsyncMock()
        self.mock_crypto_adapter = AsyncMock()
        
        self.connector.market_adapters = {
            MarketType.EQUITY: self.mock_equity_adapter,
            MarketType.CRYPTO: self.mock_crypto_adapter
        }
        
        # Replace the data validator with our mock
        self.connector.data_validator = self.mock_validator

    @pytest.mark.asyncio
    async def test_get_historical_data_single_instrument(self):
        """Test fetching historical data for a single instrument."""
        # Setup test data
        instrument = "AAPL"
        frequency = DataFrequency.DAY_1
        start_time = datetime.now() - timedelta(days=30)
        end_time = datetime.now()
        
        # Mock return data
        mock_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [98, 99, 100],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200],
            'timestamp': [
                start_time, 
                start_time + timedelta(days=1),
                start_time + timedelta(days=2)
            ]
        })
        
        self.mock_equity_adapter.get_historical_data.return_value = mock_data
        self.mock_data_normalizer.normalize_ohlcv.return_value = mock_data
        self.mock_validator.validate_ohlcv.return_value = True
        
        # Execute the method
        result = await self.connector.get_historical_data(
            instrument,
            frequency,
            start_time,
            end_time,
            DataType.OHLCV,
            MarketType.EQUITY,
            'test_provider'
        )
        
        # Assertions
        self.mock_equity_adapter.get_historical_data.assert_called_once()
        self.mock_data_normalizer.normalize_ohlcv.assert_called_once_with(mock_data, 'test_provider')
        self.mock_validator.validate_ohlcv.assert_called_once()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)

    @pytest.mark.asyncio
    async def test_get_historical_data_multiple_instruments(self):
        """Test fetching historical data for multiple instruments."""
        # Setup test data
        instruments = ["AAPL", "MSFT"]
        frequency = DataFrequency.DAY_1
        start_time = datetime.now() - timedelta(days=30)
        end_time = datetime.now()
        
        # Mock return data for each instrument
        mock_data_aapl = pd.DataFrame({
            'open': [100, 101],
            'high': [105, 106],
            'low': [98, 99],
            'close': [103, 104],
            'volume': [1000, 1100],
            'timestamp': [start_time, start_time + timedelta(days=1)]
        })
        
        mock_data_msft = pd.DataFrame({
            'open': [200, 201],
            'high': [205, 206],
            'low': [198, 199],
            'close': [203, 204],
            'volume': [2000, 2100],
            'timestamp': [start_time, start_time + timedelta(days=1)]
        })
        
        # Configure the mock to return different data for each instrument
        self.mock_equity_adapter.get_historical_data.side_effect = [
            mock_data_aapl,
            mock_data_msft
        ]
        
        self.mock_data_normalizer.normalize_ohlcv.side_effect = [
            mock_data_aapl,
            mock_data_msft
        ]
        
        self.mock_validator.validate_ohlcv.return_value = True
        
        # Execute the method
        result = await self.connector.get_historical_data(
            instruments,
            frequency,
            start_time,
            end_time,
            DataType.OHLCV,
            MarketType.EQUITY,
            'test_provider'
        )
        
        # Assertions
        self.assertEqual(self.mock_equity_adapter.get_historical_data.call_count, 2)
        self.assertEqual(self.mock_data_normalizer.normalize_ohlcv.call_count, 2)
        self.assertEqual(self.mock_validator.validate_ohlcv.call_count, 2)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        self.assertIn("AAPL", result)
        self.assertIn("MSFT", result)
        self.assertEqual(len(result["AAPL"]), 2)
        self.assertEqual(len(result["MSFT"]), 2)

    @pytest.mark.asyncio
    async def test_get_historical_data_validation_failure(self):
        """Test handling of data validation failure in historical data."""
        # Setup test data
        instrument = "AAPL"
        frequency = DataFrequency.DAY_1
        start_time = datetime.now() - timedelta(days=30)
        end_time = datetime.now()
        
        # Mock return data
        mock_data = pd.DataFrame({
            'open': [100, 101],
            'high': [105, 106],
            'low': [98, 99],
            'close': [103, 104],
            'volume': [1000, 1100],
            'timestamp': [start_time, start_time + timedelta(days=1)]
        })
        
        self.mock_equity_adapter.get_historical_data.return_value = mock_data
        self.mock_data_normalizer.normalize_ohlcv.return_value = mock_data
        
        # Set validator to fail
        self.mock_validator.validate_ohlcv.return_value = False
        
        # Execute the method
        result = await self.connector.get_historical_data(
            instrument,
            frequency,
            start_time,
            end_time,
            DataType.OHLCV,
            MarketType.EQUITY,
            'test_provider'
        )
        
        # Assertions
        self.mock_equity_adapter.get_historical_data.assert_called_once()
        self.mock_data_normalizer.normalize_ohlcv.assert_called_once()
        self.mock_validator.validate_ohlcv.assert_called_once()
        # Should still return data even with validation failure (with warning)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)

    @pytest.mark.asyncio
    async def test_get_market_data(self):
        """Test fetching current market data."""
        # Setup test data
        instruments = ["AAPL", "MSFT"]
        
        # Mock return data
        mock_data = {
            "AAPL": {
                "bid": 150.0,
                "ask": 150.5,
                "last": 150.25,
                "volume": 10000,
                "timestamp": datetime.now().timestamp()
            },
            "MSFT": {
                "bid": 250.0,
                "ask": 250.5,
                "last": 250.25,
                "volume": 8000,
                "timestamp": datetime.now().timestamp()
            }
        }
        
        normalized_data = mock_data.copy()  # For simplicity, same structure
        
        self.mock_equity_adapter.get_market_data.return_value = mock_data
        self.mock_data_normalizer.normalize_ticker.return_value = normalized_data
        
        # Clear cache to ensure fresh fetch
        self.connector.data_cache = {}
        
        # Execute the method
        result = await self.connector.get_market_data(
            instruments,
            DataType.QUOTE,
            MarketType.EQUITY,
            'test_provider'
        )
        
        # Assertions
        self.mock_equity_adapter.get_market_data.assert_called_once()
        self.mock_data_normalizer.normalize_ticker.assert_called_once_with(mock_data, 'test_provider')
        self.assertEqual(result, normalized_data)
        
        # Verify cache was populated
        cache_key = f"market_data_equity_test_provider_quote_AAPL,MSFT"
        self.assertIn(cache_key, self.connector.data_cache)
        self.assertEqual(self.connector.data_cache[cache_key]["data"], normalized_data)

    @pytest.mark.asyncio
    async def test_get_market_data_from_cache(self):
        """Test retrieving market data from cache."""
        # Setup test data
        instruments = ["AAPL", "MSFT"]
        cache_key = f"market_data_equity_test_provider_quote_AAPL,MSFT"
        
        # Preset cache
        cached_data = {
            "AAPL": {"bid": 150.0, "ask": 150.5},
            "MSFT": {"bid": 250.0, "ask": 250.5}
        }
        
        self.connector.data_cache = {
            cache_key: {
                "data": cached_data,
                "timestamp": datetime.now().timestamp()  # Fresh timestamp
            }
        }
        
        # Execute the method
        result = await self.connector.get_market_data(
            instruments,
            DataType.QUOTE,
            MarketType.EQUITY,
            'test_provider'
        )
        
        # Assertions
        self.mock_equity_adapter.get_market_data.assert_not_called()
        self.mock_data_normalizer.normalize_ticker.assert_not_called()
        self.assertEqual(result, cached_data)

    @pytest.mark.asyncio
    async def test_subscribe_to_data(self):
        """Test subscribing to real-time data updates."""
        # Setup test data
        instruments = ["AAPL", "MSFT"]
        mock_callback = AsyncMock()
        adapter_sub_id = 123
        
        self.mock_equity_adapter.subscribe_to_data.return_value = adapter_sub_id
        
        # Execute the method
        subscription_id = await self.connector.subscribe_to_data(
            instruments,
            DataType.QUOTE,
            mock_callback,
            DataFrequency.MINUTE_1,
            MarketType.EQUITY,
            'test_provider'
        )
        
        # Assertions
        self.mock_equity_adapter.subscribe_to_data.assert_called_once()
        self.assertIsInstance(subscription_id, int)
        self.assertIn(subscription_id, self.connector.subscriptions)
        
        # Verify subscription details
        subscription = self.connector.subscriptions[subscription_id]
        self.assertEqual(subscription["adapter_sub_id"], adapter_sub_id)
        self.assertEqual(subscription["market_type"], MarketType.EQUITY)
        self.assertEqual(subscription["data_type"], DataType.QUOTE)
        self.assertEqual(subscription["instruments"], instruments)
        self.assertEqual(subscription["provider"], 'test_provider')

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        """Test unsubscribing from a data stream."""
        # Setup test data
        subscription_id = 1
        adapter_sub_id = 123
        
        # Add a test subscription
        self.connector.subscriptions = {
            subscription_id: {
                "adapter_sub_id": adapter_sub_id,
                "market_type": MarketType.EQUITY,
                "data_type": DataType.QUOTE,
                "instruments": ["AAPL"],
                "provider": 'test_provider'
            }
        }
        
        self.mock_equity_adapter.unsubscribe.return_value = True
        
        # Execute the method
        result = await self.connector.unsubscribe(subscription_id)
        
        # Assertions
        self.mock_equity_adapter.unsubscribe.assert_called_once_with(adapter_sub_id)
        self.assertTrue(result)
        self.assertNotIn(subscription_id, self.connector.subscriptions)

    @pytest.mark.asyncio
    async def test_unsubscribe_invalid_id(self):
        """Test unsubscribing with an invalid subscription ID."""
        # Setup - empty subscriptions
        self.connector.subscriptions = {}
        
        # Execute the method
        result = await self.connector.unsubscribe(999)
        
        # Assertions
        self.mock_equity_adapter.unsubscribe.assert_not_called()
        self.assertFalse(result)

    @pytest.mark.asyncio
    async def test_get_market_hours(self):
        """Test getting trading hours for a market."""
        # Mock return data
        mock_hours = {
            "is_open": True,
            "open_time": "09:30:00",
            "close_time": "16:00:00",
            "timezone": "America/New_York"
        }
        
        self.mock_equity_adapter.get_market_hours.return_value = mock_hours
        
        # Execute the method
        result = await self.connector.get_market_hours(
            MarketType.EQUITY,
            'test_provider'
        )
        
        # Assertions
        self.mock_equity_adapter.get_market_hours.assert_called_once_with('test_provider')
        self.assertEqual(result, mock_hours)

    @pytest.mark.asyncio
    async def test_get_instruments(self):
        """Test getting list of available instruments."""
        # Mock return data
        mock_instruments = [
            {"symbol": "AAPL", "name": "Apple Inc."},
            {"symbol": "MSFT", "name": "Microsoft Corporation"}
        ]
        
        self.mock_equity_adapter.get_instruments.return_value = mock_instruments
        
        # Execute the method
        result = await self.connector.get_instruments(
            MarketType.EQUITY,
            'test_provider'
        )
        
        # Assertions
        self.mock_equity_adapter.get_instruments.assert_called_once_with('test_provider')
        self.assertEqual(result, mock_instruments)

    @pytest.mark.asyncio
    async def test_get_exchange_info(self):
        """Test getting information about a market and its providers."""
        # Mock return data
        mock_info = {
            "name": "Test Exchange",
            "features": ["historical_data", "real_time_quotes"],
            "supported_instruments": ["stocks", "etfs"]
        }
        
        self.mock_equity_adapter.get_exchange_info.return_value = mock_info
        
        # Execute the method
        result = await self.connector.get_exchange_info(
            MarketType.EQUITY,
            'test_provider'
        )
        
        # Assertions
        self.mock_equity_adapter.get_exchange_info.assert_called_once_with('test_provider')
        self.assertEqual(result, mock_info)

    @pytest.mark.asyncio
    async def test_get_reference_data(self):
        """Test getting reference data for instruments."""
        # Setup test data
        instruments = ["AAPL", "MSFT"]
        
        # Mock return data
        mock_data = {
            "AAPL": {
                "dividends": [
                    {"date": "2023-02-10", "amount": 0.23},
                    {"date": "2023-05-12", "amount": 0.24}
                ]
            },
            "MSFT": {
                "dividends": [
                    {"date": "2023-02-15", "amount": 0.68},
                    {"date": "2023-05-17", "amount": 0.68}
                ]
            }
        }
        
        self.mock_equity_adapter.get_reference_data.return_value = mock_data
        
        # Execute the method
        result = await self.connector.get_reference_data(
            instruments,
            "dividends",
            MarketType.EQUITY,
            'test_provider'
        )
        
        # Assertions
        self.mock_equity_adapter.get_reference_data.assert_called_once_with(
            instruments, "dividends", 'test_provider'
        )
        self.assertEqual(result, mock_data)

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test initializing all market adapters."""
        # Execute the method
        await self.connector.initialize()
        
        # Assertions
        self.mock_equity_adapter.initialize.assert_called_once()
        self.mock_crypto_adapter.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_provider_fallback(self):
        """Test provider fallback mechanisms."""
        # Setup - remove default providers
        self.connector.default_provider = {}
        
        # Case 1: Explicitly provided provider
        result = await self.connector.get_market_data(
            "AAPL",
            DataType.QUOTE,
            MarketType.EQUITY,
            'explicit_provider'
        )
        self.mock_equity_adapter.get_market_data.assert_called_with(
            ["AAPL"], DataType.QUOTE, 'explicit_provider'
        )
        
        # Reset mock
        self.mock_equity_adapter.get_market_data.reset_mock()
        
        # Case 2: Default provider from market config
        result = await self.connector.get_market_data(
            "AAPL",
            DataType.QUOTE,
            MarketType.EQUITY
        )
        self.mock_equity_adapter.get_market_data.assert_called_with(
            ["AAPL"], DataType.QUOTE, 'test_provider'
        )

    @pytest.mark.asyncio
    async def test_error_handling_in_get_historical_data(self):
        """Test error handling in get_historical_data method."""
        # Setup
        self.mock_equity_adapter.get_historical_data.side_effect = Exception("Test error")
        
        # Execute with error catching
        with self.assertRaises(Exception):
            await self.connector.get_historical_data(
                "AAPL",
                DataFrequency.DAY_1,
                datetime.now() - timedelta(days=30),
                datetime.now(),
                DataType.OHLCV,
                MarketType.EQUITY
            )
        
        # Verify error event was published
        self.mock_event_bus.publish.assert_called_once()


if __name__ == '__main__':
    unittest.main()