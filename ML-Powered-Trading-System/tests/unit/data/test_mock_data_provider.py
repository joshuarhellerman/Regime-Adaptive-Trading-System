import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import random

from data.fetchers.mock_data_provider import (
    MockDataProvider,
    DataGenerationMode,
    MarketScenario
)


class TestMockDataProvider(unittest.TestCase):
    """Unit tests for the MockDataProvider class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Set seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Default test parameters
        self.symbols = ["BTC-USD", "ETH-USD"]
        self.start_time = datetime(2023, 1, 1, 12, 0, 0)
        self.time_delta = timedelta(seconds=1)
        
        # Create sample historical data for replay mode tests
        self.historical_data = pd.DataFrame({
            'symbol': ['BTC-USD', 'BTC-USD', 'ETH-USD', 'ETH-USD'],
            'timestamp': [
                self.start_time.timestamp(),
                (self.start_time + self.time_delta).timestamp(),
                self.start_time.timestamp(),
                (self.start_time + self.time_delta).timestamp()
            ],
            'open': [40000.0, 40100.0, 2900.0, 2950.0],
            'high': [40200.0, 40300.0, 3000.0, 3050.0],
            'low': [39800.0, 39900.0, 2850.0, 2900.0],
            'close': [40100.0, 40200.0, 2950.0, 3000.0],
            'volume': [100.0, 120.0, 50.0, 60.0]
        })

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        provider = MockDataProvider()
        
        # Check default attributes
        self.assertEqual(provider.mode, DataGenerationMode.RANDOM_WALK)
        self.assertEqual(provider.scenario, MarketScenario.NORMAL)
        self.assertIsNone(provider.historical_data)
        self.assertIsNone(provider.custom_generator)
        self.assertEqual(len(provider.symbols), 5)  # Default provides 5 symbols
        self.assertEqual(provider.time_delta, timedelta(seconds=1))
        self.assertEqual(provider.noise_level, 0.001)
        self.assertFalse(provider.latency_simulation)
        
        # Check if prices are initialized
        self.assertEqual(len(provider.last_prices), len(provider.symbols))

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        provider = MockDataProvider(
            mode=DataGenerationMode.SCENARIO,
            scenario=MarketScenario.BULLISH,
            symbols=self.symbols,
            start_time=self.start_time,
            time_delta=self.time_delta,
            noise_level=0.005,
            latency_simulation=True,
            random_seed=42
        )
        
        # Check custom attributes
        self.assertEqual(provider.mode, DataGenerationMode.SCENARIO)
        self.assertEqual(provider.scenario, MarketScenario.BULLISH)
        self.assertEqual(provider.symbols, self.symbols)
        self.assertEqual(provider.start_time, self.start_time)
        self.assertEqual(provider.current_time, self.start_time)
        self.assertEqual(provider.time_delta, self.time_delta)
        self.assertEqual(provider.noise_level, 0.005)
        self.assertTrue(provider.latency_simulation)

    def test_init_validation_replay_mode(self):
        """Test validation error when using REPLAY mode without historical data."""
        with self.assertRaises(ValueError):
            MockDataProvider(mode=DataGenerationMode.REPLAY)

    def test_init_validation_custom_mode(self):
        """Test validation error when using CUSTOM mode without custom generator."""
        with self.assertRaises(ValueError):
            MockDataProvider(mode=DataGenerationMode.CUSTOM)

    def test_initialize_prices(self):
        """Test initialization of prices for symbols."""
        provider = MockDataProvider(symbols=self.symbols)
        
        # Check if prices are initialized for provided symbols
        for symbol in self.symbols:
            self.assertIn(symbol, provider.last_prices)
            self.assertGreater(provider.last_prices[symbol], 0)

    def test_get_data_random_walk(self):
        """Test get_data method in RANDOM_WALK mode."""
        provider = MockDataProvider(
            mode=DataGenerationMode.RANDOM_WALK,
            symbols=self.symbols,
            start_time=self.start_time,
            random_seed=42
        )
        
        data = provider.get_data()
        
        # Check structure of returned data
        for symbol in self.symbols:
            self.assertIn(symbol, data)
            self.assertIn('ohlcv', data[symbol])
            self.assertIn('orderbook', data[symbol])
            self.assertIn('trades', data[symbol])
            
            # Check OHLCV fields
            ohlcv = data[symbol]['ohlcv']
            self.assertIn('timestamp', ohlcv)
            self.assertIn('open', ohlcv)
            self.assertIn('high', ohlcv)
            self.assertIn('low', ohlcv)
            self.assertIn('close', ohlcv)
            self.assertIn('volume', ohlcv)
            
            # Check high >= open >= low
            self.assertGreaterEqual(ohlcv['high'], ohlcv['open'])
            self.assertGreaterEqual(ohlcv['open'], ohlcv['low'])
            
            # Check orderbook
            orderbook = data[symbol]['orderbook']
            self.assertIn('bids', orderbook)
            self.assertIn('asks', orderbook)
            self.assertGreater(len(orderbook['bids']), 0)
            self.assertGreater(len(orderbook['asks']), 0)
            
            # Check all bids are lower than all asks (no crossed book)
            max_bid = max(bid[0] for bid in orderbook['bids'])
            min_ask = min(ask[0] for ask in orderbook['asks'])
            self.assertLess(max_bid, min_ask)

    def test_get_data_replay(self):
        """Test get_data method in REPLAY mode."""
        provider = MockDataProvider(
            mode=DataGenerationMode.REPLAY,
            historical_data=self.historical_data,
            symbols=self.symbols,
            start_time=self.start_time
        )
        
        data = provider.get_data()
        
        # Check that the data matches the historical data
        for symbol in self.symbols:
            self.assertIn(symbol, data)
            ohlcv = data[symbol]['ohlcv']
            
            # Get corresponding row from historical data
            hist_row = self.historical_data[self.historical_data['symbol'] == symbol].iloc[0]
            
            # Check OHLCV values
            self.assertEqual(ohlcv['open'], hist_row['open'])
            self.assertEqual(ohlcv['high'], hist_row['high'])
            self.assertEqual(ohlcv['low'], hist_row['low'])
            self.assertEqual(ohlcv['close'], hist_row['close'])
            self.assertEqual(ohlcv['volume'], hist_row['volume'])

    def test_get_data_scenario(self):
        """Test get_data method in SCENARIO mode."""
        # Test with BULLISH scenario
        provider = MockDataProvider(
            mode=DataGenerationMode.SCENARIO,
            scenario=MarketScenario.BULLISH,
            symbols=self.symbols,
            start_time=self.start_time,
            random_seed=42
        )
        
        initial_data = provider.get_data()
        initial_prices = {symbol: data['ohlcv']['close'] for symbol, data in initial_data.items()}
        
        # Forward time and get data multiple times
        prices_history = []
        for _ in range(10):
            data = provider.get_data()
            prices = {symbol: data['ohlcv']['close'] for symbol, data in data.items()}
            prices_history.append(prices)
        
        # In a bullish scenario, we expect prices to trend upward overall
        # Calculate how many times prices increased vs decreased
        increases = 0
        decreases = 0
        for symbol in self.symbols:
            for i in range(1, len(prices_history)):
                if prices_history[i][symbol] > prices_history[i-1][symbol]:
                    increases += 1
                else:
                    decreases += 1
        
        # In a bullish scenario, increases should outnumber decreases
        self.assertGreater(increases, decreases)

    @patch('time.sleep')
    def test_simulate_latency(self, mock_sleep):
        """Test latency simulation."""
        provider = MockDataProvider(
            symbols=self.symbols,
            latency_simulation=True,
            random_seed=42
        )
        
        # Get data to trigger latency simulation
        provider.get_data()
        
        # Check if sleep was called
        mock_sleep.assert_called()

    def test_reset(self):
        """Test reset method."""
        provider = MockDataProvider(
            symbols=self.symbols,
            start_time=self.start_time
        )
        
        # Store initial prices
        initial_prices = provider.last_prices.copy()
        initial_time = provider.current_time
        
        # Get data to advance time and change prices
        for _ in range(5):
            provider.get_data()
        
        # Confirm changes
        self.assertNotEqual(provider.current_time, initial_time)
        
        # Reset
        provider.reset()
        
        # Check if reset worked
        self.assertEqual(provider.current_time, initial_time)
        for symbol in self.symbols:
            self.assertIn(symbol, provider.last_prices)

    def test_set_scenario(self):
        """Test changing scenario during simulation."""
        provider = MockDataProvider(
            mode=DataGenerationMode.SCENARIO,
            scenario=MarketScenario.NORMAL,
            symbols=self.symbols
        )
        
        # Change scenario
        provider.set_scenario(MarketScenario.FLASH_CRASH)
        
        # Check if scenario was changed
        self.assertEqual(provider.scenario, MarketScenario.FLASH_CRASH)

    def test_set_mode(self):
        """Test changing mode during simulation."""
        provider = MockDataProvider(
            mode=DataGenerationMode.RANDOM_WALK,
            symbols=self.symbols
        )
        
        # Create custom generator function
        def custom_generator(symbols, current_time):
            result = {}
            for symbol in symbols:
                result[symbol] = {
                    'ohlcv': {
                        'timestamp': current_time.timestamp(),
                        'open': 100.0,
                        'high': 110.0,
                        'low': 90.0,
                        'close': 105.0,
                        'volume': 1000.0
                    },
                    'orderbook': {'bids': [[99, 1]], 'asks': [[101, 1]], 'timestamp': current_time.timestamp()},
                    'trades': []
                }
            return result
        
        # Change mode
        provider.set_mode(
            DataGenerationMode.CUSTOM, 
            custom_generator=custom_generator
        )
        
        # Check if mode was changed
        self.assertEqual(provider.mode, DataGenerationMode.CUSTOM)
        self.assertEqual(provider.custom_generator, custom_generator)
        
        # Get data with new mode
        data = provider.get_data()
        
        # Check data matches our custom generator
        for symbol in self.symbols:
            self.assertEqual(data[symbol]['ohlcv']['close'], 105.0)

    def test_get_data_with_custom_generator(self):
        """Test get_data with a custom generator function."""
        # Define custom generator function
        def custom_generator(symbols, current_time):
            result = {}
            for symbol in symbols:
                result[symbol] = {
                    'ohlcv': {
                        'timestamp': current_time.timestamp(),
                        'open': 200.0,
                        'high': 220.0,
                        'low': 190.0,
                        'close': 210.0,
                        'volume': 500.0
                    },
                    'orderbook': {'bids': [[199, 1]], 'asks': [[201, 1]], 'timestamp': current_time.timestamp()},
                    'trades': []
                }
            return result
        
        provider = MockDataProvider(
            mode=DataGenerationMode.CUSTOM,
            custom_generator=custom_generator,
            symbols=self.symbols
        )
        
        data = provider.get_data()
        
        # Check if custom generator was used
        for symbol in self.symbols:
            self.assertEqual(data[symbol]['ohlcv']['open'], 200.0)
            self.assertEqual(data[symbol]['ohlcv']['close'], 210.0)

    def test_generate_orderbook(self):
        """Test _generate_orderbook private method."""
        provider = MockDataProvider(symbols=self.symbols)
        
        # Test with regular scenario
        orderbook = provider._generate_orderbook("BTC-USD", 40000.0)
        
        # Check structure
        self.assertIn('bids', orderbook)
        self.assertIn('asks', orderbook)
        self.assertIn('timestamp', orderbook)
        
        # Test with liquidity crisis scenario
        orderbook = provider._generate_orderbook("BTC-USD", 40000.0, MarketScenario.LIQUIDITY_CRISIS)
        
        # In liquidity crisis, we expect wider spreads
        bid_prices = [bid[0] for bid in orderbook['bids']]
        ask_prices = [ask[0] for ask in orderbook['asks']]
        
        max_bid = max(bid_prices)
        min_ask = min(ask_prices)
        
        # Check if spread is wider
        self.assertGreater((min_ask - max_bid) / 40000.0, 0.005)  # Spread should be > 0.5%

    def test_generate_trades(self):
        """Test _generate_trades private method."""
        provider = MockDataProvider(symbols=self.symbols)
        
        # Test with regular scenario
        trades = provider._generate_trades("BTC-USD", 40000.0)
        
        # Check that trades have correct structure
        for trade in trades:
            self.assertIn('timestamp', trade)
            self.assertIn('price', trade)
            self.assertIn('quantity', trade)
            self.assertIn('side', trade)
            self.assertIn(trade['side'], ['buy', 'sell'])
        
        # Test with volatile scenario that should generate more trades
        trades_volatile = provider._generate_trades("BTC-USD", 40000.0, MarketScenario.VOLATILE)
        
        # We can't assert exact numbers due to randomness, but can check general pattern
        # over multiple runs
        total_regular = 0
        total_volatile = 0
        
        for _ in range(10):
            total_regular += len(provider._generate_trades("BTC-USD", 40000.0))
            total_volatile += len(provider._generate_trades("BTC-USD", 40000.0, MarketScenario.VOLATILE))
        
        # Volatile should generate more trades on average
        self.assertGreater(total_volatile, total_regular)

    def test_historical_data_end(self):
        """Test behavior when reaching the end of historical data."""
        # Create small historical dataset
        small_hist_data = pd.DataFrame({
            'symbol': ['BTC-USD', 'ETH-USD'],
            'timestamp': [self.start_time.timestamp(), self.start_time.timestamp()],
            'price': [40000.0, 3000.0],
            'volume': [100.0, 50.0]
        })
        
        provider = MockDataProvider(
            mode=DataGenerationMode.REPLAY,
            historical_data=small_hist_data,
            symbols=self.symbols,
            start_time=self.start_time
        )
        
        # Get initial data
        initial_data = provider.get_data()
        
        # Advance past the end of the historical data
        # This should cause the provider to loop back to the beginning
        for _ in range(5):
            data = provider.get_data()
        
        # Get data again
        final_data = provider.get_data()
        
        # Data should still be available
        for symbol in self.symbols:
            self.assertIn(symbol, final_data)
            self.assertIn('ohlcv', final_data[symbol])


if __name__ == "__main__":
    unittest.main()