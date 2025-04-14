"""
Mock Data Provider

This module provides mock market data for testing trading strategies without
connecting to live exchanges. It simulates various market scenarios and can
replay historical data with optional modifications.

Dependencies:
- data.processors.data_normalizer.py: For ensuring consistent data format
"""

import logging
import time
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from enum import Enum

# Import the data normalizer to ensure consistent data format
from data.processors.data_normalizer import normalize_market_data

logger = logging.getLogger(__name__)


class DataGenerationMode(Enum):
    """Enum for different modes of generating mock data."""
    REPLAY = "replay"          # Replay historical data as-is
    RANDOM_WALK = "random_walk"  # Generate using random walk
    SCENARIO = "scenario"      # Use pre-defined market scenarios
    CUSTOM = "custom"          # Use custom data generation function


class MarketScenario(Enum):
    """Pre-defined market scenarios for simulation."""
    NORMAL = "normal"               # Normal market conditions
    BULLISH = "bullish"             # Strong upward trend
    BEARISH = "bearish"             # Strong downward trend
    VOLATILE = "volatile"           # High volatility
    FLASH_CRASH = "flash_crash"     # Sudden market crash
    RECOVERY = "recovery"           # Recovery after a crash
    SIDEWAYS = "sideways"           # Sideways/ranging market
    LIQUIDITY_CRISIS = "liquidity"  # Low liquidity conditions


class MockDataProvider:
    """
    Provides mock market data for testing trading strategies.

    This class simulates market data in various ways:
    1. Replaying historical data (with optional modifications)
    2. Generating synthetic data using random walks
    3. Simulating specific market scenarios
    4. Using custom data generation functions
    """

    def __init__(self,
                 mode: DataGenerationMode = DataGenerationMode.RANDOM_WALK,
                 historical_data: Optional[pd.DataFrame] = None,
                 scenario: MarketScenario = MarketScenario.NORMAL,
                 custom_generator: Optional[Callable] = None,
                 symbols: List[str] = None,
                 start_time: Optional[datetime] = None,
                 time_delta: timedelta = timedelta(seconds=1),
                 noise_level: float = 0.001,
                 latency_simulation: bool = False,
                 random_seed: Optional[int] = None):
        """
        Initialize the MockDataProvider.

        Args:
            mode: The data generation mode to use
            historical_data: Historical data to replay (required for REPLAY mode)
            scenario: Market scenario to simulate (for SCENARIO mode)
            custom_generator: Custom function for generating data (for CUSTOM mode)
            symbols: List of symbols to generate data for
            start_time: Start time for the simulation (default: current time)
            time_delta: Time interval between data points
            noise_level: Level of noise to add to generated data
            latency_simulation: Whether to simulate network latency
            random_seed: Seed for random number generation (for reproducibility)
        """
        self.mode = mode
        self.historical_data = historical_data
        self.scenario = scenario
        self.custom_generator = custom_generator
        self.symbols = symbols or ["BTC-USD", "ETH-USD", "SOL-USD", "AAPL", "MSFT"]
        self.start_time = start_time or datetime.now()
        self.current_time = self.start_time
        self.time_delta = time_delta
        self.noise_level = noise_level
        self.latency_simulation = latency_simulation

        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # Initialize price data for each symbol
        self.last_prices = {}
        self.initialize_prices()

        # Validate the configuration
        self._validate_config()

        logger.info(f"MockDataProvider initialized in {mode.value} mode with {len(self.symbols)} symbols")

    def _validate_config(self):
        """Validate the configuration based on the selected mode."""
        if self.mode == DataGenerationMode.REPLAY and self.historical_data is None:
            raise ValueError("Historical data must be provided when using REPLAY mode")

        if self.mode == DataGenerationMode.CUSTOM and self.custom_generator is None:
            raise ValueError("Custom generator function must be provided when using CUSTOM mode")

    def initialize_prices(self):
        """Initialize starting prices for each symbol."""
        # Define realistic starting prices for common assets
        default_prices = {
            "BTC-USD": 45000.0,
            "ETH-USD": 3000.0,
            "SOL-USD": 100.0,
            "AAPL": 150.0,
            "MSFT": 300.0,
            "AMZN": 130.0,
            "GOOGL": 140.0,
            "META": 330.0,
            "TSLA": 200.0,
            "NVDA": 450.0,
        }

        for symbol in self.symbols:
            # Use default price if available, otherwise generate a random price
            if symbol in default_prices:
                self.last_prices[symbol] = default_prices[symbol]
            else:
                # Generate a random price between 10 and 1000
                self.last_prices[symbol] = random.uniform(10, 1000)

    def get_data(self,
                 symbols: Optional[List[str]] = None,
                 data_types: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get mock market data for the specified symbols and data types.

        Args:
            symbols: List of symbols to get data for (default: all initialized symbols)
            data_types: List of data types to include (default: ['ohlcv', 'orderbook', 'trades'])

        Returns:
            Dictionary containing mock market data for each symbol and data type
        """
        symbols = symbols or self.symbols
        data_types = data_types or ['ohlcv', 'orderbook', 'trades']

        # Simulate network latency if enabled
        if self.latency_simulation:
            self._simulate_latency()

        # Update the current time
        self.current_time += self.time_delta

        # Generate data based on the selected mode
        if self.mode == DataGenerationMode.REPLAY:
            raw_data = self._replay_historical_data(symbols)
        elif self.mode == DataGenerationMode.SCENARIO:
            raw_data = self._generate_scenario_data(symbols)
        elif self.mode == DataGenerationMode.CUSTOM:
            raw_data = self.custom_generator(symbols, self.current_time)
        else:  # Default to RANDOM_WALK
            raw_data = self._generate_random_walk_data(symbols)

        # Normalize the data to ensure consistent format
        normalized_data = normalize_market_data(raw_data)

        return normalized_data

    def _simulate_latency(self):
        """Simulate network latency by adding random delays."""
        # Simulate normal latency (10-50ms)
        normal_latency = random.uniform(0.01, 0.05)

        # Occasionally simulate high latency (100-500ms) with 5% probability
        if random.random() < 0.05:
            high_latency = random.uniform(0.1, 0.5)
            time.sleep(high_latency)
            logger.debug(f"Simulated high latency: {high_latency:.3f}s")
        else:
            time.sleep(normal_latency)

    def _generate_random_walk_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Generate market data using a random walk model.

        Args:
            symbols: List of symbols to generate data for

        Returns:
            Dictionary containing generated market data
        """
        result = {}

        for symbol in symbols:
            # Get the last price for this symbol
            last_price = self.last_prices[symbol]

            # Generate a random price change (random walk)
            # The change is proportional to the price and the noise level
            price_change = last_price * self.noise_level * np.random.normal(0, 1)

            # Calculate the new price
            new_price = max(0.01, last_price + price_change)

            # Save the new price
            self.last_prices[symbol] = new_price

            # Generate OHLCV data
            timestamp = self.current_time.timestamp()
            open_price = last_price
            high_price = max(open_price, new_price) * (1 + random.uniform(0, self.noise_level))
            low_price = min(open_price, new_price) * (1 - random.uniform(0, self.noise_level))
            close_price = new_price
            volume = random.uniform(10, 1000) * last_price / 100

            # Create data structure
            result[symbol] = {
                'ohlcv': {
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                },
                'orderbook': self._generate_orderbook(symbol, new_price),
                'trades': self._generate_trades(symbol, new_price)
            }

        return result

    def _replay_historical_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Replay historical data for the specified symbols.

        Args:
            symbols: List of symbols to get data for

        Returns:
            Dictionary containing historical market data
        """
        # Get the current index in the historical data based on time
        time_diff = (self.current_time - self.start_time).total_seconds()
        index = int(time_diff / self.time_delta.total_seconds())

        # Check if we've reached the end of the historical data
        if index >= len(self.historical_data):
            logger.warning("Reached the end of historical data, restarting from beginning")
            index = index % len(self.historical_data)
            self.start_time = self.current_time

        result = {}

        for symbol in symbols:
            try:
                # Get data for this symbol at the current index
                symbol_data = self.historical_data[self.historical_data['symbol'] == symbol].iloc[index]

                # Extract OHLCV data
                ohlcv = {
                    'timestamp': self.current_time.timestamp(),
                    'open': symbol_data.get('open', symbol_data.get('price', 0)),
                    'high': symbol_data.get('high', symbol_data.get('price', 0)),
                    'low': symbol_data.get('low', symbol_data.get('price', 0)),
                    'close': symbol_data.get('close', symbol_data.get('price', 0)),
                    'volume': symbol_data.get('volume', 0)
                }

                # Use the close price for orderbook and trades
                price = ohlcv['close']

                # Save the price
                self.last_prices[symbol] = price

                result[symbol] = {
                    'ohlcv': ohlcv,
                    'orderbook': self._generate_orderbook(symbol, price),
                    'trades': self._generate_trades(symbol, price)
                }
            except (IndexError, KeyError) as e:
                logger.warning(f"Error retrieving historical data for {symbol}: {e}")
                # Fall back to random walk if historical data is not available
                result[symbol] = self._generate_random_walk_data([symbol])[symbol]

        return result

    def _generate_scenario_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Generate market data based on the selected scenario.

        Args:
            symbols: List of symbols to generate data for

        Returns:
            Dictionary containing scenario-based market data
        """
        result = {}

        # Calculate time elapsed since start (used for scenario progression)
        time_elapsed = (self.current_time - self.start_time).total_seconds()

        for symbol in symbols:
            # Get the last price for this symbol
            last_price = self.last_prices[symbol]

            # Apply scenario-specific price modifications
            if self.scenario == MarketScenario.NORMAL:
                # Normal market: mild random walk
                drift = 0.0
                volatility = 1.0

            elif self.scenario == MarketScenario.BULLISH:
                # Bullish market: positive drift
                drift = 0.01
                volatility = 1.0

            elif self.scenario == MarketScenario.BEARISH:
                # Bearish market: negative drift
                drift = -0.01
                volatility = 1.0

            elif self.scenario == MarketScenario.VOLATILE:
                # Volatile market: higher volatility
                drift = 0.0
                volatility = 3.0

            elif self.scenario == MarketScenario.FLASH_CRASH:
                # Flash crash: sudden drop followed by partial recovery
                crash_duration = 60  # seconds
                if time_elapsed < crash_duration:
                    # Calculate crash progress (0 to 1)
                    progress = time_elapsed / crash_duration
                    # Sharp initial drop (up to 20%)
                    drift = -0.2 * (1 - progress)
                    volatility = 2.0
                else:
                    # After crash: mild recovery
                    drift = 0.005
                    volatility = 1.0

            elif self.scenario == MarketScenario.RECOVERY:
                # Recovery: gradual rise after drop
                drift = 0.008
                volatility = 0.8

            elif self.scenario == MarketScenario.SIDEWAYS:
                # Sideways: no drift, low volatility
                drift = 0.0
                volatility = 0.5

            elif self.scenario == MarketScenario.LIQUIDITY_CRISIS:
                # Liquidity crisis: increased spreads, reduced depth
                drift = -0.005
                volatility = 1.5

            else:
                # Default behavior
                drift = 0.0
                volatility = 1.0

            # Apply drift and volatility to calculate price change
            price_change = (drift + volatility * self.noise_level * np.random.normal(0, 1)) * last_price

            # Calculate the new price
            new_price = max(0.01, last_price + price_change)

            # Save the new price
            self.last_prices[symbol] = new_price

            # Generate OHLCV data
            timestamp = self.current_time.timestamp()
            open_price = last_price
            high_price = max(open_price, new_price) * (1 + random.uniform(0, self.noise_level * volatility))
            low_price = min(open_price, new_price) * (1 - random.uniform(0, self.noise_level * volatility))
            close_price = new_price

            # Volume varies by scenario
            volume_multiplier = 1.0
            if self.scenario == MarketScenario.VOLATILE:
                volume_multiplier = 2.0
            elif self.scenario == MarketScenario.FLASH_CRASH:
                volume_multiplier = 5.0 if time_elapsed < 60 else 3.0
            elif self.scenario == MarketScenario.LIQUIDITY_CRISIS:
                volume_multiplier = 0.3

            volume = random.uniform(10, 1000) * last_price / 100 * volume_multiplier

            # Create data structure
            result[symbol] = {
                'ohlcv': {
                    'timestamp': timestamp,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                },
                'orderbook': self._generate_orderbook(symbol, new_price, self.scenario),
                'trades': self._generate_trades(symbol, new_price, self.scenario)
            }

        return result

    def _generate_orderbook(self,
                            symbol: str,
                            price: float,
                            scenario: Optional[MarketScenario] = None) -> Dict[str, Any]:
        """
        Generate a realistic orderbook for a given price.

        Args:
            symbol: The trading symbol
            price: The current price
            scenario: Optional scenario to adjust orderbook characteristics

        Returns:
            Dictionary containing orderbook data
        """
        # Default spread and depth parameters
        spread_pct = 0.001  # 0.1% spread
        depth_levels = 10
        base_quantity = price * 0.1  # Base quantity scales with price

        # Adjust parameters based on scenario
        if scenario == MarketScenario.VOLATILE:
            spread_pct = 0.002
            depth_decay = 1.2
        elif scenario == MarketScenario.FLASH_CRASH:
            spread_pct = 0.005
            depth_decay = 1.5
        elif scenario == MarketScenario.LIQUIDITY_CRISIS:
            spread_pct = 0.01
            depth_levels = 5
            depth_decay = 2.0
        else:
            depth_decay = 1.1  # Each level has 1.1x less quantity than previous

        # Calculate bid and ask prices
        spread_amount = price * spread_pct
        best_bid = price - spread_amount / 2
        best_ask = price + spread_amount / 2

        # Generate bid levels
        bids = []
        for i in range(depth_levels):
            bid_price = best_bid * (1 - 0.001 * i)
            quantity = base_quantity / (depth_decay ** i) * (1 + random.uniform(-0.1, 0.1))
            bids.append([bid_price, quantity])

        # Generate ask levels
        asks = []
        for i in range(depth_levels):
            ask_price = best_ask * (1 + 0.001 * i)
            quantity = base_quantity / (depth_decay ** i) * (1 + random.uniform(-0.1, 0.1))
            asks.append([ask_price, quantity])

        return {
            'timestamp': self.current_time.timestamp(),
            'bids': bids,
            'asks': asks
        }

    def _generate_trades(self,
                          symbol: str,
                          price: float,
                          scenario: Optional[MarketScenario] = None) -> List[Dict[str, Any]]:
        """
        Generate mock trades around the current price.

        Args:
            symbol: The trading symbol
            price: The current price
            scenario: Optional scenario to adjust trade characteristics

        Returns:
            List of dictionaries containing trade data
        """
        # Determine number of trades to generate (randomly 0-5)
        num_trades = random.randint(0, 5)

        # Adjust trade frequency based on scenario
        if scenario == MarketScenario.VOLATILE or scenario == MarketScenario.FLASH_CRASH:
            num_trades = random.randint(3, 10)
        elif scenario == MarketScenario.LIQUIDITY_CRISIS:
            num_trades = random.randint(0, 2)

        trades = []

        for _ in range(num_trades):
            # Generate trade price with small variation around current price
            trade_price = price * (1 + random.uniform(-0.001, 0.001))

            # Generate trade size (larger for higher-priced assets)
            trade_size = random.uniform(0.1, 10) * (10000 / price)

            # Determine if it's a buy or sell
            is_buy = random.random() > 0.5

            trades.append({
                'timestamp': self.current_time.timestamp() - random.uniform(0, self.time_delta.total_seconds()),
                'price': trade_price,
                'quantity': trade_size,
                'side': 'buy' if is_buy else 'sell'
            })

        return trades

    def reset(self):
        """Reset the mock data provider to its initial state."""
        self.current_time = self.start_time
        self.initialize_prices()
        logger.info("MockDataProvider reset to initial state")

    def set_scenario(self, scenario: MarketScenario):
        """
        Change the market scenario during simulation.

        Args:
            scenario: The new market scenario to simulate
        """
        self.scenario = scenario
        logger.info(f"Scenario changed to: {scenario.value}")

    def set_mode(self, mode: DataGenerationMode, **kwargs):
        """
        Change the data generation mode during simulation.

        Args:
            mode: The new data generation mode
            **kwargs: Additional parameters required for the new mode
        """
        self.mode = mode

        # Update relevant parameters based on the new mode
        if mode == DataGenerationMode.REPLAY and 'historical_data' in kwargs:
            self.historical_data = kwargs['historical_data']

        if mode == DataGenerationMode.SCENARIO and 'scenario' in kwargs:
            self.scenario = kwargs['scenario']

        if mode == DataGenerationMode.CUSTOM and 'custom_generator' in kwargs:
            self.custom_generator = kwargs['custom_generator']

        # Validate the new configuration
        self._validate_config()

        logger.info(f"Mode changed to: {mode.value}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create a mock data provider with default settings
    provider = MockDataProvider()

    # Get data for Bitcoin and Ethereum
    data = provider.get_data(symbols=["BTC-USD", "ETH-USD"])

    # Print the data
    for symbol, symbol_data in data.items():
        print(f"Symbol: {symbol}")
        print(f"OHLCV: {symbol_data['ohlcv']}")
        print(f"Orderbook: {len(symbol_data['orderbook']['bids'])} bids, {len(symbol_data['orderbook']['asks'])} asks")
        print(f"Trades: {len(symbol_data['trades'])} trades")
        print()

    # Example of changing scenario
    print("Changing to FLASH_CRASH scenario...")
    provider.set_scenario(MarketScenario.FLASH_CRASH)

    # Get data again
    data = provider.get_data()

    # Print just the prices
    for symbol, symbol_data in data.items():
        price = symbol_data['ohlcv']['close']
        print(f"{symbol}: {price:.2f}")