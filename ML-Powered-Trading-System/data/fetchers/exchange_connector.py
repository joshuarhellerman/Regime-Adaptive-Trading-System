"""
data/fetchers/exchange_connector.py - Market-Agnostic Exchange Connector

This module provides a unified interface for accessing market data across
different market types (equities, futures, forex, etc.) and data providers.
"""

import logging
import time
import asyncio
import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
import json

from core.event_bus import EventBus, EventTopics, create_event
from data.processors.data_normalizer import DataNormalizer
from data.processors.data_integrity import DataIntegrityValidator
from execution.exchange.connectivity_manager import ConnectivityManager
from utils.logger import get_logger

class MarketType(Enum):
    """Types of financial markets"""
    EQUITY = "equity"
    FUTURES = "futures"
    FOREX = "forex"
    FIXED_INCOME = "fixed_income"
    CRYPTO = "crypto"
    OPTIONS = "options"

class DataFrequency(Enum):
    """Standard time frequencies for market data"""
    TICK = "tick"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"

class DataType(Enum):
    """Types of market data that can be fetched"""
    OHLCV = "ohlcv"              # Open, High, Low, Close, Volume
    DEPTH = "depth"              # Order book depth
    QUOTE = "quote"              # Latest quotes
    TRADE = "trade"              # Individual trades
    FUNDAMENTAL = "fundamental"  # Fundamental data
    REFERENCE = "reference"      # Reference data (dividends, splits, etc.)

class ExchangeConnector:
    """
    Market-agnostic interface for accessing financial market data.

    This class provides a unified interface for fetching data from different
    market types (equities, futures, forex, etc.) and data providers, with
    consistent data formats regardless of the source.
    """

    def __init__(
        self,
        connectivity_manager: ConnectivityManager,
        data_normalizer: DataNormalizer,
        event_bus: EventBus,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the exchange connector.

        Args:
            connectivity_manager: Manages connectivity to exchanges
            data_normalizer: Normalizes data from different sources
            event_bus: System-wide event bus for publishing events
            config: Configuration for the exchange connector
        """
        self.logger = get_logger(__name__)
        self.connectivity_manager = connectivity_manager
        self.normalizer = data_normalizer
        self.event_bus = event_bus
        self.config = config or {}
        self.data_validator = DataIntegrityValidator()

        # Subscription management
        self.subscriptions = {}
        self.subscription_counter = 0

        # Cached data
        self.data_cache = {}
        self.cache_ttl = self.config.get('cache_ttl', 60)  # seconds

        # Market adapters
        self.market_adapters = {}

        # Default settings
        self.default_market = self.config.get('default_market', MarketType.EQUITY)
        self.default_provider = self.config.get('default_provider', {})

        # Initialize market adapters
        self._init_market_adapters()

        self.logger.info(f"Exchange connector initialized with {len(self.market_adapters)} market adapters")

    def _init_market_adapters(self):
        """
        Initialize market-specific adapters based on configuration.

        Creates adapter instances for each configured market type.
        """
        markets_config = self.config.get('markets', {})

        for market_name, market_config in markets_config.items():
            try:
                # Skip if disabled
                if not market_config.get('enabled', True):
                    continue

                market_type = MarketType(market_name)

                # Determine adapter class based on market type
                if market_type == MarketType.EQUITY:
                    from data.fetchers.adapters.equity_adapter import EquityAdapter
                    adapter = EquityAdapter(market_type, market_config, self.connectivity_manager)
                elif market_type == MarketType.FUTURES:
                    from data.fetchers.adapters.futures_adapter import FuturesAdapter
                    adapter = FuturesAdapter(market_type, market_config, self.connectivity_manager)
                elif market_type == MarketType.FOREX:
                    from data.fetchers.adapters.forex_adapter import ForexAdapter
                    adapter = ForexAdapter(market_type, market_config, self.connectivity_manager)
                elif market_type == MarketType.FIXED_INCOME:
                    from data.fetchers.adapters.fixed_income_adapter import FixedIncomeAdapter
                    adapter = FixedIncomeAdapter(market_type, market_config, self.connectivity_manager)
                elif market_type == MarketType.CRYPTO:
                    from data.fetchers.adapters.crypto_adapter import CryptoAdapter
                    adapter = CryptoAdapter(market_type, market_config, self.connectivity_manager)
                elif market_type == MarketType.OPTIONS:
                    from data.fetchers.adapters.options_adapter import OptionsAdapter
                    adapter = OptionsAdapter(market_type, market_config, self.connectivity_manager)
                else:
                    # Use a generic adapter as fallback
                    from data.fetchers.adapters.base_adapter import BaseMarketAdapter
                    adapter = BaseMarketAdapter(market_type, market_config, self.connectivity_manager)

                # Store the adapter
                self.market_adapters[market_type] = adapter
                self.logger.info(f"Initialized adapter for {market_type.value} market")

            except Exception as e:
                self.logger.error(f"Failed to initialize adapter for {market_name}: {str(e)}")

    def _get_market_adapter(self, market_type: Optional[MarketType] = None):
        """
        Get the appropriate market adapter.

        Args:
            market_type: Type of financial market

        Returns:
            Market adapter instance

        Raises:
            ValueError: If market type is not supported
        """
        if market_type is None:
            market_type = self.default_market

        if market_type not in self.market_adapters:
            raise ValueError(f"Market type {market_type.value} not supported")

        return self.market_adapters[market_type]

    def _get_provider_for_market(self, market_type: MarketType, provider: Optional[str] = None):
        """
        Get the default provider for a market if none specified.

        Args:
            market_type: Type of financial market
            provider: Data provider name

        Returns:
            Provider name
        """
        if provider is not None:
            return provider

        # Check if there's a default provider for this market
        if market_type.value in self.default_provider:
            return self.default_provider[market_type.value]

        # Get market config
        market_config = self.config.get('markets', {}).get(market_type.value, {})
        providers = market_config.get('providers', {})

        # If only one provider, use it
        if len(providers) == 1:
            return list(providers.keys())[0]

        # If there's a default provider in the market config, use it
        if 'default_provider' in market_config:
            return market_config['default_provider']

        # No provider found
        return None

    async def get_historical_data(
        self,
        instruments: Union[str, List[str]],
        frequency: Union[str, DataFrequency],
        start_time: Union[datetime, int],
        end_time: Union[datetime, int] = None,
        data_type: DataType = DataType.OHLCV,
        market_type: Optional[MarketType] = None,
        provider: Optional[str] = None,
        **kwargs
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Fetch historical data for instrument(s).

        Args:
            instruments: Single instrument or list of instruments
            frequency: Data frequency (e.g., "1m", "1h", "1d")
            start_time: Start time as datetime or timestamp
            end_time: End time (default: current time)
            data_type: Type of data to fetch
            market_type: Type of financial market
            provider: Data provider to use
            **kwargs: Additional parameters for specific markets

        Returns:
            DataFrame with historical data, or dict mapping instruments to DataFrames
        """
        # Convert single instrument to list
        single_instrument = isinstance(instruments, str)
        instruments_list = [instruments] if single_instrument else instruments

        # Convert frequency enum to string if needed
        if isinstance(frequency, DataFrequency):
            frequency = frequency.value

        # Get market adapter
        adapter = self._get_market_adapter(market_type)

        # Get provider
        provider = self._get_provider_for_market(adapter.market_type, provider)
        if provider is None:
            raise ValueError(f"No provider specified or found for {adapter.market_type.value} market")

        try:
            # Fetch data for all instruments
            results = {}

            for instrument in instruments_list:
                data = await adapter.get_historical_data(
                    instrument,
                    frequency,
                    start_time,
                    end_time,
                    data_type,
                    provider,
                    **kwargs
                )

                # Normalize the data
                if data_type == DataType.OHLCV:
                    normalized_data = self.normalizer.normalize_ohlcv(data, provider)

                    # Validate OHLCV data
                    if not self.data_validator.validate_ohlcv(normalized_data):
                        self.logger.warning(f"OHLCV data validation failed for {instrument}")
                else:
                    # For other data types, use the raw data (normalization should be added later)
                    normalized_data = data

                results[instrument] = normalized_data

            # Return single DataFrame for single instrument
            if single_instrument:
                return results[instruments]

            return results

        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}")
            await self._publish_error_event(
                "error.data.historical",
                str(market_type),
                str(instruments),
                str(e)
            )
            raise

    async def get_market_data(
        self,
        instruments: Union[str, List[str]],
        data_type: DataType = DataType.QUOTE,
        market_type: Optional[MarketType] = None,
        provider: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fetch current market data for instrument(s).

        Args:
            instruments: Single instrument or list of instruments
            data_type: Type of data to fetch (quote, depth, etc.)
            market_type: Type of financial market
            provider: Data provider to use
            **kwargs: Additional parameters for specific markets

        Returns:
            Dict with market data
        """
        # Convert single instrument to list
        instruments_list = [instruments] if isinstance(instruments, str) else instruments

        # Get market adapter
        adapter = self._get_market_adapter(market_type)

        # Get provider
        provider = self._get_provider_for_market(adapter.market_type, provider)
        if provider is None:
            raise ValueError(f"No provider specified or found for {adapter.market_type.value} market")

        # Check cache first
        cache_key = f"market_data_{adapter.market_type.value}_{provider}_{data_type.value}_{','.join(instruments_list)}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data

        try:
            # Fetch data
            data = await adapter.get_market_data(
                instruments_list,
                data_type,
                provider,
                **kwargs
            )

            # Normalize the data based on data type
            if data_type == DataType.QUOTE:
                normalized_data = self.normalizer.normalize_ticker(data, provider)
            elif data_type == DataType.DEPTH:
                normalized_data = self.normalizer.normalize_orderbook(data, provider)
            else:
                # For other data types, use the raw data (normalization should be added later)
                normalized_data = data

            # Cache the data
            self._add_to_cache(cache_key, normalized_data)

            return normalized_data

        except Exception as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            await self._publish_error_event(
                "error.data.market",
                str(market_type),
                str(instruments),
                str(e)
            )
            raise

    async def subscribe_to_data(
        self,
        instruments: Union[str, List[str]],
        data_type: DataType,
        callback: Callable[[Dict[str, Any]], None],
        frequency: Optional[Union[str, DataFrequency]] = None,
        market_type: Optional[MarketType] = None,
        provider: Optional[str] = None,
        **kwargs
    ) -> int:
        """
        Subscribe to real-time data updates.

        Args:
            instruments: Single instrument or list of instruments
            data_type: Type of data to subscribe to
            callback: Callback function for updates
            frequency: Data frequency (for OHLCV updates)
            market_type: Type of financial market
            provider: Data provider to use
            **kwargs: Additional parameters for specific markets

        Returns:
            Subscription ID
        """
        # Convert single instrument to list
        instruments_list = [instruments] if isinstance(instruments, str) else instruments

        # Convert frequency enum to string if needed
        if isinstance(frequency, DataFrequency):
            frequency = frequency.value

        # Get market adapter
        adapter = self._get_market_adapter(market_type)

        # Get provider
        provider = self._get_provider_for_market(adapter.market_type, provider)
        if provider is None:
            raise ValueError(f"No provider specified or found for {adapter.market_type.value} market")

        # Create normalized callback
        async def normalized_callback(data):
            try:
                # Normalize the data based on data type
                if data_type == DataType.OHLCV:
                    normalized_data = self.normalizer.normalize_ohlcv_update(data, provider)
                elif data_type == DataType.QUOTE:
                    normalized_data = self.normalizer.normalize_ticker_update(data, provider)
                elif data_type == DataType.DEPTH:
                    normalized_data = self.normalizer.normalize_orderbook_update(data, provider)
                elif data_type == DataType.TRADE:
                    normalized_data = self.normalizer.normalize_trade_update(data, provider)
                else:
                    # For other data types, use the raw data
                    normalized_data = data

                # Call the user callback
                await callback(normalized_data)

            except Exception as e:
                self.logger.error(f"Error in subscription callback: {str(e)}")

        try:
            # Subscribe to data
            adapter_sub_id = await adapter.subscribe_to_data(
                instruments_list,
                data_type,
                normalized_callback,
                frequency,
                provider,
                **kwargs
            )

            # Generate a system-wide subscription ID
            self.subscription_counter += 1
            subscription_id = self.subscription_counter

            # Store the subscription details
            self.subscriptions[subscription_id] = {
                "adapter_sub_id": adapter_sub_id,
                "market_type": adapter.market_type,
                "data_type": data_type,
                "instruments": instruments_list,
                "provider": provider
            }

            self.logger.info(f"Created subscription {subscription_id} for {instruments_list}")
            return subscription_id

        except Exception as e:
            self.logger.error(f"Error subscribing to data: {str(e)}")
            await self._publish_error_event(
                "error.data.subscription",
                str(market_type),
                str(instruments),
                str(e)
            )
            raise

    async def unsubscribe(self, subscription_id: int) -> bool:
        """
        Unsubscribe from a data stream.

        Args:
            subscription_id: Subscription ID

        Returns:
            True if successful, False otherwise
        """
        if subscription_id not in self.subscriptions:
            self.logger.warning(f"Subscription {subscription_id} not found")
            return False

        subscription = self.subscriptions[subscription_id]
        market_type = subscription["market_type"]
        adapter_sub_id = subscription["adapter_sub_id"]

        # Get market adapter
        adapter = self._get_market_adapter(market_type)

        try:
            # Unsubscribe
            result = await adapter.unsubscribe(adapter_sub_id)

            # Remove subscription
            del self.subscriptions[subscription_id]

            self.logger.info(f"Unsubscribed from subscription {subscription_id}")
            return result

        except Exception as e:
            self.logger.error(f"Error unsubscribing from {subscription_id}: {str(e)}")
            return False

    async def get_market_hours(
        self,
        market_type: Optional[MarketType] = None,
        provider: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get trading hours for a market.

        Args:
            market_type: Type of financial market
            provider: Data provider to use
            **kwargs: Additional parameters for specific markets

        Returns:
            Dict with market hours information
        """
        # Get market adapter
        adapter = self._get_market_adapter(market_type)

        # Get provider
        provider = self._get_provider_for_market(adapter.market_type, provider)
        if provider is None:
            raise ValueError(f"No provider specified or found for {adapter.market_type.value} market")

        try:
            # Get market hours
            hours = await adapter.get_market_hours(provider, **kwargs)
            return hours

        except Exception as e:
            self.logger.error(f"Error getting market hours: {str(e)}")
            raise

    async def get_instruments(
        self,
        market_type: Optional[MarketType] = None,
        provider: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Get list of available instruments for a market.

        Args:
            market_type: Type of financial market
            provider: Data provider to use
            **kwargs: Additional parameters for specific markets

        Returns:
            List of instrument information
        """
        # Get market adapter
        adapter = self._get_market_adapter(market_type)

        # Get provider
        provider = self._get_provider_for_market(adapter.market_type, provider)
        if provider is None:
            raise ValueError(f"No provider specified or found for {adapter.market_type.value} market")

        try:
            # Get instruments
            instruments = await adapter.get_instruments(provider, **kwargs)
            return instruments

        except Exception as e:
            self.logger.error(f"Error getting instruments: {str(e)}")
            raise

    async def get_exchange_info(
        self,
        market_type: Optional[MarketType] = None,
        provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get information about a market and its providers.

        Args:
            market_type: Type of financial market
            provider: Data provider to use

        Returns:
            Dict with market and provider information
        """
        # Get market adapter
        adapter = self._get_market_adapter(market_type)

        # Get provider
        provider = self._get_provider_for_market(adapter.market_type, provider)

        try:
            # Get exchange info
            info = await adapter.get_exchange_info(provider)
            return info

        except Exception as e:
            self.logger.error(f"Error getting exchange info: {str(e)}")
            raise

    async def get_reference_data(
        self,
        instruments: Union[str, List[str]],
        data_type: str,
        market_type: Optional[MarketType] = None,
        provider: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get reference data for instruments.

        Args:
            instruments: Single instrument or list of instruments
            data_type: Type of reference data (dividends, splits, etc.)
            market_type: Type of financial market
            provider: Data provider to use
            **kwargs: Additional parameters for specific markets

        Returns:
            Dict with reference data
        """
        # Convert single instrument to list
        instruments_list = [instruments] if isinstance(instruments, str) else instruments

        # Get market adapter
        adapter = self._get_market_adapter(market_type)

        # Get provider
        provider = self._get_provider_for_market(adapter.market_type, provider)
        if provider is None:
            raise ValueError(f"No provider specified or found for {adapter.market_type.value} market")

        try:
            # Get reference data
            data = await adapter.get_reference_data(
                instruments_list,
                data_type,
                provider,
                **kwargs
            )
            return data

        except Exception as e:
            self.logger.error(f"Error getting reference data: {str(e)}")
            raise

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """
        Get data from the cache.

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found or expired
        """
        if key not in self.data_cache:
            return None

        entry = self.data_cache[key]
        if time.time() - entry["timestamp"] > self.cache_ttl:
            # Cache entry has expired
            del self.data_cache[key]
            return None

        return entry["data"]

    def _add_to_cache(self, key: str, data: Any) -> None:
        """
        Add data to the cache.

        Args:
            key: Cache key
            data: Data to cache
        """
        self.data_cache[key] = {
            "data": data,
            "timestamp": time.time()
        }

    async def _publish_error_event(
        self,
        event_type: str,
        market: str,
        instruments: str,
        error: str
    ) -> None:
        """
        Publish an error event to the event bus.

        Args:
            event_type: Type of error event
            market: Market type
            instruments: Instruments
            error: Error message
        """
        try:
            event = create_event(
                event_type,
                {
                    "market": market,
                    "instruments": instruments,
                    "error": error,
                    "timestamp": time.time()
                }
            )
            await self.event_bus.publish(event)
        except Exception as e:
            self.logger.error(f"Error publishing error event: {str(e)}")

    async def initialize(self) -> None:
        """
        Initialize the exchange connector.

        This should be called before using the connector to ensure all
        market adapters are properly initialized.
        """
        # Initialize all market adapters
        for market_type, adapter in self.market_adapters.items():
            try:
                await adapter.initialize()
                self.logger.info(f"Initialized adapter for {market_type.value}")
            except Exception as e:
                self.logger.error(f"Error initializing adapter for {market_type.value}: {str(e)}")