"""
market_data_service.py - Unified Market Data Interface

This module provides a centralized service for accessing all market data,
with support for real-time and historical data from various sources,
data normalization, and caching.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from enum import Enum
import threading
import time
import asyncio
from pathlib import Path
import json
import os

from data.fetchers.exchange_connector import ExchangeFetcher, ExchangeType, create_data_fetcher
from data.fetchers.historical_repository import HistoricalFetcher, create_historical_fetcher
from data.fetchers.alternative_data_gateway import AlternativeDataGateway, create_alternative_data_fetcher
from data.processors.data_normalizer import normalize_market_data
from core.event_bus import EventTopics, create_event, get_event_bus, Event, EventPriority

logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Supported data sources"""
    EXCHANGE = "exchange"      # Live exchange data
    HISTORICAL = "historical"  # Historical data from files/DB
    ALTERNATIVE = "alternative"  # Alternative data sources
    COMBINED = "combined"      # Combined data from multiple sources
    AUTO = "auto"              # Automatically select best source


class DataTimeframe(Enum):
    """Standard timeframes for market data"""
    TICK = "tick"      # Tick data (no specific timeframe)
    M1 = "1m"          # 1 minute
    M5 = "5m"          # 5 minutes
    M15 = "15m"        # 15 minutes
    M30 = "30m"        # 30 minutes
    H1 = "1h"          # 1 hour
    H4 = "4h"          # 4 hours
    D1 = "1d"          # 1 day
    W1 = "1w"          # 1 week
    MN1 = "1M"         # 1 month

    @classmethod
    def from_string(cls, timeframe_str: str) -> 'DataTimeframe':
        """Convert string to DataTimeframe enum"""
        for tf in cls:
            if tf.value == timeframe_str:
                return tf
        raise ValueError(f"Invalid timeframe: {timeframe_str}")


class MarketDataSubscription:
    """Class representing a market data subscription"""
    def __init__(self,
                 symbol: str,
                 timeframe: Union[str, DataTimeframe],
                 callback: Callable[[pd.DataFrame], None],
                 source: DataSource = DataSource.AUTO,
                 options: Dict[str, Any] = None):
        self.id = f"{symbol}_{timeframe}_{source.value}_{id(self)}"
        self.symbol = symbol
        self.timeframe = timeframe.value if isinstance(timeframe, DataTimeframe) else timeframe
        self.callback = callback
        self.source = source
        self.options = options or {}
        self.active = True
        self.last_update = None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, MarketDataSubscription):
            return False
        return self.id == other.id

    def deactivate(self):
        """Deactivate the subscription"""
        self.active = False


class MarketDataService:
    """
    Centralized service for accessing market data from all sources.

    This service provides a unified interface for:
    - Real-time market data from exchanges
    - Historical data from files/databases
    - Alternative data sources
    - Normalized and merged data from multiple sources
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the market data service.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._lock = threading.RLock()
        self._subscriptions: Dict[str, MarketDataSubscription] = {}
        self._subscription_lock = threading.RLock()
        self._update_thread = None
        self._running = False
        self._cache: Dict[str, Tuple[pd.DataFrame, float]] = {}
        self._event_bus = get_event_bus()

        # Cache settings
        self.use_cache = self.config.get('use_cache', True)
        self.cache_ttl = self.config.get('cache_ttl', 300)  # seconds
        self.max_cache_items = self.config.get('max_cache_items', 500)

        # Rate limiting
        self.default_throttle = self.config.get('default_throttle', 1.0)  # seconds
        self.update_interval = self.config.get('update_interval', 1.0)  # seconds

        # Initialize data fetchers
        self._init_data_fetchers()

        # Register event handlers
        self._event_bus.subscribe(
            EventTopics.MARKET_DATA,
            self._handle_market_data_event
        )

        logger.info("Market Data Service initialized")

    def _init_data_fetchers(self):
        """Initialize data fetchers based on configuration"""
        # Exchange data fetcher
        exchange_config = self.config.get('exchange', {})
        self.exchange_fetcher = create_data_fetcher(exchange_config)

        # Historical data fetcher
        historical_config = self.config.get('historical', {})
        self.historical_fetcher = create_historical_fetcher(historical_config)

        # Alternative data fetcher
        alternative_config = self.config.get('alternative', {})
        try:
            self.alternative_fetcher = create_alternative_data_fetcher(alternative_config)
        except (ImportError, ModuleNotFoundError):
            logger.warning("Alternative data gateway not available, some features will be disabled")
            self.alternative_fetcher = None

    def start(self):
        """Start the market data service"""
        with self._lock:
            if self._running:
                logger.warning("Market data service is already running")
                return

            self._running = True
            self._update_thread = threading.Thread(
                target=self._update_loop,
                name="MarketDataUpdateThread",
                daemon=True
            )
            self._update_thread.start()

            logger.info("Market data service started")

            # Publish event
            event = create_event(
                EventTopics.COMPONENT_STARTED,
                {
                    'name': 'market_data_service',
                    'status': 'running'
                }
            )
            self._event_bus.publish(event)

    def stop(self):
        """Stop the market data service"""
        with self._lock:
            if not self._running:
                logger.warning("Market data service is not running")
                return

            self._running = False

            # Wait for update thread to finish
            if self._update_thread and self._update_thread.is_alive():
                self._update_thread.join(timeout=5.0)

            # Clear all subscriptions
            with self._subscription_lock:
                self._subscriptions.clear()

            logger.info("Market data service stopped")

            # Publish event
            event = create_event(
                EventTopics.COMPONENT_STOPPED,
                {
                    'name': 'market_data_service',
                    'status': 'stopped'
                }
            )
            self._event_bus.publish(event)

    def get_data(self,
                 symbol: str,
                 timeframe: Union[str, DataTimeframe] = DataTimeframe.D1,
                 start_date: Optional[Union[datetime, str]] = None,
                 end_date: Optional[Union[datetime, str]] = None,
                 source: DataSource = DataSource.AUTO,
                 limit: Optional[int] = None,
                 include_latest: bool = True,
                 cache: bool = True) -> pd.DataFrame:
        """
        Get market data for a symbol.

        Args:
            symbol: Symbol to get data for
            timeframe: Timeframe for the data
            start_date: Start date for historical data
            end_date: End date for historical data
            source: Data source to use
            limit: Maximum number of data points to return
            include_latest: Whether to include latest data from exchanges
            cache: Whether to use cache

        Returns:
            DataFrame with market data
        """
        # Normalize inputs
        timeframe_str = timeframe.value if isinstance(timeframe, DataTimeframe) else timeframe

        # Check cache if enabled
        if cache and self.use_cache:
            cache_key = f"{symbol}_{timeframe_str}_{start_date}_{end_date}_{source.value}_{limit}"
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                logger.debug(f"Using cached data for {symbol} {timeframe_str}")
                return cached_data

        # Determine which source to use
        result = pd.DataFrame()

        if source == DataSource.AUTO:
            # Try to determine best source based on date range
            if end_date is None or pd.to_datetime(end_date) >= pd.Timestamp.now().floor('D'):
                # Current or recent data, try exchange first
                result = self._get_exchange_data(symbol, timeframe_str, start_date, end_date, limit)

                # If exchange data is incomplete or empty, supplement with historical data
                if result.empty or (start_date is not None and result.index.min() > pd.to_datetime(start_date)):
                    hist_data = self._get_historical_data(symbol, timeframe_str, start_date, end_date, limit)

                    if not hist_data.empty:
                        if result.empty:
                            result = hist_data
                        else:
                            # Merge exchange and historical data
                            combined = pd.concat([hist_data, result])
                            combined = combined[~combined.index.duplicated(keep='last')]
                            combined.sort_index(inplace=True)
                            result = combined
            else:
                # Historical data only
                result = self._get_historical_data(symbol, timeframe_str, start_date, end_date, limit)

        elif source == DataSource.EXCHANGE:
            # Exchange data
            result = self._get_exchange_data(symbol, timeframe_str, start_date, end_date, limit)

        elif source == DataSource.HISTORICAL:
            # Historical data
            result = self._get_historical_data(symbol, timeframe_str, start_date, end_date, limit)

        elif source == DataSource.ALTERNATIVE:
            # Alternative data
            result = self._get_alternative_data(symbol, timeframe_str, start_date, end_date, limit)

        elif source == DataSource.COMBINED:
            # Combine data from all sources
            exchange_data = self._get_exchange_data(symbol, timeframe_str, start_date, end_date, limit)
            historical_data = self._get_historical_data(symbol, timeframe_str, start_date, end_date, limit)
            alternative_data = self._get_alternative_data(symbol, timeframe_str, start_date, end_date, limit)

            # Combine all data sources
            combined = pd.concat([historical_data, exchange_data, alternative_data])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined.sort_index(inplace=True)

            result = combined

        # Apply limit if specified
        if limit is not None and not result.empty:
            result = result.iloc[-limit:]

        # Add to cache if not empty
        if cache and self.use_cache and not result.empty:
            self._add_to_cache(cache_key, result)

        # Normalize data format
        if not result.empty:
            result = normalize_market_data(result, symbol, timeframe_str)

        return result

    def subscribe(self,
                 symbol: str,
                 timeframe: Union[str, DataTimeframe],
                 callback: Callable[[pd.DataFrame], None],
                 source: DataSource = DataSource.AUTO,
                 options: Dict[str, Any] = None) -> str:
        """
        Subscribe to market data updates.

        Args:
            symbol: Symbol to subscribe to
            timeframe: Timeframe for the data
            callback: Function to call with updated data
            source: Data source to use
            options: Additional options for the subscription

        Returns:
            Subscription ID
        """
        subscription = MarketDataSubscription(
            symbol=symbol,
            timeframe=timeframe,
            callback=callback,
            source=source,
            options=options
        )

        with self._subscription_lock:
            self._subscriptions[subscription.id] = subscription

        logger.debug(f"Created subscription {subscription.id} for {symbol} {subscription.timeframe}")

        # Trigger initial update
        self._update_subscription(subscription)

        return subscription.id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from market data updates.

        Args:
            subscription_id: Subscription ID

        Returns:
            True if successfully unsubscribed, False otherwise
        """
        with self._subscription_lock:
            if subscription_id not in self._subscriptions:
                logger.warning(f"Subscription {subscription_id} not found")
                return False

            # Deactivate subscription
            self._subscriptions[subscription_id].deactivate()

            # Remove from subscriptions
            del self._subscriptions[subscription_id]

            logger.debug(f"Removed subscription {subscription_id}")
            return True

    def get_subscription(self, subscription_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a subscription.

        Args:
            subscription_id: Subscription ID

        Returns:
            Subscription information dictionary
        """
        with self._subscription_lock:
            if subscription_id not in self._subscriptions:
                return None

            subscription = self._subscriptions[subscription_id]

            return {
                'id': subscription.id,
                'symbol': subscription.symbol,
                'timeframe': subscription.timeframe,
                'source': subscription.source.value,
                'active': subscription.active,
                'last_update': subscription.last_update,
                'options': subscription.options
            }

    def get_all_subscriptions(self) -> List[Dict[str, Any]]:
        """
        Get information about all subscriptions.

        Returns:
            List of subscription information dictionaries
        """
        with self._subscription_lock:
            return [
                {
                    'id': subscription.id,
                    'symbol': subscription.symbol,
                    'timeframe': subscription.timeframe,
                    'source': subscription.source.value,
                    'active': subscription.active,
                    'last_update': subscription.last_update,
                    'options': subscription.options
                }
                for subscription in self._subscriptions.values()
            ]

    def get_available_symbols(self, source: DataSource = DataSource.AUTO) -> List[str]:
        """
        Get list of available symbols.

        Args:
            source: Data source to query

        Returns:
            List of available symbols
        """
        symbols = set()

        if source in [DataSource.AUTO, DataSource.HISTORICAL]:
            try:
                hist_symbols = self.historical_fetcher.get_available_symbols()
                symbols.update(hist_symbols)
            except Exception as e:
                logger.error(f"Error getting historical symbols: {str(e)}")

        if source in [DataSource.AUTO, DataSource.EXCHANGE]:
            try:
                for exchange_id in self.exchange_fetcher.get_supported_exchanges():
                    exchange_symbols = self.exchange_fetcher.get_supported_symbols(exchange_id)
                    symbols.update(exchange_symbols)
            except Exception as e:
                logger.error(f"Error getting exchange symbols: {str(e)}")

        if source in [DataSource.AUTO, DataSource.ALTERNATIVE] and self.alternative_fetcher:
            try:
                alt_symbols = self.alternative_fetcher.get_available_symbols()
                symbols.update(alt_symbols)
            except Exception as e:
                logger.error(f"Error getting alternative symbols: {str(e)}")

        return sorted(list(symbols))

    def get_available_timeframes(self, symbol: str, source: DataSource = DataSource.AUTO) -> List[str]:
        """
        Get list of available timeframes for a symbol.

        Args:
            symbol: Symbol
            source: Data source to query

        Returns:
            List of available timeframes
        """
        timeframes = set()

        if source in [DataSource.AUTO, DataSource.HISTORICAL]:
            try:
                hist_timeframes = self.historical_fetcher.get_available_timeframes(symbol)
                timeframes.update(hist_timeframes)
            except Exception as e:
                logger.error(f"Error getting historical timeframes: {str(e)}")

        if source in [DataSource.AUTO, DataSource.EXCHANGE]:
            # Most exchanges support standard timeframes
            timeframes.update([tf.value for tf in DataTimeframe])

        if source in [DataSource.AUTO, DataSource.ALTERNATIVE] and self.alternative_fetcher:
            try:
                alt_timeframes = self.alternative_fetcher.get_available_timeframes(symbol)
                timeframes.update(alt_timeframes)
            except Exception as e:
                logger.error(f"Error getting alternative timeframes: {str(e)}")

        return sorted(list(timeframes))

    def save_data(self,
                 data: pd.DataFrame,
                 symbol: str,
                 timeframe: Union[str, DataTimeframe],
                 append: bool = True) -> bool:
        """
        Save market data to storage.

        Args:
            data: DataFrame with market data
            symbol: Symbol
            timeframe: Timeframe
            append: Whether to append to existing data

        Returns:
            True if successful, False otherwise
        """
        if data.empty:
            logger.warning(f"Cannot save empty data for {symbol}")
            return False

        timeframe_str = timeframe.value if isinstance(timeframe, DataTimeframe) else timeframe

        # Save to both CSV and database for redundancy
        csv_success = self.historical_fetcher.save_to_csv(data, symbol, timeframe_str, append)
        db_success = self.historical_fetcher.save_to_db(data, symbol, timeframe_str)

        return csv_success and db_success

    def get_status(self) -> Dict[str, Any]:
        """
        Get market data service status.

        Returns:
            Dictionary with service status information
        """
        return {
            'running': self._running,
            'subscriptions': len(self._subscriptions),
            'cache_size': len(self._cache),
            'supported_exchanges': self.exchange_fetcher.get_supported_exchanges(),
            'available_symbols_count': len(self.get_available_symbols()),
            'alternative_data_available': self.alternative_fetcher is not None
        }

    def clear_cache(self) -> None:
        """Clear the data cache."""
        with self._lock:
            self._cache.clear()
            logger.info("Market data cache cleared")

    def _update_loop(self) -> None:
        """Main update loop for subscription updates"""
        logger.debug("Market data update loop started")

        while self._running:
            try:
                # Get copy of subscriptions to avoid lock during updates
                active_subscriptions = []
                with self._subscription_lock:
                    active_subscriptions = [sub for sub in self._subscriptions.values() if sub.active]

                # Update each subscription
                for subscription in active_subscriptions:
                    try:
                        self._update_subscription(subscription)
                    except Exception as e:
                        logger.error(f"Error updating subscription {subscription.id}: {str(e)}")

                # Sleep until next update
                time.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Error in market data update loop: {str(e)}")
                time.sleep(1.0)  # Sleep to avoid tight loop in case of persistent error

    def _update_subscription(self, subscription: MarketDataSubscription) -> None:
        """
        Update a single subscription.

        Args:
            subscription: Subscription to update
        """
        # Check if subscription is active
        if not subscription.active:
            return

        # Apply rate limiting if specified in options
        throttle = subscription.options.get('throttle', self.default_throttle)
        if subscription.last_update is not None:
            elapsed = time.time() - subscription.last_update
            if elapsed < throttle:
                return

        # Get data for subscription
        try:
            # Determine date range based on options
            lookback = subscription.options.get('lookback', 100)
            start_date = None
            if isinstance(lookback, int):
                # Use limit instead of date
                limit = lookback
            else:
                # Parse lookback as timedelta
                if isinstance(lookback, str):
                    if lookback.endswith('d'):
                        days = int(lookback[:-1])
                        start_date = datetime.now() - timedelta(days=days)
                    elif lookback.endswith('h'):
                        hours = int(lookback[:-1])
                        start_date = datetime.now() - timedelta(hours=hours)
                    else:
                        # Default to days if no unit specified
                        days = int(lookback)
                        start_date = datetime.now() - timedelta(days=days)
                else:
                    # Assume timedelta object
                    start_date = datetime.now() - lookback
                limit = None

            # Get data
            data = self.get_data(
                symbol=subscription.symbol,
                timeframe=subscription.timeframe,
                start_date=start_date,
                source=subscription.source,
                limit=limit,
                cache=False  # Don't use cache for subscriptions
            )

            # Call callback with data
            if not data.empty:
                subscription.callback(data)
                subscription.last_update = time.time()

        except Exception as e:
            logger.error(f"Error updating subscription {subscription.id}: {str(e)}")

    def _get_exchange_data(self,
                         symbol: str,
                         timeframe: str,
                         start_date: Optional[Union[datetime, str]] = None,
                         end_date: Optional[Union[datetime, str]] = None,
                         limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get data from exchange.

        Args:
            symbol: Symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            limit: Maximum number of data points

        Returns:
            DataFrame with market data from exchange
        """
        try:
            if start_date is not None:
                # Historical data from exchange
                return self.exchange_fetcher.fetch_historical_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=timeframe
                )
            else:
                # Recent data from exchange
                bars = limit or 100
                return self.exchange_fetcher.fetch_recent_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    bars=bars
                )
        except Exception as e:
            logger.error(f"Error getting exchange data for {symbol} {timeframe}: {str(e)}")
            return pd.DataFrame()

    def _get_historical_data(self,
                           symbol: str,
                           timeframe: str,
                           start_date: Optional[Union[datetime, str]] = None,
                           end_date: Optional[Union[datetime, str]] = None,
                           limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get data from historical storage.

        Args:
            symbol: Symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            limit: Maximum number of data points

        Returns:
            DataFrame with market data from historical storage
        """
        try:
            data = self.historical_fetcher.fetch_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                source='auto'
            )

            # Apply limit if specified
            if limit is not None and not data.empty:
                data = data.iloc[-limit:]

            return data
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol} {timeframe}: {str(e)}")
            return pd.DataFrame()

    def _get_alternative_data(self,
                            symbol: str,
                            timeframe: str,
                            start_date: Optional[Union[datetime, str]] = None,
                            end_date: Optional[Union[datetime, str]] = None,
                            limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get data from alternative sources.

        Args:
            symbol: Symbol
            timeframe: Timeframe
            start_date: Start date
            end_date: End date
            limit: Maximum number of data points

        Returns:
            DataFrame with market data from alternative sources
        """
        if self.alternative_fetcher is None:
            return pd.DataFrame()

        try:
            data = self.alternative_fetcher.fetch_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )

            # Apply limit if specified
            if limit is not None and not data.empty:
                data = data.iloc[-limit:]

            return data
        except Exception as e:
            logger.error(f"Error getting alternative data for {symbol} {timeframe}: {str(e)}")
            return pd.DataFrame()

    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Get data from cache.

        Args:
            cache_key: Cache key

        Returns:
            DataFrame with market data or None if not found or expired
        """
        with self._lock:
            if cache_key not in self._cache:
                return None

            data, timestamp = self._cache[cache_key]

            # Check if cache is expired
            if time.time() - timestamp > self.cache_ttl:
                # Remove expired cache entry
                del self._cache[cache_key]
                return None

            # Return copy of data
            return data.copy()

    def _add_to_cache(self, cache_key: str, data: pd.DataFrame) -> None:
        """
        Add data to cache.

        Args:
            cache_key: Cache key
            data: DataFrame with market data
        """
        with self._lock:
            # Check if cache is full
            if len(self._cache) >= self.max_cache_items:
                # Remove oldest entry
                oldest_key = min(self._cache.items(), key=lambda x: x[1][1])[0]
                del self._cache[oldest_key]

            # Add to cache
            self._cache[cache_key] = (data.copy(), time.time())

    def _handle_market_data_event(self, event: Event) -> None:
        """
        Handle market data event.

        Args:
            event: Market data event
        """
        try:
            data = event.data

            # Check if event contains required fields
            if not isinstance(data, dict) or 'symbol' not in data or 'data' not in data:
                return

            symbol = data['symbol']
            market_data = data['data']
            timeframe = data.get('timeframe', '1m')

            # Find all subscriptions for this symbol and timeframe
            matching_subscriptions = []
            with self._subscription_lock:
                for subscription in self._subscriptions.values():
                    if subscription.active and subscription.symbol == symbol and subscription.timeframe == timeframe:
                        matching_subscriptions.append(subscription)

            # Update subscriptions
            for subscription in matching_subscriptions:
                try:
                    subscription.callback(market_data)
                    subscription.last_update = time.time()
                except Exception as e:
                    logger.error(f"Error processing market data event for subscription {subscription.id}: {str(e)}")

        except Exception as e:
            logger.error(f"Error handling market data event: {str(e)}")

    def get_market_depth(self, symbol: str) -> Dict[str, Any]:
        """
        Get current market depth data.

        Args:
            symbol: Symbol

        Returns:
            Dictionary with order book data
        """
        try:
            return self.exchange_fetcher.fetch_order_book(symbol)
        except Exception as e:
            logger.error(f"Error getting market depth for {symbol}: {str(e)}")
            return {}

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker data.

        Args:
            symbol: Symbol

        Returns:
            Dictionary with ticker data
        """
        try:
            return self.exchange_fetcher.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {str(e)}")
            return {}

    def get_recent_trades(self,
                        symbol: str,
                        limit: int = 100) -> pd.DataFrame:
        """
        Get recent trades for a symbol.

        Args:
            symbol: Symbol
            limit: Maximum number of trades

        Returns:
            DataFrame with trade data
        """
        try:
            return self.exchange_fetcher.fetch_trades(symbol, limit=limit)
        except Exception as e:
            logger.error(f"Error getting recent trades for {symbol}: {str(e)}")
            return pd.DataFrame()


# Singleton instance
_market_data_service_instance = None

def get_market_data_service(config: Dict[str, Any] = None) -> MarketDataService:
    """
    Get the market data service singleton instance.

    Args:
        config: Configuration dictionary (only used if instance doesn't exist)

    Returns:
        MarketDataService instance
    """
    global _market_data_service_instance
    if _market_data_service_instance is None:
        _market_data_service_instance = MarketDataService(config)
    return _market_data_service_instance