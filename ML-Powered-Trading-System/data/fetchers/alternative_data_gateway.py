"""
data/fetchers/alternative_data_gateway.py - Alternative Data Gateway

This module provides access to non-market data sources such as news, social media,
economic indicators, and other alternative data for trading models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
import asyncio
import json
import aiohttp
from enum import Enum
from pathlib import Path

from core.event_bus import EventBus, create_event
from data.processors.data_normalizer import DataNormalizer
from execution.exchange.connectivity_manager import ConnectivityManager
from utils.logger import get_logger


class AltDataType(Enum):
    """Types of alternative data"""
    NEWS = "news"  # News articles and headlines
    SOCIAL_SENTIMENT = "sentiment"  # Social media sentiment
    ECONOMIC = "economic"  # Economic indicators
    EARNINGS = "earnings"  # Earnings data
    ESG = "esg"  # Environmental, Social, and Governance
    SEC_FILINGS = "sec_filings"  # SEC filings
    INSIDER = "insider"  # Insider trading
    WEATHER = "weather"  # Weather data
    SATELLITE = "satellite"  # Satellite imagery
    CREDIT_CARD = "credit_card"  # Credit card transactions
    WEB_TRAFFIC = "web_traffic"  # Website traffic
    APP_USAGE = "app_usage"  # Mobile app usage


class AltDataFormat(Enum):
    """Data formats for alternative data"""
    STRUCTURED = "structured"  # Tabular data
    TEXT = "text"  # Unstructured text
    TIMESERIES = "timeseries"  # Time series data
    IMAGE = "image"  # Image data
    MIXED = "mixed"  # Mixed format data


class AlternativeDataGateway:
    """
    Gateway for accessing alternative data sources.

    This class provides a unified interface to various alternative data sources
    including news, social media, economic indicators, and more.
    """

    def __init__(
            self,
            connectivity_manager: ConnectivityManager,
            data_normalizer: DataNormalizer,
            event_bus: EventBus,
            config: Dict[str, Any] = None
    ):
        """
        Initialize the alternative data gateway.

        Args:
            connectivity_manager: Manages connectivity to data sources
            data_normalizer: Normalizes data from different sources
            event_bus: System-wide event bus for events
            config: Configuration for the gateway
        """
        self.logger = get_logger(__name__)
        self.connectivity_manager = connectivity_manager
        self.normalizer = data_normalizer
        self.event_bus = event_bus
        self.config = config or {}

        # Data sources configuration
        self.sources = self.config.get('sources', {})

        # Cache settings
        self.use_cache = self.config.get('use_cache', True)
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # seconds
        self.cache_dir = Path(self.config.get('cache_dir', 'data/cache/alt_data'))

        # Create cache directory if it doesn't exist
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Data subscribers
        self.subscribers = {}
        self.next_subscriber_id = 1

        # Active data streams
        self.active_streams = {}

        self.logger.info(f"Alternative data gateway initialized with {len(self.sources)} sources")

    async def get_news(
            self,
            symbols: Optional[Union[str, List[str]]] = None,
            keywords: Optional[List[str]] = None,
            start_time: Optional[Union[datetime, int]] = None,
            end_time: Optional[Union[datetime, int]] = None,
            source: Optional[str] = None,
            limit: Optional[int] = 100,
            **kwargs
    ) -> pd.DataFrame:
        """
        Fetch news data.

        Args:
            symbols: Instrument symbol(s) to filter news for
            keywords: Keywords to filter news by
            start_time: Start time for news
            end_time: End time for news
            source: News data source
            limit: Maximum number of news items to fetch
            **kwargs: Additional parameters

        Returns:
            DataFrame with news data
        """
        # Convert single symbol to list
        symbols_list = [symbols] if isinstance(symbols, str) else symbols

        # Convert timestamps to datetime if needed
        if isinstance(start_time, int):
            start_time = datetime.fromtimestamp(start_time / 1000.0)

        if isinstance(end_time, int):
            end_time = datetime.fromtimestamp(end_time / 1000.0)
        elif end_time is None:
            end_time = datetime.now()

        if start_time is None:
            start_time = end_time - timedelta(days=7)  # Default to last 7 days

        # Get default source if not specified
        if source is None:
            source = self._get_default_source(AltDataType.SOCIAL_SENTIMENT)

        # Check if source is configured
        if source not in self.sources.get(AltDataType.SOCIAL_SENTIMENT.value, {}):
            raise ValueError(f"Sentiment source {source} not configured")

        # Get platforms to include
        platforms = platforms or []

        # Get data from cache if available
        cache_key = self._get_cache_key(
            AltDataType.SOCIAL_SENTIMENT,
            source,
            symbols_list,
            platforms,
            start_time,
            end_time
        )

        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            # Get source-specific implementation
            if source == 'stocktwits':
                df = await self._get_stocktwits_sentiment(symbols_list, start_time, end_time)
            elif source == 'twitter':
                df = await self._get_twitter_sentiment(symbols_list, platforms, start_time, end_time)
            elif source == 'reddit':
                df = await self._get_reddit_sentiment(symbols_list, start_time, end_time)
            else:
                raise ValueError(f"Unsupported sentiment source: {source}")

            # Cache the data
            if self.use_cache and not df.empty:
                self._add_to_cache(cache_key, df)

            return df

        except Exception as e:
            self.logger.error(f"Error fetching sentiment data: {str(e)}")
            await self._publish_error_event(
                "error.data.alternative.sentiment",
                str(symbols_list),
                str(e)
            )
            return pd.DataFrame()

    async def get_economic_data(
            self,
            indicators: Union[str, List[str]],
            start_time: Optional[Union[datetime, int]] = None,
            end_time: Optional[Union[datetime, int]] = None,
            source: Optional[str] = None,
            **kwargs
    ) -> pd.DataFrame:
        """
        Fetch economic indicators data.

        Args:
            indicators: Economic indicator(s) to fetch
            start_time: Start time for data
            end_time: End time for data
            source: Data source
            **kwargs: Additional parameters

        Returns:
            DataFrame with economic data
        """
        # Convert single indicator to list
        indicators_list = [indicators] if isinstance(indicators, str) else indicators

        # Convert timestamps to datetime if needed
        if isinstance(start_time, int):
            start_time = datetime.fromtimestamp(start_time / 1000.0)

        if isinstance(end_time, int):
            end_time = datetime.fromtimestamp(end_time / 1000.0)
        elif end_time is None:
            end_time = datetime.now()

        if start_time is None:
            start_time = end_time - timedelta(days=365)  # Default to last year

        # Get default source if not specified
        if source is None:
            source = self._get_default_source(AltDataType.ECONOMIC)

        # Check if source is configured
        if source not in self.sources.get(AltDataType.ECONOMIC.value, {}):
            raise ValueError(f"Economic data source {source} not configured")

        # Get data from cache if available
        cache_key = self._get_cache_key(
            AltDataType.ECONOMIC,
            source,
            indicators_list,
            None,
            start_time,
            end_time
        )

        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            # Get source-specific implementation
            if source == 'fred':
                df = await self._get_fred_economic_data(indicators_list, start_time, end_time)
            elif source == 'world_bank':
                df = await self._get_world_bank_data(indicators_list, start_time, end_time)
            elif source == 'bea':
                df = await self._get_bea_data(indicators_list, start_time, end_time)
            else:
                raise ValueError(f"Unsupported economic data source: {source}")

            # Cache the data
            if self.use_cache and not df.empty:
                self._add_to_cache(cache_key, df)

            return df

        except Exception as e:
            self.logger.error(f"Error fetching economic data: {str(e)}")
            await self._publish_error_event(
                "error.data.alternative.economic",
                str(indicators_list),
                str(e)
            )
            return pd.DataFrame()

    async def get_earnings_data(
            self,
            symbols: Union[str, List[str]],
            start_time: Optional[Union[datetime, int]] = None,
            end_time: Optional[Union[datetime, int]] = None,
            include_estimates: bool = True,
            source: Optional[str] = None,
            **kwargs
    ) -> pd.DataFrame:
        """
        Fetch earnings data for symbols.

        Args:
            symbols: Symbol(s) to fetch earnings data for
            start_time: Start time for data
            end_time: End time for data
            include_estimates: Whether to include analyst estimates
            source: Data source
            **kwargs: Additional parameters

        Returns:
            DataFrame with earnings data
        """
        # Convert single symbol to list
        symbols_list = [symbols] if isinstance(symbols, str) else symbols

        # Convert timestamps to datetime if needed
        if isinstance(start_time, int):
            start_time = datetime.fromtimestamp(start_time / 1000.0)

        if isinstance(end_time, int):
            end_time = datetime.fromtimestamp(end_time / 1000.0)
        elif end_time is None:
            end_time = datetime.now()

        if start_time is None:
            start_time = end_time - timedelta(days=365)  # Default to last year

        # Get default source if not specified
        if source is None:
            source = self._get_default_source(AltDataType.EARNINGS)

        # Check if source is configured
        if source not in self.sources.get(AltDataType.EARNINGS.value, {}):
            raise ValueError(f"Earnings data source {source} not configured")

        # Get data from cache if available
        cache_key = self._get_cache_key(
            AltDataType.EARNINGS,
            source,
            symbols_list,
            [f"include_estimates={include_estimates}"],
            start_time,
            end_time
        )

        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            # Get source-specific implementation
            if source == 'alpha_vantage':
                df = await self._get_alpha_vantage_earnings(symbols_list, start_time, end_time, include_estimates)
            elif source == 'zacks':
                df = await self._get_zacks_earnings(symbols_list, start_time, end_time, include_estimates)
            else:
                raise ValueError(f"Unsupported earnings data source: {source}")

            # Cache the data
            if self.use_cache and not df.empty:
                self._add_to_cache(cache_key, df)

            return df

        except Exception as e:
            self.logger.error(f"Error fetching earnings data: {str(e)}")
            await self._publish_error_event(
                "error.data.alternative.earnings",
                str(symbols_list),
                str(e)
            )
            return pd.DataFrame()

    async def subscribe_to_alt_data(
            self,
            data_type: AltDataType,
            subjects: List[str],
            callback: Callable[[Dict[str, Any]], None],
            source: Optional[str] = None,
            **kwargs
    ) -> int:
        """
        Subscribe to alternative data updates.

        Args:
            data_type: Type of alternative data
            subjects: Subjects to subscribe to (symbols, keywords, etc.)
            callback: Callback function for data updates
            source: Data source
            **kwargs: Additional parameters

        Returns:
            Subscription ID
        """
        # Get default source if not specified
        if source is None:
            source = self._get_default_source(data_type)

        # Check if source is configured
        if source not in self.sources.get(data_type.value, {}):
            raise ValueError(f"Alternative data source {source} not configured for {data_type.value}")

        # Create subscription ID
        subscription_id = self.next_subscriber_id
        self.next_subscriber_id += 1

        # Store subscription details
        self.subscribers[subscription_id] = {
            'type': data_type,
            'subjects': subjects,
            'callback': callback,
            'source': source,
            'active': True,
            'params': kwargs
        }

        # Initialize data stream if needed
        stream_key = f"{data_type.value}_{source}"

        if stream_key not in self.active_streams:
            # Create a new stream
            self.active_streams[stream_key] = {
                'subjects': set(subjects),
                'subscribers': {subscription_id},
                'task': None
            }

            # Start the stream
            self.active_streams[stream_key]['task'] = asyncio.create_task(
                self._run_data_stream(data_type, source)
            )
        else:
            # Add to existing stream
            self.active_streams[stream_key]['subjects'].update(subjects)
            self.active_streams[stream_key]['subscribers'].add(subscription_id)

        self.logger.info(f"Created alt data subscription {subscription_id} for {data_type.value} from {source}")
        return subscription_id

    async def unsubscribe(self, subscription_id: int) -> bool:
        """
        Unsubscribe from alternative data updates.

        Args:
            subscription_id: Subscription ID

        Returns:
            True if successful, False otherwise
        """
        if subscription_id not in self.subscribers:
            self.logger.warning(f"Subscription {subscription_id} not found")
            return False

        subscription = self.subscribers[subscription_id]
        data_type = subscription['type']
        source = subscription['source']

        # Mark as inactive
        subscription['active'] = False

        # Update the stream
        stream_key = f"{data_type.value}_{source}"

        if stream_key in self.active_streams:
            # Remove subscriber
            self.active_streams[stream_key]['subscribers'].discard(subscription_id)

            # If no more subscribers, stop the stream
            if not self.active_streams[stream_key]['subscribers']:
                # Cancel the task
                task = self.active_streams[stream_key]['task']
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                # Remove the stream
                del self.active_streams[stream_key]

        # Remove the subscriber
        del self.subscribers[subscription_id]

        self.logger.info(f"Removed alt data subscription {subscription_id}")
        return True

    def _get_default_source(self, data_type: AltDataType) -> Optional[str]:
        """
        Get the default source for a data type.

        Args:
            data_type: Type of alternative data

        Returns:
            Default source name or None if not configured
        """
        type_config = self.sources.get(data_type.value, {})

        # Check if default source is specified
        if 'default' in type_config:
            return type_config['default']

        # Use the first source if available
        if type_config:
            return next(iter(type_config.keys()))

        return None

    def _get_cache_key(
            self,
            data_type: AltDataType,
            source: str,
            subjects: Optional[List[str]],
            filters: Optional[List[Any]],
            start_time: datetime,
            end_time: datetime
    ) -> str:
        """
        Generate a cache key for data.

        Args:
            data_type: Type of alternative data
            source: Data source
            subjects: Data subjects (symbols, indicators, etc.)
            filters: Additional filters
            start_time: Start time
            end_time: End time

        Returns:
            Cache key string
        """
        # Convert subjects to sorted string
        subjects_str = '_'.join(sorted(subjects)) if subjects else 'none'

        # Convert filters to string
        filters_str = '_'.join(sorted(str(f) for f in filters)) if filters else 'none'

        # Format timestamps
        start_str = start_time.strftime('%Y%m%d')
        end_str = end_time.strftime('%Y%m%d')

        # Create key
        return f"{data_type.value}_{source}_{subjects_str}_{filters_str}_{start_str}_{end_str}"

    def _get_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """
        Get data from cache.

        Args:
            key: Cache key

        Returns:
            Cached data or None if not found or expired
        """
        if not self.use_cache:
            return None

        # Check if file exists
        cache_file = self.cache_dir / f"{key}.parquet"

        if not cache_file.exists():
            return None

        # Check if expired
        if self.cache_ttl > 0:
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age > self.cache_ttl:
                return None

        try:
            # Read from cache
            return pd.read_parquet(cache_file)
        except Exception as e:
            self.logger.error(f"Error reading from cache: {str(e)}")
            return None

    def _add_to_cache(self, key: str, data: pd.DataFrame) -> None:
        """
        Add data to cache.

        Args:
            key: Cache key
            data: Data to cache
        """
        if not self.use_cache:
            return

        try:
            # Write to cache
            cache_file = self.cache_dir / f"{key}.parquet"
            data.to_parquet(cache_file)
        except Exception as e:
            self.logger.error(f"Error writing to cache: {str(e)}")

    async def _publish_error_event(
            self,
            event_type: str,
            subjects: str,
            error: str
    ) -> None:
        """
        Publish an error event to the event bus.

        Args:
            event_type: Type of error event
            subjects: Data subjects
            error: Error message
        """
        try:
            event = create_event(
                event_type,
                {
                    "subjects": subjects,
                    "error": error,
                    "timestamp": time.time()
                }
            )
            await self.event_bus.publish(event)
        except Exception as e:
            self.logger.error(f"Error publishing error event: {str(e)}")

    async def _run_data_stream(
            self,
            data_type: AltDataType,
            source: str
    ) -> None:
        """
        Run a data stream for subscribers.

        Args:
            data_type: Type of alternative data
            source: Data source
        """
        stream_key = f"{data_type.value}_{source}"

        try:
            self.logger.info(f"Starting {data_type.value} stream from {source}")

            while stream_key in self.active_streams:
                # Get current subjects
                subjects = list(self.active_streams[stream_key]['subjects'])

                if not subjects:
                    # No subjects to stream
                    await asyncio.sleep(1.0)
                    continue

                # Fetch data based on type and source
                if data_type == AltDataType.NEWS:
                    await self._stream_news(source, subjects)
                elif data_type == AltDataType.SOCIAL_SENTIMENT:
                    await self._stream_sentiment(source, subjects)
                else:
                    self.logger.warning(f"Streaming not supported for {data_type.value}")
                    break

                # Wait before next update
                await asyncio.sleep(10.0)  # Adjust based on data type

        except asyncio.CancelledError:
            self.logger.info(f"Stream {stream_key} cancelled")
        except Exception as e:
            self.logger.error(f"Error in data stream {stream_key}: {str(e)}")
        finally:
            # Clean up
            if stream_key in self.active_streams:
                del self.active_streams[stream_key]

    # Source-specific implementation methods

    async def _get_alpha_vantage_news(
            self,
            symbols: List[str],
            keywords: Optional[List[str]],
            start_time: datetime,
            end_time: datetime,
            limit: int
    ) -> pd.DataFrame:
        """Get news from Alpha Vantage"""
        # Placeholder implementation
        return pd.DataFrame()

    async def _get_newsapi_news(
            self,
            symbols: List[str],
            keywords: Optional[List[str]],
            start_time: datetime,
            end_time: datetime,
            limit: int
    ) -> pd.DataFrame:
        """Get news from News API"""
        # Placeholder implementation
        return pd.DataFrame()

    async def _get_bloomberg_news(
            self,
            symbols: List[str],
            keywords: Optional[List[str]],
            start_time: datetime,
            end_time: datetime,
            limit: int
    ) -> pd.DataFrame:
        """Get news from Bloomberg"""
        # Placeholder implementation
        return pd.DataFrame()

    async def _get_stocktwits_sentiment(
            self,
            symbols: List[str],
            start_time: datetime,
            end_time: datetime
    ) -> pd.DataFrame:
        """Get sentiment from StockTwits"""
        # Placeholder implementation
        return pd.DataFrame()

    async def _get_twitter_sentiment(
            self,
            symbols: List[str],
            platforms: List[str],
            start_time: datetime,
            end_time: datetime
    ) -> pd.DataFrame:
        """Get sentiment from Twitter"""
        # Placeholder implementation
        return pd.DataFrame()

    async def _get_reddit_sentiment(
            self,
            symbols: List[str],
            start_time: datetime,
            end_time: datetime
    ) -> pd.DataFrame:
        """Get sentiment from Reddit"""
        # Placeholder implementation
        return pd.DataFrame()

    async def _get_fred_economic_data(
            self,
            indicators: List[str],
            start_time: datetime,
            end_time: datetime
    ) -> pd.DataFrame:
        """Get economic data from FRED"""
        # Placeholder implementation
        return pd.DataFrame()

    async def _get_world_bank_data(
            self,
            indicators: List[str],
            start_time: datetime,
            end_time: datetime
    ) -> pd.DataFrame:
        """Get economic data from World Bank"""
        # Placeholder implementation
        return pd.DataFrame()

    async def _get_bea_data(
            self,
            indicators: List[str],
            start_time: datetime,
            end_time: datetime
    ) -> pd.DataFrame:
        """Get economic data from BEA"""
        # Placeholder implementation
        return pd.DataFrame()

    async def _get_alpha_vantage_earnings(
            self,
            symbols: List[str],
            start_time: datetime,
            end_time: datetime,
            include_estimates: bool
    ) -> pd.DataFrame:
        """Get earnings data from Alpha Vantage"""
        # Placeholder implementation
        return pd.DataFrame()

    async def _get_zacks_earnings(
            self,
            symbols: List[str],
            start_time: datetime,
            end_time: datetime,
            include_estimates: bool
    ) -> pd.DataFrame:
        """Get earnings data from Zacks"""
        # Placeholder implementation
        return pd.DataFrame()

    async def _stream_news(
            self,
            source: str,
            subjects: List[str]
    ) -> None:
        """Stream news updates to subscribers"""
        stream_key = f"{AltDataType.NEWS.value}_{source}"

        if stream_key not in self.active_streams:
            return

        try:
            # Fetch latest news
            news_data = []

            # Implementation depends on source
            if source == 'alpha_vantage':
                # Placeholder implementation
                pass
            elif source == 'newsapi':
                # Placeholder implementation
                pass

            # Distribute to subscribers
            if news_data:
                for sub_id in self.active_streams[stream_key]['subscribers']:
                    if sub_id in self.subscribers and self.subscribers[sub_id]['active']:
                        try:
                            await self.subscribers[sub_id]['callback'](news_data)
                        except Exception as e:
                            self.logger.error(f"Error in news subscriber callback {sub_id}: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error streaming news from {source}: {str(e)}")

    async def _stream_sentiment(
            self,
            source: str,
            subjects: List[str]
    ) -> None:
        """Stream sentiment updates to subscribers"""
        stream_key = f"{AltDataType.SOCIAL_SENTIMENT.value}_{source}"

        if stream_key not in self.active_streams:
            return

        try:
            # Fetch latest sentiment
            sentiment_data = []

            # Implementation depends on source
            if source == 'stocktwits':
                # Placeholder implementation
                pass
            elif source == 'twitter':
                # Placeholder implementation
                pass

            # Distribute to subscribers
            if sentiment_data:
                for sub_id in self.active_streams[stream_key]['subscribers']:
                    if sub_id in self.subscribers and self.subscribers[sub_id]['active']:
                        try:
                            await self.subscribers[sub_id]['callback'](sentiment_data)
                        except Exception as e:
                            self.logger.error(f"Error in sentiment subscriber callback {sub_id}: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error streaming sentiment from {source}: {str(e)}")


_time, int):
start_time = datetime.fromtimestamp(start_time / 1000.0)

if isinstance(end_time, int):
    end_time = datetime.fromtimestamp(end_time / 1000.0)
elif end_time is None:
    end_time = datetime.now()

if start_time is None:
    start_time = end_time - timedelta(days=7)  # Default to last 7 days

# Get default source if not specified
if source is None:
    source = self._get_default_source(AltDataType.NEWS)

# Check if source is configured
if source not in self.sources.get(AltDataType.NEWS.value, {}):
    raise ValueError(f"News source {source} not configured")

# Get data from cache if available
cache_key = self._get_cache_key(
    AltDataType.NEWS,
    source,
    symbols_list,
    keywords,
    start_time,
    end_time
)

cached_data = self._get_from_cache(cache_key)
if cached_data is not None:
    return cached_data

try:
    # Get source-specific implementation
    if source == 'alpha_vantage':
        df = await self._get_alpha_vantage_news(symbols_list, keywords, start_time, end_time, limit)
    elif source == 'newsapi':
        df = await self._get_newsapi_news(symbols_list, keywords, start_time, end_time, limit)
    elif source == 'bloomberg':
        df = await self._get_bloomberg_news(symbols_list, keywords, start_time, end_time, limit)
    else:
        raise ValueError(f"Unsupported news source: {source}")

    # Cache the data
    if self.use_cache and not df.empty:
        self._add_to_cache(cache_key, df)

    return df

except Exception as e:
    self.logger.error(f"Error fetching news data: {str(e)}")
    await self._publish_error_event(
        "error.data.alternative.news",
        str(symbols_list),
        str(e)
    )
    return pd.DataFrame()


async def get_sentiment(
        self,
        symbols: Optional[Union[str, List[str]]] = None,
        start_time: Optional[Union[datetime, int]] = None,
        end_time: Optional[Union[datetime, int]] = None,
        source: Optional[str] = None,
        platforms: Optional[List[str]] = None,
        **kwargs
) -> pd.DataFrame:
    """
    Fetch social media sentiment data.

    Args:
        symbols: Instrument symbol(s) to get sentiment for
        start_time: Start time for sentiment data
        end_time: End time for sentiment data
        source: Sentiment data source
        platforms: Social media platforms to include
        **kwargs: Additional parameters

    Returns:
        DataFrame with sentiment data
    """
    # Convert single symbol to list
    symbols_list = [symbols] if isinstance(symbols, str) else symbols

    # Convert timestamps to datetime if needed
    if isinstance(start