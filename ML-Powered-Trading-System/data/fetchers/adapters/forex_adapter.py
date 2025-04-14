"""
data/fetchers/adapters/forex_adapter.py - Forex Market Adapter

This module implements the market adapter for Forex (Foreign Exchange) markets,
handling the specific details of retrieving currency pair data from various providers.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import time
import json

from data.fetchers.adapters.base_adapter import BaseMarketAdapter
from data.fetchers.exchange_connector import MarketType, DataType, DataFrequency
from execution.exchange.connectivity_manager import ConnectivityManager
from utils.logger import get_logger


class ForexAdapter(BaseMarketAdapter):
    """
    Forex-specific market adapter.

    Handles the unique aspects of foreign exchange markets, including:
    - 24-hour trading windows
    - Currency pair notation
    - Pip value calculations
    - Bid/ask spread management
    - Multiple provider integration
    """

    def __init__(
            self,
            market_type: MarketType,
            config: Dict[str, Any],
            connectivity_manager: ConnectivityManager
    ):
        """
        Initialize the Forex adapter.

        Args:
            market_type: Should be MarketType.FOREX
            config: Forex-specific configuration
            connectivity_manager: Connectivity manager for network communication
        """
        super().__init__(market_type, config, connectivity_manager)

        # Forex-specific settings
        self.pip_sizes = config.get('pip_sizes', {})
        self.default_pip_size = config.get('default_pip_size', 0.0001)

        # Track major, minor, and exotic currency pairs
        self.pair_categories = config.get('pair_categories', {
            'major': ['EUR/USD', 'USD/JPY', 'GBP/USD', 'USD/CHF', 'USD/CAD', 'AUD/USD', 'NZD/USD'],
            'minor': ['EUR/GBP', 'EUR/CHF', 'EUR/CAD', 'EUR/AUD', 'GBP/JPY', 'CHF/JPY', 'EUR/JPY', 'GBP/CHF'],
            'exotic': []  # To be populated from provider data
        })

        # Metadata for currency pairs
        self.pairs_metadata = {}

        # Active websocket connections
        self.ws_connections = {}

    async def initialize(self) -> None:
        """
        Initialize the Forex adapter.

        Loads metadata for currency pairs and establishes initial connections.
        """
        await super().initialize()

        # Load metadata for common currency pairs
        await self._load_pairs_metadata()

        self.logger.info(f"Initialized Forex adapter with {len(self.pairs_metadata)} currency pairs")

    async def _load_pairs_metadata(self):
        """
        Load metadata for available currency pairs from all providers.
        """
        for provider_id in self.providers:
            try:
                # Use default provider in config
                provider = self._get_default_provider()

                # Fetch available pairs from provider
                pairs = await self.get_instruments(provider)

                # Extract and store metadata
                for pair_info in pairs:
                    symbol = pair_info['symbol']

                    # Store metadata with provider-specific details
                    if symbol not in self.pairs_metadata:
                        self.pairs_metadata[symbol] = {}

                    self.pairs_metadata[symbol][provider_id] = pair_info

                    # Determine pip size if not already configured
                    if symbol not in self.pip_sizes:
                        # Default logic based on currency pair
                        if 'JPY' in symbol:
                            self.pip_sizes[symbol] = 0.01  # 2 decimal places for JPY pairs
                        else:
                            self.pip_sizes[symbol] = 0.0001  # 4 decimal places for most pairs

                # Categorize any uncategorized pairs as exotic
                known_pairs = set(self.pair_categories['major'] + self.pair_categories['minor'])
                for pair in self.pairs_metadata:
                    if pair not in known_pairs:
                        self.pair_categories['exotic'].append(pair)

            except Exception as e:
                self.logger.error(f"Error loading metadata from provider {provider_id}: {str(e)}")

    def _get_default_provider(self) -> str:
        """
        Get the default provider for Forex data.

        Returns:
            Default provider ID
        """
        if 'default_provider' in self.settings:
            return self.settings['default_provider']

        # Use first provider as default
        if self.providers:
            return list(self.providers.keys())[0]

        raise ValueError("No Forex data providers configured")

    async def get_historical_data(
            self,
            instrument: str,
            frequency: str,
            start_time: Union[datetime, int],
            end_time: Union[datetime, int],
            data_type: DataType,
            provider: str,
            **kwargs
    ) -> pd.DataFrame:
        """
        Fetch historical Forex data.

        Args:
            instrument: Currency pair (e.g., "EUR/USD")
            frequency: Data frequency (e.g., "1m", "1h", "1d")
            start_time: Start time
            end_time: End time
            data_type: Type of data to fetch
            provider: Data provider to use
            **kwargs: Additional parameters

        Returns:
            DataFrame with historical data
        """
        self.logger.info(f"Fetching historical data for {instrument} at {frequency} frequency from {provider}")

        # Standardize currency pair format
        instrument = self._standardize_pair_format(instrument)

        # Convert datetime to timestamp if necessary
        if isinstance(start_time, datetime):
            start_time = int(start_time.timestamp())
        if isinstance(end_time, datetime):
            end_time = int(end_time.timestamp())
        elif end_time is None:
            end_time = int(datetime.now().timestamp())

        # Prepare request parameters
        params = {
            'symbol': instrument,
            'interval': frequency,
            'startTime': start_time * 1000,  # Convert to milliseconds
            'endTime': end_time * 1000,  # Convert to milliseconds
            'limit': kwargs.get('limit', 1000)
        }

        # Add provider-specific parameters
        params.update(self.api_params.get(provider, {}).get('historical_data', {}))

        # Get endpoint URL
        endpoint = await self.get_provider_endpoint(
            provider,
            'historical_data',
            symbol=self._format_pair_for_provider(instrument, provider)
        )

        try:
            # Make request
            response = await self.connectivity_manager.request(
                'GET',
                endpoint,
                params=params,
                provider_id=provider
            )

            # Check for errors
            if 'error' in response:
                self.logger.error(f"Error fetching historical data: {response['error']}")
                raise ValueError(f"Provider error: {response['error']}")

            # Process response based on data type
            if data_type == DataType.OHLCV:
                # Create DataFrame from response
                df = pd.DataFrame(response['data'])

                # Rename columns to standard format
                column_mapping = self.providers[provider].get('column_mapping', {}).get('ohlcv', {})
                df = df.rename(columns=column_mapping)

                # Ensure required columns exist
                required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                for col in required_columns:
                    if col not in df.columns:
                        self.logger.warning(f"Required column {col} not in response, adding empty column")
                        df[col] = None

                # Convert timestamp to datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)

                # Sort by timestamp
                df.sort_index(inplace=True)

                return df

            elif data_type == DataType.QUOTE:
                # For quote data, create a DataFrame with bid/ask prices
                df = pd.DataFrame(response['data'])

                # Rename columns to standard format
                column_mapping = self.providers[provider].get('column_mapping', {}).get('quote', {})
                df = df.rename(columns=column_mapping)

                # Ensure required columns exist
                required_columns = ['timestamp', 'bid', 'ask']
                for col in required_columns:
                    if col not in df.columns:
                        self.logger.warning(f"Required column {col} not in response, adding empty column")
                        df[col] = None

                # Convert timestamp to datetime
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)

                # Sort by timestamp
                df.sort_index(inplace=True)

                return df

            else:
                self.logger.warning(f"Data type {data_type.value} not fully supported for Forex historical data")
                # Return raw data as DataFrame
                return pd.DataFrame(response['data'])

        except Exception as e:
            self.logger.error(f"Error fetching historical data for {instrument}: {str(e)}")
            raise

    async def get_market_data(
            self,
            instruments: List[str],
            data_type: DataType,
            provider: str,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Fetch current Forex market data.

        Args:
            instruments: List of currency pairs
            data_type: Type of data to fetch
            provider: Data provider to use
            **kwargs: Additional parameters

        Returns:
            Dict with market data
        """
        self.logger.info(f"Fetching market data for {instruments} from {provider}")

        # Standardize currency pair formats
        instruments = [self._standardize_pair_format(instr) for instr in instruments]

        # Prepare request parameters
        params = {
            'symbols': ','.join([self._format_pair_for_provider(instr, provider) for instr in instruments])
        }

        # Add provider-specific parameters
        params.update(self.api_params.get(provider, {}).get('market_data', {}))

        # Get endpoint URL
        if data_type == DataType.QUOTE:
            endpoint_name = 'ticker'
        elif data_type == DataType.DEPTH:
            endpoint_name = 'orderbook'
        else:
            endpoint_name = data_type.value

        endpoint = await self.get_provider_endpoint(provider, endpoint_name)

        try:
            # Make request
            response = await self.connectivity_manager.request(
                'GET',
                endpoint,
                params=params,
                provider_id=provider
            )

            # Check for errors
            if 'error' in response:
                self.logger.error(f"Error fetching market data: {response['error']}")
                raise ValueError(f"Provider error: {response['error']}")

            # Process response based on data type
            result = {}

            if data_type == DataType.QUOTE:
                # Convert response to standardized format
                for pair_data in response.get('data', []):
                    symbol = self._standardize_pair_format(pair_data.get('symbol'))

                    # Skip if not in requested instruments
                    if symbol not in instruments:
                        continue

                    # Extract quote data
                    result[symbol] = {
                        'bid': float(pair_data.get('bid', 0)),
                        'ask': float(pair_data.get('ask', 0)),
                        'spread': float(pair_data.get('spread', 0)),
                        'timestamp': pair_data.get('timestamp', int(time.time() * 1000))
                    }

                    # Add pip value calculation
                    pip_size = self.pip_sizes.get(symbol, self.default_pip_size)
                    result[symbol]['pip_value'] = pip_size
                    result[symbol]['pip_cost'] = self._calculate_pip_cost(symbol, pip_size)

            elif data_type == DataType.DEPTH:
                # Convert response to standardized order book format
                for pair_data in response.get('data', []):
                    symbol = self._standardize_pair_format(pair_data.get('symbol'))

                    # Skip if not in requested instruments
                    if symbol not in instruments:
                        continue

                    # Extract order book data
                    result[symbol] = {
                        'bids': pair_data.get('bids', []),
                        'asks': pair_data.get('asks', []),
                        'timestamp': pair_data.get('timestamp', int(time.time() * 1000))
                    }

            else:
                # For other data types, return raw data
                for pair_data in response.get('data', []):
                    symbol = self._standardize_pair_format(pair_data.get('symbol'))
                    result[symbol] = pair_data

            return result

        except Exception as e:
            self.logger.error(f"Error fetching market data for {instruments}: {str(e)}")
            raise

    async def subscribe_to_data(
            self,
            instruments: List[str],
            data_type: DataType,
            callback: Callable,
            frequency: Optional[str],
            provider: str,
            **kwargs
    ) -> int:
        """
        Subscribe to real-time Forex data updates.

        Args:
            instruments: List of currency pairs
            data_type: Type of data to subscribe to
            callback: Callback function for updates
            frequency: Data frequency (for OHLCV updates)
            provider: Data provider to use
            **kwargs: Additional parameters

        Returns:
            Subscription ID
        """
        self.logger.info(f"Subscribing to {data_type.value} for {instruments} from {provider}")

        # Standardize currency pair formats
        instruments = [self._standardize_pair_format(instr) for instr in instruments]

        # Generate subscription ID
        subscription_id = self.next_subscription_id
        self.next_subscription_id += 1

        # Create subscription details
        subscription = {
            'instruments': instruments,
            'data_type': data_type,
            'callback': callback,
            'frequency': frequency,
            'provider': provider,
            'params': kwargs,
            'active': True
        }

        # Store subscription
        self.subscriptions[subscription_id] = subscription

        # Determine if we need to establish a new websocket connection
        ws_key = f"{provider}_{data_type.value}"

        if ws_key not in self.ws_connections:
            # Create new websocket connection for this provider and data type
            try:
                # Get websocket endpoint
                if data_type == DataType.QUOTE:
                    endpoint_name = 'ws_ticker'
                elif data_type == DataType.DEPTH:
                    endpoint_name = 'ws_orderbook'
                elif data_type == DataType.OHLCV:
                    endpoint_name = 'ws_kline'
                else:
                    endpoint_name = f"ws_{data_type.value}"

                ws_endpoint = await self.get_provider_endpoint(provider, endpoint_name)

                # Define message handler
                async def message_handler(message):
                    await self._handle_websocket_message(provider, data_type, message)

                # Connect to websocket
                conn = await self.connectivity_manager.connect_websocket(
                    ws_endpoint,
                    message_handler,
                    provider_id=provider
                )

                # Store connection
                self.ws_connections[ws_key] = {
                    'connection': conn,
                    'subscriptions': set([subscription_id])
                }

                # Subscribe to instruments
                subscribe_message = self._create_subscribe_message(
                    provider, data_type, instruments, frequency, kwargs
                )

                await self.connectivity_manager.send_websocket(
                    conn,
                    subscribe_message
                )

            except Exception as e:
                self.logger.error(f"Error establishing websocket connection: {str(e)}")
                # Clean up subscription
                del self.subscriptions[subscription_id]
                raise
        else:
            # Add subscription to existing connection
            self.ws_connections[ws_key]['subscriptions'].add(subscription_id)

            # Subscribe to any new instruments
            current_instruments = set()
            for sub_id in self.ws_connections[ws_key]['subscriptions']:
                if sub_id != subscription_id:
                    current_instruments.update(self.subscriptions[sub_id]['instruments'])

            new_instruments = [i for i in instruments if i not in current_instruments]

            if new_instruments:
                # Subscribe to new instruments
                subscribe_message = self._create_subscribe_message(
                    provider, data_type, new_instruments, frequency, kwargs
                )

                await self.connectivity_manager.send_websocket(
                    self.ws_connections[ws_key]['connection'],
                    subscribe_message
                )

        return subscription_id

    def _create_subscribe_message(
            self,
            provider: str,
            data_type: DataType,
            instruments: List[str],
            frequency: Optional[str],
            params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a subscription message for a websocket.

        Args:
            provider: Data provider
            data_type: Type of data
            instruments: List of instruments
            frequency: Data frequency
            params: Additional parameters

        Returns:
            Subscription message
        """
        # Get provider-specific message format
        message_format = self.providers[provider].get('ws_format', {})

        # Create base message
        message = {
            'method': message_format.get('subscribe_method', 'SUBSCRIBE'),
            'params': []
        }

        # Add instruments based on data type
        for instrument in instruments:
            formatted_pair = self._format_pair_for_provider(instrument, provider)

            if data_type == DataType.QUOTE:
                channel = message_format.get('ticker_channel', 'ticker').format(symbol=formatted_pair)
            elif data_type == DataType.DEPTH:
                channel = message_format.get('orderbook_channel', 'depth').format(symbol=formatted_pair)
            elif data_type == DataType.OHLCV:
                # Include frequency for OHLCV data
                freq = frequency or '1m'
                channel = message_format.get('kline_channel', 'kline').format(
                    symbol=formatted_pair, interval=freq
                )
            elif data_type == DataType.TRADE:
                channel = message_format.get('trade_channel', 'trade').format(symbol=formatted_pair)
            else:
                channel = message_format.get(f"{data_type.value}_channel", data_type.value).format(
                    symbol=formatted_pair
                )

            message['params'].append(channel)

        # Add provider-specific fields
        for key, value in message_format.get('additional_fields', {}).items():
            message[key] = value

        # Add request ID
        message['id'] = int(time.time() * 1000)

        return message

    async def _handle_websocket_message(
            self,
            provider: str,
            data_type: DataType,
            message: Dict[str, Any]
    ):
        """
        Handle incoming websocket message.

        Args:
            provider: Data provider
            data_type: Type of data
            message: Incoming message
        """
        try:
            # Ignore non-data messages
            if 'data' not in message:
                return

            # Extract data based on provider-specific format
            provider_format = self.providers[provider].get('ws_format', {})

            # Extract symbol and data
            symbol_field = provider_format.get('symbol_field', 'symbol')
            data_field = provider_format.get('data_field', 'data')

            symbol = message.get(symbol_field, '')
            data = message.get(data_field, {})

            if not symbol or not data:
                # Try to extract from nested structure
                if 'data' in message and isinstance(message['data'], dict):
                    symbol = message['data'].get(symbol_field, '')
                    data = message['data']

            # Standardize symbol format
            symbol = self._standardize_pair_format(symbol)

            # Process data based on type
            processed_data = {}

            if data_type == DataType.QUOTE:
                processed_data = {
                    'symbol': symbol,
                    'bid': float(data.get('bid', 0)),
                    'ask': float(data.get('ask', 0)),
                    'timestamp': data.get('timestamp', int(time.time() * 1000))
                }

                # Add spread and pip value
                processed_data['spread'] = processed_data['ask'] - processed_data['bid']
                processed_data['pip_value'] = self.pip_sizes.get(symbol, self.default_pip_size)

            elif data_type == DataType.DEPTH:
                processed_data = {
                    'symbol': symbol,
                    'bids': data.get('bids', []),
                    'asks': data.get('asks', []),
                    'timestamp': data.get('timestamp', int(time.time() * 1000))
                }

            elif data_type == DataType.OHLCV:
                kline_mapping = provider_format.get('kline_mapping', {})

                processed_data = {
                    'symbol': symbol,
                    'timestamp': data.get(kline_mapping.get('timestamp', 'timestamp'), int(time.time() * 1000)),
                    'open': float(data.get(kline_mapping.get('open', 'open'), 0)),
                    'high': float(data.get(kline_mapping.get('high', 'high'), 0)),
                    'low': float(data.get(kline_mapping.get('low', 'low'), 0)),
                    'close': float(data.get(kline_mapping.get('close', 'close'), 0)),
                    'volume': float(data.get(kline_mapping.get('volume', 'volume'), 0))
                }

            elif data_type == DataType.TRADE:
                trade_mapping = provider_format.get('trade_mapping', {})

                processed_data = {
                    'symbol': symbol,
                    'price': float(data.get(trade_mapping.get('price', 'price'), 0)),
                    'amount': float(data.get(trade_mapping.get('amount', 'amount'), 0)),
                    'side': data.get(trade_mapping.get('side', 'side'), ''),
                    'timestamp': data.get(trade_mapping.get('timestamp', 'timestamp'), int(time.time() * 1000))
                }

            else:
                # Pass through raw data for other types
                processed_data = {
                    'symbol': symbol,
                    'data': data
                }

            # Find relevant subscriptions and invoke callbacks
            ws_key = f"{provider}_{data_type.value}"

            if ws_key in self.ws_connections:
                for sub_id in self.ws_connections[ws_key]['subscriptions']:
                    subscription = self.subscriptions.get(sub_id)

                    if subscription and subscription['active']:
                        if symbol in subscription['instruments']:
                            # Call the callback with the processed data
                            await subscription['callback'](processed_data)

        except Exception as e:
            self.logger.error(f"Error handling websocket message: {str(e)}")

    async def unsubscribe(
            self,
            subscription_id: int
    ) -> bool:
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

        try:
            subscription = self.subscriptions[subscription_id]
            provider = subscription['provider']
            data_type = subscription['data_type']
            instruments = subscription['instruments']

            # Mark subscription as inactive
            subscription['active'] = False

            # Update websocket connection
            ws_key = f"{provider}_{data_type.value}"

            if ws_key in self.ws_connections:
                conn_info = self.ws_connections[ws_key]
                conn_info['subscriptions'].remove(subscription_id)

                # If no more subscriptions, unsubscribe from all instruments
                if not conn_info['subscriptions']:
                    # Create unsubscribe message
                    message_format = self.providers[provider].get('ws_format', {})

                    unsubscribe_message = {
                        'method': message_format.get('unsubscribe_method', 'UNSUBSCRIBE'),
                        'params': []
                    }

                    # Add instruments to unsubscribe
                    for instrument in instruments:
                        formatted_pair = self._format_pair_for_provider(instrument, provider)

                        if data_type == DataType.QUOTE:
                            channel = message_format.get('ticker_channel', 'ticker').format(symbol=formatted_pair)
                        elif data_type == DataType.DEPTH:
                            channel = message_format.get('orderbook_channel', 'depth').format(symbol=formatted_pair)
                        elif data_type == DataType.OHLCV:
                            freq = subscription['frequency'] or '1m'
                            channel = message_format.get('kline_channel', 'kline').format(
                                symbol=formatted_pair, interval=freq
                            )
                        else:
                            channel = message_format.get(f"{data_type.value}_channel", data_type.value).format(
                                symbol=formatted_pair
                            )

                        unsubscribe_message['params'].append(channel)

                    # Add provider-specific fields
                    for key, value in message_format.get('additional_fields', {}).items():
                        unsubscribe_message[key] = value

                    # Add request ID
                    unsubscribe_message['id'] = int(time.time() * 1000)

                    # Send unsubscribe message
                    await self.connectivity_manager.send_websocket(
                        conn_info['connection'],
                        unsubscribe_message
                    )

                    # Close connection if no more subscriptions
                    await self.connectivity_manager.close_websocket(conn_info['connection'])
                    del self.ws_connections[ws_key]

                # Otherwise, check if we need to unsubscribe from specific instruments
                else:
                    remaining_instruments = set()
                    for sub_id in conn_info['subscriptions']:
                        remaining_instruments.update(self.subscriptions[sub_id]['instruments'])

                    instruments_to_unsubscribe = [i for i in instruments if i not in remaining_instruments]

                    if instruments_to_unsubscribe:
                        # Create unsubscribe message for specific instruments
                        message_format = self.providers[provider].get('ws_format', {})

                        unsubscribe_message = {
                            'method': message_format.get('unsubscribe_method', 'UNSUBSCRIBE'),
                            'params': []
                        }

                        # Add instruments to unsubscribe
                        for instrument in instruments_to_unsubscribe:
                            formatted_pair = self._format_pair_for_provider(instrument, provider)

                            if data_type == DataType.QUOTE:
                                channel = message_format.get('ticker_channel', 'ticker').format(symbol=formatted_pair)
                            elif data_type == DataType.DEPTH:
                                channel = message_format.get('orderbook_channel', 'depth').format(symbol=formatted_pair)
                            elif data_type == DataType.OHLCV:
                                freq = subscription['frequency'] or '1m'
                                channel = message_format.get('kline_channel', 'kline').format(
                                    symbol=formatted_pair, interval=freq
                                )
                            else:
                                channel = message_format.get(f"{data_type.value}_channel", data_type.value).format(
                                    symbol=formatted_pair
                                )

                            unsubscribe_message['params'].append(channel)

                        # Add provider-specific fields
                        for key, value in message_format.get('additional_fields', {}).items():
                            unsubscribe_message[key] = value

                        # Add request ID
                        unsubscribe_message['id'] = int(time.time() * 1000)

                        # Send unsubscribe message
                        await self.connectivity_manager.send_websocket(
                            conn_info['connection'],
                            unsubscribe_message
                        )

            # Remove subscription
            del self.subscriptions[subscription_id]

            return True

        except Exception as e:
            self.logger.error(f"Error unsubscribing from {subscription_id}: {str(e)}")
            return False

    async def get_instruments(
            self,
            provider: str,
            **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Get list of available Forex currency pairs.

        Args:
            provider: Data provider to use
            **kwargs: Additional parameters

        Returns:
            List of currency pair information
        """
        self.logger.info(f"Fetching available Forex instruments from {provider}")

        # Prepare request parameters
        params = {}

        # Add provider-specific parameters
        params.update(self.api_params.get(provider, {}).get('instruments', {}))

        # Get endpoint URL
        endpoint = await self.get_provider_endpoint(provider, 'symbols')

        try:
            # Make request
            response = await self.connectivity_manager.request(
                'GET',
                endpoint,
                params=params,
                provider_id=provider
            )

            # Check for errors
            if 'error' in response:
                self.logger.error(f"Error fetching instruments: {response['error']}")
                raise ValueError(f"Provider error: {response['error']}")

            # Process symbols from response
            symbols = []

            # Get field mappings for this provider
            field_mapping = self.providers[provider].get('field_mapping', {}).get('instrument', {})

            # Extract symbols based on provider response format
            symbols_data = response.get('data', [])

            if not symbols_data and 'symbols' in response:
                symbols_data = response.get('symbols', [])

            for symbol_data in symbols_data:
                # Extract fields based on mapping
                symbol = symbol_data.get(field_mapping.get('symbol', 'symbol'), '')

                # Skip non-forex pairs if filter exists
                if 'type' in symbol_data and field_mapping.get('type', 'type') in symbol_data:
                    pair_type = symbol_data.get(field_mapping.get('type', 'type'))
                    if pair_type != 'forex' and kwargs.get('forex_only', True):
                        continue

                # Standardize symbol format
                symbol = self._standardize_pair_format(symbol)

                # Create standardized instrument info
                instrument_info = {
                    'symbol': symbol,
                    'base_currency': symbol.split('/')[0] if '/' in symbol else symbol[:3],
                    'quote_currency': symbol.split('/')[1] if '/' in symbol else symbol[3:],
                    'pip_size': symbol_data.get(field_mapping.get('pip_size', 'pip_size'),
                                                self.pip_sizes.get(symbol, self.default_pip_size)),
                    'min_size': float(symbol_data.get(field_mapping.get('min_size', 'min_size'), 0.01)),
                    'max_size': float(symbol_data.get(field_mapping.get('max_size', 'max_size'), 1000000)),
                    'min_notional': float(symbol_data.get(field_mapping.get('min_notional', 'min_notional'), 1)),
                    'price_precision': int(symbol_data.get(field_mapping.get('price_precision', 'price_precision'), 5)),
                    'size_precision': int(symbol_data.get(field_mapping.get('size_precision', 'size_precision'), 2)),
                    'trading_fees': float(symbol_data.get(field_mapping.get('trading_fees', 'trading_fees'), 0)),
                }

                # Add provider-specific data
                instrument_info['provider_data'] = {k: v for k, v in symbol_data.items()
                                                    if k not in field_mapping.values()}

                symbols.append(instrument_info)

            return symbols