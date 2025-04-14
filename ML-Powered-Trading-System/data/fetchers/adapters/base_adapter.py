"""
data/fetchers/adapters/base_adapter.py - Base Market Adapter Interface

This module defines the base interface for market-specific adapters that handle
the details of communicating with different market types and data providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
import pandas as pd
from enum import Enum

from execution.exchange.connectivity_manager import ConnectivityManager
from utils.logger import get_logger
from data.fetchers.exchange_connector import MarketType, DataType, DataFrequency


class BaseMarketAdapter(ABC):
    """
    Base interface for market-specific adapters.

    Market adapters handle the specific details of different market types
    (equities, futures, forex, etc.) while presenting a unified interface
    to the exchange connector.
    """

    def __init__(
            self,
            market_type: MarketType,
            config: Dict[str, Any],
            connectivity_manager: ConnectivityManager
    ):
        """
        Initialize the market adapter.

        Args:
            market_type: Type of financial market
            config: Market-specific configuration
            connectivity_manager: Connectivity manager for network communication
        """
        self.market_type = market_type
        self.config = config
        self.connectivity_manager = connectivity_manager
        self.logger = get_logger(f"{__name__}.{market_type.value}")

        # Provider configurations
        self.providers = config.get('providers', {})

        # Market-specific settings
        self.settings = config.get('settings', {})

        # Provider endpoints
        self.endpoints = {}

        # Provider API parameters
        self.api_params = {}

        # Market hours
        self.market_hours = config.get('market_hours', {})

        # Active subscriptions
        self.subscriptions = {}
        self.next_subscription_id = 1

    async def initialize(self) -> None:
        """
        Initialize the adapter.

        This should be called before using the adapter to establish
        necessary connections and load market metadata.
        """
        # Load provider endpoints and parameters
        for provider_id, provider_config in self.providers.items():
            self.endpoints[provider_id] = provider_config.get('endpoints', {})
            self.api_params[provider_id] = provider_config.get('parameters', {})

        self.logger.info(f"Initialized {self.market_type.value} adapter with {len(self.providers)} providers")

    async def get_provider_endpoint(
            self,
            provider: str,
            endpoint_name: str,
            **kwargs
    ) -> str:
        """
        Get the full endpoint URL for a provider API endpoint.

        Args:
            provider: Provider ID
            endpoint_name: Name of the endpoint
            **kwargs: Parameters to substitute in the endpoint URL

        Returns:
            Full endpoint URL
        """
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not configured for {self.market_type.value}")

        if endpoint_name not in self.endpoints[provider]:
            raise ValueError(f"Endpoint {endpoint_name} not configured for provider {provider}")

        # Get base URL
        base_url = self.providers[provider].get('api_url', '')

        # Get endpoint path
        endpoint = self.endpoints[provider][endpoint_name]

        # Substitute parameters in the endpoint path
        if kwargs:
            endpoint = endpoint.format(**kwargs)

        # Combine base URL and endpoint
        full_url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        return full_url

    @abstractmethod
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
        Fetch historical data for an instrument.

        Args:
            instrument: Instrument identifier
            frequency: Data frequency
            start_time: Start time
            end_time: End time
            data_type: Type of data to fetch
            provider: Data provider to use
            **kwargs: Additional parameters

        Returns:
            DataFrame with historical data
        """
        pass

    @abstractmethod
    async def get_market_data(
            self,
            instruments: List[str],
            data_type: DataType,
            provider: str,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Fetch current market data for instruments.

        Args:
            instruments: List of instrument identifiers
            data_type: Type of data to fetch
            provider: Data provider to use
            **kwargs: Additional parameters

        Returns:
            Dict with market data
        """
        pass

    @abstractmethod
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
        Subscribe to real-time data updates.

        Args:
            instruments: List of instrument identifiers
            data_type: Type of data to subscribe to
            callback: Callback function for updates
            frequency: Data frequency (for OHLCV updates)
            provider: Data provider to use
            **kwargs: Additional parameters

        Returns:
            Subscription ID
        """
        pass

    @abstractmethod
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
        pass

    async def get_market_hours(
            self,
            provider: str,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Get trading hours for the market.

        Args:
            provider: Data provider to use
            **kwargs: Additional parameters

        Returns:
            Dict with market hours information
        """
        # Default implementation uses configured market hours
        return self.market_hours

    @abstractmethod
    async def get_instruments(
            self,
            provider: str,
            **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Get list of available instruments.

        Args:
            provider: Data provider to use
            **kwargs: Additional parameters

        Returns:
            List of instrument information
        """
        pass

    async def get_exchange_info(
            self,
            provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get information about the market and its providers.

        Args:
            provider: Data provider to use (if None, returns info for all providers)

        Returns:
            Dict with market and provider information
        """
        if provider:
            # Return info for specific provider
            if provider not in self.providers:
                raise ValueError(f"Provider {provider} not configured for {self.market_type.value}")

            return {
                "market_type": self.market_type.value,
                "provider": provider,
                "settings": self.providers[provider]
            }
        else:
            # Return info for all providers
            return {
                "market_type": self.market_type.value,
                "providers": list(self.providers.keys()),
                "settings": self.settings
            }

    @abstractmethod
    async def get_reference_data(
            self,
            instruments: List[str],
            data_type: str,
            provider: str,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Get reference data for instruments.

        Args:
            instruments: List of instrument identifiers
            data_type: Type of reference data
            provider: Data provider to use
            **kwargs: Additional parameters

        Returns:
            Dict with reference data
        """
        pass