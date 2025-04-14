"""
exchange_gateway.py - Abstract base class for exchange connectivity

This module defines the ExchangeGateway abstract base class that all specific
exchange implementations (like OandaGateway) should inherit from. It provides
the common interface for order management, position tracking, and account operations.
"""

import abc
import logging
import time
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime

# Import from core modules
from core.event_bus import EventBus

# Configure logger
logger = logging.getLogger(__name__)


class ExchangeType(Enum):
    """Types of exchanges supported by the system"""
    SPOT = "spot"  # Spot markets (stocks, forex, etc.)
    FUTURES = "futures"  # Futures markets
    OPTIONS = "options"  # Options markets
    CRYPTO_SPOT = "crypto_spot"  # Cryptocurrency spot markets
    CRYPTO_DERIVATIVES = "crypto_derivatives"  # Cryptocurrency futures/options


class ExchangeCapabilities(Enum):
    """Capabilities that may be supported by an exchange"""
    MARKET_ORDERS = "market_orders"
    LIMIT_ORDERS = "limit_orders"
    STOP_ORDERS = "stop_orders"
    TRAILING_STOPS = "trailing_stops"
    OCO_ORDERS = "oco_orders"  # One-cancels-other orders
    MARGIN_TRADING = "margin_trading"
    SHORT_SELLING = "short_selling"
    STREAMING_QUOTES = "streaming_quotes"
    STREAMING_ORDERBOOK = "streaming_orderbook"
    STREAMING_TRADES = "streaming_trades"
    HISTORICAL_DATA = "historical_data"
    ACCOUNT_BALANCE = "account_balance"
    POSITION_MANAGEMENT = "position_management"
    ORDER_HISTORY = "order_history"
    TRADE_HISTORY = "trade_history"


class ExchangeGateway(abc.ABC):
    """
    Abstract base class for exchange gateways.

    This class defines the interface that all exchange implementations
    must follow to ensure consistency across the trading system.
    """

    def __init__(
            self,
            exchange_id: str,
            exchange_name: str,
            exchange_type: ExchangeType,
            capabilities: List[ExchangeCapabilities],
            event_bus: Optional[EventBus] = None,
            **kwargs
    ):
        """
        Initialize the exchange gateway.

        Args:
            exchange_id: Unique identifier for this exchange
            exchange_name: Human-readable name of the exchange
            exchange_type: Type of exchange
            capabilities: List of capabilities supported by this exchange
            event_bus: System event bus for publishing events
            **kwargs: Additional exchange-specific parameters
        """
        self.exchange_id = exchange_id
        self.exchange_name = exchange_name
        self.exchange_type = exchange_type
        self.capabilities = capabilities
        self.event_bus = event_bus

        # Track connection state
        self.connected = False
        self.last_connection_time = None
        self.connection_error = None

        logger.info(f"Initialized {exchange_name} gateway ({exchange_id})")

    @abc.abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the exchange.

        Returns:
            True if connection is successful, False otherwise
        """
        pass

    @abc.abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the exchange.

        Returns:
            True if disconnection is successful
        """
        pass

    @abc.abstractmethod
    def is_connected(self) -> bool:
        """
        Check if currently connected to the exchange.

        Returns:
            True if connected, False otherwise
        """
        return self.connected

    @abc.abstractmethod
    def place_order(self, order: Any) -> str:
        """
        Place an order on the exchange.

        Args:
            order: The order to place

        Returns:
            exchange_order_id: The exchange's order ID

        Raises:
            Exception: If order placement fails
        """
        pass

    @abc.abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order on the exchange.

        Args:
            order_id: The ID of the order to cancel

        Returns:
            True if cancellation is successful, False otherwise
        """
        pass

    @abc.abstractmethod
    def get_order(self, order_id: str) -> Optional[Any]:
        """
        Get order details from the exchange.

        Args:
            order_id: The ID of the order to retrieve

        Returns:
            The order if found, None otherwise
        """
        pass

    @abc.abstractmethod
    def get_open_orders(self) -> List[Any]:
        """
        Get all open orders from the exchange.

        Returns:
            List of open orders
        """
        pass

    @abc.abstractmethod
    def get_position(self, instrument: str) -> float:
        """
        Get current position for an instrument.

        Args:
            instrument: The instrument to get position for

        Returns:
            Current position size (positive for long, negative for short)
        """
        pass

    @abc.abstractmethod
    def get_all_positions(self) -> Dict[str, float]:
        """
        Get all current positions.

        Returns:
            Dictionary mapping instruments to position sizes
        """
        pass

    @abc.abstractmethod
    def close_position(self, instrument: str, amount: Optional[float] = None) -> bool:
        """
        Close position for an instrument.

        Args:
            instrument: Instrument to close position for
            amount: Optional amount to close (None = close all)

        Returns:
            True if position was closed successfully
        """
        pass

    @abc.abstractmethod
    def get_balance(self, currency: str) -> float:
        """
        Get current balance for a currency.

        Args:
            currency: The currency to get balance for

        Returns:
            Current balance
        """
        pass

    @abc.abstractmethod
    def get_all_balances(self) -> Dict[str, float]:
        """
        Get all current balances.

        Returns:
            Dictionary mapping currencies to balances
        """
        pass

    @abc.abstractmethod
    def get_ticker(self, instrument: str) -> Dict[str, Any]:
        """
        Get current ticker data for an instrument.

        Args:
            instrument: The instrument to get ticker for

        Returns:
            Dictionary with ticker data
        """
        pass

    @abc.abstractmethod
    def get_instrument_details(self, instrument: str) -> Dict[str, Any]:
        """
        Get details about an instrument.

        Args:
            instrument: The instrument to get details for

        Returns:
            Dictionary with instrument details
        """
        pass

    @abc.abstractmethod
    def get_orderbook(self, instrument: str, depth: int = 10) -> Dict[str, Any]:
        """
        Get current orderbook for an instrument.

        Args:
            instrument: The instrument to get orderbook for
            depth: Depth of the orderbook to retrieve

        Returns:
            Dictionary with orderbook data
        """
        pass

    @abc.abstractmethod
    def get_recent_trades(self, instrument: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent trades for an instrument.

        Args:
            instrument: The instrument to get trades for
            limit: Maximum number of trades to retrieve

        Returns:
            List of recent trades
        """
        pass

    @abc.abstractmethod
    def get_candles(
            self,
            instrument: str,
            timeframe: str,
            start: Optional[datetime] = None,
            end: Optional[datetime] = None,
            limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical candles for an instrument.

        Args:
            instrument: The instrument to get candles for
            timeframe: Candle timeframe (e.g., "1m", "1h", "1d")
            start: Start time
            end: End time
            limit: Maximum number of candles to retrieve

        Returns:
            List of candles
        """
        pass

    @abc.abstractmethod
    def get_exchange_info(self) -> Dict[str, Any]:
        """
        Get information about the exchange.

        Returns:
            Dictionary with exchange information
        """
        pass

    @abc.abstractmethod
    def get_server_time(self) -> datetime:
        """
        Get current server time from the exchange.

        Returns:
            Current server time
        """
        pass

    def supports_capability(self, capability: ExchangeCapabilities) -> bool:
        """
        Check if the exchange supports a specific capability.

        Args:
            capability: Capability to check

        Returns:
            True if supported, False otherwise
        """
        return capability in self.capabilities

    def get_capabilities(self) -> List[str]:
        """
        Get list of supported capabilities.

        Returns:
            List of capability names
        """
        return [cap.value for cap in self.capabilities]

    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the exchange gateway.

        Returns:
            Dictionary with status information
        """
        return {
            "exchange_id": self.exchange_id,
            "exchange_name": self.exchange_name,
            "exchange_type": self.exchange_type.value,
            "connected": self.connected,
            "last_connection_time": self.last_connection_time,
            "connection_error": str(self.connection_error) if self.connection_error else None,
            "capabilities": self.get_capabilities()
        }

    def format_instrument(self, base: str, quote: str) -> str:
        """
        Format a trading pair into the exchange's expected format.

        Args:
            base: Base currency/asset
            quote: Quote currency/asset

        Returns:
            Formatted instrument symbol
        """
        # Default implementation, override as needed for specific exchanges
        return f"{base}/{quote}"

    def parse_instrument(self, instrument: str) -> Tuple[str, str]:
        """
        Parse an instrument symbol into base and quote components.

        Args:
            instrument: Instrument symbol

        Returns:
            Tuple of (base, quote)
        """
        # Default implementation, override as needed for specific exchanges
        parts = instrument.split('/')
        if len(parts) >= 2:
            return parts[0], parts[1]
        return instrument, ""

    def convert_timeframe(self, timeframe: str) -> str:
        """
        Convert a standard timeframe string to exchange-specific format.

        Args:
            timeframe: Standard timeframe (e.g., "1m", "1h", "1d")

        Returns:
            Exchange-specific timeframe string
        """
        # Default implementation, override as needed for specific exchanges
        return timeframe

    def standardize_instrument(self, exchange_instrument: str) -> str:
        """
        Convert exchange-specific instrument format to standard format.

        Args:
            exchange_instrument: Exchange-specific instrument format

        Returns:
            Standardized instrument format
        """
        # Default implementation, override as needed for specific exchanges
        return exchange_instrument

    def _log_request(self, method: str, endpoint: str, params: Any = None, data: Any = None) -> None:
        """
        Log an API request for debugging.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            data: Request data
        """
        logger.debug(f"API Request: {method} {endpoint}")
        if params:
            logger.debug(f"Params: {params}")
        if data:
            logger.debug(f"Data: {data}")

    def _log_response(self, response: Any, timing_ms: float) -> None:
        """
        Log an API response for debugging.

        Args:
            response: API response
            timing_ms: Request timing in milliseconds
        """
        logger.debug(f"API Response: {response} ({timing_ms:.2f}ms)")

    def _handle_error(self, method: str, endpoint: str, error: Exception) -> None:
        """
        Handle and log an API error.

        Args:
            method: HTTP method
            endpoint: API endpoint
            error: The exception that occurred
        """
        logger.error(f"API Error in {method} {endpoint}: {str(error)}")
        self.connection_error = error