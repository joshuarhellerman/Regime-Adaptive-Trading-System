"""
oanda_gateway.py - Implementation of the ExchangeGateway interface for Oanda forex broker

This module provides connectivity to the Oanda REST API for forex trading.
It handles order placement, cancellation, retrieval, and position/account management.
"""

import time
import uuid
import logging
import requests
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue

# Import from core modules
from core.event_bus import EventBus, EventType
from core.state_manager import StateManager

# Import from data module
from data.market_data_service import MarketDataService

# Import from execution module
from execution.order.order import Order, OrderStatus, OrderType, TimeInForce, Side
from execution.order.order_book import OrderBook
from execution.fill.fill_model import Fill
from execution.exchange.exchange_gateway import ExchangeGateway
from execution.exchange.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class OandaOrderType(Enum):
    """Mapping of system order types to Oanda-specific order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    MARKET_IF_TOUCHED = "MARKET_IF_TOUCHED"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_STOP_LOSS = "TRAILING_STOP_LOSS"


class OandaTimeInForce(Enum):
    """Mapping of system TIF to Oanda-specific TIF"""
    GTC = "GTC"  # Good Till Cancelled
    GTD = "GTD"  # Good Till Date
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill


class OandaGateway(ExchangeGateway):
    """
    Oanda gateway implementation of the ExchangeGateway interface.

    This class provides connectivity to the Oanda REST API for forex trading.
    It handles order placement, cancellation, retrieval, and position/account management.
    """

    def __init__(
            self,
            api_key: str,
            account_id: str,
            base_url: str,
            practice: bool = True,
            market_data_service: Optional[MarketDataService] = None,
            state_manager: Optional[StateManager] = None,
            event_bus: Optional[EventBus] = None,
    ):
        # Initialize the base class with Oanda-specific information
        super().__init__(
            exchange_id="oanda",
            exchange_name="OANDA",
            exchange_type=ExchangeType.SPOT,
            capabilities=[
                ExchangeCapabilities.MARKET_ORDERS,
                ExchangeCapabilities.LIMIT_ORDERS,
                ExchangeCapabilities.STOP_ORDERS,
                ExchangeCapabilities.STREAMING_QUOTES,
                ExchangeCapabilities.HISTORICAL_DATA,
                ExchangeCapabilities.ACCOUNT_BALANCE,
                ExchangeCapabilities.POSITION_MANAGEMENT,
                ExchangeCapabilities.ORDER_HISTORY,
                ExchangeCapabilities.TRADE_HISTORY
            ],
            event_bus=event_bus
        )
        """
        Initialize the Oanda gateway.

        Args:
            api_key: Oanda API key for authentication
            account_id: Oanda account ID
            base_url: Base URL for Oanda API (differs for practice vs live)
            practice: Whether this is a practice account
            market_data_service: Service providing market data
            state_manager: System state manager
            event_bus: System event bus
        """
        self.api_key = api_key
        self.account_id = account_id
        self.base_url = base_url
        self.practice = practice
        self.market_data_service = market_data_service
        self.state_manager = state_manager
        self.event_bus = event_bus

        # API endpoints
        self.api_version = "v3"
        self.accounts_endpoint = f"{self.base_url}/{self.api_version}/accounts"
        self.account_endpoint = f"{self.accounts_endpoint}/{self.account_id}"
        self.orders_endpoint = f"{self.account_endpoint}/orders"
        self.positions_endpoint = f"{self.account_endpoint}/positions"
        self.instruments_endpoint = f"{self.base_url}/{self.api_version}/instruments"

        # Headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "RFC3339"
        }

        # Internal state
        self.connected = False
        self.open_orders: Dict[str, Order] = {}
        self.positions: Dict[str, Dict[str, Any]] = {}

        # Rate limiter to ensure we don't exceed API limits
        self.rate_limiter = RateLimiter(
            max_requests=120,  # Oanda limits to 120 requests per second
            time_window=60.0
        )

        # Order ID to Oanda order ID mapping
        self.order_id_map: Dict[str, str] = {}

        # Background update thread
        self.update_thread = None
        self.running = False

        logger.info(f"Oanda gateway initialized for account {account_id} (practice: {practice})")

    def connect(self) -> bool:
        """
        Establish connection to Oanda API.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            # Test connection by retrieving account details
            response = self._send_request("GET", self.account_endpoint)

            if response.status_code == 200:
                account_data = response.json()
                logger.info(f"Connected to Oanda account: {self.account_id}")
                logger.debug(f"Account details: {account_data}")

                # Start background update thread
                self.running = True
                self.update_thread = threading.Thread(
                    target=self._background_updates,
                    daemon=True,
                    name=f"oanda-updates-{self.account_id}"
                )
                self.update_thread.start()

                self.connected = True
                return True
            else:
                logger.error(f"Failed to connect to Oanda: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error connecting to Oanda: {e}", exc_info=True)
            return False

    def disconnect(self) -> bool:
        """
        Disconnect from Oanda API.

        Returns:
            True if disconnection is successful
        """
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5.0)

        self.connected = False
        logger.info("Disconnected from Oanda API")
        return True

    def place_order(self, order: Order) -> str:
        """
        Place an order on Oanda.

        Args:
            order: The order to place

        Returns:
            order_id: The exchange order ID

        Raises:
            Exception: If order placement fails
        """
        if not self.connected:
            raise Exception("Not connected to Oanda API")

        # Format the order for Oanda API
        oanda_order = self._format_order_for_oanda(order)

        # Send the request
        response = self._send_request("POST", self.orders_endpoint, json=oanda_order)

        if response.status_code == 201:  # Created
            order_response = response.json()
            oanda_order_id = order_response.get("orderCreateTransaction", {}).get("id")

            if not oanda_order_id:
                raise Exception(f"Failed to extract Oanda order ID from response: {order_response}")

            # Update order with exchange ID
            order.exchange_order_id = oanda_order_id
            order.status = OrderStatus.NEW
            order.timestamp_exchange_ack = datetime.now()

            # Store in open orders
            self.open_orders[oanda_order_id] = order
            self.order_id_map[order.id] = oanda_order_id

            # Emit event
            if self.event_bus:
                self.event_bus.emit(
                    EventType.ORDER_NEW,
                    {
                        "order": order,
                        "exchange_id": "oanda",
                        "timestamp": datetime.now()
                    }
                )

            logger.info(f"Order placed on Oanda: {oanda_order_id}")
            return oanda_order_id
        else:
            error_msg = f"Failed to place order on Oanda: {response.status_code} - {response.text}"
            logger.error(error_msg)

            # Mark order as rejected
            order.status = OrderStatus.REJECTED
            order.status_message = error_msg

            # Emit event
            if self.event_bus:
                self.event_bus.emit(
                    EventType.ORDER_REJECTED,
                    {
                        "order": order,
                        "exchange_id": "oanda",
                        "reason": error_msg,
                        "timestamp": datetime.now()
                    }
                )

            raise Exception(error_msg)

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order on Oanda.

        Args:
            order_id: The ID of the order to cancel

        Returns:
            True if cancellation is successful, False otherwise
        """
        if not self.connected:
            logger.error("Not connected to Oanda API")
            return False

        # Look up Oanda order ID
        oanda_order_id = self.order_id_map.get(order_id)
        if not oanda_order_id:
            logger.warning(f"Cannot find Oanda order ID for order {order_id}")
            return False

        # Send cancel request
        cancel_endpoint = f"{self.orders_endpoint}/{oanda_order_id}/cancel"
        response = self._send_request("PUT", cancel_endpoint)

        if response.status_code == 200:
            cancel_response = response.json()
            logger.info(f"Order {oanda_order_id} canceled successfully")

            # Update order status
            if oanda_order_id in self.open_orders:
                order = self.open_orders[oanda_order_id]
                order.status = OrderStatus.CANCELED
                order.timestamp_updated = datetime.now()

                # Emit event
                if self.event_bus:
                    self.event_bus.emit(
                        EventType.ORDER_CANCELED,
                        {
                            "order": order,
                            "exchange_id": "oanda",
                            "timestamp": datetime.now()
                        }
                    )

            return True
        else:
            error_msg = f"Failed to cancel order {oanda_order_id}: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return False

    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order details from Oanda.

        Args:
            order_id: The ID of the order to retrieve

        Returns:
            The order if found, None otherwise
        """
        if not self.connected:
            logger.error("Not connected to Oanda API")
            return None

        # Look up Oanda order ID
        oanda_order_id = self.order_id_map.get(order_id)
        if not oanda_order_id:
            logger.warning(f"Cannot find Oanda order ID for order {order_id}")
            return None

        # Fetch order details
        order_endpoint = f"{self.orders_endpoint}/{oanda_order_id}"
        response = self._send_request("GET", order_endpoint)

        if response.status_code == 200:
            order_response = response.json()
            order_data = order_response.get("order", {})

            if oanda_order_id in self.open_orders:
                # Update existing order
                order = self.open_orders[oanda_order_id]
                self._update_order_from_oanda(order, order_data)
                return order.copy()
            else:
                # Create new order object
                order = self._create_order_from_oanda(order_data)
                if order:
                    self.open_orders[oanda_order_id] = order
                    self.order_id_map[order.id] = oanda_order_id
                return order
        else:
            logger.error(f"Failed to get order {oanda_order_id}: {response.status_code} - {response.text}")
            return None

    def get_open_orders(self) -> List[Order]:
        """
        Get all open orders from Oanda.

        Returns:
            List of open orders
        """
        if not self.connected:
            logger.error("Not connected to Oanda API")
            return []

        # Fetch open orders
        response = self._send_request("GET", self.orders_endpoint)

        if response.status_code == 200:
            orders_response = response.json()
            orders_data = orders_response.get("orders", [])

            open_orders = []
            for order_data in orders_data:
                oanda_order_id = order_data.get("id")

                if oanda_order_id in self.open_orders:
                    # Update existing order
                    order = self.open_orders[oanda_order_id]
                    self._update_order_from_oanda(order, order_data)
                    open_orders.append(order.copy())
                else:
                    # Create new order object
                    order = self._create_order_from_oanda(order_data)
                    if order:
                        self.open_orders[oanda_order_id] = order
                        self.order_id_map[order.id] = oanda_order_id
                        open_orders.append(order.copy())

            return open_orders
        else:
            logger.error(f"Failed to get open orders: {response.status_code} - {response.text}")
            return []

    def get_position(self, instrument: str) -> float:
        """
        Get current position for an instrument.

        Args:
            instrument: The instrument to get position for

        Returns:
            Current position size (positive for long, negative for short)
        """
        if not self.connected:
            logger.error("Not connected to Oanda API")
            return 0.0

        # Format instrument for Oanda
        oanda_instrument = self._format_instrument_for_oanda(instrument)

        # Fetch position details
        position_endpoint = f"{self.positions_endpoint}/{oanda_instrument}"
        response = self._send_request("GET", position_endpoint)

        if response.status_code == 200:
            position_response = response.json()
            position_data = position_response.get("position", {})

            # Update internal position tracking
            self.positions[instrument] = position_data

            # Calculate net position
            long_units = float(position_data.get("long", {}).get("units", 0))
            short_units = float(position_data.get("short", {}).get("units", 0))

            # Oanda reports short units as negative
            return long_units + short_units
        elif response.status_code == 404:
            # No position for this instrument
            return 0.0
        else:
            logger.error(f"Failed to get position for {instrument}: {response.status_code} - {response.text}")
            return 0.0

    def get_balance(self, currency: str) -> float:
        """
        Get current balance for a currency.

        Args:
            currency: The currency to get balance for

        Returns:
            Current balance
        """
        if not self.connected:
            logger.error("Not connected to Oanda API")
            return 0.0

        # Fetch account details
        response = self._send_request("GET", self.account_endpoint)

        if response.status_code == 200:
            account_data = response.json()
            account = account_data.get("account", {})

            # Oanda accounts typically have a single currency
            account_currency = account.get("currency")

            if account_currency == currency:
                balance = float(account.get("balance", 0))
                return balance
            else:
                logger.warning(f"Requested balance for {currency}, but account currency is {account_currency}")
                return 0.0
        else:
            logger.error(f"Failed to get account details: {response.status_code} - {response.text}")
            return 0.0

    def get_all_balances(self) -> Dict[str, float]:
        """
        Get all current balances.

        Returns:
            Dictionary mapping currencies to balances
        """
        if not self.connected:
            logger.error("Not connected to Oanda API")
            return {}

        # Fetch account details
        response = self._send_request("GET", self.account_endpoint)

        if response.status_code == 200:
            account_data = response.json()
            account = account_data.get("account", {})

            # Oanda accounts typically have a single currency
            account_currency = account.get("currency", "")
            balance = float(account.get("balance", 0))

            return {account_currency: balance}
        else:
            logger.error(f"Failed to get account details: {response.status_code} - {response.text}")
            return {}

    def get_all_positions(self) -> Dict[str, float]:
        """
        Get all current positions.

        Returns:
            Dictionary mapping instruments to position sizes
        """
        if not self.connected:
            logger.error("Not connected to Oanda API")
            return {}

        # Fetch positions
        response = self._send_request("GET", self.positions_endpoint)

        if response.status_code == 200:
            positions_response = response.json()
            positions_data = positions_response.get("positions", [])

            # Format into dictionary
            result = {}
            for position_data in positions_data:
                oanda_instrument = position_data.get("instrument")
                instrument = self._parse_oanda_instrument(oanda_instrument)

                # Calculate net position
                long_units = float(position_data.get("long", {}).get("units", 0))
                short_units = float(position_data.get("short", {}).get("units", 0))
                net_position = long_units + short_units  # short units are already negative

                if net_position != 0:
                    result[instrument] = net_position

            return result
        else:
            logger.error(f"Failed to get positions: {response.status_code} - {response.text}")
            return {}

    def is_connected(self) -> bool:
        """
        Check if currently connected to the exchange.

        Returns:
            True if connected, False otherwise
        """
        return self.connected

    def get_ticker(self, instrument: str) -> Dict[str, Any]:
        """
        Get current ticker data for an instrument.

        Args:
            instrument: The instrument to get ticker for

        Returns:
            Dictionary with ticker data
        """
        if not self.connected:
            logger.error("Not connected to Oanda API")
            return {}

        # Format instrument for Oanda
        oanda_instrument = self._format_instrument_for_oanda(instrument)

        # Get pricing data
        pricing_endpoint = f"{self.base_url}/{self.api_version}/accounts/{self.account_id}/pricing"
        params = {
            "instruments": oanda_instrument
        }

        response = self._send_request("GET", pricing_endpoint, params=params)

        if response.status_code == 200:
            pricing_data = response.json()
            prices = pricing_data.get("prices", [])

            if not prices:
                logger.warning(f"No pricing data available for {instrument}")
                return {}

            price_data = prices[0]

            # Format ticker data
            # Calculate mid price if both bid and ask are available
            bid = float(price_data.get("bids", [{}])[0].get("price", 0)) if price_data.get("bids") else None
            ask = float(price_data.get("asks", [{}])[0].get("price", 0)) if price_data.get("asks") else None
            mid = (bid + ask) / 2 if bid is not None and ask is not None else None

            ticker = {
                "instrument": instrument,
                "time": price_data.get("time"),
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "status": price_data.get("status"),
                "tradeable": price_data.get("tradeable", False)
            }

            return ticker
        else:
            logger.error(f"Failed to get ticker for {instrument}: {response.status_code} - {response.text}")
            return {}

    def _format_order_for_oanda(self, order: Order) -> Dict[str, Any]:
        """
        Format an order for the Oanda API.

        Args:
            order: The order to format

        Returns:
            Oanda API order structure
        """
        # Convert instrument to Oanda format
        instrument = self._format_instrument_for_oanda(order.instrument)

        # Map order type to Oanda type
        if order.type == OrderType.MARKET:
            oanda_type = OandaOrderType.MARKET.value
        elif order.type == OrderType.LIMIT:
            oanda_type = OandaOrderType.LIMIT.value
        elif order.type == OrderType.STOP:
            oanda_type = OandaOrderType.STOP.value
        else:
            raise ValueError(f"Unsupported order type: {order.type}")

        # Map time in force to Oanda TIF
        if order.time_in_force == TimeInForce.GTC:
            oanda_tif = OandaTimeInForce.GTC.value
        elif order.time_in_force == TimeInForce.IOC:
            oanda_tif = OandaTimeInForce.IOC.value
        elif order.time_in_force == TimeInForce.FOK:
            oanda_tif = OandaTimeInForce.FOK.value
        else:
            oanda_tif = OandaTimeInForce.GTC.value  # Default to GTC

        # Basic order structure
        oanda_order = {
            "order": {
                "type": oanda_type,
                "instrument": instrument,
                "units": str(order.quantity) if order.side == Side.BUY else str(-order.quantity),
                "timeInForce": oanda_tif,
                "positionFill": "DEFAULT"
            }
        }

        # Add price for limit and stop orders
        if order.type == OrderType.LIMIT:
            oanda_order["order"]["price"] = str(order.price)
        elif order.type == OrderType.STOP:
            oanda_order["order"]["price"] = str(order.price)

        # Add client order ID if present
        if order.id:
            oanda_order["order"]["clientExtensions"] = {
                "id": order.id,
                "tag": "system"
            }

        return oanda_order

    def _format_instrument_for_oanda(self, instrument: str) -> str:
        """
        Format instrument for Oanda API.

        Args:
            instrument: System instrument format (e.g., "EUR/USD")

        Returns:
            Oanda formatted instrument (e.g., "EUR_USD")
        """
        return instrument.replace("/", "_")

    def _parse_oanda_instrument(self, oanda_instrument: str) -> str:
        """
        Parse Oanda instrument format to system format.

        Args:
            oanda_instrument: Oanda instrument format (e.g., "EUR_USD")

        Returns:
            System instrument format (e.g., "EUR/USD")
        """
        return oanda_instrument.replace("_", "/")

    def _create_order_from_oanda(self, order_data: Dict[str, Any]) -> Optional[Order]:
        """
        Create an Order object from Oanda order data.

        Args:
            order_data: Oanda order details

        Returns:
            Order object or None if conversion fails
        """
        try:
            # Extract basic order info
            oanda_order_id = order_data.get("id")
            oanda_instrument = order_data.get("instrument")
            units = float(order_data.get("units", 0))
            oanda_type = order_data.get("type")

            # Convert Oanda instrument to system format
            instrument = self._parse_oanda_instrument(oanda_instrument)

            # Determine order side from units
            side = Side.BUY if units > 0 else Side.SELL

            # Map Oanda order type to system type
            if oanda_type == OandaOrderType.MARKET.value:
                order_type = OrderType.MARKET
            elif oanda_type == OandaOrderType.LIMIT.value:
                order_type = OrderType.LIMIT
            elif oanda_type == OandaOrderType.STOP.value:
                order_type = OrderType.STOP
            else:
                order_type = OrderType.MARKET  # Default

            # Extract price if present
            price = float(order_data.get("price", 0))

            # Map Oanda status to system status
            oanda_state = order_data.get("state")
            if oanda_state == "PENDING":
                status = OrderStatus.PENDING
            elif oanda_state == "FILLED":
                status = OrderStatus.FILLED
            elif oanda_state == "CANCELLED":
                status = OrderStatus.CANCELED
            elif oanda_state == "TRIGGERED":
                status = OrderStatus.NEW
            elif oanda_state == "PARTIALLY_FILLED":
                status = OrderStatus.PARTIALLY_FILLED
            else:
                status = OrderStatus.NEW  # Default

            # Extract client order ID if present
            client_extensions = order_data.get("clientExtensions", {})
            client_id = client_extensions.get("id")

            # Create Order object
            order = Order(
                id=client_id or str(uuid.uuid4()),
                instrument=instrument,
                quantity=abs(units),
                side=side,
                type=order_type,
                price=price,
                status=status,
                exchange_order_id=oanda_order_id
            )

            # Additional fields if present
            filled_units = float(order_data.get("filledUnits", 0))
            order.filled_quantity = abs(filled_units)

            # Parse timestamps
            create_time = order_data.get("createTime")
            if create_time:
                order.timestamp_created = datetime.fromisoformat(create_time.replace("Z", "+00:00"))

            fill_time = order_data.get("fillingTransactionID")
            if fill_time:
                order.timestamp_updated = datetime.fromisoformat(fill_time.replace("Z", "+00:00"))

            return order

        except Exception as e:
            logger.error(f"Error creating order from Oanda data: {e}", exc_info=True)
            return None

    def _update_order_from_oanda(self, order: Order, order_data: Dict[str, Any]) -> None:
        """
        Update an existing order with data from Oanda.

        Args:
            order: Order to update
            order_data: Oanda order details
        """
        try:
            # Map Oanda status to system status
            oanda_state = order_data.get("state")
            if oanda_state == "PENDING":
                order.status = OrderStatus.PENDING
            elif oanda_state == "FILLED":
                order.status = OrderStatus.FILLED
            elif oanda_state == "CANCELLED":
                order.status = OrderStatus.CANCELED
            elif oanda_state == "TRIGGERED":
                order.status = OrderStatus.NEW
            elif oanda_state == "PARTIALLY_FILLED":
                order.status = OrderStatus.PARTIALLY_FILLED

            # Update filled quantity
            filled_units = float(order_data.get("filledUnits", 0))
            order.filled_quantity = abs(filled_units)

            # Update timestamps
            fill_time = order_data.get("fillingTransactionID")
            if fill_time:
                order.timestamp_updated = datetime.fromisoformat(fill_time.replace("Z", "+00:00"))

        except Exception as e:
            logger.error(f"Error updating order from Oanda data: {e}", exc_info=True)

    def _background_updates(self) -> None:
        """Background thread for regular updates from Oanda."""
        update_interval = 5.0  # Seconds between updates

        while self.running:
            try:
                # Update open orders
                self._update_open_orders()

                # Update positions
                self._update_positions()

                # Sleep until next update
                time.sleep(update_interval)

            except Exception as e:
                logger.error(f"Error in background updates: {e}", exc_info=True)
                time.sleep(update_interval)  # Sleep and try again

    def _update_open_orders(self) -> None:
        """Update status of all open orders."""
        try:
            # Fetch open orders
            response = self._send_request("GET", self.orders_endpoint)

            if response.status_code == 200:
                orders_response = response.json()
                orders_data = orders_response.get("orders", [])

                for order_data in orders_data:
                    oanda_order_id = order_data.get("id")

                    if oanda_order_id in self.open_orders:
                        # Update existing order
                        order = self.open_orders[oanda_order_id]
                        old_status = order.status
                        old_filled = order.filled_quantity

                        self._update_order_from_oanda(order, order_data)

                        # Check for status changes and emit events
                        if self.event_bus and order.status != old_status:
                            if order.status == OrderStatus.FILLED:
                                # Calculate fill details
                                fill_qty = order.quantity - old_filled
                                fill_price = float(order_data.get("price", 0))

                                # Create fill object
                                fill = Fill(
                                    id=str(uuid.uuid4()),
                                    order_id=order.id,
                                    exchange_order_id=oanda_order_id,
                                    instrument=order.instrument,
                                    side=order.side,
                                    quantity=fill_qty,
                                    price=fill_price,
                                    fee=0.0,  # Oanda doesn't report fees directly
                                    fee_currency=self._get_quote_currency(order.instrument),
                                    timestamp=order.timestamp_updated or datetime.now(),
                                    is_maker=False
                                )

                                self.event_bus.emit(
                                    EventType.ORDER_FILLED,
                                    {
                                        "order": order,
                                        "fill": fill,
                                        "exchange_id": "oanda",
                                        "timestamp": datetime.now()
                                    }
                                )

                            elif order.status == OrderStatus.PARTIALLY_FILLED:
                                # Calculate fill details
                                fill_qty = order.filled_quantity - old_filled
                                fill_price = float(order_data.get("price", 0))

                                # Create fill object
                                fill = Fill(
                                    id=str(uuid.uuid4()),
                                    order_id=order.id,
                                    exchange_order_id=oanda_order_id,
                                    instrument=order.instrument,
                                    side=order.side,
                                    quantity=fill_qty,
                                    price=fill_price,
                                    fee=0.0,  # Oanda doesn't report fees directly
                                    fee_currency=self._get_quote_currency(order.instrument),
                                    timestamp=order.timestamp_updated or datetime.now(),
                                    is_maker=False
                                )

                                self.event_bus.emit(
                                    EventType.ORDER_PARTIALLY_FILLED,
                                    {
                                        "order": order,
                                        "fill": fill,
                                        "exchange_id": "oanda",
                                        "timestamp": datetime.now()
                                    }
                                )

                            elif order.status == OrderStatus.CANCELED:
                                self.event_bus.emit(
                                    EventType.ORDER_CANCELED,
                                    {
                                        "order": order,
                                        "exchange_id": "oanda",
                                        "timestamp": datetime.now()
                                    }
                                )
                    else:
                        # New order we didn't know about
                        order = self._create_order_from_oanda(order_data)
                        if order:
                            self.open_orders[oanda_order_id] = order
                            self.order_id_map[order.id] = oanda_order_id
            else:
                logger.error(f"Failed to get open orders: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Error updating open orders: {e}", exc_info=True)

    def _update_positions(self) -> None:
        """Update all positions."""
        try:
            # Fetch positions
            response = self._send_request("GET", self.positions_endpoint)

            if response.status_code == 200:
                positions_response = response.json()
                positions_data = positions_response.get("positions", [])

                # Clear current positions
                self.positions.clear()

                for position_data in positions_data:
                    oanda_instrument = position_data.get("instrument")
                    instrument = self._parse_oanda_instrument(oanda_instrument)

                    # Store position data
                    self.positions[instrument] = position_data

                    # Process position if needed
                    long_units = float(position_data.get("long", {}).get("units", 0))
                    short_units = float(position_data.get("short", {}).get("units", 0))

                    # Could emit position update events here if needed

            else:
                logger.error(f"Failed to get positions: {response.status_code} - {response.text}")

        except Exception as e:
            logger.error(f"Error updating positions: {e}", exc_info=True)

    def _send_request(
            self,
            method: str,
            url: str,
            params: Optional[Dict[str, Any]] = None,
            json: Optional[Dict[str, Any]] = None,
            headers: Optional[Dict[str, str]] = None
    ) -> requests.Response:
        """
        Send a request to the Oanda API with rate limiting.

        Args:
            method: HTTP method
            url: Request URL
            params: URL parameters
            json: JSON body
            headers: Additional headers

        Returns:
            Response object
        """
        # Apply rate limiting
        self.rate_limiter.acquire()

        # Merge headers
        request_headers = self.headers.copy()
        if headers:
            request_headers.update(headers)

        # Send the request
        try:
            response = requests.request(
                method=method,
                url=url,
                params=params,
                json=json,
                headers=request_headers,
                timeout=30  # 30 second timeout
            )

            # Check for rate limit errors
            if response.status_code == 429:
                logger.warning(f"Rate limit exceeded: {response.text}")
                if self.rate_limiter:
                    self.rate_limiter.report_error(is_rate_limit_error=True)

            # Check for other errors that might need backoff
            elif response.status_code >= 500:
                logger.warning(f"Server error: {response.status_code} - {response.text}")
                if self.rate_limiter:
                    self.rate_limiter.report_error(is_rate_limit_error=False)

            # Success case
            elif response.status_code < 400:
                if self.rate_limiter:
                    self.rate_limiter.report_success()

            return response

        except Exception as e:
            logger.error(f"Request error: {e}", exc_info=True)
            raise