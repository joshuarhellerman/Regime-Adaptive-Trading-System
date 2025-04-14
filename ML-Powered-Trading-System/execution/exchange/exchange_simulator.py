"""
exchange_simulator.py - Realistic exchange simulation for paper trading

This module provides a high-fidelity simulation of exchange behavior for paper trading.
It implements realistic market dynamics, latency, fills, fees, and liquidity constraints.
The simulator is designed to work with any exchange gateway and be adaptable to different
market types (forex, crypto, equities, etc.).
"""

import time
import uuid
import random
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import queue

# Import from core modules
from core.event_bus import EventBus, EventType
from core.state_manager import StateManager

# Import from data module
from data.market_data_service import MarketDataService
from data.storage.market_snapshot import MarketSnapshot

# Import from execution module
from execution.order.order import Order, OrderStatus, OrderType, TimeInForce, Side
from execution.order.order_book import OrderBook
from execution.fill.fill_model import Fill
from execution.exchange.exchange_gateway import ExchangeGateway

logger = logging.getLogger(__name__)

class SimulationMode(Enum):
    """Enum for different simulation fidelity modes."""
    BASIC = 1       # Simple, instant fills at mid price
    REALISTIC = 2   # Realistic fills with market impact
    DETAILED = 3    # Highly detailed with order book simulation

@dataclass
class SimulationConfig:
    """Configuration for the exchange simulator."""
    mode: SimulationMode = SimulationMode.REALISTIC

    # Latency simulation (milliseconds)
    base_latency_ms: int = 50  # Base latency
    latency_stdev_ms: int = 10  # Standard deviation of latency
    exchange_processing_ms: int = 25  # Additional exchange processing time

    # Fill simulation
    partial_fill_probability: float = 0.3  # Probability of partial fills
    min_fill_ratio: float = 0.6  # Minimum ratio for partial fills

    # Market impact (price slippage)
    market_impact_factor: float = 0.2  # Higher = more impact
    liquidity_factor: Dict[str, float] = field(default_factory=dict)  # Per-instrument liquidity

    # Rejection simulation
    rejection_probability: float = 0.01  # Probability of order rejection

    # Fee structure
    maker_fee_rate: float = 0.0010  # 0.10% (10 bps) for maker orders
    taker_fee_rate: float = 0.0015  # 0.15% (15 bps) for taker orders

    # Order book depth simulation
    simulate_order_book: bool = True  # Whether to simulate order book
    book_depth: int = 10  # Number of levels to simulate

    # Market conditions
    volatility_factor: float = 1.0  # Adjust for more/less volatile simulation

    # Time acceleration (for faster simulation if needed)
    time_acceleration: float = 1.0  # 1.0 = real-time, 2.0 = 2x speed

    # Market-specific configurations
    forex_pip_value: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default values for configs."""
        # Default liquidity factors for different asset classes if not provided
        if not self.liquidity_factor:
            self.liquidity_factor = {
                "default": 1.0,
                "forex": 2.0,    # Higher liquidity in forex
                "crypto": 0.5,   # Lower liquidity in crypto
                "equities": 1.5  # Medium-high liquidity in equities
            }

        # Default pip values for major forex pairs if not provided
        if not self.forex_pip_value:
            self.forex_pip_value = {
                "EUR/USD": 0.0001,
                "GBP/USD": 0.0001,
                "USD/JPY": 0.01,
                "USD/CHF": 0.0001,
                "AUD/USD": 0.0001,
                "USD/CAD": 0.0001,
                "NZD/USD": 0.0001,
                "default": 0.0001  # Default pip value
            }


class ExchangeSimulator(ExchangeGateway):
    """
    Exchange simulator that provides realistic paper trading functionality.

    This simulator mimics real exchange behavior including:
    - Market impact and slippage
    - Realistic fill patterns (partial fills)
    - Exchange latency
    - Fee structures
    - Order book dynamics
    - Exchange-specific quirks

    It's designed to be compatible with the ExchangeGateway interface
    to allow seamless switching between paper and live trading.
    """

    def __init__(
        self,
        exchange_id: str,
        market_data_service: MarketDataService,
        state_manager: StateManager,
        event_bus: EventBus,
        config: Optional[SimulationConfig] = None
    ):
        """
        Initialize the exchange simulator.

        Args:
            exchange_id: Unique identifier for this simulator instance
            market_data_service: Service providing market data
            state_manager: System state manager
            event_bus: System event bus
            config: Simulation configuration parameters
        """
        self.exchange_id = exchange_id
        self.market_data_service = market_data_service
        self.state_manager = state_manager
        self.event_bus = event_bus
        self.config = config or SimulationConfig()

        # Track all open orders
        self.open_orders: Dict[str, Order] = {}

        # Track positions and balances
        self.positions: Dict[str, float] = {}
        self.balances: Dict[str, float] = {}

        # Simulated order books (price -> volume)
        self.order_books: Dict[str, OrderBook] = {}

        # Internal processing queue
        self.order_queue = queue.Queue()

        # Start processing thread
        self.running = True
        self.processing_thread = threading.Thread(
            target=self._process_order_queue,
            daemon=True,
            name=f"exchange-sim-{exchange_id}"
        )
        self.processing_thread.start()

        # Subscribe to market data updates
        # This will allow the simulator to update its internal state based on market data
        self.event_bus.subscribe(
            EventType.MARKET_DATA_UPDATE,
            self._handle_market_data_update
        )

        logger.info(f"Exchange simulator initialized for {exchange_id}")

    def connect(self) -> bool:
        """
        Simulate establishing a connection to the exchange.

        Returns:
            True if connection is successful (always in simulation)
        """
        logger.info(f"Simulated connection established to {self.exchange_id}")
        return True

    def disconnect(self) -> bool:
        """
        Simulate disconnecting from the exchange.

        Returns:
            True if disconnection is successful
        """
        self.running = False
        self.processing_thread.join(timeout=2.0)
        logger.info(f"Simulated connection to {self.exchange_id} closed")
        return True

    def place_order(self, order: Order) -> str:
        """
        Place an order on the simulated exchange.

        Args:
            order: The order to place

        Returns:
            order_id: The exchange order ID (generated by simulator)
        """
        # Generate exchange-specific order ID if not present
        if not order.exchange_order_id:
            order.exchange_order_id = f"sim_{self.exchange_id}_{uuid.uuid4()}"

        # Clone the order to avoid external mutation
        sim_order = order.copy()

        # Update status
        sim_order.status = OrderStatus.PENDING
        sim_order.timestamp_created = datetime.now()

        # Add to open orders
        self.open_orders[sim_order.exchange_order_id] = sim_order

        # Simulate exchange latency before acknowledgement
        latency = self._simulate_latency()

        # Put in processing queue with simulated exchange latency
        self.order_queue.put((sim_order, time.time() + latency / 1000))

        logger.debug(f"Order placed in simulator queue: {sim_order.exchange_order_id}, "
                    f"processing in {latency}ms")

        # Emit event for order acceptance
        self.event_bus.emit(
            EventType.ORDER_ACCEPTED,
            {
                "order": sim_order,
                "exchange_id": self.exchange_id,
                "timestamp": datetime.now()
            }
        )

        return sim_order.exchange_order_id

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order on the simulated exchange.

        Args:
            order_id: The ID of the order to cancel

        Returns:
            True if cancellation is successful, False otherwise
        """
        if order_id not in self.open_orders:
            logger.warning(f"Cannot cancel order {order_id}: not found in open orders")
            return False

        order = self.open_orders[order_id]

        # If order is already in terminal state, cannot cancel
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]:
            logger.warning(f"Cannot cancel order {order_id}: already in state {order.status}")
            return False

        # Simulate exchange latency for cancellation
        latency = self._simulate_latency()

        # Schedule the cancellation
        self.order_queue.put(("cancel", order_id, time.time() + latency / 1000))

        logger.debug(f"Cancellation request queued for {order_id}, "
                    f"processing in {latency}ms")

        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order details from the simulated exchange.

        Args:
            order_id: The ID of the order to retrieve

        Returns:
            The order if found, None otherwise
        """
        if order_id in self.open_orders:
            # Return a copy to prevent external mutation
            return self.open_orders[order_id].copy()
        return None

    def get_open_orders(self) -> List[Order]:
        """
        Get all open orders from the simulated exchange.

        Returns:
            List of open orders
        """
        return [
            order.copy() for order in self.open_orders.values()
            if order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]
        ]

    def get_position(self, instrument: str) -> float:
        """
        Get current position for an instrument.

        Args:
            instrument: The instrument to get position for

        Returns:
            Current position size (positive for long, negative for short)
        """
        return self.positions.get(instrument, 0.0)

    def get_balance(self, currency: str) -> float:
        """
        Get current balance for a currency.

        Args:
            currency: The currency to get balance for

        Returns:
            Current balance
        """
        return self.balances.get(currency, 0.0)

    def set_balance(self, currency: str, amount: float) -> None:
        """
        Set balance for paper trading simulation.

        Args:
            currency: The currency to set balance for
            amount: The balance amount
        """
        self.balances[currency] = amount
        logger.info(f"Set simulated balance for {currency}: {amount}")

    def _process_order_queue(self) -> None:
        """Process the order queue in a separate thread."""
        while self.running:
            try:
                # Get next item from queue, but don't block indefinitely
                try:
                    item = self.order_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Process based on item type
                if isinstance(item, tuple) and item[0] == "cancel":
                    _, order_id, process_time = item
                    # Wait until processing time
                    self._wait_until(process_time)
                    self._process_cancellation(order_id)
                else:
                    order, process_time = item
                    # Wait until processing time
                    self._wait_until(process_time)
                    self._process_order(order)

                # Mark as done
                self.order_queue.task_done()

            except Exception as e:
                logger.error(f"Error in order queue processing: {e}", exc_info=True)

    def _wait_until(self, target_time: float) -> None:
        """Wait until the specified time."""
        now = time.time()
        if now < target_time:
            time.sleep((target_time - now) / self.config.time_acceleration)

    def _process_order(self, order: Order) -> None:
        """
        Process an order through the simulated exchange.

        Args:
            order: The order to process
        """
        # Check if order should be rejected
        if random.random() < self.config.rejection_probability:
            self._reject_order(order, "Simulated random rejection")
            return

        # Get current market data
        try:
            market_data = self.market_data_service.get_last_price(order.instrument)
        except Exception as e:
            self._reject_order(order, f"Failed to get market data: {e}")
            return

        # Update order status to NEW (acknowledged by exchange)
        order.status = OrderStatus.NEW
        order.timestamp_exchange_ack = datetime.now()

        # Emit order accepted event
        self.event_bus.emit(
            EventType.ORDER_NEW,
            {
                "order": order,
                "exchange_id": self.exchange_id,
                "timestamp": datetime.now()
            }
        )

        # Process order based on type
        if order.type == OrderType.MARKET:
            self._process_market_order(order, market_data)
        elif order.type == OrderType.LIMIT:
            self._process_limit_order(order, market_data)
        else:
            # For other order types like STOP, STOP_LIMIT, etc.
            # (implementation would be added based on requirements)
            self._reject_order(order, f"Order type {order.type} not supported yet")

    def _process_market_order(self, order: Order, market_data: Dict[str, Any]) -> None:
        """
        Process a market order.

        Args:
            order: The market order to process
            market_data: Current market data
        """
        # Calculate execution price with slippage
        execution_price = self._calculate_execution_price(
            order.instrument,
            order.side,
            order.quantity,
            market_data
        )

        # Determine if order will be filled completely or partially
        if random.random() < self.config.partial_fill_probability:
            # Simulate partial fill
            fill_ratio = random.uniform(
                self.config.min_fill_ratio,
                0.99  # Max partial fill ratio
            )
            filled_qty = order.quantity * fill_ratio

            # Create partial fill
            self._create_fill(
                order,
                filled_qty,
                execution_price,
                is_partial=True
            )

            # Schedule remainder for later fills
            remaining_qty = order.quantity - filled_qty
            if remaining_qty > 0:
                # Clone original order with remaining quantity
                remainder_order = order.copy()
                remainder_order.quantity = remaining_qty

                # Schedule fill with additional delay
                additional_delay = self._simulate_latency() * 2
                self.order_queue.put(
                    (remainder_order, time.time() + additional_delay / 1000)
                )

                logger.debug(f"Scheduled remaining fill for {order.exchange_order_id}, "
                           f"qty: {remaining_qty}, delay: {additional_delay}ms")
        else:
            # Simulate complete fill
            self._create_fill(
                order,
                order.quantity,
                execution_price,
                is_partial=False
            )

    def _process_limit_order(self, order: Order, market_data: Dict[str, Any]) -> None:
        """
        Process a limit order.

        Args:
            order: The limit order to process
            market_data: Current market data
        """
        current_price = (
            market_data.get("bid", 0) if order.side == Side.SELL else
            market_data.get("ask", 0)
        )

        # Check if limit order would execute immediately
        if ((order.side == Side.BUY and order.price >= current_price) or
            (order.side == Side.SELL and order.price <= current_price)):

            # Immediate execution - similar to market order but respects limit price
            execution_price = min(order.price, current_price) if order.side == Side.BUY else \
                             max(order.price, current_price)

            # Apply some randomness to execution price (within limit constraints)
            if order.side == Side.BUY:
                execution_price = min(
                    order.price,  # Never execute above limit for buy
                    execution_price * (1 + random.uniform(0, 0.0002))  # Small random factor
                )
            else:  # SELL
                execution_price = max(
                    order.price,  # Never execute below limit for sell
                    execution_price * (1 - random.uniform(0, 0.0002))  # Small random factor
                )

            # Determine if order will be filled completely or partially
            if random.random() < self.config.partial_fill_probability:
                # Simulate partial fill
                fill_ratio = random.uniform(
                    self.config.min_fill_ratio,
                    0.99  # Max partial fill ratio
                )
                filled_qty = order.quantity * fill_ratio

                # Create partial fill
                self._create_fill(
                    order,
                    filled_qty,
                    execution_price,
                    is_partial=True
                )

                # Remaining quantity stays in open orders
                order.quantity -= filled_qty
            else:
                # Simulate complete fill
                self._create_fill(
                    order,
                    order.quantity,
                    execution_price,
                    is_partial=False
                )
        else:
            # Limit not hit immediately, order remains open
            logger.debug(f"Limit order {order.exchange_order_id} placed in book, "
                       f"waiting for price {order.price}")

            # For simulation purposes, we'll periodically check if the order
            # should be filled based on market data updates
            pass

    def _process_cancellation(self, order_id: str) -> None:
        """
        Process an order cancellation.

        Args:
            order_id: The ID of the order to cancel
        """
        if order_id not in self.open_orders:
            logger.warning(f"Cannot cancel order {order_id}: not found")
            return

        order = self.open_orders[order_id]

        # Check if order can be canceled
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED]:
            logger.warning(f"Cannot cancel order {order_id}: already in state {order.status}")
            return

        # Update order status
        order.status = OrderStatus.CANCELED
        order.timestamp_updated = datetime.now()

        # Emit cancellation event
        self.event_bus.emit(
            EventType.ORDER_CANCELED,
            {
                "order": order,
                "exchange_id": self.exchange_id,
                "timestamp": datetime.now()
            }
        )

        logger.info(f"Order {order_id} canceled successfully")

    def _reject_order(self, order: Order, reason: str) -> None:
        """
        Reject an order with the given reason.

        Args:
            order: The order to reject
            reason: The reason for rejection
        """
        # Update order status
        order.status = OrderStatus.REJECTED
        order.status_message = reason
        order.timestamp_updated = datetime.now()

        # Emit rejection event
        self.event_bus.emit(
            EventType.ORDER_REJECTED,
            {
                "order": order,
                "exchange_id": self.exchange_id,
                "reason": reason,
                "timestamp": datetime.now()
            }
        )

        logger.warning(f"Order {order.exchange_order_id} rejected: {reason}")

    def _create_fill(
        self,
        order: Order,
        quantity: float,
        price: float,
        is_partial: bool
    ) -> None:
        """
        Create a fill for an order.

        Args:
            order: The order being filled
            quantity: The quantity filled
            price: The fill price
            is_partial: Whether this is a partial fill
        """
        # Update order status
        if is_partial:
            order.status = OrderStatus.PARTIALLY_FILLED
        else:
            order.status = OrderStatus.FILLED

        order.filled_quantity += quantity
        order.timestamp_updated = datetime.now()

        # Calculate fees
        is_maker = order.type == OrderType.LIMIT
        fee_rate = self.config.maker_fee_rate if is_maker else self.config.taker_fee_rate
        fee_amount = price * quantity * fee_rate
        fee_currency = self._get_fee_currency(order.instrument)

        # Create fill object
        fill = Fill(
            id=str(uuid.uuid4()),
            order_id=order.id,
            exchange_order_id=order.exchange_order_id,
            instrument=order.instrument,
            side=order.side,
            quantity=quantity,
            price=price,
            fee=fee_amount,
            fee_currency=fee_currency,
            timestamp=datetime.now(),
            is_maker=is_maker
        )

        # Update positions
        self._update_position(order, quantity, price, fee_amount, fee_currency)

        # Emit fill event
        self.event_bus.emit(
            EventType.ORDER_FILLED if order.status == OrderStatus.FILLED
            else EventType.ORDER_PARTIALLY_FILLED,
            {
                "order": order,
                "fill": fill,
                "exchange_id": self.exchange_id,
                "timestamp": datetime.now()
            }
        )

        logger.info(
            f"Order {order.exchange_order_id} {'partially ' if is_partial else ''}filled: "
            f"{quantity} @ {price}, fees: {fee_amount} {fee_currency}"
        )

    def _update_position(
        self,
        order: Order,
        quantity: float,
        price: float,
        fee_amount: float,
        fee_currency: str
    ) -> None:
        """
        Update positions and balances based on fills.

        Args:
            order: The filled order
            quantity: The filled quantity
            price: The fill price
            fee_amount: The fee amount
            fee_currency: The fee currency
        """
        # Extract base and quote currency from instrument
        base_ccy, quote_ccy = self._parse_instrument(order.instrument)

        # Update position for base currency
        position_delta = quantity if order.side == Side.BUY else -quantity
        self.positions[order.instrument] = self.positions.get(order.instrument, 0) + position_delta

        # Update balances
        if order.side == Side.BUY:
            # Deduct quote currency (e.g., USD)
            quote_amount = quantity * price
            self.balances[quote_ccy] = self.balances.get(quote_ccy, 0) - quote_amount

            # Add base currency (e.g., BTC)
            self.balances[base_ccy] = self.balances.get(base_ccy, 0) + quantity
        else:  # SELL
            # Add quote currency
            quote_amount = quantity * price
            self.balances[quote_ccy] = self.balances.get(quote_ccy, 0) + quote_amount

            # Deduct base currency
            self.balances[base_ccy] = self.balances.get(base_ccy, 0) - quantity

        # Deduct fees
        self.balances[fee_currency] = self.balances.get(fee_currency, 0) - fee_amount

        logger.debug(
            f"Updated positions for {order.instrument}: {self.positions[order.instrument]}, "
            f"Balances: {self.balances}"
        )

    def _simulate_latency(self) -> float:
        """
        Simulate realistic exchange latency.

        Returns:
            Simulated latency in milliseconds
        """
        # Base latency with normal distribution
        latency = max(1, np.random.normal(
            self.config.base_latency_ms,
            self.config.latency_stdev_ms
        ))

        # Add exchange processing time
        latency += self.config.exchange_processing_ms

        # Occasionally add significant delay to simulate network issues
        if random.random() < 0.01:  # 1% chance of additional delay
            latency += random.uniform(100, 500)  # 100-500ms additional delay

        return latency

    def _calculate_execution_price(
        self,
        instrument: str,
        side: Side,
        quantity: float,
        market_data: Dict[str, Any]
    ) -> float:
        """
        Calculate execution price with realistic slippage based on order size.

        Args:
            instrument: The trading instrument
            side: Order side (BUY/SELL)
            quantity: Order quantity
            market_data: Current market data

        Returns:
            Execution price with slippage
        """
        # Get bid/ask from market data
        bid = market_data.get("bid", 0)
        ask = market_data.get("ask", 0)

        # If bid/ask not available, use last price
        if bid == 0 or ask == 0:
            mid = market_data.get("price", 0)
            spread = mid * 0.0002  # Assume 2bp spread if not available
            bid = mid - spread / 2
            ask = mid + spread / 2

        # Get market liquidity factor for this instrument
        liquidity = self._get_liquidity_factor(instrument)

        # Base price before slippage
        base_price = ask if side == Side.BUY else bid

        # Calculate market impact (slippage)
        # Higher quantity and lower liquidity = more slippage
        impact_factor = (quantity * self.config.market_impact_factor) / liquidity

        # Apply impact to price
        if side == Side.BUY:
            # Buy orders move price up
            execution_price = base_price * (1 + impact_factor)
        else:
            # Sell orders move price down
            execution_price = base_price * (1 - impact_factor)

        # Add some randomness to execution price
        noise_factor = 0.0001  # 1bp noise
        execution_price *= (1 + random.uniform(-noise_factor, noise_factor))

        return execution_price

    def _get_liquidity_factor(self, instrument: str) -> float:
        """
        Get liquidity factor for an instrument.

        Args:
            instrument: The instrument

        Returns:
            Liquidity factor (higher = more liquid = less slippage)
        """
        # Determine instrument type (forex, crypto, etc.)
        instrument_type = self._get_instrument_type(instrument)

        # Get liquidity factor from config
        return self.config.liquidity_factor.get(
            instrument_type,
            self.config.liquidity_factor.get("default", 1.0)
        )

    def _get_instrument_type(self, instrument: str) -> str:
        """
        Determine instrument type based on symbol.

        Args:
            instrument: The instrument symbol

        Returns:
            Instrument type (forex, crypto, equities)
        """
        # Simple heuristic based on symbol format
        if '/' in instrument:
            # Forex typically uses format like "EUR/USD"
            return "forex"
        elif '-' in instrument or instrument.endswith("USDT") or instrument.endswith("BTC"):
            # Common crypto formats
            return "crypto"
        elif '.' in instrument:
            # Equities often have exchange suffix like "AAPL.US"
            return "equities"
        else:
            return "default"

    def _parse_instrument(self, instrument: str) -> Tuple[str, str]:
        """
        Parse base and quote currency from instrument.

        Args:
            instrument: The instrument symbol

        Returns:
            Tuple of (base_currency, quote_currency)
        """
        instrument_type = self._get_instrument_type(instrument)

        if instrument_type == "forex":
            # Forex typically uses format like "EUR/USD"
            parts = instrument.split('/')
            if len(parts) == 2:
                return parts[0], parts[1]

        elif instrument_type == "crypto":
            # Handle different crypto formats
            if '-' in instrument:
                # Format like "BTC-USDT"
                parts = instrument.split('-')
                if len(parts) == 2:
                    return parts[0], parts[1]
            else:
                # Format like "BTCUSDT"
                for quote in ["USDT", "BTC", "ETH", "USD", "BUSD"]:
                    if instrument.endswith(quote):
                        base = instrument[:-len(quote)]
                        return base, quote

        # Default fallback - split in middle if all else fails
        mid = len(instrument) // 2
        return instrument[:mid], instrument[mid:]

    def _get_fee_currency(self, instrument: str) -> str:
        """
        Determine fee currency based on instrument.

        Args:
            instrument: The instrument symbol

        Returns:
            Fee currency
        """
        instrument_type = self._get_instrument_type(instrument)

        if instrument_type == "forex":
            # Forex fees typically in quote currency
            _, quote = self._parse_instrument(instrument)
            return quote

        elif instrument_type == "crypto":
            # Some exchanges charge in platform token, others in quote currency
            _, quote = self._parse_instrument(instrument)

            # Simulate some exchanges using platform token for fees
            if self.exchange_id == "binance":
                return "BNB"  # Binance uses BNB for discounted fees
            elif self.exchange_id == "ftx":
                return "FTT"  # FTX uses FTT for discounted fees
            else:
                return quote

        else:  # Default for other instrument types
            # Extract quote currency
            _, quote = self._parse_instrument(instrument)
            return quote

    def _handle_market_data_update(self, event_data: Dict[str, Any]) -> None:
        """
        Handle market data updates to potentially fill limit orders.

        Args:
            event_data: Market data event information
        """
        instrument = event_data.get("instrument")
        if not instrument:
            return

        # Get market data
        market_data = event_data.get("data", {})
        if not market_data:
            return

        # Check for limit orders that can be filled
        self._check_limit_orders_for_instrument(instrument, market_data)

    def _check_limit_orders_for_instrument(
        self,
        instrument: str,
        market_data: Dict[str, Any]
    ) -> None:
        """
        Check if any limit orders for this instrument can be filled.

        Args:
            instrument: The instrument to check
            market_data: Current market data
        """
        # Get current price
        bid = market_data.get("bid", 0)
        ask = market_data.get("ask", 0)

        # If bid/ask not available, use last price
        if bid == 0 or ask == 0:
            price = market_data.get("price", 0)
            spread = price * 0.0002  # Assume 2bp spread if not available
            bid = price - spread / 2
            ask = price + spread / 2

        # Find limit orders for this instrument that can be filled
        orders_to_process = []

        for order_id, order in list(self.open_orders.items()):
            if (order.instrument == instrument and
                order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED] and
                order.type == OrderType.LIMIT):

                should_fill = False

                # Check if price conditions are met
                if order.side == Side.BUY and order.price >= bid:
                    should_fill = True
                elif order.side == Side.SELL and order.price <= ask:
                    should_fill = True

                if should_fill:
                    # Clone order for processing
                    orders_to_process.append(order.copy())

        # Process orders that need to be filled
        for order in orders_to_process:
            # Calculate execution price
            execution_price = self._calculate_execution_price(
                order.instrument,
                order.side,
                order.quantity,
                market_data
            )

            # Respect limit price boundary
            if order.side == Side.BUY:
                execution_price = min(execution_price, order.price)
            else:
                execution_price = max(execution_price, order.price)

            # Create fill
            self._create_fill(
                order,
                order.quantity,
                execution_price,
                is_partial=False
            )

    def run_scheduled_updates(self) -> None:
        """
        Run scheduled updates for the simulator, like checking limit orders.

        This method should be called periodically to simulate exchange behavior
        even when no new market data is received.
        """
        # Process any limit orders that might be fillable
        for instrument in set(order.instrument for order in self.open_orders.values()):
            try:
                # Get latest market data for this instrument
                market_data = self.market_data_service.get_last_price(instrument)
                self._check_limit_orders_for_instrument(instrument, market_data)
            except Exception as e:
                logger.error(f"Error checking limit orders for {instrument}: {e}")

    def reset(self) -> None:
        """
        Reset simulator state (for testing or initialization).
        """
        # Cancel all open orders
        for order_id in list(self.open_orders.keys()):
            self._process_cancellation(order_id)

        # Clear positions and balances
        self.positions.clear()
        self.balances.clear()

        logger.info(f"Exchange simulator {self.exchange_id} reset completed")

    def simulate_order_book(self, instrument: str, market_data: Dict[str, Any]) -> OrderBook:
        """
        Generate a simulated order book based on market data.

        Args:
            instrument: The instrument to simulate
            market_data: Current market data

        Returns:
            Simulated order book
        """
        if not self.config.simulate_order_book:
            return OrderBook(instrument)

        order_book = OrderBook(instrument)

        # Get current price points
        mid = (market_data.get("bid", 0) + market_data.get("ask", 0)) / 2
        if mid == 0:
            mid = market_data.get("price", 0)

        spread = market_data.get("ask", mid * 1.0001) - market_data.get("bid", mid * 0.9999)

        # Get liquidity factor for instrument
        liquidity = self._get_liquidity_factor(instrument)

        # Generate simulated order book
        for i in range(self.config.book_depth):
            # Price gap increases as we move away from mid
            price_step = spread * (1 + i * 0.25)

            # Bid side (buy orders)
            bid_price = mid - price_step / 2
            bid_volume = random.uniform(5, 15) * liquidity * (1 - i * 0.05)
            order_book.add_bid(bid_price, bid_volume)

            # Ask side (sell orders)
            ask_price = mid + price_step / 2
            ask_volume = random.uniform(5, 15) * liquidity * (1 - i * 0.05)
            order_book.add_ask(ask_price, ask_volume)

        return order_book

    def format_instrument_for_exchange(self, instrument: str) -> str:
        """
        Format instrument symbol according to this exchange's conventions.

        Args:
            instrument: The generic instrument symbol (e.g., "BTC/USD")

        Returns:
            Exchange-specific formatted symbol
        """
        # Simple mapping for forex and crypto
        if self.exchange_id == "binance":
            # Binance uses no separator for crypto
            return instrument.replace("/", "")

        elif self.exchange_id == "oanda":
            # Oanda uses "_" for forex
            return instrument.replace("/", "_")

        elif self.exchange_id == "coinbase":
            # Coinbase uses "-" for crypto
            return instrument.replace("/", "-")

        elif self.exchange_id == "interactive_brokers":
            # IB uses different formats for different asset classes
            if self._get_instrument_type(instrument) == "forex":
                return instrument  # Keep as is for forex
            else:
                # Add exchange suffix for stocks if not present
                if "." not in instrument:
                    return f"{instrument}.US"
                return instrument

        # Default: return as is
        return instrument

    def parse_exchange_instrument(self, exchange_symbol: str) -> str:
        """
        Parse exchange-specific symbol into standardized format.

        Args:
            exchange_symbol: The exchange-specific symbol

        Returns:
            Standardized instrument symbol
        """
        # Reverse of format_instrument_for_exchange
        if self.exchange_id == "binance":
            # Try to identify base/quote boundary for crypto
            for quote in ["USDT", "BTC", "ETH", "USD", "BUSD"]:
                if exchange_symbol.endswith(quote):
                    base = exchange_symbol[:-len(quote)]
                    return f"{base}/{quote}"

        elif self.exchange_id == "oanda":
            # Convert Oanda's "_" to "/"
            return exchange_symbol.replace("_", "/")

        elif self.exchange_id == "coinbase":
            # Convert Coinbase's "-" to "/"
            return exchange_symbol.replace("-", "/")

        elif self.exchange_id == "interactive_brokers":
            # Strip exchange suffix for stocks
            if "." in exchange_symbol:
                return exchange_symbol.split(".")[0]

        # Default: return as is
        return exchange_symbol