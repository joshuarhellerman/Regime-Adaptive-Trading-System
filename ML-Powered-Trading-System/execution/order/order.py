"""
Order Module

This module defines the core Order classes and related enums for representing
different types of trading orders in the system.
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any, Union
import uuid
import logging

# Configure logger
logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Types of orders supported by the system"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    OCO = "oco"  # One-Cancels-Other
    ICEBERG = "iceberg"  # Hidden/Iceberg
    FOK = "fok"  # Fill-or-Kill
    IOC = "ioc"  # Immediate-or-Cancel

class OrderSide(Enum):
    """Side of the order (buy or sell)"""
    BUY = "buy"
    SELL = "sell"

    def is_buy(self) -> bool:
        """Check if this is a buy order"""
        return self == OrderSide.BUY

    def is_sell(self) -> bool:
        """Check if this is a sell order"""
        return self == OrderSide.SELL

class OrderStatus(Enum):
    """Possible states of an order"""
    CREATED = "created"            # Initial state when order is created locally
    PENDING = "pending"            # Order is being prepared for submission
    PENDING_CANCEL = "pending_cancel"  # Cancel request has been submitted
    PENDING_UPDATE = "pending_update"  # Update request has been submitted
    OPEN = "open"                  # Order has been submitted and is active
    PARTIALLY_FILLED = "partially_filled"  # Order is partially filled
    FILLED = "filled"              # Order is completely filled
    CANCELED = "canceled"          # Order has been canceled
    REJECTED = "rejected"          # Order was rejected by the exchange
    EXPIRED = "expired"            # Order has expired (e.g., day orders at market close)
    ERROR = "error"                # Error occurred with the order
    WORKING = "working"            # Special state for orders being managed by execution strategies

class TimeInForce(Enum):
    """Validity period for an order"""
    GTC = "gtc"    # Good Till Canceled
    IOC = "ioc"    # Immediate Or Cancel
    FOK = "fok"    # Fill Or Kill
    DAY = "day"    # Valid for the trading day
    GTD = "gtd"    # Good Till Date

@dataclass
class Order:
    """
    Base class for all orders in the system.

    This class represents an order with all essential properties and methods
    for tracking its lifecycle from creation to execution/cancellation.
    """
    # Basic order information
    symbol: str                          # Trading pair or instrument
    side: OrderSide                      # Buy or sell
    quantity: float                      # Order quantity
    order_type: OrderType                # Type of order

    # Price information
    price: Optional[float] = None        # Limit price
    stop_price: Optional[float] = None   # Stop price for stop orders

    # Order parameters
    time_in_force: TimeInForce = TimeInForce.GTC  # How long the order is valid

    # Status and tracking
    status: OrderStatus = OrderStatus.CREATED    # Current order status
    status_message: str = ""             # Additional status information
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # Unique local ID
    exchange_order_id: Optional[str] = None  # ID assigned by exchange
    client_order_id: Optional[str] = None  # Custom client order ID
    parent_order_id: Optional[str] = None  # Parent order ID for child orders
    strategy_id: Optional[str] = None    # ID of strategy that created order

    # Timestamps
    creation_time: datetime = field(default_factory=datetime.utcnow)  # When order was created
    update_time: Optional[datetime] = None  # Last time order was updated
    submission_time: Optional[datetime] = None  # When order was submitted to exchange
    execution_time: Optional[datetime] = None  # When order was executed/filled

    # Fill information
    filled_quantity: float = 0.0         # Amount filled so far
    average_price: Optional[float] = None  # Average price of all fills

    # Exchange and other metadata
    exchange_id: Optional[str] = None    # Exchange this order is for
    exchange_account: Optional[str] = None  # Account ID on the exchange
    params: Dict[str, Any] = field(default_factory=dict)  # Additional parameters
    tags: List[str] = field(default_factory=list)  # Tags for categorization

    def __post_init__(self):
        """
        Validate and initialize order after creation.
        """
        # Set default values
        if self.update_time is None:
            self.update_time = self.creation_time

        # Validate order configuration
        self._validate()

        # Log order creation
        logger.info(f"Created {self.order_type.value} {self.side.value} order for {self.symbol}: "
                   f"quantity={self.quantity}, price={self.price}, id={self.order_id}")

    def _validate(self):
        """Validate order configuration"""
        # Validate price for limit orders
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and self.price is None:
            raise ValueError(f"Price must be specified for {self.order_type.value} orders")

        # Validate stop price for stop orders
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
            raise ValueError(f"Stop price must be specified for {self.order_type.value} orders")

    @property
    def is_active(self) -> bool:
        """Check if the order is still active in the market"""
        return self.status in [
            OrderStatus.CREATED,
            OrderStatus.PENDING,
            OrderStatus.OPEN,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.WORKING,
            OrderStatus.PENDING_UPDATE
        ]

    @property
    def is_complete(self) -> bool:
        """Check if the order has completed execution"""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.ERROR
        ]

    @property
    def remaining_quantity(self) -> float:
        """Calculate the remaining quantity to be filled"""
        return self.quantity - self.filled_quantity

    @property
    def fill_percentage(self) -> float:
        """Calculate the percentage of the order that has been filled"""
        if self.quantity <= 0:
            return 0.0
        return (self.filled_quantity / self.quantity) * 100.0

    def copy(self) -> 'Order':
        """Create a copy of this order"""
        # Create a dict representation and create a new order from it
        order_dict = self.to_dict()

        # When copying, we want to carry over the same ID
        return Order.from_dict(order_dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert order to dictionary for serialization.

        Returns:
            Dict containing all order data
        """
        result = {
            # Basic order info
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "order_id": self.order_id,

            # Price info
            "price": self.price,
            "stop_price": self.stop_price,

            # Order parameters
            "time_in_force": self.time_in_force.value,

            # Status and tracking
            "status": self.status.value,
            "status_message": self.status_message,
            "exchange_order_id": self.exchange_order_id,
            "client_order_id": self.client_order_id,
            "parent_order_id": self.parent_order_id,
            "strategy_id": self.strategy_id,

            # Timestamps
            "creation_time": self.creation_time.isoformat() if self.creation_time else None,
            "update_time": self.update_time.isoformat() if self.update_time else None,
            "submission_time": self.submission_time.isoformat() if self.submission_time else None,
            "execution_time": self.execution_time.isoformat() if self.execution_time else None,

            # Fill information
            "filled_quantity": self.filled_quantity,
            "average_price": self.average_price,

            # Exchange and other metadata
            "exchange_id": self.exchange_id,
            "exchange_account": self.exchange_account,
            "params": self.params,
            "tags": self.tags
        }

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """
        Create an order instance from a dictionary.

        Args:
            data: Dictionary containing order data

        Returns:
            Order instance
        """
        # Make a copy to avoid modifying the original
        order_data = data.copy()

        # Convert string enums back to enum types
        if "side" in order_data and isinstance(order_data["side"], str):
            order_data["side"] = OrderSide(order_data["side"])

        if "order_type" in order_data and isinstance(order_data["order_type"], str):
            order_data["order_type"] = OrderType(order_data["order_type"])

        if "status" in order_data and isinstance(order_data["status"], str):
            order_data["status"] = OrderStatus(order_data["status"])

        if "time_in_force" in order_data and isinstance(order_data["time_in_force"], str):
            order_data["time_in_force"] = TimeInForce(order_data["time_in_force"])

        # Convert ISO timestamps back to datetime
        datetime_fields = [
            "creation_time", "update_time", "submission_time", "execution_time"
        ]

        for field in datetime_fields:
            if field in order_data and order_data[field] and isinstance(order_data[field], str):
                order_data[field] = datetime.fromisoformat(order_data[field])

        return cls(**order_data)

    def __str__(self) -> str:
        """String representation of the order"""
        return (f"{self.order_type.value.capitalize()} {self.side.value} order for {self.symbol}: "
                f"{self.filled_quantity}/{self.quantity} filled @ {self.average_price or self.price}, "
                f"status={self.status.value}")