"""
Order Manager Module

This module provides a high-level interface for managing orders throughout their lifecycle.
It serves as the primary interface between strategies and the execution service,
handling order creation, updates, cancellations, and monitoring.
"""

import logging
import threading
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Callable, Set
import queue
import uuid

from core.event_bus import EventTopics, create_event, get_event_bus, Event, EventSubscriber
from execution.order.order import Order, OrderStatus, OrderSide, OrderType, TimeInForce
from execution.order.order_book import OrderBook
from execution.order.order_factory import OrderFactory

# Configure logger
logger = logging.getLogger(__name__)


class OrderManagerMode(Enum):
    """Mode for the order manager"""
    LIVE = "live"  # Live trading
    PAPER = "paper"  # Paper trading
    BACKTEST = "backtest"  # Backtesting


class OrderManager(EventSubscriber):
    """
    Order Manager for handling all order lifecycle operations.

    This class provides a high-level interface for order management, abstracting
    away the details of order execution. It manages the lifetime of orders from
    creation through execution, updates, and cancellation.

    Responsibilities:
    - Create and validate orders
    - Submit orders to the execution service
    - Track and update order status
    - Cancel orders
    - Provide order status notifications
    - Handle order replacements and modifications
    """

    def __init__(self,
                 execution_service,
                 order_book: Optional[OrderBook] = None,
                 order_factory: Optional[OrderFactory] = None,
                 config: Dict[str, Any] = None,
                 mode: OrderManagerMode = OrderManagerMode.PAPER):
        """
        Initialize the order manager.

        Args:
            execution_service: The execution service for submitting orders
            order_book: Optional order book instance (will create if not provided)
            order_factory: Optional order factory instance (will create if not provided)
            config: Configuration dictionary
            mode: Operating mode (live, paper, backtest)
        """
        self.execution_service = execution_service
        self.order_book = order_book or OrderBook()
        self.order_factory = order_factory or OrderFactory()
        self.config = config or {}
        self.mode = mode

        # Initialize event bus and subscribe to relevant topics
        self._event_bus = get_event_bus()
        self._event_subscriptions = []

        # Setup callback registry
        self._order_callbacks: Dict[str, List[Callable]] = {}
        self._global_callbacks: List[Callable] = []

        # Setup state management
        self._running = False
        self._order_status_cache: Dict[str, OrderStatus] = {}
        self._lock = threading.RLock()

        # User-configurable settings
        self._default_exchange = self.config.get("default_exchange")
        self._status_polling_interval = self.config.get("status_polling_interval", 1.0)

        # Statistics
        self._stats = {
            "orders_created": 0,
            "orders_submitted": 0,
            "orders_filled": 0,
            "orders_canceled": 0,
            "orders_rejected": 0,
            "orders_error": 0
        }

        logger.info(f"OrderManager initialized in {mode.value} mode")

    def start(self) -> None:
        """Start the order manager and subscribe to events"""
        with self._lock:
            if self._running:
                logger.warning("OrderManager already running")
                return

            self._running = True

            # Subscribe to order events
            self._subscribe_to_events()

            # Start polling thread if configured
            if self._status_polling_interval > 0:
                self._polling_thread = threading.Thread(
                    target=self._status_polling_worker,
                    daemon=True,
                    name="OrderStatusPolling"
                )
                self._polling_thread.start()

            logger.info("OrderManager started")

    def stop(self) -> None:
        """Stop the order manager and unsubscribe from events"""
        with self._lock:
            if not self._running:
                logger.warning("OrderManager not running")
                return

            self._running = False

            # Unsubscribe from events
            self._unsubscribe_from_events()

            logger.info("OrderManager stopped")

    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant event topics"""
        topics = [
            EventTopics.ORDER_CREATED,
            EventTopics.ORDER_SUBMITTED,
            EventTopics.ORDER_OPEN,
            EventTopics.ORDER_FILLED,
            EventTopics.ORDER_PARTIALLY_FILLED,
            EventTopics.ORDER_CANCELED,
            EventTopics.ORDER_REJECTED,
            EventTopics.ORDER_EXPIRED,
            EventTopics.ORDER_ERROR,
            EventTopics.ORDER_STATUS_UPDATED
        ]

        for topic in topics:
            subscription_id = self._event_bus.subscribe(topic, self._handle_order_event)
            self._event_subscriptions.append((topic, subscription_id))

        logger.debug(f"Subscribed to {len(topics)} event topics")

    def _unsubscribe_from_events(self) -> None:
        """Unsubscribe from all events"""
        for topic, subscription_id in self._event_subscriptions:
            self._event_bus.unsubscribe(topic, subscription_id)

        self._event_subscriptions = []
        logger.debug("Unsubscribed from all events")

    def _handle_order_event(self, event: Event) -> None:
        """
        Handle an order event from the event bus.

        Args:
            event: The event to handle
        """
        try:
            # Extract order from event data
            order_data = event.data.get("order")
            if not order_data:
                logger.warning(f"Order event missing order data: {event.topic}")
                return

            # Convert to Order object if it's a dict
            order = order_data
            if isinstance(order_data, dict):
                order = Order.from_dict(order_data)

            # Update order status cache
            self._order_status_cache[order.order_id] = order.status

            # Update stats
            if event.topic == EventTopics.ORDER_FILLED:
                self._stats["orders_filled"] += 1
            elif event.topic == EventTopics.ORDER_CANCELED:
                self._stats["orders_canceled"] += 1
            elif event.topic == EventTopics.ORDER_REJECTED:
                self._stats["orders_rejected"] += 1
            elif event.topic == EventTopics.ORDER_ERROR:
                self._stats["orders_error"] += 1

            # Call order-specific callbacks
            self._call_order_callbacks(order.order_id, event.topic, order)

            # Call global callbacks
            self._call_global_callbacks(event.topic, order)

        except Exception as e:
            logger.error(f"Error handling order event: {str(e)}", exc_info=True)

    def _status_polling_worker(self) -> None:
        """Worker thread for polling order status"""
        logger.info(f"Order status polling started with interval {self._status_polling_interval}s")

        while self._running:
            try:
                # Get active orders
                active_orders = self.get_active_orders()

                for order in active_orders:
                    # Get latest status from execution service
                    current_status = self.execution_service.get_order_status(order.order_id)

                    # If status has changed, update our cache
                    if current_status and current_status != self._order_status_cache.get(order.order_id):
                        self._order_status_cache[order.order_id] = current_status

                        # Get full order details
                        updated_order = self.execution_service.get_order(order.order_id)
                        if updated_order:
                            # Update order book
                            self.order_book.update_order(updated_order)

                            # Log status change
                            logger.info(f"Order {order.order_id} status changed: {current_status.value}")

                # Sleep until next poll
                time.sleep(self._status_polling_interval)

            except Exception as e:
                logger.error(f"Error in order status polling: {str(e)}", exc_info=True)
                time.sleep(5)  # Avoid spinning on persistent errors

    def create_market_order(self,
                            symbol: str,
                            side: Union[OrderSide, str],
                            quantity: float,
                            **kwargs) -> Order:
        """
        Create a market order.

        Args:
            symbol: Trading symbol
            side: Order side (buy or sell)
            quantity: Order quantity
            **kwargs: Additional order parameters

        Returns:
            The created order
        """
        # Apply default exchange if not specified
        if "exchange_id" not in kwargs and self._default_exchange:
            kwargs["exchange_id"] = self._default_exchange

        # Create the order
        order = self.order_factory.create_market_order(symbol, side, quantity, **kwargs)

        # Add to order book
        self.order_book.add_order(order)

        # Update stats
        self._stats["orders_created"] += 1

        logger.info(f"Created market {order.side.value} order for {symbol}: "
                    f"quantity={quantity}, id={order.order_id}")

        return order

    def create_limit_order(self,
                           symbol: str,
                           side: Union[OrderSide, str],
                           quantity: float,
                           price: float,
                           **kwargs) -> Order:
        """
        Create a limit order.

        Args:
            symbol: Trading symbol
            side: Order side (buy or sell)
            quantity: Order quantity
            price: Limit price
            **kwargs: Additional order parameters

        Returns:
            The created order
        """
        # Apply default exchange if not specified
        if "exchange_id" not in kwargs and self._default_exchange:
            kwargs["exchange_id"] = self._default_exchange

        # Create the order
        order = self.order_factory.create_limit_order(symbol, side, quantity, price, **kwargs)

        # Add to order book
        self.order_book.add_order(order)

        # Update stats
        self._stats["orders_created"] += 1

        logger.info(f"Created limit {order.side.value} order for {symbol}: "
                    f"quantity={quantity}, price={price}, id={order.order_id}")

        return order

    def create_stop_order(self,
                          symbol: str,
                          side: Union[OrderSide, str],
                          quantity: float,
                          stop_price: float,
                          **kwargs) -> Order:
        """
        Create a stop order.

        Args:
            symbol: Trading symbol
            side: Order side (buy or sell)
            quantity: Order quantity
            stop_price: Stop price
            **kwargs: Additional order parameters

        Returns:
            The created order
        """
        # Apply default exchange if not specified
        if "exchange_id" not in kwargs and self._default_exchange:
            kwargs["exchange_id"] = self._default_exchange

        # Create the order
        order = self.order_factory.create_stop_order(symbol, side, quantity, stop_price, **kwargs)

        # Add to order book
        self.order_book.add_order(order)

        # Update stats
        self._stats["orders_created"] += 1

        logger.info(f"Created stop {order.side.value} order for {symbol}: "
                    f"quantity={quantity}, stop_price={stop_price}, id={order.order_id}")

        return order

    def create_stop_limit_order(self,
                                symbol: str,
                                side: Union[OrderSide, str],
                                quantity: float,
                                stop_price: float,
                                limit_price: float,
                                **kwargs) -> Order:
        """
        Create a stop-limit order.

        Args:
            symbol: Trading symbol
            side: Order side (buy or sell)
            quantity: Order quantity
            stop_price: Stop price
            limit_price: Limit price
            **kwargs: Additional order parameters

        Returns:
            The created order
        """
        # Apply default exchange if not specified
        if "exchange_id" not in kwargs and self._default_exchange:
            kwargs["exchange_id"] = self._default_exchange

        # Create the order
        order = self.order_factory.create_stop_limit_order(
            symbol, side, quantity, stop_price, limit_price, **kwargs
        )

        # Add to order book
        self.order_book.add_order(order)

        # Update stats
        self._stats["orders_created"] += 1

        logger.info(f"Created stop-limit {order.side.value} order for {symbol}: "
                    f"quantity={quantity}, stop_price={stop_price}, limit_price={limit_price}, "
                    f"id={order.order_id}")

        return order

    def submit_order(self, order: Order) -> bool:
        """
        Submit an order for execution.

        Args:
            order: Order to submit

        Returns:
            True if the order was submitted successfully
        """
        if not self._running:
            raise RuntimeError("OrderManager is not running")

        try:
            # Submit to execution service
            self.execution_service.submit_order(order)

            # Update stats
            self._stats["orders_submitted"] += 1

            logger.info(f"Submitted order {order.order_id} for execution")
            return True

        except Exception as e:
            logger.error(f"Error submitting order {order.order_id}: {str(e)}")

            # Update order status
            order.status = OrderStatus.ERROR
            order.status_message = f"Submission error: {str(e)}"
            self.order_book.update_order(order)

            return False

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: ID of the order to cancel

        Returns:
            True if the cancel request was accepted
        """
        if not self._running:
            raise RuntimeError("OrderManager is not running")

        try:
            # Request cancellation
            result = self.execution_service.cancel_order(order_id)

            if result:
                logger.info(f"Cancel request accepted for order {order_id}")
            else:
                logger.warning(f"Cancel request rejected for order {order_id}")

            return result

        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {str(e)}")
            return False

    def update_order(self, order_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing order.

        Args:
            order_id: ID of the order to update
            updates: Dictionary of updates to apply

        Returns:
            True if the update request was accepted
        """
        if not self._running:
            raise RuntimeError("OrderManager is not running")

        try:
            # Request update
            result = self.execution_service.update_order(order_id, updates)

            if result:
                logger.info(f"Update request accepted for order {order_id}")
            else:
                logger.warning(f"Update request rejected for order {order_id}")

            return result

        except Exception as e:
            logger.error(f"Error updating order {order_id}: {str(e)}")
            return False

    def replace_order(self,
                      order_id: str,
                      new_price: Optional[float] = None,
                      new_quantity: Optional[float] = None,
                      **kwargs) -> Optional[str]:
        """
        Replace an order with a new one.

        This is a common operation that cancels an existing order and creates a new one
        with updated parameters. It handles the two operations atomically.

        Args:
            order_id: ID of the order to replace
            new_price: New price (for limit orders)
            new_quantity: New quantity
            **kwargs: Additional parameters for the new order

        Returns:
            ID of the new order if successful, None otherwise
        """
        if not self._running:
            raise RuntimeError("OrderManager is not running")

        # Get the original order
        original_order = self.order_book.get_order(order_id)
        if not original_order:
            logger.warning(f"Order not found for replacement: {order_id}")
            return None

        # Check if order can be replaced
        if not original_order.is_active:
            logger.warning(f"Cannot replace inactive order: {order_id}, status={original_order.status.value}")
            return None

        # Create a new order with the same parameters but updated values
        new_order_params = original_order.to_dict()

        # Remove fields that should not be copied
        for field in ["order_id", "status", "filled_quantity", "average_price",
                      "creation_time", "update_time", "submission_time", "execution_time",
                      "exchange_order_id"]:
            new_order_params.pop(field, None)

        # Update with new values
        if new_price is not None and hasattr(original_order, "price"):
            new_order_params["price"] = new_price

        if new_quantity is not None:
            new_order_params["quantity"] = new_quantity

        # Add link to original order
        if "params" not in new_order_params:
            new_order_params["params"] = {}
        new_order_params["params"]["replaced_order_id"] = order_id

        # Apply any additional updates
        for key, value in kwargs.items():
            if key in new_order_params:
                new_order_params[key] = value

        try:
            # Create new order
            new_order = Order.from_dict(new_order_params)

            # Add to order book
            self.order_book.add_order(new_order)

            # Submit new order
            self.submit_order(new_order)

            # Cancel old order
            self.cancel_order(order_id)

            logger.info(f"Replaced order {order_id} with new order {new_order.order_id}")
            return new_order.order_id

        except Exception as e:
            logger.error(f"Error replacing order {order_id}: {str(e)}")
            return None

    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get an order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order or None if not found
        """
        # Try local order book first
        order = self.order_book.get_order(order_id)
        if order:
            return order

        # If not found, try execution service
        return self.execution_service.get_order(order_id)

    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """
        Get the current status of an order.

        Args:
            order_id: Order ID

        Returns:
            Order status or None if not found
        """
        # Check cache first
        if order_id in self._order_status_cache:
            return self._order_status_cache[order_id]

        # Try local order book
        order = self.order_book.get_order(order_id)
        if order:
            self._order_status_cache[order_id] = order.status
            return order.status

        # Try execution service
        status = self.execution_service.get_order_status(order_id)
        if status:
            self._order_status_cache[order_id] = status

        return status

    def get_active_orders(self) -> List[Order]:
        """
        Get all active orders.

        Returns:
            List of active orders
        """
        return self.order_book.get_active_orders()

    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """
        Get all orders for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of orders for the symbol
        """
        return self.order_book.get_orders_by_symbol(symbol)

    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """
        Get all orders with a specific status.

        Args:
            status: Order status

        Returns:
            List of orders with the status
        """
        return self.order_book.get_orders_by_status(status)

    def get_position_by_symbol(self, symbol: str) -> float:
        """
        Calculate net position for a symbol based on order fills.

        Args:
            symbol: Trading symbol

        Returns:
            Net position (positive for long, negative for short)
        """
        return self.order_book.get_position_by_symbol(symbol)

    def register_order_callback(self, order_id: str, callback: Callable) -> None:
        """
        Register a callback for a specific order.

        The callback will be called when any event occurs for the order.

        Args:
            order_id: Order ID
            callback: Callback function (takes event_topic and order as arguments)
        """
        with self._lock:
            if order_id not in self._order_callbacks:
                self._order_callbacks[order_id] = []

            self._order_callbacks[order_id].append(callback)

    def unregister_order_callback(self, order_id: str, callback: Callable) -> bool:
        """
        Unregister a callback for a specific order.

        Args:
            order_id: Order ID
            callback: Callback function to remove

        Returns:
            True if the callback was found and removed
        """
        with self._lock:
            if order_id in self._order_callbacks:
                if callback in self._order_callbacks[order_id]:
                    self._order_callbacks[order_id].remove(callback)
                    return True

        return False

    def register_global_callback(self, callback: Callable) -> None:
        """
        Register a global callback for all orders.

        The callback will be called when any event occurs for any order.

        Args:
            callback: Callback function (takes event_topic and order as arguments)
        """
        with self._lock:
            self._global_callbacks.append(callback)

    def unregister_global_callback(self, callback: Callable) -> bool:
        """
        Unregister a global callback.

        Args:
            callback: Callback function to remove

        Returns:
            True if the callback was found and removed
        """
        with self._lock:
            if callback in self._global_callbacks:
                self._global_callbacks.remove(callback)
                return True

        return False

    def _call_order_callbacks(self, order_id: str, event_topic: str, order: Order) -> None:
        """
        Call callbacks registered for a specific order.

        Args:
            order_id: Order ID
            event_topic: Event topic
            order: Order object
        """
        with self._lock:
            callbacks = self._order_callbacks.get(order_id, [])

        for callback in callbacks:
            try:
                callback(event_topic, order)
            except Exception as e:
                logger.error(f"Error in order callback: {str(e)}", exc_info=True)

    def _call_global_callbacks(self, event_topic: str, order: Order) -> None:
        """
        Call global callbacks.

        Args:
            event_topic: Event topic
            order: Order object
        """
        with self._lock:
            callbacks = self._global_callbacks.copy()

        for callback in callbacks:
            try:
                callback(event_topic, order)
            except Exception as e:
                logger.error(f"Error in global callback: {str(e)}", exc_info=True)

    def wait_for_order_status(self,
                              order_id: str,
                              target_status: List[OrderStatus],
                              timeout_seconds: float = 30.0) -> Optional[Order]:
        """
        Wait for an order to reach a specific status.

        Args:
            order_id: Order ID
            target_status: List of target statuses to wait for
            timeout_seconds: Maximum time to wait in seconds

        Returns:
            Order if it reached the target status, None if timed out
        """
        start_time = time.time()
        poll_interval = 0.1  # seconds

        while time.time() - start_time < timeout_seconds:
            order = self.get_order(order_id)

            if order and order.status in target_status:
                return order

            time.sleep(poll_interval)

        # Timeout
        logger.warning(f"Timeout waiting for order {order_id} to reach status {[s.value for s in target_status]}")
        return None

    def get_stats(self) -> Dict[str, int]:
        """
        Get order statistics.

        Returns:
            Dictionary of statistics
        """
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset all statistics to zero"""
        with self._lock:
            for key in self._stats:
                self._stats[key] = 0

    def __del__(self):
        """Clean up resources when the object is destroyed"""
        if self._running:
            self.stop()