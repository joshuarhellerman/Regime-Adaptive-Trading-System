"""
Order Book Module

This module provides an order book implementation for tracking order state
throughout the order lifecycle. It maintains a collection of all orders in the system,
provides lookup functionality, and acts as the central repository for order state.
"""

import logging
import threading
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from pathlib import Path
import pandas as pd
from collections import defaultdict

from .order import Order, OrderStatus, OrderSide, OrderType

# Configure logger
logger = logging.getLogger(__name__)

class OrderBook:
    """
    Order book for tracking all orders in the system.

    This class maintains a collection of all orders, provides lookup methods,
    and tracks order state changes. It can also persist orders to disk and
    load them back for recovery.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the order book.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._orders: Dict[str, Order] = {}
        self._lock = threading.RLock()
        self._order_history: Dict[str, List[Dict[str, Any]]] = {}

        # Indices for quick lookups
        self._orders_by_status: Dict[OrderStatus, Set[str]] = defaultdict(set)
        self._orders_by_symbol: Dict[str, Set[str]] = defaultdict(set)
        self._orders_by_exchange: Dict[str, Set[str]] = defaultdict(set)
        self._orders_by_strategy: Dict[str, Set[str]] = defaultdict(set)
        self._child_orders: Dict[str, Set[str]] = defaultdict(set)  # parent_id -> set of child_ids

        # Configure persistence
        self._persistence_enabled = self.config.get('persistence_enabled', True)
        self._persistence_dir = Path(self.config.get('persistence_dir', 'data/orders'))
        if self._persistence_enabled:
            self._persistence_dir.mkdir(parents=True, exist_ok=True)

        # Auto-cleanup settings
        self._cleanup_after_days = self.config.get('cleanup_after_days', 30)
        self._last_cleanup = datetime.now()
        self._cleanup_interval_hours = self.config.get('cleanup_interval_hours', 24)

        logger.info(f"OrderBook initialized, persistence {'enabled' if self._persistence_enabled else 'disabled'}")

    def add_order(self, order: Order) -> bool:
        """
        Add an order to the order book.

        Args:
            order: Order to add

        Returns:
            True if the order was added, False if it already exists
        """
        with self._lock:
            # Check if order already exists
            if order.order_id in self._orders:
                logger.warning(f"Order {order.order_id} already exists in order book")
                return False

            # Add to main collection
            self._orders[order.order_id] = order

            # Add to indices
            self._orders_by_status[order.status].add(order.order_id)
            self._orders_by_symbol[order.symbol].add(order.order_id)

            if order.exchange_id:
                self._orders_by_exchange[order.exchange_id].add(order.order_id)

            if order.strategy_id:
                self._orders_by_strategy[order.strategy_id].add(order.order_id)

            if order.parent_order_id:
                self._child_orders[order.parent_order_id].add(order.order_id)

            # Record in history
            self._record_history(order)

            # Persist order
            if self._persistence_enabled:
                self._persist_order(order)

            logger.debug(f"Added order {order.order_id} to order book")
            return True

    def update_order(self, order: Order) -> bool:
        """
        Update an existing order in the order book.

        Args:
            order: Updated order

        Returns:
            True if the order was updated, False if not found
        """
        with self._lock:
            # Check if order exists
            if order.order_id not in self._orders:
                logger.warning(f"Order {order.order_id} not found for update")
                return False

            # Get the old order
            old_order = self._orders[order.order_id]

            # Update indices if status changed
            if old_order.status != order.status:
                self._orders_by_status[old_order.status].discard(order.order_id)
                self._orders_by_status[order.status].add(order.order_id)
                logger.info(f"Order {order.order_id} status changed: {old_order.status.value} -> {order.status.value}")

            # Update indices if exchange changed
            if old_order.exchange_id != order.exchange_id:
                if old_order.exchange_id:
                    self._orders_by_exchange[old_order.exchange_id].discard(order.order_id)
                if order.exchange_id:
                    self._orders_by_exchange[order.exchange_id].add(order.order_id)

            # Update indices if strategy changed
            if old_order.strategy_id != order.strategy_id:
                if old_order.strategy_id:
                    self._orders_by_strategy[old_order.strategy_id].discard(order.order_id)
                if order.strategy_id:
                    self._orders_by_strategy[order.strategy_id].add(order.order_id)

            # Update indices if parent changed
            if old_order.parent_order_id != order.parent_order_id:
                if old_order.parent_order_id:
                    self._child_orders[old_order.parent_order_id].discard(order.order_id)
                if order.parent_order_id:
                    self._child_orders[order.parent_order_id].add(order.order_id)

            # Update main collection
            self._orders[order.order_id] = order

            # Record in history
            self._record_history(order)

            # Persist order
            if self._persistence_enabled:
                self._persist_order(order)

            logger.debug(f"Updated order {order.order_id} in order book")
            return True

    def remove_order(self, order_id: str) -> bool:
        """
        Remove an order from the order book.

        Args:
            order_id: ID of the order to remove

        Returns:
            True if the order was removed, False if not found
        """
        with self._lock:
            # Check if order exists
            if order_id not in self._orders:
                logger.warning(f"Order {order_id} not found for removal")
                return False

            # Get the order
            order = self._orders[order_id]

            # Remove from indices
            self._orders_by_status[order.status].discard(order_id)
            self._orders_by_symbol[order.symbol].discard(order_id)

            if order.exchange_id:
                self._orders_by_exchange[order.exchange_id].discard(order_id)

            if order.strategy_id:
                self._orders_by_strategy[order.strategy_id].discard(order_id)

            if order.parent_order_id:
                self._child_orders[order.parent_order_id].discard(order_id)

            # Remove from main collection
            del self._orders[order_id]

            # We keep the history for reference

            logger.debug(f"Removed order {order_id} from order book")
            return True

    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get an order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order or None if not found
        """
        with self._lock:
            return self._orders.get(order_id)

    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """
        Get all orders with a specific status.

        Args:
            status: Order status

        Returns:
            List of orders with the specified status
        """
        with self._lock:
            return [self._orders[order_id] for order_id in self._orders_by_status[status]
                   if order_id in self._orders]  # Guard against inconsistency

    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """
        Get all orders for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of orders for the specified symbol
        """
        with self._lock:
            return [self._orders[order_id] for order_id in self._orders_by_symbol[symbol]
                   if order_id in self._orders]  # Guard against inconsistency

    def get_orders_by_exchange(self, exchange_id: str) -> List[Order]:
        """
        Get all orders for a specific exchange.

        Args:
            exchange_id: Exchange ID

        Returns:
            List of orders for the specified exchange
        """
        with self._lock:
            return [self._orders[order_id] for order_id in self._orders_by_exchange[exchange_id]
                   if order_id in self._orders]  # Guard against inconsistency

    def get_orders_by_strategy(self, strategy_id: str) -> List[Order]:
        """
        Get all orders for a specific strategy.

        Args:
            strategy_id: Strategy ID

        Returns:
            List of orders for the specified strategy
        """
        with self._lock:
            return [self._orders[order_id] for order_id in self._orders_by_strategy[strategy_id]
                   if order_id in self._orders]  # Guard against inconsistency

    def get_child_orders(self, parent_order_id: str) -> List[Order]:
        """
        Get all child orders for a parent order.

        Args:
            parent_order_id: Parent order ID

        Returns:
            List of child orders
        """
        with self._lock:
            return [self._orders[order_id] for order_id in self._child_orders[parent_order_id]
                   if order_id in self._orders]  # Guard against inconsistency

    def get_active_orders(self) -> List[Order]:
        """
        Get all active orders (orders that are not in a terminal state).

        Returns:
            List of active orders
        """
        with self._lock:
            active_statuses = [
                OrderStatus.CREATED,
                OrderStatus.PENDING,
                OrderStatus.OPEN,
                OrderStatus.PARTIALLY_FILLED,
                OrderStatus.PENDING_CANCEL,
                OrderStatus.PENDING_UPDATE,
                OrderStatus.WORKING
            ]

            active_orders = []
            for status in active_statuses:
                active_orders.extend(self.get_orders_by_status(status))

            return active_orders

    def get_orders_by_side(self, side: OrderSide) -> List[Order]:
        """
        Get all orders with a specific side.

        Args:
            side: Order side (buy or sell)

        Returns:
            List of orders with the specified side
        """
        with self._lock:
            return [order for order in self._orders.values() if order.side == side]

    def get_orders_by_type(self, order_type: OrderType) -> List[Order]:
        """
        Get all orders with a specific type.

        Args:
            order_type: Order type

        Returns:
            List of orders with the specified type
        """
        with self._lock:
            return [order for order in self._orders.values() if order.order_type == order_type]

    def get_all_orders(self) -> List[Order]:
        """
        Get all orders in the order book.

        Returns:
            List of all orders
        """
        with self._lock:
            return list(self._orders.values())

    def get_order_count(self) -> int:
        """
        Get the total number of orders in the order book.

        Returns:
            Number of orders
        """
        with self._lock:
            return len(self._orders)

    def get_order_count_by_status(self) -> Dict[str, int]:
        """
        Get the number of orders by status.

        Returns:
            Dictionary mapping status names to counts
        """
        with self._lock:
            return {status.value: len(orders) for status, orders in self._orders_by_status.items()}

    def get_position_by_symbol(self, symbol: str) -> float:
        """
        Calculate the net position for a symbol based on active orders.

        Args:
            symbol: Trading symbol

        Returns:
            Net position (positive for long, negative for short)
        """
        with self._lock:
            position = 0.0
            orders = self.get_orders_by_symbol(symbol)

            for order in orders:
                # Only count filled quantity
                if order.filled_quantity > 0:
                    if order.side == OrderSide.BUY:
                        position += order.filled_quantity
                    else:
                        position -= order.filled_quantity

            return position

    def _record_history(self, order: Order) -> None:
        """
        Record order in history.

        Args:
            order: Order to record
        """
        if order.order_id not in self._order_history:
            self._order_history[order.order_id] = []

        # Add current state to history
        history_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": order.status.value,
            "filled_quantity": order.filled_quantity,
            "average_price": order.average_price
        }

        self._order_history[order.order_id].append(history_entry)

    def get_order_history(self, order_id: str) -> List[Dict[str, Any]]:
        """
        Get history of an order.

        Args:
            order_id: Order ID

        Returns:
            List of history entries for the order
        """
        with self._lock:
            return self._order_history.get(order_id, [])[:]  # Return a copy

    def export_orders_csv(self, file_path: Optional[str] = None) -> str:
        """
        Export all orders to a CSV file.

        Args:
            file_path: Path to save CSV file (optional)

        Returns:
            Path to the saved file
        """
        with self._lock:
            # Convert orders to dict for DataFrame
            orders_data = []
            for order in self._orders.values():
                order_dict = order.to_dict()
                # Flatten nested data for CSV
                order_dict.pop('params', None)
                order_dict.pop('tags', None)
                orders_data.append(order_dict)

            # Create DataFrame
            df = pd.DataFrame(orders_data)

            # Create default file path if not provided
            if file_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"order_export_{timestamp}.csv"

            # Save to CSV
            df.to_csv(file_path, index=False)
            logger.info(f"Exported {len(orders_data)} orders to {file_path}")

            return file_path

    def _persist_order(self, order: Order) -> None:
        """
        Persist order to disk.

        Args:
            order: Order to persist
        """
        if not self._persistence_enabled:
            return

        try:
            # Convert order to dict
            order_dict = order.to_dict()

            # Create file path
            file_path = self._persistence_dir / f"{order.order_id}.json"

            # Save to file
            with open(file_path, 'w') as f:
                json.dump(order_dict, f, indent=2)

        except Exception as e:
            logger.error(f"Error persisting order {order.order_id}: {str(e)}")

    def load_persisted_orders(self) -> int:
        """
        Load persisted orders from disk.

        Returns:
            Number of orders loaded
        """
        if not self._persistence_enabled:
            return 0

        count = 0

        # Get all JSON files in persistence directory
        for file_path in self._persistence_dir.glob("*.json"):
            try:
                # Load order from file
                with open(file_path, 'r') as f:
                    order_dict = json.load(f)

                # Create order from dict
                order = Order.from_dict(order_dict)

                # Add to order book
                if self.add_order(order):
                    count += 1

            except Exception as e:
                logger.error(f"Error loading order from {file_path}: {str(e)}")

        logger.info(f"Loaded {count} persisted orders")
        return count

    def cleanup_old_orders(self, days: Optional[int] = None) -> int:
        """
        Remove old completed orders from memory (but not from persistence).

        Args:
            days: Number of days to keep orders (default from config)

        Returns:
            Number of orders removed
        """
        with self._lock:
            # Use default if not specified
            if days is None:
                days = self._cleanup_after_days

            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=days)

            # Find orders to remove
            to_remove = []
            for order_id, order in self._orders.items():
                # Only remove completed orders
                if not order.is_complete:
                    continue

                # Check if older than cutoff
                if order.update_time and order.update_time < cutoff_date:
                    to_remove.append(order_id)

            # Remove orders
            for order_id in to_remove:
                self.remove_order(order_id)

            logger.info(f"Cleaned up {len(to_remove)} old orders")
            return len(to_remove)

    def auto_cleanup(self) -> bool:
        """
        Perform automatic cleanup if needed based on configured interval.

        Returns:
            True if cleanup was performed, False otherwise
        """
        # Check if cleanup is due
        now = datetime.now()
        hours_since_last = (now - self._last_cleanup).total_seconds() / 3600

        if hours_since_last >= self._cleanup_interval_hours:
            # Perform cleanup
            self.cleanup_old_orders()
            self._last_cleanup = now
            return True

        return False

    def find_orders(self, criteria: Dict[str, Any]) -> List[Order]:
        """
        Find orders matching the given criteria.

        Args:
            criteria: Dictionary of criteria to match

        Returns:
            List of matching orders
        """
        with self._lock:
            matches = []

            for order in self._orders.values():
                match = True

                for key, value in criteria.items():
                    # Handle special case for nested dict params
                    if key == 'params' and isinstance(value, dict):
                        for param_key, param_value in value.items():
                            if param_key not in order.params or order.params[param_key] != param_value:
                                match = False
                                break
                    # Handle special case for status as string
                    elif key == 'status' and isinstance(value, str):
                        if order.status.value != value:
                            match = False
                    # Handle all other cases
                    elif not hasattr(order, key) or getattr(order, key) != value:
                        match = False

                if match:
                    matches.append(order)

            return matches

    def group_orders_by(self, attribute: str) -> Dict[Any, List[Order]]:
        """
        Group orders by a specific attribute.

        Args:
            attribute: Attribute to group by

        Returns:
            Dictionary mapping attribute values to lists of orders
        """
        with self._lock:
            result = defaultdict(list)

            for order in self._orders.values():
                if hasattr(order, attribute):
                    key = getattr(order, attribute)
                    # Handle enum values
                    if hasattr(key, 'value'):
                        key = key.value
                    result[key].append(order)

            return dict(result)

    def bulk_update_status(self, order_ids: List[str], new_status: OrderStatus,
                          message: Optional[str] = None) -> int:
        """
        Update status for multiple orders at once.

        Args:
            order_ids: List of order IDs to update
            new_status: New status to set
            message: Optional status message

        Returns:
            Number of orders successfully updated
        """
        with self._lock:
            updated = 0

            for order_id in order_ids:
                order = self.get_order(order_id)
                if order:
                    order.status = new_status
                    if message:
                        order.status_message = message
                    order.update_time = datetime.utcnow()

                    if self.update_order(order):
                        updated += 1

            return updated

    def purge_all(self) -> int:
        """
        Remove all orders from the order book (for testing/reset).

        Returns:
            Number of orders removed
        """
        with self._lock:
            count = len(self._orders)

            # Clear all collections
            self._orders.clear()
            self._orders_by_status.clear()
            self._orders_by_symbol.clear()
            self._orders_by_exchange.clear()
            self._orders_by_strategy.clear()
            self._child_orders.clear()

            logger.warning(f"Purged {count} orders from order book")
            return count

    def __len__(self) -> int:
        """Get the number of orders in the order book"""
        return self.get_order_count()

    def __contains__(self, order_id: str) -> bool:
        """Check if an order exists in the order book"""
        return order_id in self._orders