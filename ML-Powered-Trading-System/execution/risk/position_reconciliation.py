"""
position_reconciliation.py - Position tracking and reconciliation

This module provides position tracking and reconciliation capabilities,
ensuring that the system's internal position state matches the exchange's
position state. It handles position updates, reconciliation with exchange
positions, and provides a consistent view of positions to other system
components.

The module implements proactive position tracking and periodic reconciliation
to maintain position accuracy in the face of potential execution errors,
exchange issues, or system failures.
"""

import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass

from core.event_bus import EventTopics, create_event, get_event_bus
from execution.order.order import Order, OrderStatus, OrderSide

# Configure logger
logger = logging.getLogger(__name__)


class PositionReconciliationStatus(Enum):
    """Status of a position reconciliation"""
    MATCHED = "matched"  # Positions match
    ADJUSTED = "adjusted"  # Position was adjusted to match exchange
    FAILED = "failed"  # Reconciliation failed


@dataclass
class Position:
    """Position information"""
    symbol: str
    size: float
    avg_price: float
    updated_at: datetime
    exchange_id: str
    open_orders: Optional[List[str]] = None

    def __post_init__(self):
        if self.open_orders is None:
            self.open_orders = []


class PositionManager:
    """
    Manages positions and ensures consistency with exchange state.

    Responsibilities:
    - Track positions across all instruments
    - Update positions based on executed orders
    - Reconcile positions with exchange data
    - Provide position information to other components
    - Track position metrics and history
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the position manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._event_bus = get_event_bus()

        # Positions by symbol
        self.positions: Dict[str, Position] = {}

        # Historical positions
        self.position_history: Dict[str, List[Dict[str, Any]]] = {}

        # Position reconciliation settings
        self.reconciliation_settings = {
            "auto_reconcile": self.config.get("auto_reconcile", True),
            "reconciliation_interval_minutes": self.config.get("reconciliation_interval_minutes", 60),
            "reconciliation_size_tolerance": self.config.get("reconciliation_size_tolerance", 0.0),
            "reconciliation_price_tolerance": self.config.get("reconciliation_price_tolerance", 0.005),  # 0.5%
            "max_retry_count": self.config.get("max_retry_count", 3)
        }

        # Last reconciliation time
        self.last_reconciliation_time = None

        logger.info("PositionManager initialized")

    def get_position(self, symbol: str) -> float:
        """
        Get current position size for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position size (positive for long, negative for short, 0 for no position)
        """
        position = self.positions.get(symbol)
        return position.size if position else 0.0

    def get_position_details(self, symbol: str) -> Optional[Position]:
        """
        Get detailed position information for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position object or None if no position
        """
        return self.positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        """
        Get all current positions.

        Returns:
            Dictionary mapping symbols to Position objects
        """
        return self.positions.copy()

    def get_position_value(self, symbol: str, current_price: Optional[float] = None) -> float:
        """
        Get current position value.

        Args:
            symbol: Trading symbol
            current_price: Current market price (optional, uses average price if not provided)

        Returns:
            Position value in quote currency
        """
        position = self.positions.get(symbol)
        if not position or position.size == 0:
            return 0.0

        price = current_price if current_price is not None else position.avg_price
        return position.size * price

    def get_total_position_value(self, price_provider=None) -> float:
        """
        Get total value of all positions.

        Args:
            price_provider: Optional callable to get current prices

        Returns:
            Total position value in quote currency
        """
        total_value = 0.0

        for symbol, position in self.positions.items():
            if position.size == 0:
                continue

            if price_provider:
                try:
                    price = price_provider(symbol)
                except Exception as e:
                    logger.warning(f"Error getting price for {symbol}: {str(e)}")
                    price = position.avg_price
            else:
                price = position.avg_price

            total_value += abs(position.size * price)

        return total_value

    def get_net_exposure(self, price_provider=None) -> float:
        """
        Get net market exposure (long - short).

        Args:
            price_provider: Optional callable to get current prices

        Returns:
            Net exposure value
        """
        net_value = 0.0

        for symbol, position in self.positions.items():
            if position.size == 0:
                continue

            if price_provider:
                try:
                    price = price_provider(symbol)
                except Exception as e:
                    logger.warning(f"Error getting price for {symbol}: {str(e)}")
                    price = position.avg_price
            else:
                price = position.avg_price

            net_value += position.size * price

        return net_value

    def update_from_fill(self, order: Order, fill_size: float, fill_price: float) -> None:
        """
        Update position based on an order fill.

        Args:
            order: Filled order
            fill_size: Size of the fill
            fill_price: Price of the fill
        """
        symbol = order.symbol

        # Get current position or create new one
        position = self.positions.get(symbol)
        if not position:
            position = Position(
                symbol=symbol,
                size=0.0,
                avg_price=0.0,
                updated_at=datetime.now(),
                exchange_id=order.exchange_id,
                open_orders=[]
            )
            self.positions[symbol] = position

        # Calculate new position
        current_size = position.size
        current_value = current_size * position.avg_price if current_size != 0 else 0

        # Apply fill based on order side
        if order.side == OrderSide.BUY:
            fill_value = fill_size * fill_price
            new_size = current_size + fill_size
            new_value = current_value + fill_value
        else:  # SELL
            fill_value = fill_size * fill_price
            new_size = current_size - fill_size
            new_value = current_value - fill_value

        # Calculate new average price
        if new_size != 0:
            # If position flipped from long to short or vice versa, reset average price
            if current_size * new_size <= 0:  # Different signs or zero
                position.avg_price = fill_price
            else:
                position.avg_price = abs(new_value / new_size)
        else:
            position.avg_price = 0

        # Update position
        old_size = position.size
        position.size = new_size
        position.updated_at = datetime.now()

        # Remove order from open orders if present
        if order.order_id in position.open_orders:
            position.open_orders.remove(order.order_id)

        # Add to position history
        self._add_to_history(symbol, old_size, new_size, position.avg_price, order.order_id)

        # Publish position update event
        self._publish_position_event(symbol, new_size, position.avg_price, order.order_id)

        logger.info(f"Updated position for {symbol}: {old_size} -> {new_size}, avg price: {position.avg_price}")

    def set_position(self, symbol: str, size: float, avg_price: Optional[float] = None, exchange_id: Optional[str] = None) -> None:
        """
        Set position directly (for reconciliation or initialization).

        Args:
            symbol: Trading symbol
            size: New position size
            avg_price: Average price (optional)
            exchange_id: Exchange ID (optional)
        """
        position = self.positions.get(symbol)
        old_size = position.size if position else 0

        if not position:
            if avg_price is None:
                raise ValueError(f"Average price required for new position: {symbol}")

            if exchange_id is None:
                raise ValueError(f"Exchange ID required for new position: {symbol}")

            position = Position(
                symbol=symbol,
                size=size,
                avg_price=avg_price,
                updated_at=datetime.now(),
                exchange_id=exchange_id,
                open_orders=[]
            )
            self.positions[symbol] = position
        else:
            position.size = size
            if avg_price is not None:
                position.avg_price = avg_price
            position.updated_at = datetime.now()

        # Add to position history
        self._add_to_history(symbol, old_size, size, position.avg_price, "manual_update")

        # Publish position update event
        self._publish_position_event(symbol, size, position.avg_price, "manual_update")

        logger.info(f"Set position for {symbol}: {old_size} -> {size}, avg price: {position.avg_price}")

    def add_open_order(self, symbol: str, order_id: str, exchange_id: str) -> None:
        """
        Add an open order to a position.

        Args:
            symbol: Trading symbol
            order_id: Order ID
            exchange_id: Exchange ID
        """
        position = self.positions.get(symbol)
        if not position:
            position = Position(
                symbol=symbol,
                size=0.0,
                avg_price=0.0,
                updated_at=datetime.now(),
                exchange_id=exchange_id,
                open_orders=[order_id]
            )
            self.positions[symbol] = position
        else:
            if order_id not in position.open_orders:
                position.open_orders.append(order_id)

        logger.debug(f"Added open order {order_id} to position {symbol}, total open orders: {len(position.open_orders)}")

    def remove_open_order(self, symbol: str, order_id: str) -> None:
        """
        Remove an open order from a position.

        Args:
            symbol: Trading symbol
            order_id: Order ID
        """
        position = self.positions.get(symbol)
        if position and order_id in position.open_orders:
            position.open_orders.remove(order_id)
            logger.debug(f"Removed open order {order_id} from position {symbol}, remaining open orders: {len(position.open_orders)}")

    def reconcile_with_exchange(self, exchange_gateway) -> Dict[str, PositionReconciliationStatus]:
        """
        Reconcile positions with exchange data.

        Args:
            exchange_gateway: Exchange gateway to query

        Returns:
            Dictionary mapping symbols to reconciliation status
        """
        now = datetime.now()

        # Skip if we've run recently
        if self.last_reconciliation_time and (now - self.last_reconciliation_time).total_seconds() < self.reconciliation_settings["reconciliation_interval_minutes"] * 60:
            logger.debug("Skipping position reconciliation, too soon since last run")
            return {}

        self.last_reconciliation_time = now
        logger.info("Starting position reconciliation with exchange")

        results = {}

        try:
            # Get exchange positions
            exchange_positions = exchange_gateway.get_all_positions()

            if exchange_positions is None:
                logger.error("Failed to retrieve positions from exchange")
                return {}

            # Set of processed symbols
            processed_symbols = set()

            # Check each internal position against exchange
            for symbol, position in self.positions.items():
                processed_symbols.add(symbol)

                # Skip positions with open orders as they may be in flux
                if position.open_orders:
                    logger.debug(f"Skipping reconciliation for {symbol} with open orders: {position.open_orders}")
                    results[symbol] = PositionReconciliationStatus.MATCHED
                    continue

                # Check if position exists on exchange
                if symbol not in exchange_positions:
                    if position.size != 0:
                        if self.reconciliation_settings["auto_reconcile"]:
                            logger.warning(f"Position {symbol} not found on exchange but exists internally with size {position.size}, resetting")
                            old_size = position.size
                            position.size = 0
                            position.avg_price = 0
                            position.updated_at = now

                            # Add to history
                        self._add_to_history(symbol, old_size, exchange_size, position.avg_price, "reconciliation")

                        # Publish event
                        self._publish_position_event(symbol, exchange_size, position.avg_price, "reconciliation")

                        results[symbol] = PositionReconciliationStatus.ADJUSTED
                    else:
                        logger.warning(f"Position size mismatch for {symbol}: internal={position.size}, exchange={exchange_size}")
                        results[symbol] = PositionReconciliationStatus.FAILED
                else:
                    # Check average price if size is non-zero
                    if position.size != 0 and exchange_size != 0:
                        price_diff_pct = abs(exchange_price - position.avg_price) / position.avg_price if position.avg_price > 0 else 0
                        price_tolerance = self.reconciliation_settings["reconciliation_price_tolerance"]

                        if price_diff_pct > price_tolerance:
                            if self.reconciliation_settings["auto_reconcile"]:
                                logger.warning(f"Position price mismatch for {symbol}: internal={position.avg_price}, exchange={exchange_price}, adjusting")
                                position.avg_price = exchange_price
                                position.updated_at = now

                                # Publish event
                                self._publish_position_event(symbol, position.size, exchange_price, "reconciliation")

                                results[symbol] = PositionReconciliationStatus.ADJUSTED
                            else:
                                logger.warning(f"Position price mismatch for {symbol}: internal={position.avg_price}, exchange={exchange_price}")
                                results[symbol] = PositionReconciliationStatus.FAILED
                        else:
                            results[symbol] = PositionReconciliationStatus.MATCHED
                    else:
                        results[symbol] = PositionReconciliationStatus.MATCHED

            # Check for positions on exchange that aren't tracked internally
            for symbol, exchange_position in exchange_positions.items():
                if symbol in processed_symbols:
                    continue

                exchange_size = exchange_position.get('size', 0)

                # Skip zero positions
                if exchange_size == 0:
                    continue

                exchange_price = exchange_position.get('avg_price', 0)
                exchange_id = exchange_position.get('exchange_id', '')

                if self.reconciliation_settings["auto_reconcile"]:
                    logger.warning(f"Found position on exchange not tracked internally: {symbol}, size={exchange_size}, adding")

                    # Add position
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        size=exchange_size,
                        avg_price=exchange_price,
                        updated_at=now,
                        exchange_id=exchange_id,
                        open_orders=[]
                    )

                    # Add to history
                    self._add_to_history(symbol, 0, exchange_size, exchange_price, "reconciliation")

                    # Publish event
                    self._publish_position_event(symbol, exchange_size, exchange_price, "reconciliation")

                    results[symbol] = PositionReconciliationStatus.ADJUSTED
                else:
                    logger.warning(f"Found position on exchange not tracked internally: {symbol}, size={exchange_size}")
                    results[symbol] = PositionReconciliationStatus.FAILED

            # Count results by status
            status_counts = {status: 0 for status in PositionReconciliationStatus}
            for status in results.values():
                status_counts[status] += 1

            logger.info(f"Position reconciliation completed: {status_counts[PositionReconciliationStatus.MATCHED]} matched, "
                       f"{status_counts[PositionReconciliationStatus.ADJUSTED]} adjusted, "
                       f"{status_counts[PositionReconciliationStatus.FAILED]} failed")

            # Publish reconciliation event
            event = create_event(
                EventTopics.POSITION_RECONCILIATION_COMPLETED,
                {
                    "timestamp": now.timestamp(),
                    "matched": status_counts[PositionReconciliationStatus.MATCHED],
                    "adjusted": status_counts[PositionReconciliationStatus.ADJUSTED],
                    "failed": status_counts[PositionReconciliationStatus.FAILED]
                }
            )
            self._event_bus.publish(event)

            return results

        except Exception as e:
            logger.error(f"Error reconciling positions: {str(e)}", exc_info=True)
            return {symbol: PositionReconciliationStatus.FAILED for symbol in self.positions}

    def _add_to_history(self, symbol: str, old_size: float, new_size: float, price: float, source: str) -> None:
        """
        Add position change to history.

        Args:
            symbol: Trading symbol
            old_size: Previous position size
            new_size: New position size
            price: Position price
            source: Source of the change (order ID or 'reconciliation', etc.)
        """
        if symbol not in self.position_history:
            self.position_history[symbol] = []

        self.position_history[symbol].append({
            "timestamp": datetime.now().timestamp(),
            "old_size": old_size,
            "new_size": new_size,
            "price": price,
            "source": source
        })

        # Limit history size
        max_history = self.config.get("max_position_history", 100)
        if len(self.position_history[symbol]) > max_history:
            self.position_history[symbol] = self.position_history[symbol][-max_history:]

    def _publish_position_event(self, symbol: str, size: float, price: float, source: str) -> None:
        """
        Publish position update event.

        Args:
            symbol: Trading symbol
            size: Position size
            price: Position price
            source: Source of the update
        """
        event = create_event(
            EventTopics.POSITION_UPDATED,
            {
                "symbol": symbol,
                "size": size,
                "avg_price": price,
                "value": size * price,
                "timestamp": time.time(),
                "source": source
            }
        )
        self._event_bus.publish(event)

    def get_position_history(self, symbol: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get position history for a symbol.

        Args:
            symbol: Trading symbol
            limit: Optional limit on number of records

        Returns:
            List of position history records
        """
        history = self.position_history.get(symbol, [])

        if limit:
            history = history[-limit:]

        return history

    def calculate_realized_pnl(self, symbol: str, fills: List[Dict[str, Any]]) -> float:
        """
        Calculate realized P&L for a symbol based on fills.

        Args:
            symbol: Trading symbol
            fills: List of fill records

        Returns:
            Realized P&L
        """
        if not fills:
            return 0.0

        # Sort fills by timestamp
        sorted_fills = sorted(fills, key=lambda x: x.get("timestamp", 0))

        # FIFO implementation for P&L calculation
        realized_pnl = 0.0
        position = 0.0
        queue = []  # (size, price) tuples

        for fill in sorted_fills:
            side = fill.get("side")
            size = fill.get("size", 0)
            price = fill.get("price", 0)

            if side == "buy":
                # Add to position
                position += size
                queue.append((size, price))
            elif side == "sell":
                # Close position (FIFO)
                remaining = size
                while remaining > 0 and queue:
                    entry_size, entry_price = queue[0]

                    # Calculate size to close
                    close_size = min(entry_size, remaining)

                    # Calculate P&L for this portion
                    pnl = close_size * (price - entry_price)
                    realized_pnl += pnl

                    # Update remaining size
                    remaining -= close_size

                    # Update or remove entry
                    if close_size == entry_size:
                        queue.pop(0)
                    else:
                        queue[0] = (entry_size - close_size, entry_price)

                # Update position
                position -= size

                # Handle short positions
                if position < 0 and remaining > 0:
                    queue.append((remaining, price))

        return realized_pnl

    def reset(self) -> None:
        """Reset position manager state."""
        self.positions = {}
        self.position_history = {}
        self.last_reconciliation_time = None

        logger.info("Position manager state reset")

    def calculate_portfolio_metrics(self, price_provider=None) -> Dict[str, Any]:
        """
        Calculate portfolio metrics based on current positions.

        Args:
            price_provider: Optional callable to get current prices

        Returns:
            Dictionary of portfolio metrics
        """
        total_long_value = 0.0
        total_short_value = 0.0
        position_count = 0
        symbols = []

        for symbol, position in self.positions.items():
            if position.size == 0:
                continue

            position_count += 1
            symbols.append(symbol)

            if price_provider:
                try:
                    price = price_provider(symbol)
                except Exception as e:
                    logger.warning(f"Error getting price for {symbol}: {str(e)}")
                    price = position.avg_price
            else:
                price = position.avg_price

            position_value = position.size * price

            if position.size > 0:
                total_long_value += position_value
            else:
                total_short_value += abs(position_value)

        total_exposure = total_long_value + total_short_value
        net_exposure = total_long_value - total_short_value

        return {
            "position_count": position_count,
            "symbols": symbols,
            "total_long_value": total_long_value,
            "total_short_value": total_short_value,
            "total_exposure": total_exposure,
            "net_exposure": net_exposure,
            "exposure_ratio": total_short_value / total_long_value if total_long_value > 0 else 0.0,
            "positions": self.positions
        }
                            self._add_to_history(symbol, old_size, 0, 0, "reconciliation")

                            # Publish event
                            self._publish_position_event(symbol, 0, 0, "reconciliation")

                            results[symbol] = PositionReconciliationStatus.ADJUSTED
                        else:
                            logger.warning(f"Position {symbol} not found on exchange but exists internally with size {position.size}")
                            results[symbol] = PositionReconciliationStatus.FAILED
                    else:
                        results[symbol] = PositionReconciliationStatus.MATCHED

                    continue

                # Compare with exchange position
                exchange_position = exchange_positions[symbol]
                exchange_size = exchange_position.get('size', 0)
                exchange_price = exchange_position.get('avg_price', 0)

                # Check position size within tolerance
                size_diff = abs(exchange_size - position.size)
                size_tolerance = self.reconciliation_settings["reconciliation_size_tolerance"]

                if size_diff > size_tolerance:
                    if self.reconciliation_settings["auto_reconcile"]:
                        logger.warning(f"Position size mismatch for {symbol}: internal={position.size}, exchange={exchange_size}, adjusting")
                        old_size = position.size
                        position.size = exchange_size
                        position.updated_at = now

                        # Add to history