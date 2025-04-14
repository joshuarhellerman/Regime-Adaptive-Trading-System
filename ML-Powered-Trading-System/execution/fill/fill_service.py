"""
fill_service.py - Service for processing and reconciling order fills

This module provides a central service for processing fills, updating orders,
and ensuring position reconciliation. It serves as the system's source of truth
for fill data and manages the fill lifecycle.
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from datetime import datetime
import uuid
import json
from pathlib import Path
import os

from core.event_bus import EventBus, Event, EventTopics, EventPriority, create_event
from core.state_manager import StateManager, StateScope
from execution.order.order import Order, OrderStatus
from execution.order.order_book import OrderBook
from execution.fill.fill_model import Fill
from execution.risk.position_reconciliation import PositionReconciliation
from execution.risk.post_trade_reconciliation import PostTradeReconciliation

logger = logging.getLogger(__name__)

class FillService:
    """
    Service for processing and reconciling order fills.

    Responsibilities:
    - Process fill events from exchanges
    - Update order status based on fills
    - Ensure fill data is persisted
    - Trigger position reconciliation
    - Provide fill history and analytics
    """

    def __init__(self,
                event_bus: EventBus,
                state_manager: StateManager,
                order_book: OrderBook,
                position_reconciliation: Optional[PositionReconciliation] = None,
                post_trade_reconciliation: Optional[PostTradeReconciliation] = None,
                config: Dict[str, Any] = None):
        """
        Initialize the fill service.

        Args:
            event_bus: System event bus
            state_manager: State management system
            order_book: Order tracking system
            position_reconciliation: Position reconciliation service
            post_trade_reconciliation: Post-trade reconciliation service
            config: Configuration dictionary
        """
        self.event_bus = event_bus
        self.state_manager = state_manager
        self.order_book = order_book
        self.position_reconciliation = position_reconciliation
        self.post_trade_reconciliation = post_trade_reconciliation
        self.config = config or {}

        # Fill storage
        self._fills: Dict[str, Fill] = {}  # fill_id -> Fill
        self._order_fills: Dict[str, List[str]] = {}  # order_id -> List[fill_id]
        self._lock = threading.RLock()

        # Persistence
        self._persistence_enabled = self.config.get('persistence_enabled', True)
        self._persistence_dir = Path(self.config.get('fill_persistence_dir', 'data/fills'))
        if self._persistence_enabled:
            self._persistence_dir.mkdir(parents=True, exist_ok=True)

        # Register with state manager
        self._register_state()

        # Subscribe to fill events
        self._subscribe_to_events()

        logger.info("FillService initialized")

    def _register_state(self):
        """Register fill service state with state manager."""
        # Register fills as persistent state
        self.state_manager.set_scope("fills", StateScope.PERSISTENT)

    def _subscribe_to_events(self):
        """Subscribe to fill-related events."""
        # Subscribe to fill events from exchanges
        self.event_bus.subscribe("exchange.fill.*", self.process_fill)

        # Subscribe to fill events from paper trading
        self.event_bus.subscribe("paper.order.filled", self.process_fill)

        # Subscribe to simulation fills
        self.event_bus.subscribe("simulation.fill", self.process_fill)

    def process_fill(self, event: Event) -> None:
        """
        Process a fill event.

        Args:
            event: Fill event from exchange or simulation
        """
        try:
            fill_data = event.data

            # Create Fill object if needed
            if not isinstance(fill_data, Fill):
                fill = Fill.from_dict(fill_data)
            else:
                fill = fill_data

            # Start a transaction for atomic state updates
            with self.state_manager.transaction():
                self._process_fill_internal(fill)
        except Exception as e:
            logger.error(f"Error processing fill: {e}", exc_info=True)

    def _process_fill_internal(self, fill: Fill) -> None:
        """
        Internal method to process a fill.

        Args:
            fill: The fill to process
        """
        with self._lock:
            # Store fill
            self._fills[fill.fill_id] = fill

            # Update order fills index
            if fill.order_id not in self._order_fills:
                self._order_fills[fill.order_id] = []
            self._order_fills[fill.order_id].append(fill.fill_id)

            # Update state manager
            self.state_manager.set(f"fills.{fill.fill_id}", fill.to_dict())

            # Get order
            order = self.order_book.get_order(fill.order_id)
            if not order:
                logger.error(f"Fill received for unknown order: {fill.order_id}")
                return

            # Update order with fill information
            updated_order = self._apply_fill_to_order(order, fill)

            # Update order in order book
            self.order_book.update_order(updated_order)

            # Persist fill if enabled
            if self._persistence_enabled:
                self._persist_fill(fill)

            # Publish fill event
            self._publish_fill_event(fill)

            # Trigger position reconciliation
            self._trigger_position_reconciliation(fill)

            # Trigger post-trade reconciliation
            self._trigger_post_trade_reconciliation(fill)

    def _apply_fill_to_order(self, order: Order, fill: Fill) -> Order:
        """
        Apply a fill to an order, updating its state.

        Args:
            order: The order to update
            fill: The fill to apply

        Returns:
            Updated order
        """
        # Create a copy of the order for updating
        updated_order = order.copy()

        # Calculate new filled quantity
        new_filled_quantity = updated_order.filled_quantity + float(fill.quantity)

        # Calculate new average price
        if updated_order.filled_quantity == 0:
            # First fill
            new_average_price = float(fill.price)
        else:
            # Weighted average
            current_value = updated_order.filled_quantity * (updated_order.average_price or 0)
            new_value = float(fill.quantity) * float(fill.price)
            new_average_price = (current_value + new_value) / new_filled_quantity

        # Update order fields
        updated_order.filled_quantity = new_filled_quantity
        updated_order.average_price = new_average_price
        updated_order.update_time = datetime.utcnow()

        # Update status based on fill
        if abs(new_filled_quantity - updated_order.quantity) < 1e-8:
            # Order is completely filled
            updated_order.status = OrderStatus.FILLED
            updated_order.execution_time = datetime.utcnow()
        elif new_filled_quantity > 0:
            # Order is partially filled
            updated_order.status = OrderStatus.PARTIALLY_FILLED

        # Add fill information to order metadata
        if 'fills' not in updated_order.params:
            updated_order.params['fills'] = []

        updated_order.params['fills'].append(fill.fill_id)

        logger.info(f"Applied fill {fill.fill_id} to order {order.order_id}: "
                  f"{float(fill.quantity)} @ {float(fill.price)}, "
                  f"total filled: {new_filled_quantity}/{updated_order.quantity}")

        return updated_order

    def _publish_fill_event(self, fill: Fill) -> None:
        """
        Publish a fill event.

        Args:
            fill: Fill to publish event for
        """
        # Create and publish fill event
        event = create_event(
            f"execution.fill.{fill.instrument}",
            fill.to_dict(),
            priority=EventPriority.HIGH,
            source="fill_service"
        )

        self.event_bus.publish(event)

    def _trigger_position_reconciliation(self, fill: Fill) -> None:
        """
        Trigger position reconciliation after a fill.

        Args:
            fill: Fill that triggered reconciliation
        """
        if self.position_reconciliation:
            try:
                self.position_reconciliation.reconcile_fill(fill)
            except Exception as e:
                logger.error(f"Error in position reconciliation: {e}")

    def _trigger_post_trade_reconciliation(self, fill: Fill) -> None:
        """
        Trigger post-trade reconciliation after a fill.

        Args:
            fill: Fill that triggered reconciliation
        """
        if self.post_trade_reconciliation:
            try:
                self.post_trade_reconciliation.process_fill(fill)
            except Exception as e:
                logger.error(f"Error in post-trade reconciliation: {e}")

    def _persist_fill(self, fill: Fill) -> None:
        """
        Persist a fill to disk.

        Args:
            fill: Fill to persist
        """
        if not self._persistence_enabled:
            return

        try:
            # Create file path
            file_path = self._persistence_dir / f"{fill.fill_id}.json"

            # Write fill to file
            with open(file_path, 'w') as f:
                json.dump(fill.to_dict(), f, indent=2)

        except Exception as e:
            logger.error(f"Error persisting fill {fill.fill_id}: {e}")

    def get_fill(self, fill_id: str) -> Optional[Fill]:
        """
        Get a fill by ID.

        Args:
            fill_id: Fill ID

        Returns:
            Fill or None if not found
        """
        with self._lock:
            return self._fills.get(fill_id)

    def get_fills_for_order(self, order_id: str) -> List[Fill]:
        """
        Get all fills for an order.

        Args:
            order_id: Order ID

        Returns:
            List of fills for the order
        """
        with self._lock:
            fill_ids = self._order_fills.get(order_id, [])
            return [self._fills[fill_id] for fill_id in fill_ids if fill_id in self._fills]

    def get_fills_for_instrument(self, instrument: str,
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None) -> List[Fill]:
        """
        Get fills for an instrument in a time range.

        Args:
            instrument: Instrument symbol
            start_time: Start of time range
            end_time: End of time range

        Returns:
            List of fills matching criteria
        """
        with self._lock:
            fills = []

            for fill in self._fills.values():
                if fill.instrument != instrument:
                    continue

                if start_time and fill.timestamp < start_time:
                    continue

                if end_time and fill.timestamp > end_time:
                    continue

                fills.append(fill)

            # Sort by timestamp
            return sorted(fills, key=lambda f: f.timestamp)

    def calculate_average_price(self, order_id: str) -> Optional[float]:
        """
        Calculate the volume-weighted average price for an order.

        Args:
            order_id: Order ID

        Returns:
            VWAP or None if no fills
        """
        fills = self.get_fills_for_order(order_id)

        if not fills:
            return None

        total_value = sum(float(fill.quantity) * float(fill.price) for fill in fills)
        total_quantity = sum(float(fill.quantity) for fill in fills)

        if total_quantity == 0:
            return None

        return total_value / total_quantity

    def calculate_total_fees(self, order_id: str) -> float:
        """
        Calculate total fees for an order.

        Args:
            order_id: Order ID

        Returns:
            Total fees
        """
        fills = self.get_fills_for_order(order_id)
        return sum(float(fill.fees) for fill in fills)

    def reconcile_fills(self, instrument: str,
                      start_time: datetime,
                      end_time: datetime) -> Dict[str, Any]:
        """
        Reconcile fills with exchange records for a given time period.

        Args:
            instrument: Instrument symbol
            start_time: Start of time range
            end_time: End of time range

        Returns:
            Reconciliation results
        """
        # This would typically query the exchange for fills and compare
        # with local records to ensure no discrepancies
        # For now, return a placeholder
        return {
            "instrument": instrument,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "local_fills": len(self.get_fills_for_instrument(instrument, start_time, end_time)),
            "exchange_fills": 0,
            "discrepancies": []
        }

    def load_persisted_fills(self) -> int:
        """
        Load persisted fills from disk.

        Returns:
            Number of fills loaded
        """
        if not self._persistence_enabled:
            return 0

        count = 0

        try:
            # Get all JSON files in persistence directory
            for file_path in self._persistence_dir.glob("*.json"):
                try:
                    # Load fill from file
                    with open(file_path, 'r') as f:
                        fill_data = json.load(f)

                    # Create Fill object
                    fill = Fill.from_dict(fill_data)

                    # Process fill (without re-persisting)
                    with self._lock:
                        # Store fill
                        self._fills[fill.fill_id] = fill

                        # Update order fills index
                        if fill.order_id not in self._order_fills:
                            self._order_fills[fill.order_id] = []
                        if fill.fill_id not in self._order_fills[fill.order_id]:
                            self._order_fills[fill.order_id].append(fill.fill_id)

                        # Update state manager
                        self.state_manager.set(f"fills.{fill.fill_id}", fill.to_dict())

                    count += 1

                except Exception as e:
                    logger.error(f"Error loading fill from {file_path}: {e}")

            logger.info(f"Loaded {count} persisted fills")

        except Exception as e:
            logger.error(f"Error loading persisted fills: {e}")

        return count