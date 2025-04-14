"""
post_trade_reconciliation.py - Trade verification and reconciliation

This module provides post-trade verification and reconciliation capabilities,
ensuring that trades have been executed correctly and that the system's internal
state matches the exchange's state. It identifies and resolves discrepancies
to maintain accurate position tracking and performance measurement.

The module implements various reconciliation checks and can trigger corrective
actions when discrepancies are detected.
"""

import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass

from core.event_bus import EventTopics, create_event, get_event_bus
from execution.order.order import Order, OrderStatus, Fill

# Configure logger
logger = logging.getLogger(__name__)


class ReconciliationStatus(Enum):
    """Status of a reconciliation check"""
    SUCCESS = "success"  # Everything matches
    WARNING = "warning"  # Minor discrepancies
    ERROR = "error"  # Major discrepancies
    UNVERIFIED = "unverified"  # Could not verify


@dataclass
class ReconciliationResult:
    """Result of a reconciliation check"""
    status: ReconciliationStatus
    message: str
    check_name: str
    details: Optional[Dict[str, Any]] = None
    corrective_actions: Optional[List[Dict[str, Any]]] = None


class PostTradeReconciliation:
    """
    Performs post-trade reconciliation to verify that trades have been executed
    correctly and that internal state matches exchange state.

    Responsibilities:
    - Verify executed trades against exchange reports
    - Reconcile internal state with exchange state
    - Identify and resolve discrepancies
    - Maintain audit trail of reconciliation activities
    - Trigger corrective actions when needed
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the post-trade reconciliation module.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._event_bus = get_event_bus()

        # Reconciliation history
        self._reconciliation_history = []

        # Configurable thresholds
        self.thresholds = {
            "price_tolerance": self.config.get("price_tolerance", 0.001),  # 0.1%
            "quantity_tolerance": self.config.get("quantity_tolerance", 0.0),  # No tolerance by default
            "reconciliation_interval_minutes": self.config.get("reconciliation_interval_minutes", 60),
            "max_retry_count": self.config.get("max_retry_count", 3),
            "retry_interval_seconds": self.config.get("retry_interval_seconds", 30),
        }

        # Exchange-specific thresholds
        self.exchange_thresholds = self.config.get("exchange_thresholds", {})

        # Last reconciliation time
        self.last_reconciliation_time = None

        logger.info("PostTradeReconciliation initialized")

    def reconcile_trade(self, order: Order, exchange_gateway) -> ReconciliationResult:
        """
        Reconcile a single trade against exchange data.

        Args:
            order: Executed order to reconcile
            exchange_gateway: Exchange gateway to query

        Returns:
            ReconciliationResult
        """
        # Skip if order not filled or partially filled
        if order.status not in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
            return ReconciliationResult(
                status=ReconciliationStatus.UNVERIFIED,
                message=f"Order not filled, status: {order.status.value}",
                check_name="trade_reconciliation"
            )

        try:
            # Get exchange order details
            exchange_order = exchange_gateway.get_order_details(order.order_id, order.exchange_id)

            if not exchange_order:
                return ReconciliationResult(
                    status=ReconciliationStatus.ERROR,
                    message=f"Order {order.order_id} not found on exchange",
                    check_name="trade_reconciliation",
                    details={"order_id": order.order_id}
                )

            # Check if statuses match
            if exchange_order.status != order.status:
                return ReconciliationResult(
                    status=ReconciliationStatus.ERROR,
                    message=f"Status mismatch: internal={order.status.value}, exchange={exchange_order.status.value}",
                    check_name="trade_reconciliation",
                    details={
                        "order_id": order.order_id,
                        "internal_status": order.status.value,
                        "exchange_status": exchange_order.status.value
                    },
                    corrective_actions=[
                        {"action": "update_order_status", "order_id": order.order_id, "status": exchange_order.status.value}
                    ]
                )

            # Check filled quantity
            quantity_diff = abs(exchange_order.filled_quantity - order.filled_quantity)
            quantity_tolerance = self.thresholds["quantity_tolerance"]

            # Check if exchange-specific tolerance exists
            if order.exchange_id in self.exchange_thresholds:
                exchange_tolerance = self.exchange_thresholds[order.exchange_id].get("quantity_tolerance")
                if exchange_tolerance is not None:
                    quantity_tolerance = exchange_tolerance

            if quantity_diff > quantity_tolerance:
                return ReconciliationResult(
                    status=ReconciliationStatus.ERROR,
                    message=f"Quantity mismatch: internal={order.filled_quantity}, exchange={exchange_order.filled_quantity}",
                    check_name="trade_reconciliation",
                    details={
                        "order_id": order.order_id,
                        "internal_quantity": order.filled_quantity,
                        "exchange_quantity": exchange_order.filled_quantity,
                        "difference": quantity_diff
                    },
                    corrective_actions=[
                        {"action": "update_filled_quantity", "order_id": order.order_id, "quantity": exchange_order.filled_quantity}
                    ]
                )

            # Check average price if filled
            if order.filled_quantity > 0 and exchange_order.filled_quantity > 0:
                price_diff_pct = abs(exchange_order.average_price - order.average_price) / order.average_price if order.average_price > 0 else 0
                price_tolerance = self.thresholds["price_tolerance"]

                # Check if exchange-specific tolerance exists
                if order.exchange_id in self.exchange_thresholds:
                    exchange_tolerance = self.exchange_thresholds[order.exchange_id].get("price_tolerance")
                    if exchange_tolerance is not None:
                        price_tolerance = exchange_tolerance

                if price_diff_pct > price_tolerance:
                    return ReconciliationResult(
                        status=ReconciliationStatus.ERROR,
                        message=f"Price mismatch: internal={order.average_price}, exchange={exchange_order.average_price}",
                        check_name="trade_reconciliation",
                        details={
                            "order_id": order.order_id,
                            "internal_price": order.average_price,
                            "exchange_price": exchange_order.average_price,
                            "difference_pct": price_diff_pct
                        },
                        corrective_actions=[
                            {"action": "update_average_price", "order_id": order.order_id, "price": exchange_order.average_price}
                        ]
                    )

                # Additional check for execution time if available
                if hasattr(exchange_order, 'execution_time') and hasattr(order, 'execution_time'):
                    if exchange_order.execution_time and order.execution_time:
                        time_diff = abs((exchange_order.execution_time - order.execution_time).total_seconds())
                        if time_diff > 60:  # More than a minute difference
                            return ReconciliationResult(
                                status=ReconciliationStatus.WARNING,
                                message=f"Execution time mismatch: internal={order.execution_time}, exchange={exchange_order.execution_time}",
                                check_name="trade_reconciliation",
                                details={
                                    "order_id": order.order_id,
                                    "internal_time": order.execution_time.isoformat(),
                                    "exchange_time": exchange_order.execution_time.isoformat(),
                                    "difference_seconds": time_diff
                                },
                                corrective_actions=[
                                    {"action": "update_execution_time", "order_id": order.order_id, "time": exchange_order.execution_time.isoformat()}
                                ]
                            )

                # All checks passed
                return ReconciliationResult(
                    status=ReconciliationStatus.SUCCESS,
                    message=f"Trade successfully reconciled: {order.order_id}",
                    check_name="trade_reconciliation",
                    details={
                        "order_id": order.order_id,
                        "filled_quantity": order.filled_quantity,
                        "average_price": order.average_price,
                        "symbol": order.symbol
                    }
                )

        except Exception as e:
            logger.error(f"Error reconciling trade {order.order_id}: {str(e)}", exc_info=True)
            return ReconciliationResult(
                status=ReconciliationStatus.ERROR,
                message=f"Reconciliation error: {str(e)}",
                check_name="trade_reconciliation",
                details={
                    "order_id": order.order_id,
                    "error": str(e)
                }
            )

    def reconcile_positions(self, positions: Dict[str, Dict], exchange_gateway) -> List[ReconciliationResult]:
        """
        Reconcile all positions against exchange data.

        Args:
            positions: Dictionary of internal positions
            exchange_gateway: Exchange gateway to query

        Returns:
            List of ReconciliationResults
        """
        results = []

        try:
            # Get all positions from exchange
            exchange_positions = exchange_gateway.get_all_positions()

            if exchange_positions is None:
                return [ReconciliationResult(
                    status=ReconciliationStatus.ERROR,
                    message="Failed to retrieve positions from exchange",
                    check_name="position_reconciliation"
                )]

            # Check all internal positions against exchange
            for symbol, position in positions.items():
                position_size = position.get('size', 0)

                # Skip zero positions
                if position_size == 0:
                    continue

                # Check if position exists on exchange
                if symbol not in exchange_positions:
                    results.append(ReconciliationResult(
                        status=ReconciliationStatus.ERROR,
                        message=f"Position {symbol} not found on exchange",
                        check_name="position_reconciliation",
                        details={
                            "symbol": symbol,
                            "internal_size": position_size
                        },
                        corrective_actions=[
                            {"action": "reconcile_position", "symbol": symbol, "size": 0}
                        ]
                    ))
                    continue

                # Compare sizes
                exchange_size = exchange_positions[symbol].get('size', 0)
                size_diff = abs(exchange_size - position_size)
                quantity_tolerance = self.thresholds["quantity_tolerance"]

                if size_diff > quantity_tolerance:
                    results.append(ReconciliationResult(
                        status=ReconciliationStatus.ERROR,
                        message=f"Position size mismatch for {symbol}: internal={position_size}, exchange={exchange_size}",
                        check_name="position_reconciliation",
                        details={
                            "symbol": symbol,
                            "internal_size": position_size,
                            "exchange_size": exchange_size,
                            "difference": size_diff
                        },
                        corrective_actions=[
                            {"action": "reconcile_position", "symbol": symbol, "size": exchange_size}
                        ]
                    ))
                else:
                    results.append(ReconciliationResult(
                        status=ReconciliationStatus.SUCCESS,
                        message=f"Position {symbol} successfully reconciled",
                        check_name="position_reconciliation",
                        details={
                            "symbol": symbol,
                            "size": position_size
                        }
                    ))

            # Check for positions on exchange that don't exist internally
            for symbol, position in exchange_positions.items():
                exchange_size = position.get('size', 0)

                # Skip zero positions
                if exchange_size == 0:
                    continue

                # Check if position exists internally
                if symbol not in positions or positions[symbol].get('size', 0) == 0:
                    results.append(ReconciliationResult(
                        status=ReconciliationStatus.ERROR,
                        message=f"Position {symbol} exists on exchange but not internally",
                        check_name="position_reconciliation",
                        details={
                            "symbol": symbol,
                            "exchange_size": exchange_size
                        },
                        corrective_actions=[
                            {"action": "add_position", "symbol": symbol, "size": exchange_size}
                        ]
                    ))

            return results

        except Exception as e:
            logger.error(f"Error reconciling positions: {str(e)}", exc_info=True)
            return [ReconciliationResult(
                status=ReconciliationStatus.ERROR,
                message=f"Position reconciliation error: {str(e)}",
                check_name="position_reconciliation",
                details={
                    "error": str(e)
                }
            )]

    def apply_corrective_actions(self, result: ReconciliationResult, order_book, position_manager=None) -> bool:
        """
        Apply corrective actions from reconciliation results.

        Args:
            result: ReconciliationResult with corrective actions
            order_book: OrderBook for order updates
            position_manager: PositionManager for position updates (optional)

        Returns:
            True if actions were applied successfully, False otherwise
        """
        if not result.corrective_actions:
            return True

        try:
            for action in result.corrective_actions:
                action_type = action.get("action")

                if action_type == "update_order_status":
                    order_id = action.get("order_id")
                    status = action.get("status")

                    order = order_book.get_order(order_id)
                    if order:
                        order.status = status
                        order_book.update_order(order)
                        logger.info(f"Updated order {order_id} status to {status}")

                elif action_type == "update_filled_quantity":
                    order_id = action.get("order_id")
                    quantity = action.get("quantity")

                    order = order_book.get_order(order_id)
                    if order:
                        order.filled_quantity = quantity
                        order_book.update_order(order)
                        logger.info(f"Updated order {order_id} filled quantity to {quantity}")

                elif action_type == "update_average_price":
                    order_id = action.get("order_id")
                    price = action.get("price")

                    order = order_book.get_order(order_id)
                    if order:
                        order.average_price = price
                        order_book.update_order(order)
                        logger.info(f"Updated order {order_id} average price to {price}")

                elif action_type == "update_execution_time":
                    order_id = action.get("order_id")
                    time_str = action.get("time")

                    order = order_book.get_order(order_id)
                    if order:
                        order.execution_time = datetime.fromisoformat(time_str)
                        order_book.update_order(order)
                        logger.info(f"Updated order {order_id} execution time to {time_str}")

                elif action_type in ["reconcile_position", "add_position"]:
                    if not position_manager:
                        logger.warning(f"Cannot apply position action without position manager: {action}")
                        continue

                    symbol = action.get("symbol")
                    size = action.get("size")

                    position_manager.set_position(symbol, size)
                    logger.info(f"Updated position for {symbol} to {size}")

                else:
                    logger.warning(f"Unknown corrective action: {action_type}")

            return True

        except Exception as e:
            logger.error(f"Error applying corrective actions: {str(e)}", exc_info=True)
            return False

    def run_scheduled_reconciliation(self, order_book, position_manager, exchange_gateway) -> Dict[str, Any]:
        """
        Run a scheduled reconciliation of orders and positions.

        Args:
            order_book: OrderBook for orders
            position_manager: PositionManager for positions
            exchange_gateway: Exchange gateway for querying exchange data

        Returns:
            Dictionary with reconciliation results
        """
        now = datetime.now()

        # Skip if we've run recently
        if self.last_reconciliation_time and (now - self.last_reconciliation_time).total_seconds() < self.thresholds["reconciliation_interval_minutes"] * 60:
            return {
                "status": "skipped",
                "reason": "Too soon since last reconciliation"
            }

        self.last_reconciliation_time = now

        # Get active orders from order book
        active_orders = order_book.get_active_orders()
        recent_completed_orders = order_book.get_recently_completed_orders(
            timedelta(hours=24)  # Reconcile completed orders from last 24 hours
        )

        # Reconcile orders
        order_results = []

        for order in active_orders + recent_completed_orders:
            result = self.reconcile_trade(order, exchange_gateway)
            order_results.append(result)

            # Apply corrective actions if needed
            if result.status in [ReconciliationStatus.ERROR, ReconciliationStatus.WARNING] and result.corrective_actions:
                self.apply_corrective_actions(result, order_book, position_manager)

        # Reconcile positions
        positions = position_manager.get_all_positions() if position_manager else {}
        position_results = self.reconcile_positions(positions, exchange_gateway)

        # Apply position corrective actions
        for result in position_results:
            if result.status == ReconciliationStatus.ERROR and result.corrective_actions:
                self.apply_corrective_actions(result, order_book, position_manager)

        # Generate summary
        order_success = sum(1 for r in order_results if r.status == ReconciliationStatus.SUCCESS)
        order_warnings = sum(1 for r in order_results if r.status == ReconciliationStatus.WARNING)
        order_errors = sum(1 for r in order_results if r.status == ReconciliationStatus.ERROR)
        order_unverified = sum(1 for r in order_results if r.status == ReconciliationStatus.UNVERIFIED)

        position_success = sum(1 for r in position_results if r.status == ReconciliationStatus.SUCCESS)
        position_warnings = sum(1 for r in position_results if r.status == ReconciliationStatus.WARNING)
        position_errors = sum(1 for r in position_results if r.status == ReconciliationStatus.ERROR)

        # Log summary
        logger.info(f"Reconciliation completed: Orders - {order_success} success, {order_warnings} warnings, "
                   f"{order_errors} errors, {order_unverified} unverified; Positions - {position_success} success, "
                   f"{position_warnings} warnings, {position_errors} errors")

        # Create reconciliation report
        report = {
            "timestamp": now.isoformat(),
            "orders": {
                "total": len(order_results),
                "success": order_success,
                "warnings": order_warnings,
                "errors": order_errors,
                "unverified": order_unverified,
                "details": [
                    {
                        "order_id": r.details.get("order_id") if r.details else None,
                        "status": r.status.value,
                        "message": r.message
                    }
                    for r in order_results
                ]
            },
            "positions": {
                "total": len(position_results),
                "success": position_success,
                "warnings": position_warnings,
                "errors": position_errors,
                "details": [
                    {
                        "symbol": r.details.get("symbol") if r.details else None,
                        "status": r.status.value,
                        "message": r.message
                    }
                    for r in position_results
                ]
            }
        }

        # Publish reconciliation event
        event = create_event(
            EventTopics.RECONCILIATION_COMPLETED,
            {
                "timestamp": now.timestamp(),
                "order_success": order_success,
                "order_warnings": order_warnings,
                "order_errors": order_errors,
                "position_success": position_success,
                "position_warnings": position_warnings,
                "position_errors": position_errors
            }
        )
        self._event_bus.publish(event)

        return report

    def check_fills_against_exchange(self, order: Order, exchange_gateway) -> ReconciliationResult:
        """
        Check order fills against exchange execution reports.

        Args:
            order: Order to check
            exchange_gateway: Exchange gateway for API access

        Returns:
            ReconciliationResult
        """
        try:
            # Get fills from exchange
            exchange_fills = exchange_gateway.get_fills_for_order(order.order_id)

            if not exchange_fills:
                return ReconciliationResult(
                    status=ReconciliationStatus.UNVERIFIED,
                    message=f"No fills found on exchange for order {order.order_id}",
                    check_name="fill_reconciliation"
                )

            # Calculate total filled from exchange fills
            exchange_filled_quantity = sum(fill.get("quantity", 0) for fill in exchange_fills)
            exchange_total_value = sum(fill.get("quantity", 0) * fill.get("price", 0) for fill in exchange_fills)

            # Calculate average price from exchange fills
            exchange_avg_price = exchange_total_value / exchange_filled_quantity if exchange_filled_quantity > 0 else 0

            # Compare with internal order
            quantity_diff = abs(exchange_filled_quantity - order.filled_quantity)
            quantity_tolerance = self.thresholds["quantity_tolerance"]

            # Check for fill quantity mismatch
            if quantity_diff > quantity_tolerance:
                return ReconciliationResult(
                    status=ReconciliationStatus.ERROR,
                    message=f"Fill quantity mismatch: internal={order.filled_quantity}, exchange={exchange_filled_quantity}",
                    check_name="fill_reconciliation",
                    details={
                        "order_id": order.order_id,
                        "internal_quantity": order.filled_quantity,
                        "exchange_quantity": exchange_filled_quantity,
                        "difference": quantity_diff
                    },
                    corrective_actions=[
                        {"action": "update_filled_quantity", "order_id": order.order_id, "quantity": exchange_filled_quantity}
                    ]
                )

            # Check price if filled
            if order.filled_quantity > 0 and exchange_filled_quantity > 0:
                price_diff_pct = abs(exchange_avg_price - order.average_price) / order.average_price if order.average_price > 0 else 0
                price_tolerance = self.thresholds["price_tolerance"]

                # Check if exchange-specific tolerance exists
                if order.exchange_id in self.exchange_thresholds:
                    exchange_tolerance = self.exchange_thresholds[order.exchange_id].get("price_tolerance")
                    if exchange_tolerance is not None:
                        price_tolerance = exchange_tolerance

                if price_diff_pct > price_tolerance:
                    return ReconciliationResult(
                        status=ReconciliationStatus.ERROR,
                        message=f"Fill price mismatch: internal={order.average_price}, exchange={exchange_avg_price}",
                        check_name="fill_reconciliation",
                        details={
                            "order_id": order.order_id,
                            "internal_price": order.average_price,
                            "exchange_price": exchange_avg_price,
                            "difference_pct": price_diff_pct
                        },
                        corrective_actions=[
                            {"action": "update_average_price", "order_id": order.order_id, "price": exchange_avg_price}
                        ]
                    )

            # All checks passed
            return ReconciliationResult(
                status=ReconciliationStatus.SUCCESS,
                message=f"Fills successfully reconciled for order {order.order_id}",
                check_name="fill_reconciliation",
                details={
                    "order_id": order.order_id,
                    "filled_quantity": exchange_filled_quantity,
                    "average_price": exchange_avg_price,
                    "fill_count": len(exchange_fills)
                }
            )

        except Exception as e:
            logger.error(f"Error checking fills for order {order.order_id}: {str(e)}", exc_info=True)
            return ReconciliationResult(
                status=ReconciliationStatus.ERROR,
                message=f"Fill reconciliation error: {str(e)}",
                check_name="fill_reconciliation",
                details={
                    "order_id": order.order_id,
                    "error": str(e)
                }
            )

    def verify_exchange_status(self, exchange_id: str, exchange_gateway) -> bool:
        """
        Verify exchange status and connectivity.

        Args:
            exchange_id: Exchange ID
            exchange_gateway: Exchange gateway for API access

        Returns:
            True if exchange is operational, False otherwise
        """
        try:
            # Check exchange status
            status = exchange_gateway.check_exchange_status(exchange_id)

            if not status.get("operational", False):
                logger.warning(f"Exchange {exchange_id} reported non-operational status: {status}")
                return False

            # Check connectivity by making a simple request
            test_result = exchange_gateway.test_connectivity(exchange_id)

            if not test_result.get("success", False):
                logger.warning(f"Connectivity test failed for exchange {exchange_id}: {test_result}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error verifying exchange status for {exchange_id}: {str(e)}", exc_info=True)
            return False

    def load_reconciliation_thresholds(self, config: Dict[str, Any]) -> None:
        """
        Load reconciliation thresholds from config.

        Args:
            config: Configuration dictionary
        """
        if not config:
            return

        # Update general thresholds
        for key in ["price_tolerance", "quantity_tolerance", "reconciliation_interval_minutes",
                   "max_retry_count", "retry_interval_seconds"]:
            if key in config:
                self.thresholds[key] = config[key]

        # Update exchange-specific thresholds
        if "exchange_thresholds" in config:
            self.exchange_thresholds.update(config["exchange_thresholds"])

        logger.info(f"Loaded reconciliation thresholds: general={self.thresholds}, exchange-specific={self.exchange_thresholds}")

    def reset(self) -> None:
        """Reset reconciliation state."""
        self.last_reconciliation_time = None
        logger.info("Post-trade reconciliation state reset")