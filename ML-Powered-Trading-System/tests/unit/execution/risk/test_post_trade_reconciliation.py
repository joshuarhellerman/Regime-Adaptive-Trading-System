import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

# Import the module to test
from core.event_bus import EventTopics
from execution.order.order import Order, OrderStatus
from execution.exchange.risk.post_trade_reconciliation import (
    PostTradeReconciliation,
    ReconciliationStatus,
    ReconciliationResult
)


class TestPostTradeReconciliation(unittest.TestCase):
    """Test suite for PostTradeReconciliation class"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Configure mock event bus
        self.mock_event_bus = MagicMock()
        self.patcher = patch('execution.exchange.risk.post_trade_reconciliation.get_event_bus',
                            return_value=self.mock_event_bus)
        self.patcher.start()

        # Create PostTradeReconciliation instance
        self.config = {
            "price_tolerance": 0.002,  # 0.2%
            "quantity_tolerance": 0.0001,  # Slight tolerance for floating-point issues
            "reconciliation_interval_minutes": 30,
            "max_retry_count": 2,
            "retry_interval_seconds": 15,
            "exchange_thresholds": {
                "exchange_1": {
                    "price_tolerance": 0.005,  # 0.5% for this exchange
                    "quantity_tolerance": 0.0002
                }
            }
        }
        self.reconciliation = PostTradeReconciliation(config=self.config)

        # Set up mock order, exchange_gateway, order_book, and position_manager
        self.mock_order = self._create_mock_order()
        self.mock_exchange_gateway = self._create_mock_exchange_gateway()
        self.mock_order_book = self._create_mock_order_book()
        self.mock_position_manager = self._create_mock_position_manager()

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        self.patcher.stop()

    def _create_mock_order(self):
        """Create a mock Order for testing."""
        mock_order = Mock(spec=Order)
        mock_order.order_id = "test_order_123"
        mock_order.exchange_id = "exchange_1"
        mock_order.symbol = "BTC-USD"
        mock_order.status = OrderStatus.FILLED
        mock_order.filled_quantity = 1.0
        mock_order.average_price = 50000.0
        mock_order.execution_time = datetime.now()
        return mock_order

    def _create_mock_exchange_gateway(self):
        """Create a mock exchange gateway for testing."""
        mock_gateway = Mock()
        # Setup default behavior here
        mock_gateway.get_order_details.return_value = None
        mock_gateway.get_all_positions.return_value = {}
        mock_gateway.get_fills_for_order.return_value = []
        mock_gateway.check_exchange_status.return_value = {"operational": True}
        mock_gateway.test_connectivity.return_value = {"success": True}
        return mock_gateway

    def _create_mock_order_book(self):
        """Create a mock order book for testing."""
        mock_order_book = Mock()
        mock_order_book.get_order.return_value = self.mock_order
        mock_order_book.get_active_orders.return_value = [self.mock_order]
        mock_order_book.get_recently_completed_orders.return_value = []
        return mock_order_book

    def _create_mock_position_manager(self):
        """Create a mock position manager for testing."""
        mock_pos_manager = Mock()
        mock_pos_manager.get_all_positions.return_value = {
            "BTC-USD": {"size": 1.0, "entry_price": 50000.0},
            "ETH-USD": {"size": 10.0, "entry_price": 3000.0}
        }
        return mock_pos_manager

    def test_init(self):
        """Test initialization with config values."""
        self.assertEqual(self.reconciliation.thresholds["price_tolerance"], 0.002)
        self.assertEqual(self.reconciliation.thresholds["quantity_tolerance"], 0.0001)
        self.assertEqual(self.reconciliation.thresholds["reconciliation_interval_minutes"], 30)
        self.assertEqual(self.reconciliation.exchange_thresholds["exchange_1"]["price_tolerance"], 0.005)

    def test_reconcile_trade_matching(self):
        """Test trade reconciliation when internal and exchange data match."""
        # Setup mock exchange order that matches internal order
        exchange_order = Mock()
        exchange_order.status = OrderStatus.FILLED
        exchange_order.filled_quantity = 1.0
        exchange_order.average_price = 50000.0
        exchange_order.execution_time = self.mock_order.execution_time
        self.mock_exchange_gateway.get_order_details.return_value = exchange_order

        # Run reconciliation
        result = self.reconciliation.reconcile_trade(self.mock_order, self.mock_exchange_gateway)

        # Verify results
        self.assertEqual(result.status, ReconciliationStatus.SUCCESS)
        self.assertTrue("successfully reconciled" in result.message.lower())
        self.assertEqual(result.check_name, "trade_reconciliation")
        self.assertEqual(result.details["order_id"], "test_order_123")
        self.assertIsNone(result.corrective_actions)

    def test_reconcile_trade_status_mismatch(self):
        """Test trade reconciliation when order status doesn't match."""
        # Setup mock exchange order with different status
        exchange_order = Mock()
        exchange_order.status = OrderStatus.PARTIALLY_FILLED
        exchange_order.filled_quantity = 0.5
        exchange_order.average_price = 50000.0
        self.mock_exchange_gateway.get_order_details.return_value = exchange_order

        # Run reconciliation
        result = self.reconciliation.reconcile_trade(self.mock_order, self.mock_exchange_gateway)

        # Verify results
        self.assertEqual(result.status, ReconciliationStatus.ERROR)
        self.assertTrue("status mismatch" in result.message.lower())
        self.assertEqual(result.details["internal_status"], self.mock_order.status.value)
        self.assertEqual(result.details["exchange_status"], exchange_order.status.value)

        # Verify corrective actions
        self.assertIsNotNone(result.corrective_actions)
        action = result.corrective_actions[0]
        self.assertEqual(action["action"], "update_order_status")
        self.assertEqual(action["order_id"], "test_order_123")
        self.assertEqual(action["status"], exchange_order.status.value)

    def test_reconcile_trade_quantity_mismatch(self):
        """Test trade reconciliation when quantities don't match."""
        # Setup mock exchange order with different quantity
        exchange_order = Mock()
        exchange_order.status = OrderStatus.FILLED
        exchange_order.filled_quantity = 1.1  # Different quantity
        exchange_order.average_price = 50000.0
        self.mock_exchange_gateway.get_order_details.return_value = exchange_order

        # Run reconciliation
        result = self.reconciliation.reconcile_trade(self.mock_order, self.mock_exchange_gateway)

        # Verify results
        self.assertEqual(result.status, ReconciliationStatus.ERROR)
        self.assertTrue("quantity mismatch" in result.message.lower())
        self.assertEqual(result.details["internal_quantity"], 1.0)
        self.assertEqual(result.details["exchange_quantity"], 1.1)
        
        # Verify corrective actions
        self.assertIsNotNone(result.corrective_actions)
        action = result.corrective_actions[0]
        self.assertEqual(action["action"], "update_filled_quantity")
        self.assertEqual(action["quantity"], 1.1)

    def test_reconcile_trade_price_mismatch(self):
        """Test trade reconciliation when prices don't match."""
        # Setup mock exchange order with different price
        exchange_order = Mock()
        exchange_order.status = OrderStatus.FILLED
        exchange_order.filled_quantity = 1.0
        exchange_order.average_price = 50500.0  # 1% difference
        self.mock_exchange_gateway.get_order_details.return_value = exchange_order

        # Run reconciliation 
        result = self.reconciliation.reconcile_trade(self.mock_order, self.mock_exchange_gateway)

        # Verify results
        self.assertEqual(result.status, ReconciliationStatus.ERROR)
        self.assertTrue("price mismatch" in result.message.lower())
        self.assertEqual(result.details["internal_price"], 50000.0)
        self.assertEqual(result.details["exchange_price"], 50500.0)

        # Verify corrective actions
        self.assertIsNotNone(result.corrective_actions)
        action = result.corrective_actions[0]
        self.assertEqual(action["action"], "update_average_price")
        self.assertEqual(action["price"], 50500.0)

    def test_reconcile_trade_exchange_specific_tolerance(self):
        """Test exchange-specific tolerance thresholds."""
        # Setup mock exchange order with price difference within exchange-specific tolerance
        exchange_order = Mock()
        exchange_order.status = OrderStatus.FILLED
        exchange_order.filled_quantity = 1.0
        # 0.4% difference, within the 0.5% tolerance for exchange_1
        exchange_order.average_price = 50200.0
        self.mock_exchange_gateway.get_order_details.return_value = exchange_order

        # Run reconciliation
        result = self.reconciliation.reconcile_trade(self.mock_order, self.mock_exchange_gateway)

        # Should be SUCCESS because we're within the exchange-specific tolerance
        self.assertEqual(result.status, ReconciliationStatus.SUCCESS)

    def test_reconcile_trade_execution_time_mismatch(self):
        """Test reconciliation with execution time mismatch."""
        # Setup mock exchange order with different execution time
        exchange_order = Mock()
        exchange_order.status = OrderStatus.FILLED
        exchange_order.filled_quantity = 1.0
        exchange_order.average_price = 50000.0
        exchange_order.execution_time = self.mock_order.execution_time + timedelta(minutes=2)
        self.mock_exchange_gateway.get_order_details.return_value = exchange_order

        # Run reconciliation
        result = self.reconciliation.reconcile_trade(self.mock_order, self.mock_exchange_gateway)

        # Should be a warning since time difference is over a minute
        self.assertEqual(result.status, ReconciliationStatus.WARNING)
        self.assertTrue("execution time mismatch" in result.message.lower())

    def test_reconcile_trade_not_filled(self):
        """Test reconciliation for orders that aren't filled."""
        unfilled_order = Mock(spec=Order)
        unfilled_order.status = OrderStatus.OPEN
        
        result = self.reconciliation.reconcile_trade(unfilled_order, self.mock_exchange_gateway)
        
        self.assertEqual(result.status, ReconciliationStatus.UNVERIFIED)
        self.assertTrue("not filled" in result.message.lower())

    def test_reconcile_trade_order_not_found(self):
        """Test reconciliation when order is not found on exchange."""
        # Setup exchange gateway to return None for order details
        self.mock_exchange_gateway.get_order_details.return_value = None
        
        result = self.reconciliation.reconcile_trade(self.mock_order, self.mock_exchange_gateway)
        
        self.assertEqual(result.status, ReconciliationStatus.ERROR)
        self.assertTrue("not found on exchange" in result.message.lower())

    def test_reconcile_trade_error(self):
        """Test reconciliation error handling."""
        # Setup exchange gateway to raise an exception
        self.mock_exchange_gateway.get_order_details.side_effect = Exception("API error")
        
        result = self.reconciliation.reconcile_trade(self.mock_order, self.mock_exchange_gateway)
        
        self.assertEqual(result.status, ReconciliationStatus.ERROR)
        self.assertTrue("reconciliation error" in result.message.lower())
        self.assertEqual(result.details["error"], "API error")

    def test_reconcile_positions_matching(self):
        """Test position reconciliation when positions match."""
        # Setup exchange positions matching internal positions
        exchange_positions = {
            "BTC-USD": {"size": 1.0, "mark_price": 50100.0},
            "ETH-USD": {"size": 10.0, "mark_price": 3050.0}
        }
        self.mock_exchange_gateway.get_all_positions.return_value = exchange_positions
        
        results = self.reconciliation.reconcile_positions(
            self.mock_position_manager.get_all_positions(), 
            self.mock_exchange_gateway
        )
        
        # Should have 2 successful reconciliations
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertEqual(result.status, ReconciliationStatus.SUCCESS)
            self.assertTrue("successfully reconciled" in result.message.lower())

    def test_reconcile_positions_size_mismatch(self):
        """Test position reconciliation when sizes don't match."""
        # Setup exchange positions with different size for BTC
        exchange_positions = {
            "BTC-USD": {"size": 1.2, "mark_price": 50100.0},  # Different size
            "ETH-USD": {"size": 10.0, "mark_price": 3050.0}
        }
        self.mock_exchange_gateway.get_all_positions.return_value = exchange_positions
        
        results = self.reconciliation.reconcile_positions(
            self.mock_position_manager.get_all_positions(), 
            self.mock_exchange_gateway
        )
        
        # Find the BTC result
        btc_result = next(r for r in results if r.details and r.details.get("symbol") == "BTC-USD")
        
        self.assertEqual(btc_result.status, ReconciliationStatus.ERROR)
        self.assertTrue("position size mismatch" in btc_result.message.lower())
        self.assertEqual(btc_result.details["internal_size"], 1.0)
        self.assertEqual(btc_result.details["exchange_size"], 1.2)

    def test_reconcile_positions_missing_internally(self):
        """Test reconciliation when position exists on exchange but not internally."""
        # Setup exchange positions with an extra position
        exchange_positions = {
            "BTC-USD": {"size": 1.0, "mark_price": 50100.0},
            "ETH-USD": {"size": 10.0, "mark_price": 3050.0},
            "SOL-USD": {"size": 50.0, "mark_price": 100.0}  # Extra position
        }
        self.mock_exchange_gateway.get_all_positions.return_value = exchange_positions
        
        results = self.reconciliation.reconcile_positions(
            self.mock_position_manager.get_all_positions(), 
            self.mock_exchange_gateway
        )
        
        # Find the SOL result
        sol_result = next(r for r in results if r.details and r.details.get("symbol") == "SOL-USD")
        
        self.assertEqual(sol_result.status, ReconciliationStatus.ERROR)
        self.assertTrue("exists on exchange but not internally" in sol_result.message.lower())
        self.assertEqual(sol_result.details["exchange_size"], 50.0)

    def test_reconcile_positions_missing_on_exchange(self):
        """Test reconciliation when position exists internally but not on exchange."""
        # Setup exchange positions missing ETH
        exchange_positions = {
            "BTC-USD": {"size": 1.0, "mark_price": 50100.0}
            # ETH is missing
        }
        self.mock_exchange_gateway.get_all_positions.return_value = exchange_positions
        
        results = self.reconciliation.reconcile_positions(
            self.mock_position_manager.get_all_positions(), 
            self.mock_exchange_gateway
        )
        
        # Find the ETH result
        eth_result = next(r for r in results if r.details and r.details.get("symbol") == "ETH-USD")
        
        self.assertEqual(eth_result.status, ReconciliationStatus.ERROR)
        self.assertTrue("not found on exchange" in eth_result.message.lower())
        self.assertEqual(eth_result.details["internal_size"], 10.0)

    def test_apply_corrective_actions(self):
        """Test applying corrective actions."""
        # Create a reconciliation result with corrective actions
        result = ReconciliationResult(
            status=ReconciliationStatus.ERROR,
            message="Test error",
            check_name="test_check",
            details={"order_id": "test_order_123"},
            corrective_actions=[
                {"action": "update_order_status", "order_id": "test_order_123", "status": "partially_filled"}
            ]
        )
        
        # Apply the corrective actions
        success = self.reconciliation.apply_corrective_actions(
            result, self.mock_order_book, self.mock_position_manager
        )
        
        # Verify order book was updated
        self.assertTrue(success)
        self.mock_order_book.update_order.assert_called_once()

    def test_apply_corrective_actions_position(self):
        """Test applying position corrective actions."""
        # Create a reconciliation result with position corrective actions
        result = ReconciliationResult(
            status=ReconciliationStatus.ERROR,
            message="Position error",
            check_name="position_reconciliation",
            details={"symbol": "BTC-USD"},
            corrective_actions=[
                {"action": "reconcile_position", "symbol": "BTC-USD", "size": 1.2}
            ]
        )
        
        # Apply the corrective actions
        success = self.reconciliation.apply_corrective_actions(
            result, self.mock_order_book, self.mock_position_manager
        )
        
        # Verify position manager was updated
        self.assertTrue(success)
        self.mock_position_manager.set_position.assert_called_once_with("BTC-USD", 1.2)

    def test_run_scheduled_reconciliation(self):
        """Test running scheduled reconciliation."""
        # Setup mock objects for full reconciliation
        exchange_order = Mock()
        exchange_order.status = OrderStatus.FILLED
        exchange_order.filled_quantity = 1.0
        exchange_order.average_price = 50000.0
        exchange_order.execution_time = self.mock_order.execution_time
        self.mock_exchange_gateway.get_order_details.return_value = exchange_order
        
        exchange_positions = {
            "BTC-USD": {"size": 1.0, "mark_price": 50100.0},
            "ETH-USD": {"size": 10.0, "mark_price": 3050.0}
        }
        self.mock_exchange_gateway.get_all_positions.return_value = exchange_positions
        
        # Run reconciliation
        report = self.reconciliation.run_scheduled_reconciliation(
            self.mock_order_book, self.mock_position_manager, self.mock_exchange_gateway
        )
        
        # Verify results
        self.assertIn("timestamp", report)
        self.assertEqual(report["orders"]["success"], 1)  # One order successfully reconciled
        self.assertEqual(report["positions"]["success"], 2)  # Two positions successfully reconciled
        
        # Check that event was published
        self.mock_event_bus.publish.assert_called_once()
        event = self.mock_event_bus.publish.call_args[0][0]
        self.assertEqual(event.topic, EventTopics.RECONCILIATION_COMPLETED)

    def test_run_scheduled_reconciliation_too_soon(self):
        """Test that reconciliation is skipped if run too soon."""
        # Set last reconciliation time to recent time
        self.reconciliation.last_reconciliation_time = datetime.now() - timedelta(minutes=10)
        
        # Run reconciliation
        report = self.reconciliation.run_scheduled_reconciliation(
            self.mock_order_book, self.mock_position_manager, self.mock_exchange_gateway
        )
        
        # Verify it was skipped
        self.assertEqual(report["status"], "skipped")
        self.mock_exchange_gateway.get_order_details.assert_not_called()
        self.mock_event_bus.publish.assert_not_called()

    def test_check_fills_against_exchange_matching(self):
        """Test fill checking when fills match."""
        # Setup exchange fills
        exchange_fills = [
            {"quantity": 0.5, "price": 50000.0, "timestamp": "2023-01-01T10:00:00Z"},
            {"quantity": 0.5, "price": 50000.0, "timestamp": "2023-01-01T10:00:01Z"}
        ]
        self.mock_exchange_gateway.get_fills_for_order.return_value = exchange_fills
        
        # Run check
        result = self.reconciliation.check_fills_against_exchange(
            self.mock_order, self.mock_exchange_gateway
        )
        
        # Verify results
        self.assertEqual(result.status, ReconciliationStatus.SUCCESS)
        self.assertTrue("successfully reconciled" in result.message.lower())
        self.assertEqual(result.details["fill_count"], 2)

    def test_check_fills_against_exchange_quantity_mismatch(self):
        """Test fill checking when quantities don't match."""
        # Setup exchange fills with different total quantity
        exchange_fills = [
            {"quantity": 0.6, "price": 50000.0, "timestamp": "2023-01-01T10:00:00Z"},
            {"quantity": 0.5, "price": 50000.0, "timestamp": "2023-01-01T10:00:01Z"}
        ]
        self.mock_exchange_gateway.get_fills_for_order.return_value = exchange_fills
        
        # Run check
        result = self.reconciliation.check_fills_against_exchange(
            self.mock_order, self.mock_exchange_gateway
        )
        
        # Verify results
        self.assertEqual(result.status, ReconciliationStatus.ERROR)
        self.assertTrue("fill quantity mismatch" in result.message.lower())
        self.assertEqual(result.details["internal_quantity"], 1.0)
        self.assertEqual(result.details["exchange_quantity"], 1.1)

    def test_verify_exchange_status_operational(self):
        """Test exchange status verification when operational."""
        # Run verification
        result = self.reconciliation.verify_exchange_status(
            "exchange_1", self.mock_exchange_gateway
        )
        
        # Verify result
        self.assertTrue(result)
        self.mock_exchange_gateway.check_exchange_status.assert_called_once_with("exchange_1")
        self.mock_exchange_gateway.test_connectivity.assert_called_once_with("exchange_1")

    def test_verify_exchange_status_non_operational(self):
        """Test exchange status verification when not operational."""
        # Setup mock to return non-operational status
        self.mock_exchange_gateway.check_exchange_status.return_value = {"operational": False}
        
        # Run verification
        result = self.reconciliation.verify_exchange_status(
            "exchange_1", self.mock_exchange_gateway
        )
        
        # Verify result
        self.assertFalse(result)
        self.mock_exchange_gateway.test_connectivity.assert_not_called()  # Should short-circuit

    def test_load_reconciliation_thresholds(self):
        """Test loading reconciliation thresholds from config."""
        # Create new config
        new_config = {
            "price_tolerance": 0.01,
            "quantity_tolerance": 0.005,
            "exchange_thresholds": {
                "exchange_2": {
                    "price_tolerance": 0.02
                }
            }
        }
        
        # Load thresholds
        self.reconciliation.load_reconciliation_thresholds(new_config)
        
        # Verify thresholds were updated
        self.assertEqual(self.reconciliation.thresholds["price_tolerance"], 0.01)
        self.assertEqual(self.reconciliation.thresholds["quantity_tolerance"], 0.005)
        self.assertEqual(self.reconciliation.exchange_thresholds["exchange_2"]["price_tolerance"], 0.02)
        
        # Original values should be preserved
        self.assertEqual(self.reconciliation.thresholds["reconciliation_interval_minutes"], 30)
        self.assertEqual(self.reconciliation.exchange_thresholds["exchange_1"]["price_tolerance"], 0.005)

    def test_reset(self):
        """Test resetting reconciliation state."""
        # Set last reconciliation time
        self.reconciliation.last_reconciliation_time = datetime.now()
        
        # Reset
        self.reconciliation.reset()
        
        # Verify state was reset
        self.assertIsNone(self.reconciliation.last_reconciliation_time)


if __name__ == "__main__":
    unittest.main()