"""
Unit tests for the FillService class.

This module tests the functionality of the FillService, including fill processing,
order updates, persistence, and reconciliation.
"""
import unittest
from unittest.mock import MagicMock, patch, call
import tempfile
import json
import os
from pathlib import Path
from datetime import datetime, timezone

from core.event_bus import EventBus, Event, EventPriority
from core.state_manager import StateManager
from execution.order.order import Order, OrderStatus
from execution.order.order_book import OrderBook
from execution.fill.fill_model import Fill
from execution.risk.position_reconciliation import PositionReconciliation
from execution.risk.post_trade_reconciliation import PostTradeReconciliation
from execution.fill.fill_service import FillService

class TestFillService(unittest.TestCase):
    """Test cases for FillService."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mocks for dependencies
        self.event_bus = MagicMock(spec=EventBus)
        self.state_manager = MagicMock(spec=StateManager)
        self.order_book = MagicMock(spec=OrderBook)
        self.position_reconciliation = MagicMock(spec=PositionReconciliation)
        self.post_trade_reconciliation = MagicMock(spec=PostTradeReconciliation)

        # Create a temporary directory for fill persistence
        self.temp_dir = tempfile.TemporaryDirectory()
        self.persistence_dir = Path(self.temp_dir.name)

        # Create configuration
        self.config = {
            'persistence_enabled': True,
            'fill_persistence_dir': str(self.persistence_dir)
        }

        # Create FillService instance
        self.fill_service = FillService(
            event_bus=self.event_bus,
            state_manager=self.state_manager,
            order_book=self.order_book,
            position_reconciliation=self.position_reconciliation,
            post_trade_reconciliation=self.post_trade_reconciliation,
            config=self.config
        )

        # Sample order and fill data
        self.sample_order = Order(
            order_id="order-123",
            instrument="BTC-USD",
            quantity=1.0,
            price=50000.0,
            side="buy",
            order_type="limit",
            status=OrderStatus.PENDING
        )

        self.sample_fill = Fill(
            fill_id="fill-123",
            order_id="order-123",
            instrument="BTC-USD",
            quantity=0.5,
            price=50000.0,
            side="buy",
            timestamp=datetime.now(timezone.utc),
            fees=10.0,
            exchange="binance"
        )

    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test FillService initialization."""
        # Check event subscriptions
        self.event_bus.subscribe.assert_any_call("exchange.fill.*", self.fill_service.process_fill)
        self.event_bus.subscribe.assert_any_call("paper.order.filled", self.fill_service.process_fill)
        self.event_bus.subscribe.assert_any_call("simulation.fill", self.fill_service.process_fill)

        # Check state registration
        self.state_manager.set_scope.assert_called_once_with("fills", any)

    def test_process_fill(self):
        """Test processing a fill event."""
        # Set up test
        fill_event = MagicMock(spec=Event)
        fill_event.data = self.sample_fill.to_dict()

        self.order_book.get_order.return_value = self.sample_order

        # Mock transaction context manager
        transaction_context = MagicMock()
        self.state_manager.transaction.return_value = transaction_context
        transaction_context.__enter__ = MagicMock()
        transaction_context.__exit__ = MagicMock()

        # Call method
        self.fill_service.process_fill(fill_event)

        # Check that order was retrieved
        self.order_book.get_order.assert_called_once_with("order-123")

        # Check fill was stored
        fill = self.fill_service.get_fill("fill-123")
        self.assertIsNotNone(fill)
        self.assertEqual(fill.fill_id, "fill-123")

        # Check state was updated
        self.state_manager.set.assert_called_once_with(
            "fills.fill-123",
            self.sample_fill.to_dict()
        )

        # Check order was updated
        self.order_book.update_order.assert_called_once()
        args = self.order_book.update_order.call_args[0]
        updated_order = args[0]
        self.assertEqual(updated_order.status, OrderStatus.PARTIALLY_FILLED)
        self.assertEqual(updated_order.filled_quantity, 0.5)
        self.assertEqual(updated_order.average_price, 50000.0)

        # Check fill event was published
        self.event_bus.publish.assert_called_once()

        # Check persistence
        expected_file = self.persistence_dir / "fill-123.json"
        self.assertTrue(expected_file.exists())

        # Check reconciliation was triggered
        self.position_reconciliation.reconcile_fill.assert_called_once_with(any)
        self.post_trade_reconciliation.process_fill.assert_called_once_with(any)

    def test_process_fill_complete_order(self):
        """Test processing a fill that completes an order."""
        # Set up test with a fill that matches the entire order quantity
        fill = Fill(
            fill_id="fill-456",
            order_id="order-123",
            instrument="BTC-USD",
            quantity=1.0,  # Full quantity
            price=50000.0,
            side="buy",
            timestamp=datetime.now(timezone.utc),
            fees=10.0,
            exchange="binance"
        )

        fill_event = MagicMock(spec=Event)
        fill_event.data = fill.to_dict()

        self.order_book.get_order.return_value = self.sample_order

        # Mock transaction context manager
        transaction_context = MagicMock()
        self.state_manager.transaction.return_value = transaction_context
        transaction_context.__enter__ = MagicMock()
        transaction_context.__exit__ = MagicMock()

        # Call method
        self.fill_service.process_fill(fill_event)

        # Check order was updated to FILLED status
        self.order_book.update_order.assert_called_once()
        args = self.order_book.update_order.call_args[0]
        updated_order = args[0]
        self.assertEqual(updated_order.status, OrderStatus.FILLED)
        self.assertEqual(updated_order.filled_quantity, 1.0)

    def test_process_fill_missing_order(self):
        """Test processing a fill for a non-existent order."""
        # Set up test with order_book returning None
        fill_event = MagicMock(spec=Event)
        fill_event.data = self.sample_fill.to_dict()

        self.order_book.get_order.return_value = None

        # Mock transaction context manager
        transaction_context = MagicMock()
        self.state_manager.transaction.return_value = transaction_context
        transaction_context.__enter__ = MagicMock()
        transaction_context.__exit__ = MagicMock()

        # Call method
        self.fill_service.process_fill(fill_event)

        # Check fill was stored despite missing order
        fill = self.fill_service.get_fill("fill-123")
        self.assertIsNotNone(fill)

        # Order update should not have been called
        self.order_book.update_order.assert_not_called()

    def test_process_fill_event_object(self):
        """Test processing a fill as an object rather than dict."""
        # Set up test with Fill object directly
        fill_event = MagicMock(spec=Event)
        fill_event.data = self.sample_fill

        self.order_book.get_order.return_value = self.sample_order

        # Mock transaction context manager
        transaction_context = MagicMock()
        self.state_manager.transaction.return_value = transaction_context
        transaction_context.__enter__ = MagicMock()
        transaction_context.__exit__ = MagicMock()

        # Call method
        self.fill_service.process_fill(fill_event)

        # Check fill was stored
        fill = self.fill_service.get_fill("fill-123")
        self.assertIsNotNone(fill)

    def test_process_fill_exception(self):
        """Test handling exceptions during fill processing."""
        # Set up test with exception during processing
        fill_event = MagicMock(spec=Event)
        fill_event.data = self.sample_fill.to_dict()

        self.order_book.get_order.side_effect = Exception("Test exception")

        # Mock transaction context manager
        transaction_context = MagicMock()
        self.state_manager.transaction.return_value = transaction_context
        transaction_context.__enter__ = MagicMock()
        transaction_context.__exit__ = MagicMock()

        # Call method (should not raise exception)
        self.fill_service.process_fill(fill_event)

        # Check order update was not called
        self.order_book.update_order.assert_not_called()

    def test_get_fill(self):
        """Test retrieving a fill by ID."""
        # Setup - add a fill
        self.fill_service._fills["fill-123"] = self.sample_fill

        # Call method
        fill = self.fill_service.get_fill("fill-123")

        # Check result
        self.assertEqual(fill, self.sample_fill)

        # Test non-existent fill
        fill = self.fill_service.get_fill("nonexistent")
        self.assertIsNone(fill)

    def test_get_fills_for_order(self):
        """Test retrieving fills for an order."""
        # Setup - add fills
        fill1 = Fill(
            fill_id="fill-1",
            order_id="order-abc",
            instrument="BTC-USD",
            quantity=0.5,
            price=50000.0,
            side="buy",
            timestamp=datetime.now(timezone.utc),
            fees=10.0,
            exchange="binance"
        )

        fill2 = Fill(
            fill_id="fill-2",
            order_id="order-abc",
            instrument="BTC-USD",
            quantity=0.5,
            price=50100.0,
            side="buy",
            timestamp=datetime.now(timezone.utc),
            fees=10.0,
            exchange="binance"
        )

        self.fill_service._fills["fill-1"] = fill1
        self.fill_service._fills["fill-2"] = fill2
        self.fill_service._order_fills["order-abc"] = ["fill-1", "fill-2"]

        # Call method
        fills = self.fill_service.get_fills_for_order("order-abc")

        # Check result
        self.assertEqual(len(fills), 2)
        self.assertIn(fill1, fills)
        self.assertIn(fill2, fills)

        # Test order with no fills
        fills = self.fill_service.get_fills_for_order("nonexistent")
        self.assertEqual(len(fills), 0)

    def test_get_fills_for_instrument(self):
        """Test retrieving fills for an instrument within a time range."""
        # Setup - add fills
        now = datetime.now(timezone.utc)

        # Create fills with different timestamps
        fill1 = Fill(
            fill_id="fill-1",
            order_id="order-1",
            instrument="BTC-USD",
            quantity=0.5,
            price=50000.0,
            side="buy",
            timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            fees=10.0,
            exchange="binance"
        )

        fill2 = Fill(
            fill_id="fill-2",
            order_id="order-2",
            instrument="BTC-USD",
            quantity=0.5,
            price=51000.0,
            side="buy",
            timestamp=datetime(2023, 2, 1, tzinfo=timezone.utc),
            fees=10.0,
            exchange="binance"
        )

        fill3 = Fill(
            fill_id="fill-3",
            order_id="order-3",
            instrument="ETH-USD",  # Different instrument
            quantity=0.5,
            price=2000.0,
            side="buy",
            timestamp=datetime(2023, 2, 1, tzinfo=timezone.utc),
            fees=10.0,
            exchange="binance"
        )

        self.fill_service._fills = {
            "fill-1": fill1,
            "fill-2": fill2,
            "fill-3": fill3
        }

        # Call method with different time ranges
        # No time range
        fills = self.fill_service.get_fills_for_instrument("BTC-USD")
        self.assertEqual(len(fills), 2)
        self.assertIn(fill1, fills)
        self.assertIn(fill2, fills)

        # With start time
        fills = self.fill_service.get_fills_for_instrument(
            "BTC-USD",
            start_time=datetime(2023, 1, 15, tzinfo=timezone.utc)
        )
        self.assertEqual(len(fills), 1)
        self.assertIn(fill2, fills)

        # With end time
        fills = self.fill_service.get_fills_for_instrument(
            "BTC-USD",
            end_time=datetime(2023, 1, 15, tzinfo=timezone.utc)
        )
        self.assertEqual(len(fills), 1)
        self.assertIn(fill1, fills)

        # With both start and end time
        fills = self.fill_service.get_fills_for_instrument(
            "BTC-USD",
            start_time=datetime(2022, 12, 1, tzinfo=timezone.utc),
            end_time=datetime(2023, 1, 15, tzinfo=timezone.utc)
        )
        self.assertEqual(len(fills), 1)
        self.assertIn(fill1, fills)

        # Test non-existent instrument
        fills = self.fill_service.get_fills_for_instrument("nonexistent")
        self.assertEqual(len(fills), 0)

    def test_calculate_average_price(self):
        """Test calculating volume-weighted average price for an order."""
        # Setup - add fills
        fill1 = Fill(
            fill_id="fill-1",
            order_id="order-abc",
            instrument="BTC-USD",
            quantity=1.0,
            price=50000.0,
            side="buy",
            timestamp=datetime.now(timezone.utc),
            fees=10.0,
            exchange="binance"
        )

        fill2 = Fill(
            fill_id="fill-2",
            order_id="order-abc",
            instrument="BTC-USD",
            quantity=2.0,  # Different quantity
            price=51000.0,  # Different price
            side="buy",
            timestamp=datetime.now(timezone.utc),
            fees=20.0,
            exchange="binance"
        )

        self.fill_service._fills["fill-1"] = fill1
        self.fill_service._fills["fill-2"] = fill2
        self.fill_service._order_fills["order-abc"] = ["fill-1", "fill-2"]

        # Call method
        avg_price = self.fill_service.calculate_average_price("order-abc")

        # Expected VWAP: (1.0*50000 + 2.0*51000) / (1.0 + 2.0) = 50666.67
        expected_vwap = (1.0 * 50000.0 + 2.0 * 51000.0) / 3.0
        self.assertAlmostEqual(avg_price, expected_vwap)

        # Test order with no fills
        avg_price = self.fill_service.calculate_average_price("nonexistent")
        self.assertIsNone(avg_price)

    def test_calculate_total_fees(self):
        """Test calculating total fees for an order."""
        # Setup - add fills
        fill1 = Fill(
            fill_id="fill-1",
            order_id="order-abc",
            instrument="BTC-USD",
            quantity=1.0,
            price=50000.0,
            side="buy",
            timestamp=datetime.now(timezone.utc),
            fees=10.0,
            exchange="binance"
        )

        fill2 = Fill(
            fill_id="fill-2",
            order_id="order-abc",
            instrument="BTC-USD",
            quantity=2.0,
            price=51000.0,
            side="buy",
            timestamp=datetime.now(timezone.utc),
            fees=20.0,
            exchange="binance"
        )

        self.fill_service._fills["fill-1"] = fill1
        self.fill_service._fills["fill-2"] = fill2
        self.fill_service._order_fills["order-abc"] = ["fill-1", "fill-2"]

        # Call method
        total_fees = self.fill_service.calculate_total_fees("order-abc")

        # Expected total fees: 10.0 + 20.0 = 30.0
        self.assertEqual(total_fees, 30.0)

        # Test order with no fills
        total_fees = self.fill_service.calculate_total_fees("nonexistent")
        self.assertEqual(total_fees, 0.0)

    def test_reconcile_fills(self):
        """Test fill reconciliation functionality."""
        # Setup - add fills
        start_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2023, 2, 1, tzinfo=timezone.utc)

        fill1 = Fill(
            fill_id="fill-1",
            order_id="order-1",
            instrument="BTC-USD",
            quantity=1.0,
            price=50000.0,
            side="buy",
            timestamp=datetime(2023, 1, 15, tzinfo=timezone.utc),  # Within time range
            fees=10.0,
            exchange="binance"
        )

        fill2 = Fill(
            fill_id="fill-2",
            order_id="order-2",
            instrument="BTC-USD",
            quantity=2.0,
            price=51000.0,
            side="buy",
            timestamp=datetime(2023, 1, 20, tzinfo=timezone.utc),  # Within time range
            fees=20.0,
            exchange="binance"
        )

        self.fill_service._fills = {
            "fill-1": fill1,
            "fill-2": fill2
        }

        # Call method
        result = self.fill_service.reconcile_fills("BTC-USD", start_time, end_time)

        # Check result
        self.assertEqual(result["instrument"], "BTC-USD")
        self.assertEqual(result["local_fills"], 2)
        self.assertEqual(result["exchange_fills"], 0)  # Placeholder in implementation
        self.assertEqual(len(result["discrepancies"]), 0)  # Empty in implementation

    @patch('json.load')
    @patch('builtins.open', create=True)
    def test_load_persisted_fills(self, mock_open, mock_json_load):
        """Test loading persisted fills from disk."""
        # Setup mock for file loading
        file_path1 = self.persistence_dir / "fill-1.json"
        file_path2 = self.persistence_dir / "fill-2.json"

        # Create mock files in the directory
        file_path1.touch()
        file_path2.touch()

        # Setup mock for fill data
        fill1_dict = {
            "fill_id": "fill-1",
            "order_id": "order-1",
            "instrument": "BTC-USD",
            "quantity": 1.0,
            "price": 50000.0,
            "side": "buy",
            "timestamp": "2023-01-15T00:00:00+00:00",
            "fees": 10.0,
            "exchange": "binance"
        }

        fill2_dict = {
            "fill_id": "fill-2",
            "order_id": "order-2",
            "instrument": "BTC-USD",
            "quantity": 2.0,
            "price": 51000.0,
            "side": "buy",
            "timestamp": "2023-01-20T00:00:00+00:00",
            "fees": 20.0,
            "exchange": "binance"
        }

        # Set up json.load to return different values on consecutive calls
        mock_json_load.side_effect = [fill1_dict, fill2_dict]

        # Call method
        count = self.fill_service.load_persisted_fills()

        # Check results
        self.assertEqual(count, 2)
        self.assertEqual(len(self.fill_service._fills), 2)
        self.assertTrue("fill-1" in self.fill_service._fills)
        self.assertTrue("fill-2" in self.fill_service._fills)
        self.assertTrue("order-1" in self.fill_service._order_fills)
        self.assertTrue("order-2" in self.fill_service._order_fills)

        # Verify state manager calls
        self.state_manager.set.assert_any_call("fills.fill-1", any)
        self.state_manager.set.assert_any_call("fills.fill-2", any)

    def test_apply_fill_to_order_first_fill(self):
        """Test applying a fill to an order with no previous fills."""
        # Create an order with no fills
        order = Order(
            order_id="order-123",
            instrument="BTC-USD",
            quantity=1.0,
            price=50000.0,
            side="buy",
            order_type="limit",
            status=OrderStatus.PENDING
        )

        # Create a fill
        fill = Fill(
            fill_id="fill-123",
            order_id="order-123",
            instrument="BTC-USD",
            quantity=0.5,
            price=50100.0,
            side="buy",
            timestamp=datetime.now(timezone.utc),
            fees=10.0,
            exchange="binance"
        )

        # Call method
        updated_order = self.fill_service._apply_fill_to_order(order, fill)

        # Check results
        self.assertEqual(updated_order.filled_quantity, 0.5)
        self.assertEqual(updated_order.average_price, 50100.0)
        self.assertEqual(updated_order.status, OrderStatus.PARTIALLY_FILLED)
        self.assertTrue('fills' in updated_order.params)
        self.assertEqual(updated_order.params['fills'], ["fill-123"])

    def test_apply_fill_to_order_additional_fill(self):
        """Test applying an additional fill to an order with existing fills."""
        # Create an order with existing fill data
        order = Order(
            order_id="order-123",
            instrument="BTC-USD",
            quantity=1.0,
            price=50000.0,
            side="buy",
            order_type="limit",
            status=OrderStatus.PARTIALLY_FILLED,
            filled_quantity=0.5,
            average_price=50000.0
        )
        order.params['fills'] = ["fill-previous"]

        # Create a new fill
        fill = Fill(
            fill_id="fill-new",
            order_id="order-123",
            instrument="BTC-USD",
            quantity=0.5,  # This will complete the order
            price=50200.0,  # Different price
            side="buy",
            timestamp=datetime.now(timezone.utc),
            fees=10.0,
            exchange="binance"
        )

        # Call method
        updated_order = self.fill_service._apply_fill_to_order(order, fill)

        # Check results
        self.assertEqual(updated_order.filled_quantity, 1.0)
        # Expected average price: (0.5*50000 + 0.5*50200) / 1.0 = 50100.0
        self.assertEqual(updated_order.average_price, 50100.0)
        self.assertEqual(updated_order.status, OrderStatus.FILLED)  # Fully filled
        self.assertTrue('fills' in updated_order.params)
        self.assertEqual(updated_order.params['fills'], ["fill-previous", "fill-new"])
        self.assertIsNotNone(updated_order.execution_time)

    def test_persistence_disabled(self):
        """Test that persistence operations are skipped when disabled."""
        # Create a service with persistence disabled
        config = {'persistence_enabled': False}
        fill_service = FillService(
            event_bus=self.event_bus,
            state_manager=self.state_manager,
            order_book=self.order_book,
            position_reconciliation=self.position_reconciliation,
            post_trade_reconciliation=self.post_trade_reconciliation,
            config=config
        )

        # Set up test
        fill_event = MagicMock(spec=Event)
        fill_event.data = self.sample_fill.to_dict()

        self.order_book.get_order.return_value = self.sample_order

        # Mock transaction context manager
        transaction_context = MagicMock()
        self.state_manager.transaction.return_value = transaction_context
        transaction_context.__enter__ = MagicMock()
        transaction_context.__exit__ = MagicMock()

        # Call method
        fill_service.process_fill(fill_event)

        # Check no file was created
        expected_file = self.persistence_dir / "fill-123.json"
        self.assertFalse(expected_file.exists())

        # Test load_persisted_fills returns 0
        count = fill_service.load_persisted_fills()
        self.assertEqual(count, 0)


if __name__ == '__main__':
    unittest.main()