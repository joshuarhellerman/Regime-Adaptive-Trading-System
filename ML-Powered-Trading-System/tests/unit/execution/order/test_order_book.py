import unittest
import tempfile
import shutil
import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from unittest.mock import patch, MagicMock

from execution.exchange.order.order_book import OrderBook
from execution.exchange.order.order import Order, OrderStatus, OrderSide, OrderType


class TestOrderBook(unittest.TestCase):
    """Test suite for OrderBook class."""

    def setUp(self):
        """Set up for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'persistence_enabled': True,
            'persistence_dir': self.temp_dir,
            'cleanup_after_days': 7,
            'cleanup_interval_hours': 12
        }
        self.order_book = OrderBook(self.config)
        
        # Create some test orders
        self.test_orders = [
            Order(
                order_id="order1",
                symbol="BTC/USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=1.0,
                price=50000.0,
                status=OrderStatus.OPEN,
                exchange_id="exchange1",
                strategy_id="strategy1"
            ),
            Order(
                order_id="order2",
                symbol="ETH/USD",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=10.0,
                price=3000.0,
                status=OrderStatus.PENDING,
                exchange_id="exchange1",
                strategy_id="strategy2"
            ),
            Order(
                order_id="order3",
                symbol="BTC/USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=0.5,
                price=49000.0,
                status=OrderStatus.FILLED,
                exchange_id="exchange2",
                strategy_id="strategy1",
                filled_quantity=0.5,
                average_price=49000.0
            ),
            Order(
                order_id="order4",
                symbol="ETH/USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=5.0,
                price=0.0,
                status=OrderStatus.PARTIALLY_FILLED,
                exchange_id="exchange2",
                strategy_id="strategy2",
                filled_quantity=2.0,
                average_price=3100.0,
                parent_order_id="order2"
            )
        ]

    def tearDown(self):
        """Clean up after each test."""
        shutil.rmtree(self.temp_dir)

    def _add_test_orders(self):
        """Helper to add test orders to the order book."""
        for order in self.test_orders:
            self.order_book.add_order(order)

    def test_init(self):
        """Test initialization of OrderBook."""
        self.assertEqual(self.order_book.get_order_count(), 0)
        self.assertEqual(len(self.order_book), 0)
        self.assertTrue(self.order_book._persistence_enabled)
        self.assertEqual(self.order_book._persistence_dir, Path(self.temp_dir))
        self.assertEqual(self.order_book._cleanup_after_days, 7)
        self.assertEqual(self.order_book._cleanup_interval_hours, 12)

    def test_add_order(self):
        """Test adding orders to the order book."""
        order = self.test_orders[0]
        # Add the order
        result = self.order_book.add_order(order)
        self.assertTrue(result)
        self.assertEqual(self.order_book.get_order_count(), 1)
        
        # Try adding the same order again
        result = self.order_book.add_order(order)
        self.assertFalse(result)
        self.assertEqual(self.order_book.get_order_count(), 1)
        
        # Add another order
        result = self.order_book.add_order(self.test_orders[1])
        self.assertTrue(result)
        self.assertEqual(self.order_book.get_order_count(), 2)

    def test_get_order(self):
        """Test retrieving an order by ID."""
        self._add_test_orders()
        
        # Get existing order
        order = self.order_book.get_order("order1")
        self.assertIsNotNone(order)
        self.assertEqual(order.order_id, "order1")
        
        # Get non-existent order
        order = self.order_book.get_order("non_existent")
        self.assertIsNone(order)

    def test_update_order(self):
        """Test updating an existing order."""
        self._add_test_orders()
        
        # Get and modify an order
        order = self.order_book.get_order("order1")
        order.status = OrderStatus.FILLED
        order.filled_quantity = 1.0
        order.average_price = 50100.0
        
        # Update the order
        result = self.order_book.update_order(order)
        self.assertTrue(result)
        
        # Check if the order was updated
        updated_order = self.order_book.get_order("order1")
        self.assertEqual(updated_order.status, OrderStatus.FILLED)
        self.assertEqual(updated_order.filled_quantity, 1.0)
        self.assertEqual(updated_order.average_price, 50100.0)
        
        # Try updating a non-existent order
        non_existent = Order(
            order_id="non_existent",
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            price=50000.0
        )
        result = self.order_book.update_order(non_existent)
        self.assertFalse(result)

    def test_remove_order(self):
        """Test removing an order from the order book."""
        self._add_test_orders()
        
        # Remove an existing order
        result = self.order_book.remove_order("order1")
        self.assertTrue(result)
        self.assertEqual(self.order_book.get_order_count(), 3)
        self.assertIsNone(self.order_book.get_order("order1"))
        
        # Try removing the same order again
        result = self.order_book.remove_order("order1")
        self.assertFalse(result)
        
        # Make sure indices are updated
        orders_by_symbol = self.order_book.get_orders_by_symbol("BTC/USD")
        self.assertEqual(len(orders_by_symbol), 1)  # Only order3 should remain
        self.assertEqual(orders_by_symbol[0].order_id, "order3")

    def test_get_orders_by_status(self):
        """Test retrieving orders by status."""
        self._add_test_orders()
        
        # Get orders by status
        open_orders = self.order_book.get_orders_by_status(OrderStatus.OPEN)
        self.assertEqual(len(open_orders), 1)
        self.assertEqual(open_orders[0].order_id, "order1")
        
        pending_orders = self.order_book.get_orders_by_status(OrderStatus.PENDING)
        self.assertEqual(len(pending_orders), 1)
        self.assertEqual(pending_orders[0].order_id, "order2")
        
        # Check a status with no orders
        cancelled_orders = self.order_book.get_orders_by_status(OrderStatus.CANCELLED)
        self.assertEqual(len(cancelled_orders), 0)

    def test_get_orders_by_symbol(self):
        """Test retrieving orders by symbol."""
        self._add_test_orders()
        
        # Get orders by symbol
        btc_orders = self.order_book.get_orders_by_symbol("BTC/USD")
        self.assertEqual(len(btc_orders), 2)
        self.assertIn(btc_orders[0].order_id, ["order1", "order3"])
        self.assertIn(btc_orders[1].order_id, ["order1", "order3"])
        
        eth_orders = self.order_book.get_orders_by_symbol("ETH/USD")
        self.assertEqual(len(eth_orders), 2)
        self.assertIn(eth_orders[0].order_id, ["order2", "order4"])
        self.assertIn(eth_orders[1].order_id, ["order2", "order4"])
        
        # Check a symbol with no orders
        ltc_orders = self.order_book.get_orders_by_symbol("LTC/USD")
        self.assertEqual(len(ltc_orders), 0)

    def test_get_orders_by_exchange(self):
        """Test retrieving orders by exchange."""
        self._add_test_orders()
        
        # Get orders by exchange
        exchange1_orders = self.order_book.get_orders_by_exchange("exchange1")
        self.assertEqual(len(exchange1_orders), 2)
        self.assertIn(exchange1_orders[0].order_id, ["order1", "order2"])
        self.assertIn(exchange1_orders[1].order_id, ["order1", "order2"])
        
        exchange2_orders = self.order_book.get_orders_by_exchange("exchange2")
        self.assertEqual(len(exchange2_orders), 2)
        self.assertIn(exchange2_orders[0].order_id, ["order3", "order4"])
        self.assertIn(exchange2_orders[1].order_id, ["order3", "order4"])
        
        # Check an exchange with no orders
        other_orders = self.order_book.get_orders_by_exchange("other_exchange")
        self.assertEqual(len(other_orders), 0)

    def test_get_orders_by_strategy(self):
        """Test retrieving orders by strategy."""
        self._add_test_orders()
        
        # Get orders by strategy
        strategy1_orders = self.order_book.get_orders_by_strategy("strategy1")
        self.assertEqual(len(strategy1_orders), 2)
        self.assertIn(strategy1_orders[0].order_id, ["order1", "order3"])
        self.assertIn(strategy1_orders[1].order_id, ["order1", "order3"])
        
        strategy2_orders = self.order_book.get_orders_by_strategy("strategy2")
        self.assertEqual(len(strategy2_orders), 2)
        self.assertIn(strategy2_orders[0].order_id, ["order2", "order4"])
        self.assertIn(strategy2_orders[1].order_id, ["order2", "order4"])
        
        # Check a strategy with no orders
        other_orders = self.order_book.get_orders_by_strategy("other_strategy")
        self.assertEqual(len(other_orders), 0)

    def test_get_child_orders(self):
        """Test retrieving child orders for a parent order."""
        self._add_test_orders()
        
        # Get child orders for a parent
        child_orders = self.order_book.get_child_orders("order2")
        self.assertEqual(len(child_orders), 1)
        self.assertEqual(child_orders[0].order_id, "order4")
        
        # Check a parent with no children
        no_children = self.order_book.get_child_orders("order1")
        self.assertEqual(len(no_children), 0)

    def test_get_active_orders(self):
        """Test retrieving all active orders."""
        self._add_test_orders()
        
        # Get active orders
        active_orders = self.order_book.get_active_orders()
        self.assertEqual(len(active_orders), 3)  # order1 (OPEN), order2 (PENDING), order4 (PARTIALLY_FILLED)
        
        # Change status of one active order to a terminal state
        order1 = self.order_book.get_order("order1")
        order1.status = OrderStatus.CANCELLED
        self.order_book.update_order(order1)
        
        # Check active orders again
        active_orders = self.order_book.get_active_orders()
        self.assertEqual(len(active_orders), 2)  # order2 (PENDING), order4 (PARTIALLY_FILLED)

    def test_get_orders_by_side(self):
        """Test retrieving orders by side."""
        self._add_test_orders()
        
        # Get orders by side
        buy_orders = self.order_book.get_orders_by_side(OrderSide.BUY)
        self.assertEqual(len(buy_orders), 3)  # order1, order3, order4
        
        sell_orders = self.order_book.get_orders_by_side(OrderSide.SELL)
        self.assertEqual(len(sell_orders), 1)  # order2

    def test_get_orders_by_type(self):
        """Test retrieving orders by type."""
        self._add_test_orders()
        
        # Get orders by type
        market_orders = self.order_book.get_orders_by_type(OrderType.MARKET)
        self.assertEqual(len(market_orders), 2)  # order1, order4
        
        limit_orders = self.order_book.get_orders_by_type(OrderType.LIMIT)
        self.assertEqual(len(limit_orders), 2)  # order2, order3

    def test_get_all_orders(self):
        """Test retrieving all orders in the order book."""
        self._add_test_orders()
        
        # Get all orders
        all_orders = self.order_book.get_all_orders()
        self.assertEqual(len(all_orders), 4)

    def test_get_order_count(self):
        """Test getting the total order count."""
        self.assertEqual(self.order_book.get_order_count(), 0)
        
        self._add_test_orders()
        self.assertEqual(self.order_book.get_order_count(), 4)
        
        self.order_book.remove_order("order1")
        self.assertEqual(self.order_book.get_order_count(), 3)

    def test_get_order_count_by_status(self):
        """Test getting order counts by status."""
        self._add_test_orders()
        
        # Get order counts by status
        status_counts = self.order_book.get_order_count_by_status()
        self.assertEqual(status_counts[OrderStatus.OPEN.value], 1)
        self.assertEqual(status_counts[OrderStatus.PENDING.value], 1)
        self.assertEqual(status_counts[OrderStatus.FILLED.value], 1)
        self.assertEqual(status_counts[OrderStatus.PARTIALLY_FILLED.value], 1)
        
        # Check a status with no orders
        self.assertFalse(OrderStatus.CANCELLED.value in status_counts)

    def test_get_position_by_symbol(self):
        """Test calculating net position for a symbol."""
        self._add_test_orders()
        
        # Calculate position for BTC/USD
        # order1: BUY 1.0, not filled
        # order3: BUY 0.5, filled 0.5
        position = self.order_book.get_position_by_symbol("BTC/USD")
        self.assertEqual(position, 0.5)
        
        # Calculate position for ETH/USD
        # order2: SELL 10.0, not filled
        # order4: BUY 5.0, filled 2.0
        position = self.order_book.get_position_by_symbol("ETH/USD")
        self.assertEqual(position, 2.0)
        
        # Update order4 to be filled more
        order4 = self.order_book.get_order("order4")
        order4.filled_quantity = 5.0
        self.order_book.update_order(order4)
        
        # Calculate position again
        position = self.order_book.get_position_by_symbol("ETH/USD")
        self.assertEqual(position, 5.0)
        
        # Now fill order2 (SELL)
        order2 = self.order_book.get_order("order2")
        order2.status = OrderStatus.FILLED
        order2.filled_quantity = 10.0
        self.order_book.update_order(order2)
        
        # Calculate position again
        position = self.order_book.get_position_by_symbol("ETH/USD")
        self.assertEqual(position, -5.0)  # 5.0 bought - 10.0 sold

    def test_order_history(self):
        """Test order history recording."""
        # Add an order and check history
        self.order_book.add_order(self.test_orders[0])
        history = self.order_book.get_order_history("order1")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["status"], OrderStatus.OPEN.value)
        
        # Update the order and check history
        order = self.order_book.get_order("order1")
        order.status = OrderStatus.PARTIALLY_FILLED
        order.filled_quantity = 0.5
        order.average_price = 50000.0
        self.order_book.update_order(order)
        
        history = self.order_book.get_order_history("order1")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[1]["status"], OrderStatus.PARTIALLY_FILLED.value)
        self.assertEqual(history[1]["filled_quantity"], 0.5)
        self.assertEqual(history[1]["average_price"], 50000.0)
        
        # Update again and check history
        order.status = OrderStatus.FILLED
        order.filled_quantity = 1.0
        self.order_book.update_order(order)
        
        history = self.order_book.get_order_history("order1")
        self.assertEqual(len(history), 3)
        self.assertEqual(history[2]["status"], OrderStatus.FILLED.value)
        self.assertEqual(history[2]["filled_quantity"], 1.0)

    def test_export_orders_csv(self):
        """Test exporting orders to CSV."""
        self._add_test_orders()
        
        # Export to a specific file
        csv_path = os.path.join(self.temp_dir, "export_test.csv")
        result_path = self.order_book.export_orders_csv(csv_path)
        self.assertEqual(result_path, csv_path)
        self.assertTrue(os.path.exists(csv_path))
        
        # Read the CSV back and check contents
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 4)
        self.assertIn("order_id", df.columns)
        self.assertIn("symbol", df.columns)
        self.assertIn("status", df.columns)
        
        # Check if default path works
        result_path = self.order_book.export_orders_csv()
        self.assertTrue(os.path.exists(result_path))

    def test_persistence(self):
        """Test order persistence to disk."""
        # Add orders
        self._add_test_orders()
        
        # Check if files were created
        for order in self.test_orders:
            file_path = os.path.join(self.temp_dir, f"{order.order_id}.json")
            self.assertTrue(os.path.exists(file_path))
        
        # Create a new order book and load persisted orders
        new_order_book = OrderBook(self.config)
        loaded_count = new_order_book.load_persisted_orders()
        self.assertEqual(loaded_count, 4)
        self.assertEqual(new_order_book.get_order_count(), 4)
        
        # Check if all orders were loaded correctly
        for order_id in ["order1", "order2", "order3", "order4"]:
            self.assertIsNotNone(new_order_book.get_order(order_id))

    def test_cleanup_old_orders(self):
        """Test cleaning up old orders."""
        self._add_test_orders()
        
        # Set update times for orders
        now = datetime.utcnow()
        for i, order_id in enumerate(["order1", "order2", "order3", "order4"]):
            order = self.order_book.get_order(order_id)
            # order1 and order2 are recent, order3 and order4 are old
            if i < 2:
                order.update_time = now - timedelta(days=1)
            else:
                order.update_time = now - timedelta(days=10)
            self.order_book.update_order(order)
        
        # Make order3 and order4 complete (terminal status)
        order3 = self.order_book.get_order("order3")
        order3.is_complete = True  # Set for testing, normally derived from status
        self.order_book.update_order(order3)
        
        order4 = self.order_book.get_order("order4")
        order4.status = OrderStatus.CANCELLED
        order4.is_complete = True  # Set for testing, normally derived from status
        self.order_book.update_order(order4)
        
        # Clean up orders older than 7 days
        removed = self.order_book.cleanup_old_orders(7)
        self.assertEqual(removed, 2)  # order3 and order4
        self.assertEqual(self.order_book.get_order_count(), 2)
        self.assertIsNotNone(self.order_book.get_order("order1"))
        self.assertIsNotNone(self.order_book.get_order("order2"))
        self.assertIsNone(self.order_book.get_order("order3"))
        self.assertIsNone(self.order_book.get_order("order4"))

    def test_auto_cleanup(self):
        """Test automatic cleanup based on configured interval."""
        self._add_test_orders()
        
        # Set update times for orders
        now = datetime.utcnow()
        for order_id in ["order3", "order4"]:
            order = self.order_book.get_order(order_id)
            order.update_time = now - timedelta(days=10)
            order.is_complete = True  # Set for testing
            self.order_book.update_order(order)
        
        # Set last cleanup time to be recent
        self.order_book._last_cleanup = now - timedelta(hours=1)
        result = self.order_book.auto_cleanup()
        self.assertFalse(result)  # No cleanup performed
        self.assertEqual(self.order_book.get_order_count(), 4)
        
        # Set last cleanup time to be old
        self.order_book._last_cleanup = now - timedelta(hours=15)
        result = self.order_book.auto_cleanup()
        self.assertTrue(result)  # Cleanup performed
        self.assertEqual(self.order_book.get_order_count(), 2)  # order1 and order2 remain

    def test_find_orders(self):
        """Test finding orders by criteria."""
        self._add_test_orders()
        
        # Find by symbol
        orders = self.order_book.find_orders({"symbol": "BTC/USD"})
        self.assertEqual(len(orders), 2)
        
        # Find by multiple criteria
        orders = self.order_book.find_orders({
            "symbol": "BTC/USD",
            "side": OrderSide.BUY
        })
        self.assertEqual(len(orders), 2)
        
        # Find by status as string
        orders = self.order_book.find_orders({"status": "OPEN"})
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].order_id, "order1")
        
        # Find with no matches
        orders = self.order_book.find_orders({
            "symbol": "BTC/USD",
            "side": OrderSide.SELL
        })
        self.assertEqual(len(orders), 0)

    def test_group_orders_by(self):
        """Test grouping orders by attribute."""
        self._add_test_orders()
        
        # Group by symbol
        grouped = self.order_book.group_orders_by("symbol")
        self.assertEqual(len(grouped), 2)  # Two symbols
        self.assertEqual(len(grouped["BTC/USD"]), 2)
        self.assertEqual(len(grouped["ETH/USD"]), 2)
        
        # Group by status
        grouped = self.order_book.group_orders_by("status")
        self.assertEqual(len(grouped), 4)  # Four statuses
        self.assertEqual(len(grouped["OPEN"]), 1)
        self.assertEqual(len(grouped["PENDING"]), 1)
        self.assertEqual(len(grouped["FILLED"]), 1)
        self.assertEqual(len(grouped["PARTIALLY_FILLED"]), 1)
        
        # Group by side
        grouped = self.order_book.group_orders_by("side")
        self.assertEqual(len(grouped), 2)  # Two sides
        self.assertEqual(len(grouped["BUY"]), 3)
        self.assertEqual(len(grouped["SELL"]), 1)

    def test_bulk_update_status(self):
        """Test bulk updating order status."""
        self._add_test_orders()
        
        # Update status for multiple orders
        updated = self.order_book.bulk_update_status(
            ["order1", "order2"], 
            OrderStatus.CANCELLED,
            "Cancelled by user"
        )
        self.assertEqual(updated, 2)
        
        # Check if orders were updated
        order1 = self.order_book.get_order("order1")
        self.assertEqual(order1.status, OrderStatus.CANCELLED)
        self.assertEqual(order1.status_message, "Cancelled by user")
        
        order2 = self.order_book.get_order("order2")
        self.assertEqual(order2.status, OrderStatus.CANCELLED)
        self.assertEqual(order2.status_message, "Cancelled by user")
        
        # Try with some non-existent orders
        updated = self.order_book.bulk_update_status(
            ["order1", "non_existent"],
            OrderStatus.OPEN
        )
        self.assertEqual(updated, 1)  # Only order1 should be updated

    def test_purge_all(self):
        """Test purging all orders."""
        self._add_test_orders()
        self.assertEqual(self.order_book.get_order_count(), 4)
        
        # Purge all orders
        removed = self.order_book.purge_all()
        self.assertEqual(removed, 4)
        self.assertEqual(self.order_book.get_order_count(), 0)
        
        # Check if all collections are empty
        self.assertEqual(len(self.order_book.get_all_orders()), 0)
        self.assertEqual(len(self.order_book.get_orders_by_symbol("BTC/USD")), 0)
        self.assertEqual(len(self.order_book.get_orders_by_status(OrderStatus.OPEN)), 0)

    def test_contains(self):
        """Test the __contains__ special method."""
        self._add_test_orders()
        
        self.assertTrue("order1" in self.order_book)
        self.assertFalse("non_existent" in self.order_book)

    def test_len(self):
        """Test the __len__ special method."""
        self.assertEqual(len(self.order_book), 0)
        
        self._add_test_orders()
        self.assertEqual(len(self.order_book), 4)


if __name__ == '__main__':
    unittest.main()