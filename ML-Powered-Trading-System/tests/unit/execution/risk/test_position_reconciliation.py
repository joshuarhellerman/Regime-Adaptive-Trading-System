import unittest
from unittest.mock import Mock, patch, MagicMock
import time
from datetime import datetime, timedelta
from decimal import Decimal

# Import modules to test
from execution.order.order import Order, OrderStatus, OrderSide
from execution.exchange.risk.position_reconciliation import (
    Position, PositionManager, PositionReconciliationStatus
)


class TestPosition(unittest.TestCase):
    """Test the Position dataclass."""

    def test_position_init(self):
        """Test Position initialization."""
        now = datetime.now()
        position = Position(
            symbol="BTC-USD",
            size=1.5,
            avg_price=50000.0,
            updated_at=now,
            exchange_id="exchange1"
        )
        
        self.assertEqual(position.symbol, "BTC-USD")
        self.assertEqual(position.size, 1.5)
        self.assertEqual(position.avg_price, 50000.0)
        self.assertEqual(position.updated_at, now)
        self.assertEqual(position.exchange_id, "exchange1")
        self.assertEqual(position.open_orders, [])
    
    def test_position_init_with_open_orders(self):
        """Test Position initialization with open orders."""
        now = datetime.now()
        position = Position(
            symbol="ETH-USD",
            size=10.0,
            avg_price=2000.0,
            updated_at=now,
            exchange_id="exchange2",
            open_orders=["order1", "order2"]
        )
        
        self.assertEqual(position.symbol, "ETH-USD")
        self.assertEqual(position.open_orders, ["order1", "order2"])


class TestPositionManager(unittest.TestCase):
    """Test the PositionManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.event_bus_mock = Mock()
        
        # Patch the get_event_bus function to return our mock
        self.event_bus_patcher = patch('execution.exchange.risk.position_reconciliation.get_event_bus', 
                                      return_value=self.event_bus_mock)
        self.event_bus_patcher.start()
        
        # Create a test config
        self.test_config = {
            "auto_reconcile": True,
            "reconciliation_interval_minutes": 30,
            "reconciliation_size_tolerance": 0.01,
            "reconciliation_price_tolerance": 0.01,
            "max_retry_count": 2,
            "max_position_history": 50
        }
        
        # Create a position manager with our test config
        self.position_manager = PositionManager(config=self.test_config)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.event_bus_patcher.stop()
    
    def test_init(self):
        """Test PositionManager initialization."""
        # Check that the position manager was initialized correctly
        self.assertEqual(self.position_manager.positions, {})
        self.assertEqual(self.position_manager.position_history, {})
        self.assertIsNone(self.position_manager.last_reconciliation_time)
        
        # Check that config values were set correctly
        self.assertEqual(self.position_manager.reconciliation_settings["auto_reconcile"], True)
        self.assertEqual(self.position_manager.reconciliation_settings["reconciliation_interval_minutes"], 30)
        self.assertEqual(self.position_manager.reconciliation_settings["reconciliation_size_tolerance"], 0.01)
        self.assertEqual(self.position_manager.reconciliation_settings["reconciliation_price_tolerance"], 0.01)
        self.assertEqual(self.position_manager.reconciliation_settings["max_retry_count"], 2)
    
    def test_get_position_no_position(self):
        """Test getting position size when no position exists."""
        self.assertEqual(self.position_manager.get_position("BTC-USD"), 0.0)
    
    def test_get_position_with_position(self):
        """Test getting position size when position exists."""
        # Create a test position
        self.position_manager.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            size=2.5,
            avg_price=45000.0,
            updated_at=datetime.now(),
            exchange_id="exchange1"
        )
        
        self.assertEqual(self.position_manager.get_position("BTC-USD"), 2.5)
    
    def test_get_position_details_no_position(self):
        """Test getting position details when no position exists."""
        self.assertIsNone(self.position_manager.get_position_details("BTC-USD"))
    
    def test_get_position_details_with_position(self):
        """Test getting position details when position exists."""
        position = Position(
            symbol="BTC-USD",
            size=2.5,
            avg_price=45000.0,
            updated_at=datetime.now(),
            exchange_id="exchange1"
        )
        self.position_manager.positions["BTC-USD"] = position
        
        self.assertEqual(self.position_manager.get_position_details("BTC-USD"), position)
    
    def test_get_all_positions(self):
        """Test getting all positions."""
        # Create test positions
        position1 = Position(
            symbol="BTC-USD",
            size=2.5,
            avg_price=45000.0,
            updated_at=datetime.now(),
            exchange_id="exchange1"
        )
        position2 = Position(
            symbol="ETH-USD",
            size=-10.0,
            avg_price=2000.0,
            updated_at=datetime.now(),
            exchange_id="exchange1"
        )
        
        self.position_manager.positions["BTC-USD"] = position1
        self.position_manager.positions["ETH-USD"] = position2
        
        positions = self.position_manager.get_all_positions()
        self.assertEqual(len(positions), 2)
        self.assertEqual(positions["BTC-USD"], position1)
        self.assertEqual(positions["ETH-USD"], position2)
        
        # Verify it's a copy, not the original
        positions["BTC-USD"].size = 5.0
        self.assertEqual(self.position_manager.positions["BTC-USD"].size, 2.5)
    
    def test_get_position_value(self):
        """Test getting position value."""
        # Create a test position
        self.position_manager.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            size=2.5,
            avg_price=45000.0,
            updated_at=datetime.now(),
            exchange_id="exchange1"
        )
        
        # Test with no current price (should use avg_price)
        self.assertEqual(self.position_manager.get_position_value("BTC-USD"), 2.5 * 45000.0)
        
        # Test with current price
        self.assertEqual(self.position_manager.get_position_value("BTC-USD", 50000.0), 2.5 * 50000.0)
        
        # Test with no position
        self.assertEqual(self.position_manager.get_position_value("ETH-USD"), 0.0)
        
        # Test with zero position
        self.position_manager.positions["ETH-USD"] = Position(
            symbol="ETH-USD",
            size=0.0,
            avg_price=2000.0,
            updated_at=datetime.now(),
            exchange_id="exchange1"
        )
        self.assertEqual(self.position_manager.get_position_value("ETH-USD"), 0.0)
    
    def test_get_total_position_value(self):
        """Test getting total position value."""
        # Create test positions
        self.position_manager.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            size=2.5,
            avg_price=45000.0,
            updated_at=datetime.now(),
            exchange_id="exchange1"
        )
        self.position_manager.positions["ETH-USD"] = Position(
            symbol="ETH-USD",
            size=-10.0,
            avg_price=2000.0,
            updated_at=datetime.now(),
            exchange_id="exchange1"
        )
        self.position_manager.positions["XRP-USD"] = Position(
            symbol="XRP-USD",
            size=0.0,
            avg_price=1.0,
            updated_at=datetime.now(),
            exchange_id="exchange1"
        )
        
        # Test with no price provider (should use avg_price)
        expected_value = (2.5 * 45000.0) + abs(-10.0 * 2000.0)
        self.assertEqual(self.position_manager.get_total_position_value(), expected_value)
        
        # Test with price provider
        mock_price_provider = lambda symbol: {
            "BTC-USD": 50000.0,
            "ETH-USD": 2500.0
        }.get(symbol, 0.0)
        
        expected_value = (2.5 * 50000.0) + abs(-10.0 * 2500.0)
        self.assertEqual(self.position_manager.get_total_position_value(mock_price_provider), expected_value)
        
        # Test with price provider that raises exception
        def failing_price_provider(symbol):
            if symbol == "ETH-USD":
                raise Exception("Price not available")
            return {
                "BTC-USD": 50000.0
            }.get(symbol, 0.0)
        
        # Should use avg_price for ETH-USD
        expected_value = (2.5 * 50000.0) + abs(-10.0 * 2000.0)
        self.assertEqual(self.position_manager.get_total_position_value(failing_price_provider), expected_value)
    
    def test_get_net_exposure(self):
        """Test getting net market exposure."""
        # Create test positions
        self.position_manager.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            size=2.5,
            avg_price=45000.0,
            updated_at=datetime.now(),
            exchange_id="exchange1"
        )
        self.position_manager.positions["ETH-USD"] = Position(
            symbol="ETH-USD",
            size=-10.0,
            avg_price=2000.0,
            updated_at=datetime.now(),
            exchange_id="exchange1"
        )
        
        # Test with no price provider (should use avg_price)
        expected_net = (2.5 * 45000.0) + (-10.0 * 2000.0)
        self.assertEqual(self.position_manager.get_net_exposure(), expected_net)
        
        # Test with price provider
        mock_price_provider = lambda symbol: {
            "BTC-USD": 50000.0,
            "ETH-USD": 2500.0
        }.get(symbol, 0.0)
        
        expected_net = (2.5 * 50000.0) + (-10.0 * 2500.0)
        self.assertEqual(self.position_manager.get_net_exposure(mock_price_provider), expected_net)
    
    def test_update_from_fill_new_position(self):
        """Test updating position from fill when no position exists."""
        # Create a test order
        order = Mock(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            exchange_id="exchange1",
            order_id="order1"
        )
        
        # Update from fill
        self.position_manager.update_from_fill(order, 1.5, 50000.0)
        
        # Check position was created
        self.assertIn("BTC-USD", self.position_manager.positions)
        position = self.position_manager.positions["BTC-USD"]
        self.assertEqual(position.size, 1.5)
        self.assertEqual(position.avg_price, 50000.0)
        self.assertEqual(position.exchange_id, "exchange1")
        
        # Check position history was updated
        self.assertIn("BTC-USD", self.position_manager.position_history)
        self.assertEqual(len(self.position_manager.position_history["BTC-USD"]), 1)
        history_entry = self.position_manager.position_history["BTC-USD"][0]
        self.assertEqual(history_entry["old_size"], 0.0)
        self.assertEqual(history_entry["new_size"], 1.5)
        self.assertEqual(history_entry["price"], 50000.0)
        self.assertEqual(history_entry["source"], "order1")
        
        # Check event was published
        self.event_bus_mock.publish.assert_called_once()
    
    def test_update_from_fill_existing_position_same_side(self):
        """Test updating position from fill with existing position on same side."""
        # Create an existing position
        self.position_manager.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            size=1.0,
            avg_price=45000.0,
            updated_at=datetime.now(),
            exchange_id="exchange1"
        )
        
        # Create a test order (buying more)
        order = Mock(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            exchange_id="exchange1",
            order_id="order1"
        )
        
        # Update from fill
        self.position_manager.update_from_fill(order, 1.5, 50000.0)
        
        # Check position was updated
        position = self.position_manager.positions["BTC-USD"]
        self.assertEqual(position.size, 2.5)  # 1.0 + 1.5
        
        # Check avg price calculation: (1.0 * 45000 + 1.5 * 50000) / 2.5
        expected_avg_price = (1.0 * 45000.0 + 1.5 * 50000.0) / 2.5
        self.assertAlmostEqual(position.avg_price, expected_avg_price)
    
    def test_update_from_fill_existing_position_opposite_side(self):
        """Test updating position from fill with existing position on opposite side."""
        # Create an existing position (long)
        self.position_manager.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            size=1.0,
            avg_price=45000.0,
            updated_at=datetime.now(),
            exchange_id="exchange1"
        )
        
        # Create a test order (selling more than current position)
        order = Mock(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            exchange_id="exchange1",
            order_id="order1"
        )
        
        # Update from fill
        self.position_manager.update_from_fill(order, 2.0, 50000.0)
        
        # Check position was updated
        position = self.position_manager.positions["BTC-USD"]
        self.assertEqual(position.size, -1.0)  # 1.0 - 2.0
        
        # When position flips, avg price should be the fill price
        self.assertEqual(position.avg_price, 50000.0)
    
    def test_update_from_fill_position_to_zero(self):
        """Test updating position from fill reducing position to zero."""
        # Create an existing position
        self.position_manager.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            size=1.5,
            avg_price=45000.0,
            updated_at=datetime.now(),
            exchange_id="exchange1",
            open_orders=["order1"]
        )
        
        # Create a test order (selling exactly current position)
        order = Mock(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            exchange_id="exchange1",
            order_id="order1"
        )
        
        # Update from fill
        self.position_manager.update_from_fill(order, 1.5, 50000.0)
        
        # Check position was updated
        position = self.position_manager.positions["BTC-USD"]
        self.assertEqual(position.size, 0.0)
        self.assertEqual(position.avg_price, 0.0)
        
        # Check order was removed from open orders
        self.assertEqual(position.open_orders, [])
    
    def test_set_position_new(self):
        """Test setting a new position manually."""
        # Set new position
        self.position_manager.set_position("BTC-USD", 2.0, 45000.0, "exchange1")
        
        # Check position was created
        self.assertIn("BTC-USD", self.position_manager.positions)
        position = self.position_manager.positions["BTC-USD"]
        self.assertEqual(position.size, 2.0)
        self.assertEqual(position.avg_price, 45000.0)
        self.assertEqual(position.exchange_id, "exchange1")
        
        # Check position history was updated
        self.assertIn("BTC-USD", self.position_manager.position_history)
        self.assertEqual(len(self.position_manager.position_history["BTC-USD"]), 1)
        history_entry = self.position_manager.position_history["BTC-USD"][0]
        self.assertEqual(history_entry["old_size"], 0)
        self.assertEqual(history_entry["new_size"], 2.0)
        self.assertEqual(history_entry["price"], 45000.0)
        self.assertEqual(history_entry["source"], "manual_update")
        
        # Check event was published
        self.event_bus_mock.publish.assert_called_once()
    
    def test_set_position_existing(self):
        """Test setting an existing position manually."""
        # Create an existing position
        self.position_manager.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            size=1.0,
            avg_price=45000.0,
            updated_at=datetime.now(),
            exchange_id="exchange1"
        )
        
        # Set position
        self.position_manager.set_position("BTC-USD", 2.0, 50000.0)
        
        # Check position was updated
        position = self.position_manager.positions["BTC-USD"]
        self.assertEqual(position.size, 2.0)
        self.assertEqual(position.avg_price, 50000.0)
    
    def test_set_position_missing_parameters(self):
        """Test setting a new position with missing parameters."""
        # Should raise exception for missing avg_price
        with self.assertRaises(ValueError):
            self.position_manager.set_position("BTC-USD", 2.0, exchange_id="exchange1")
        
        # Should raise exception for missing exchange_id
        with self.assertRaises(ValueError):
            self.position_manager.set_position("BTC-USD", 2.0, avg_price=45000.0)
    
    def test_add_open_order_new_position(self):
        """Test adding an open order to a new position."""
        # Add open order
        self.position_manager.add_open_order("BTC-USD", "order1", "exchange1")
        
        # Check position was created
        self.assertIn("BTC-USD", self.position_manager.positions)
        position = self.position_manager.positions["BTC-USD"]
        self.assertEqual(position.size, 0.0)
        self.assertEqual(position.avg_price, 0.0)
        self.assertEqual(position.exchange_id, "exchange1")
        self.assertEqual(position.open_orders, ["order1"])
    
    def test_add_open_order_existing_position(self):
        """Test adding an open order to an existing position."""
        # Create an existing position
        self.position_manager.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            size=1.0,
            avg_price=45000.0,
            updated_at=datetime.now(),
            exchange_id="exchange1",
            open_orders=["order1"]
        )
        
        # Add open order
        self.position_manager.add_open_order("BTC-USD", "order2", "exchange1")
        
        # Check order was added
        position = self.position_manager.positions["BTC-USD"]
        self.assertEqual(position.open_orders, ["order1", "order2"])
        
        # Add duplicate order (should not be added)
        self.position_manager.add_open_order("BTC-USD", "order1", "exchange1")
        self.assertEqual(position.open_orders, ["order1", "order2"])
    
    def test_remove_open_order(self):
        """Test removing an open order."""
        # Create an existing position with open orders
        self.position_manager.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            size=1.0,
            avg_price=45000.0,
            updated_at=datetime.now(),
            exchange_id="exchange1",
            open_orders=["order1", "order2"]
        )
        
        # Remove order
        self.position_manager.remove_open_order("BTC-USD", "order1")
        
        # Check order was removed
        position = self.position_manager.positions["BTC-USD"]
        self.assertEqual(position.open_orders, ["order2"])
        
        # Remove non-existent order (should not error)
        self.position_manager.remove_open_order("BTC-USD", "order3")
        self.assertEqual(position.open_orders, ["order2"])
        
        # Remove order from non-existent position (should not error)
        self.position_manager.remove_open_order("ETH-USD", "order1")
    
    @patch('execution.exchange.risk.position_reconciliation.datetime')
    def test_reconcile_with_exchange_too_soon(self, mock_datetime):
        """Test reconciliation skipped if run too soon."""
        # Set up mock datetime
        now = datetime.now()
        mock_datetime.now.return_value = now
        
        # Set last reconciliation time
        self.position_manager.last_reconciliation_time = now - timedelta(minutes=10)
        
        # Create mock exchange gateway
        mock_exchange = Mock()
        
        # Reconcile (should be skipped)
        results = self.position_manager.reconcile_with_exchange(mock_exchange)
        
        # Check no exchange calls were made and empty results returned
        mock_exchange.get_all_positions.assert_not_called()
        self.assertEqual(results, {})
    
    @patch('execution.exchange.risk.position_reconciliation.datetime')
    def test_reconcile_with_exchange_failed_to_get_positions(self, mock_datetime):
        """Test reconciliation when exchange fails to return positions."""
        # Set up mock datetime
        now = datetime.now()
        mock_datetime.now.return_value = now
        
        # Create mock exchange gateway that returns None
        mock_exchange = Mock()
        mock_exchange.get_all_positions.return_value = None
        
        # Reconcile
        results = self.position_manager.reconcile_with_exchange(mock_exchange)
        
        # Check exchange call was made and empty results returned
        mock_exchange.get_all_positions.assert_called_once()
        self.assertEqual(results, {})
        self.assertEqual(self.position_manager.last_reconciliation_time, now)
    
    @patch('execution.exchange.risk.position_reconciliation.datetime')
    def test_reconcile_with_exchange_positions_match(self, mock_datetime):
        """Test reconciliation when positions match."""
        # Set up mock datetime
        now = datetime.now()
        mock_datetime.now.return_value = now
        
        # Create test positions
        self.position_manager.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            size=2.5,
            avg_price=45000.0,
            updated_at=now - timedelta(hours=1),
            exchange_id="exchange1"
        )
        
        # Create mock exchange gateway
        mock_exchange = Mock()
        mock_exchange.get_all_positions.return_value = {
            "BTC-USD": {
                "size": 2.5,
                "avg_price": 45000.0,
                "exchange_id": "exchange1"
            }
        }
        
        # Reconcile
        results = self.position_manager.reconcile_with_exchange(mock_exchange)
        
        # Check results
        self.assertEqual(results["BTC-USD"], PositionReconciliationStatus.MATCHED)
        
        # Check event was published
        self.event_bus_mock.publish.assert_called_once()
    
    @patch('execution.exchange.risk.position_reconciliation.datetime')
    def test_reconcile_with_exchange_position_size_mismatch(self, mock_datetime):
        """Test reconciliation when position sizes don't match."""
        # Set up mock datetime
        now = datetime.now()
        mock_datetime.now.return_value = now
        
        # Create test positions
        self.position_manager.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            size=2.5,
            avg_price=45000.0,
            updated_at=now - timedelta(hours=1),
            exchange_id="exchange1"
        )
        
        # Create mock exchange gateway with different size
        mock_exchange = Mock()
        mock_exchange.get_all_positions.return_value = {
            "BTC-USD": {
                "size": 3.0,  # Different size
                "avg_price": 45000.0,
                "exchange_id": "exchange1"
            }
        }
        
        # Reconcile
        results = self.position_manager.reconcile_with_exchange(mock_exchange)
        
        # Check results
        self.assertEqual(results["BTC-USD"], PositionReconciliationStatus.ADJUSTED)
        
        # Check position was adjusted
        position = self.position_manager.positions["BTC-USD"]
        self.assertEqual(position.size, 3.0)
        
        # Check position history was updated
        history_entry = self.position_manager.position_history["BTC-USD"][-1]
        self.assertEqual(history_entry["old_size"], 2.5)
        self.assertEqual(history_entry["new_size"], 3.0)
        self.assertEqual(history_entry["source"], "reconciliation")
    
    @patch('execution.exchange.risk.position_reconciliation.datetime')
    def test_reconcile_with_exchange_position_price_mismatch(self, mock_datetime):
        """Test reconciliation when position prices don't match."""
        # Set up mock datetime
        now = datetime.now()
        mock_datetime.now.return_value = now
        
        # Create test positions
        self.position_manager.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            size=2.5,
            avg_price=45000.0,
            updated_at=now - timedelta(hours=1),
            exchange_id="exchange1"
        )
        
        # Create mock exchange gateway with different price
        mock_exchange = Mock()
        mock_exchange.get_all_positions.return_value = {
            "BTC-USD": {
                "size": 2.5,
                "avg_price": 46000.0,  # Different price (more than 0.5% difference)
                "exchange_id": "exchange1"
            }
        }
        
        # Reconcile
        results = self.position_manager.reconcile_with_exchange(mock_exchange)
        
        # Check results
        self.assertEqual(results["BTC-USD"], PositionReconciliationStatus.ADJUSTED)
        
        # Check position was adjusted
        position = self.position_manager.positions["BTC-USD"]
        self.assertEqual(position.avg_price, 46000.0)
    
    @patch('execution.exchange.risk.position_reconciliation.datetime')
    def test_reconcile_with_exchange_missing_internal_position(self, mock_datetime):
        """Test reconciliation when exchange has position not found internally."""
        # Set up mock datetime
        now = datetime.now()
        mock_datetime.now.return_value = now
        
        # Create mock exchange gateway with position not tracked internally
        mock_exchange = Mock()
        mock_exchange.get_all_positions.return_value = {
            "ETH-USD": {
                "size": 10.0,
                "avg_price": 2000.0,
                "exchange_id": "exchange1"
            }
        }
        
        # Reconcile
        results = self.position_manager.reconcile_with_exchange(mock_exchange)
        
        # Check results
        self.assertEqual(results["ETH-USD"], PositionReconciliationStatus.ADJUSTED)
        
        # Check position was added
        self.assertIn("ETH-USD", self.position_manager.positions)
        position = self.position_manager.positions["ETH-USD"]
        self.assertEqual(position.size, 10.0)
        self.assertEqual(position.avg_price, 2000.0)
        self.assertEqual(position.exchange_id, "exchange1")
    
    @patch('execution.exchange.risk.position_reconciliation.datetime')
    def test_reconcile_with_exchange_position_not_found_on_exchange(self, mock_datetime):
        """Test reconciliation when internal position not found on exchange."""
        # Set up mock datetime
        now = datetime.now()
        mock_datetime.now.return_value = now
        
        # Create test positions
        self.position_manager.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            size=2.5,
            avg_price=45000.0,
            updated_at=now - timedelta(hours=1),
            exchange_id="exchange1"
        )
        
        # Create mock exchange gateway with no positions
        mock_exchange = Mock()
        mock_exchange.get_all_positions.return_value = {}
        
        # Reconcile
        results = self.position_manager.reconcile_with_exchange(mock_exchange)
        
        # Check results
        self.assertEqual(results["BTC-USD"], PositionReconciliationStatus.ADJUSTED)
        
        # Check position was reset
        position = self.position_manager.positions["BTC-USD"]
        self.assertEqual(position.size, 0.0)
        self.assertEqual(position.avg_price, 0.0)
    
    @patch('execution.exchange.risk.position_reconciliation.datetime')
    def test_reconcile_with_exchange_skip_positions_with_open_orders(self, mock_datetime):
        """Test reconciliation skips positions with open orders."""
        # Set up mock datetime
        now = datetime.now()
        mock_datetime.now.return_value = now
        
        # Create test positions with open orders
        self.position_manager.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            size=2.5,
            avg_price=45000.0,
            updated_at=now - timedelta(hours=1),
            exchange_id="exchange1",
            open_orders=["order1"]
        )
        
        # Create mock exchange gateway with different size
        mock_exchange = Mock()
        mock_exchange.get_all_positions.return_value = {
            "BTC-USD": {
                "size": 3.0,  # Different size
                "avg_price": 45000.0,
                "exchange_id": "exchange1"
            }
        }
        
        # Reconcile
        results = self.position_manager.reconcile_with_exchange(mock_exchange)
        
        # Check results - should be marked as matched because it has open orders
        self.assertEqual(results["BTC-USD"], PositionReconciliationStatus.MATCHED)
        
        # Check position was NOT adjusted (still has original size)
        position = self.position_manager.positions["BTC-USD"]
        self.assertEqual(position.size, 2.5)
    
    @patch('execution.exchange.risk.position_reconciliation.datetime')
    def test_reconcile_with_exchange_exception(self, mock_datetime):
        """Test reconciliation when an exception occurs."""
        # Set up mock datetime
        now = datetime.now()
        mock_datetime.now.return_value = now
        
        # Create test positions
        self.position_manager.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            size=2.5,
            avg_price=45000.0,
            updated_at=now - timedelta(hours=1),
            exchange_id="exchange1"
        )
        
        # Create mock exchange gateway that raises exception
        mock_exchange = Mock()
        mock_exchange.get_all_positions.side_effect = Exception("Exchange error")
        
        # Reconcile
        results = self.position_manager.reconcile_with_exchange(mock_exchange)
        
        # Check results - should be all failed
        self.assertEqual(results["BTC-USD"], PositionReconciliationStatus.FAILED)
    
    def test_get_position_history(self):
        """Test getting position history."""
        # Create test position
        self.position_manager.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            size=2.5,
            avg_price=45000.0,
            updated_at=datetime.now(),
            exchange_id="exchange1"
        )
        
        # Add history entries
        now = time.time()
        self.position_manager.position_history["BTC-USD"] = [
            {
                "timestamp": now - 300,
                "old_size": 0.0,
                "new_size": 1.0,
                "price": 44000.0,
                "source": "order1"
            },
            {
                "timestamp": now - 200,
                "old_size": 1.0,
                "new_size": 2.0,
                "price": 45000.0,
                "source": "order2"
            },
            {
                "timestamp": now - 100,
                "old_size": 2.0,
                "new_size": 2.5,
                "price": 45000.0,
                "source": "order3"
            }
        ]
        
        # Get full history
        history = self.position_manager.get_position_history("BTC-USD")
        self.assertEqual(len(history), 3)
        
        # Get limited history
        history = self.position_manager.get_position_history("BTC-USD", limit=2)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["new_size"], 2.0)
        self.assertEqual(history[1]["new_size"], 2.5)
        
        # Get history for non-existent symbol
        history = self.position_manager.get_position_history("ETH-USD")
        self.assertEqual(history, [])
    
    def test_calculate_realized_pnl(self):
        """Test calculating realized P&L from fills."""
        # Create fills (FIFO order)
        fills = [
            {"side": "buy", "size": 1.0, "price": 40000.0, "timestamp": 1000},
            {"side": "buy", "size": 1.0, "price": 45000.0, "timestamp": 2000},
            {"side": "sell", "size": 1.5, "price": 50000.0, "timestamp": 3000},
            {"side": "buy", "size": 1.0, "price": 48000.0, "timestamp": 4000},
        ]
        
        # Calculate P&L
        pnl = self.position_manager.calculate_realized_pnl("BTC-USD", fills)
        
        # First sell closes 1.0 @ 40000 -> 50000 = profit of 10000
        # Then sells 0.5 @ 45000 -> 50000 = profit of 2500
        # Total profit = 12500
        self.assertEqual(pnl, 12500.0)
        
        # Test with empty fills
        pnl = self.position_manager.calculate_realized_pnl("BTC-USD", [])
        self.assertEqual(pnl, 0.0)
        
        # Test with only buys
        fills = [
            {"side": "buy", "size": 1.0, "price": 40000.0, "timestamp": 1000},
            {"side": "buy", "size": 1.0, "price": 45000.0, "timestamp": 2000},
        ]
        pnl = self.position_manager.calculate_realized_pnl("BTC-USD", fills)
        self.assertEqual(pnl, 0.0)
        
        # Test with short position
        fills = [
            {"side": "sell", "size": 1.0, "price": 50000.0, "timestamp": 1000},
            {"side": "buy", "size": 0.5, "price": 45000.0, "timestamp": 2000},
        ]
        # Profit = 0.5 * (50000 - 45000) = 2500
        pnl = self.position_manager.calculate_realized_pnl("BTC-USD", fills)
        self.assertEqual(pnl, 2500.0)
    
    def test_reset(self):
        """Test resetting position manager state."""
        # Create test positions and history
        self.position_manager.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            size=2.5,
            avg_price=45000.0,
            updated_at=datetime.now(),
            exchange_id="exchange1"
        )
        self.position_manager.position_history["BTC-USD"] = [
            {
                "timestamp": time.time(),
                "old_size": 0.0,
                "new_size": 2.5,
                "price": 45000.0,
                "source": "order1"
            }
        ]
        self.position_manager.last_reconciliation_time = datetime.now()
        
        # Reset state
        self.position_manager.reset()
        
        # Check everything was cleared
        self.assertEqual(self.position_manager.positions, {})
        self.assertEqual(self.position_manager.position_history, {})
        self.assertIsNone(self.position_manager.last_reconciliation_time)
    
    def test_calculate_portfolio_metrics(self):
        """Test calculating portfolio metrics."""
        # Create test positions
        self.position_manager.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            size=2.5,
            avg_price=45000.0,
            updated_at=datetime.now(),
            exchange_id="exchange1"
        )
        self.position_manager.positions["ETH-USD"] = Position(
            symbol="ETH-USD",
            size=-10.0,
            avg_price=2000.0,
            updated_at=datetime.now(),
            exchange_id="exchange1"
        )
        self.position_manager.positions["XRP-USD"] = Position(
            symbol="XRP-USD",
            size=0.0,
            avg_price=1.0,
            updated_at=datetime.now(),
            exchange_id="exchange1"
        )
        
        # Calculate metrics without price provider
        metrics = self.position_manager.calculate_portfolio_metrics()
        
        # Check metrics
        self.assertEqual(metrics["position_count"], 2)  # XRP-USD is zero size
        self.assertEqual(set(metrics["symbols"]), {"BTC-USD", "ETH-USD"})
        
        # Check values
        self.assertEqual(metrics["total_long_value"], 2.5 * 45000.0)
        self.assertEqual(metrics["total_short_value"], 10.0 * 2000.0)
        self.assertEqual(metrics["total_exposure"], (2.5 * 45000.0) + (10.0 * 2000.0))
        self.assertEqual(metrics["net_exposure"], (2.5 * 45000.0) - (10.0 * 2000.0))
        self.assertEqual(metrics["exposure_ratio"], (10.0 * 2000.0) / (2.5 * 45000.0))
        
        # Calculate metrics with price provider
        mock_price_provider = lambda symbol: {
            "BTC-USD": 50000.0,
            "ETH-USD": 2500.0
        }.get(symbol, 0.0)
        
        metrics = self.position_manager.calculate_portfolio_metrics(mock_price_provider)
        
        # Check values with updated prices
        self.assertEqual(metrics["total_long_value"], 2.5 * 50000.0)
        self.assertEqual(metrics["total_short_value"], 10.0 * 2500.0)
        self.assertEqual(metrics["total_exposure"], (2.5 * 50000.0) + (10.0 * 2500.0))
        self.assertEqual(metrics["net_exposure"], (2.5 * 50000.0) - (10.0 * 2500.0))
        self.assertEqual(metrics["exposure_ratio"], (10.0 * 2500.0) / (2.5 * 50000.0))
    
    def test_add_to_history_limit(self):
        """Test that history entries are limited."""
        # Set small history limit
        self.position_manager.config["max_position_history"] = 3
        
        # Create test position
        self.position_manager.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            size=1.0,
            avg_price=45000.0,
            updated_at=datetime.now(),
            exchange_id="exchange1"
        )
        
        # Add multiple history entries
        for i in range(5):
            self.position_manager._add_to_history(
                "BTC-USD", 
                1.0 + i, 
                1.0 + i + 1, 
                45000.0, 
                f"order{i}"
            )
        
        # Check history was limited to 3 entries
        history = self.position_manager.position_history["BTC-USD"]
        self.assertEqual(len(history), 3)
        
        # Check we kept the most recent entries
        self.assertEqual(history[0]["source"], "order2")
        self.assertEqual(history[1]["source"], "order3")
        self.assertEqual(history[2]["source"], "order4")
    
    def test_publish_position_event(self):
        """Test publishing position update event."""
        # Publish event
        self.position_manager._publish_position_event(
            "BTC-USD", 
            2.5, 
            45000.0, 
            "test_source"
        )
        
        # Check event bus was called
        self.event_bus_mock.publish.assert_called_once()
        
        # Get event data
        event = self.event_bus_mock.publish.call_args[0][0]
        
        # Check event data
        self.assertEqual(event.topic, "position_updated")
        self.assertEqual(event.data["symbol"], "BTC-USD")
        self.assertEqual(event.data["size"], 2.5)
        self.assertEqual(event.data["avg_price"], 45000.0)
        self.assertEqual(event.data["value"], 2.5 * 45000.0)
        self.assertEqual(event.data["source"], "test_source")
        self.assertIn("timestamp", event.data)


if __name__ == "__main__":
    unittest.main()