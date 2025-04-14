import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from execution.exchange.order.order import (
    Order, OrderType, OrderSide, OrderStatus, TimeInForce
)

class TestOrderEnums:
    """Test cases for order-related enums"""
    
    def test_order_type_values(self):
        """Test OrderType enum values"""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        assert OrderType.STOP_LIMIT.value == "stop_limit"
        assert OrderType.TRAILING_STOP.value == "trailing_stop"
        assert OrderType.OCO.value == "oco"
        assert OrderType.ICEBERG.value == "iceberg"
        assert OrderType.FOK.value == "fok"
        assert OrderType.IOC.value == "ioc"
    
    def test_order_side_values(self):
        """Test OrderSide enum values and helper methods"""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"
        
        # Test helper methods
        assert OrderSide.BUY.is_buy() is True
        assert OrderSide.BUY.is_sell() is False
        assert OrderSide.SELL.is_buy() is False
        assert OrderSide.SELL.is_sell() is True
    
    def test_order_status_values(self):
        """Test OrderStatus enum values"""
        assert OrderStatus.CREATED.value == "created"
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.PENDING_CANCEL.value == "pending_cancel"
        assert OrderStatus.PENDING_UPDATE.value == "pending_update"
        assert OrderStatus.OPEN.value == "open"
        assert OrderStatus.PARTIALLY_FILLED.value == "partially_filled"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELED.value == "canceled"
        assert OrderStatus.REJECTED.value == "rejected"
        assert OrderStatus.EXPIRED.value == "expired"
        assert OrderStatus.ERROR.value == "error"
        assert OrderStatus.WORKING.value == "working"
    
    def test_time_in_force_values(self):
        """Test TimeInForce enum values"""
        assert TimeInForce.GTC.value == "gtc"
        assert TimeInForce.IOC.value == "ioc"
        assert TimeInForce.FOK.value == "fok"
        assert TimeInForce.DAY.value == "day"
        assert TimeInForce.GTD.value == "gtd"


class TestOrderCreation:
    """Test cases for order creation and validation"""
    
    def test_create_market_order(self):
        """Test creating a basic market order"""
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=1.5,
            order_type=OrderType.MARKET
        )
        
        assert order.symbol == "BTC/USD"
        assert order.side == OrderSide.BUY
        assert order.quantity == 1.5
        assert order.order_type == OrderType.MARKET
        assert order.status == OrderStatus.CREATED
        assert order.time_in_force == TimeInForce.GTC
        assert order.filled_quantity == 0.0
        assert order.average_price is None
        assert order.price is None
        assert order.stop_price is None
        assert isinstance(order.order_id, str)
        assert isinstance(order.creation_time, datetime)
        assert order.update_time == order.creation_time
        assert order.exchange_order_id is None
        assert order.params == {}
        assert order.tags == []
    
    def test_create_limit_order(self):
        """Test creating a limit order"""
        order = Order(
            symbol="ETH/USD",
            side=OrderSide.SELL,
            quantity=2.0,
            order_type=OrderType.LIMIT,
            price=3000.0
        )
        
        assert order.symbol == "ETH/USD"
        assert order.side == OrderSide.SELL
        assert order.quantity == 2.0
        assert order.order_type == OrderType.LIMIT
        assert order.price == 3000.0
        assert order.status == OrderStatus.CREATED
    
    def test_create_stop_order(self):
        """Test creating a stop order"""
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.SELL,
            quantity=0.5,
            order_type=OrderType.STOP,
            stop_price=40000.0
        )
        
        assert order.symbol == "BTC/USD"
        assert order.side == OrderSide.SELL
        assert order.quantity == 0.5
        assert order.order_type == OrderType.STOP
        assert order.stop_price == 40000.0
        assert order.status == OrderStatus.CREATED
    
    def test_create_stop_limit_order(self):
        """Test creating a stop limit order"""
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.SELL,
            quantity=0.5,
            order_type=OrderType.STOP_LIMIT,
            price=39500.0,
            stop_price=40000.0
        )
        
        assert order.symbol == "BTC/USD"
        assert order.side == OrderSide.SELL
        assert order.quantity == 0.5
        assert order.order_type == OrderType.STOP_LIMIT
        assert order.price == 39500.0
        assert order.stop_price == 40000.0
        assert order.status == OrderStatus.CREATED
    
    def test_create_order_with_optional_fields(self):
        """Test creating an order with additional optional fields"""
        creation_time = datetime.utcnow() - timedelta(minutes=5)
        order = Order(
            symbol="ETH/BTC",
            side=OrderSide.BUY,
            quantity=10.0,
            order_type=OrderType.LIMIT,
            price=0.07,
            time_in_force=TimeInForce.DAY,
            client_order_id="client123",
            exchange_id="binance",
            exchange_account="main",
            strategy_id="trend_following",
            params={"post_only": True},
            tags=["automated", "strategy1"],
            creation_time=creation_time
        )
        
        assert order.symbol == "ETH/BTC"
        assert order.time_in_force == TimeInForce.DAY
        assert order.client_order_id == "client123"
        assert order.exchange_id == "binance"
        assert order.exchange_account == "main"
        assert order.strategy_id == "trend_following"
        assert order.params == {"post_only": True}
        assert order.tags == ["automated", "strategy1"]
        assert order.creation_time == creation_time
        assert order.update_time == creation_time


class TestOrderValidation:
    """Test cases for order validation"""
    
    def test_limit_order_requires_price(self):
        """Test that limit orders require a price"""
        with pytest.raises(ValueError, match="Price must be specified for limit orders"):
            Order(
                symbol="BTC/USD",
                side=OrderSide.BUY,
                quantity=1.0,
                order_type=OrderType.LIMIT,
                price=None
            )
    
    def test_stop_order_requires_stop_price(self):
        """Test that stop orders require a stop price"""
        with pytest.raises(ValueError, match="Stop price must be specified for stop orders"):
            Order(
                symbol="BTC/USD",
                side=OrderSide.SELL,
                quantity=1.0,
                order_type=OrderType.STOP
            )
    
    def test_stop_limit_order_requires_prices(self):
        """Test that stop limit orders require both prices"""
        with pytest.raises(ValueError, match="Price must be specified for stop_limit orders"):
            Order(
                symbol="BTC/USD",
                side=OrderSide.SELL,
                quantity=1.0,
                order_type=OrderType.STOP_LIMIT,
                stop_price=40000.0
            )
            
        with pytest.raises(ValueError, match="Stop price must be specified for stop_limit orders"):
            Order(
                symbol="BTC/USD",
                side=OrderSide.SELL,
                quantity=1.0,
                order_type=OrderType.STOP_LIMIT,
                price=39500.0
            )


class TestOrderProperties:
    """Test cases for order properties"""
    
    def test_is_active_property(self):
        """Test is_active property"""
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=1.0,
            order_type=OrderType.MARKET
        )
        
        # Test active states
        active_statuses = [
            OrderStatus.CREATED,
            OrderStatus.PENDING,
            OrderStatus.OPEN,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.WORKING,
            OrderStatus.PENDING_UPDATE
        ]
        
        for status in active_statuses:
            order.status = status
            assert order.is_active is True
        
        # Test inactive states
        inactive_statuses = [
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.ERROR
        ]
        
        for status in inactive_statuses:
            order.status = status
            assert order.is_active is False
    
    def test_is_complete_property(self):
        """Test is_complete property"""
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=1.0,
            order_type=OrderType.MARKET
        )
        
        # Test incomplete states
        incomplete_statuses = [
            OrderStatus.CREATED,
            OrderStatus.PENDING,
            OrderStatus.OPEN,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.WORKING,
            OrderStatus.PENDING_UPDATE
        ]
        
        for status in incomplete_statuses:
            order.status = status
            assert order.is_complete is False
        
        # Test complete states
        complete_statuses = [
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.ERROR
        ]
        
        for status in complete_statuses:
            order.status = status
            assert order.is_complete is True
    
    def test_remaining_quantity_property(self):
        """Test remaining_quantity property"""
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=2.0,
            order_type=OrderType.MARKET
        )
        
        assert order.remaining_quantity == 2.0
        
        order.filled_quantity = 0.5
        assert order.remaining_quantity == 1.5
        
        order.filled_quantity = 2.0
        assert order.remaining_quantity == 0.0
    
    def test_fill_percentage_property(self):
        """Test fill_percentage property"""
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=2.0,
            order_type=OrderType.MARKET
        )
        
        assert order.fill_percentage == 0.0
        
        order.filled_quantity = 0.5
        assert order.fill_percentage == 25.0
        
        order.filled_quantity = 1.0
        assert order.fill_percentage == 50.0
        
        order.filled_quantity = 2.0
        assert order.fill_percentage == 100.0
        
        # Test edge case with zero quantity
        order.quantity = 0.0
        assert order.fill_percentage == 0.0


class TestOrderSerialization:
    """Test cases for order serialization and deserialization"""
    
    def test_to_dict(self):
        """Test converting order to dictionary"""
        creation_time = datetime(2023, 1, 1, 12, 0, 0)
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=1.5,
            order_type=OrderType.LIMIT,
            price=50000.0,
            time_in_force=TimeInForce.GTC,
            client_order_id="test123",
            exchange_id="binance",
            strategy_id="trend_following",
            tags=["test", "automated"],
            creation_time=creation_time
        )
        
        order_dict = order.to_dict()
        
        assert order_dict["symbol"] == "BTC/USDT"
        assert order_dict["side"] == "buy"
        assert order_dict["quantity"] == 1.5
        assert order_dict["order_type"] == "limit"
        assert order_dict["price"] == 50000.0
        assert order_dict["time_in_force"] == "gtc"
        assert order_dict["status"] == "created"
        assert order_dict["client_order_id"] == "test123"
        assert order_dict["exchange_id"] == "binance"
        assert order_dict["strategy_id"] == "trend_following"
        assert order_dict["tags"] == ["test", "automated"]
        assert order_dict["creation_time"] == "2023-01-01T12:00:00"
    
    def test_from_dict(self):
        """Test creating order from dictionary"""
        order_dict = {
            "symbol": "ETH/USDT",
            "side": "sell",
            "quantity": 2.0,
            "order_type": "limit",
            "price": 3000.0,
            "status": "open",
            "time_in_force": "day",
            "order_id": "test-order-id",
            "exchange_order_id": "exchange-order-id",
            "creation_time": "2023-02-01T10:00:00",
            "update_time": "2023-02-01T10:05:00",
            "filled_quantity": 1.0,
            "average_price": 3010.0,
            "exchange_id": "kraken",
            "tags": ["test"]
        }
        
        order = Order.from_dict(order_dict)
        
        assert order.symbol == "ETH/USDT"
        assert order.side == OrderSide.SELL
        assert order.quantity == 2.0
        assert order.order_type == OrderType.LIMIT
        assert order.price == 3000.0
        assert order.status == OrderStatus.OPEN
        assert order.time_in_force == TimeInForce.DAY
        assert order.order_id == "test-order-id"
        assert order.exchange_order_id == "exchange-order-id"
        assert order.creation_time == datetime(2023, 2, 1, 10, 0, 0)
        assert order.update_time == datetime(2023, 2, 1, 10, 5, 0)
        assert order.filled_quantity == 1.0
        assert order.average_price == 3010.0
        assert order.exchange_id == "kraken"
        assert order.tags == ["test"]
    
    def test_round_trip_serialization(self):
        """Test round-trip serialization (to_dict -> from_dict)"""
        original_order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=1.0,
            order_type=OrderType.LIMIT,
            price=45000.0,
            time_in_force=TimeInForce.GTC,
            client_order_id="round-trip-test",
            exchange_id="gemini",
            strategy_id="mean_reversion",
            tags=["test", "round-trip"]
        )
        
        # Serialize and then deserialize
        order_dict = original_order.to_dict()
        reconstructed_order = Order.from_dict(order_dict)
        
        # Verify all key attributes match
        assert reconstructed_order.symbol == original_order.symbol
        assert reconstructed_order.side == original_order.side
        assert reconstructed_order.quantity == original_order.quantity
        assert reconstructed_order.order_type == original_order.order_type
        assert reconstructed_order.price == original_order.price
        assert reconstructed_order.time_in_force == original_order.time_in_force
        assert reconstructed_order.status == original_order.status
        assert reconstructed_order.order_id == original_order.order_id
        assert reconstructed_order.client_order_id == original_order.client_order_id
        assert reconstructed_order.exchange_id == original_order.exchange_id
        assert reconstructed_order.strategy_id == original_order.strategy_id
        assert reconstructed_order.tags == original_order.tags


class TestOrderCopy:
    """Test cases for order copying"""
    
    def test_copy_method(self):
        """Test that the copy method creates an independent copy"""
        original_order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=1.0,
            order_type=OrderType.MARKET,
            client_order_id="original",
            params={"post_only": True},
            tags=["original"]
        )
        
        copied_order = original_order.copy()
        
        # Verify all attributes are the same
        assert copied_order.symbol == original_order.symbol
        assert copied_order.side == original_order.side
        assert copied_order.quantity == original_order.quantity
        assert copied_order.order_type == original_order.order_type
        assert copied_order.status == original_order.status
        assert copied_order.client_order_id == original_order.client_order_id
        assert copied_order.params == original_order.params
        assert copied_order.tags == original_order.tags
        
        # Verify it's a different object
        assert id(copied_order) != id(original_order)
        
        # Verify that modifying the copy doesn't affect the original
        copied_order.quantity = 2.0
        copied_order.status = OrderStatus.OPEN
        copied_order.client_order_id = "modified"
        copied_order.params["post_only"] = False
        copied_order.tags.append("modified")
        
        assert original_order.quantity == 1.0
        assert original_order.status == OrderStatus.CREATED
        assert original_order.client_order_id == "original"
        assert original_order.params == {"post_only": True}
        assert original_order.tags == ["original"]


class TestOrderLogging:
    """Test cases for order logging"""
    
    @patch('execution.exchange.order.order.logger')
    def test_order_creation_logging(self, mock_logger):
        """Test that order creation is logged"""
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=1.0,
            order_type=OrderType.MARKET
        )
        
        mock_logger.info.assert_called_once()
        log_msg = mock_logger.info.call_args[0][0]
        
        # Verify the log message contains key information
        assert "Created market buy order" in log_msg
        assert "BTC/USD" in log_msg
        assert "quantity=1.0" in log_msg
        assert order.order_id in log_msg


class TestOrderStringRepresentation:
    """Test cases for order string representation"""
    
    def test_str_method(self):
        """Test the string representation of an order"""
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=1.5,
            order_type=OrderType.LIMIT,
            price=50000.0
        )
        
        order_str = str(order)
        
        assert "Limit buy order for BTC/USD" in order_str
        assert "0/1.5 filled" in order_str
        assert "status=created" in order_str
        
        # Update the order status and filled amount
        order.status = OrderStatus.PARTIALLY_FILLED
        order.filled_quantity = 0.5
        order.average_price = 49500.0
        
        order_str = str(order)
        assert "0.5/1.5 filled @ 49500.0" in order_str
        assert "status=partially_filled" in order_str