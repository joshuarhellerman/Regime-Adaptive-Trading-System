"""
Tests for the OrderFactory class.

This module tests the functionality of the OrderFactory class, ensuring that
all order creation methods work correctly with appropriate parameters.
"""

import unittest
from unittest.mock import patch
import logging
from datetime import datetime

from execution.exchange.order.order import (
    Order, OrderType, OrderSide, OrderStatus, TimeInForce
)
from execution.exchange.order.order_factory import OrderFactory


class TestOrderFactory(unittest.TestCase):
    """Test cases for the OrderFactory class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.factory = OrderFactory()

        # Configure with some defaults for testing
        self.factory.configure(
            default_exchange_id="binance",
            default_account="test_account",
            default_time_in_force=TimeInForce.GTC,
            default_tags=["test", "automated"],
            default_params={"test_param": "test_value"}
        )

    def test_init(self):
        """Test OrderFactory initialization."""
        factory = OrderFactory()
        self.assertIsNone(factory.default_exchange_id)
        self.assertIsNone(factory.default_account)
        self.assertEqual(factory.default_time_in_force, TimeInForce.GTC)
        self.assertEqual(factory.default_tags, [])
        self.assertEqual(factory.default_params, {})

    def test_configure(self):
        """Test OrderFactory configuration."""
        factory = OrderFactory()

        # Configure with new values
        factory.configure(
            default_exchange_id="kraken",
            default_account="main_account",
            default_time_in_force=TimeInForce.IOC,
            default_tags=["production"],
            default_params={"client_id": "test"}
        )

        # Check if configuration was applied correctly
        self.assertEqual(factory.default_exchange_id, "kraken")
        self.assertEqual(factory.default_account, "main_account")
        self.assertEqual(factory.default_time_in_force, TimeInForce.IOC)
        self.assertEqual(factory.default_tags, ["production"])
        self.assertEqual(factory.default_params, {"client_id": "test"})

        # Test partial configuration
        factory.configure(default_exchange_id="coinbase")
        self.assertEqual(factory.default_exchange_id, "coinbase")
        self.assertEqual(factory.default_account, "main_account")  # Unchanged

    def test_prepare_common_params(self):
        """Test _prepare_common_params method."""
        # Test with empty kwargs
        params = self.factory._prepare_common_params()
        self.assertEqual(params["exchange_id"], "binance")
        self.assertEqual(params["exchange_account"], "test_account")
        self.assertEqual(params["time_in_force"], TimeInForce.GTC)
        self.assertEqual(set(params["tags"]), {"test", "automated"})
        self.assertEqual(params["params"], {"test_param": "test_value"})

        # Test with overriding kwargs
        params = self.factory._prepare_common_params(
            exchange_id="override",
            exchange_account="override_account",
            time_in_force=TimeInForce.FOK,
            tags=["custom"],
            params={"custom_param": "custom_value"}
        )
        self.assertEqual(params["exchange_id"], "override")
        self.assertEqual(params["exchange_account"], "override_account")
        self.assertEqual(params["time_in_force"], TimeInForce.FOK)
        self.assertEqual(set(params["tags"]), {"custom", "test", "automated"})
        self.assertEqual(params["params"], {
            "test_param": "test_value",
            "custom_param": "custom_value"
        })

    def test_create_market_order(self):
        """Test creating market orders."""
        # Test with enum side
        order = self.factory.create_market_order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=1.5
        )

        self.assertEqual(order.symbol, "BTC/USD")
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertEqual(order.quantity, 1.5)
        self.assertEqual(order.order_type, OrderType.MARKET)
        self.assertEqual(order.exchange_id, "binance")
        self.assertEqual(order.exchange_account, "test_account")

        # Test with string side
        order = self.factory.create_market_order(
            symbol="ETH/USD",
            side="sell",
            quantity=2.0,
            exchange_id="custom_exchange"  # Override default
        )

        self.assertEqual(order.symbol, "ETH/USD")
        self.assertEqual(order.side, OrderSide.SELL)
        self.assertEqual(order.quantity, 2.0)
        self.assertEqual(order.exchange_id, "custom_exchange")

    def test_create_limit_order(self):
        """Test creating limit orders."""
        order = self.factory.create_limit_order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000.0
        )

        self.assertEqual(order.symbol, "BTC/USD")
        self.assertEqual(order.side, OrderSide.BUY)
        self.assertEqual(order.quantity, 1.0)
        self.assertEqual(order.order_type, OrderType.LIMIT)
        self.assertEqual(order.price, 50000.0)

        # Test with custom parameters
        order = self.factory.create_limit_order(
            symbol="ETH/USD",
            side="sell",
            quantity=5.0,
            price=3000.0,
            time_in_force=TimeInForce.IOC
        )

        self.assertEqual(order.symbol, "ETH/USD")
        self.assertEqual(order.side, OrderSide.SELL)
        self.assertEqual(order.time_in_force, TimeInForce.IOC)
        self.assertEqual(order.price, 3000.0)

    def test_create_stop_order(self):
        """Test creating stop orders."""
        order = self.factory.create_stop_order(
            symbol="BTC/USD",
            side=OrderSide.SELL,
            quantity=1.0,
            stop_price=45000.0
        )

        self.assertEqual(order.symbol, "BTC/USD")
        self.assertEqual(order.side, OrderSide.SELL)
        self.assertEqual(order.quantity, 1.0)
        self.assertEqual(order.order_type, OrderType.STOP)
        self.assertEqual(order.stop_price, 45000.0)
        self.assertNotIn("price", vars(order))

    def test_create_stop_limit_order(self):
        """Test creating stop-limit orders."""
        order = self.factory.create_stop_limit_order(
            symbol="BTC/USD",
            side=OrderSide.SELL,
            quantity=1.0,
            stop_price=45000.0,
            limit_price=44000.0
        )

        self.assertEqual(order.symbol, "BTC/USD")
        self.assertEqual(order.side, OrderSide.SELL)
        self.assertEqual(order.quantity, 1.0)
        self.assertEqual(order.order_type, OrderType.STOP_LIMIT)
        self.assertEqual(order.stop_price, 45000.0)
        self.assertEqual(order.price, 44000.0)

    def test_create_trailing_stop_order(self):
        """Test creating trailing stop orders."""
        # Test with trail_amount
        order = self.factory.create_trailing_stop_order(
            symbol="BTC/USD",
            side=OrderSide.SELL,
            quantity=1.0,
            trail_amount=1000.0
        )

        self.assertEqual(order.symbol, "BTC/USD")
        self.assertEqual(order.side, OrderSide.SELL)
        self.assertEqual(order.quantity, 1.0)
        self.assertEqual(order.order_type, OrderType.TRAILING_STOP)
        self.assertEqual(order.params["trail_amount"], 1000.0)
        self.assertNotIn("trail_percent", order.params)

        # Test with trail_percent
        order = self.factory.create_trailing_stop_order(
            symbol="ETH/USD",
            side="sell",
            quantity=5.0,
            trail_percent=2.5
        )

        self.assertEqual(order.symbol, "ETH/USD")
        self.assertEqual(order.params["trail_percent"], 2.5)
        self.assertNotIn("trail_amount", order.params)

        # Test with both parameters
        order = self.factory.create_trailing_stop_order(
            symbol="LTC/USD",
            side="buy",
            quantity=10.0,
            trail_amount=50.0,
            trail_percent=1.0
        )

        self.assertEqual(order.params["trail_amount"], 50.0)
        self.assertEqual(order.params["trail_percent"], 1.0)

        # Test with missing trailing parameters
        with self.assertRaises(ValueError):
            self.factory.create_trailing_stop_order(
                symbol="BTC/USD",
                side=OrderSide.SELL,
                quantity=1.0
            )

    def test_create_order_generic(self):
        """Test the generic create_order method."""
        # Test market order
        order = self.factory.create_order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=1.0,
            order_type=OrderType.MARKET
        )

        self.assertEqual(order.order_type, OrderType.MARKET)

        # Test limit order
        order = self.factory.create_order(
            symbol="BTC/USD",
            side="buy",
            quantity=1.0,
            order_type="limit",
            price=50000.0
        )

        self.assertEqual(order.order_type, OrderType.LIMIT)
        self.assertEqual(order.price, 50000.0)

        # Test with missing required parameters
        with self.assertRaises(ValueError):
            self.factory.create_order(
                symbol="BTC/USD",
                side="buy",
                quantity=1.0,
                order_type=OrderType.LIMIT
                # Missing price
            )

        with self.assertRaises(ValueError):
            self.factory.create_order(
                symbol="BTC/USD",
                side="buy",
                quantity=1.0,
                order_type=OrderType.STOP
                # Missing stop_price
            )

        with self.assertRaises(ValueError):
            self.factory.create_order(
                symbol="BTC/USD",
                side="buy",
                quantity=1.0,
                order_type=OrderType.STOP_LIMIT,
                price=50000.0
                # Missing stop_price
            )

        with self.assertRaises(ValueError):
            self.factory.create_order(
                symbol="BTC/USD",
                side="buy",
                quantity=1.0,
                order_type=OrderType.TRAILING_STOP
                # Missing trail_amount or trail_percent
            )

    def test_create_bulk_orders(self):
        """Test creating multiple orders at once."""
        orders_data = [
            {
                "symbol": "BTC/USD",
                "side": "buy",
                "quantity": 1.0,
                "order_type": "market"
            },
            {
                "symbol": "ETH/USD",
                "side": "sell",
                "quantity": 5.0,
                "order_type": "limit",
                "price": 3000.0
            },
            {
                "symbol": "LTC/USD",
                "side": "sell",
                "quantity": 10.0,
                "order_type": "stop",
                "stop_price": 150.0
            }
        ]

        orders = self.factory.create_bulk_orders(orders_data)

        self.assertEqual(len(orders), 3)
        self.assertEqual(orders[0].symbol, "BTC/USD")
        self.assertEqual(orders[0].order_type, OrderType.MARKET)
        self.assertEqual(orders[1].symbol, "ETH/USD")
        self.assertEqual(orders[1].price, 3000.0)
        self.assertEqual(orders[2].symbol, "LTC/USD")
        self.assertEqual(orders[2].stop_price, 150.0)

        # Test with one invalid order
        orders_data.append({
            "symbol": "XRP/USD",
            "side": "buy",
            "quantity": 1000.0,
            "order_type": "limit"
            # Missing price
        })

        with patch.object(logging.getLogger('execution.exchange.order.order_factory'), 'error') as mock_logger:
            orders = self.factory.create_bulk_orders(orders_data)
            self.assertEqual(len(orders), 3)  # Only valid orders returned
            mock_logger.assert_called_once()  # Error was logged

    def test_order_with_custom_tags_and_params(self):
        """Test order creation with custom tags and parameters."""
        order = self.factory.create_market_order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=1.0,
            tags=["special"],
            params={"algo_id": "test123"}
        )

        # Should include both default and custom tags
        self.assertIn("test", order.tags)
        self.assertIn("automated", order.tags)
        self.assertIn("special", order.tags)

        # Should include both default and custom params
        self.assertEqual(order.params["test_param"], "test_value")
        self.assertEqual(order.params["algo_id"], "test123")


if __name__ == "__main__":
    unittest.main()