"""
Tests for the ExchangeGateway abstract base class.

This module tests the functionality of the ExchangeGateway class by creating
a mock implementation and verifying its behavior.
"""

import unittest
from unittest.mock import Mock, patch
import pytest
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from core.event_bus import EventBus
from execution.exchange.exchange_gateway import (
    ExchangeGateway,
    ExchangeType,
    ExchangeCapabilities
)


class MockExchangeGateway(ExchangeGateway):
    """Mock implementation of ExchangeGateway for testing"""

    def __init__(self, **kwargs):
        super().__init__(
            exchange_id="mock_exchange",
            exchange_name="Mock Exchange",
            exchange_type=ExchangeType.SPOT,
            capabilities=[
                ExchangeCapabilities.MARKET_ORDERS,
                ExchangeCapabilities.LIMIT_ORDERS,
                ExchangeCapabilities.STREAMING_QUOTES
            ],
            **kwargs
        )
        # Mock data for testing
        self._balances = {"USD": 10000.0, "EUR": 5000.0}
        self._positions = {"BTC/USD": 1.5, "ETH/USD": -0.5}
        self._orders = {}
        self._open_orders = []
        self._server_time = datetime.now()

    def connect(self) -> bool:
        self.connected = True
        self.last_connection_time = datetime.now()
        self.connection_error = None
        return True

    def disconnect(self) -> bool:
        self.connected = False
        return True

    def is_connected(self) -> bool:
        return super().is_connected()

    def place_order(self, order: Any) -> str:
        order_id = f"order_{len(self._orders) + 1}"
        self._orders[order_id] = order
        self._open_orders.append(order_id)
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._open_orders:
            self._open_orders.remove(order_id)
            return True
        return False

    def get_order(self, order_id: str) -> Optional[Any]:
        return self._orders.get(order_id)

    def get_open_orders(self) -> List[Any]:
        return [self._orders[order_id] for order_id in self._open_orders]

    def get_position(self, instrument: str) -> float:
        return self._positions.get(instrument, 0.0)

    def get_all_positions(self) -> Dict[str, float]:
        return self._positions.copy()

    def close_position(self, instrument: str, amount: Optional[float] = None) -> bool:
        if instrument in self._positions:
            if amount is None or abs(amount) >= abs(self._positions[instrument]):
                del self._positions[instrument]
            else:
                self._positions[instrument] -= amount
            return True
        return False

    def get_balance(self, currency: str) -> float:
        return self._balances.get(currency, 0.0)

    def get_all_balances(self) -> Dict[str, float]:
        return self._balances.copy()

    def get_ticker(self, instrument: str) -> Dict[str, Any]:
        return {
            "bid": 100.0,
            "ask": 101.0,
            "last": 100.5,
            "volume": 1000.0,
            "timestamp": datetime.now()
        }

    def get_instrument_details(self, instrument: str) -> Dict[str, Any]:
        base, quote = self.parse_instrument(instrument)
        return {
            "symbol": instrument,
            "base_currency": base,
            "quote_currency": quote,
            "min_size": 0.001,
            "max_size": 100.0,
            "price_precision": 2,
            "size_precision": 4
        }

    def get_orderbook(self, instrument: str, depth: int = 10) -> Dict[str, Any]:
        return {
            "bids": [[100.0, 1.5], [99.0, 2.5]],
            "asks": [[101.0, 1.0], [102.0, 2.0]],
            "timestamp": datetime.now()
        }

    def get_recent_trades(self, instrument: str, limit: int = 100) -> List[Dict[str, Any]]:
        return [
            {
                "id": "t1",
                "price": 100.5,
                "amount": 0.1,
                "side": "buy",
                "timestamp": datetime.now()
            },
            {
                "id": "t2",
                "price": 100.6,
                "amount": 0.2,
                "side": "sell",
                "timestamp": datetime.now()
            }
        ][:limit]

    def get_candles(
            self,
            instrument: str,
            timeframe: str,
            start: Optional[datetime] = None,
            end: Optional[datetime] = None,
            limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        return [
            {
                "timestamp": datetime.now(),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 100.0
            },
            {
                "timestamp": datetime.now(),
                "open": 100.5,
                "high": 102.0,
                "low": 100.0,
                "close": 101.5,
                "volume": 120.0
            }
        ][:limit]

    def get_exchange_info(self) -> Dict[str, Any]:
        return {
            "name": self.exchange_name,
            "id": self.exchange_id,
            "type": self.exchange_type.value,
            "capabilities": self.get_capabilities(),
            "trading_pairs": ["BTC/USD", "ETH/USD", "LTC/USD"],
            "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"]
        }

    def get_server_time(self) -> datetime:
        return self._server_time


class TestExchangeGateway(unittest.TestCase):
    """Test cases for ExchangeGateway"""

    def setUp(self):
        """Set up test fixtures"""
        self.event_bus = Mock(spec=EventBus)
        self.gateway = MockExchangeGateway(event_bus=self.event_bus)

    def test_init(self):
        """Test gateway initialization"""
        self.assertEqual(self.gateway.exchange_id, "mock_exchange")
        self.assertEqual(self.gateway.exchange_name, "Mock Exchange")
        self.assertEqual(self.gateway.exchange_type, ExchangeType.SPOT)
        self.assertEqual(len(self.gateway.capabilities), 3)
        self.assertFalse(self.gateway.connected)
        self.assertIsNone(self.gateway.last_connection_time)
        self.assertIsNone(self.gateway.connection_error)

    def test_connect_disconnect(self):
        """Test connection management"""
        self.assertFalse(self.gateway.connected)

        # Test connect
        result = self.gateway.connect()
        self.assertTrue(result)
        self.assertTrue(self.gateway.connected)
        self.assertIsNotNone(self.gateway.last_connection_time)

        # Test is_connected
        self.assertTrue(self.gateway.is_connected())

        # Test disconnect
        result = self.gateway.disconnect()
        self.assertTrue(result)
        self.assertFalse(self.gateway.connected)
        self.assertFalse(self.gateway.is_connected())

    def test_capabilities(self):
        """Test capability checking"""
        self.assertTrue(self.gateway.supports_capability(ExchangeCapabilities.MARKET_ORDERS))
        self.assertTrue(self.gateway.supports_capability(ExchangeCapabilities.LIMIT_ORDERS))
        self.assertFalse(self.gateway.supports_capability(ExchangeCapabilities.OCO_ORDERS))

        capabilities = self.gateway.get_capabilities()
        self.assertEqual(len(capabilities), 3)
        self.assertIn("market_orders", capabilities)
        self.assertIn("limit_orders", capabilities)
        self.assertIn("streaming_quotes", capabilities)

    def test_order_operations(self):
        """Test order management operations"""
        # Connect first
        self.gateway.connect()

        # Place an order
        test_order = {"symbol": "BTC/USD", "type": "limit", "side": "buy", "price": 10000, "amount": 0.1}
        order_id = self.gateway.place_order(test_order)
        self.assertTrue(order_id.startswith("order_"))

        # Get order details
        order = self.gateway.get_order(order_id)
        self.assertEqual(order, test_order)

        # Get open orders
        open_orders = self.gateway.get_open_orders()
        self.assertEqual(len(open_orders), 1)
        self.assertEqual(open_orders[0], test_order)

        # Cancel order
        result = self.gateway.cancel_order(order_id)
        self.assertTrue(result)

        # Verify cancellation
        open_orders = self.gateway.get_open_orders()
        self.assertEqual(len(open_orders), 0)

        # Cancel non-existent order
        result = self.gateway.cancel_order("non_existent_order")
        self.assertFalse(result)

    def test_position_operations(self):
        """Test position management operations"""
        # Get position
        position = self.gateway.get_position("BTC/USD")
        self.assertEqual(position, 1.5)

        # Get non-existent position
        position = self.gateway.get_position("XRP/USD")
        self.assertEqual(position, 0.0)

        # Get all positions
        positions = self.gateway.get_all_positions()
        self.assertEqual(len(positions), 2)
        self.assertEqual(positions["BTC/USD"], 1.5)
        self.assertEqual(positions["ETH/USD"], -0.5)

        # Close partial position
        result = self.gateway.close_position("BTC/USD", 0.5)
        self.assertTrue(result)
        position = self.gateway.get_position("BTC/USD")
        self.assertEqual(position, 1.0)

        # Close full position
        result = self.gateway.close_position("BTC/USD")
        self.assertTrue(result)
        position = self.gateway.get_position("BTC/USD")
        self.assertEqual(position, 0.0)

        # Close non-existent position
        result = self.gateway.close_position("XRP/USD")
        self.assertFalse(result)

    def test_balance_operations(self):
        """Test balance operations"""
        # Get balance
        balance = self.gateway.get_balance("USD")
        self.assertEqual(balance, 10000.0)

        # Get non-existent balance
        balance = self.gateway.get_balance("BTC")
        self.assertEqual(balance, 0.0)

        # Get all balances
        balances = self.gateway.get_all_balances()
        self.assertEqual(len(balances), 2)
        self.assertEqual(balances["USD"], 10000.0)
        self.assertEqual(balances["EUR"], 5000.0)

    def test_market_data_operations(self):
        """Test market data operations"""
        # Test get_ticker
        ticker = self.gateway.get_ticker("BTC/USD")
        self.assertIsInstance(ticker, dict)
        self.assertIn("bid", ticker)
        self.assertIn("ask", ticker)

        # Test get_instrument_details
        details = self.gateway.get_instrument_details("BTC/USD")
        self.assertEqual(details["symbol"], "BTC/USD")
        self.assertEqual(details["base_currency"], "BTC")
        self.assertEqual(details["quote_currency"], "USD")

        # Test get_orderbook
        orderbook = self.gateway.get_orderbook("BTC/USD")
        self.assertIn("bids", orderbook)
        self.assertIn("asks", orderbook)

        # Test get_recent_trades
        trades = self.gateway.get_recent_trades("BTC/USD", limit=1)
        self.assertEqual(len(trades), 1)
        self.assertIn("price", trades[0])

        # Test get_candles
        candles = self.gateway.get_candles("BTC/USD", "1h", limit=1)
        self.assertEqual(len(candles), 1)
        self.assertIn("open", candles[0])
        self.assertIn("close", candles[0])

    def test_exchange_info(self):
        """Test exchange information methods"""
        # Test get_exchange_info
        info = self.gateway.get_exchange_info()
        self.assertEqual(info["name"], "Mock Exchange")
        self.assertEqual(info["id"], "mock_exchange")

        # Test get_server_time
        time = self.gateway.get_server_time()
        self.assertIsInstance(time, datetime)

    def test_instrument_formatting(self):
        """Test instrument formatting methods"""
        # Test format_instrument
        instrument = self.gateway.format_instrument("BTC", "USD")
        self.assertEqual(instrument, "BTC/USD")

        # Test parse_instrument
        base, quote = self.gateway.parse_instrument("BTC/USD")
        self.assertEqual(base, "BTC")
        self.assertEqual(quote, "USD")

        # Test edge case
        base, quote = self.gateway.parse_instrument("BTCUSD")
        self.assertEqual(base, "BTCUSD")
        self.assertEqual(quote, "")

    def test_timeframe_conversion(self):
        """Test timeframe conversion"""
        timeframe = self.gateway.convert_timeframe("1h")
        self.assertEqual(timeframe, "1h")

    def test_standardize_instrument(self):
        """Test instrument standardization"""
        instrument = self.gateway.standardize_instrument("BTCUSD")
        self.assertEqual(instrument, "BTCUSD")

    def test_status(self):
        """Test status reporting"""
        # Test before connection
        status = self.gateway.get_status()
        self.assertEqual(status["exchange_id"], "mock_exchange")
        self.assertEqual(status["exchange_name"], "Mock Exchange")
        self.assertEqual(status["exchange_type"], "spot")
        self.assertFalse(status["connected"])

        # Test after connection
        self.gateway.connect()
        status = self.gateway.get_status()
        self.assertTrue(status["connected"])
        self.assertIsNotNone(status["last_connection_time"])

    def test_error_handling(self):
        """Test error handling methods"""
        # Test _handle_error
        error = Exception("Test error")
        self.gateway._handle_error("GET", "/api/test", error)
        self.assertEqual(self.gateway.connection_error, error)

    @patch("logging.Logger.debug")
    def test_logging(self, mock_debug):
        """Test logging methods"""
        # Test _log_request
        self.gateway._log_request("GET", "/api/test", {"param": "value"}, {"data": "test"})
        self.assertEqual(mock_debug.call_count, 3)

        # Test _log_response
        self.gateway._log_response({"result": "success"}, 123.45)
        self.assertEqual(mock_debug.call_count, 4)


if __name__ == "__main__":
    unittest.main()