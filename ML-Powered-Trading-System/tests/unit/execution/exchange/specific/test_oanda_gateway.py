"""
Test module for OandaGateway implementation.

This module tests the functionality of the OandaGateway class
for interacting with the Oanda forex broker API.
"""

import unittest
from unittest.mock import MagicMock, patch, ANY
import json
import datetime
import uuid
from decimal import Decimal

# Import from module being tested
from execution.exchange.exchange_specific.oanda_gateway import OandaGateway, OandaOrderType, OandaTimeInForce
from execution.order.order import Order, OrderStatus, OrderType, Side, TimeInForce
from execution.exchange.exchange_gateway import ExchangeType, ExchangeCapabilities


class TestOandaGateway(unittest.TestCase):
    """Test cases for the OandaGateway class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock dependencies
        self.mock_event_bus = MagicMock()
        self.mock_market_data_service = MagicMock()
        self.mock_state_manager = MagicMock()

        # Initialize gateway with mocked dependencies
        self.gateway = OandaGateway(
            api_key="test_api_key",
            account_id="test_account_id",
            base_url="https://api-fxtrade.oanda.com",
            practice=True,
            market_data_service=self.mock_market_data_service,
            state_manager=self.mock_state_manager,
            event_bus=self.mock_event_bus
        )

        # Mock requests.response for connection test
        self.mock_response = MagicMock()
        self.mock_response.status_code = 200
        self.mock_response.json.return_value = {"account": {"id": "test_account_id"}}

        # Override the send_request method to return our mock response
        self.gateway._send_request = MagicMock(return_value=self.mock_response)

        # Connect the gateway
        self.gateway.connect()

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        self.gateway.disconnect()

    def test_initialization(self):
        """Test proper initialization of OandaGateway."""
        self.assertEqual(self.gateway.exchange_id, "oanda")
        self.assertEqual(self.gateway.exchange_name, "OANDA")
        self.assertEqual(self.gateway.exchange_type, ExchangeType.SPOT)
        self.assertEqual(self.gateway.api_key, "test_api_key")
        self.assertEqual(self.gateway.account_id, "test_account_id")
        self.assertEqual(self.gateway.base_url, "https://api-fxtrade.oanda.com")
        self.assertEqual(self.gateway.practice, True)

        # Check that capabilities are properly set
        self.assertIn(ExchangeCapabilities.MARKET_ORDERS, self.gateway.capabilities)
        self.assertIn(ExchangeCapabilities.LIMIT_ORDERS, self.gateway.capabilities)
        self.assertIn(ExchangeCapabilities.STOP_ORDERS, self.gateway.capabilities)

    def test_connect(self):
        """Test connection to Oanda API."""
        # Reset connection state for this test
        self.gateway.connected = False

        # Mock a successful response
        self.mock_response.status_code = 200
        self.mock_response.json.return_value = {"account": {"id": "test_account_id"}}

        result = self.gateway.connect()

        # Verify connection was successful
        self.assertTrue(result)
        self.assertTrue(self.gateway.connected)

        # Verify API call
        self.gateway._send_request.assert_called_with(
            "GET", self.gateway.account_endpoint
        )

    def test_connect_failure(self):
        """Test connection failure handling."""
        # Reset connection state for this test
        self.gateway.connected = False

        # Mock a failed response
        self.mock_response.status_code = 401
        self.mock_response.text = "Unauthorized"

        result = self.gateway.connect()

        # Verify connection failed
        self.assertFalse(result)
        self.assertFalse(self.gateway.connected)

    def test_disconnect(self):
        """Test disconnection from Oanda API."""
        # Ensure we start connected
        self.gateway.connected = True
        self.gateway.running = True

        # Create a mock thread that returns immediately
        self.gateway.update_thread = MagicMock()

        result = self.gateway.disconnect()

        # Verify disconnection was successful
        self.assertTrue(result)
        self.assertFalse(self.gateway.connected)
        self.assertFalse(self.gateway.running)
        self.gateway.update_thread.join.assert_called_once()

    def test_place_market_order(self):
        """Test placing a market order."""
        # Create test order
        order = Order(
            id="test_order_1",
            instrument="EUR/USD",
            quantity=1000,
            side=Side.BUY,
            type=OrderType.MARKET,
            price=None
        )

        # Mock a successful order placement response
        self.mock_response.status_code = 201
        self.mock_response.json.return_value = {
            "orderCreateTransaction": {
                "id": "123456",
                "type": "MARKET_ORDER",
                "instrument": "EUR_USD",
                "units": "1000",
                "time": "2023-01-01T12:00:00Z",
                "reason": "CLIENT_ORDER"
            }
        }

        # Place the order
        result = self.gateway.place_order(order)

        # Verify the order was placed successfully
        self.assertEqual(result, "123456")
        self.assertEqual(order.exchange_order_id, "123456")
        self.assertEqual(order.status, OrderStatus.NEW)

        # Verify the order is stored
        self.assertIn("123456", self.gateway.open_orders)
        self.assertEqual(self.gateway.order_id_map[order.id], "123456")

        # Verify the event was emitted
        self.mock_event_bus.emit.assert_called_once()

    def test_place_order_failure(self):
        """Test handling of failed order placement."""
        # Create test order
        order = Order(
            id="test_order_2",
            instrument="EUR/USD",
            quantity=1000,
            side=Side.BUY,
            type=OrderType.MARKET,
            price=None
        )

        # Mock a failed order placement response
        self.mock_response.status_code = 400
        self.mock_response.text = "Invalid order"

        # Attempt to place the order and expect an exception
        with self.assertRaises(Exception):
            self.gateway.place_order(order)

        # Verify the order status was updated
        self.assertEqual(order.status, OrderStatus.REJECTED)

        # Verify the event was emitted
        self.mock_event_bus.emit.assert_called_once()

    def test_place_limit_order(self):
        """Test placing a limit order."""
        # Create test order
        order = Order(
            id="test_order_3",
            instrument="EUR/USD",
            quantity=1000,
            side=Side.BUY,
            type=OrderType.LIMIT,
            price=Decimal("1.10000")
        )

        # Mock a successful order placement response
        self.mock_response.status_code = 201
        self.mock_response.json.return_value = {
            "orderCreateTransaction": {
                "id": "123457",
                "type": "LIMIT_ORDER",
                "instrument": "EUR_USD",
                "units": "1000",
                "price": "1.10000",
                "time": "2023-01-01T12:00:00Z",
                "reason": "CLIENT_ORDER"
            }
        }

        # Place the order
        result = self.gateway.place_order(order)

        # Verify the order was placed successfully
        self.assertEqual(result, "123457")

        # Verify the API call included the limit price
        expected_order_data = {
            "order": {
                "type": "LIMIT",
                "instrument": "EUR_USD",
                "units": "1000",
                "timeInForce": "GTC",
                "positionFill": "DEFAULT",
                "price": "1.10000",
                "clientExtensions": {
                    "id": "test_order_3",
                    "tag": "system"
                }
            }
        }

        self.gateway._send_request.assert_called_with(
            "POST",
            self.gateway.orders_endpoint,
            json=ANY
        )

        # Extract the JSON argument from the call
        actual_json = self.gateway._send_request.call_args[1]["json"]

        # Compare the relevant parts
        self.assertEqual(actual_json["order"]["type"], expected_order_data["order"]["type"])
        self.assertEqual(actual_json["order"]["instrument"], expected_order_data["order"]["instrument"])
        self.assertEqual(actual_json["order"]["units"], expected_order_data["order"]["units"])
        self.assertEqual(actual_json["order"]["price"], expected_order_data["order"]["price"])

    def test_cancel_order(self):
        """Test cancelling an order."""
        # Create and place a test order first
        order = Order(
            id="test_order_4",
            instrument="EUR/USD",
            quantity=1000,
            side=Side.BUY,
            type=OrderType.LIMIT,
            price=Decimal("1.10000")
        )

        # Mock an order placement
        order.exchange_order_id = "123458"
        self.gateway.open_orders["123458"] = order
        self.gateway.order_id_map["test_order_4"] = "123458"

        # Mock a successful cancellation response
        self.mock_response.status_code = 200
        self.mock_response.json.return_value = {
            "orderCancelTransaction": {
                "id": "123459",
                "type": "ORDER_CANCEL",
                "orderID": "123458",
                "time": "2023-01-01T12:01:00Z",
                "reason": "CLIENT_REQUEST"
            }
        }

        # Cancel the order
        result = self.gateway.cancel_order("test_order_4")

        # Verify the order was cancelled successfully
        self.assertTrue(result)
        self.assertEqual(order.status, OrderStatus.CANCELED)

        # Verify the API call
        cancel_endpoint = f"{self.gateway.orders_endpoint}/123458/cancel"
        self.gateway._send_request.assert_called_with("PUT", cancel_endpoint)

        # Verify the event was emitted
        self.mock_event_bus.emit.assert_called_once()

    def test_cancel_order_failure(self):
        """Test handling of failed order cancellation."""
        # Create and place a test order first
        order = Order(
            id="test_order_5",
            instrument="EUR/USD",
            quantity=1000,
            side=Side.BUY,
            type=OrderType.LIMIT,
            price=Decimal("1.10000")
        )

        # Mock an order placement
        order.exchange_order_id = "123460"
        self.gateway.open_orders["123460"] = order
        self.gateway.order_id_map["test_order_5"] = "123460"

        # Mock a failed cancellation response
        self.mock_response.status_code = 404
        self.mock_response.text = "Order not found"

        # Attempt to cancel the order
        result = self.gateway.cancel_order("test_order_5")

        # Verify the cancellation failed
        self.assertFalse(result)

        # Verify the API call
        cancel_endpoint = f"{self.gateway.orders_endpoint}/123460/cancel"
        self.gateway._send_request.assert_called_with("PUT", cancel_endpoint)

    def test_get_order(self):
        """Test retrieving an order."""
        # Create a test order ID and Oanda ID mapping
        order_id = "test_order_6"
        oanda_order_id = "123461"
        self.gateway.order_id_map[order_id] = oanda_order_id

        # Mock a successful order retrieval response
        self.mock_response.status_code = 200
        self.mock_response.json.return_value = {
            "order": {
                "id": oanda_order_id,
                "instrument": "EUR_USD",
                "units": "1000",
                "price": "1.10000",
                "type": "LIMIT",
                "state": "PENDING",
                "createTime": "2023-01-01T12:00:00Z",
                "clientExtensions": {
                    "id": order_id,
                    "tag": "system"
                }
            }
        }

        # Retrieve the order
        order = self.gateway.get_order(order_id)

        # Verify the order was retrieved successfully
        self.assertIsNotNone(order)
        self.assertEqual(order.id, order_id)
        self.assertEqual(order.exchange_order_id, oanda_order_id)
        self.assertEqual(order.instrument, "EUR/USD")
        self.assertEqual(order.quantity, 1000)
        self.assertEqual(order.side, Side.BUY)
        self.assertEqual(order.type, OrderType.LIMIT)
        self.assertEqual(order.price, 1.10000)
        self.assertEqual(order.status, OrderStatus.PENDING)

        # Verify the API call
        order_endpoint = f"{self.gateway.orders_endpoint}/{oanda_order_id}"
        self.gateway._send_request.assert_called_with("GET", order_endpoint)

    def test_get_open_orders(self):
        """Test retrieving all open orders."""
        # Mock a successful open orders response
        self.mock_response.status_code = 200
        self.mock_response.json.return_value = {
            "orders": [
                {
                    "id": "123462",
                    "instrument": "EUR_USD",
                    "units": "1000",
                    "price": "1.10000",
                    "type": "LIMIT",
                    "state": "PENDING",
                    "createTime": "2023-01-01T12:00:00Z"
                },
                {
                    "id": "123463",
                    "instrument": "USD_JPY",
                    "units": "-2000",
                    "price": "110.000",
                    "type": "LIMIT",
                    "state": "PENDING",
                    "createTime": "2023-01-01T12:01:00Z"
                }
            ]
        }

        # Retrieve open orders
        orders = self.gateway.get_open_orders()

        # Verify orders were retrieved successfully
        self.assertEqual(len(orders), 2)

        # Check first order
        self.assertEqual(orders[0].exchange_order_id, "123462")
        self.assertEqual(orders[0].instrument, "EUR/USD")
        self.assertEqual(orders[0].quantity, 1000)
        self.assertEqual(orders[0].side, Side.BUY)

        # Check second order
        self.assertEqual(orders[1].exchange_order_id, "123463")
        self.assertEqual(orders[1].instrument, "USD/JPY")
        self.assertEqual(orders[1].quantity, 2000)
        self.assertEqual(orders[1].side, Side.SELL)

        # Verify the API call
        self.gateway._send_request.assert_called_with("GET", self.gateway.orders_endpoint)

    def test_get_position(self):
        """Test retrieving a position for an instrument."""
        instrument = "EUR/USD"

        # Mock a successful position response
        self.mock_response.status_code = 200
        self.mock_response.json.return_value = {
            "position": {
                "instrument": "EUR_USD",
                "long": {
                    "units": "5000",
                    "averagePrice": "1.10000"
                },
                "short": {
                    "units": "-2000",
                    "averagePrice": "1.12000"
                }
            }
        }

        # Retrieve the position
        position_size = self.gateway.get_position(instrument)

        # Verify position was retrieved successfully
        self.assertEqual(position_size, 3000.0)  # 5000 - 2000 = 3000

        # Verify the API call
        position_endpoint = f"{self.gateway.positions_endpoint}/EUR_USD"
        self.gateway._send_request.assert_called_with("GET", position_endpoint)

    def test_get_balance(self):
        """Test retrieving account balance."""
        currency = "USD"

        # Mock a successful account response
        self.mock_response.status_code = 200
        self.mock_response.json.return_value = {
            "account": {
                "id": "test_account_id",
                "currency": "USD",
                "balance": "10000.50"
            }
        }

        # Retrieve the balance
        balance = self.gateway.get_balance(currency)

        # Verify balance was retrieved successfully
        self.assertEqual(balance, 10000.50)

        # Verify the API call
        self.gateway._send_request.assert_called_with("GET", self.gateway.account_endpoint)

    def test_get_all_balances(self):
        """Test retrieving all account balances."""
        # Mock a successful account response
        self.mock_response.status_code = 200
        self.mock_response.json.return_value = {
            "account": {
                "id": "test_account_id",
                "currency": "USD",
                "balance": "10000.50"
            }
        }

        # Retrieve all balances
        balances = self.gateway.get_all_balances()

        # Verify balances were retrieved successfully
        self.assertEqual(len(balances), 1)
        self.assertEqual(balances["USD"], 10000.50)

        # Verify the API call
        self.gateway._send_request.assert_called_with("GET", self.gateway.account_endpoint)

    def test_get_all_positions(self):
        """Test retrieving all positions."""
        # Mock a successful positions response
        self.mock_response.status_code = 200
        self.mock_response.json.return_value = {
            "positions": [
                {
                    "instrument": "EUR_USD",
                    "long": {
                        "units": "5000",
                        "averagePrice": "1.10000"
                    },
                    "short": {
                        "units": "-2000",
                        "averagePrice": "1.12000"
                    }
                },
                {
                    "instrument": "USD_JPY",
                    "long": {
                        "units": "0",
                        "averagePrice": "0"
                    },
                    "short": {
                        "units": "-10000",
                        "averagePrice": "110.000"
                    }
                }
            ]
        }

        # Retrieve all positions
        positions = self.gateway.get_all_positions()

        # Verify positions were retrieved successfully
        self.assertEqual(len(positions), 2)
        self.assertEqual(positions["EUR/USD"], 3000.0)  # 5000 - 2000
        self.assertEqual(positions["USD/JPY"], -10000.0)

        # Verify the API call
        self.gateway._send_request.assert_called_with("GET", self.gateway.positions_endpoint)

    def test_get_ticker(self):
        """Test retrieving ticker data for an instrument."""
        instrument = "EUR/USD"

        # Mock a successful pricing response
        self.mock_response.status_code = 200
        self.mock_response.json.return_value = {
            "prices": [
                {
                    "instrument": "EUR_USD",
                    "time": "2023-01-01T12:00:00Z",
                    "status": "tradeable",
                    "tradeable": True,
                    "bids": [
                        {"price": "1.10000", "liquidity": 10000000}
                    ],
                    "asks": [
                        {"price": "1.10010", "liquidity": 10000000}
                    ]
                }
            ]
        }

        # Retrieve ticker data
        ticker = self.gateway.get_ticker(instrument)

        # Verify ticker data was retrieved successfully
        self.assertEqual(ticker["instrument"], "EUR/USD")
        self.assertEqual(ticker["bid"], 1.10000)
        self.assertEqual(ticker["ask"], 1.10010)
        self.assertEqual(ticker["mid"], 1.10005)  # (1.10000 + 1.10010) / 2
        self.assertEqual(ticker["status"], "tradeable")
        self.assertEqual(ticker["tradeable"], True)

        # Verify the API call
        pricing_endpoint = f"{self.gateway.base_url}/{self.gateway.api_version}/accounts/{self.gateway.account_id}/pricing"
        self.gateway._send_request.assert_called_with(
            "GET",
            pricing_endpoint,
            params={"instruments": "EUR_USD"}
        )

    def test_format_instrument_for_oanda(self):
        """Test currency pair formatting for Oanda API."""
        self.assertEqual(self.gateway._format_instrument_for_oanda("EUR/USD"), "EUR_USD")
        self.assertEqual(self.gateway._format_instrument_for_oanda("USD/JPY"), "USD_JPY")
        self.assertEqual(self.gateway._format_instrument_for_oanda("GBP/NZD"), "GBP_NZD")

    def test_parse_oanda_instrument(self):
        """Test parsing Oanda currency pair format to system format."""
        self.assertEqual(self.gateway._parse_oanda_instrument("EUR_USD"), "EUR/USD")
        self.assertEqual(self.gateway._parse_oanda_instrument("USD_JPY"), "USD/JPY")
        self.assertEqual(self.gateway._parse_oanda_instrument("GBP_NZD"), "GBP/NZD")

    @patch('threading.Thread')
    def test_background_updates(self, mock_thread):
        """Test that background updates are started and stopped correctly."""
        # Stop the existing thread first
        self.gateway.disconnect()

        # Reset connection state
        self.gateway.connected = False

        # Mock thread creation
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance

        # Connect again with mocked thread
        self.gateway.connect()

        # Verify thread was started properly
        mock_thread.assert_called_once()
        mock_thread_instance.start.assert_called_once()

        # Now disconnect
        self.gateway.disconnect()

        # Verify thread was joined
        mock_thread_instance.join.assert_called_once()
        self.assertFalse(self.gateway.running)


if __name__ == '__main__':
    unittest.main()