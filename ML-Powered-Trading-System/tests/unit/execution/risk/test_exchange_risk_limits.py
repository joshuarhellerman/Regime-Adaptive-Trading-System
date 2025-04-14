"""
Test suite for ExchangeRiskLimits class.

These tests validate the functionality of the exchange-specific risk limits
and ensure proper validation of orders against exchange constraints.
"""

import json
import os
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from execution.exchange.risk.exchange_risk_limits import ExchangeRiskLimits, RateLimitType


class TestExchangeRiskLimits(unittest.TestCase):
    """Test cases for ExchangeRiskLimits class."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a test config with mock exchange limits
        self.test_config = {
            "data_dir": "test_data/exchange_limits",
            "status_check_interval_minutes": 30,
            "exchanges": {
                "exchange1": {
                    "enabled": True,
                    "max_order_size": 100.0,
                    "min_order_size": 0.01,
                    "max_notional_value": 10000.0,
                    "min_notional_value": 10.0,
                    "max_leverage": 20.0,
                    "price_precision": {"BTC/USDT": 2, "ETH/USDT": 2},
                    "quantity_precision": {"BTC/USDT": 5, "ETH/USDT": 4},
                    "default_price_precision": 2,
                    "default_quantity_precision": 5,
                    "rate_limits": [
                        {"type": "orders", "window_seconds": 60, "max_count": 10},
                        {"type": "requests", "window_seconds": 10, "max_count": 50}
                    ],
                    "supported_order_types": ["market", "limit", "stop_limit"],
                    "supported_time_in_force": ["GTC", "IOC", "FOK"],
                    "trading_hours": {
                        "default": {"24/7": True},
                        "BTC/USDT": {
                            "monday": [{"open": "00:00:00", "close": "23:59:59"}],
                            "tuesday": [{"open": "00:00:00", "close": "23:59:59"}],
                            "wednesday": [{"open": "00:00:00", "close": "23:59:59"}],
                            "thursday": [{"open": "00:00:00", "close": "23:59:59"}],
                            "friday": [{"open": "00:00:00", "close": "23:59:59"}],
                            "saturday": [{"open": "00:00:00", "close": "23:59:59"}],
                            "sunday": [{"open": "00:00:00", "close": "23:59:59"}]
                        },
                        "RESTRICTED/USDT": {
                            "monday": [{"open": "09:00:00", "close": "17:00:00"}],
                            "tuesday": [{"open": "09:00:00", "close": "17:00:00"}],
                            "wednesday": [{"open": "09:00:00", "close": "17:00:00"}],
                            "thursday": [{"open": "09:00:00", "close": "17:00:00"}],
                            "friday": [{"open": "09:00:00", "close": "17:00:00"}],
                            "saturday": [],
                            "sunday": []
                        }
                    }
                },
                "exchange2": {
                    "enabled": False,
                    "max_order_size": 50.0,
                    "min_order_size": 0.05,
                    "max_notional_value": 5000.0,
                    "min_notional_value": 20.0,
                    "max_leverage": 10.0,
                    "supported_order_types": ["market", "limit"],
                    "supported_time_in_force": ["GTC"],
                    "rate_limits": [
                        {"type": "orders", "window_seconds": 60, "max_count": 5}
                    ]
                }
            }
        }

        # Create test directory
        os.makedirs("test_data/exchange_limits", exist_ok=True)

        # Initialize the risk limits manager
        self.risk_limits = ExchangeRiskLimits(self.test_config)

        # Add specific instrument limits for testing
        self.risk_limits.update_instrument_limits(
            "exchange1",
            "ETH/USDT",
            {
                "min_order_size": 0.05,
                "max_order_size": 50.0,
                "min_notional_value": 20.0
            }
        )

    def tearDown(self):
        """Clean up after each test."""
        # Remove test data directory
        import shutil
        if os.path.exists("test_data"):
            shutil.rmtree("test_data")

    def test_init(self):
        """Test initialization of ExchangeRiskLimits."""
        self.assertEqual(len(self.risk_limits.exchange_limits), 2)
        self.assertIn("exchange1", self.risk_limits.exchange_limits)
        self.assertIn("exchange2", self.risk_limits.exchange_limits)
        self.assertEqual(self.risk_limits.exchange_limits["exchange1"]["status"], "online")
        self.assertEqual(self.risk_limits.exchange_limits["exchange2"]["status"], "online")

        # Check that data directory was created
        self.assertTrue(os.path.exists("test_data/exchange_limits"))

        # Check rate limit windows initialization
        self.assertIn("exchange1", self.risk_limits.rate_limit_windows)
        self.assertIn("orders", self.risk_limits.rate_limit_windows["exchange1"])
        self.assertEqual(len(self.risk_limits.rate_limit_windows["exchange1"]["orders"]), 1)
        self.assertEqual(self.risk_limits.rate_limit_windows["exchange1"]["orders"][0]["max_count"], 10)

    def test_update_exchange_status(self):
        """Test updating exchange status."""
        self.risk_limits.update_exchange_status("exchange1", "limited")
        self.assertEqual(self.risk_limits.exchange_limits["exchange1"]["status"], "limited")

        # Test updating non-existent exchange
        self.risk_limits.update_exchange_status("nonexistent", "offline")
        self.assertNotIn("nonexistent", self.risk_limits.exchange_limits)

    def test_get_exchange_status(self):
        """Test getting exchange status."""
        self.assertEqual(self.risk_limits.get_exchange_status("exchange1"), "online")
        self.assertEqual(self.risk_limits.get_exchange_status("nonexistent"), "unknown")

    def test_is_exchange_available(self):
        """Test checking exchange availability."""
        self.assertTrue(self.risk_limits.is_exchange_available("exchange1"))

        # Update status to limited
        self.risk_limits.update_exchange_status("exchange1", "limited")
        self.assertTrue(self.risk_limits.is_exchange_available("exchange1"))

        # Update status to offline
        self.risk_limits.update_exchange_status("exchange1", "offline")
        self.assertFalse(self.risk_limits.is_exchange_available("exchange1"))

        # Test non-existent exchange
        self.assertFalse(self.risk_limits.is_exchange_available("nonexistent"))

    def test_check_order_size_limits(self):
        """Test checking order size limits."""
        # Valid order
        is_valid, reason = self.risk_limits.check_order_size_limits("exchange1", "BTC/USDT", 10.0, 500.0)
        self.assertTrue(is_valid)
        self.assertEqual(reason, "")

        # Size too small
        is_valid, reason = self.risk_limits.check_order_size_limits("exchange1", "BTC/USDT", 0.001, 500.0)
        self.assertFalse(is_valid)
        self.assertIn("below minimum", reason)

        # Size too large
        is_valid, reason = self.risk_limits.check_order_size_limits("exchange1", "BTC/USDT", 200.0, 500.0)
        self.assertFalse(is_valid)
        self.assertIn("exceeds maximum", reason)

        # Notional value too small
        is_valid, reason = self.risk_limits.check_order_size_limits("exchange1", "BTC/USDT", 0.01, 0.1)
        self.assertFalse(is_valid)
        self.assertIn("below minimum", reason)

        # Notional value too large
        is_valid, reason = self.risk_limits.check_order_size_limits("exchange1", "BTC/USDT", 50.0, 500.0)
        self.assertFalse(is_valid)
        self.assertIn("exceeds maximum", reason)

        # Check symbol-specific limits
        is_valid, reason = self.risk_limits.check_order_size_limits("exchange1", "ETH/USDT", 0.03, 500.0)
        self.assertFalse(is_valid)
        self.assertIn("below minimum", reason)

        # Check disabled exchange
        is_valid, reason = self.risk_limits.check_order_size_limits("exchange2", "BTC/USDT", 10.0, 500.0)
        self.assertFalse(is_valid)
        self.assertIn("disabled", reason)

        # Check offline exchange
        self.risk_limits.update_exchange_status("exchange1", "offline")
        is_valid, reason = self.risk_limits.check_order_size_limits("exchange1", "BTC/USDT", 10.0, 500.0)
        self.assertFalse(is_valid)
        self.assertIn("offline", reason)

        # Unknown exchange
        is_valid, reason = self.risk_limits.check_order_size_limits("nonexistent", "BTC/USDT", 10.0, 500.0)
        self.assertFalse(is_valid)
        self.assertIn("Unknown exchange", reason)

    def test_check_price_precision(self):
        """Test checking and adjusting price precision."""
        # Test symbol with defined precision
        is_valid, adjusted_price = self.risk_limits.check_price_precision("exchange1", "BTC/USDT", 123.456789)
        self.assertTrue(is_valid)
        self.assertEqual(adjusted_price, 123.46)

        # Test symbol with undefined precision (should use default)
        is_valid, adjusted_price = self.risk_limits.check_price_precision("exchange1", "LTC/USDT", 123.456789)
        self.assertTrue(is_valid)
        self.assertEqual(adjusted_price, 123.46)  # Using default precision of 2

        # Test unknown exchange
        is_valid, adjusted_price = self.risk_limits.check_price_precision("nonexistent", "BTC/USDT", 123.456789)
        self.assertFalse(is_valid)
        self.assertEqual(adjusted_price, 123.456789)  # Unchanged

    def test_check_quantity_precision(self):
        """Test checking and adjusting quantity precision."""
        # Test symbol with defined precision
        is_valid, adjusted_qty = self.risk_limits.check_quantity_precision("exchange1", "BTC/USDT", 1.123456789)
        self.assertTrue(is_valid)
        self.assertEqual(adjusted_qty, 1.12346)  # Rounded to 5 decimals

        # Test symbol with defined precision (different value)
        is_valid, adjusted_qty = self.risk_limits.check_quantity_precision("exchange1", "ETH/USDT", 1.123456789)
        self.assertTrue(is_valid)
        self.assertEqual(adjusted_qty, 1.1235)  # Rounded to 4 decimals

        # Test symbol with undefined precision (should use default)
        is_valid, adjusted_qty = self.risk_limits.check_quantity_precision("exchange1", "LTC/USDT", 1.123456789)
        self.assertTrue(is_valid)
        self.assertEqual(adjusted_qty, 1.12346)  # Using default precision of 5

        # Test unknown exchange
        is_valid, adjusted_qty = self.risk_limits.check_quantity_precision("nonexistent", "BTC/USDT", 1.123456789)
        self.assertFalse(is_valid)
        self.assertEqual(adjusted_qty, 1.123456789)  # Unchanged

    def test_check_order_type_supported(self):
        """Test checking if order type is supported."""
        # Supported order types
        self.assertTrue(self.risk_limits.check_order_type_supported("exchange1", "market"))
        self.assertTrue(self.risk_limits.check_order_type_supported("exchange1", "limit"))
        self.assertTrue(self.risk_limits.check_order_type_supported("exchange1", "stop_limit"))

        # Unsupported order type
        self.assertFalse(self.risk_limits.check_order_type_supported("exchange1", "trailing_stop"))

        # Different exchange with different supported types
        self.assertTrue(self.risk_limits.check_order_type_supported("exchange2", "market"))
        self.assertFalse(self.risk_limits.check_order_type_supported("exchange2", "stop_limit"))

        # Unknown exchange
        self.assertFalse(self.risk_limits.check_order_type_supported("nonexistent", "market"))

    def test_check_time_in_force_supported(self):
        """Test checking if time in force is supported."""
        # Supported time in force
        self.assertTrue(self.risk_limits.check_time_in_force_supported("exchange1", "GTC"))
        self.assertTrue(self.risk_limits.check_time_in_force_supported("exchange1", "IOC"))

        # Unsupported time in force
        self.assertFalse(self.risk_limits.check_time_in_force_supported("exchange1", "GTD"))

        # Different exchange with different supported types
        self.assertTrue(self.risk_limits.check_time_in_force_supported("exchange2", "GTC"))
        self.assertFalse(self.risk_limits.check_time_in_force_supported("exchange2", "IOC"))

        # Unknown exchange
        self.assertFalse(self.risk_limits.check_time_in_force_supported("nonexistent", "GTC"))

    def test_check_rate_limit(self):
        """Test checking rate limits."""
        # Initially, should be allowed
        is_allowed, _ = self.risk_limits.check_rate_limit("exchange1", "orders")
        self.assertTrue(is_allowed)

        # Record several requests to hit limit
        for _ in range(10):
            self.risk_limits.record_request("exchange1", "orders")

        # Now should be rate limited
        is_allowed, wait_seconds = self.risk_limits.check_rate_limit("exchange1", "orders")
        self.assertFalse(is_allowed)
        self.assertGreater(wait_seconds, 0)

        # Test different limit type
        is_allowed, _ = self.risk_limits.check_rate_limit("exchange1", "requests")
        self.assertTrue(is_allowed)  # Different limit type should be allowed

        # Unknown exchange
        is_allowed, _ = self.risk_limits.check_rate_limit("nonexistent", "orders")
        self.assertFalse(is_allowed)

        # Unknown limit type
        is_allowed, _ = self.risk_limits.check_rate_limit("exchange1", "nonexistent")
        self.assertTrue(is_allowed)  # Should default to allowed if type unknown

    def test_record_request(self):
        """Test recording requests for rate limiting."""
        # Record a request
        self.risk_limits.record_request("exchange1", "orders")

        # Check that it was recorded
        window = self.risk_limits.rate_limit_windows["exchange1"]["orders"][0]
        self.assertEqual(len(window["timestamps"]), 1)

        # Record multiple requests
        for _ in range(5):
            self.risk_limits.record_request("exchange1", "orders")

        # Check that all were recorded
        self.assertEqual(len(window["timestamps"]), 6)

        # Test recording for non-existent exchange/limit type (should not error)
        self.risk_limits.record_request("nonexistent", "orders")
        self.risk_limits.record_request("exchange1", "nonexistent")

    def test_update_instrument_limits(self):
        """Test updating instrument-specific limits."""
        # Update limits for new symbol
        self.risk_limits.update_instrument_limits("exchange1", "LTC/USDT", {
            "min_order_size": 0.1,
            "max_order_size": 30.0
        })

        # Check that limits were added
        self.assertIn("LTC/USDT", self.risk_limits.instrument_limits["exchange1"])
        self.assertEqual(self.risk_limits.instrument_limits["exchange1"]["LTC/USDT"]["min_order_size"], 0.1)

        # Update existing limits
        self.risk_limits.update_instrument_limits("exchange1", "ETH/USDT", {
            "max_leverage": 5.0
        })

        # Check that limits were updated and existing ones preserved
        self.assertEqual(self.risk_limits.instrument_limits["exchange1"]["ETH/USDT"]["min_order_size"], 0.05)
        self.assertEqual(self.risk_limits.instrument_limits["exchange1"]["ETH/USDT"]["max_leverage"], 5.0)

        # Update for new exchange
        self.risk_limits.update_instrument_limits("exchange3", "BTC/USDT", {
            "min_order_size": 0.01
        })

        # Check that new exchange was added
        self.assertIn("exchange3", self.risk_limits.instrument_limits)
        self.assertEqual(self.risk_limits.instrument_limits["exchange3"]["BTC/USDT"]["min_order_size"], 0.01)

    def test_load_limits_from_exchange(self):
        """Test loading limits from exchange API."""
        # Create mock exchange gateway
        mock_gateway = MagicMock()
        mock_gateway.get_exchange_info.return_value = {
            "status": "online",
            "rate_limits": [
                {"type": "orders", "window_seconds": 60, "max_count": 20}
            ],
            "trading_hours": {
                "default": {"24/7": True}
            },
            "symbols": [
                {
                    "symbol": "BTC/USDT",
                    "min_order_size": 0.001,
                    "max_order_size": 50.0,
                    "min_notional_value": 5.0,
                    "price_precision": 1,
                    "quantity_precision": 3
                },
                {
                    "symbol": "ETH/USDT",
                    "min_order_size": 0.01,
                    "max_order_size": 100.0,
                    "price_precision": 2,
                    "quantity_precision": 4
                }
            ]
        }

        # Load limits
        result = self.risk_limits.load_limits_from_exchange("exchange1", mock_gateway)
        self.assertTrue(result)

        # Check that limits were updated
        self.assertEqual(self.risk_limits.exchange_limits["exchange1"]["status"], "online")
        self.assertEqual(len(self.risk_limits.exchange_limits["exchange1"]["rate_limits"]), 1)
        self.assertEqual(self.risk_limits.exchange_limits["exchange1"]["rate_limits"][0]["max_count"], 20)

        # Check that instrument limits were updated
        self.assertEqual(self.risk_limits.instrument_limits["exchange1"]["BTC/USDT"]["min_order_size"], 0.001)
        self.assertEqual(self.risk_limits.exchange_limits["exchange1"]["price_precision"]["BTC/USDT"], 1)

        # Test error handling
        mock_gateway.get_exchange_info.side_effect = Exception("API error")
        result = self.risk_limits.load_limits_from_exchange("exchange1", mock_gateway)
        self.assertFalse(result)

    @patch('execution.exchange.risk.exchange_risk_limits.datetime')
    def test_check_trading_hours(self, mock_datetime):
        """Test checking trading hours."""
        # Mock the current time to be Monday at 12:00
        mock_now = MagicMock()
        mock_now.strftime.return_value = "monday"
        mock_now.time.return_value = datetime.strptime("12:00:00", "%H:%M:%S").time()
        mock_datetime.utcnow.return_value = mock_now

        # Test 24/7 trading symbol
        self.assertTrue(self.risk_limits.check_trading_hours("exchange1", "BTC/USDT"))

        # Test restricted hours symbol - within trading hours
        self.assertTrue(self.risk_limits.check_trading_hours("exchange1", "RESTRICTED/USDT"))

        # Test restricted hours symbol - outside trading hours
        mock_now.time.return_value = datetime.strptime("18:00:00", "%H:%M:%S").time()
        self.assertFalse(self.risk_limits.check_trading_hours("exchange1", "RESTRICTED/USDT"))

        # Test restricted hours symbol - weekend
        mock_now.strftime.return_value = "saturday"
        self.assertFalse(self.risk_limits.check_trading_hours("exchange1", "RESTRICTED/USDT"))

        # Test symbol with no specific hours (should use default)
        mock_now.strftime.return_value = "monday"
        self.assertTrue(self.risk_limits.check_trading_hours("exchange1", "LTC/USDT"))

        # Test exchange with no trading hours defined
        del self.risk_limits.exchange_limits["exchange1"]["trading_hours"]
        self.assertTrue(self.risk_limits.check_trading_hours("exchange1", "BTC/USDT"))

        # Unknown exchange
        self.assertFalse(self.risk_limits.check_trading_hours("nonexistent", "BTC/USDT"))

    def test_check_leverage_limit(self):
        """Test checking leverage limits."""
        # Within limit
        is_valid, adjusted_leverage = self.risk_limits.check_leverage_limit("exchange1", "BTC/USDT", 10.0)
        self.assertTrue(is_valid)
        self.assertEqual(adjusted_leverage, 10.0)

        # Exceeds limit
        is_valid, adjusted_leverage = self.risk_limits.check_leverage_limit("exchange1", "BTC/USDT", 30.0)
        self.assertTrue(is_valid)
        self.assertEqual(adjusted_leverage, 20.0)  # Should be adjusted to max

        # Add symbol-specific leverage limit
        self.risk_limits.update_instrument_limits("exchange1", "ETH/USDT", {"max_leverage": 5.0})

        # Test symbol-specific limit
        is_valid, adjusted_leverage = self.risk_limits.check_leverage_limit("exchange1", "ETH/USDT", 10.0)
        self.assertTrue(is_valid)
        self.assertEqual(adjusted_leverage, 5.0)

        # Unknown exchange
        is_valid, adjusted_leverage = self.risk_limits.check_leverage_limit("nonexistent", "BTC/USDT", 10.0)
        self.assertFalse(is_valid)
        self.assertEqual(adjusted_leverage, 10.0)  # Unchanged

    def test_validate_order(self):
        """Test comprehensive order validation."""
        # Valid market order
        is_valid, error, adjusted = self.risk_limits.validate_order(
            "exchange1", "BTC/USDT", "market", "GTC", 1.0, 1000.0
        )
        self.assertTrue(is_valid)
        self.assertEqual(error, "")

        # Valid limit order
        is_valid, error, adjusted = self.risk_limits.validate_order(
            "exchange1", "BTC/USDT", "limit", "GTC", 1.0, 1000.0
        )
        self.assertTrue(is_valid)
        self.assertEqual(error, "")

        # Limit order without price
        is_valid, error, adjusted = self.risk_limits.validate_order(
            "exchange1", "BTC/USDT", "limit", "GTC", 1.0
        )
        self.assertFalse(is_valid)
        self.assertIn("requires price", error)

        # Order with precision adjustment
        is_valid, error, adjusted = self.risk_limits.validate_order(
            "exchange1", "BTC/USDT", "limit", "GTC", 1.123456789, 1000.123456789
        )
        self.assertTrue(is_valid)
        self.assertEqual(adjusted["size"], 1.12346)
        self.assertEqual(adjusted["price"], 1000.12)

        # Unsupported order type
        is_valid, error, adjusted = self.risk_limits.validate_order(
            "exchange1", "BTC/USDT", "trailing_stop", "GTC", 1.0, 1000.0
        )
        self.assertFalse(is_valid)
        self.assertIn("not supported", error)

        # Unsupported time in force
        is_valid, error, adjusted = self.risk_limits.validate_order(
            "exchange1", "BTC/USDT", "limit", "GTD", 1.0, 1000.0
        )
        self.assertFalse(is_valid)
        self.assertIn("not supported", error)

        # Order too small
        is_valid, error, adjusted = self.risk_limits.validate_order(
            "exchange1", "BTC/USDT", "limit", "GTC", 0.001, 1000.0
        )
        self.assertFalse(is_valid)
        self.assertIn("below minimum", error)

        # Order too large
        is_valid, error, adjusted = self.risk_limits.validate_order(
            "exchange1", "BTC/USDT", "limit", "GTC", 200.0, 1000.0
        )
        self.assertFalse(is_valid)
        self.assertIn("exceeds maximum", error)

        # Order with leverage adjustment
        is_valid, error, adjusted = self.risk_limits.validate_order(
            "exchange1", "BTC/USDT", "limit", "GTC", 1.0, 1000.0, 30.0
        )
        self.assertTrue(is_valid)
        self.assertEqual(adjusted["leverage"], 20.0)

        # Exchange not available
        self.risk_limits.update_exchange_status("exchange1", "offline")
        is_valid, error, adjusted = self.risk_limits.validate_order(
            "exchange1", "BTC/USDT", "limit", "GTC", 1.0, 1000.0
        )
        self.assertFalse(is_valid)
        self.assertIn("not available", error)

        # Unknown exchange
        is_valid, error, adjusted = self.risk_limits.validate_order(
            "nonexistent", "BTC/USDT", "limit", "GTC", 1.0, 1000.0
        )
        self.assertFalse(is_valid)
        self.assertIn("not available", error)

        # Test rate limit exceeded
        self.risk_limits.update_exchange_status("exchange1", "online")
        # Fill rate limit
        for _ in range(10):
            self.risk_limits.record_request("exchange1", "orders")

        is_valid, error, adjusted = self.risk_limits.validate_order(
            "exchange1", "BTC/USDT", "market", "GTC", 1.0, 1000.0
        )
        self.assertFalse(is_valid)
        self.assertIn("Rate limit exceeded", error)

    def test_should_check_exchange_status(self):
        """Test determining if exchange status should be checked."""
        # Initially should check (initialized with time 24h ago)
        self.assertTrue(self.risk_limits.should_check_exchange_status("exchange1"))

        # Update last check time
        self.risk_limits.last_status_check["exchange1"] = datetime.now()
        self.assertFalse(self.risk_limits.should_check_exchange_status("exchange1"))

        # Set last check to be more than interval minutes ago
        self.risk_limits.last_status_check["exchange1"] = datetime.now() - timedelta(minutes=60)
        self.assertTrue(self.risk_limits.should_check_exchange_status("exchange1"))

        # Unknown exchange
        self.assertTrue(self.risk_limits.should_check_exchange_status("nonexistent"))

    def test_get_limits_summary(self):
        """Test getting limits summary."""
        # Get summary for exchange
        summary = self.risk_limits.get_limits_summary("exchange1")
        self.assertEqual(summary["exchange_id"], "exchange1")
        self.assertEqual(summary["status"], "online")
        self.assertEqual(summary["general_limits"]["max_order_size"], 100.0)
        self.assertIn("rate_limits", summary)

        # Get summary with symbol
        summary = self.risk_limits.get_limits_summary("exchange1", "ETH/USDT")
        self.assertIn("symbol_limits", summary)
        self.assertEqual(summary["symbol_limits"]["min_order_size"], 0.05)

        # Unknown exchange
        summary = self.risk_limits.get_limits_summary("nonexistent")
        self.assertIn("error", summary)

    def test_reset(self):
        """Test resetting rate limit tracking."""
        # Record some requests
        self.risk_limits.record_request("exchange1", "orders")
        self.risk_limits.record_request("exchange1", "orders")

        # Verify they're recorded
        window = self.risk_limits.rate_limit_windows["exchange1"]["orders"][0]
        self.assertEqual(len(window["timestamps"]), 2)

        # Reset
        self.risk_limits.reset()

        # Verify timestamps are cleared
        self.assertEqual(len(window["timestamps"]), 0)

        # Verify status check times are reset
        self.assertLess(
            (datetime.now() - self.risk_limits.last_status_check["exchange1"]).total_seconds(),
            -86000  # Close to 24 hours ago
        )

    @patch('json.dump')
    def test_save_limits_to_file(self, mock_json_dump):
        """Test saving limits to file."""
        # Call private method
        self.risk_limits._save_limits_to_file("exchange1")

        # Check that json.dump was called
        mock_json_dump.assert_called_once()

    @patch('json.load')
    def test_load_limits_from_file(self, mock_json_load):
        """Test loading limits from file."""
        # Mock json.load to return test data
        mock_json_load.return_value = {
            "exchange_limits": {
                "status": "limited",
                "max_order_size": 75.0,
                "min_order_size": 0.02,
                "rate_limits": [
                    {"type": "orders", "window_seconds": 60, "max_count": 15}
                ]
            },
            "instrument_limits": {
                "BTC/USDT": {"min_order_size": 0.002}
            }
        }

        # Mock file existence
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', MagicMock()):
            # Load limits
            result = self.risk_limits._load_limits_from_file("exchange1")
            self.assertTrue(result)

            # Check that limits were updated
            self.assertEqual(self.risk_limits.exchange_limits["exchange1"]["status"], "limited")
            self.assertEqual(self.risk_limits.exchange_limits["exchange1"]["max_order_size"], 75.0)

            # Check that instrument limits were updated
            self.assertEqual(self.risk_limits.instrument_limits["exchange1"]["BTC/USDT"]["min_order_size"], 0.002)

        # Test file not found
        with patch('pathlib.Path.exists', return_value=False):
            result = self.risk_limits._load_limits_from_file("exchange1")
            self.assertFalse(result)

        # Test error handling
        mock_json_load.side_effect = Exception("JSON error")
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', MagicMock()):
            result = self.risk_limits._load_limits_from_file("exchange1")
            self.assertFalse(result)load):
        """Test loading limits from file."""
        # Mock json.load to return test data
        mock_json_