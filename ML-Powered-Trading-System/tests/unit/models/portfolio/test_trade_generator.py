"""
Unit tests for the TradeGenerator class.

This module tests the functionality of the trade generator module,
which converts portfolio weight changes into concrete trade instructions.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import asdict

from models.portfolio.trade_generator import (
    TradeGenerator,
    TradeInstruction,
    TradeGenerationResult
)
from models.portfolio.portfolio import Portfolio


class TestTradeInstruction(unittest.TestCase):
    """Tests for the TradeInstruction class."""

    def test_initialization(self):
        """Test basic initialization of a TradeInstruction."""
        instruction = TradeInstruction(
            symbol="AAPL",
            quantity=100.0,
            side="buy",
            limit_price=150.0
        )

        self.assertEqual(instruction.symbol, "AAPL")
        self.assertEqual(instruction.quantity, 100.0)
        self.assertEqual(instruction.side, "BUY")  # Should be uppercase
        self.assertEqual(instruction.limit_price, 150.0)
        self.assertEqual(instruction.order_type, "MARKET")  # Default
        self.assertIsNotNone(instruction.id)  # Should auto-generate ID

    def test_value_property(self):
        """Test the value property of TradeInstruction."""
        # With limit price
        instruction1 = TradeInstruction(
            symbol="AAPL",
            quantity=100.0,
            side="BUY",
            limit_price=150.0
        )
        self.assertEqual(instruction1.value, 15000.0)

        # Without limit price, using metadata
        instruction2 = TradeInstruction(
            symbol="AAPL",
            quantity=100.0,
            side="BUY",
            metadata={"current_price": 155.0}
        )
        self.assertEqual(instruction2.value, 15500.0)

        # Without any price information
        instruction3 = TradeInstruction(
            symbol="AAPL",
            quantity=100.0,
            side="BUY"
        )
        self.assertEqual(instruction3.value, 0.0)

    def test_string_representation(self):
        """Test string representation of the trade instruction."""
        instruction1 = TradeInstruction(
            symbol="AAPL",
            quantity=100.0,
            side="BUY",
            limit_price=150.0
        )
        self.assertEqual(str(instruction1), "BUY 100.0 AAPL @ MARKET 150.0")

        instruction2 = TradeInstruction(
            symbol="MSFT",
            quantity=50.0,
            side="SELL"
        )
        self.assertEqual(str(instruction2), "SELL 50.0 MSFT @ MARKET")


class TestTradeGenerationResult(unittest.TestCase):
    """Tests for the TradeGenerationResult class."""

    def test_initialization(self):
        """Test basic initialization of a TradeGenerationResult."""
        trade1 = TradeInstruction(symbol="AAPL", quantity=100.0, side="BUY", limit_price=150.0)
        trade2 = TradeInstruction(symbol="MSFT", quantity=50.0, side="SELL", limit_price=200.0)

        result = TradeGenerationResult(
            trades=[trade1, trade2],
            total_value=25000.0
        )

        self.assertEqual(len(result.trades), 2)
        self.assertEqual(result.total_value, 25000.0)
        self.assertIsInstance(result.timestamp, datetime)
        self.assertEqual(result.metadata, {})

    def test_string_representation(self):
        """Test string representation of the trade generation result."""
        trade1 = TradeInstruction(symbol="AAPL", quantity=100.0, side="BUY", limit_price=150.0)
        trade2 = TradeInstruction(symbol="MSFT", quantity=50.0, side="SELL", limit_price=200.0)

        result = TradeGenerationResult(
            trades=[trade1, trade2],
            total_value=25000.0
        )

        self.assertEqual(str(result), "Generated 2 trades with total value 25000.00")


class TestTradeGenerator(unittest.TestCase):
    """Tests for the TradeGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.trade_generator = TradeGenerator(
            min_trade_value=100.0,
            min_trade_quantity=1.0,
            round_quantities=True,
            cash_key="CASH"
        )

        # Create mock portfolio
        self.portfolio = MagicMock(spec=Portfolio)
        self.portfolio.total_value = 100000.0
        self.portfolio.weights = {
            "AAPL": 0.3,  # 30% in AAPL
            "MSFT": 0.2,  # 20% in MSFT
            "GOOG": 0.1,  # 10% in GOOG
            "CASH": 0.4   # 40% in cash
        }

        # Mock positions with prices
        position_aapl = MagicMock()
        position_aapl.current_price = 150.0
        position_msft = MagicMock()
        position_msft.current_price = 200.0
        position_goog = MagicMock()
        position_goog.current_price = 2500.0

        self.portfolio.positions = {
            "AAPL": position_aapl,
            "MSFT": position_msft,
            "GOOG": position_goog,
            "CASH": MagicMock()
        }

    def test_initialization(self):
        """Test TradeGenerator initialization."""
        generator = TradeGenerator(
            min_trade_value=200.0,
            min_trade_quantity=2.0,
            max_trade_value=10000.0,
            round_quantities=False,
            cash_key="USD"
        )

        self.assertEqual(generator.min_trade_value, 200.0)
        self.assertEqual(generator.min_trade_quantity, 2.0)
        self.assertEqual(generator.max_trade_value, 10000.0)
        self.assertEqual(generator.round_quantities, False)
        self.assertEqual(generator.cash_key, "USD")

    def test_generate_trades_basics(self):
        """Test basic trade generation functionality."""
        # Target weights that require some rebalancing
        target_weights = {
            "AAPL": 0.4,  # Increase AAPL by 10%
            "MSFT": 0.15,  # Decrease MSFT by 5%
            "GOOG": 0.15,  # Increase GOOG by 5%
            "CASH": 0.3    # Decrease CASH by 10%
        }

        result = self.trade_generator.generate_trades(
            current_portfolio=self.portfolio,
            target_weights=target_weights
        )

        # Verify results
        self.assertIsInstance(result, TradeGenerationResult)

        # We should have 3 trades (one for each non-cash asset)
        self.assertEqual(len(result.trades), 3)

        # Check specific trades
        trades_by_symbol = {t.symbol: t for t in result.trades}

        # AAPL: +10% of portfolio value / price
        aapl_trade = trades_by_symbol.get("AAPL")
        self.assertIsNotNone(aapl_trade)
        self.assertEqual(aapl_trade.side, "BUY")
        self.assertEqual(aapl_trade.quantity, round(0.1 * 100000.0 / 150.0))

        # MSFT: -5% of portfolio value / price
        msft_trade = trades_by_symbol.get("MSFT")
        self.assertIsNotNone(msft_trade)
        self.assertEqual(msft_trade.side, "SELL")
        self.assertEqual(msft_trade.quantity, round(0.05 * 100000.0 / 200.0))

        # GOOG: +5% of portfolio value / price
        goog_trade = trades_by_symbol.get("GOOG")
        self.assertIsNotNone(goog_trade)
        self.assertEqual(goog_trade.side, "BUY")
        self.assertEqual(goog_trade.quantity, round(0.05 * 100000.0 / 2500.0))

    def test_generate_trades_with_external_prices(self):
        """Test generating trades with externally provided prices."""
        target_weights = {
            "AAPL": 0.4,
            "MSFT": 0.15,
            "GOOG": 0.15,
            "CASH": 0.3
        }

        # Use different prices than what's in the portfolio
        prices = {
            "AAPL": 160.0,
            "MSFT": 220.0,
            "GOOG": 2600.0
        }

        result = self.trade_generator.generate_trades(
            current_portfolio=self.portfolio,
            target_weights=target_weights,
            prices=prices
        )

        # Verify trades use the provided prices
        trades_by_symbol = {t.symbol: t for t in result.trades}

        aapl_trade = trades_by_symbol.get("AAPL")
        self.assertEqual(aapl_trade.limit_price, 160.0)
        self.assertEqual(aapl_trade.quantity, round(0.1 * 100000.0 / 160.0))

    def test_generate_trades_with_lot_sizes(self):
        """Test generating trades with lot size constraints."""
        target_weights = {
            "AAPL": 0.4,
            "MSFT": 0.15,
            "GOOG": 0.15,
            "CASH": 0.3
        }

        # Define lot sizes
        lot_sizes = {
            "AAPL": 10.0,  # Must trade in multiples of 10
            "MSFT": 5.0,   # Must trade in multiples of 5
            "GOOG": 1.0    # Must trade in multiples of 1 (no effect)
        }

        result = self.trade_generator.generate_trades(
            current_portfolio=self.portfolio,
            target_weights=target_weights,
            lot_sizes=lot_sizes
        )

        # Verify trades are rounded to lot sizes
        trades_by_symbol = {t.symbol: t for t in result.trades}

        aapl_trade = trades_by_symbol.get("AAPL")
        expected_qty = round((0.1 * 100000.0 / 150.0) / 10.0) * 10.0
        self.assertEqual(aapl_trade.quantity, expected_qty)

        msft_trade = trades_by_symbol.get("MSFT")
        expected_qty = round((0.05 * 100000.0 / 200.0) / 5.0) * 5.0
        self.assertEqual(msft_trade.quantity, expected_qty)

    def test_min_trade_value_filter(self):
        """Test that trades below minimum value are filtered out."""
        # Set up a scenario with a very small weight change
        target_weights = {
            "AAPL": 0.301,  # Just 0.1% increase, should be below min value
            "MSFT": 0.2,
            "GOOG": 0.1,
            "CASH": 0.399
        }

        # Set a higher min trade value
        trade_generator = TradeGenerator(min_trade_value=200.0)

        result = trade_generator.generate_trades(
            current_portfolio=self.portfolio,
            target_weights=target_weights
        )

        # We expect no trades since the only change is below min value
        self.assertEqual(len(result.trades), 0)

    def test_min_trade_quantity_filter(self):
        """Test that trades below minimum quantity are filtered out."""
        # Set up a scenario with a small weight change that results in < min quantity
        target_weights = {
            "AAPL": 0.3,
            "MSFT": 0.2,
            "GOOG": 0.101,  # 0.1% increase for expensive stock
            "CASH": 0.399
        }

        # With high-priced GOOG at $2500, a 0.1% change is 0.001 * 100000 / 2500 = 0.04 shares
        trade_generator = TradeGenerator(min_trade_quantity=0.1)

        result = trade_generator.generate_trades(
            current_portfolio=self.portfolio,
            target_weights=target_weights
        )

        # We expect no trades since the only change results in quantity below min
        self.assertEqual(len(result.trades), 0)

    def test_max_trade_value_cap(self):
        """Test that trades are capped at maximum value."""
        target_weights = {
            "AAPL": 0.5,  # 20% increase - would be $20,000
            "MSFT": 0.1,  # 10% decrease
            "GOOG": 0.1,
            "CASH": 0.3
        }

        # Set max trade value to $10,000
        trade_generator = TradeGenerator(max_trade_value=10000.0)

        result = trade_generator.generate_trades(
            current_portfolio=self.portfolio,
            target_weights=target_weights
        )

        # Check if AAPL trade is capped
        aapl_trade = next((t for t in result.trades if t.symbol == "AAPL"), None)
        self.assertIsNotNone(aapl_trade)

        # Expected quantity capped at max_value / price
        expected_qty = round(10000.0 / 150.0)
        self.assertEqual(aapl_trade.quantity, expected_qty)

    def test_generate_trades_empty_inputs(self):
        """Test generate_trades with empty inputs."""
        # Empty target weights
        result = self.trade_generator.generate_trades(
            current_portfolio=self.portfolio,
            target_weights={}
        )
        self.assertEqual(len(result.trades), 0)

    def test_generate_trades_invalid_prices(self):
        """Test generate_trades with invalid prices."""
        target_weights = {"AAPL": 0.4, "CASH": 0.6}

        # Invalid (zero) price
        prices = {"AAPL": 0.0}

        result = self.trade_generator.generate_trades(
            current_portfolio=self.portfolio,
            target_weights=target_weights,
            prices=prices
        )

        # Should skip the trade due to invalid price
        self.assertEqual(len(result.trades), 0)

    @patch('models.portfolio.trade_generator.PortfolioOptimizer')
    def test_generate_trades_from_signals(self, mock_optimizer_class):
        """Test generating trades from alpha signals."""
        # Set up mock optimizer
        mock_optimizer = MagicMock()
        mock_optimizer_class.return_value = mock_optimizer

        # Mock the optimizer to return some target weights
        mock_optimizer.alpha_to_weights.return_value = {
            "AAPL": 0.4,
            "MSFT": 0.15,
            "GOOG": 0.15,
            "CASH": 0.3
        }

        # Create alpha signals
        alpha_signals = {
            "AAPL": 0.8,
            "MSFT": 0.3,
            "GOOG": 0.5
        }

        # Call the method
        result = self.trade_generator.generate_trades_from_signals(
            current_portfolio=self.portfolio,
            alpha_signals=alpha_signals
        )

        # Verify optimizer was called correctly
        mock_optimizer.alpha_to_weights.assert_called_once_with(
            alpha_results=alpha_signals,
            current_portfolio=self.portfolio.weights,
            risk_model=None
        )

        # Check that trades were generated based on the optimizer's output
        self.assertEqual(len(result.trades), 3)

    def test_generate_rebalance_trades(self):
        """Test generating rebalance trades from portfolio target weights."""
        # Set target weights in the portfolio
        self.portfolio.target_weights = {
            "AAPL": 0.35,
            "MSFT": 0.25,
            "GOOG": 0.15,
            "CASH": 0.25
        }

        result = self.trade_generator.generate_rebalance_trades(self.portfolio)

        # Verify trades were generated using portfolio's target weights
        self.assertEqual(len(result.trades), 3)

        trades_by_symbol = {t.symbol: t for t in result.trades}

        # AAPL: +5% of portfolio value
        aapl_trade = trades_by_symbol.get("AAPL")
        self.assertEqual(aapl_trade.side, "BUY")
        self.assertEqual(aapl_trade.quantity, round(0.05 * 100000.0 / 150.0))

    def test_generate_rebalance_trades_no_target_weights(self):
        """Test generating rebalance trades when no target weights are set."""
        # Portfolio with no target weights
        self.portfolio.target_weights = None

        result = self.trade_generator.generate_rebalance_trades(self.portfolio)

        # Should return empty result
        self.assertEqual(len(result.trades), 0)
        self.assertEqual(result.total_value, 0.0)

    def test_split_large_trades(self):
        """Test splitting large trades into smaller chunks."""
        # Create a large trade
        large_trade = TradeInstruction(
            symbol="AAPL",
            quantity=1000.0,
            side="BUY",
            limit_price=150.0
        )

        # Split by value
        result = self.trade_generator.split_large_trades(
            trades=[large_trade],
            max_value=50000.0  # Max value $50k (less than 1000 * 150 = $150k)
        )

        # Should split into 3 trades (150k / 50k = 3)
        self.assertEqual(len(result), 3)

        # Check split quantities
        quantities = [t.quantity for t in result]
        self.assertEqual(sum(quantities), large_trade.quantity)

        # Check metadata
        self.assertEqual(result[0].metadata["parent_id"], large_trade.id)
        self.assertEqual(result[0].metadata["chunk"], 1)
        self.assertEqual(result[0].metadata["total_chunks"], 3)

    def test_split_large_trades_by_quantity(self):
        """Test splitting large trades based on max quantity."""
        large_trade = TradeInstruction(
            symbol="AAPL",
            quantity=1000.0,
            side="BUY",
            limit_price=150.0
        )

        # Split by quantity
        result = self.trade_generator.split_large_trades(
            trades=[large_trade],
            max_value=1000000.0,  # Very high max value (won't trigger)
            max_quantity=300.0    # Max quantity 300 shares
        )

        # Should split into 4 trades (1000 / 300 ≈ 3.33 → 4)
        self.assertEqual(len(result), 4)

        # Check quantities
        quantities = [t.quantity for t in result]
        self.assertEqual(sum(quantities), large_trade.quantity)

        # First 3 chunks should have equal quantities
        self.assertEqual(quantities[0], quantities[1])
        self.assertEqual(quantities[1], quantities[2])

    def test_split_large_trades_mixed_list(self):
        """Test splitting a mixed list of trades."""
        trades = [
            TradeInstruction(symbol="AAPL", quantity=1000.0, side="BUY", limit_price=150.0),
            TradeInstruction(symbol="MSFT", quantity=100.0, side="SELL", limit_price=200.0),
            TradeInstruction(symbol="GOOG", quantity=50.0, side="BUY", limit_price=2500.0)
        ]

        # Only the AAPL and GOOG trades exceed the max value
        result = self.trade_generator.split_large_trades(
            trades=trades,
            max_value=20000.0
        )

        # AAPL should split into several trades, MSFT remains one trade,
        # GOOG splits into several (because 50 * 2500 > 20000)
        self.assertTrue(len(result) > 3)

        # Check that we have the right total quantities
        quantities_by_symbol = {}
        for trade in result:
            if trade.symbol not in quantities_by_symbol:
                quantities_by_symbol[trade.symbol] = 0
            quantities_by_symbol[trade.symbol] += trade.quantity

        self.assertEqual(quantities_by_symbol["AAPL"], 1000.0)
        self.assertEqual(quantities_by_symbol["MSFT"], 100.0)
        self.assertEqual(quantities_by_symbol["GOOG"], 50.0)


if __name__ == "__main__":
    unittest.main()