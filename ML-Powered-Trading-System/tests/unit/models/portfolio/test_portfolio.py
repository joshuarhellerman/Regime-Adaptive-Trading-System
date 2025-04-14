import unittest
from datetime import datetime
from unittest.mock import patch
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal

from models.portfolio.portfolio import Portfolio, Position


class TestPosition(unittest.TestCase):
    """Unit tests for the Position class."""

    def test_position_initialization(self):
        """Test Position initialization with default and custom values."""
        # Test with minimal required arguments
        pos1 = Position("AAPL", 10, 150.0)
        self.assertEqual(pos1.symbol, "AAPL")
        self.assertEqual(pos1.quantity, 10)
        self.assertEqual(pos1.current_price, 150.0)
        self.assertEqual(pos1.cost_basis, 0.0)
        self.assertIsNone(pos1.sector)
        self.assertIsInstance(pos1.last_update, datetime)

        # Test with all arguments
        test_date = datetime(2023, 1, 1)
        pos2 = Position("MSFT", 5, 250.0, 240.0, "Technology", test_date)
        self.assertEqual(pos2.symbol, "MSFT")
        self.assertEqual(pos2.quantity, 5)
        self.assertEqual(pos2.current_price, 250.0)
        self.assertEqual(pos2.cost_basis, 240.0)
        self.assertEqual(pos2.sector, "Technology")
        self.assertEqual(pos2.last_update, test_date)

    def test_position_properties(self):
        """Test Position property calculations."""
        pos = Position("GOOG", 2, 2000.0, 1800.0)

        # Test market_value property
        self.assertEqual(pos.market_value, 4000.0)

        # Test pnl property
        self.assertEqual(pos.pnl, 400.0)  # (2000 - 1800) * 2

        # Test pnl_percentage property
        self.assertAlmostEqual(pos.pnl_percentage, 11.11111111111111)  # (2000/1800 - 1) * 100

        # Test zero cost basis case
        pos_zero_cost = Position("ZERO", 5, 100.0, 0.0)
        self.assertEqual(pos_zero_cost.pnl_percentage, 0.0)

        # Test zero quantity case
        pos_zero_qty = Position("ZERO", 0, 100.0, 50.0)
        self.assertEqual(pos_zero_qty.pnl_percentage, 0.0)


class TestPortfolio(unittest.TestCase):
    """Unit tests for the Portfolio class."""

    def setUp(self):
        """Set up common test fixtures."""
        self.portfolio = Portfolio(cash=10000.0, name="Test Portfolio")

        # Add some positions
        self.portfolio.add_position(Position("AAPL", 10, 150.0, 140.0, "Technology"))
        self.portfolio.add_position(Position("MSFT", 5, 250.0, 240.0, "Technology"))
        self.portfolio.add_position(Position("JNJ", 8, 170.0, 165.0, "Healthcare"))

    def test_init(self):
        """Test Portfolio initialization."""
        portfolio = Portfolio(cash=5000.0, name="New Portfolio")
        self.assertEqual(portfolio.cash, 5000.0)
        self.assertEqual(portfolio.name, "New Portfolio")
        self.assertEqual(len(portfolio.positions), 0)
        self.assertEqual(portfolio.target_weights, {})
        self.assertIsInstance(portfolio.last_update, datetime)

    def test_add_position_new(self):
        """Test adding a new position to the portfolio."""
        portfolio = Portfolio(1000.0)
        position = Position("GOOG", 1, 2000.0, 1900.0, "Technology")

        # Add new position
        portfolio.add_position(position)

        # Verify position was added
        self.assertIn("GOOG", portfolio.positions)
        self.assertEqual(portfolio.positions["GOOG"].quantity, 1)
        self.assertEqual(portfolio.positions["GOOG"].cost_basis, 1900.0)

    def test_add_position_existing(self):
        """Test adding to an existing position in the portfolio."""
        # Initial state
        self.assertEqual(self.portfolio.positions["AAPL"].quantity, 10)
        self.assertEqual(self.portfolio.positions["AAPL"].cost_basis, 140.0)

        # Add more to the position with different cost basis
        new_position = Position("AAPL", 5, 155.0, 150.0, "Technology")
        self.portfolio.add_position(new_position)

        # Verify position was updated with weighted average cost basis
        self.assertEqual(self.portfolio.positions["AAPL"].quantity, 15)
        expected_cost_basis = ((10 * 140.0) + (5 * 150.0)) / 15
        self.assertAlmostEqual(self.portfolio.positions["AAPL"].cost_basis, expected_cost_basis)
        self.assertEqual(self.portfolio.positions["AAPL"].current_price, 155.0)

    def test_add_position_zero_total(self):
        """Test case where adding a position results in zero total quantity."""
        # Add negative position that cancels out existing
        negative_position = Position("AAPL", -10, 155.0, 155.0, "Technology")
        self.portfolio.add_position(negative_position)

        # Position should still exist but with quantity 0
        self.assertIn("AAPL", self.portfolio.positions)
        self.assertEqual(self.portfolio.positions["AAPL"].quantity, 0)

    def test_update_prices(self):
        """Test updating prices for multiple positions."""
        new_prices = {
            "AAPL": 160.0,
            "MSFT": 260.0,
            "JNJ": 175.0
        }

        self.portfolio.update_prices(new_prices)

        # Verify prices were updated
        self.assertEqual(self.portfolio.positions["AAPL"].current_price, 160.0)
        self.assertEqual(self.portfolio.positions["MSFT"].current_price, 260.0)
        self.assertEqual(self.portfolio.positions["JNJ"].current_price, 175.0)

    def test_total_value(self):
        """Test total portfolio value calculation."""
        # Expected: (10*150 + 5*250 + 8*170) + 10000 = 13850
        expected_value = (10 * 150.0) + (5 * 250.0) + (8 * 170.0) + 10000
        self.assertEqual(self.portfolio.total_value, expected_value)

    def test_market_value(self):
        """Test market value calculation (excluding cash)."""
        # Expected: 10*150 + 5*250 + 8*170 = 3850
        expected_value = (10 * 150.0) + (5 * 250.0) + (8 * 170.0)
        self.assertEqual(self.portfolio.market_value, expected_value)

    def test_weights(self):
        """Test portfolio weight calculations."""
        total_value = self.portfolio.total_value
        expected_weights = {
            "AAPL": (10 * 150.0) / total_value,
            "MSFT": (5 * 250.0) / total_value,
            "JNJ": (8 * 170.0) / total_value,
            "CASH": 10000.0 / total_value
        }

        weights = self.portfolio.weights

        for symbol, expected_weight in expected_weights.items():
            self.assertAlmostEqual(weights[symbol], expected_weight)

    def test_weights_empty_portfolio(self):
        """Test weight calculations for an empty portfolio."""
        empty_portfolio = Portfolio(0.0)
        weights = empty_portfolio.weights

        self.assertEqual(weights, {"CASH": 0.0})

    def test_get_position_value(self):
        """Test getting value of a specific position."""
        # Expected: 10 * 150.0 = 1500
        self.assertEqual(self.portfolio.get_position_value("AAPL"), 1500.0)

        # Non-existent position
        self.assertEqual(self.portfolio.get_position_value("NONEXISTENT"), 0.0)

    def test_get_position_weight(self):
        """Test getting weight of a specific position."""
        total_value = self.portfolio.total_value
        expected_weight = (10 * 150.0) / total_value

        self.assertAlmostEqual(self.portfolio.get_position_weight("AAPL"), expected_weight)

        # Non-existent position
        self.assertEqual(self.portfolio.get_position_weight("NONEXISTENT"), 0.0)

    def test_set_target_weights_valid(self):
        """Test setting valid target weights."""
        target_weights = {
            "AAPL": 0.3,
            "MSFT": 0.2,
            "JNJ": 0.2,
            "CASH": 0.3
        }

        self.portfolio.set_target_weights(target_weights)
        self.assertEqual(self.portfolio.target_weights, target_weights)

    def test_set_target_weights_invalid(self):
        """Test setting invalid target weights (sum != 1.0)."""
        target_weights = {
            "AAPL": 0.4,
            "MSFT": 0.4,
            "JNJ": 0.4,
        }

        with self.assertRaises(ValueError):
            self.portfolio.set_target_weights(target_weights)

    def test_get_rebalance_trades_no_targets(self):
        """Test rebalance calculation with no target weights set."""
        trades = self.portfolio.get_rebalance_trades()
        self.assertEqual(trades, {})

    def test_get_rebalance_trades(self):
        """Test rebalance trade calculations."""
        # Set target weights
        target_weights = {
            "AAPL": 0.2,
            "MSFT": 0.3,
            "JNJ": 0.1,
            "CASH": 0.4
        }
        self.portfolio.set_target_weights(target_weights)

        # Calculate expected trades
        total_value = self.portfolio.total_value
        trades = self.portfolio.get_rebalance_trades()

        # Expected trades for each position
        expected_aapl_value = total_value * 0.2
        current_aapl_value = 10 * 150.0
        expected_aapl_diff = expected_aapl_value - current_aapl_value
        expected_aapl_qty = expected_aapl_diff / 150.0

        expected_msft_value = total_value * 0.3
        current_msft_value = 5 * 250.0
        expected_msft_diff = expected_msft_value - current_msft_value
        expected_msft_qty = expected_msft_diff / 250.0

        expected_jnj_value = total_value * 0.1
        current_jnj_value = 8 * 170.0
        expected_jnj_diff = expected_jnj_value - current_jnj_value
        expected_jnj_qty = expected_jnj_diff / 170.0

        expected_cash_diff = (total_value * 0.4) - 10000.0

        # Check trades (allowing for small floating-point differences)
        self.assertAlmostEqual(trades["AAPL"], expected_aapl_qty)
        self.assertAlmostEqual(trades["MSFT"], expected_msft_qty)
        self.assertAlmostEqual(trades["JNJ"], expected_jnj_qty)
        self.assertAlmostEqual(trades["CASH"], expected_cash_diff)

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        df = self.portfolio.to_dataframe()

        # Check DataFrame shape
        self.assertEqual(df.shape[0], 4)  # 3 positions + cash
        self.assertEqual(df.shape[1], 9)  # 9 columns

        # Check all positions and cash are included
        symbols = set(df['symbol'].values)
        self.assertEqual(symbols, {"AAPL", "MSFT", "JNJ", "CASH"})

        # Check some values
        aapl_row = df[df['symbol'] == 'AAPL'].iloc[0]
        self.assertEqual(aapl_row['quantity'], 10)
        self.assertEqual(aapl_row['price'], 150.0)
        self.assertEqual(aapl_row['market_value'], 1500.0)
        self.assertEqual(aapl_row['sector'], 'Technology')

    def test_to_dataframe_empty(self):
        """Test conversion to DataFrame with an empty portfolio."""
        empty_portfolio = Portfolio(1000.0)
        df = empty_portfolio.to_dataframe()

        # Check DataFrame shape for empty portfolio (just cash)
        self.assertEqual(df.shape[0], 1)
        self.assertEqual(df['symbol'].iloc[0], 'CASH')

    def test_get_sector_exposures(self):
        """Test sector exposure calculations."""
        exposures = self.portfolio.get_sector_exposures()
        total_value = self.portfolio.total_value

        # Expected exposures
        expected_exposures = {
            "Technology": ((10 * 150.0) + (5 * 250.0)) / total_value,
            "Healthcare": (8 * 170.0) / total_value,
            "Cash": 10000.0 / total_value
        }

        for sector, expected in expected_exposures.items():
            self.assertAlmostEqual(exposures[sector], expected)

    def test_str_representation(self):
        """Test string representation of portfolio."""
        expected_str = (f"Portfolio 'Test Portfolio': 3 positions, "
                        f"${self.portfolio.market_value:.2f} invested, "
                        f"${self.portfolio.cash:.2f} cash, "
                        f"${self.portfolio.total_value:.2f} total")
        self.assertEqual(str(self.portfolio), expected_str)

    def test_summary(self):
        """Test portfolio summary generation."""
        summary = self.portfolio.summary()

        # Check that summary contains key information
        self.assertIn("Portfolio Summary: Test Portfolio", summary)
        self.assertIn(f"Total Value: ${self.portfolio.total_value:.2f}", summary)
        self.assertIn("Top Positions:", summary)
        self.assertIn("Sector Allocation:", summary)
        self.assertIn("Technology:", summary)
        self.assertIn("Healthcare:", summary)
        self.assertIn("Cash:", summary)


if __name__ == "__main__":
    unittest.main()