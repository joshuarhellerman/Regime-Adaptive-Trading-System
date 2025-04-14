"""
Unit tests for the AlphaResult module.

This module tests the functionality of the AlphaResult class, AlphaResultCollection,
and related utility functions.
"""

import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest import mock

# Import the module to test
from models.alpha.alpha_result import (
    AlphaResult,
    AlphaResultCollection,
    SignalType,
    SignalDirection,
    SignalTimeframe,
    SignalStrength,
    AlphaSource,
    create_alpha_result_from_strategy,
    merge_alpha_results
)

# Import related components
from execution.order.order import OrderSide


class TestSignalDirection(unittest.TestCase):
    """Test the SignalDirection enum."""

    def test_to_order_side_conversion(self):
        """Test conversion from SignalDirection to OrderSide."""
        self.assertEqual(SignalDirection.LONG.to_order_side, OrderSide.BUY)
        self.assertEqual(SignalDirection.SHORT.to_order_side, OrderSide.SELL)

        # Test that neutral direction raises an error
        with self.assertRaises(ValueError):
            _ = SignalDirection.NEUTRAL.to_order_side


class TestAlphaResult(unittest.TestCase):
    """Test the AlphaResult class."""

    def setUp(self):
        """Set up test instances before each test."""
        # Create a basic AlphaResult instance
        self.basic_result = AlphaResult(
            symbol="AAPL",
            direction=SignalDirection.LONG,
            conviction=0.75
        )

        # Create a more detailed AlphaResult
        self.detailed_result = AlphaResult(
            symbol="MSFT",
            direction=SignalDirection.SHORT,
            conviction=0.85,
            model_id="model_123",
            strategy_id="strategy_456",
            signal_type=SignalType.MOMENTUM,
            timeframe=SignalTimeframe.SWING,
            source=AlphaSource.MACHINE_LEARNING,
            target_price=280.0,
            stop_price=300.0,
            expected_return=-0.05,
            expected_volatility=0.02,
            probability=0.7,
            expiration=datetime.utcnow() + timedelta(days=5),
            time_to_expiration=120.0,
            execution_window=4.0,
            urgency=0.8,
            max_impact=0.001,
            max_position_size=1000,
            risk_contribution=0.03,
            correlation_factor=0.2,
            features={"rsi": -2.1, "macd": -0.8},
            tags=["tech", "earnings"],
            notes="Earnings report coming up"
        )

    def test_initialization(self):
        """Test initialization of AlphaResult objects."""
        # Test basic initialization
        self.assertEqual(self.basic_result.symbol, "AAPL")
        self.assertEqual(self.basic_result.direction, SignalDirection.LONG)
        self.assertEqual(self.basic_result.conviction, 0.75)

        # Ensure defaults are set
        self.assertEqual(self.basic_result.signal_type, SignalType.DIRECTIONAL)
        self.assertIsNotNone(self.basic_result.signal_id)

        # Test detailed initialization
        self.assertEqual(self.detailed_result.symbol, "MSFT")
        self.assertEqual(self.detailed_result.features["rsi"], -2.1)
        self.assertEqual(len(self.detailed_result.tags), 2)

    def test_validation(self):
        """Test validation during initialization."""
        # Test conviction out of range
        with self.assertRaises(ValueError):
            AlphaResult(symbol="AAPL", direction=SignalDirection.LONG, conviction=1.5)

        with self.assertRaises(ValueError):
            AlphaResult(symbol="AAPL", direction=SignalDirection.LONG, conviction=-0.1)

        # Test urgency out of range
        with self.assertRaises(ValueError):
            AlphaResult(
                symbol="AAPL",
                direction=SignalDirection.LONG,
                conviction=0.5,
                urgency=1.5
            )

    def test_strength_category(self):
        """Test the strength_category property."""
        # Test various conviction levels
        test_cases = [
            (0.1, SignalStrength.VERY_WEAK),
            (0.3, SignalStrength.WEAK),
            (0.5, SignalStrength.MODERATE),
            (0.7, SignalStrength.STRONG),
            (0.9, SignalStrength.VERY_STRONG)
        ]

        for conviction, expected_strength in test_cases:
            result = AlphaResult(
                symbol="TEST",
                direction=SignalDirection.LONG,
                conviction=conviction
            )
            self.assertEqual(result.strength_category, expected_strength)

    def test_is_valid(self):
        """Test the is_valid property."""
        # No expiration should be valid
        self.assertTrue(self.basic_result.is_valid)

        # Future expiration should be valid
        future_result = AlphaResult(
            symbol="AAPL",
            direction=SignalDirection.LONG,
            conviction=0.5,
            expiration=datetime.utcnow() + timedelta(hours=1)
        )
        self.assertTrue(future_result.is_valid)

        # Past expiration should be invalid
        past_result = AlphaResult(
            symbol="AAPL",
            direction=SignalDirection.LONG,
            conviction=0.5,
            expiration=datetime.utcnow() - timedelta(hours=1)
        )
        self.assertFalse(past_result.is_valid)

    def test_time_since_generation(self):
        """Test the time_since_generation property."""
        # Mock the current time
        with mock.patch('models.alpha.alpha_result.datetime') as mock_datetime:
            # Set the "now" time
            mock_now = datetime(2023, 1, 1, 12, 0, 0)
            mock_datetime.utcnow.return_value = mock_now

            # Create a result with a timestamp 2 hours ago
            result = AlphaResult(
                symbol="AAPL",
                direction=SignalDirection.LONG,
                conviction=0.5,
                timestamp=mock_now - timedelta(hours=2)
            )

            # The time since generation should be 2 hours
            self.assertEqual(result.time_since_generation, 2.0)

    def test_primary_feature(self):
        """Test the primary_feature property."""
        # No features
        self.assertIsNone(self.basic_result.primary_feature)

        # With features
        self.assertEqual(self.detailed_result.primary_feature, ("rsi", -2.1))

        # Test absolute values
        result = AlphaResult(
            symbol="TEST",
            direction=SignalDirection.LONG,
            conviction=0.5,
            features={"f1": 1.0, "f2": -3.0, "f3": 2.0}
        )
        self.assertEqual(result.primary_feature, ("f2", -3.0))

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result_dict = self.detailed_result.to_dict()

        # Check core fields
        self.assertEqual(result_dict["symbol"], "MSFT")
        self.assertEqual(result_dict["direction"], "short")
        self.assertEqual(result_dict["conviction"], 0.85)

        # Check nested structures
        self.assertEqual(result_dict["features"]["rsi"], -2.1)
        self.assertEqual(result_dict["tags"], ["tech", "earnings"])

        # Check computed properties
        self.assertEqual(result_dict["strength_category"], "VERY_STRONG")
        self.assertIn("is_valid", result_dict)

    def test_from_dict(self):
        """Test creation from dictionary."""
        # Convert to dict and back
        result_dict = self.detailed_result.to_dict()
        reconstructed = AlphaResult.from_dict(result_dict)

        # Verify core properties
        self.assertEqual(reconstructed.symbol, self.detailed_result.symbol)
        self.assertEqual(reconstructed.direction, self.detailed_result.direction)
        self.assertEqual(reconstructed.conviction, self.detailed_result.conviction)

        # Verify enum conversions
        self.assertEqual(reconstructed.signal_type, SignalType.MOMENTUM)
        self.assertEqual(reconstructed.timeframe, SignalTimeframe.SWING)

        # Verify nested structures
        self.assertEqual(reconstructed.features["rsi"], -2.1)
        self.assertEqual(reconstructed.tags, ["tech", "earnings"])

        # Verify timestamps converted correctly
        self.assertIsInstance(reconstructed.timestamp, datetime)
        self.assertIsInstance(reconstructed.expiration, datetime)

    def test_str_representation(self):
        """Test string representation."""
        expected_str = "Directional long signal for AAPL: conviction=0.75, timeframe=daily"
        self.assertEqual(str(self.basic_result), expected_str)


class TestAlphaResultCollection(unittest.TestCase):
    """Test the AlphaResultCollection class."""

    def setUp(self):
        """Set up test instances before each test."""
        # Create several AlphaResult instances
        self.result1 = AlphaResult(
            symbol="AAPL",
            direction=SignalDirection.LONG,
            conviction=0.75
        )

        self.result2 = AlphaResult(
            symbol="MSFT",
            direction=SignalDirection.SHORT,
            conviction=0.6
        )

        self.result3 = AlphaResult(
            symbol="AAPL",
            direction=SignalDirection.SHORT,
            conviction=0.55,
            expiration=datetime.utcnow() - timedelta(hours=1)  # Expired
        )

        # Create collection
        self.collection = AlphaResultCollection([self.result1, self.result2, self.result3])

    def test_initialization(self):
        """Test initialization of AlphaResultCollection."""
        # Test empty initialization
        empty_collection = AlphaResultCollection()
        self.assertEqual(empty_collection.count, 0)

        # Test initialization with signals
        self.assertEqual(self.collection.count, 3)
        self.assertIsNotNone(self.collection.collection_id)
        self.assertIsInstance(self.collection.timestamp, datetime)

    def test_add_signal(self):
        """Test adding signals to collection."""
        # Create new collection
        collection = AlphaResultCollection()
        self.assertEqual(collection.count, 0)

        # Add a signal
        collection.add_signal(self.result1)
        self.assertEqual(collection.count, 1)

        # Add another signal
        collection.add_signal(self.result2)
        self.assertEqual(collection.count, 2)

    def test_remove_signal(self):
        """Test removing signals from collection."""
        # Get initial count
        initial_count = self.collection.count

        # Remove existing signal
        signal_id = self.result1.signal_id
        result = self.collection.remove_signal(signal_id)

        # Check result and count
        self.assertTrue(result)
        self.assertEqual(self.collection.count, initial_count - 1)

        # Try to remove non-existent signal
        result = self.collection.remove_signal("non_existent_id")
        self.assertFalse(result)
        self.assertEqual(self.collection.count, initial_count - 1)

    def test_get_signal(self):
        """Test getting signals by ID."""
        # Get existing signal
        signal_id = self.result1.signal_id
        signal = self.collection.get_signal(signal_id)
        self.assertEqual(signal, self.result1)

        # Try to get non-existent signal
        signal = self.collection.get_signal("non_existent_id")
        self.assertIsNone(signal)

    def test_get_signals_by_symbol(self):
        """Test getting signals by symbol."""
        # Get signals for AAPL
        aapl_signals = self.collection.get_signals_by_symbol("AAPL")
        self.assertEqual(len(aapl_signals), 2)
        self.assertEqual({s.symbol for s in aapl_signals}, {"AAPL"})

        # Get signals for MSFT
        msft_signals = self.collection.get_signals_by_symbol("MSFT")
        self.assertEqual(len(msft_signals), 1)
        self.assertEqual(msft_signals[0].symbol, "MSFT")

        # Get signals for non-existent symbol
        none_signals = self.collection.get_signals_by_symbol("GOOG")
        self.assertEqual(len(none_signals), 0)

    def test_get_signals_by_direction(self):
        """Test getting signals by direction."""
        # Get long signals
        long_signals = self.collection.get_signals_by_direction(SignalDirection.LONG)
        self.assertEqual(len(long_signals), 1)
        self.assertEqual(long_signals[0].direction, SignalDirection.LONG)

        # Get short signals
        short_signals = self.collection.get_signals_by_direction(SignalDirection.SHORT)
        self.assertEqual(len(short_signals), 2)
        self.assertTrue(all(s.direction == SignalDirection.SHORT for s in short_signals))

    def test_symbols_property(self):
        """Test the symbols property."""
        # Check symbols list
        symbols = self.collection.symbols
        self.assertEqual(len(symbols), 2)
        self.assertEqual(set(symbols), {"AAPL", "MSFT"})
        self.assertEqual(symbols, sorted(symbols))  # Ensure sorted

    def test_count_property(self):
        """Test the count property."""
        self.assertEqual(self.collection.count, 3)

    def test_valid_signals_property(self):
        """Test the valid_signals property."""
        # Only two signals are valid (one is expired)
        valid_signals = self.collection.valid_signals
        self.assertEqual(len(valid_signals), 2)

        # All valid signals should have is_valid=True
        self.assertTrue(all(s.is_valid for s in valid_signals))

    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        # Convert to DataFrame
        df = self.collection.to_dataframe()

        # Check DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)  # 3 rows

        # Check columns
        self.assertIn("symbol", df.columns)
        self.assertIn("direction", df.columns)
        self.assertIn("conviction", df.columns)

        # Check values
        symbols = df["symbol"].tolist()
        self.assertEqual(set(symbols), {"AAPL", "MSFT"})

    def test_to_dict(self):
        """Test conversion to dictionary."""
        # Convert to dict
        collection_dict = self.collection.to_dict()

        # Check dict structure
        self.assertIn("collection_id", collection_dict)
        self.assertIn("timestamp", collection_dict)
        self.assertIn("count", collection_dict)
        self.assertIn("symbols", collection_dict)
        self.assertIn("signals", collection_dict)

        # Check values
        self.assertEqual(collection_dict["count"], 3)
        self.assertEqual(set(collection_dict["symbols"]), {"AAPL", "MSFT"})
        self.assertEqual(len(collection_dict["signals"]), 3)

    def test_from_dict(self):
        """Test creation from dictionary."""
        # Convert to dict and back
        collection_dict = self.collection.to_dict()
        reconstructed = AlphaResultCollection.from_dict(collection_dict)

        # Check core properties
        self.assertEqual(reconstructed.collection_id, self.collection.collection_id)
        self.assertEqual(reconstructed.count, self.collection.count)
        self.assertEqual(set(reconstructed.symbols), set(self.collection.symbols))

        # Check signals reconstruction
        self.assertEqual(len(reconstructed.signals), len(self.collection.signals))

        # Get first signal from each and compare
        orig_signal = self.collection.signals[0]
        reconstructed_signal = reconstructed.signals[0]
        self.assertEqual(reconstructed_signal.symbol, orig_signal.symbol)
        self.assertEqual(reconstructed_signal.direction, orig_signal.direction)

    def test_str_representation(self):
        """Test string representation."""
        expected_str = "AlphaResultCollection with 3 signals for 2 symbols"
        self.assertEqual(str(self.collection), expected_str)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions for alpha results."""

    def test_create_alpha_result_from_strategy(self):
        """Test creation of AlphaResult from strategy output."""
        # Test long signal
        strategy_result = {
            "signal": "long",
            "conviction": 0.8,
            "target_price": 200.0,
            "stop_loss": 180.0,
            "expected_return": 0.1,
            "probability": 0.75,
            "features": {"rsi": 70, "macd": 2.5},
            "notes": "Breakout pattern"
        }

        result = create_alpha_result_from_strategy(
            strategy_result=strategy_result,
            strategy_id="test_strategy",
            symbol="AAPL"
        )

        # Check core properties
        self.assertEqual(result.symbol, "AAPL")
        self.assertEqual(result.direction, SignalDirection.LONG)
        self.assertEqual(result.conviction, 0.8)
        self.assertEqual(result.strategy_id, "test_strategy")

        # Check additional properties
        self.assertEqual(result.target_price, 200.0)
        self.assertEqual(result.stop_price, 180.0)
        self.assertEqual(result.expected_return, 0.1)
        self.assertEqual(result.probability, 0.75)
        self.assertEqual(result.features["rsi"], 70)
        self.assertEqual(result.notes, "Breakout pattern")

        # Test short signal
        strategy_result = {
            "signal": "short",
            "conviction": 0.6
        }

        result = create_alpha_result_from_strategy(
            strategy_result=strategy_result,
            strategy_id="test_strategy",
            symbol="AAPL"
        )

        self.assertEqual(result.direction, SignalDirection.SHORT)

        # Test neutral signal
        strategy_result = {
            "signal": "neutral",
            "conviction": 0.3
        }

        result = create_alpha_result_from_strategy(
            strategy_result=strategy_result,
            strategy_id="test_strategy",
            symbol="AAPL"
        )

        self.assertEqual(result.direction, SignalDirection.NEUTRAL)

        # Test missing signal
        strategy_result = {
            "conviction": 0.5
        }

        result = create_alpha_result_from_strategy(
            strategy_result=strategy_result,
            strategy_id="test_strategy",
            symbol="AAPL"
        )

        self.assertEqual(result.direction, SignalDirection.NEUTRAL)

    def test_merge_alpha_results_empty(self):
        """Test merging with empty list."""
        with self.assertRaises(ValueError):
            merge_alpha_results([], "AAPL")

    def test_merge_alpha_results_wrong_symbol(self):
        """Test merging with wrong symbol."""
        result = AlphaResult(
            symbol="MSFT",
            direction=SignalDirection.LONG,
            conviction=0.5
        )

        with self.assertRaises(ValueError):
            merge_alpha_results([result], "AAPL")

    def test_merge_alpha_results_single(self):
        """Test merging with single result."""
        result = AlphaResult(
            symbol="AAPL",
            direction=SignalDirection.LONG,
            conviction=0.5
        )

        merged = merge_alpha_results([result], "AAPL")
        self.assertEqual(merged, result)  # Should return the same object

    def test_merge_alpha_results_conviction_weighted(self):
        """Test merging with conviction_weighted method."""
        # Create results with opposing directions
        result1 = AlphaResult(
            symbol="AAPL",
            direction=SignalDirection.LONG,
            conviction=0.8,
            target_price=200.0,
            expected_return=0.1,
            tags=["momentum"]
        )

        result2 = AlphaResult(
            symbol="AAPL",
            direction=SignalDirection.SHORT,
            conviction=0.2,
            target_price=180.0,
            expected_return=-0.05,
            tags=["reversal"]
        )

        # Merge results
        merged = merge_alpha_results(
            [result1, result2],
            "AAPL",
            merge_method="conviction_weighted"
        )

        # Check direction and conviction
        self.assertEqual(merged.direction, SignalDirection.LONG)  # 0.8 vs 0.2
        self.assertEqual(merged.conviction, 0.6)  # |0.8 - 0.2| / 1.0

        # Check averaged properties
        self.assertEqual(merged.target_price, 190.0)  # (200 + 180) / 2
        self.assertEqual(merged.expected_return, 0.025)  # (0.1 + -0.05) / 2

        # Check combined tags
        self.assertEqual(set(merged.tags), {"momentum", "reversal"})

        # Check metadata
        self.assertIn("source_signals", merged.metadata)
        self.assertEqual(merged.metadata["merge_method"], "conviction_weighted")

    def test_merge_alpha_results_majority_vote(self):
        """Test merging with majority_vote method."""
        # Create 3 results, 2 long and 1 short
        result1 = AlphaResult(
            symbol="AAPL",
            direction=SignalDirection.LONG,
            conviction=0.5
        )

        result2 = AlphaResult(
            symbol="AAPL",
            direction=SignalDirection.LONG,
            conviction=0.6
        )

        result3 = AlphaResult(
            symbol="AAPL",
            direction=SignalDirection.SHORT,
            conviction=0.9
        )

        # Merge results
        merged = merge_alpha_results(
            [result1, result2, result3],
            "AAPL",
            merge_method="majority_vote"
        )

        # Check direction and conviction
        self.assertEqual(merged.direction, SignalDirection.LONG)  # 2 vs 1
        self.assertEqual(merged.conviction, 2/3)  # 2 out of 3 votes

    def test_merge_alpha_results_highest_conviction(self):
        """Test merging with highest_conviction method."""
        # Create 3 results with varying convictions
        result1 = AlphaResult(
            symbol="AAPL",
            direction=SignalDirection.LONG,
            conviction=0.5
        )

        result2 = AlphaResult(
            symbol="AAPL",
            direction=SignalDirection.LONG,
            conviction=0.7
        )

        result3 = AlphaResult(
            symbol="AAPL",
            direction=SignalDirection.SHORT,
            conviction=0.9
        )

        # Merge results
        merged = merge_alpha_results(
            [result1, result2, result3],
            "AAPL",
            merge_method="highest_conviction"
        )

        # Should select result3 since it has highest conviction
        self.assertEqual(merged.direction, SignalDirection.SHORT)
        self.assertEqual(merged.conviction, 0.9)

    def test_merge_alpha_results_unknown_method(self):
        """Test merging with unknown method."""
        result = AlphaResult(
            symbol="AAPL",
            direction=SignalDirection.LONG,
            conviction=0.5
        )

        with self.assertRaises(ValueError):
            merge_alpha_results([result], "AAPL", merge_method="unknown_method")


if __name__ == "__main__":
    unittest.main()