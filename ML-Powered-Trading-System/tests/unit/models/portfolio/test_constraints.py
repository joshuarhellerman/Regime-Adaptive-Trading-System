import unittest
import numpy as np
from typing import Dict
import pandas as pd

from models.portfolio.constraints import (
    PortfolioConstraint,
    LeverageConstraint,
    PositionSizeConstraint,
    SectorConstraint,
    TurnoverConstraint
)


class TestLeverageConstraint(unittest.TestCase):
    """Test suite for LeverageConstraint."""

    def test_init(self):
        """Test initialization parameters."""
        # Test with default parameters
        constraint = LeverageConstraint()
        self.assertEqual(constraint.max_leverage, 1.0)
        self.assertTrue(constraint.exclude_cash)

        # Test with custom parameters
        constraint = LeverageConstraint(max_leverage=2.0, exclude_cash=False)
        self.assertEqual(constraint.max_leverage, 2.0)
        self.assertFalse(constraint.exclude_cash)

        # Test with invalid parameters
        with self.assertRaises(ValueError):
            LeverageConstraint(max_leverage=0)
        with self.assertRaises(ValueError):
            LeverageConstraint(max_leverage=-1.0)

    def test_is_satisfied(self):
        """Test constraint satisfaction check."""
        constraint = LeverageConstraint(max_leverage=1.5, exclude_cash=True)

        # Test weights with leverage within limit
        weights = {'A': 0.3, 'B': 0.4, 'C': -0.2, 'CASH': 0.5}
        self.assertTrue(constraint.is_satisfied(weights))

        # Test weights with leverage at the limit
        weights = {'A': 0.5, 'B': 0.5, 'C': -0.5, 'CASH': 0.5}
        self.assertTrue(constraint.is_satisfied(weights))

        # Test weights with leverage exceeding limit
        weights = {'A': 0.8, 'B': 0.7, 'C': -0.5, 'CASH': 0.0}
        self.assertFalse(constraint.is_satisfied(weights))

        # Test without cash exclusion
        constraint = LeverageConstraint(max_leverage=1.5, exclude_cash=False)
        weights = {'A': 0.3, 'B': 0.4, 'C': -0.2, 'CASH': 0.5}
        self.assertTrue(constraint.is_satisfied(weights))

        weights = {'A': 0.8, 'B': 0.7, 'C': -0.5, 'CASH': 0.5}
        self.assertFalse(constraint.is_satisfied(weights))

    def test_apply(self):
        """Test constraint application."""
        constraint = LeverageConstraint(max_leverage=1.0, exclude_cash=True)

        # Test weights already satisfying constraint
        original_weights = {'A': 0.3, 'B': 0.4, 'C': -0.2, 'CASH': 0.5}
        weights = constraint.apply(original_weights)
        self.assertEqual(weights, original_weights)

        # Test weights requiring scaling
        original_weights = {'A': 0.6, 'B': 0.5, 'C': -0.3, 'CASH': 0.2}
        weights = constraint.apply(original_weights)

        # Check that leverage is now at or below max
        non_cash_sum = sum(abs(weights[s]) for s in weights if s != 'CASH')
        self.assertLessEqual(non_cash_sum, constraint.max_leverage)

        # Check that weights sum to 1.0
        self.assertAlmostEqual(sum(weights.values()), 1.0)

        # Test without cash exclusion
        constraint = LeverageConstraint(max_leverage=1.2, exclude_cash=False)
        original_weights = {'A': 0.6, 'B': 0.5, 'C': -0.3, 'CASH': 0.2}
        weights = constraint.apply(original_weights)

        # Check that total leverage is now at or below max
        leverage = sum(abs(weights[s]) for s in weights)
        self.assertLessEqual(leverage, constraint.max_leverage)

        # Proportions should be maintained
        self.assertAlmostEqual(weights['A'] / original_weights['A'],
                               weights['B'] / original_weights['B'],
                               places=6)


class TestPositionSizeConstraint(unittest.TestCase):
    """Test suite for PositionSizeConstraint."""

    def test_init(self):
        """Test initialization parameters."""
        # Test with default parameters
        constraint = PositionSizeConstraint()
        self.assertEqual(constraint.max_size, 0.2)
        self.assertEqual(constraint.min_size, 0.0)
        self.assertTrue(constraint.exclude_cash)

        # Test with custom parameters
        constraint = PositionSizeConstraint(max_size=0.3, min_size=0.05, exclude_cash=False)
        self.assertEqual(constraint.max_size, 0.3)
        self.assertEqual(constraint.min_size, 0.05)
        self.assertFalse(constraint.exclude_cash)

        # Test with invalid parameters
        with self.assertRaises(ValueError):
            PositionSizeConstraint(max_size=0)
        with self.assertRaises(ValueError):
            PositionSizeConstraint(max_size=1.1)
        with self.assertRaises(ValueError):
            PositionSizeConstraint(min_size=-0.1)
        with self.assertRaises(ValueError):
            PositionSizeConstraint(max_size=0.2, min_size=0.3)

    def test_is_satisfied(self):
        """Test constraint satisfaction check."""
        constraint = PositionSizeConstraint(max_size=0.3, min_size=0.05, exclude_cash=True)

        # Test weights with positions within limits
        weights = {'A': 0.25, 'B': 0.2, 'C': -0.15, 'D': 0.1, 'CASH': 0.3}
        self.assertTrue(constraint.is_satisfied(weights))

        # Test weights with positions exceeding max size
        weights = {'A': 0.35, 'B': 0.2, 'C': -0.15, 'D': 0.1, 'CASH': 0.2}
        self.assertFalse(constraint.is_satisfied(weights))

        # Test weights with positions below min size
        weights = {'A': 0.25, 'B': 0.2, 'C': -0.03, 'D': 0.1, 'CASH': 0.42}
        self.assertFalse(constraint.is_satisfied(weights))

        # Test with cash excluded
        constraint = PositionSizeConstraint(max_size=0.3, min_size=0.05, exclude_cash=False)
        weights = {'A': 0.25, 'B': 0.2, 'C': -0.15, 'D': 0.1, 'CASH': 0.5}
        self.assertFalse(constraint.is_satisfied(weights))  # Cash exceeds max_size

    def test_apply(self):
        """Test constraint application."""
        constraint = PositionSizeConstraint(max_size=0.25, min_size=0.05, exclude_cash=True)

        # Test weights already satisfying constraint
        original_weights = {'A': 0.20, 'B': 0.15, 'C': -0.15, 'D': 0.1, 'CASH': 0.4}
        weights = constraint.apply(original_weights)
        self.assertEqual(weights, original_weights)

        # Test weights with positions exceeding max size
        original_weights = {'A': 0.35, 'B': 0.15, 'C': -0.15, 'D': 0.1, 'CASH': 0.25}
        weights = constraint.apply(original_weights)

        # Check max size constraint
        for symbol, weight in weights.items():
            if symbol != 'CASH':
                self.assertLessEqual(abs(weight), constraint.max_size)

        # Check weights sum to 1.0
        self.assertAlmostEqual(sum(weights.values()), 1.0)

        # Test weights with positions below min size
        original_weights = {'A': 0.25, 'B': 0.15, 'C': -0.03, 'D': 0.1, 'CASH': 0.47}
        weights = constraint.apply(original_weights)

        # Check min size constraint
        for symbol, weight in weights.items():
            if symbol != 'CASH' and abs(weight) > 0:
                self.assertGreaterEqual(abs(weight), constraint.min_size)

        # Check weights sum to 1.0
        self.assertAlmostEqual(sum(weights.values()), 1.0)

        # Test with zero weights
        original_weights = {'A': 0.25, 'B': 0.15, 'C': 0.0, 'D': 0.1, 'CASH': 0.5}
        weights = constraint.apply(original_weights)
        self.assertEqual(weights['C'], 0.0)  # Zero weight should remain zero


class TestSectorConstraint(unittest.TestCase):
    """Test suite for SectorConstraint."""

    def test_init(self):
        """Test initialization parameters."""
        # Test with default parameters
        constraint = SectorConstraint()
        self.assertEqual(constraint.max_sector_exposure, 0.3)
        self.assertEqual(constraint.min_sector_exposure, 0.0)
        self.assertEqual(constraint.sector_mapping, {})

        # Test with custom parameters
        sector_mapping = {'A': 'Tech', 'B': 'Tech', 'C': 'Finance', 'D': 'Energy'}
        constraint = SectorConstraint(max_sector_exposure=0.4, min_sector_exposure=0.1,
                                      sector_mapping=sector_mapping)
        self.assertEqual(constraint.max_sector_exposure, 0.4)
        self.assertEqual(constraint.min_sector_exposure, 0.1)
        self.assertEqual(constraint.sector_mapping, sector_mapping)

        # Test with invalid parameters
        with self.assertRaises(ValueError):
            SectorConstraint(max_sector_exposure=0)
        with self.assertRaises(ValueError):
            SectorConstraint(max_sector_exposure=1.1)
        with self.assertRaises(ValueError):
            SectorConstraint(min_sector_exposure=-0.1)
        with self.assertRaises(ValueError):
            SectorConstraint(max_sector_exposure=0.3, min_sector_exposure=0.4)

    def test_is_satisfied(self):
        """Test constraint satisfaction check."""
        sector_mapping = {'A': 'Tech', 'B': 'Tech', 'C': 'Finance', 'D': 'Energy', 'E': 'Energy'}
        constraint = SectorConstraint(max_sector_exposure=0.4, min_sector_exposure=0.1,
                                      sector_mapping=sector_mapping)

        # Test weights with sectors within limits
        weights = {'A': 0.2, 'B': 0.1, 'C': 0.3, 'D': 0.05, 'E': 0.05, 'CASH': 0.3}
        self.assertTrue(constraint.is_satisfied(weights))

        # Test weights with sectors exceeding max exposure
        weights = {'A': 0.3, 'B': 0.2, 'C': 0.1, 'D': 0.1, 'E': 0.1, 'CASH': 0.2}
        self.assertFalse(constraint.is_satisfied(weights))  # Tech exceeds 0.4

        # Test weights with sectors below min exposure
        weights = {'A': 0.2, 'B': 0.1, 'C': 0.3, 'D': 0.05, 'E': 0.0, 'CASH': 0.35}
        self.assertFalse(constraint.is_satisfied(weights))  # Energy below 0.1

        # Test with metadata sector mapping
        constraint = SectorConstraint(max_sector_exposure=0.4, min_sector_exposure=0.1)
        metadata = {'sector_mapping': sector_mapping}
        weights = {'A': 0.2, 'B': 0.1, 'C': 0.3, 'D': 0.05, 'E': 0.05, 'CASH': 0.3}
        self.assertTrue(constraint.is_satisfied(weights, metadata))

    def test_apply(self):
        """Test constraint application."""
        sector_mapping = {'A': 'Tech', 'B': 'Tech', 'C': 'Finance', 'D': 'Energy', 'E': 'Energy'}
        constraint = SectorConstraint(max_sector_exposure=0.4, min_sector_exposure=0.1,
                                      sector_mapping=sector_mapping)

        # Test weights already satisfying constraint
        original_weights = {'A': 0.2, 'B': 0.1, 'C': 0.3, 'D': 0.05, 'E': 0.05, 'CASH': 0.3}
        weights = constraint.apply(original_weights)
        self.assertEqual(weights, original_weights)

        # Test weights with sectors exceeding max exposure
        original_weights = {'A': 0.3, 'B': 0.2, 'C': 0.2, 'D': 0.1, 'E': 0.0, 'CASH': 0.2}
        weights = constraint.apply(original_weights)

        # Check sector exposures
        tech_exposure = weights['A'] + weights['B']
        finance_exposure = weights['C']
        energy_exposure = weights['D'] + weights['E']

        self.assertLessEqual(tech_exposure, constraint.max_sector_exposure)
        self.assertLessEqual(finance_exposure, constraint.max_sector_exposure)
        self.assertLessEqual(energy_exposure, constraint.max_sector_exposure)

        # Check weights sum to 1.0
        self.assertAlmostEqual(sum(weights.values()), 1.0)

        # Test weights with sectors below min exposure
        original_weights = {'A': 0.2, 'B': 0.1, 'C': 0.3, 'D': 0.05, 'E': 0.0, 'CASH': 0.35}
        weights = constraint.apply(original_weights)

        # Check sector exposures
        tech_exposure = weights['A'] + weights['B']
        finance_exposure = weights['C']
        energy_exposure = weights['D'] + weights['E']

        if energy_exposure > 0:  # If sector not eliminated
            self.assertGreaterEqual(energy_exposure, constraint.min_sector_exposure)

        # Check weights sum to 1.0
        self.assertAlmostEqual(sum(weights.values()), 1.0)

        # Test with metadata sector mapping
        constraint = SectorConstraint(max_sector_exposure=0.4, min_sector_exposure=0.1)
        metadata = {'sector_mapping': sector_mapping}
        original_weights = {'A': 0.3, 'B': 0.2, 'C': 0.2, 'D': 0.1, 'E': 0.0, 'CASH': 0.2}
        weights = constraint.apply(original_weights, metadata)

        tech_exposure = weights['A'] + weights['B']
        self.assertLessEqual(tech_exposure, constraint.max_sector_exposure)


class TestTurnoverConstraint(unittest.TestCase):
    """Test suite for TurnoverConstraint."""

    def test_init(self):
        """Test initialization parameters."""
        # Test with default parameters
        constraint = TurnoverConstraint()
        self.assertEqual(constraint.max_turnover, 0.2)
        self.assertEqual(constraint.current_weights, {})

        # Test with custom parameters
        current_weights = {'A': 0.3, 'B': 0.3, 'C': 0.2, 'CASH': 0.2}
        constraint = TurnoverConstraint(max_turnover=0.3, current_weights=current_weights)
        self.assertEqual(constraint.max_turnover, 0.3)
        self.assertEqual(constraint.current_weights, current_weights)

        # Test with invalid parameters
        with self.assertRaises(ValueError):
            TurnoverConstraint(max_turnover=-0.1)
        with self.assertRaises(ValueError):
            TurnoverConstraint(max_turnover=2.1)

    def test_is_satisfied(self):
        """Test constraint satisfaction check."""
        current_weights = {'A': 0.3, 'B': 0.3, 'C': 0.2, 'CASH': 0.2}
        constraint = TurnoverConstraint(max_turnover=0.3, current_weights=current_weights)

        # Test weights with turnover within limit
        weights = {'A': 0.4, 'B': 0.2, 'C': 0.2, 'CASH': 0.2}
        # Turnover: |0.4-0.3| + |0.2-0.3| + |0.2-0.2| + |0.2-0.2| = 0.2
        self.assertTrue(constraint.is_satisfied(weights))

        # Test weights with turnover at the limit
        weights = {'A': 0.45, 'B': 0.15, 'C': 0.2, 'CASH': 0.2}
        # Turnover: |0.45-0.3| + |0.15-0.3| + |0.2-0.2| + |0.2-0.2| = 0.3
        self.assertTrue(constraint.is_satisfied(weights))

        # Test weights with turnover exceeding limit
        weights = {'A': 0.5, 'B': 0.1, 'C': 0.2, 'CASH': 0.2}
        # Turnover: |0.5-0.3| + |0.1-0.3| + |0.2-0.2| + |0.2-0.2| = 0.4
        self.assertFalse(constraint.is_satisfied(weights))

        # Test with new symbols
        weights = {'A': 0.3, 'B': 0.0, 'D': 0.3, 'CASH': 0.4}
        # Turnover: |0.3-0.3| + |0.0-0.3| + |0.0-0.2| + |0.3-0.0| + |0.4-0.2| = 0.4
        self.assertFalse(constraint.is_satisfied(weights))

        # Test with metadata current weights
        constraint = TurnoverConstraint(max_turnover=0.3)
        metadata = {'current_weights': current_weights}
        weights = {'A': 0.4, 'B': 0.2, 'C': 0.2, 'CASH': 0.2}
        self.assertTrue(constraint.is_satisfied(weights, metadata))

    def test_apply(self):
        """Test constraint application."""
        current_weights = {'A': 0.3, 'B': 0.3, 'C': 0.2, 'CASH': 0.2}
        constraint = TurnoverConstraint(max_turnover=0.2, current_weights=current_weights)

        # Test weights already satisfying constraint
        original_weights = {'A': 0.35, 'B': 0.25, 'C': 0.2, 'CASH': 0.2}
        weights = constraint.apply(original_weights)
        self.assertEqual(weights, original_weights)

        # Test weights with turnover exceeding limit
        original_weights = {'A': 0.5, 'B': 0.1, 'C': 0.2, 'CASH': 0.2}
        weights = constraint.apply(original_weights)

        # Calculate turnover
        all_symbols = set(list(weights.keys()) + list(current_weights.keys()))
        turnover = 0.0
        for symbol in all_symbols:
            current = current_weights.get(symbol, 0.0)
            target = weights.get(symbol, 0.0)
            turnover += abs(target - current)
        turnover /= 2.0

        # Check turnover is at or below max
        self.assertLessEqual(turnover, constraint.max_turnover)

        # Check weights sum to 1.0
        self.assertAlmostEqual(sum(weights.values()), 1.0)

        # Test with new symbols
        original_weights = {'A': 0.3, 'B': 0.0, 'D': 0.3, 'CASH': 0.4}
        weights = constraint.apply(original_weights)

        # Calculate turnover
        all_symbols = set(list(weights.keys()) + list(current_weights.keys()))
        turnover = 0.0
        for symbol in all_symbols:
            current = current_weights.get(symbol, 0.0)
            target = weights.get(symbol, 0.0)
            turnover += abs(target - current)
        turnover /= 2.0

        # Check turnover is at or below max
        self.assertLessEqual(turnover, constraint.max_turnover)

        # Check weights sum to 1.0
        self.assertAlmostEqual(sum(weights.values()), 1.0)

        # Test with metadata current weights
        constraint = TurnoverConstraint(max_turnover=0.2)
        metadata = {'current_weights': current_weights}
        original_weights = {'A': 0.5, 'B': 0.1, 'C': 0.2, 'CASH': 0.2}
        weights = constraint.apply(original_weights, metadata)

        # Calculate turnover
        all_symbols = set(list(weights.keys()) + list(current_weights.keys()))
        turnover = 0.0
        for symbol in all_symbols:
            current = current_weights.get(symbol, 0.0)
            target = weights.get(symbol, 0.0)
            turnover += abs(target - current)
        turnover /= 2.0

        # Check turnover is at or below max
        self.assertLessEqual(turnover, constraint.max_turnover)


class TestMultipleConstraints(unittest.TestCase):
    """Test suite for combining multiple constraints."""

    def test_multiple_constraints(self):
        """Test applying multiple constraints in sequence."""
        # Create a set of initial weights
        initial_weights = {'A': 0.4, 'B': 0.3, 'C': -0.2, 'D': 0.3, 'CASH': 0.2}

        # Create constraints
        leverage_constraint = LeverageConstraint(max_leverage=1.0, exclude_cash=True)
        position_constraint = PositionSizeConstraint(max_size=0.25, min_size=0.05, exclude_cash=True)
        current_weights = {'A': 0.25, 'B': 0.25, 'C': -0.1, 'D': 0.2, 'CASH': 0.4}
        turnover_constraint = TurnoverConstraint(max_turnover=0.2, current_weights=current_weights)
        sector_mapping = {'A': 'Tech', 'B': 'Tech', 'C': 'Finance', 'D': 'Energy'}
        sector_constraint = SectorConstraint(max_sector_exposure=0.4, min_sector_exposure=0.1,
                                             sector_mapping=sector_mapping)

        # Apply constraints in sequence
        weights = initial_weights.copy()

        # First constraint
        weights = leverage_constraint.apply(weights)
        self.assertTrue(leverage_constraint.is_satisfied(weights))

        # Second constraint
        weights = position_constraint.apply(weights)
        self.assertTrue(leverage_constraint.is_satisfied(weights))
        self.assertTrue(position_constraint.is_satisfied(weights))

        # Third constraint
        weights = sector_constraint.apply(weights)
        self.assertTrue(leverage_constraint.is_satisfied(weights))
        self.assertTrue(position_constraint.is_satisfied(weights))
        self.assertTrue(sector_constraint.is_satisfied(weights))

        # Fourth constraint
        weights = turnover_constraint.apply(weights)
        self.assertTrue(leverage_constraint.is_satisfied(weights))
        # Note: Other constraints might no longer be satisfied after applying turnover constraint

        # Check final weights sum to 1.0
        self.assertAlmostEqual(sum(weights.values()), 1.0)


if __name__ == '__main__':
    unittest.main()