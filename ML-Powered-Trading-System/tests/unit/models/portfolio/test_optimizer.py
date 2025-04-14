"""
Unit tests for the portfolio optimizer module.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock

from models.portfolio.optimizer import PortfolioOptimizer, OptimizationResult
from models.portfolio.constraints import LeverageConstraint, PositionSizeConstraint, PortfolioConstraint
from models.portfolio.objectives import MaximizeReturn, MinimizeRisk, MaximizeSharpe


class TestPortfolioOptimizer(unittest.TestCase):
    """Test cases for the PortfolioOptimizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.alpha_signals = {
            'AAPL': 0.05,
            'MSFT': 0.03,
            'GOOG': 0.04,
            'AMZN': 0.02
        }

        # Create a simple covariance matrix
        symbols = list(self.alpha_signals.keys())
        cov_matrix = np.array([
            [0.04, 0.02, 0.02, 0.01],
            [0.02, 0.05, 0.01, 0.01],
            [0.02, 0.01, 0.03, 0.01],
            [0.01, 0.01, 0.01, 0.06]
        ])
        self.covariance_matrix = pd.DataFrame(
            cov_matrix,
            index=symbols,
            columns=symbols
        )

        # Initial portfolio weights
        self.initial_weights = {
            'AAPL': 0.3,
            'MSFT': 0.2,
            'GOOG': 0.25,
            'AMZN': 0.15,
            'CASH': 0.1
        }

        # Default optimizer for testing
        self.optimizer = PortfolioOptimizer()

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        optimizer = PortfolioOptimizer()

        # Check that default constraints are set
        self.assertEqual(len(optimizer.constraints), 2)
        self.assertIsInstance(optimizer.constraints[0], LeverageConstraint)
        self.assertIsInstance(optimizer.constraints[1], PositionSizeConstraint)

        # Check default objective
        self.assertIsInstance(optimizer.objective, MaximizeReturn)

        # Check default bounds
        self.assertEqual(optimizer.default_bounds, (-1.0, 1.0))

        # Check cash key
        self.assertEqual(optimizer.cash_key, 'CASH')

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        objective = MinimizeRisk()
        constraints = [
            LeverageConstraint(max_leverage=0.8),
            PositionSizeConstraint(max_size=0.2, min_size=0.05)
        ]
        default_bounds = (-0.5, 0.5)
        cash_key = 'USD'

        optimizer = PortfolioOptimizer(
            objective=objective,
            constraints=constraints,
            default_bounds=default_bounds,
            cash_key=cash_key
        )

        self.assertEqual(optimizer.objective, objective)
        self.assertEqual(optimizer.constraints, constraints)
        self.assertEqual(optimizer.default_bounds, default_bounds)
        self.assertEqual(optimizer.cash_key, cash_key)

    def test_optimize_successful(self):
        """Test successful optimization."""
        result = self.optimizer.optimize(
            alpha_signals=self.alpha_signals,
            initial_weights=self.initial_weights,
            covariance_matrix=self.covariance_matrix
        )

        # Check result type
        self.assertIsInstance(result, OptimizationResult)

        # Check optimization status
        self.assertTrue(result.success)

        # Check weights sum to 1
        self.assertAlmostEqual(sum(result.weights.values()), 1.0, delta=1e-6)

        # Check all symbols are present in the weights
        for symbol in list(self.alpha_signals.keys()) + ['CASH']:
            self.assertIn(symbol, result.weights)

    def test_optimize_with_missing_initial_weights(self):
        """Test optimization with missing initial weights."""
        # Remove some initial weights
        partial_weights = {'AAPL': 0.5, 'MSFT': 0.5}

        result = self.optimizer.optimize(
            alpha_signals=self.alpha_signals,
            initial_weights=partial_weights,
            covariance_matrix=self.covariance_matrix
        )

        # Check all symbols are present in the weights
        for symbol in list(self.alpha_signals.keys()) + ['CASH']:
            self.assertIn(symbol, result.weights)

        # Check weights sum to 1
        self.assertAlmostEqual(sum(result.weights.values()), 1.0, delta=1e-6)

    def test_optimize_with_custom_bounds(self):
        """Test optimization with custom bounds."""
        bounds = {
            'AAPL': (0.0, 0.4),
            'MSFT': (0.0, 0.3),
            'CASH': (0.2, 1.0)
        }

        result = self.optimizer.optimize(
            alpha_signals=self.alpha_signals,
            initial_weights=self.initial_weights,
            covariance_matrix=self.covariance_matrix,
            bounds=bounds
        )

        # Check bounds are respected
        self.assertLessEqual(result.weights['AAPL'], 0.4)
        self.assertLessEqual(result.weights['MSFT'], 0.3)
        self.assertGreaterEqual(result.weights['CASH'], 0.2)

    @patch('scipy.optimize.minimize')
    def test_optimize_failure(self, mock_minimize):
        """Test handling of optimization failures."""
        # Mock the minimize function to simulate failure
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.message = "Optimization failed"
        mock_result.x = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Equal weights
        mock_result.nit = 0
        mock_result.nfev = 0
        mock_minimize.return_value = mock_result

        result = self.optimizer.optimize(
            alpha_signals=self.alpha_signals,
            initial_weights=self.initial_weights,
            covariance_matrix=self.covariance_matrix
        )

        # Check result status
        self.assertFalse(result.success)
        self.assertEqual(result.message, "Optimization failed")

        # Check that weights still sum to 1
        self.assertAlmostEqual(sum(result.weights.values()), 1.0, delta=1e-6)

    def test_optimize_with_exception(self):
        """Test handling of exceptions during optimization."""
        with patch('scipy.optimize.minimize', side_effect=Exception("Test exception")):
            result = self.optimizer.optimize(
                alpha_signals=self.alpha_signals,
                initial_weights=self.initial_weights,
                covariance_matrix=self.covariance_matrix
            )

            # Check result status
            self.assertFalse(result.success)
            self.assertEqual(result.message, "Test exception")

            # Check that initial weights are returned
            self.assertEqual(result.weights, self.initial_weights)

    def test_alpha_to_weights(self):
        """Test the alpha_to_weights convenience method."""
        weights = self.optimizer.alpha_to_weights(
            alpha_results=self.alpha_signals,
            current_portfolio=self.initial_weights,
            risk_model=self.covariance_matrix
        )

        # Check that weights are returned
        self.assertIsInstance(weights, dict)

        # Check all symbols are present
        for symbol in list(self.alpha_signals.keys()) + ['CASH']:
            self.assertIn(symbol, weights)

        # Check weights sum to 1
        self.assertAlmostEqual(sum(weights.values()), 1.0, delta=1e-6)

    def test_alpha_to_weights_with_failure(self):
        """Test alpha_to_weights with optimization failure."""
        with patch.object(PortfolioOptimizer, 'optimize') as mock_optimize:
            # Mock optimization failure
            mock_optimize.return_value = OptimizationResult(
                weights={},
                objective_value=0.0,
                success=False,
                message="Test failure",
                optimization_time=0.0
            )

            # Should return current portfolio on failure
            weights = self.optimizer.alpha_to_weights(
                alpha_results=self.alpha_signals,
                current_portfolio=self.initial_weights,
                risk_model=self.covariance_matrix
            )

            self.assertEqual(weights, self.initial_weights)

    def test_alpha_to_weights_without_current_portfolio(self):
        """Test alpha_to_weights with no current portfolio and optimization failure."""
        with patch.object(PortfolioOptimizer, 'optimize') as mock_optimize:
            # Mock optimization failure
            mock_optimize.return_value = OptimizationResult(
                weights={},
                objective_value=0.0,
                success=False,
                message="Test failure",
                optimization_time=0.0
            )

            # Should return equal weights on failure with no current portfolio
            weights = self.optimizer.alpha_to_weights(
                alpha_results=self.alpha_signals,
                current_portfolio=None,
                risk_model=self.covariance_matrix
            )

            # Check all positive alpha symbols have equal weight
            expected_weight = 1.0 / (len(self.alpha_signals) + 1)  # +1 for cash
            for symbol in self.alpha_signals:
                self.assertAlmostEqual(weights[symbol], expected_weight)
            self.assertAlmostEqual(weights['CASH'], expected_weight)

    def test_optimize_with_different_objectives(self):
        """Test optimization with different objectives."""
        results = self.optimizer.optimize_with_different_objectives(
            alpha_signals=self.alpha_signals,
            risk_model=self.covariance_matrix,
            initial_weights=self.initial_weights
        )

        # Check we have results for all objectives
        self.assertIn('max_return', results)
        self.assertIn('min_risk', results)
        self.assertIn('max_sharpe', results)

        # Check all results are OptimizationResult objects
        for result in results.values():
            self.assertIsInstance(result, OptimizationResult)
            self.assertTrue(result.success)

        # Results should be different for different objectives
        self.assertNotEqual(
            results['max_return'].weights,
            results['min_risk'].weights
        )


class TestOptimizationResult(unittest.TestCase):
    """Test cases for the OptimizationResult class."""

    def test_initialization(self):
        """Test initialization of OptimizationResult."""
        weights = {'AAPL': 0.5, 'MSFT': 0.5}
        result = OptimizationResult(
            weights=weights,
            objective_value=1.0,
            success=True,
            message="Optimization successful"
        )

        self.assertEqual(result.weights, weights)
        self.assertEqual(result.objective_value, 1.0)
        self.assertTrue(result.success)
        self.assertEqual(result.message, "Optimization successful")
        self.assertIsInstance(result.timestamp, datetime)
        self.assertEqual(result.metadata, {})

    def test_str_method(self):
        """Test the __str__ method of OptimizationResult."""
        # Successful result
        success_result = OptimizationResult(
            weights={'AAPL': 0.5, 'MSFT': 0.5},
            objective_value=1.0,
            success=True,
            message="Optimization successful"
        )
        self.assertIn("succeeded", str(success_result))
        self.assertIn("1.000000", str(success_result))

        # Failed result
        failed_result = OptimizationResult(
            weights={'AAPL': 0.5, 'MSFT': 0.5},
            objective_value=0.0,
            success=False,
            message="Optimization failed"
        )
        self.assertIn("failed", str(failed_result))
        self.assertIn("0.000000", str(failed_result))


if __name__ == '__main__':
    unittest.main()