"""
Unit tests for portfolio objectives module.

This test suite verifies that portfolio objective functions work as expected,
including return maximization, risk minimization, Sharpe ratio maximization,
and drawdown minimization.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Import the objectives module
from models.portfolio.objectives import (
    PortfolioObjective,
    MaximizeReturn,
    MinimizeRisk,
    MaximizeSharpe,
    MinimizeMaximumDrawdown
)


class TestPortfolioObjectives(unittest.TestCase):
    """Test cases for portfolio objectives."""

    def setUp(self):
        """Set up test fixtures for all test methods."""
        # Sample weights for a portfolio
        self.weights = {
            'AAPL': 0.25,
            'MSFT': 0.25,
            'AMZN': 0.25,
            'GOOG': 0.25,
        }

        # Sample expected returns
        self.expected_returns = {
            'AAPL': 0.10,
            'MSFT': 0.08,
            'AMZN': 0.12,
            'GOOG': 0.09,
        }

        # Sample covariance matrix
        data = {
            'AAPL': [0.04, 0.02, 0.015, 0.018],
            'MSFT': [0.02, 0.03, 0.01, 0.015],
            'AMZN': [0.015, 0.01, 0.05, 0.02],
            'GOOG': [0.018, 0.015, 0.02, 0.035],
        }
        self.cov_matrix = pd.DataFrame(
            data,
            index=['AAPL', 'MSFT', 'AMZN', 'GOOG'],
            columns=['AAPL', 'MSFT', 'AMZN', 'GOOG']
        )

        # Sample historical returns
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = np.random.normal(0.001, 0.02, size=(100, 4))
        self.historical_returns = pd.DataFrame(
            data,
            index=dates,
            columns=['AAPL', 'MSFT', 'AMZN', 'GOOG']
        )

        # Common metadata dictionary
        self.metadata = {
            'expected_returns': self.expected_returns,
            'covariance_matrix': self.cov_matrix,
            'historical_returns': self.historical_returns,
            'risk_free_rate': 0.02
        }

    def test_abstract_base_class(self):
        """Test that PortfolioObjective is an abstract base class."""
        with self.assertRaises(TypeError):
            obj = PortfolioObjective()  # Should fail as an abstract class

    def test_portfolio_objective_str(self):
        """Test string representation of a portfolio objective."""

        # Create a concrete subclass for testing
        class TestObjective(PortfolioObjective):
            def evaluate(self, weights, metadata):
                return 1.0

        obj = TestObjective()
        self.assertEqual(str(obj), "TestObjective")

    def test_maximize_return(self):
        """Test that MaximizeReturn correctly calculates expected portfolio return."""
        # Create objective with constructor-provided expected returns
        obj1 = MaximizeReturn(self.expected_returns)
        result1 = obj1.evaluate(self.weights, {})

        # Expected result: 0.25*0.10 + 0.25*0.08 + 0.25*0.12 + 0.25*0.09 = 0.0975
        expected_result = 0.0975
        self.assertAlmostEqual(result1, expected_result)

        # Test with expected returns provided in metadata
        obj2 = MaximizeReturn()
        result2 = obj2.evaluate(self.weights, {'expected_returns': self.expected_returns})
        self.assertAlmostEqual(result2, expected_result)

    def test_maximize_return_no_data(self):
        """Test that MaximizeReturn raises an error when no expected returns are provided."""
        obj = MaximizeReturn()
        with self.assertRaises(ValueError):
            obj.evaluate(self.weights, {})

    def test_maximize_return_missing_asset(self):
        """Test handling of assets not in expected returns."""
        weights = self.weights.copy()
        weights['MISSING'] = 0.1  # Add an asset that's not in expected returns
        weights['AAPL'] -= 0.1  # Adjust to maintain sum of 1

        obj = MaximizeReturn(self.expected_returns)

        # Capture printed warnings
        with patch('builtins.print') as mock_print:
            result = obj.evaluate(weights, {})

        # Check that a warning was printed
        mock_print.assert_called_with("Warning: No expected return for MISSING with weight 0.1")

        # Expected result should ignore the missing asset
        expected_result = 0.225 * 0.10 + 0.25 * 0.08 + 0.25 * 0.12 + 0.25 * 0.09
        self.assertAlmostEqual(result, expected_result)

    def test_maximize_return_with_cash(self):
        """Test MaximizeReturn with cash position."""
        weights = {
            'AAPL': 0.20,
            'MSFT': 0.20,
            'AMZN': 0.20,
            'GOOG': 0.20,
            'CASH': 0.20
        }

        obj = MaximizeReturn(self.expected_returns)
        result = obj.evaluate(weights, {})

        # Expected result: 0.20*0.10 + 0.20*0.08 + 0.20*0.12 + 0.20*0.09 + 0.20*0 = 0.078
        expected_result = 0.078
        self.assertAlmostEqual(result, expected_result)

    def test_maximize_return_str(self):
        """Test string representation of MaximizeReturn."""
        obj = MaximizeReturn()
        self.assertEqual(str(obj), "MaximizeReturn")

    def test_minimize_risk(self):
        """Test that MinimizeRisk correctly calculates portfolio variance."""
        # Create objective with constructor-provided covariance matrix
        obj1 = MinimizeRisk(self.cov_matrix)
        result1 = obj1.evaluate(self.weights, {})

        # Calculate expected variance manually
        weight_vector = np.array([0.25, 0.25, 0.25, 0.25])
        expected_variance = weight_vector.T @ self.cov_matrix.values @ weight_vector

        # Result should be negative variance
        self.assertAlmostEqual(result1, -expected_variance)

        # Test with covariance matrix provided in metadata
        obj2 = MinimizeRisk()
        result2 = obj2.evaluate(self.weights, {'covariance_matrix': self.cov_matrix})
        self.assertAlmostEqual(result2, -expected_variance)

    def test_minimize_risk_no_data(self):
        """Test that MinimizeRisk raises an error when no covariance matrix is provided."""
        obj = MinimizeRisk()
        with self.assertRaises(ValueError):
            obj.evaluate(self.weights, {})

    def test_minimize_risk_cash_position(self):
        """Test that MinimizeRisk handles cash positions correctly."""
        weights = {
            'AAPL': 0.20,
            'MSFT': 0.20,
            'AMZN': 0.20,
            'GOOG': 0.20,
            'CASH': 0.20  # Cash should be ignored in risk calculation
        }

        obj = MinimizeRisk(self.cov_matrix)
        result = obj.evaluate(weights, {})

        # Calculate expected variance manually (ignoring cash)
        weight_vector = np.array([0.20, 0.20, 0.20, 0.20])
        sub_cov = self.cov_matrix.values
        expected_variance = weight_vector.T @ sub_cov @ weight_vector

        # Result should be negative variance
        self.assertAlmostEqual(result, -expected_variance)

    def test_minimize_risk_str(self):
        """Test string representation of MinimizeRisk."""
        obj = MinimizeRisk()
        self.assertEqual(str(obj), "MinimizeRisk")

    def test_maximize_sharpe(self):
        """Test that MaximizeSharpe correctly calculates Sharpe ratio."""
        risk_free_rate = 0.02
        obj = MaximizeSharpe(
            self.expected_returns,
            self.cov_matrix,
            risk_free_rate
        )
        result = obj.evaluate(self.weights, {})

        # Calculate expected return
        expected_return = sum(self.weights[s] * self.expected_returns[s] for s in self.weights)

        # Calculate expected variance
        weight_vector = np.array([0.25, 0.25, 0.25, 0.25])
        expected_variance = weight_vector.T @ self.cov_matrix.values @ weight_vector
        expected_volatility = np.sqrt(expected_variance)

        # Calculate expected Sharpe ratio
        expected_sharpe = (expected_return - risk_free_rate) / expected_volatility

        self.assertAlmostEqual(result, expected_sharpe)

    def test_maximize_sharpe_metadata(self):
        """Test MaximizeSharpe using data from metadata."""
        obj = MaximizeSharpe()
        result = obj.evaluate(self.weights, self.metadata)

        # Calculate expected return
        expected_return = sum(self.weights[s] * self.expected_returns[s] for s in self.weights)

        # Calculate expected variance
        weight_vector = np.array([0.25, 0.25, 0.25, 0.25])
        expected_variance = weight_vector.T @ self.cov_matrix.values @ weight_vector
        expected_volatility = np.sqrt(expected_variance)

        # Calculate expected Sharpe ratio
        expected_sharpe = (expected_return - self.metadata['risk_free_rate']) / expected_volatility

        self.assertAlmostEqual(result, expected_sharpe)

    def test_maximize_sharpe_no_data(self):
        """Test that MaximizeSharpe raises an error when necessary data is missing."""
        obj = MaximizeSharpe()
        with self.assertRaises(ValueError):
            obj.evaluate(self.weights, {})

    def test_maximize_sharpe_zero_variance(self):
        """Test MaximizeSharpe with zero variance portfolio."""
        # Create a special covariance matrix with zeros
        zero_cov = pd.DataFrame(
            np.zeros((4, 4)),
            index=['AAPL', 'MSFT', 'AMZN', 'GOOG'],
            columns=['AAPL', 'MSFT', 'AMZN', 'GOOG']
        )

        obj = MaximizeSharpe(self.expected_returns, zero_cov, 0.02)
        result = obj.evaluate(self.weights, {})

        # Should return negative infinity for zero variance
        self.assertEqual(result, float('-inf'))

    def test_maximize_sharpe_str(self):
        """Test string representation of MaximizeSharpe."""
        obj = MaximizeSharpe(risk_free_rate=0.03)
        self.assertEqual(str(obj), "MaximizeSharpe(risk_free_rate=0.03)")

    def test_minimize_maximum_drawdown(self):
        """Test that MinimizeMaximumDrawdown correctly calculates maximum drawdown."""
        # Create a deterministic returns dataframe for testing
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        returns_data = {
            'AAPL': [0.01, -0.02, 0.03, -0.01, 0.02],
            'MSFT': [0.02, -0.01, 0.01, -0.02, 0.01],
        }
        historical_returns = pd.DataFrame(returns_data, index=dates)

        weights = {'AAPL': 0.6, 'MSFT': 0.4}

        # Calculate expected portfolio returns
        portfolio_returns = np.array([
            0.6 * 0.01 + 0.4 * 0.02,  # 0.014
            0.6 * -0.02 + 0.4 * -0.01,  # -0.016
            0.6 * 0.03 + 0.4 * 0.01,  # 0.022
            0.6 * -0.01 + 0.4 * -0.02,  # -0.014
            0.6 * 0.02 + 0.4 * 0.01,  # 0.016
        ])

        # Calculate cumulative returns: [1.014, 0.998, 1.019, 1.005, 1.021]
        cumulative_returns = (1 + portfolio_returns).cumprod()

        # Calculate running maximum: [1.014, 1.014, 1.019, 1.019, 1.021]
        running_max = np.maximum.accumulate(cumulative_returns)

        # Calculate drawdowns: [0, -0.016, 0, -0.014, 0]
        drawdowns = (cumulative_returns / running_max) - 1.0

        # Expected maximum drawdown
        expected_max_drawdown = np.min(drawdowns)

        obj = MinimizeMaximumDrawdown(historical_returns)
        result = obj.evaluate(weights, {})

        # Result should be negative of max drawdown
        self.assertAlmostEqual(result, -expected_max_drawdown)

    def test_minimize_maximum_drawdown_metadata(self):
        """Test MinimizeMaximumDrawdown with historical returns from metadata."""
        obj = MinimizeMaximumDrawdown()
        result = obj.evaluate(self.weights, {'historical_returns': self.historical_returns})

        # We can't easily calculate the expected result manually for random data,
        # but we can check that the result is negative (or zero for perfect portfolios)
        self.assertLessEqual(result, 0)

    def test_minimize_maximum_drawdown_no_data(self):
        """Test that MinimizeMaximumDrawdown raises an error when no historical returns are provided."""
        obj = MinimizeMaximumDrawdown()
        with self.assertRaises(ValueError):
            obj.evaluate(self.weights, {})

    def test_minimize_maximum_drawdown_empty_portfolio(self):
        """Test MinimizeMaximumDrawdown with an empty portfolio."""
        obj = MinimizeMaximumDrawdown(self.historical_returns)
        result = obj.evaluate({}, {})

        # An empty portfolio should have zero drawdown
        self.assertEqual(result, 0.0)

    def test_minimize_maximum_drawdown_str(self):
        """Test string representation of MinimizeMaximumDrawdown."""
        obj = MinimizeMaximumDrawdown()
        self.assertEqual(str(obj), "MinimizeMaximumDrawdown")


if __name__ == '__main__':
    unittest.main()