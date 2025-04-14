"""
Unit tests for the risk model module.

This test suite validates the factor-based risk decomposition and forecasting
capabilities of the RiskModel class.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock
import pickle
import os
import tempfile

from models.portfolio.risk_model import RiskModel, RiskDecomposition


class TestRiskModel(unittest.TestCase):
    """Test suite for the RiskModel class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample data for testing
        np.random.seed(42)  # For reproducibility

        # Create sample dates
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')

        # Create sample asset returns
        self.assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB']
        self.asset_returns = pd.DataFrame(
            np.random.normal(0.001, 0.02, size=(100, len(self.assets))),
            index=dates,
            columns=self.assets
        )

        # Create sample factor returns
        self.factors = ['Market', 'Size', 'Value', 'Momentum', 'Volatility']
        self.factor_returns = pd.DataFrame(
            np.random.normal(0.0005, 0.01, size=(100, len(self.factors))),
            index=dates,
            columns=self.factors
        )

        # Create correlations between factors and returns for realism
        # Market factor affects all stocks
        self.asset_returns['AAPL'] += 1.2 * self.factor_returns['Market']
        self.asset_returns['MSFT'] += 1.0 * self.factor_returns['Market']
        self.asset_returns['GOOGL'] += 0.9 * self.factor_returns['Market']
        self.asset_returns['AMZN'] += 1.3 * self.factor_returns['Market']
        self.asset_returns['FB'] += 1.1 * self.factor_returns['Market']

        # Other factor influences
        self.asset_returns['AAPL'] += 0.5 * self.factor_returns['Momentum'] - 0.3 * self.factor_returns['Value']
        self.asset_returns['MSFT'] += 0.2 * self.factor_returns['Size'] + 0.4 * self.factor_returns['Momentum']
        self.asset_returns['GOOGL'] += 0.3 * self.factor_returns['Value'] + 0.2 * self.factor_returns['Volatility']
        self.asset_returns['AMZN'] -= 0.6 * self.factor_returns['Value'] + 0.4 * self.factor_returns['Size']
        self.asset_returns['FB'] += 0.5 * self.factor_returns['Volatility'] + 0.3 * self.factor_returns['Momentum']

        # Create a risk model instance
        self.risk_model = RiskModel()

        # Sample portfolio weights
        self.portfolio_weights = {
            'AAPL': 0.25,
            'MSFT': 0.20,
            'GOOGL': 0.20,
            'AMZN': 0.15,
            'FB': 0.15,
            'CASH': 0.05
        }

    def test_init(self):
        """Test initialization of the RiskModel class."""
        model = RiskModel()
        self.assertIsNone(model.factor_covariance)
        self.assertIsNone(model.factor_exposures)
        self.assertEqual(model.specific_variances, {})
        self.assertEqual(model.estimation_universe, [])
        self.assertEqual(model.factor_names, [])

        # Test initialization with parameters
        factor_cov = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]], columns=['A', 'B'], index=['A', 'B'])
        factor_exp = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]], columns=['A', 'B'], index=['Asset1', 'Asset2'])
        spec_var = {'Asset1': 0.01, 'Asset2': 0.02}

        model = RiskModel(
            factor_covariance=factor_cov,
            factor_exposures=factor_exp,
            specific_variances=spec_var,
            estimation_universe=['Asset1', 'Asset2'],
            factor_names=['A', 'B']
        )

        pd.testing.assert_frame_equal(model.factor_covariance, factor_cov)
        pd.testing.assert_frame_equal(model.factor_exposures, factor_exp)
        self.assertEqual(model.specific_variances, spec_var)
        self.assertEqual(model.estimation_universe, ['Asset1', 'Asset2'])
        self.assertEqual(model.factor_names, ['A', 'B'])

    def test_estimate_from_returns(self):
        """Test estimation of risk model parameters from historical returns."""
        self.risk_model.estimate_from_returns(
            returns=self.asset_returns,
            factors=self.factor_returns,
            min_history=60,
            shrinkage=0.1
        )

        # Check that model parameters were estimated
        self.assertIsNotNone(self.risk_model.factor_covariance)
        self.assertIsNotNone(self.risk_model.factor_exposures)
        self.assertTrue(len(self.risk_model.specific_variances) > 0)

        # Check dimensions
        self.assertEqual(self.risk_model.factor_covariance.shape, (len(self.factors), len(self.factors)))
        self.assertEqual(self.risk_model.factor_exposures.shape, (len(self.assets), len(self.factors)))
        self.assertEqual(len(self.risk_model.specific_variances), len(self.assets))

        # Check factor names and estimation universe
        self.assertEqual(self.risk_model.factor_names, self.factors)
        self.assertEqual(set(self.risk_model.estimation_universe), set(self.assets))

        # Check factor exposures to make sure they reflect our constructed relationships
        # Market factor should have strong positive exposure for all stocks
        for asset in self.assets:
            self.assertGreater(self.risk_model.factor_exposures.loc[asset, 'Market'], 0.5)

        # Check specific factor exposures we built into the data
        self.assertGreater(self.risk_model.factor_exposures.loc['AAPL', 'Momentum'], 0)
        self.assertLess(self.risk_model.factor_exposures.loc['AAPL', 'Value'], 0)
        self.assertGreater(self.risk_model.factor_exposures.loc['MSFT', 'Size'], 0)
        self.assertGreater(self.risk_model.factor_exposures.loc['GOOGL', 'Value'], 0)
        self.assertLess(self.risk_model.factor_exposures.loc['AMZN', 'Value'], 0)
        self.assertGreater(self.risk_model.factor_exposures.loc['FB', 'Volatility'], 0)

    def test_estimate_from_returns_insufficient_history(self):
        """Test estimation with insufficient history."""
        with patch('logging.Logger.warning') as mock_warning:
            self.risk_model.estimate_from_returns(
                returns=self.asset_returns.iloc[:10],  # Only 10 periods
                factors=self.factor_returns.iloc[:10],
                min_history=60,
                shrinkage=0.1
            )
            mock_warning.assert_called()

    def test_estimate_from_returns_misaligned_indexes(self):
        """Test estimation with misaligned indexes."""
        with patch('logging.Logger.warning') as mock_warning:
            # Create misaligned data
            misaligned_factors = self.factor_returns.copy()
            misaligned_factors.index = pd.date_range(start='2020-02-01', periods=100, freq='D')

            self.risk_model.estimate_from_returns(
                returns=self.asset_returns,
                factors=misaligned_factors,
                min_history=60,
                shrinkage=0.1
            )
            mock_warning.assert_called()

    def test_get_asset_covariance_matrix(self):
        """Test construction of asset covariance matrix."""
        # First estimate the model
        self.risk_model.estimate_from_returns(
            returns=self.asset_returns,
            factors=self.factor_returns
        )

        # Get covariance matrix for all assets
        cov_matrix = self.risk_model.get_asset_covariance_matrix()

        # Check dimensions
        self.assertEqual(cov_matrix.shape, (len(self.assets), len(self.assets)))

        # Check properties of a valid covariance matrix
        # 1. Symmetric
        pd.testing.assert_frame_equal(cov_matrix, cov_matrix.T, check_dtype=False)

        # 2. Positive semi-definite (all eigenvalues >= 0)
        eigenvalues = np.linalg.eigvals(cov_matrix)
        for ev in eigenvalues:
            self.assertGreaterEqual(ev, -1e-10)  # Allow for numerical imprecision

        # 3. Diagonal entries are variances (non-negative)
        for asset in self.assets:
            self.assertGreaterEqual(cov_matrix.loc[asset, asset], 0)

        # Test with subset of assets
        subset = ['AAPL', 'MSFT']
        sub_cov = self.risk_model.get_asset_covariance_matrix(assets=subset)
        self.assertEqual(sub_cov.shape, (len(subset), len(subset)))
        self.assertEqual(list(sub_cov.index), subset)
        self.assertEqual(list(sub_cov.columns), subset)

    def test_get_asset_covariance_matrix_not_estimated(self):
        """Test getting covariance matrix before model estimation."""
        with self.assertRaises(ValueError):
            self.risk_model.get_asset_covariance_matrix()

    def test_get_asset_covariance_matrix_invalid_assets(self):
        """Test getting covariance matrix with invalid assets."""
        # First estimate the model
        self.risk_model.estimate_from_returns(
            returns=self.asset_returns,
            factors=self.factor_returns
        )

        # Try with invalid assets
        with self.assertRaises(ValueError):
            self.risk_model.get_asset_covariance_matrix(assets=['INVALID1', 'INVALID2'])

    def test_decompose_portfolio_risk(self):
        """Test portfolio risk decomposition."""
        # First estimate the model
        self.risk_model.estimate_from_returns(
            returns=self.asset_returns,
            factors=self.factor_returns
        )

        # Decompose risk for the test portfolio
        decomposition = self.risk_model.decompose_portfolio_risk(self.portfolio_weights)

        # Check that the decomposition is a RiskDecomposition object
        self.assertIsInstance(decomposition, RiskDecomposition)

        # Check that the risk values are positive
        self.assertGreater(decomposition.total_risk, 0)
        self.assertGreater(decomposition.systematic_risk, 0)
        self.assertGreater(decomposition.specific_risk, 0)

        # Check that the risk components sum approximately to total risk
        # (squared, since we're dealing with variances)
        self.assertAlmostEqual(
            decomposition.total_risk**2,
            decomposition.systematic_risk**2 + decomposition.specific_risk**2,
            places=10
        )

        # Check factor and asset contributions
        self.assertEqual(set(decomposition.factor_contributions.keys()), set(self.factors))
        self.assertEqual(set(decomposition.asset_contributions.keys()), set(self.assets))

        # Sum of factor contributions should equal systematic risk
        factor_contrib_sum = sum(decomposition.factor_contributions.values())
        self.assertAlmostEqual(factor_contrib_sum, decomposition.systematic_risk, places=5)

        # Sum of asset contributions should equal total risk
        asset_contrib_sum = sum(decomposition.asset_contributions.values())
        self.assertAlmostEqual(asset_contrib_sum, decomposition.total_risk, places=5)

        # Check factor exposures dict
        for asset in self.assets:
            self.assertIn(asset, decomposition.factor_exposures)
            self.assertEqual(set(decomposition.factor_exposures[asset].keys()), set(self.factors))

    def test_decompose_portfolio_risk_empty_portfolio(self):
        """Test risk decomposition for empty portfolio."""
        # First estimate the model
        self.risk_model.estimate_from_returns(
            returns=self.asset_returns,
            factors=self.factor_returns
        )

        # Test with empty portfolio
        empty_weights = {'CASH': 1.0}
        decomposition = self.risk_model.decompose_portfolio_risk(empty_weights)

        # All risk metrics should be zero
        self.assertEqual(decomposition.total_risk, 0.0)
        self.assertEqual(decomposition.systematic_risk, 0.0)
        self.assertEqual(decomposition.specific_risk, 0.0)
        self.assertEqual(decomposition.factor_contributions, {})
        self.assertEqual(decomposition.asset_contributions, {})
        self.assertEqual(decomposition.factor_exposures, {})

    def test_decompose_portfolio_risk_not_estimated(self):
        """Test risk decomposition before model estimation."""
        with self.assertRaises(ValueError):
            self.risk_model.decompose_portfolio_risk(self.portfolio_weights)

    def test_risk_decomposition_properties(self):
        """Test properties of the RiskDecomposition class."""
        # Create a risk decomposition object
        decomp = RiskDecomposition(
            total_risk=0.1,
            systematic_risk=0.08,
            specific_risk=0.06,
            factor_contributions={'Market': 0.05, 'Size': 0.03},
            asset_contributions={'AAPL': 0.04, 'MSFT': 0.06},
            factor_exposures={'AAPL': {'Market': 1.2}, 'MSFT': {'Market': 1.0}}
        )

        # Check systematic_pct
        expected_systematic_pct = (0.08 / 0.1) * 100.0
        self.assertAlmostEqual(decomp.systematic_pct, expected_systematic_pct)

        # Check specific_pct
        expected_specific_pct = (0.06 / 0.1) * 100.0
        self.assertAlmostEqual(decomp.specific_pct, expected_specific_pct)

        # Test with zero total risk
        decomp = RiskDecomposition(
            total_risk=0.0,
            systematic_risk=0.0,
            specific_risk=0.0,
            factor_contributions={},
            asset_contributions={},
            factor_exposures={}
        )
        self.assertEqual(decomp.systematic_pct, 0.0)
        self.assertEqual(decomp.specific_pct, 0.0)

    def test_get_risk_report(self):
        """Test generation of risk report."""
        # First estimate the model
        self.risk_model.estimate_from_returns(
            returns=self.asset_returns,
            factors=self.factor_returns
        )

        # Get risk report
        report = self.risk_model.get_risk_report(self.portfolio_weights)

        # Check the report content
        self.assertIn('total_risk', report)
        self.assertIn('systematic_risk', report)
        self.assertIn('specific_risk', report)
        self.assertIn('systematic_pct', report)
        self.assertIn('specific_pct', report)
        self.assertIn('value_at_risk_95', report)
        self.assertIn('factor_exposures', report)
        self.assertIn('top_asset_contributors', report)
        self.assertIn('top_factor_contributors', report)
        self.assertIn('timestamp', report)

        # Check VaR calculation (95% confidence = 1.645 standard deviations)
        expected_var = 1.645 * report['total_risk']
        self.assertAlmostEqual(report['value_at_risk_95'], expected_var, places=10)

        # Check factor exposures
        self.assertEqual(set(report['factor_exposures'].keys()), set(self.factors))

        # Check top contributors - should have at most 5 items
        self.assertLessEqual(len(report['top_asset_contributors']), 5)
        self.assertLessEqual(len(report['top_factor_contributors']), 5)

    def test_get_risk_report_error(self):
        """Test risk report error handling."""
        # Create a risk model that will cause an error when decompose_portfolio_risk is called
        model = RiskModel()
        model.factor_covariance = pd.DataFrame()  # Empty dataframe will cause error
        model.factor_exposures = pd.DataFrame()   # Empty dataframe will cause error

        # The report should contain an error message but not crash
        report = model.get_risk_report(self.portfolio_weights)
        self.assertIn('error', report)
        self.assertIn('timestamp', report)

    def test_stress_test_portfolio(self):
        """Test portfolio stress testing."""
        # First estimate the model
        self.risk_model.estimate_from_returns(
            returns=self.asset_returns,
            factors=self.factor_returns
        )

        # Define factor shocks
        factor_shocks = {
            'Market': -3.0,  # Market down 3 standard deviations
            'Value': 1.0,    # Value up 1 standard deviation
            'Size': -1.5     # Size down 1.5 standard deviations
        }

        # Run stress test
        results = self.risk_model.stress_test_portfolio(
            weights=self.portfolio_weights,
            factor_shocks=factor_shocks
        )

        # Check the results structure
        self.assertIn('expected_loss', results)
        self.assertIn('factor_contributions', results)
        self.assertIn('asset_returns', results)

        # The expected loss should be positive (representing a loss)
        # because we have significant negative market shock
        self.assertGreater(results['expected_loss'], 0)

        # Market should be a major contributor to loss
        self.assertIn('Market', results['factor_contributions'])
        self.assertGreater(results['factor_contributions']['Market'], 0)

        # Check asset returns
        for asset in self.assets:
            self.assertIn(asset, results['asset_returns'])
            # Stocks with high market beta should have large negative returns
            if asset in ['AAPL', 'AMZN']:  # High market beta stocks in our setup
                self.assertLess(results['asset_returns'][asset], -0.01)  # Should be quite negative

    def test_stress_test_portfolio_empty(self):
        """Test stress testing with empty portfolio."""
        # First estimate the model
        self.risk_model.estimate_from_returns(
            returns=self.asset_returns,
            factors=self.factor_returns
        )

        # Test with empty portfolio
        empty_weights = {'CASH': 1.0}
        factor_shocks = {'Market': -3.0}

        results = self.risk_model.stress_test_portfolio(empty_weights, factor_shocks)
        self.assertEqual(results['expected_loss'], 0.0)

    def test_stress_test_portfolio_not_estimated(self):
        """Test stress testing before model estimation."""
        factor_shocks = {'Market': -3.0}
        with self.assertRaises(ValueError):
            self.risk_model.stress_test_portfolio(self.portfolio_weights, factor_shocks)

    def test_save_and_load(self):
        """Test saving and loading the risk model."""
        # First estimate the model
        self.risk_model.estimate_from_returns(
            returns=self.asset_returns,
            factors=self.factor_returns
        )

        # Save the model to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            temp_path = tmp.name

        try:
            self.risk_model.save(temp_path)

            # Load the model
            loaded_model = RiskModel.load(temp_path)

            # Check that loaded model has the same parameters
            pd.testing.assert_frame_equal(loaded_model.factor_covariance, self.risk_model.factor_covariance)
            pd.testing.assert_frame_equal(loaded_model.factor_exposures, self.risk_model.factor_exposures)
            self.assertEqual(loaded_model.specific_variances, self.risk_model.specific_variances)
            self.assertEqual(loaded_model.estimation_universe, self.risk_model.estimation_universe)
            self.assertEqual(loaded_model.factor_names, self.risk_model.factor_names)

            # Check that loaded model gives same results
            decomp1 = self.risk_model.decompose_portfolio_risk(self.portfolio_weights)
            decomp2 = loaded_model.decompose_portfolio_risk(self.portfolio_weights)

            self.assertAlmostEqual(decomp1.total_risk, decomp2.total_risk)
            self.assertAlmostEqual(decomp1.systematic_risk, decomp2.systematic_risk)
            self.assertAlmostEqual(decomp1.specific_risk, decomp2.specific_risk)

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_error_handling(self):
        """Test error handling during save operation."""
        # Mock pickle.dump to raise an exception
        with patch('pickle.dump', side_effect=Exception('Mocked error')), \
             patch('logging.Logger.error') as mock_error:
            self.risk_model.save('invalid_file.pkl')
            mock_error.assert_called()

    def test_load_error_handling(self):
        """Test error handling during load operation."""
        # Try to load from a non-existent file
        with patch('logging.Logger.error') as mock_error:
            model = RiskModel.load('non_existent_file.pkl')
            mock_error.assert_called()

            # Should return an empty model
            self.assertIsNone(model.factor_covariance)
            self.assertIsNone(model.factor_exposures)
            self.assertEqual(model.specific_variances, {})


if __name__ == '__main__':
    unittest.main()