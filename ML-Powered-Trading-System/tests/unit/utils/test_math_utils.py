import unittest
import numpy as np
import pandas as pd
from scipy import stats
import warnings
import math
import sys
import os

# Add parent directory to path to allow importing from utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.math_utils import (
    safe_computation,
    winsorize,
    robust_mean,
    robust_std,
    modified_z_score,
    exponential_weighted_std,
    calculate_returns,
    realized_volatility,
    parkinson_volatility,
    garman_klass_volatility,
    calculate_var,
    calculate_cvar,
    hurst_exponent,
    half_life_mean_reversion,
    detrend_time_series,
    detect_outliers,
    lowpass_filter,
    robust_correlation,
    correlation_significance,
    ewma_correlation,
    shrink_covariance,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    calculate_drawdowns,
    information_ratio,
    omega_ratio,
    minimize_portfolio_variance,
    maximize_sharpe_ratio,
    risk_budgeting_weights,
    rolling_window_dataset,
    train_test_split_time_series,
    directional_movement_index,
    z_score_normalization
)


class TestSafeComputationDecorator(unittest.TestCase):
    """Test the safe_computation decorator."""

    def test_normal_execution(self):
        @safe_computation(default_value=999)
        def add_numbers(a, b):
            return a + b
        
        self.assertEqual(add_numbers(2, 3), 5)
    
    def test_exception_handling(self):
        @safe_computation(default_value=999)
        def divide_by_zero(a, b):
            return a / b
        
        self.assertEqual(divide_by_zero(1, 0), 999)
        
    def test_default_nan(self):
        @safe_computation()  # Use default (np.nan)
        def raise_error():
            raise ValueError("Test error")
        
        self.assertTrue(np.isnan(raise_error()))


class TestBasicStatisticalFunctions(unittest.TestCase):
    """Test basic statistical functions."""
    
    def setUp(self):
        # Sample data for tests
        self.normal_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.outlier_data = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        self.small_data = np.array([1.0, 2.0])
        self.empty_data = np.array([])
        self.nan_data = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    
    def test_winsorize(self):
        # Test with normal data
        result = winsorize(self.normal_data, limits=(0.1, 0.1))
        np.testing.assert_allclose(result, np.array([1.4, 2.0, 3.0, 4.0, 4.6]), rtol=1e-5)
        
        # Test with outlier data
        result = winsorize(self.outlier_data, limits=(0.1, 0.1))
        # With outliers, should limit the upper value more dramatically
        self.assertTrue(result[-1] < 100.0)
        
        # Test with small data - should return unchanged
        result = winsorize(self.small_data)
        np.testing.assert_array_equal(result, self.small_data)
        
        # Test with empty data - should return unchanged
        result = winsorize(self.empty_data)
        np.testing.assert_array_equal(result, self.empty_data)
        
        # Test with NaN data - NaNs should be preserved
        result = winsorize(self.nan_data)
        self.assertTrue(np.isnan(result[1]))

    def test_robust_mean(self):
        # Test with normal data
        self.assertAlmostEqual(robust_mean(self.normal_data, trim_pct=0.2), 3.0)
        
        # Test with outlier data
        self.assertAlmostEqual(robust_mean(self.outlier_data, trim_pct=0.2), 3.0)
        
        # Test with NaN data
        mean_val = robust_mean(self.nan_data)
        self.assertFalse(np.isnan(mean_val))  # Should handle NaNs properly
        
        # Test with small data - should use regular mean
        self.assertEqual(robust_mean(self.small_data), 1.5)

    def test_robust_std(self):
        # Test with normal data
        expected_std = np.std(self.normal_data)
        self.assertAlmostEqual(robust_std(self.normal_data), expected_std, places=5)
        
        # Test with outlier data - robust std should be less than regular std
        regular_std = np.std(self.outlier_data)
        robust = robust_std(self.outlier_data)
        self.assertLess(robust, regular_std)
        
        # Test with small data
        self.assertAlmostEqual(robust_std(self.small_data), np.std(self.small_data))
        
        # Test with empty data
        self.assertTrue(np.isnan(robust_std(self.empty_data)))

    def test_modified_z_score(self):
        # Test with normal data
        z_scores = modified_z_score(self.normal_data)
        self.assertEqual(len(z_scores), len(self.normal_data))
        
        # Middle value should have z-score near 0
        self.assertAlmostEqual(z_scores[2], 0.0, places=5)
        
        # Test with outlier data
        z_scores = modified_z_score(self.outlier_data)
        # Outlier should have high z-score
        self.assertGreater(abs(z_scores[-1]), 3.0)
        
        # Test with small data
        z_scores = modified_z_score(self.small_data)
        np.testing.assert_array_equal(z_scores, np.zeros_like(self.small_data))

    def test_exponential_weighted_std(self):
        # Test with normal data
        ew_std = exponential_weighted_std(self.normal_data, alpha=0.1)
        self.assertFalse(np.isnan(ew_std))
        
        # Test with NaN data
        ew_std = exponential_weighted_std(self.nan_data)
        self.assertFalse(np.isnan(ew_std))
        
        # Test with different alpha values
        ew_std1 = exponential_weighted_std(self.normal_data, alpha=0.1)
        ew_std2 = exponential_weighted_std(self.normal_data, alpha=0.5)
        self.assertNotEqual(ew_std1, ew_std2)


class TestReturnsAndVolatilityCalculations(unittest.TestCase):
    """Test returns and volatility calculation functions."""
    
    def setUp(self):
        # Sample price data
        self.prices = np.array([100.0, 102.0, 101.0, 103.0, 105.0, 104.0])
        self.returns = np.array([0.02, -0.01, 0.02, 0.019, -0.01])
        self.high_prices = np.array([105.0, 103.0, 104.0, 107.0, 108.0])
        self.low_prices = np.array([99.0, 100.0, 100.0, 102.0, 103.0])
        self.open_prices = np.array([100.0, 102.0, 101.0, 103.0, 105.0])
        self.close_prices = np.array([102.0, 101.0, 103.0, 105.0, 104.0])
    
    def test_calculate_returns(self):
        # Test log returns
        log_returns = calculate_returns(self.prices, method='log')
        self.assertEqual(len(log_returns), len(self.prices) - 1)
        self.assertAlmostEqual(log_returns[0], np.log(102.0/100.0), places=10)
        
        # Test simple returns
        simple_returns = calculate_returns(self.prices, method='simple')
        self.assertAlmostEqual(simple_returns[0], 0.02, places=10)
        
        # Test pct_change
        pct_returns = calculate_returns(self.prices, method='pct_change')
        self.assertAlmostEqual(pct_returns[0], 0.02, places=10)
        
        # Test with invalid method
        with self.assertRaises(ValueError):
            calculate_returns(self.prices, method='invalid')
        
        # Test with small data
        small_prices = np.array([100.0])
        result = calculate_returns(small_prices)
        np.testing.assert_array_equal(result, np.zeros(1))

    def test_realized_volatility(self):
        # Test with normal returns
        vol = realized_volatility(self.returns, annualization_factor=252)
        self.assertGreater(vol, 0)
        
        # Test with different annualization factors
        vol1 = realized_volatility(self.returns, annualization_factor=252)
        vol2 = realized_volatility(self.returns, annualization_factor=12)
        self.assertNotEqual(vol1, vol2)
        
        # Test with empty data
        vol = realized_volatility(np.array([]))
        self.assertEqual(vol, 0.0)
        
        # Test with NaN values
        returns_with_nan = np.array([0.01, np.nan, 0.02, -0.01])
        vol = realized_volatility(returns_with_nan)
        self.assertFalse(np.isnan(vol))

    def test_parkinson_volatility(self):
        vol = parkinson_volatility(self.high_prices, self.low_prices)
        self.assertGreater(vol, 0)
        
        # Test with different annualization factors
        vol1 = parkinson_volatility(self.high_prices, self.low_prices, annualization_factor=252)
        vol2 = parkinson_volatility(self.high_prices, self.low_prices, annualization_factor=12)
        self.assertNotEqual(vol1, vol2)
        
        # Test with empty data
        vol = parkinson_volatility(np.array([]), np.array([]))
        self.assertEqual(vol, 0.0)

    def test_garman_klass_volatility(self):
        vol = garman_klass_volatility(
            self.open_prices, self.high_prices, self.low_prices, self.close_prices
        )
        self.assertGreater(vol, 0)
        
        # Test with different annualization factors
        vol1 = garman_klass_volatility(
            self.open_prices, self.high_prices, self.low_prices, self.close_prices,
            annualization_factor=252
        )
        vol2 = garman_klass_volatility(
            self.open_prices, self.high_prices, self.low_prices, self.close_prices,
            annualization_factor=12
        )
        self.assertNotEqual(vol1, vol2)
        
        # Test with empty data
        vol = garman_klass_volatility(np.array([]), np.array([]), np.array([]), np.array([]))
        self.assertEqual(vol, 0.0)

    def test_calculate_var(self):
        # Test historical VaR
        var = calculate_var(self.returns, alpha=0.05, method='historical')
        self.assertLess(var, 0)  # Negative value for 5% worst case
        
        # Test parametric VaR
        var = calculate_var(self.returns, alpha=0.05, method='parametric')
        self.assertLess(var, 0)
        
        # Test Cornish-Fisher VaR
        var = calculate_var(self.returns, alpha=0.05, method='cornish_fisher')
        self.assertLess(var, 0)
        
        # Test with invalid method
        with self.assertRaises(ValueError):
            calculate_var(self.returns, method='invalid')
        
        # Test with empty data
        var = calculate_var(np.array([]))
        self.assertEqual(var, 0.0)

    def test_calculate_cvar(self):
        # Test CVaR calculation
        cvar = calculate_cvar(self.returns, alpha=0.05)
        self.assertLess(cvar, 0)  # CVaR should be negative for loss
        
        # CVaR should be worse (more negative) than VaR
        var = calculate_var(self.returns, alpha=0.05, method='historical')
        self.assertLess(cvar, var)
        
        # Test with empty data
        cvar = calculate_cvar(np.array([]))
        self.assertEqual(cvar, 0.0)


class TestTimeSeriesAnalysis(unittest.TestCase):
    """Test time series analysis functions."""
    
    def setUp(self):
        # Generate random walk (Hurst ~ 0.5)
        np.random.seed(42)
        self.random_walk = np.cumsum(np.random.normal(0, 1, 100))
        
        # Generate trending series (Hurst > 0.5)
        self.trending = np.linspace(0, 10, 100) + np.random.normal(0, 0.1, 100)
        
        # Generate mean-reverting series (Hurst < 0.5)
        x = np.linspace(0, 5, 100)
        self.mean_reverting = np.sin(x) + np.random.normal(0, 0.1, 100)
        
        # Linear trend for detrending
        self.linear_trend = np.linspace(0, 10, 100) + np.random.normal(0, 1, 100)
        
        # Series with outliers
        self.outlier_series = np.random.normal(0, 1, 100)
        self.outlier_series[50] = 10.0  # Add outlier
    
    def test_hurst_exponent(self):
        # Test random walk (should be near 0.5)
        h, r2 = hurst_exponent(self.random_walk)
        self.assertIsNotNone(h)
        self.assertGreater(h, 0.3)
        self.assertLess(h, 0.7)
        
        # Test trending series (should be > 0.5)
        h, r2 = hurst_exponent(self.trending)
        self.assertIsNotNone(h)
        self.assertGreater(h, 0.5)
        
        # Test mean-reverting series (should be < 0.5)
        h, r2 = hurst_exponent(self.mean_reverting)
        self.assertIsNotNone(h)
        self.assertLess(h, 0.5)
        
        # Test with small data
        h, r2 = hurst_exponent(np.random.normal(0, 1, 10))
        self.assertIsNone(h)
        self.assertIsNone(r2)

    def test_half_life_mean_reversion(self):
        # Test mean-reverting series (should have finite half-life)
        half_life, lambda_coef, t_stat = half_life_mean_reversion(self.mean_reverting)
        self.assertIsNotNone(half_life)
        self.assertLess(lambda_coef, 0)  # Lambda should be negative for mean-reversion
        
        # Test trending series (should have infinite or very large half-life)
        half_life, lambda_coef, t_stat = half_life_mean_reversion(self.trending)
        self.assertEqual(half_life, np.inf)
        
        # Test with small data
        half_life, lambda_coef, t_stat = half_life_mean_reversion(np.random.normal(0, 1, 3))
        self.assertIsNone(half_life)
        self.assertIsNone(lambda_coef)
        self.assertIsNone(t_stat)

    def test_detrend_time_series(self):
        # Test linear detrending
        detrended = detrend_time_series(self.linear_trend, method='linear')
        self.assertEqual(len(detrended), len(self.linear_trend))
        # Detrended series should have mean close to zero
        self.assertAlmostEqual(np.mean(detrended), 0, places=1)
        
        # Test polynomial detrending
        detrended = detrend_time_series(self.linear_trend, method='polynomial_2')
        self.assertAlmostEqual(np.mean(detrended), 0, places=1)
        
        # Test ewma detrending
        detrended = detrend_time_series(self.linear_trend, method='ewma')
        self.assertEqual(len(detrended), len(self.linear_trend))
        
        # Test with invalid method
        with self.assertRaises(ValueError):
            detrend_time_series(self.linear_trend, method='invalid')
        
        # Test with NaN values
        series_with_nan = self.linear_trend.copy()
        series_with_nan[10] = np.nan
        detrended = detrend_time_series(series_with_nan)
        self.assertTrue(np.isnan(detrended[10]))

    def test_detect_outliers(self):
        # Test zscore method
        outliers, z_scores = detect_outliers(self.outlier_series, method='zscore')
        self.assertTrue(outliers[50])  # Index 50 should be flagged as outlier
        self.assertGreater(abs(z_scores[50]), 3.0)
        
        # Test modified_zscore method
        outliers, z_scores = detect_outliers(self.outlier_series, method='modified_zscore')
        self.assertTrue(outliers[50])
        
        # Test iqr method
        outliers, z_scores = detect_outliers(self.outlier_series, method='iqr')
        self.assertTrue(outliers[50])
        
        # Test with invalid method
        with self.assertRaises(ValueError):
            detect_outliers(self.outlier_series, method='invalid')
        
        # Test with NaN values
        series_with_nan = self.outlier_series.copy()
        series_with_nan[10] = np.nan
        outliers, z_scores = detect_outliers(series_with_nan)
        self.assertFalse(outliers[10])  # NaN should not be flagged as outlier
        self.assertTrue(np.isnan(series_with_nan[10]))  # Original NaN preserved

    def test_lowpass_filter(self):
        # Create noisy signal
        x = np.linspace(0, 10, 1000)
        signal = np.sin(x) + 0.2 * np.sin(20 * x)
        
        # Apply filter
        filtered = lowpass_filter(signal, cutoff=0.1, fs=10, order=3)
        self.assertEqual(len(filtered), len(signal))
        
        # Filtered signal should be smoother (less variance)
        self.assertLess(np.var(filtered), np.var(signal))
        
        # Test with NaN values
        signal_with_nan = signal.copy()
        signal_with_nan[100:110] = np.nan
        filtered = lowpass_filter(signal_with_nan, cutoff=0.1)
        # NaNs should be preserved in output
        self.assertTrue(np.isnan(filtered[105]))
        
        # Test with small data
        small_signal = np.sin(np.linspace(0, 10, 5))
        filtered = lowpass_filter(small_signal, cutoff=0.1, order=2)
        self.assertEqual(len(filtered), len(small_signal))


class TestCorrelationAndCovarianceFunctions(unittest.TestCase):
    """Test correlation and covariance functions."""
    
    def setUp(self):
        np.random.seed(42)
        self.x = np.random.normal(0, 1, 100)
        
        # Create correlated series
        self.y_strong_pos = self.x + np.random.normal(0, 0.1, 100)  # Strong positive correlation
        self.y_weak_pos = self.x + np.random.normal(0, 1, 100)      # Weak positive correlation
        self.y_neg = -self.x + np.random.normal(0, 0.1, 100)        # Negative correlation
        self.y_uncorr = np.random.normal(0, 1, 100)                 # Uncorrelated
        
        # Create series with NaNs
        self.x_with_nan = self.x.copy()
        self.x_with_nan[10:15] = np.nan
        
        # Create returns for covariance
        self.returns = np.random.normal(0, 1, (100, 3))  # 100 days, 3 assets
    
    def test_robust_correlation(self):
        # Test strong positive correlation
        corr = robust_correlation(self.x, self.y_strong_pos, method='spearman')
        self.assertGreater(corr, 0.9)
        
        # Test weak positive correlation
        corr = robust_correlation(self.x, self.y_weak_pos, method='spearman')
        self.assertGreater(corr, 0)
        
        # Test negative correlation
        corr = robust_correlation(self.x, self.y_neg, method='spearman')
        self.assertLess(corr, -0.9)
        
        # Test uncorrelated series
        corr = robust_correlation(self.x, self.y_uncorr, method='spearman')
        self.assertGreater(abs(corr), -0.5)
        self.assertLess(abs(corr), 0.5)
        
        # Test with different methods
        corr_pearson = robust_correlation(self.x, self.y_strong_pos, method='pearson')
        corr_kendall = robust_correlation(self.x, self.y_strong_pos, method='kendall')
        self.assertNotEqual(corr_pearson, corr_kendall)
        
        # Test with invalid method
        with self.assertRaises(ValueError):
            robust_correlation(self.x, self.y_strong_pos, method='invalid')
        
        # Test with NaN values
        corr = robust_correlation(self.x_with_nan, self.y_strong_pos)
        self.assertFalse(np.isnan(corr))

    def test_correlation_significance(self):
        # Test strong correlation (should have low p-value)
        corr, p_value = correlation_significance(self.x, self.y_strong_pos)
        self.assertGreater(corr, 0.9)
        self.assertLess(p_value, 0.01)
        
        # Test uncorrelated series (should have high p-value)
        corr, p_value = correlation_significance(self.x, self.y_uncorr)
        self.assertGreater(p_value, 0.01)
        
        # Test with NaN values
        corr, p_value = correlation_significance(self.x_with_nan, self.y_strong_pos)
        self.assertFalse(np.isnan(corr))
        self.assertFalse(np.isnan(p_value))

    def test_ewma_correlation(self):
        # Test EWMA correlation
        corr = ewma_correlation(self.x, self.y_strong_pos, alpha=0.1)
        self.assertGreater(corr, 0.9)
        
        # Test with different alpha
        corr1 = ewma_correlation(self.x, self.y_strong_pos, alpha=0.1)
        corr2 = ewma_correlation(self.x, self.y_strong_pos, alpha=0.5)
        self.assertNotEqual(corr1, corr2)
        
        # Test with NaN values
        corr = ewma_correlation(self.x_with_nan, self.y_strong_pos)
        self.assertFalse(np.isnan(corr))

    def test_shrink_covariance(self):
        # Test with no shrinkage
        cov_original = np.cov(self.returns.T)
        cov_shrink = shrink_covariance(self.returns, shrinkage=0)
        np.testing.assert_allclose(cov_original, cov_shrink, rtol=1e-5)
        
        # Test with full shrinkage to target
        cov_shrink = shrink_covariance(self.returns, shrinkage=1)
        self.assertTrue(np.all(np.diag(np.diag(cov_original)) == cov_shrink))
        
        # Test automatic shrinkage estimation
        cov_shrink = shrink_covariance(self.returns)
        self.assertFalse(np.array_equal(cov_original, cov_shrink))
        
        # Test with 1D array
        var_shrink = shrink_covariance(self.x)
        self.assertAlmostEqual(var_shrink, np.var(self.x))


class TestFinancialMetrics(unittest.TestCase):
    """Test financial metrics functions."""
    
    def setUp(self):
        np.random.seed(42)
        # Create positive-mean returns
        self.returns_pos = np.random.normal(0.001, 0.01, 252)  # 1 year, positive mean
        # Create negative-mean returns
        self.returns_neg = np.random.normal(-0.001, 0.01, 252)  # 1 year, negative mean
        # Create benchmark returns
        self.benchmark = np.random.normal(0.0005, 0.01, 252)  # 1 year, lower mean
        # Create covariance matrix and expected returns for portfolio optimization
        self.cov_matrix = np.array([
            [0.04, 0.02, 0.01],
            [0.02, 0.09, 0.03],
            [0.01, 0.03, 0.16]
        ])
        self.expected_returns = np.array([0.10, 0.15, 0.20])
    
    def test_sharpe_ratio(self):
        # Test positive returns (should have positive Sharpe)
        sharpe = sharpe_ratio(self.returns_pos, risk_free_rate=0.0)
        self.assertGreater(sharpe, 0)
        
        # Test negative returns (should have negative Sharpe)
        sharpe = sharpe_ratio(self.returns_neg, risk_free_rate=0.0)
        self.assertLess(sharpe, 0)
        
        # Test with risk-free rate
        sharpe1 = sharpe_ratio(self.returns_pos, risk_free_rate=0.0)
        sharpe2 = sharpe_ratio(self.returns_pos, risk_free_rate=0.02)
        self.assertNotEqual(sharpe1, sharpe2)
        
        # Test with different annualization factors
        sharpe1 = sharpe_ratio(self.returns_pos, annualization_factor=252)
        sharpe2 = sharpe_ratio(self.returns_pos, annualization_factor=12)
        self.assertNotEqual(sharpe1, sharpe2)
        
        # Test with NaN values
        returns_with_nan = self.returns_pos.copy()
        returns_with_nan[10:15] = np.nan
        sharpe = sharpe_ratio(returns_with_nan)
        self.assertFalse(np.isnan(sharpe))

    def test_sortino_ratio(self):
        # Test positive returns (should have positive Sortino)
        sortino = sortino_ratio(self.returns_pos, risk_free_rate=0.0)
        self.assertGreater(sortino, 0)
        
        # Test negative returns (should have negative Sortino)
        sortino = sortino_ratio(self.returns_neg, risk_free_rate=0.0)
        self.assertLess(sortino, 0)
        
        # Sortino should be higher than Sharpe for positive returns
        # because it only penalizes downside risk
        sharpe = sharpe_ratio(self.returns_pos)
        sortino = sortino_ratio(self.returns_pos)
        self.assertGreater(sortino, sharpe)
        
        # Test with target return
        sortino1 = sortino_ratio(self.returns_pos, target_return=0.0)
        sortino2 = sortino_ratio(self.returns_pos, target_return=0.001)
        self.assertNotEqual(sortino1, sortino2)
        
        # Test with NaN values
        returns_with_nan = self.returns_pos.copy()
        returns_with_nan[10:15] = np.nan
        sortino = sortino_ratio(returns_with_nan)
        self.assertFalse(np.isnan(sortino))

    def test_calmar_ratio(self):
        # Create returns with a significant drawdown
        returns = np.ones(100) * 0.001
        returns[40:50] = -0.05  # 10-day drawdown
        
        # Test Calmar ratio
        calmar = calmar_ratio(returns, annualization_factor=252)
        self.assertGreater(calmar, 0)  # Should be positive for positive returns
        
        # Test with different windows
        calmar1 = calmar_ratio(returns, window=None)  # Full period
        calmar2 = calmar_ratio(returns, window=20)    # 20-day window
        self.assertNotEqual(calmar1, calmar2)
        
        # Test with NaN values
        returns_with_nan = returns.copy()
        returns_with_nan[10:15] = np.nan
        calmar = calmar_ratio(returns_with_nan)
        self.assertFalse(np.isnan(calmar))

    def test_calculate_drawdowns(self):
        # Create returns with a significant drawdown
        returns = np.ones(100) * 0.001
        returns[40:50] = -0.05  # 10-day drawdown
        
        # Test drawdown calculation
        drawdowns, max_dd = calculate_drawdowns(returns)
        self.assertEqual(len(drawdowns), len(returns))
        self.assertLess(max_dd, 0)  # Max drawdown should be negative
        
        # The largest drawdown should be after the negative returns
        self.assertEqual(np.argmin(drawdowns), 49)
        
        # Test with NaN values
        returns_with_nan = returns.copy()
        returns_with_nan[10:15] = np.nan
        drawdowns, max_dd = calculate_drawdowns(returns_with_nan)
        self.assertFalse(np.isnan(max_dd))
        
    def test_information_ratio(self):
        # Test with strategy outperforming benchmark
        ir = information_ratio(self.returns_pos, self.benchmark)
        self.assertGreater(ir, 0)
        
        # Test with strategy underperforming benchmark
        ir = information_ratio(self.returns_neg, self.benchmark)
        self.assertLess(ir, 0)
        
        # Test with different annualization factors
        ir1 = information_ratio(self.returns_pos, self.benchmark, annualization_factor=252)
        ir2 = information_ratio(self.returns_pos, self.benchmark, annualization_factor=12)
        self.assertNotEqual(ir1, ir2)
        
        # Test with different length arrays
        ir = information_ratio(self.returns_pos[:200], self.benchmark)
        self.assertFalse(np.isnan(ir))
        
        # Test with NaN values
        returns_with_nan = self.returns_pos.copy()
        returns_with_nan[10:15] = np.nan
        ir = information_ratio(returns_with_nan, self.benchmark)
        self.assertFalse(np.isnan(ir))
        
    def test_omega_ratio(self):
        # Test positive returns (should have omega > 1)
        omega = omega_ratio(self.returns_pos, threshold=0.0)
        self.assertGreater(omega, 1.0)
        
        # Test negative returns (should have omega < 1)
        omega = omega_ratio(self.returns_neg, threshold=0.0)
        self.assertLess(omega, 1.0)
        
        # Test with different thresholds
        omega1 = omega_ratio(self.returns_pos, threshold=0.0)
        omega2 = omega_ratio(self.returns_pos, threshold=0.001)
        self.assertNotEqual(omega1, omega2)
        
        # Test with all returns above threshold
        omega = omega_ratio(np.ones(100) * 0.01, threshold=0.0)
        self.assertEqual(omega, float('inf'))
        
        # Test with NaN values
        returns_with_nan = self.returns_pos.copy()
        returns_with_nan[10:15] = np.nan
        omega = omega_ratio(returns_with_nan)
        self.assertFalse(np.isnan(omega))
        
    def test_minimize_portfolio_variance(self):
        # Test minimum variance portfolio
        weights = minimize_portfolio_variance(self.cov_matrix)
        
        # Weights should sum to 1
        self.assertAlmostEqual(np.sum(weights), 1.0, places=6)
        
        # All weights should be between 0 and 1 (no leverage, no short)
        self.assertTrue(np.all(weights >= 0))
        self.assertTrue(np.all(weights <= 1))
        
        # First asset should have highest weight (lowest variance)
        self.assertEqual(np.argmax(weights), 0)
        
        # Test with custom bounds
        bounds = [(0, 0.5), (0, 0.5), (0, 0.5)]  # Restrict each asset to max 50%
        weights = minimize_portfolio_variance(self.cov_matrix, bounds=bounds)
        self.assertTrue(np.all(weights <= 0.5))
        
        # Test with custom constraints
        # Allow for shorting (weights can be negative)
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = [(-0.5, 1), (-0.5, 1), (-0.5, 1)]  # Allow shorting up to -50%
        weights = minimize_portfolio_variance(self.cov_matrix, constraints=constraints, bounds=bounds)
        self.assertAlmostEqual(np.sum(weights), 1.0, places=6)
        
    def test_maximize_sharpe_ratio(self):
        # Test maximum Sharpe ratio portfolio
        weights = maximize_sharpe_ratio(self.expected_returns, self.cov_matrix)
        
        # Weights should sum to 1
        self.assertAlmostEqual(np.sum(weights), 1.0, places=6)
        
        # All weights should be between 0 and 1 (no leverage, no short)
        self.assertTrue(np.all(weights >= 0))
        self.assertTrue(np.all(weights <= 1))
        
        # Third asset should have highest weight (highest return/risk ratio)
        self.assertEqual(np.argmax(weights), 2)
        
        # Test with risk-free rate
        weights1 = maximize_sharpe_ratio(self.expected_returns, self.cov_matrix, risk_free_rate=0.0)
        weights2 = maximize_sharpe_ratio(self.expected_returns, self.cov_matrix, risk_free_rate=0.05)
        # Different risk-free rates should give different weights
        self.assertFalse(np.array_equal(weights1, weights2))
        
    def test_risk_budgeting_weights(self):
        # Test equal risk contribution portfolio
        weights = risk_budgeting_weights(self.cov_matrix)
        
        # Weights should sum to 1
        self.assertAlmostEqual(np.sum(weights), 1.0, places=6)
        
        # All weights should be between 0 and 1 (no leverage, no short)
        self.assertTrue(np.all(weights >= 0))
        self.assertTrue(np.all(weights <= 1))
        
        # First asset should have highest weight (lowest risk)
        self.assertEqual(np.argmax(weights), 0)
        
        # Test with custom risk budget
        risk_budget = np.array([0.5, 0.25, 0.25])  # First asset gets 50% of risk
        weights = risk_budgeting_weights(self.cov_matrix, risk_budget=risk_budget)
        self.assertAlmostEqual(np.sum(weights), 1.0, places=6)


class TestMachineLearningUtilities(unittest.TestCase):
    """Test machine learning utility functions."""
    
    def setUp(self):
        np.random.seed(42)
        # Create sample time series data
        self.X = np.random.normal(0, 1, (100, 5))  # 100 samples, 5 features
        self.y = np.random.normal(0, 1, 100)       # 100 target values
    
    def test_rolling_window_dataset(self):
        # Test with window size 10, forecast horizon 1
        X_windows, y_targets = rolling_window_dataset(self.X, self.y, window_size=10)
        
        # Check shapes
        self.assertEqual(X_windows.shape[0], 90)  # 100 - 10 = 90 windows
        self.assertEqual(X_windows.shape[1], 10)  # Window size
        self.assertEqual(X_windows.shape[2], 5)   # Features
        self.assertEqual(y_targets.shape[0], 90)  # 90 targets
        
        # Test with different forecast horizon
        X_windows, y_targets = rolling_window_dataset(self.X, self.y, window_size=10, forecast_horizon=5)
        self.assertEqual(X_windows.shape[0], 86)  # 100 - 10 - 5 + 1 = 86 windows
        
        # Test with different step size
        X_windows, y_targets = rolling_window_dataset(self.X, self.y, window_size=10, step=2)
        self.assertEqual(X_windows.shape[0], 45)  # (100 - 10) // 2 = 45 windows
        
        # Test with mismatched X, y
        with self.assertRaises(ValueError):
            rolling_window_dataset(self.X, self.y[:50], window_size=10)
    
    def test_train_test_split_time_series(self):
        # Test with just train/test split
        X_train, X_test, y_train, y_test = train_test_split_time_series(self.X, self.y, test_size=0.2)
        
        # Check shapes
        self.assertEqual(X_train.shape[0], 80)  # 80% of data
        self.assertEqual(X_test.shape[0], 20)   # 20% of data
        self.assertEqual(y_train.shape[0], 80)
        self.assertEqual(y_test.shape[0], 20)
        
        # Verify the split maintains temporal order
        self.assertTrue(np.array_equal(X_train, self.X[:80]))
        self.assertTrue(np.array_equal(X_test, self.X[80:]))
        
        # Test with train/val/test split
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split_time_series(
            self.X, self.y, test_size=0.2, validation_size=0.25
        )
        
        # Check shapes
        self.assertEqual(X_train.shape[0], 60)  # 60% of data (80% * 75%)
        self.assertEqual(X_val.shape[0], 20)    # 20% of data (80% * 25%)
        self.assertEqual(X_test.shape[0], 20)   # 20% of data
        
        # Verify the split maintains temporal order
        self.assertTrue(np.array_equal(X_train, self.X[:60]))
        self.assertTrue(np.array_equal(X_val, self.X[60:80]))
        self.assertTrue(np.array_equal(X_test, self.X[80:]))
        
        # Test with mismatched X, y
        with self.assertRaises(ValueError):
            train_test_split_time_series(self.X, self.y[:50])
    
    def test_purged_cross_validation_splits(self):
        # Test with 5 splits, no embargo
        splits = purged_cross_validation_splits(self.X, self.y, n_splits=5)
        
        # Should have 5 splits
        self.assertEqual(len(splits), 5)
        
        # Each split should be a tuple of (train_idx, test_idx)
        for train_idx, test_idx in splits:
            # Train and test should be disjoint
            self.assertEqual(len(np.intersect1d(train_idx, test_idx)), 0)
            
            # All indices should be covered
            self.assertEqual(len(np.union1d(train_idx, test_idx)), 100)
        
        # Test with embargo
        splits = purged_cross_validation_splits(self.X, self.y, n_splits=5, embargo_size=0.2)
        
        # Should still have 5 splits
        self.assertEqual(len(splits), 5)
        
        # With embargo, train and test should have a gap between them
        for i, (train_idx, test_idx) in enumerate(splits):
            if i < 4:  # Skip last fold which doesn't have data after it
                # Get min/max indices
                min_test = np.min(test_idx)
                max_test = np.max(test_idx)
                
                # Check for embargo gap after test set
                for j in range(max_test + 1, min(max_test + 5, 100)):  # Embargo should be 4 samples (20% of 20)
                    self.assertFalse(j in train_idx)


class TestDirectionalMovementAndZScore(unittest.TestCase):
    """Test directional movement and z-score functions."""
    
    def setUp(self):
        np.random.seed(42)
        # Create sample OHLC data
        self.dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        self.close = np.cumsum(np.random.normal(0, 1, 100)) + 100
        self.high = self.close + np.random.uniform(0, 2, 100)
        self.low = self.close - np.random.uniform(0, 2, 100)
        self.open = self.close - np.random.normal(0, 1, 100)
    
    def test_directional_movement_index(self):
        # Test DMI calculation
        adx, pdi, ndi = directional_movement_index(self.high, self.low, self.close)
        
        # ADX should be between 0 and 100
        self.assertTrue(np.all((adx >= 0) | np.isnan(adx)))
        self.assertTrue(np.all((adx <= 100) | np.isnan(adx)))
        
        # PDI and NDI should be between 0 and 100
        self.assertTrue(np.all((pdi >= 0) | np.isnan(pdi)))
        self.assertTrue(np.all((pdi <= 100) | np.isnan(pdi)))
        self.assertTrue(np.all((ndi >= 0) | np.isnan(ndi)))
        self.assertTrue(np.all((ndi <= 100) | np.isnan(ndi)))
        
        # First value should be NaN
        self.assertTrue(np.isnan(adx[0]))
        self.assertTrue(np.isnan(pdi[0]))
        self.assertTrue(np.isnan(ndi[0]))
        
        # Test with different window
        adx1, pdi1, ndi1 = directional_movement_index(self.high, self.low, self.close, window=14)
        adx2, pdi2, ndi2 = directional_movement_index(self.high, self.low, self.close, window=7)
        # Different windows should give different results
        self.assertFalse(np.array_equal(adx1, adx2))
        
        # Test with small data
        adx, pdi, ndi = directional_movement_index(self.high[:5], self.low[:5], self.close[:5])
        self.assertIsNone(adx)
    
    def test_z_score_normalization(self):
        # Test static normalization
        z_scores = z_score_normalization(self.close)
        
        # Z-scores should have mean ≈ 0 and std ≈ 1
        self.assertAlmostEqual(np.mean(z_scores), 0, places=10)
        self.assertAlmostEqual(np.std(z_scores), 1, places=10)
        
        # Test rolling normalization
        z_scores = z_score_normalization(self.close, window=20)
        
        # First values should be 0 (not enough data)
        self.assertTrue(np.all(z_scores[:5] == 0))
        
        # Last values should be normalized
        self.assertTrue(np.abs(np.mean(z_scores[-20:])) < 0.5)
        
        # Test with center=False
        z_scores = z_score_normalization(self.close, window=20, center=False)
        
        # First values should still be 0
        self.assertTrue(np.all(z_scores[:5] == 0))
        
        # Test with NaN values
        data_with_nan = self.close.copy()
        data_with_nan[10:15] = np.nan
        z_scores = z_score_normalization(data_with_nan)
        self.assertTrue(np.isnan(z_scores[10:15]).all())


if __name__ == '__main__':
    unittest.main()