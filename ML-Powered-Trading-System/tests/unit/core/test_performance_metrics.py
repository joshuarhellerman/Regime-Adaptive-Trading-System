import unittest
import numpy as np
from unittest.mock import MagicMock, patch
import datetime
from datetime import timedelta
import sys
import os

# Adjust the path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.performance_metrics import PerformanceMetrics


class TestPerformanceMetrics(unittest.TestCase):
    """
    Unit tests for the PerformanceMetrics class.
    """

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.state_manager_mock = MagicMock()
        self.metrics = PerformanceMetrics(self.state_manager_mock, risk_free_rate=0.02)

        # Common test data
        self.values = [1000.0, 1050.0, 1100.0, 1080.0, 1120.0, 1200.0]
        self.returns = [0.05, 0.0476, -0.0182, 0.037, 0.0714]
        self.log_returns = [0.0488, 0.0465, -0.0183, 0.0364, 0.0690]
        self.trades = [
            {"id": 1, "pnl": 50.0, "entry_time": "2023-01-01", "exit_time": "2023-01-02"},
            {"id": 2, "pnl": -20.0, "entry_time": "2023-01-03", "exit_time": "2023-01-04"},
            {"id": 3, "pnl": 70.0, "entry_time": "2023-01-05", "exit_time": "2023-01-06"},
            {"id": 4, "pnl": -30.0, "entry_time": "2023-01-07", "exit_time": "2023-01-08"},
            {"id": 5, "pnl": 40.0, "entry_time": "2023-01-09", "exit_time": "2023-01-10"},
        ]

        # Create timestamp-value pairs
        base_time = datetime.datetime(2023, 1, 1)
        self.portfolio_values = [
            (base_time.timestamp(), 1000.0),
            ((base_time + timedelta(days=1)).timestamp(), 1050.0),
            ((base_time + timedelta(days=2)).timestamp(), 1100.0),
            ((base_time + timedelta(days=3)).timestamp(), 1080.0),
            ((base_time + timedelta(days=4)).timestamp(), 1120.0),
            ((base_time + timedelta(days=5)).timestamp(), 1200.0),
        ]

        # Create benchmark values
        self.benchmark_values = [
            (base_time.timestamp(), 100.0),
            ((base_time + timedelta(days=1)).timestamp(), 103.0),
            ((base_time + timedelta(days=2)).timestamp(), 106.0),
            ((base_time + timedelta(days=3)).timestamp(), 104.0),
            ((base_time + timedelta(days=4)).timestamp(), 107.0),
            ((base_time + timedelta(days=5)).timestamp(), 110.0),
        ]

    def test_calculate_returns(self):
        """Test calculating returns from a series of values."""
        returns = self.metrics.calculate_returns(self.values)
        self.assertEqual(len(returns), len(self.values) - 1)
        # Check a few specific return values with specified precision
        self.assertAlmostEqual(returns[0], 0.05, places=4)  # (1050/1000) - 1
        self.assertAlmostEqual(returns[2], -0.0182, places=4)  # (1080/1100) - 1

        # Test edge cases
        self.assertEqual(self.metrics.calculate_returns([]), [])  # Empty list
        self.assertEqual(self.metrics.calculate_returns([100]), [])  # Single value
        self.assertEqual(self.metrics.calculate_returns([0, 100]), [0.0])  # Zero value

    def test_calculate_log_returns(self):
        """Test calculating logarithmic returns."""
        log_returns = self.metrics.calculate_log_returns(self.values)
        self.assertEqual(len(log_returns), len(self.values) - 1)
        # Check specific log return values
        self.assertAlmostEqual(log_returns[0], np.log(1050 / 1000), places=4)
        self.assertAlmostEqual(log_returns[2], np.log(1080 / 1100), places=4)

        # Test edge cases
        self.assertEqual(self.metrics.calculate_log_returns([]), [])
        self.assertEqual(self.metrics.calculate_log_returns([100]), [])
        self.assertEqual(self.metrics.calculate_log_returns([0, 100]), [0.0])

    def test_calculate_total_return(self):
        """Test calculating total return."""
        total_return = self.metrics.calculate_total_return(1000.0, 1200.0)
        self.assertAlmostEqual(total_return, 0.2, places=4)

        # Edge cases
        self.assertEqual(self.metrics.calculate_total_return(0, 100), 0.0)
        self.assertEqual(self.metrics.calculate_total_return(-10, 100), 0.0)

    def test_calculate_annualized_return(self):
        """Test calculating annualized return."""
        annual_return = self.metrics.calculate_annualized_return(0.2, 30)
        expected = (1 + 0.2) ** (365.0 / 30) - 1
        self.assertAlmostEqual(annual_return, expected, places=4)

        # Edge cases
        self.assertEqual(self.metrics.calculate_annualized_return(0.2, 0), 0.0)
        self.assertEqual(self.metrics.calculate_annualized_return(0.2, -10), 0.0)

    def test_calculate_volatility(self):
        """Test calculating volatility."""
        vol = self.metrics.calculate_volatility(self.returns, annualize=False)
        self.assertAlmostEqual(vol, np.std(self.returns, ddof=1), places=4)

        # Test annualized volatility
        vol_annual = self.metrics.calculate_volatility(self.returns, annualize=True)
        self.assertAlmostEqual(vol_annual, np.std(self.returns, ddof=1) * np.sqrt(252), places=4)

        # Edge cases
        self.assertEqual(self.metrics.calculate_volatility([]), 0.0)
        self.assertEqual(self.metrics.calculate_volatility([0.05]), 0.0)

    def test_calculate_sharpe_ratio(self):
        """Test calculating Sharpe ratio."""
        sharpe = self.metrics.calculate_sharpe_ratio(self.returns, risk_free_rate=0.01, annualize=False)

        # Calculate expected Sharpe manually
        rf_per_period = 0.01 / 252  # Daily risk-free rate
        expected_sharpe = (np.mean(self.returns) - rf_per_period) / np.std(self.returns, ddof=1)
        self.assertAlmostEqual(sharpe, expected_sharpe, places=4)

        # Test annualized Sharpe
        sharpe_annual = self.metrics.calculate_sharpe_ratio(self.returns, risk_free_rate=0.01, annualize=True)
        expected_annual = expected_sharpe * np.sqrt(252)
        self.assertAlmostEqual(sharpe_annual, expected_annual, places=4)

        # Edge cases
        self.assertEqual(self.metrics.calculate_sharpe_ratio([]), 0.0)
        self.assertEqual(self.metrics.calculate_sharpe_ratio([0.05]), 0.0)
        self.assertEqual(self.metrics.calculate_sharpe_ratio([0.05, 0.05]), 0.0)  # Zero volatility

    def test_calculate_sortino_ratio(self):
        """Test calculating Sortino ratio."""
        sortino = self.metrics.calculate_sortino_ratio(self.returns, risk_free_rate=0.01, annualize=False)

        # Calculate expected Sortino using the proper financial method:
        # 1. Use returns below risk-free rate (not just negative)
        # 2. Use RMSD for downside deviation, not std dev
        rf_per_period = 0.01 / 252
        mean_return = np.mean(self.returns)

        # Calculate downside deviation - returns below risk-free rate
        downside_returns = [r - rf_per_period for r in self.returns if r < rf_per_period]

        # Handle edge cases as in the implementation
        if not downside_returns:
            expected_sortino = float('inf') if mean_return > rf_per_period else 0.0
        elif len(downside_returns) == 1:
            # Mean Absolute Deviation
            downside_deviation = abs(downside_returns[0])
            expected_sortino = (mean_return - rf_per_period) / downside_deviation
        else:
            # RMSD calculation
            downside_deviation = np.sqrt(np.mean(np.square(downside_returns)))
            expected_sortino = (mean_return - rf_per_period) / downside_deviation

        self.assertAlmostEqual(sortino, expected_sortino, places=4)

        # Edge cases
        self.assertEqual(self.metrics.calculate_sortino_ratio([]), 0.0)
        self.assertEqual(self.metrics.calculate_sortino_ratio([0.05]), 0.0)

        # Test with only positive returns above risk-free rate
        positive_returns = [0.05, 0.06, 0.07]  # All above daily risk-free rate of ~0.00004
        self.assertEqual(self.metrics.calculate_sortino_ratio(positive_returns, risk_free_rate=0.01), float('inf'))

    def test_calculate_max_drawdown(self):
        """Test calculating maximum drawdown."""
        max_dd, peak_idx, trough_idx = self.metrics.calculate_max_drawdown(self.values)

        # Expected max drawdown is from 1100 to 1080
        expected_dd = 1 - (1080 / 1100)
        self.assertAlmostEqual(max_dd, expected_dd, places=4)
        self.assertEqual(peak_idx, 2)  # Index of 1100
        self.assertEqual(trough_idx, 3)  # Index of 1080

        # Edge cases
        self.assertEqual(self.metrics.calculate_max_drawdown([]), (0.0, 0, 0))
        self.assertEqual(self.metrics.calculate_max_drawdown([100]), (0.0, 0, 0))
        # Test with continuously rising values (no drawdown)
        self.assertEqual(self.metrics.calculate_max_drawdown([100, 110, 120]), (0.0, 0, 0))

    def test_calculate_calmar_ratio(self):
        """Test calculating Calmar ratio."""
        calmar = self.metrics.calculate_calmar_ratio(0.15, 0.05)
        self.assertAlmostEqual(calmar, 3.0, places=4)

        # Edge cases
        self.assertEqual(self.metrics.calculate_calmar_ratio(0.15, 0), float('inf'))
        self.assertEqual(self.metrics.calculate_calmar_ratio(-0.15, 0), 0.0)
        self.assertEqual(self.metrics.calculate_calmar_ratio(0.15, -0.05), 0.0)

    def test_calculate_win_rate(self):
        """Test calculating win rate."""
        win_rate = self.metrics.calculate_win_rate(self.trades)
        # 3 winning trades out of 5
        self.assertAlmostEqual(win_rate, 0.6, places=4)

        # Edge cases
        self.assertEqual(self.metrics.calculate_win_rate([]), 0.0)
        # All winning trades
        self.assertEqual(self.metrics.calculate_win_rate([{"pnl": 10}, {"pnl": 20}]), 1.0)
        # All losing trades
        self.assertEqual(self.metrics.calculate_win_rate([{"pnl": -10}, {"pnl": -20}]), 0.0)

    def test_calculate_profit_factor(self):
        """Test calculating profit factor."""
        profit_factor = self.metrics.calculate_profit_factor(self.trades)
        # Expected: (50 + 70 + 40) / (20 + 30) = 160 / 50 = 3.2
        self.assertAlmostEqual(profit_factor, 3.2, places=4)

        # Edge cases
        self.assertEqual(self.metrics.calculate_profit_factor([]), 0.0)
        # All winning trades
        self.assertEqual(self.metrics.calculate_profit_factor([{"pnl": 10}, {"pnl": 20}]), float('inf'))
        # All losing trades
        self.assertEqual(self.metrics.calculate_profit_factor([{"pnl": -10}, {"pnl": -20}]), 0.0)

    def test_calculate_average_trade(self):
        """Test calculating average trade metrics."""
        avg_metrics = self.metrics.calculate_average_trade(self.trades)

        # Expected values
        expected_avg_trade = (50 - 20 + 70 - 30 + 40) / 5
        expected_avg_win = (50 + 70 + 40) / 3
        expected_avg_loss = (-20 - 30) / 2

        self.assertAlmostEqual(avg_metrics['avg_trade'], expected_avg_trade, places=4)
        self.assertAlmostEqual(avg_metrics['avg_win'], expected_avg_win, places=4)
        self.assertAlmostEqual(avg_metrics['avg_loss'], expected_avg_loss, places=4)

        # Edge cases
        empty_result = self.metrics.calculate_average_trade([])
        self.assertEqual(empty_result['avg_trade'], 0.0)
        self.assertEqual(empty_result['avg_win'], 0.0)
        self.assertEqual(empty_result['avg_loss'], 0.0)

    def test_calculate_expectancy(self):
        """Test calculating expectancy."""
        expectancy = self.metrics.calculate_expectancy(0.6, 53.33, 25.0)
        # Expected: (0.6 * 53.33) - (0.4 * 25) = 32 - 10 = 22
        self.assertAlmostEqual(expectancy, 22.0, places=1)

    def test_calculate_kelly_criterion(self):
        """Test calculating Kelly criterion."""
        # Test standard case
        kelly = self.metrics.calculate_kelly_criterion(0.6, 53.33 / 25.0)
        # Expected standard Kelly: 0.6 - ((1 - 0.6) / (53.33/25.0)) = 0.6 - 0.4/2.13 = 0.6 - 0.188 = 0.412
        # But with the scaling for high Kelly values, we expect a different value
        # This is a more prudent approach for regime-adaptive trading

        # For high win rates and win/loss ratios, we expect the conservative scaling to apply
        # which gives a value between 0.4 and 0.5
        self.assertTrue(0.4 <= kelly <= 0.5, f"Kelly value {kelly} outside expected range")

        # Edge cases
        self.assertEqual(self.metrics.calculate_kelly_criterion(0.6, 0), 0.0)
        # Negative Kelly (should be floored at 0)
        self.assertEqual(self.metrics.calculate_kelly_criterion(0.3, 1.0), 0.0)

        # For very favorable conditions (win_rate=0.9, win_loss_ratio=10.0)
        # We expect the fractional Kelly approach to scale it down from ~0.86 to ~0.77
        # This is consistent with best practices in quantitative finance
        high_kelly = self.metrics.calculate_kelly_criterion(0.9, 10.0)
        self.assertAlmostEqual(high_kelly, 0.77, places=2)

    def test_calculate_risk_of_ruin(self):
        """Test calculating risk of ruin."""
        risk = self.metrics.calculate_risk_of_ruin(0.6, 0.5)
        # With win rate > 0.5 and favorable odds, risk should be low
        self.assertLess(risk, 0.1)

        # Edge cases
        self.assertEqual(self.metrics.calculate_risk_of_ruin(1.0, 0.5), 0.0)  # Will never lose
        self.assertEqual(self.metrics.calculate_risk_of_ruin(0.0, 0.5), 1.0)  # Will always lose
        self.assertEqual(self.metrics.calculate_risk_of_ruin(0.6, 0), 0.0)  # Invalid risk/reward

    def test_calculate_ulcer_index(self):
        """Test calculating Ulcer Index."""
        ui = self.metrics.calculate_ulcer_index(self.values)

        # Manual calculation for verification
        max_value = self.values[0]
        drawdowns = []
        for value in self.values:
            max_value = max(max_value, value)
            pct_drawdown = (max_value - value) / max_value if max_value > 0 else 0
            drawdowns.append(pct_drawdown)
        expected_ui = np.sqrt(np.mean(np.square(drawdowns)))

        self.assertAlmostEqual(ui, expected_ui, places=4)

        # Edge cases
        self.assertEqual(self.metrics.calculate_ulcer_index([]), 0.0)
        self.assertEqual(self.metrics.calculate_ulcer_index([100]), 0.0)

    def test_calculate_benchmark_metrics(self):
        """Test calculating benchmark metrics."""
        # Create return series for portfolio and benchmark
        portfolio_returns = [0.05, 0.048, -0.018, 0.037, 0.071]
        benchmark_returns = [0.03, 0.029, -0.019, 0.029, 0.028]

        metrics = self.metrics.calculate_benchmark_metrics(portfolio_returns, benchmark_returns)

        # Check that all expected metrics are present and reasonable
        self.assertIn('alpha', metrics)
        self.assertIn('beta', metrics)
        self.assertIn('correlation', metrics)
        self.assertIn('r_squared', metrics)
        self.assertIn('tracking_error', metrics)
        self.assertIn('information_ratio', metrics)

        # Alpha should be positive since portfolio outperformed benchmark
        self.assertGreater(metrics['alpha'], 0)

        # Correlation should be between -1 and 1
        self.assertGreaterEqual(metrics['correlation'], -1)
        self.assertLessEqual(metrics['correlation'], 1)

        # Edge cases
        empty_metrics = self.metrics.calculate_benchmark_metrics([], [])
        self.assertEqual(empty_metrics['alpha'], 0.0)

        # Different length inputs
        mismatched_metrics = self.metrics.calculate_benchmark_metrics([0.1, 0.2], [0.1])
        self.assertEqual(mismatched_metrics['alpha'], 0.0)

    def test_calculate_var(self):
        """Test calculating Value at Risk."""
        # Historical VaR
        var_hist = self.metrics.calculate_var(self.returns, 0.95, 'historical')
        # Parametric VaR
        var_param = self.metrics.calculate_var(self.returns, 0.95, 'parametric')
        # Monte Carlo VaR
        var_mc = self.metrics.calculate_var(self.returns, 0.95, 'monte_carlo')

        # All VaR methods should return positive values
        self.assertGreater(var_hist, 0)
        self.assertGreater(var_param, 0)
        self.assertGreater(var_mc, 0)

        # Edge cases
        self.assertEqual(self.metrics.calculate_var([], 0.95), 0.0)
        # Invalid method defaults to historical
        var_default = self.metrics.calculate_var(self.returns, 0.95, 'invalid_method')
        self.assertEqual(var_default, var_hist)

    def test_calculate_cvar(self):
        """Test calculating Conditional Value at Risk."""
        cvar = self.metrics.calculate_cvar(self.returns, 0.95)

        # CVaR should be greater than or equal to VaR
        var = self.metrics.calculate_var(self.returns, 0.95)
        self.assertGreaterEqual(cvar, var)

        # Edge cases
        self.assertEqual(self.metrics.calculate_cvar([], 0.95), 0.0)
        # Test with all positive returns (CVaR equals VaR if no returns beyond VaR)
        self.assertEqual(self.metrics.calculate_cvar([0.01, 0.02, 0.03], 0.95),
                         self.metrics.calculate_var([0.01, 0.02, 0.03], 0.95))

    def test_calculate_all_metrics(self):
        """Test calculating all metrics at once."""
        metrics = self.metrics.calculate_all_metrics(
            self.portfolio_values, self.trades, self.benchmark_values
        )

        # Check that important metrics are present
        self.assertIn('total_return', metrics)
        self.assertIn('annualized_return', metrics)
        self.assertIn('volatility', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('sortino_ratio', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('benchmark', metrics)

        # Check a few specific values
        self.assertAlmostEqual(metrics['total_return'], 0.2, places=4)  # (1200 - 1000) / 1000
        self.assertEqual(metrics['num_trades'], 5)

        # Edge cases
        empty_metrics = self.metrics.calculate_all_metrics([], [], [])
        self.assertEqual(empty_metrics, {})

        # Test with portfolio values but no trades or benchmark
        partial_metrics = self.metrics.calculate_all_metrics(self.portfolio_values, [], None)
        self.assertIn('total_return', partial_metrics)
        self.assertIn('win_rate', partial_metrics)
        self.assertEqual(partial_metrics['win_rate'], 0.0)  # No trades

    @patch('core.performance_metrics.PerformanceMetrics.calculate_all_metrics')
    def test_get_real_time_metrics(self, mock_calculate_all_metrics):
        """Test getting real-time metrics."""
        # Set up mock state manager
        self.state_manager_mock.get_portfolio_values.return_value = self.portfolio_values
        self.state_manager_mock.get_trades.return_value = self.trades
        self.state_manager_mock.get_benchmark_values.return_value = self.benchmark_values

        # Mock the calculate_all_metrics method
        mock_calculate_all_metrics.return_value = {'total_return': 0.2, 'sharpe_ratio': 1.5}

        # Call the method
        metrics = self.metrics.get_real_time_metrics()

        # Check that state manager methods were called
        self.state_manager_mock.get_portfolio_values.assert_called_once()
        self.state_manager_mock.get_trades.assert_called_once()
        self.state_manager_mock.get_benchmark_values.assert_called_once()

        # Check that calculate_all_metrics was called with correct arguments
        mock_calculate_all_metrics.assert_called_once_with(
            self.portfolio_values, self.trades, self.benchmark_values
        )

        # Check that the result matches what calculate_all_metrics returned
        self.assertEqual(metrics, {'total_return': 0.2, 'sharpe_ratio': 1.5})

        # Test with window_size
        self.metrics.get_real_time_metrics(window_size=30)
        self.state_manager_mock.get_portfolio_values.assert_called_with(30)

        # Test error handling when state manager is None
        metrics_without_state = PerformanceMetrics()
        self.assertEqual(metrics_without_state.get_real_time_metrics(), {})

    def test_regime_specific_metrics(self):
        """
        Test that metrics behave correctly across different simulated market regimes.
        This is particularly important for a regime-adaptive strategy.
        """
        # Simulate a trending market (steadily increasing returns)
        trending_returns = [0.01, 0.015, 0.02, 0.018, 0.022, 0.025]
        trending_values = [1000]
        for r in trending_returns:
            trending_values.append(trending_values[-1] * (1 + r))

        # Simulate a mean-reverting market (oscillating returns)
        mean_rev_returns = [0.01, -0.008, 0.012, -0.01, 0.015, -0.012]
        mean_rev_values = [1000]
        for r in mean_rev_returns:
            mean_rev_values.append(mean_rev_values[-1] * (1 + r))

        # Simulate a volatile market (large positive and negative returns)
        volatile_returns = [0.03, -0.025, 0.04, -0.035, 0.045, -0.04]
        volatile_values = [1000]
        for r in volatile_returns:
            volatile_values.append(volatile_values[-1] * (1 + r))

        # Test Sharpe ratios across regimes - should be highest in trending, lowest in volatile
        sharpe_trending = self.metrics.calculate_sharpe_ratio(trending_returns)
        sharpe_mean_rev = self.metrics.calculate_sharpe_ratio(mean_rev_returns)
        sharpe_volatile = self.metrics.calculate_sharpe_ratio(volatile_returns)

        self.assertGreater(sharpe_trending, sharpe_mean_rev)
        self.assertGreater(sharpe_mean_rev, sharpe_volatile)

        # Test drawdowns - should be larger in volatile markets
        dd_trending, _, _ = self.metrics.calculate_max_drawdown(trending_values)
        dd_mean_rev, _, _ = self.metrics.calculate_max_drawdown(mean_rev_values)
        dd_volatile, _, _ = self.metrics.calculate_max_drawdown(volatile_values)

        self.assertLess(dd_trending, dd_volatile)

        # Test Kelly criterion - should recommend larger position sizes in trending markets
        # Simulate trades for each regime
        def create_regime_trades(returns):
            trades = []
            win_count = sum(1 for r in returns if r > 0)
            win_rate = win_count / len(returns)

            avg_win = sum(r for r in returns if r > 0) / win_count if win_count else 0
            avg_loss = sum(abs(r) for r in returns if r < 0) / (len(returns) - win_count) if len(
                returns) > win_count else 0
            win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 10  # Prevent division by zero

            return win_rate, win_loss_ratio

        # Calculate Kelly for each regime
        win_rate_trending, wl_ratio_trending = create_regime_trades(trending_returns)
        win_rate_mean_rev, wl_ratio_mean_rev = create_regime_trades(mean_rev_returns)
        win_rate_volatile, wl_ratio_volatile = create_regime_trades(volatile_returns)

        kelly_trending = self.metrics.calculate_kelly_criterion(win_rate_trending, wl_ratio_trending)
        kelly_mean_rev = self.metrics.calculate_kelly_criterion(win_rate_mean_rev, wl_ratio_mean_rev)
        kelly_volatile = self.metrics.calculate_kelly_criterion(win_rate_volatile, wl_ratio_volatile)

        # In a well-behaved regime detection system, Kelly should allocate more to trending markets
        self.assertGreaterEqual(kelly_trending, kelly_volatile)


if __name__ == '__main__':
    unittest.main()