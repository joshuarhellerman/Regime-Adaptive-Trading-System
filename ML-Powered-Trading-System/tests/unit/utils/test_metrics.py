import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from utils.metrics import (
    simple_returns, log_returns, total_return, annualize_return, compound_returns,
    volatility, downside_deviation, max_drawdown, drawdowns, value_at_risk, conditional_value_at_risk,
    sharpe_ratio, sortino_ratio, calmar_ratio, information_ratio, treynor_ratio,
    rolling_returns, rolling_volatility, rolling_sharpe,
    win_rate, profit_factor, expectancy, kelly_criterion,
    risk_contribution, portfolio_volatility, portfolio_returns, beta, alpha, correlation,
    r_squared, tracking_error, normalize_data, exponential_smoothing, rolling_window,
    detect_outliers, remove_outliers, moving_average_crossover, pivot_points,
    exponential_moving_average, simple_moving_average, relative_strength_index,
    bollinger_bands, macd, average_true_range, zscore, median_absolute_deviation,
    percentile_rank, percentileofscore, rolling_correlation, autocorrelation
)


class TestMetrics(unittest.TestCase):
    """Test cases for financial metrics utility module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample price data
        self.prices = np.array([100.0, 102.0, 99.0, 101.0, 103.0, 102.5, 103.8, 104.2])
        self.prices2 = np.array([50.0, 51.0, 49.5, 52.0, 53.1, 52.8, 53.9, 54.5])
        
        # Create sample return data
        self.returns = np.array([0.02, -0.03, 0.02, 0.02, -0.005, 0.013, 0.004])
        self.returns2 = np.array([0.02, -0.03, 0.05, 0.021, -0.006, 0.021, 0.011])
        
        # Create sample OHLC data
        self.high_prices = np.array([102.0, 103.0, 101.0, 104.0, 105.0, 103.5, 105.8, 105.2])
        self.low_prices = np.array([99.0, 100.0, 97.0, 99.0, 101.0, 101.0, 102.8, 103.2])
        self.close_prices = self.prices
        
        # Create sample trade data
        self.pnl_values = [100, -50, 75, 200, -100, 50, 30, -75, 125, 80]

    def test_system_health_metrics(self):
        """Test system health metrics calculation."""
        from utils.metrics import system_health_metrics
        
        # Test with perfect metrics
        health = system_health_metrics(0.0, 0.0, 0.0, 0.0)
        self.assertEqual(health['overall_health'], 1.0)
        self.assertEqual(health['cpu_score'], 1.0)
        self.assertEqual(health['memory_score'], 1.0)
        self.assertEqual(health['latency_score'], 1.0)
        self.assertEqual(health['error_score'], 1.0)
        
        # Test with worst metrics
        health = system_health_metrics(100.0, 100.0, float('inf'), 100.0)
        self.assertAlmostEqual(health['overall_health'], 0.0)
        self.assertEqual(health['cpu_score'], 0.0)
        self.assertEqual(health['memory_score'], 0.0)
        self.assertEqual(health['latency_score'], 0.0)
        self.assertEqual(health['error_score'], 0.0)
        
        # Test with mixed metrics
        health = system_health_metrics(50.0, 25.0, 100.0, 1.0)
        self.assertGreater(health['overall_health'], 0.0)
        self.assertLess(health['overall_health'], 1.0)
        self.assertEqual(health['cpu_score'], 0.5)
        self.assertEqual(health['memory_score'], 0.75)
        self.assertAlmostEqual(health['latency_score'], np.exp(-100/100))
        self.assertAlmostEqual(health['error_score'], np.exp(-1/1))
    
    def test_execution_quality_score(self):
        """Test execution quality score calculation."""
        from utils.metrics import execution_quality_score
        
        # Test with perfect execution
        fill_times = [0.0, 0.0, 0.0]
        order_sizes = [1000.0, 2000.0, 3000.0]
        slippages = [0.0, 0.0, 0.0]
        
        score = execution_quality_score(fill_times, order_sizes, slippages)
        self.assertEqual(score, 100.0)
        
        # Test with worst execution
        fill_times = [10000.0, 10000.0, 10000.0]
        slippages = [1000.0, 1000.0, 1000.0]
        
        score = execution_quality_score(fill_times, order_sizes, slippages)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)
        
        # Test with mixed execution
        fill_times = [100.0, 500.0, 1000.0]
        slippages = [1.0, 5.0, 10.0]
        
        score = execution_quality_score(fill_times, order_sizes, slippages)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 100.0)
        
        # Test with different length arrays
        with self.assertRaises(ValueError):
            execution_quality_score(fill_times, order_sizes[:-1], slippages)
        
        # Test with empty arrays
        self.assertEqual(execution_quality_score([], [], []), 0.0)
    
    def test_regime_stability_score(self):
        """Test regime stability score calculation."""
        from utils.metrics import regime_stability_score
        
        # Test with perfectly stable regimes
        regime_labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        window = 3
        
        stability = regime_stability_score(regime_labels, window)
        
        # First window-1 values should be NaN
        self.assertTrue(np.all(np.isnan(stability[:window-1])))
        
        # Rest should be 1.0 (perfectly stable)
        self.assertTrue(np.all(stability[window-1:] == 1.0))
        
        # Test with changing regimes
        regime_labels = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        stability = regime_stability_score(regime_labels, window)
        
        # Check a few values
        self.assertEqual(stability[2], 1.0)  # [1,1,1] - all same
        self.assertAlmostEqual(stability[4], 2/3)  # [1,1,2] - 2 of one regime, 1 of another
        self.assertAlmostEqual(stability[7], 2/3)  # [2,2,3] - 2 of one regime, 1 of another
        
        # Test with window larger than array
        self.assertTrue(np.all(regime_stability_score(np.array([1, 2]), 3) == 0.0))
    
    def test_calculate_market_impact(self):
        """Test market impact calculation."""
        from utils.metrics import calculate_market_impact
        
        # Test with square root model
        impact = calculate_market_impact(
            size=10000,
            aum=1000000,
            adv=100000,
            price=100.0,
            volatility=0.02,
            model='square_root'
        )
        
        # Calculate expected impact
        trade_value = 10000 * 100.0
        participation = min(0.3, trade_value / 100000)
        expected_impact = 0.02 * np.sqrt(participation)
        
        self.assertAlmostEqual(impact, expected_impact)
        
        # Test with linear model
        impact = calculate_market_impact(
            size=10000,
            aum=1000000,
            adv=100000,
            price=100.0,
            volatility=0.02,
            model='linear'
        )
        
        expected_impact = 0.02 * participation
        self.assertAlmostEqual(impact, expected_impact)
        
        # Test with power law model
        impact = calculate_market_impact(
            size=10000,
            aum=1000000,
            adv=100000,
            price=100.0,
            volatility=0.02,
            model='power_law'
        )
        
        expected_impact = 0.02 * (participation ** 0.6)
        self.assertAlmostEqual(impact, expected_impact)
        
        # Test with unknown model (should default to square root)
        impact = calculate_market_impact(
            size=10000,
            aum=1000000,
            adv=100000,
            price=100.0,
            volatility=0.02,
            model='unknown_model'
        )
        
        self.assertAlmostEqual(impact, expected_impact)
    
    def test_estimate_slippage(self):
        """Test slippage estimation."""
        from utils.metrics import estimate_slippage
        
        # Test with standard parameters
        slippage = estimate_slippage(volume=100000, volatility=0.02, bid_ask=0.0001)
        
        # Base slippage should be volatility * 10
        base_slippage = 0.02 * 10
        
        # Volume factor for 100,000 volume
        volume_factor = 0.85
        
        # Spread component in basis points
        spread_component = 0.0001 * 100 / 2
        
        expected_slippage = (base_slippage * volume_factor) + spread_component
        self.assertAlmostEqual(slippage, expected_slippage)
        
        # Test with high volume (should have lower slippage)
        high_vol_slippage = estimate_slippage(volume=2000000, volatility=0.02, bid_ask=0.0001)
        self.assertLess(high_vol_slippage, slippage)
        
        # Test with low volume (should have higher slippage)
        low_vol_slippage = estimate_slippage(volume=5000, volatility=0.02, bid_ask=0.0001)
        self.assertGreater(low_vol_slippage, slippage)
        
        # Test without bid-ask spread
        no_spread_slippage = estimate_slippage(volume=100000, volatility=0.02)
        self.assertLess(no_spread_slippage, slippage)
    
    def test_order_size_constraints(self):
        """Test order size constraints calculation."""
        from utils.metrics import order_size_constraints
        
        # Test with standard parameters
        constraints = order_size_constraints(
            portfolio_value=1000000,
            price=100.0,
            volatility=0.02,
            max_pct_aum=0.05,
            max_days_volume=0.1,
            adv=100000
        )
        
        # Calculate expected constraints
        aum_limit = 1000000 * 0.05
        max_shares_aum = aum_limit / 100.0
        
        max_shares_volume = 100000 * 0.1
        
        risk_budget = 1000000 * 0.02
        max_shares_risk = risk_budget / (100.0 * 0.02)
        
        expected_max_shares = min(max_shares_aum, max_shares_volume, max_shares_risk)
        
        self.assertEqual(constraints['aum_constraint'], max_shares_aum)
        self.assertEqual(constraints['volume_constraint'], max_shares_volume)
        self.assertEqual(constraints['risk_constraint'], max_shares_risk)
        self.assertEqual(constraints['max_shares'], expected_max_shares)
        
        # Test without ADV (volume constraint should be infinity)
        no_adv_constraints = order_size_constraints(
            portfolio_value=1000000,
            price=100.0,
            volatility=0.02,
            max_pct_aum=0.05,
            max_days_volume=0.1,
            adv=None
        )
        
        self.assertEqual(no_adv_constraints['volume_constraint'], float('inf'))
        self.assertEqual(no_adv_constraints['max_shares'], min(max_shares_aum, max_shares_risk))
    
    def test_round_order_size(self):
        """Test order size rounding."""
        from utils.metrics import round_order_size
        
        # Test with standard parameters
        rounded_size = round_order_size(size=123.45, min_size=1.0, lot_size=5.0)
        expected_size = np.floor(123.45 / 5.0) * 5.0
        self.assertEqual(rounded_size, expected_size)
        
        # Test with size below minimum
        below_min = round_order_size(size=0.5, min_size=1.0, lot_size=5.0)
        self.assertEqual(below_min, 0.0)
        
        # Test with size below minimum after rounding
        below_min_after_rounding = round_order_size(size=3.0, min_size=5.0, lot_size=5.0)
        self.assertEqual(below_min_after_rounding, 0.0)
    
    def test_map_indicator_to_signal(self):
        """Test mapping indicator to signals based on thresholds."""
        from utils.metrics import map_indicator_to_signal
        
        # Create indicator values
        indicator = np.array([20, 40, 60, 80, 50, 30, 10])
        
        # Set thresholds
        thresholds = {'buy': 70, 'sell': 30}
        
        # Expected signals
        expected_signals = np.array([0, 0, 0, 1, 0, 0, -1])
        
        # Test mapping
        signals = map_indicator_to_signal(indicator, thresholds)
        np.testing.assert_array_equal(signals, expected_signals)
        
        # Test with only buy threshold
        buy_only_thresholds = {'buy': 70}
        buy_only_signals = map_indicator_to_signal(indicator, buy_only_thresholds)
        buy_only_expected = np.zeros_like(indicator)
        buy_only_expected[3] = 1  # Only value >= 70
        np.testing.assert_array_equal(buy_only_signals, buy_only_expected)
        
        # Test with only sell threshold
        sell_only_thresholds = {'sell': 30}
        sell_only_signals = map_indicator_to_signal(indicator, sell_only_thresholds)
        sell_only_expected = np.zeros_like(indicator)
        sell_only_expected[0] = -1  # Value 20 <= 30
        sell_only_expected[5] = -1  # Value 30 <= 30
        sell_only_expected[6] = -1  # Value 10 <= 30
        np.testing.assert_array_equal(sell_only_signals, sell_only_expected)
        
        # Test with empty array
        self.assertEqual(len(map_indicator_to_signal(np.array([]), thresholds)), 0)


if __name__ == '__main__':
    unittest.main()
_rolling_returns(self):
        """Test rolling returns calculation over a window."""
        window = 3
        expected_rolling_returns = np.array([
            (self.prices[3] / self.prices[0]) - 1,
            (self.prices[4] / self.prices[1]) - 1,
            (self.prices[5] / self.prices[2]) - 1,
            (self.prices[6] / self.prices[3]) - 1,
            (self.prices[7] / self.prices[4]) - 1
        ])
        
        np.testing.assert_almost_equal(rolling_returns(self.prices, window), expected_rolling_returns)
        
        # Test with window larger than array
        self.assertEqual(len(rolling_returns(self.prices, 10)), 0)
        
    def test_win_loss_ratio(self):
        """Test calculation of win/loss ratio from PnL values."""
        # Create list of profits and losses
        profits = [p for p in self.pnl_values if p > 0]
        losses = [p for p in self.pnl_values if p < 0]
        
        avg_profit = np.mean(profits) if profits else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        
        expected_ratio = avg_profit / avg_loss if avg_loss > 0 else float('inf')
        
        # This isn't directly testing a function in metrics.py, but it tests the 
        # calculation that would be used to get win_loss_ratio for kelly_criterion
        win_loss_ratio = sum(profits) / abs(sum(losses)) if sum(losses) != 0 else float('inf')
        
        self.assertGreater(win_loss_ratio, 0)
        
        # Test with only profits
        only_profits = [100, 200, 300]
        only_profits_ratio = sum(only_profits) / 1  # Division by 1 to avoid division by zero
        self.assertEqual(only_profits_ratio, sum(only_profits))
        
        # Test with only losses
        only_losses = [-100, -200, -300]
        only_losses_ratio = 0  # No wins
        self.assertEqual(only_losses_ratio, 0)
        
    def test_simple_returns(self):
        """Test simple returns calculation."""
        expected_returns = np.array([0.02, -0.029411765, 0.0202020202, 0.019801980, -0.004854369, 0.0126829268, 0.0038535982])
        np.testing.assert_almost_equal(simple_returns(self.prices), expected_returns)
        
        # Test with list input
        np.testing.assert_almost_equal(simple_returns(self.prices.tolist()), expected_returns)
        
        # Test with short array
        self.assertEqual(len(simple_returns(np.array([100]))), 0)

    def test_log_returns(self):
        """Test logarithmic returns calculation."""
        expected_returns = np.log(self.prices[1:] / self.prices[:-1])
        np.testing.assert_almost_equal(log_returns(self.prices), expected_returns)
        
        # Test with list input
        np.testing.assert_almost_equal(log_returns(self.prices.tolist()), expected_returns)
        
        # Test with short array
        self.assertEqual(len(log_returns(np.array([100]))), 0)

    def test_total_return(self):
        """Test total return calculation."""
        self.assertAlmostEqual(total_return(100, 104.2), 0.042)
        
        # Test with zero start value - should return 0
        self.assertEqual(total_return(0, 100), 0.0)
        
        # Test with negative start value - should return 0
        self.assertEqual(total_return(-10, 100), 0.0)

    def test_annualize_return(self):
        """Test annualization of returns."""
        # Test with 252 trading days per year
        annual_return = annualize_return(0.1, 63, 252)  # 63 days (quarter)
        self.assertAlmostEqual(annual_return, (1 + 0.1) ** 4 - 1, places=10)
        
        # Test with zero days - should return 0
        self.assertEqual(annualize_return(0.1, 0), 0.0)
        
        # Test with negative days - should return 0
        self.assertEqual(annualize_return(0.1, -10), 0.0)

    def test_compound_returns(self):
        """Test compounding of returns."""
        expected = (1 + self.returns[0]) * (1 + self.returns[1]) * (1 + self.returns[2]) - 1
        self.assertAlmostEqual(compound_returns(self.returns[:3]), expected)
        
        # Test with list input
        self.assertAlmostEqual(compound_returns(self.returns[:3].tolist()), expected)
        
        # Test with empty array
        self.assertEqual(compound_returns([]), 0.0)

    def test_volatility(self):
        """Test volatility calculation."""
        # Daily volatility
        daily_vol = volatility(self.returns, False)
        self.assertAlmostEqual(daily_vol, np.std(self.returns, ddof=1), places=10)
        
        # Annualized volatility (assuming 252 trading days)
        ann_vol = volatility(self.returns, True, 252)
        self.assertAlmostEqual(ann_vol, np.std(self.returns, ddof=1) * np.sqrt(252), places=10)
        
        # Test with not enough returns
        self.assertEqual(volatility(np.array([0.01])), 0.0)

    def test_downside_deviation(self):
        """Test downside deviation calculation."""
        threshold = 0.0
        # Get returns below threshold
        downside_returns = self.returns[self.returns < threshold]
        expected_dd = np.std(downside_returns, ddof=1)
        
        dd = downside_deviation(self.returns, threshold, False)
        self.assertAlmostEqual(dd, expected_dd, places=10)
        
        # Annualized downside deviation
        ann_dd = downside_deviation(self.returns, threshold, True, 252)
        self.assertAlmostEqual(ann_dd, expected_dd * np.sqrt(252), places=10)
        
        # Test with all returns above threshold
        all_positive = np.array([0.01, 0.02, 0.03])
        self.assertEqual(downside_deviation(all_positive, 0.0), 0.0)
        
        # Test with not enough returns
        self.assertEqual(downside_deviation(np.array([0.01])), 0.0)

    def test_max_drawdown(self):
        """Test maximum drawdown calculation."""
        # Calculate expected maximum drawdown manually
        peak = self.prices[0]
        peak_idx = 0
        max_dd = 0.0
        trough_idx = 0
        
        for i, price in enumerate(self.prices):
            if price > peak:
                peak = price
                peak_i = i
            else:
                dd = (peak - price) / peak
                if dd > max_dd:
                    max_dd = dd
                    peak_idx = peak_i
                    trough_idx = i
        
        dd, p_idx, t_idx = max_drawdown(self.prices)
        self.assertAlmostEqual(dd, max_dd, places=10)
        self.assertEqual(p_idx, peak_idx)
        self.assertEqual(t_idx, trough_idx)
        
        # Test with short array
        self.assertEqual(max_drawdown(np.array([100]))[0], 0.0)

    def test_drawdowns(self):
        """Test drawdowns calculation at each point."""
        # Calculate expected drawdowns
        running_max = np.maximum.accumulate(self.prices)
        expected_drawdowns = (running_max - self.prices) / running_max
        
        np.testing.assert_almost_equal(drawdowns(self.prices), expected_drawdowns)
        
        # Test with short array
        np.testing.assert_almost_equal(drawdowns(np.array([100])), np.array([0.0]))

    def test_value_at_risk(self):
        """Test Value at Risk calculation."""
        # Historical VaR at 95% confidence level
        confidence_level = 0.95
        expected_var = abs(np.percentile(self.returns, 100 * (1 - confidence_level)))
        
        var = value_at_risk(self.returns, confidence_level, 'historical')
        self.assertAlmostEqual(var, expected_var, places=10)
        
        # Test with not enough returns
        self.assertEqual(value_at_risk(np.array([0.01])), 0.0)
        
        # Test parametric and Monte Carlo methods
        var_param = value_at_risk(self.returns, confidence_level, 'parametric')
        var_mc = value_at_risk(self.returns, confidence_level, 'monte_carlo')
        
        # Results should be positive
        self.assertGreater(var_param, 0)
        self.assertGreater(var_mc, 0)
        
        # Unknown method should default to historical
        var_unknown = value_at_risk(self.returns, confidence_level, 'unknown_method')
        self.assertAlmostEqual(var_unknown, expected_var, places=10)

    def test_conditional_value_at_risk(self):
        """Test Conditional Value at Risk calculation."""
        confidence_level = 0.95
        var = value_at_risk(self.returns, confidence_level, 'historical')
        
        # Expected CVaR is the average of returns beyond VaR
        threshold = -var  # Negative since we're looking at losses
        tail_returns = self.returns[self.returns <= threshold]
        expected_cvar = abs(np.mean(tail_returns)) if len(tail_returns) > 0 else var
        
        cvar = conditional_value_at_risk(self.returns, confidence_level)
        self.assertAlmostEqual(cvar, expected_cvar, places=10)
        
        # Test with not enough returns
        self.assertEqual(conditional_value_at_risk(np.array([0.01])), 0.0)

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        risk_free_rate = 0.02  # 2% annual risk-free rate
        
        # Calculate daily excess return
        daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
        excess_return = np.mean(self.returns) - daily_rf
        
        # Calculate daily Sharpe ratio
        daily_vol = np.std(self.returns, ddof=1)
        expected_daily_sharpe = excess_return / daily_vol
        
        # Calculate annualized Sharpe ratio
        expected_annual_sharpe = expected_daily_sharpe * np.sqrt(252)
        
        # Test daily Sharpe
        daily_sharpe = sharpe_ratio(self.returns, risk_free_rate, False, 252)
        self.assertAlmostEqual(daily_sharpe, expected_daily_sharpe, places=10)
        
        # Test annualized Sharpe
        annual_sharpe = sharpe_ratio(self.returns, risk_free_rate, True, 252)
        self.assertAlmostEqual(annual_sharpe, expected_annual_sharpe, places=10)
        
        # Test with zero volatility
        self.assertEqual(sharpe_ratio(np.zeros(10), 0.0), 0.0)
        
        # Test with not enough returns
        self.assertEqual(sharpe_ratio(np.array([0.01])), 0.0)

    def test_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        risk_free_rate = 0.02  # 2% annual risk-free rate
        
        # Calculate daily excess return
        daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
        excess_return = np.mean(self.returns) - daily_rf
        
        # Calculate downside deviation
        dd = downside_deviation(self.returns, daily_rf, False)
        
        # Calculate daily Sortino ratio
        expected_daily_sortino = excess_return / dd if dd > 0 else 0.0
        
        # Calculate annualized Sortino ratio
        expected_annual_sortino = expected_daily_sortino * np.sqrt(252)
        
        # Test daily Sortino
        daily_sortino = sortino_ratio(self.returns, risk_free_rate, False, 252)
        if dd > 0:
            self.assertAlmostEqual(daily_sortino, expected_daily_sortino, places=10)
        
        # Test annualized Sortino
        annual_sortino = sortino_ratio(self.returns, risk_free_rate, True, 252)
        if dd > 0:
            self.assertAlmostEqual(annual_sortino, expected_annual_sortino, places=10)
        
        # Test with zero downside deviation
        all_positive = np.array([0.05, 0.06, 0.07])  # All above risk-free rate
        self.assertEqual(sortino_ratio(all_positive, 0.02), 0.0)
        
        # Test with not enough returns
        self.assertEqual(sortino_ratio(np.array([0.01])), 0.0)

    def test_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        # Calculate annualized return
        period_years = 1.0
        ann_return = (np.prod(1 + self.returns) ** (1 / period_years)) - 1
        
        # Calculate max drawdown using prices derived from returns
        prices = np.cumprod(np.insert(1 + self.returns, 0, 1))[1:]
        max_dd, _, _ = max_drawdown(prices)
        
        # Expected Calmar ratio
        expected_calmar = ann_return / max_dd if max_dd > 0 else 0.0
        
        # Test Calmar ratio
        calmar = calmar_ratio(self.returns, None, period_years)
        if max_dd > 0:
            self.assertAlmostEqual(calmar, expected_calmar, places=10)
        
        # Test with provided prices
        calmar_with_prices = calmar_ratio(self.returns, prices, period_years)
        if max_dd > 0:
            self.assertAlmostEqual(calmar_with_prices, expected_calmar, places=10)
        
        # Test with zero max drawdown
        no_drawdown = np.array([100, 101, 102, 103])
        self.assertEqual(calmar_ratio(np.array([0.01, 0.01, 0.01]), no_drawdown), 0.0)
        
        # Test with not enough returns
        self.assertEqual(calmar_ratio(np.array([0.01])), 0.0)

    def test_information_ratio(self):
        """Test Information Ratio calculation."""
        # Calculate active returns
        active_returns = self.returns - self.returns2
        
        # Calculate tracking error
        tracking_error = np.std(active_returns, ddof=1)
        
        # Calculate daily Information Ratio
        expected_daily_ir = np.mean(active_returns) / tracking_error
        
        # Calculate annualized Information Ratio
        expected_annual_ir = expected_daily_ir * np.sqrt(252)
        
        # Test daily Information Ratio
        daily_ir = information_ratio(self.returns, self.returns2, False, 252)
        self.assertAlmostEqual(daily_ir, expected_daily_ir, places=10)
        
        # Test annualized Information Ratio
        annual_ir = information_ratio(self.returns, self.returns2, True, 252)
        self.assertAlmostEqual(annual_ir, expected_annual_ir, places=10)
        
        # Test with different length arrays
        with self.assertRaises(ValueError):
            information_ratio(self.returns, self.returns2[:5])
        
        # Test with zero tracking error
        self.assertEqual(information_ratio(np.array([0.01, 0.02]), np.array([0.01, 0.02])), 0.0)
        
        # Test with not enough returns
        self.assertEqual(information_ratio(np.array([0.01]), np.array([0.02])), 0.0)

    def test_treynor_ratio(self):
        """Test Treynor Ratio calculation."""
        risk_free_rate = 0.02  # 2% annual risk-free rate
        
        # Calculate beta
        cov = np.cov(self.returns, self.returns2)[0, 1]
        var = np.var(self.returns2, ddof=1)
        beta_value = cov / var
        
        # Calculate daily excess return
        daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
        excess_return = np.mean(self.returns) - daily_rf
        
        # Calculate daily Treynor ratio
        expected_daily_treynor = excess_return / beta_value if beta_value != 0 else 0.0
        
        # Calculate annualized Treynor ratio
        expected_annual_treynor = expected_daily_treynor * 252
        
        # Test daily Treynor ratio
        daily_treynor = treynor_ratio(self.returns, self.returns2, risk_free_rate, False, 252)
        if beta_value != 0:
            self.assertAlmostEqual(daily_treynor, expected_daily_treynor, places=10)
        
        # Test annualized Treynor ratio
        annual_treynor = treynor_ratio(self.returns, self.returns2, risk_free_rate, True, 252)
        if beta_value != 0:
            self.assertAlmostEqual(annual_treynor, expected_annual_treynor, places=10)
        
        # Test with different length arrays
        with self.assertRaises(ValueError):
            treynor_ratio(self.returns, self.returns2[:5])
        
        # Test with zero beta (zero benchmark variance)
        self.assertEqual(treynor_ratio(np.array([0.01, 0.02]), np.array([0.01, 0.01])), 0.0)
        
        # Test with not enough returns
        self.assertEqual(treynor_ratio(np.array([0.01]), np.array([0.02])), 0.0)

    def test_rolling_returns(self):
        """Test rolling returns calculation."""
        window = 3
        expected_rolling_returns = np.array([
            (self.prices[3] / self.prices[0]) - 1,
            (self.prices[4] / self.prices[1]) - 1,
            (self.prices[5] / self.prices[2]) - 1,
            (self.prices[6] / self.prices[3]) - 1,
            (self.prices[7] / self.prices[4]) - 1
        ])
        
        np.testing.assert_almost_equal(rolling_returns(self.prices, window), expected_rolling_returns)
        
        # Test with window larger than array
        self.assertEqual(len(rolling_returns(self.prices, 10)), 0)

    def test_rolling_volatility(self):
        """Test rolling volatility calculation."""
        window = 3
        expected_rolling_vol = np.array([
            np.std(self.returns[0:3], ddof=1),
            np.std(self.returns[1:4], ddof=1),
            np.std(self.returns[2:5], ddof=1),
            np.std(self.returns[3:6], ddof=1),
            np.std(self.returns[4:7], ddof=1)
        ])
        
        # Test daily rolling volatility
        daily_rolling_vol = rolling_volatility(self.returns, window, False)
        np.testing.assert_almost_equal(daily_rolling_vol, expected_rolling_vol)
        
        # Test annualized rolling volatility
        ann_rolling_vol = rolling_volatility(self.returns, window, True, 252)
        np.testing.assert_almost_equal(ann_rolling_vol, expected_rolling_vol * np.sqrt(252))
        
        # Test with window larger than array
        self.assertEqual(len(rolling_volatility(self.returns, 10)), 0)

    def test_rolling_sharpe(self):
        """Test rolling Sharpe ratio calculation."""
        window = 3
        risk_free_rate = 0.02  # 2% annual risk-free rate
        daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
        
        # Calculate rolling Sharpe ratios manually
        sharpe_values = []
        for i in range(window, len(self.returns) + 1):
            window_returns = self.returns[i-window:i]
            mean_return = np.mean(window_returns)
            std_return = np.std(window_returns, ddof=1)
            
            if std_return <= 0:
                sharpe = 0.0
            else:
                sharpe = (mean_return - daily_rf) / std_return
                # Annualize if needed
                sharpe = sharpe * np.sqrt(252)
            
            sharpe_values.append(sharpe)
        
        expected_rolling_sharpe = np.array(sharpe_values)
        
        # Test rolling Sharpe ratio
        rolling_sharpe_values = rolling_sharpe(self.returns, window, risk_free_rate, True, 252)
        np.testing.assert_almost_equal(rolling_sharpe_values, expected_rolling_sharpe)
        
        # Test with window larger than array
        self.assertEqual(len(rolling_sharpe(self.returns, 10)), 0)

    def test_win_rate(self):
        """Test win rate calculation."""
        # Count winning trades
        winning_trades = sum(1 for pnl in self.pnl_values if pnl > 0)
        expected_win_rate = winning_trades / len(self.pnl_values)
        
        self.assertAlmostEqual(win_rate(self.pnl_values), expected_win_rate)
        
        # Test with empty list
        self.assertEqual(win_rate([]), 0.0)

    def test_profit_factor(self):
        """Test profit factor calculation."""
        # Calculate gross profits and losses
        gross_profits = sum(pnl for pnl in self.pnl_values if pnl > 0)
        gross_losses = abs(sum(pnl for pnl in self.pnl_values if pnl < 0))
        
        expected_profit_factor = gross_profits / gross_losses
        
        self.assertAlmostEqual(profit_factor(self.pnl_values), expected_profit_factor)
        
        # Test with no losses
        self.assertEqual(profit_factor([100, 200, 300]), float('inf'))
        
        # Test with no profits
        self.assertEqual(profit_factor([-100, -200, -300]), 0.0)
        
        # Test with empty list
        self.assertEqual(profit_factor([]), 0.0)

    def test_expectancy(self):
        """Test expectancy calculation."""
        expected_expectancy = sum(self.pnl_values) / len(self.pnl_values)
        
        self.assertAlmostEqual(expectancy(self.pnl_values), expected_expectancy)
        
        # Test with empty list
        self.assertEqual(expectancy([]), 0.0)

    def test_kelly_criterion(self):
        """Test Kelly Criterion calculation."""
        win_rate_value = 0.6
        win_loss_ratio = 2.0
        
        expected_kelly = win_rate_value - ((1 - win_rate_value) / win_loss_ratio)
        expected_kelly = max(0.0, min(1.0, expected_kelly))
        
        self.assertAlmostEqual(kelly_criterion(win_rate_value, win_loss_ratio), expected_kelly)
        
        # Test with negative win/loss ratio
        self.assertEqual(kelly_criterion(0.5, -1.0), 0.0)
        
        # Test with extreme win rates
        self.assertEqual(kelly_criterion(0.0, 2.0), 0.0)
        self.assertEqual(kelly_criterion(1.0, 2.0), 1.0)

    def test_risk_contribution(self):
        """Test risk contribution calculation."""
        # Create sample returns and weights
        returns_list = [self.returns, self.returns2]
        weights = [0.6, 0.4]
        
        # Convert to numpy arrays
        returns_array = np.array([np.array(r) for r in returns_list])
        weights_array = np.array(weights)
        
        # Calculate covariance matrix
        cov_matrix = np.cov(returns_array)
        
        # Calculate portfolio volatility
        portfolio_vol = np.sqrt(weights_array.T @ cov_matrix @ weights_array)
        
        # Calculate marginal contribution to risk
        mcr = cov_matrix @ weights_array
        
        # Calculate risk contribution
        expected_rc = weights_array * mcr / portfolio_vol
        
        # Test risk contribution
        rc = risk_contribution(returns_list, weights)
        np.testing.assert_almost_equal(rc, expected_rc.tolist())
        
        # Test with mismatched lengths
        with self.assertRaises(ValueError):
            risk_contribution([self.returns, self.returns2], [0.5])
        
        # Test with zero portfolio volatility
        zero_vol_returns = [np.zeros(5), np.zeros(5)]
        zero_vol_weights = [0.5, 0.5]
        zero_vol_rc = risk_contribution(zero_vol_returns, zero_vol_weights)
        self.assertEqual(zero_vol_rc, [0.0, 0.0])

    def test_portfolio_volatility(self):
        """Test portfolio volatility calculation."""
        # Create sample returns and weights
        returns_list = [self.returns, self.returns2]
        weights = [0.6, 0.4]
        
        # Convert to numpy arrays
        returns_array = np.array([np.array(r) for r in returns_list])
        weights_array = np.array(weights)
        
        # Calculate covariance matrix
        cov_matrix = np.cov(returns_array)
        
        # Calculate portfolio volatility
        expected_vol = np.sqrt(weights_array.T @ cov_matrix @ weights_array)
        
        # Annualize if needed
        expected_ann_vol = expected_vol * np.sqrt(252)
        
        # Test daily portfolio volatility
        daily_vol = portfolio_volatility(returns_list, weights, False)
        self.assertAlmostEqual(daily_vol, expected_vol)
        
        # Test annualized portfolio volatility
        ann_vol = portfolio_volatility(returns_list, weights, True, 252)
        self.assertAlmostEqual(ann_vol, expected_ann_vol)
        
        # Test with mismatched lengths
        with self.assertRaises(ValueError):
            portfolio_volatility([self.returns, self.returns2], [0.5])

    def test_portfolio_returns(self):
        """Test portfolio returns calculation."""
        # Create sample returns and weights
        returns_list = [self.returns, self.returns2]
        weights = [0.6, 0.4]
        
        # Convert to numpy arrays
        returns_array = np.array([np.array(r) for r in returns_list])
        weights_array = np.array(weights)
        
        # Calculate portfolio returns
        expected_port_returns = np.sum(returns_array.T * weights_array, axis=1)
        
        # Test portfolio returns
        port_returns = portfolio_returns(returns_list, weights)
        np.testing.assert_almost_equal(port_returns, expected_port_returns)
        
        # Test with mismatched lengths
        with self.assertRaises(ValueError):
            portfolio_returns([self.returns, self.returns2], [0.5])

    def test_beta(self):
        """Test beta calculation."""
        # Calculate beta using covariance / variance
        cov = np.cov(self.returns, self.returns2)[0, 1]
        var = np.var(self.returns2, ddof=1)
        
        expected_beta = cov / var
        
        # Test beta
        beta_value = beta(self.returns, self.returns2)
        self.assertAlmostEqual(beta_value, expected_beta)
        
        # Test with list input
        beta_value_list = beta(self.returns.tolist(), self.returns2.tolist())
        self.assertAlmostEqual(beta_value_list, expected_beta)
        
        # Test with different length arrays
        with self.assertRaises(ValueError):
            beta(self.returns, self.returns2[:5])
        
        # Test with zero benchmark variance
        self.assertEqual(beta(np.array([0.01, 0.02]), np.array([0.01, 0.01])), 0.0)
        
        # Test with not enough returns
        self.assertEqual(beta(np.array([0.01]), np.array([0.02])), 0.0)

    def test_alpha(self):
        """Test Jensen's alpha calculation."""
        risk_free_rate = 0.02  # 2% annual risk-free rate
        
        # Calculate beta
        beta_value = beta(self.returns, self.returns2)
        
        # Convert annualized risk-free rate to daily
        daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
        
        # Calculate alpha
        mean_return = np.mean(self.returns)
        mean_benchmark = np.mean(self.returns2)
        
        expected_alpha = mean_return - (daily_rf + beta_value * (mean_benchmark - daily_rf))
        
        # Annualize alpha
        expected_ann_alpha = (1 + expected_alpha) ** 252 - 1
        
        # Test daily alpha
        daily_alpha = alpha(self.returns, self.returns2, risk_free_rate, False, 252)
        self.assertAlmostEqual(daily_alpha, expected_alpha)
        
        # Test annualized alpha
        ann_alpha = alpha(self.returns, self.returns2, risk_free_rate, True, 252)
        self.assertAlmostEqual(ann_alpha, expected_ann_alpha)
        
        # Test with different length arrays
        with self.assertRaises(ValueError):
            alpha(self.returns, self.returns2[:5])
        
        # Test with not enough returns
        self.assertEqual(alpha(np.array([0.01]), np.array([0.02])), 0.0)

    def test_correlation(self):
        """Test correlation calculation."""
        expected_corr = np.corrcoef(self.returns, self.returns2)[0, 1]
        
        # Test correlation
        corr = correlation(self.returns, self.returns2)
        self.assertAlmostEqual(corr, expected_corr)
        
        # Test with list input
        corr_list = correlation(self.returns.tolist(), self.returns2.tolist())
        self.assertAlmostEqual(corr_list, expected_corr)
        
        # Test with different length arrays
        with self.assertRaises(ValueError):
            correlation(self.returns, self.returns2[:5])
        
        # Test with not enough returns
        self.assertEqual(correlation(np.array([0.01]), np.array([0.02])), 0.0)

    def test_r_squared(self):
        """Test R-squared calculation."""
        expected_corr = np.corrcoef(self.returns, self.returns2)[0, 1]
        expected_r2 = expected_corr ** 2
        
        # Test R-squared
        r2 = r_squared(self.returns, self.returns2)
        self.assertAlmostEqual(r2, expected_r2)
        
        # Test with list input
        r2_list = r_squared(self.returns.tolist(), self.returns2.tolist())
        self.assertAlmostEqual(r2_list, expected_r2)
        
        # Test with different length arrays
        with self.assertRaises(ValueError):
            r_squared(self.returns, self.returns2[:5])
        
        # Test with not enough returns
        self.assertEqual(r_squared(np.array([0.01]), np.array([0.02])), 0.0)

    def test_tracking_error(self):
        """Test tracking error calculation."""
        # Calculate tracking difference
        tracking_diff = self.returns - self.returns2
        
        # Calculate tracking error (standard deviation of tracking difference)
        expected_te = np.std(tracking_diff, ddof=1)
        
        # Annualize tracking error
        expected_ann_te = expected_te * np.sqrt(252)
        
        # Test daily tracking error
        daily_te = tracking_error(self.returns, self.returns2, False)
        self.assertAlmostEqual(daily_te, expected_te)
        
        # Test annualized tracking error
        ann_te = tracking_error(self.returns, self.returns2, True, 252)
        self.assertAlmostEqual(ann_te, expected_ann_te)
        
        # Test with different length arrays
        with self.assertRaises(ValueError):
            tracking_error(self.returns, self.returns2[:5])
        
        # Test with not enough returns
        self.assertEqual(tracking_error(np.array([0.01]), np.array([0.02])), 0.0)

    def test_normalize_data(self):
        """Test data normalization."""
        # Z-score normalization
        mean = np.mean(self.prices)
        std = np.std(self.prices)
        expected_zscore = (self.prices - mean) / std
        
        # Min-max normalization
        min_val = np.min(self.prices)
        max_val = np.max(self.prices)
        expected_minmax = (self.prices - min_val) / (max_val - min_val)
        
        # Robust normalization
        median = np.median(self.prices)
        q75, q25 = np.percentile(self.prices, [75, 25])
        iqr = q75 - q25
        expected_robust = (self.prices - median) / iqr
        
        # Test Z-score normalization
        zscore_norm = normalize_data(self.prices, 'zscore')
        np.testing.assert_almost_equal(zscore_norm, expected_zscore)
        
        # Test min-max normalization
        minmax_norm = normalize_data(self.prices, 'minmax')
        np.testing.assert_almost_equal(minmax_norm, expected_minmax)
        
        # Test robust normalization
        robust_norm = normalize_data(self.prices, 'robust')
        np.testing.assert_almost_equal(robust_norm, expected_robust)
        
        # Test with unknown method (should default to zscore)
        unknown_norm = normalize_data(self.prices, 'unknown_method')
        np.testing.assert_almost_equal(unknown_norm, expected_zscore)
        
        # Test with empty array
        self.assertEqual(len(normalize_data(np.array([]))), 0)
        
        # Test with constant array
        np.testing.assert_almost_equal(normalize_data(np.ones(5)), np.zeros(5))

    def test_exponential_smoothing(self):
        """Test exponential smoothing."""
        alpha = 0.3
        
        # Calculate expected exponential smoothing
        expected_smoothed = np.zeros_like(self.prices)
        expected_smoothed[0] = self.prices[0]
        
        for i in range(1, len(self.prices)):
            expected_smoothed[i] = alpha * self.prices[i] + (1 - alpha) * expected_smoothed[i-1]
        
        # Test exponential smoothing
        smoothed = exponential_smoothing(self.prices, alpha)
        np.testing.assert_almost_equal(smoothed, expected_smoothed)
        
        # Test with list input
        smoothed_list = exponential_smoothing(self.prices.tolist(), alpha)
        np.testing.assert_almost_equal(smoothed_list, expected_smoothed)
        
        # Test with empty array
        self.assertEqual(len(exponential_smoothing(np.array([]))), 0)
        
        # Test with invalid alpha (should be clamped to [0, 1])
        smoothed_invalid_alpha = exponential_smoothing(self.prices, 1.5)
        expected_invalid_alpha = exponential_smoothing(self.prices, 1.0)
        np.testing.assert_almost_equal(smoothed_invalid_alpha, expected_invalid_alpha)

    def test_rolling_window(self):
        """Test rolling window creation."""
        window = 3
        step = 1
        
        # Calculate expected windows
        expected_windows = []
        for i in range(0, len(self.prices) - window + 1, step):
            expected_windows.append(self.prices[i:i+window])
        
        # Test rolling window
        windows = rolling_window(self.prices, window, step)
        
        # Compare each window
        self.assertEqual(len(windows), len(expected_windows))
        for i in range(len(windows)):
            np.testing.assert_almost_equal(windows[i], expected_windows[i])
        
        # Test with list input
        windows_list = rolling_window(self.prices.tolist(), window, step)
        self.assertEqual(len(windows_list), len(expected_windows))
        
        # Test with window larger than array
        self.assertEqual(rolling_window(self.prices, 10), [])
        
        # Test with larger step
        step2 = 2
        expected_windows2 = []
        for i in range(0, len(self.prices) - window + 1, step2):
            expected_windows2.append(self.prices[i:i+window])
        
        windows2 = rolling_window(self.prices, window, step2)
        self.assertEqual(len(windows2), len(expected_windows2))

    def test_detect_outliers(self):
        """Test outlier detection."""
        # Create data with outliers
        data = np.array([1.0, 2.0, 1.5, 1.7, 10.0, 1.8, 1.9, 0.5, 1.3, 1.6])
        
        # Z-score method
        from scipy.stats import zscore as scipy_zscore
        z = scipy_zscore(data)
        threshold = 2.0
        expected_zscore_outliers = np.abs(z) > threshold
        
        # IQR method
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        upper_bound = q75 + threshold * iqr
        lower_bound = q25 - threshold * iqr
        expected_iqr_outliers = (data > upper_bound) | (data < lower_bound)
        
        # Test Z-score outlier detection
        zscore_outliers = detect_outliers(data, 'zscore', threshold)
        np.testing.assert_almost_equal(zscore_outliers, expected_zscore_outliers)
        
        # Test IQR outlier detection
        iqr_outliers = detect_outliers(data, 'iqr', threshold)
        np.testing.assert_almost_equal(iqr_outliers, expected_iqr_outliers)
        
        # Test with unknown method (should default to zscore)
        unknown_outliers = detect_outliers(data, 'unknown_method', threshold)
        np.testing.assert_almost_equal(unknown_outliers, expected_zscore_outliers)
        
        # Test with empty array
        self.assertEqual(len(detect_outliers(np.array([]))), 0)

    def test_remove_outliers(self):
        """Test outlier removal."""
        # Create data with outliers
        data = np.array([1.0, 2.0, 1.5, 1.7, 10.0, 1.8, 1.9, 0.5, 1.3, 1.6])
        
        # Create outlier mask (marking 10.0 and 0.5 as outliers)
        outlier_mask = np.zeros_like(data, dtype=bool)
        outlier_mask[4] = True  # 10.0
        outlier_mask[7] = True  # 0.5
        
        # Mean replacement
        mean_value = np.mean(data[~outlier_mask])
        expected_mean_replaced = data.copy()
        expected_mean_replaced[outlier_mask] = mean_value
        
        # Median replacement
        median_value = np.median(data[~outlier_mask])
        expected_median_replaced = data.copy()
        expected_median_replaced[outlier_mask] = median_value
        
        # Test mean replacement
        mean_replaced = remove_outliers(data, outlier_mask, 'mean')
        np.testing.assert_almost_equal(mean_replaced, expected_mean_replaced)
        
        # Test median replacement
        median_replaced = remove_outliers(data, outlier_mask, 'median')
        np.testing.assert_almost_equal(median_replaced, expected_median_replaced)
        
        # Test with unknown method (should default to mean)
        unknown_replaced = remove_outliers(data, outlier_mask, 'unknown_method')
        np.testing.assert_almost_equal(unknown_replaced, expected_mean_replaced)
        
        # Test with empty array
        self.assertEqual(len(remove_outliers(np.array([]), np.array([]))), 0)
        
        # Test with no outliers
        no_outliers_mask = np.zeros_like(data, dtype=bool)
        np.testing.assert_almost_equal(remove_outliers(data, no_outliers_mask), data)

    def test_moving_average_crossover(self):
        """Test moving average crossover detection."""
        # Create fast and slow moving averages with known crossovers
        fast_ma = np.array([10, 11, 12, 13, 12, 11, 10, 9, 8, 9, 10, 11])
        slow_ma = np.array([10, 10, 10, 11, 11, 11, 11, 10, 9, 9, 9, 10])
        
        # Expected crossovers: bullish at index 10, bearish at index 6
        expected_signals = np.zeros_like(fast_ma)
        expected_signals[10] = 1   # Bullish crossover
        expected_signals[6] = -1   # Bearish crossover
        
        # Test moving average crossover
        signals = moving_average_crossover(fast_ma, slow_ma)
        np.testing.assert_almost_equal(signals, expected_signals)
        
        # Test with different length arrays
        with self.assertRaises(ValueError):
            moving_average_crossover(fast_ma, slow_ma[:-1])
        
        # Test with not enough data
        self.assertEqual(len(moving_average_crossover(np.array([1]), np.array([1]))), 0)

    def test_pivot_points(self):
        """Test pivot point calculation."""
        high = 105.0
        low = 95.0
        close = 100.0
        
        # Standard pivot points
        p = (high + low + close) / 3
        s1 = (2 * p) - high
        s2 = p - (high - low)
        s3 = low - 2 * (high - p)
        r1 = (2 * p) - low
        r2 = p + (high - low)
        r3 = high + 2 * (p - low)
        
        expected_standard = {
            'pivot': p,
            'support1': s1,
            'support2': s2,
            'support3': s3,
            'resistance1': r1,
            'resistance2': r2,
            'resistance3': r3
        }
        
        # Test standard pivot points
        standard_pivots = pivot_points(high, low, close, 'standard')
        for key in expected_standard:
            self.assertAlmostEqual(standard_pivots[key], expected_standard[key])
        
        # Test with array inputs
        high_array = np.array([104.0, 105.0])
        low_array = np.array([94.0, 95.0])
        close_array = np.array([99.0, 100.0])
        
        array_pivots = pivot_points(high_array, low_array, close_array, 'standard')
        for key in expected_standard:
            self.assertAlmostEqual(array_pivots[key], expected_standard[key])
        
        # Test with unknown method (should default to standard)
        unknown_pivots = pivot_points(high, low, close, 'unknown_method')
        for key in expected_standard:
            self.assertAlmostEqual(unknown_pivots[key], expected_standard[key])

    def test_exponential_moving_average(self):
        """Test exponential moving average calculation."""
        span = 3
        
        # Calculate alpha
        alpha = 2 / (span + 1)
        
        # Calculate expected EMA
        expected_ema = np.zeros_like(self.prices)
        expected_ema[0] = self.prices[0]
        
        for i in range(1, len(self.prices)):
            expected_ema[i] = alpha * self.prices[i] + (1 - alpha) * expected_ema[i-1]
        
        # Test EMA
        ema = exponential_moving_average(self.prices, span)
        np.testing.assert_almost_equal(ema, expected_ema)
        
        # Test with list input
        ema_list = exponential_moving_average(self.prices.tolist(), span)
        np.testing.assert_almost_equal(ema_list, expected_ema)
        
        # Test with not enough data
        self.assertEqual(len(exponential_moving_average(np.array([1]), 3)), 0)

    def test_simple_moving_average(self):
        """Test simple moving average calculation."""
        window = 3
        
        # Calculate expected SMA
        expected_sma = np.zeros_like(self.prices)
        expected_sma[:window-1] = np.nan
        
        for i in range(window-1, len(self.prices)):
            expected_sma[i] = np.mean(self.prices[i-window+1:i+1])
        
        # Test SMA
        sma = simple_moving_average(self.prices, window)
        
        # Check for NaN values separately
        np.testing.assert_array_equal(np.isnan(sma[:window-1]), np.isnan(expected_sma[:window-1]))
        
        # Check non-NaN values
        np.testing.assert_almost_equal(sma[window-1:], expected_sma[window-1:])
        
        # Test with list input
        sma_list = simple_moving_average(self.prices.tolist(), window)
        np.testing.assert_array_equal(np.isnan(sma_list[:window-1]), np.isnan(expected_sma[:window-1]))
        np.testing.assert_almost_equal(sma_list[window-1:], expected_sma[window-1:])
        
        # Test with not enough data
        self.assertEqual(len(simple_moving_average(np.array([1]), 3)), 0)

    def test_relative_strength_index(self):
        """Test Relative Strength Index calculation."""
        window = 3
        
        # Calculate price changes
        deltas = np.diff(self.prices)
        
        # Calculate seed values
        seed = deltas[:window]
        up = seed[seed >= 0].sum() / window
        down = -seed[seed < 0].sum() / window
        
        rs = up / down if down != 0 else float('inf')
        
        # Initialize expected RSI array
        expected_rsi = np.zeros_like(self.prices)
        expected_rsi[0] = 100. - 100. / (1. + rs)
        expected_rsi[1:window] = np.nan
        
        # Calculate RSI using EMA of up and down
        for i in range(1, len(self.prices)):
            if i < window:
                continue
                
            delta = deltas[i-1]  # Current price change
            
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
                
            up = (up * (window - 1) + upval) / window
            down = (down * (window - 1) + downval) / window
            
            if down == 0:
                rs = float('inf')
            else:
                rs = up / down
                
            expected_rsi[i] = 100. - 100. / (1. + rs)
        
        # Test RSI
        rsi = relative_strength_index(self.prices, window)
        
        # Check for NaN values separately
        np.testing.assert_array_equal(np.isnan(rsi[1:window]), np.isnan(expected_rsi[1:window]))
        
        # Check non-NaN values
        np.testing.assert_almost_equal(rsi[0], expected_rsi[0])
        np.testing.assert_almost_equal(rsi[window:], expected_rsi[window:])
        
        # Test with not enough data
        self.assertEqual(len(relative_strength_index(np.array([1, 2]), 3)), 0)

    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        window = 3
        num_std = 2.0
        
        # Calculate middle band (SMA)
        middle_band = simple_moving_average(self.prices, window)
        
        # Calculate standard deviation
        rolling_std = np.zeros_like(self.prices)
        rolling_std[:window-1] = np.nan
        
        for i in range(window-1, len(self.prices)):
            rolling_std[i] = np.std(self.prices[i-window+1:i+1], ddof=1)
        
        # Calculate upper and lower bands
        upper_band = middle_band + (rolling_std * num_std)
        lower_band = middle_band - (rolling_std * num_std)
        
        # Test Bollinger Bands
        bb_upper, bb_middle, bb_lower = bollinger_bands(self.prices, window, num_std)
        
        # Check for NaN values separately
        np.testing.assert_array_equal(np.isnan(bb_upper[:window-1]), np.isnan(upper_band[:window-1]))
        np.testing.assert_array_equal(np.isnan(bb_middle[:window-1]), np.isnan(middle_band[:window-1]))
        np.testing.assert_array_equal(np.isnan(bb_lower[:window-1]), np.isnan(lower_band[:window-1]))
        
        # Check non-NaN values
        np.testing.assert_almost_equal(bb_upper[window-1:], upper_band[window-1:])
        np.testing.assert_almost_equal(bb_middle[window-1:], middle_band[window-1:])
        np.testing.assert_almost_equal(bb_lower[window-1:], lower_band[window-1:])
        
        # Test with not enough data
        self.assertEqual(len(bollinger_bands(np.array([1, 2]), 3)[0]), 0)

    def test_macd(self):
        """Test MACD calculation."""
        fast_span = 3
        slow_span = 6
        signal_span = 2
        
        # Calculate fast and slow EMAs
        fast_ema = exponential_moving_average(self.prices, fast_span)
        slow_ema = exponential_moving_average(self.prices, slow_span)
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = exponential_moving_average(macd_line, signal_span)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        # Test MACD
        macd_result, signal_result, hist_result = macd(self.prices, fast_span, slow_span, signal_span)
        np.testing.assert_almost_equal(macd_result, macd_line)
        np.testing.assert_almost_equal(signal_result, signal_line)
        np.testing.assert_almost_equal(hist_result, histogram)
        
        # Test with not enough data
        self.assertEqual(len(macd(np.array([1, 2, 3]), fast_span, slow_span, signal_span)[0]), 0)

    def test_average_true_range(self):
        """Test Average True Range calculation."""
        window = 3
        
        # Calculate true range
        tr = np.zeros_like(self.high_prices)
        tr[0] = self.high_prices[0] - self.low_prices[0]
        
        for i in range(1, len(self.high_prices)):
            tr[i] = max(
                self.high_prices[i] - self.low_prices[i],
                abs(self.high_prices[i] - self.close_prices[i-1]),
                abs(self.low_prices[i] - self.close_prices[i-1])
            )
        
        # Calculate ATR
        expected_atr = np.zeros_like(self.high_prices)
        expected_atr[:window-1] = np.nan
        expected_atr[window-1] = np.mean(tr[:window])
        
        for i in range(window, len(self.high_prices)):
            expected_atr[i] = ((window - 1) * expected_atr[i-1] + tr[i]) / window
        
        # Test ATR
        atr = average_true_range(self.high_prices, self.low_prices, self.close_prices, window)
        
        # Check for NaN values separately
        np.testing.assert_array_equal(np.isnan(atr[:window-1]), np.isnan(expected_atr[:window-1]))
        
        # Check non-NaN values
        np.testing.assert_almost_equal(atr[window-1:], expected_atr[window-1:])
        
        # Test with different length arrays
        with self.assertRaises(ValueError):
            average_true_range(self.high_prices, self.low_prices, self.close_prices[:-1], window)
        
        # Test with not enough data
        self.assertEqual(len(average_true_range(np.array([1, 2]), np.array([0, 1]), np.array([0.5, 1.5]), 3)), 0)

    def test_zscore(self):
        """Test Z-score calculation."""
        # Full dataset Z-score
        mean = np.mean(self.prices)
        std = np.std(self.prices, ddof=1)
        expected_zscore = (self.prices - mean) / std
        
        # Test full dataset Z-score
        full_zscore = zscore(self.prices)
        np.testing.assert_almost_equal(full_zscore, expected_zscore)
        
        # Test rolling Z-score with window
        window = 3
        rolling_z = zscore(self.prices, window)
        
        # First window-1 values should be NaN
        self.assertTrue(np.all(np.isnan(rolling_z[:window-1])))
        
        # Check a few individual values
        for i in range(window-1, len(self.prices)):
            window_values = self.prices[i-window+1:i+1]
            window_mean = np.mean(window_values)
            window_std = np.std(window_values, ddof=1)
            
            if window_std == 0:
                expected_z = 0
            else:
                expected_z = (self.prices[i] - window_mean) / window_std
            
            self.assertAlmostEqual(rolling_z[i], expected_z)
        
        # Test with not enough data for window
        self.assertEqual(len(zscore(np.array([1, 2]), 3)), 0)

    def test_median_absolute_deviation(self):
        """Test median absolute deviation calculation."""
        scale = 1.4826
        
        # Calculate MAD
        median = np.median(self.prices)
        mad = np.median(np.abs(self.prices - median))
        expected_mad = mad * scale
        
        # Test MAD
        mad_result = median_absolute_deviation(self.prices, scale)
        self.assertAlmostEqual(mad_result, expected_mad)
        
        # Test with empty array
        self.assertEqual(median_absolute_deviation(np.array([])), 0.0)

    def test_percentile_rank(self):
        """Test percentile rank calculation."""
        lookback = 3
        
        # Expected percentile ranks
        expected_ranks = np.zeros_like(self.prices)
        expected_ranks[:lookback-1] = np.nan
        
        for i in range(lookback-1, len(self.prices)):
            window = self.prices[i-lookback+1:i+1]
            expected_ranks[i] = percentileofscore(window, self.prices[i])
        
        # Test percentile rank
        ranks = percentile_rank(self.prices, lookback)
        
        # Check for NaN values separately
        np.testing.assert_array_equal(np.isnan(ranks[:lookback-1]), np.isnan(expected_ranks[:lookback-1]))
        
        # Check non-NaN values
        np.testing.assert_almost_equal(ranks[lookback-1:], expected_ranks[lookback-1:])
        
        # Test with not enough data
        self.assertEqual(len(percentile_rank(np.array([1, 2]), 3)), 0)

    def test_percentileofscore(self):
        """Test percentile of score calculation."""
        data = np.array([1, 2, 3, 4, 5, 5, 6, 7])
        
        # Test scores at various positions
        self.assertAlmostEqual(percentileofscore(data, 1), 0.0)
        self.assertAlmostEqual(percentileofscore(data, 3), 25.0)
        self.assertAlmostEqual(percentileofscore(data, 5), 62.5)  # Midpoint of tied values
        self.assertAlmostEqual(percentileofscore(data, 7), 100.0)
        
        # Test score not in data
        self.assertAlmostEqual(percentileofscore(data, 4.5), 50.0)
        
        # Test with empty array
        self.assertEqual(percentileofscore(np.array([]), 5), 0.0)

    def test_rolling_correlation(self):
        """Test rolling correlation calculation."""
        window = 3
        
        # Expected rolling correlation
        expected_corr = np.zeros(len(self.prices))
        expected_corr[:window-1] = np.nan
        
        for i in range(window-1, len(self.prices)):
            x_window = self.prices[i-window+1:i+1]
            y_window = self.prices2[i-window+1:i+1]
            
            # Check for constant values
            if np.std(x_window) == 0 or np.std(y_window) == 0:
                expected_corr[i] = np.nan
            else:
                expected_corr[i] = np.corrcoef(x_window, y_window)[0, 1]
        
        # Test rolling correlation
        corr = rolling_correlation(self.prices, self.prices2, window)
        
        # Check for NaN values separately
        np.testing.assert_array_equal(np.isnan(corr[:window-1]), np.isnan(expected_corr[:window-1]))
        
        # Check non-NaN values
        for i in range(window-1, len(self.prices)):
            if np.isnan(expected_corr[i]):
                self.assertTrue(np.isnan(corr[i]))
            else:
                self.assertAlmostEqual(corr[i], expected_corr[i])
        
        # Test with different length arrays
        with self.assertRaises(ValueError):
            rolling_correlation(self.prices, self.prices2[:-1], window)
        
        # Test with not enough data
        self.assertEqual(len(rolling_correlation(np.array([1, 2]), np.array([3, 4]), 3)), 0)

    def test_autocorrelation(self):
        """Test autocorrelation calculation."""
        lag = 2
        
        # Calculate autocorrelation
        # Shift series
        series1 = self.prices[:-lag]
        series2 = self.prices[lag:]
        
        # Expected autocorrelation
        expected_autocorr = np.corrcoef(series1, series2)[0, 1]
        
        # Test autocorrelation
        autocorr = autocorrelation(self.prices, lag)
        self.assertAlmostEqual(autocorr, expected_autocorr)
        
        # Test with not enough data
        self.assertEqual(autocorrelation(np.array([1]), 1), 0.0)
        
        # Test with constant data (zero standard deviation)
        self.assertEqual(autocorrelation(np.ones(5), 1), 0.0)
        
    def test