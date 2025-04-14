"""
Performance Metrics Module

This module handles real-time calculation of trading performance metrics.
Unlike the PerformanceTracker that handles storage and historical analysis,
this module focuses on efficient, low-latency calculation of key metrics.

The module adheres to the following architectural principles:
- Stateless where possible (calculation functions don't maintain state)
- Strict performance budget (all critical calculations optimized)
- Single source of truth (relies on state_manager for data)
- Observable system (all calculations can be monitored)
"""

import logging
import numpy as np
import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from functools import lru_cache

# Configure logger
logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Handles real-time calculation of trading performance metrics.

    This class follows the architecture guidelines:
    - Only calculates metrics, doesn't store historical data
    - Optimized for low latency calculations
    - Clearly defined interfaces with other components
    - Explicit error handling
    """

    def __init__(self, state_manager=None, risk_free_rate: float = 0.0):
        """
        Initialize the performance metrics calculator.

        Args:
            state_manager: Reference to the system state manager
            risk_free_rate: Risk-free rate for Sharpe ratio calculation (annualized)
        """
        self.state_manager = state_manager
        self.risk_free_rate = risk_free_rate
        logger.info("Performance metrics initialized")

    def calculate_returns(self, values: List[float]) -> List[float]:
        """
        Calculate period-to-period returns from a series of values.

        Args:
            values: List of portfolio/asset values

        Returns:
            List of returns (as decimals, not percentages)
        """
        if len(values) < 2:
            return []

        returns = []
        for i in range(1, len(values)):
            if values[i-1] > 0:  # Protect against division by zero
                ret = (values[i] / values[i-1]) - 1.0
            else:
                ret = 0.0
            returns.append(ret)

        return returns

    def calculate_log_returns(self, values: List[float]) -> List[float]:
        """
        Calculate logarithmic returns from a series of values.

        Args:
            values: List of portfolio/asset values

        Returns:
            List of log returns
        """
        if len(values) < 2:
            return []

        log_returns = []
        for i in range(1, len(values)):
            if values[i-1] > 0 and values[i] > 0:  # Protect against invalid log
                log_ret = np.log(values[i] / values[i-1])
            else:
                log_ret = 0.0
            log_returns.append(log_ret)

        return log_returns

    def calculate_total_return(self,
                              initial_value: float,
                              current_value: float) -> float:
        """
        Calculate total return given initial and current values.

        Args:
            initial_value: Starting value
            current_value: Ending value

        Returns:
            Total return as decimal
        """
        if initial_value <= 0:
            logger.warning("Initial value is zero or negative, cannot calculate return")
            return 0.0

        return (current_value / initial_value) - 1.0

    def calculate_annualized_return(self,
                                   total_return: float,
                                   days: int) -> float:
        """
        Calculate annualized return from total return and number of days.

        Args:
            total_return: Total return as decimal
            days: Number of days in the period

        Returns:
            Annualized return as decimal
        """
        if days <= 0:
            logger.warning("Days parameter is zero or negative, cannot calculate annualized return")
            return 0.0

        return (1 + total_return) ** (365.0 / days) - 1.0

    def calculate_volatility(self,
                            returns: List[float],
                            annualize: bool = True,
                            trading_days_per_year: int = 252) -> float:
        """
        Calculate volatility (standard deviation of returns).

        Args:
            returns: List of period returns
            annualize: Whether to annualize the volatility
            trading_days_per_year: Number of trading days in a year

        Returns:
            Volatility as decimal
        """
        if len(returns) < 2:
            logger.warning("Not enough returns to calculate volatility")
            return 0.0

        vol = np.std(returns, ddof=1)

        if annualize:
            # Scale based on return frequency
            vol = vol * np.sqrt(trading_days_per_year)

        return vol

    def calculate_sharpe_ratio(self,
                              returns: List[float],
                              risk_free_rate: Optional[float] = None,
                              annualize: bool = True,
                              trading_days_per_year: int = 252) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: List of period returns
            risk_free_rate: Risk-free rate (annualized), uses instance value if None
            annualize: Whether to annualize the result
            trading_days_per_year: Number of trading days in a year

        Returns:
            Sharpe ratio
        """
        if len(returns) < 2:
            logger.warning("Not enough returns to calculate Sharpe ratio")
            return 0.0

        rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate

        # Convert annualized risk-free rate to per-period
        if annualize:
            per_period_rf = (1 + rf_rate) ** (1 / trading_days_per_year) - 1
        else:
            per_period_rf = rf_rate / trading_days_per_year

        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return <= 0:
            logger.warning("Zero or negative standard deviation, cannot calculate Sharpe ratio")
            return 0.0

        sharpe = (mean_return - per_period_rf) / std_return

        # Annualize if requested
        if annualize:
            sharpe = sharpe * np.sqrt(trading_days_per_year)

        return sharpe

    def calculate_sortino_ratio(self,
                               returns: List[float],
                               risk_free_rate: Optional[float] = None,
                               annualize: bool = True,
                               trading_days_per_year: int = 252) -> float:
        """
        Calculate Sortino ratio (using downside deviation instead of standard deviation).

        Args:
            returns: List of period returns
            risk_free_rate: Risk-free rate (annualized), uses instance value if None
            annualize: Whether to annualize the result
            trading_days_per_year: Number of trading days in a year

        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            logger.warning("Not enough returns to calculate Sortino ratio")
            return 0.0

        rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate

        # Convert annualized risk-free rate to per-period
        if annualize:
            per_period_rf = (1 + rf_rate) ** (1 / trading_days_per_year) - 1
        else:
            per_period_rf = rf_rate / trading_days_per_year

        mean_return = np.mean(returns)

        # Calculate downside deviation (standard deviation of negative returns only)
        negative_returns = [r for r in returns if r < 0]

        if not negative_returns:
            logger.info("No negative returns, cannot calculate meaningful Sortino ratio")
            return float('inf') if mean_return > per_period_rf else 0.0

        downside_deviation = np.std(negative_returns, ddof=1)

        if downside_deviation <= 0:
            logger.warning("Zero or negative downside deviation, cannot calculate Sortino ratio")
            return 0.0

        sortino = (mean_return - per_period_rf) / downside_deviation

        # Annualize if requested
        if annualize:
            sortino = sortino * np.sqrt(trading_days_per_year)

        return sortino

    def calculate_max_drawdown(self, values: List[float]) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown from a series of values.

        Args:
            values: List of portfolio/asset values

        Returns:
            Tuple of (maximum drawdown as decimal, peak index, trough index)
        """
        if len(values) < 2:
            logger.warning("Not enough values to calculate maximum drawdown")
            return 0.0, 0, 0

        max_drawdown = 0.0
        peak_idx = 0
        trough_idx = 0
        peak = values[0]
        peak_i = 0

        for i, value in enumerate(values):
            if value > peak:
                peak = value
                peak_i = i
            else:
                drawdown = 1 - (value / peak)
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    peak_idx = peak_i
                    trough_idx = i

        return max_drawdown, peak_idx, trough_idx

    def calculate_calmar_ratio(self,
                              annualized_return: float,
                              max_drawdown: float) -> float:
        """
        Calculate Calmar ratio (annualized return divided by maximum drawdown).

        Args:
            annualized_return: Annualized return as decimal
            max_drawdown: Maximum drawdown as decimal

        Returns:
            Calmar ratio
        """
        if max_drawdown <= 0:
            logger.warning("Zero or negative maximum drawdown, cannot calculate Calmar ratio")
            return 0.0 if annualized_return <= 0 else float('inf')

        return annualized_return / max_drawdown

    def calculate_win_rate(self, trades: List[Dict[str, Any]]) -> float:
        """
        Calculate win rate from trade history.

        Args:
            trades: List of trade dictionaries, each with a 'pnl' key

        Returns:
            Win rate as decimal
        """
        if not trades:
            return 0.0

        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        return len(winning_trades) / len(trades)

    def calculate_profit_factor(self, trades: List[Dict[str, Any]]) -> float:
        """
        Calculate profit factor (gross profit / gross loss).

        Args:
            trades: List of trade dictionaries, each with a 'pnl' key

        Returns:
            Profit factor
        """
        if not trades:
            return 0.0

        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]

        gross_profit = sum(t.get('pnl', 0) for t in winning_trades)
        gross_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))

        if gross_loss <= 0:
            return 0.0 if gross_profit <= 0 else float('inf')

        return gross_profit / gross_loss

    def calculate_average_trade(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate average trade metrics.

        Args:
            trades: List of trade dictionaries, each with a 'pnl' key

        Returns:
            Dictionary with 'avg_trade', 'avg_win', and 'avg_loss'
        """
        if not trades:
            return {'avg_trade': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0}

        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]

        avg_trade = sum(t.get('pnl', 0) for t in trades) / len(trades) if trades else 0.0
        avg_win = sum(t.get('pnl', 0) for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
        avg_loss = sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0.0

        return {
            'avg_trade': avg_trade,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }

    def calculate_expectancy(self,
                            win_rate: float,
                            avg_win: float,
                            avg_loss: float) -> float:
        """
        Calculate expectancy (expected value of a trade).

        Args:
            win_rate: Win rate as decimal
            avg_win: Average win amount
            avg_loss: Average loss amount (positive number)

        Returns:
            Expectancy value
        """
        return (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))

    def calculate_kelly_criterion(self,
                                 win_rate: float,
                                 win_loss_ratio: float) -> float:
        """
        Calculate Kelly Criterion optimal position size.

        Args:
            win_rate: Win rate as decimal
            win_loss_ratio: Ratio of average win to average loss

        Returns:
            Kelly percentage (between 0 and 1)
        """
        if win_loss_ratio <= 0:
            return 0.0

        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)

        # Cap at 1.0 and floor at 0.0
        return max(0.0, min(1.0, kelly))

    def calculate_risk_of_ruin(self,
                              win_rate: float,
                              risk_reward_ratio: float) -> float:
        """
        Calculate risk of ruin (probability of losing all capital).

        Args:
            win_rate: Win rate as decimal
            risk_reward_ratio: Risk to reward ratio (e.g., 1.0 for risking $1 to make $1)

        Returns:
            Risk of ruin probability
        """
        if win_rate >= 1.0 or win_rate <= 0.0:
            logger.warning("Win rate must be between 0 and 1 for risk of ruin calculation")
            return 0.0 if win_rate >= 1.0 else 1.0

        if risk_reward_ratio <= 0:
            logger.warning("Risk reward ratio must be positive for risk of ruin calculation")
            return 0.0

        q = 1.0 - win_rate
        p = win_rate

        if p > q:
            return ((q / p) ** 1) ** 1
        else:
            return 1.0

    def calculate_ulcer_index(self, values: List[float]) -> float:
        """
        Calculate Ulcer Index (measure of downside risk).

        Args:
            values: List of portfolio/asset values

        Returns:
            Ulcer Index
        """
        if len(values) < 2:
            return 0.0

        # Calculate percentage drawdown at each point
        max_value = values[0]
        drawdowns = []

        for value in values:
            max_value = max(max_value, value)
            pct_drawdown = (max_value - value) / max_value if max_value > 0 else 0
            drawdowns.append(pct_drawdown)

        # Square, average, and take square root
        return np.sqrt(np.mean(np.square(drawdowns)))

    def calculate_benchmark_metrics(self,
                                   returns: List[float],
                                   benchmark_returns: List[float]) -> Dict[str, float]:
        """
        Calculate metrics relative to a benchmark.

        Args:
            returns: List of strategy/portfolio returns
            benchmark_returns: List of benchmark returns

        Returns:
            Dictionary with alpha, beta, and other metrics
        """
        if len(returns) != len(benchmark_returns) or len(returns) < 2:
            logger.warning("Returns and benchmark returns must have the same length and at least 2 points")
            return {
                'alpha': 0.0,
                'beta': 0.0,
                'correlation': 0.0,
                'r_squared': 0.0,
                'tracking_error': 0.0,
                'information_ratio': 0.0
            }

        # Calculate beta (covariance / variance)
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns, ddof=1)

        if benchmark_variance > 0:
            beta = covariance / benchmark_variance
        else:
            beta = 0.0

        # Calculate alpha (Jensen's alpha)
        mean_return = np.mean(returns)
        mean_benchmark_return = np.mean(benchmark_returns)
        alpha = mean_return - (beta * mean_benchmark_return)

        # Calculate correlation
        if np.std(returns) > 0 and np.std(benchmark_returns) > 0:
            correlation = np.corrcoef(returns, benchmark_returns)[0, 1]
        else:
            correlation = 0.0

        # Calculate R-squared
        r_squared = correlation ** 2

        # Calculate tracking error
        tracking_diff = np.array(returns) - np.array(benchmark_returns)
        tracking_error = np.std(tracking_diff, ddof=1)

        # Calculate information ratio
        if tracking_error > 0:
            information_ratio = (mean_return - mean_benchmark_return) / tracking_error
        else:
            information_ratio = 0.0

        return {
            'alpha': alpha,
            'beta': beta,
            'correlation': correlation,
            'r_squared': r_squared,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio
        }

    def calculate_var(self,
                     returns: List[float],
                     confidence_level: float = 0.95,
                     method: str = 'historical') -> float:
        """
        Calculate Value at Risk.

        Args:
            returns: List of returns
            confidence_level: Confidence level (default 0.95 for 95%)
            method: VaR method ('historical', 'parametric', or 'monte_carlo')

        Returns:
            Value at Risk as a positive decimal
        """
        if not returns:
            return 0.0

        if method == 'historical':
            # Historical VaR - simply use the quantile of returns
            return abs(np.quantile(returns, 1 - confidence_level))

        elif method == 'parametric':
            # Parametric VaR - assume normal distribution
            mean = np.mean(returns)
            std = np.std(returns, ddof=1)
            z_score = abs(np.percentile(np.random.normal(0, 1, 10000), (1 - confidence_level) * 100))
            return abs(mean - z_score * std)

        elif method == 'monte_carlo':
            # Simple Monte Carlo - resample from historical returns
            np.random.seed(42)  # For reproducibility
            simulations = 10000
            sampled_returns = np.random.choice(returns, size=simulations, replace=True)
            return abs(np.quantile(sampled_returns, 1 - confidence_level))

        else:
            logger.warning(f"Unknown VaR method: {method}, using historical")
            return abs(np.quantile(returns, 1 - confidence_level))

    def calculate_cvar(self,
                      returns: List[float],
                      confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).

        Args:
            returns: List of returns
            confidence_level: Confidence level (default 0.95 for 95%)

        Returns:
            Conditional Value at Risk as a positive decimal
        """
        if not returns:
            return 0.0

        # Calculate VaR first
        var = self.calculate_var(returns, confidence_level, 'historical')

        # CVaR is the average of returns beyond VaR
        tail_returns = [r for r in returns if r <= -var]

        if not tail_returns:
            return var  # Fallback if no returns beyond VaR

        return abs(np.mean(tail_returns))

    def calculate_all_metrics(self,
                             portfolio_values: List[Tuple[float, float]],
                             trades: List[Dict[str, Any]],
                             benchmark_values: Optional[List[Tuple[float, float]]] = None) -> Dict[str, Any]:
        """
        Calculate all performance metrics at once.

        Args:
            portfolio_values: List of (timestamp, value) tuples
            trades: List of trade dictionaries
            benchmark_values: Optional list of (timestamp, value) tuples for benchmark

        Returns:
            Dictionary with all performance metrics
        """
        try:
            # Extract timestamps and values
            timestamps, values = zip(*portfolio_values) if portfolio_values else ([], [])

            if len(values) < 2:
                logger.warning("Not enough portfolio values to calculate metrics")
                return {}

            # Calculate returns
            returns = self.calculate_returns(values)

            # Calculate benchmark returns if available
            if benchmark_values and len(benchmark_values) >= 2:
                b_timestamps, b_values = zip(*benchmark_values)
                benchmark_returns = self.calculate_returns(b_values)
            else:
                benchmark_returns = None

            # Calculate basic performance metrics
            initial_value = values[0]
            current_value = values[-1]
            total_return = self.calculate_total_return(initial_value, current_value)

            # Calculate time-based metrics if we have timestamps
            if timestamps:
                start_date = datetime.datetime.fromtimestamp(timestamps[0]).date()
                end_date = datetime.datetime.fromtimestamp(timestamps[-1]).date()
                days = (end_date - start_date).days

                annualized_return = self.calculate_annualized_return(total_return, days) if days > 0 else 0.0
            else:
                start_date = None
                end_date = None
                days = 0
                annualized_return = 0.0

            # Calculate risk metrics
            volatility = self.calculate_volatility(returns)
            max_drawdown, peak_idx, trough_idx = self.calculate_max_drawdown(values)

            # Calculate ratio metrics
            sharpe_ratio = self.calculate_sharpe_ratio(returns)
            sortino_ratio = self.calculate_sortino_ratio(returns)
            calmar_ratio = self.calculate_calmar_ratio(annualized_return, max_drawdown) if max_drawdown > 0 else 0.0

            # Calculate risk metrics
            var_95 = self.calculate_var(returns, 0.95)
            cvar_95 = self.calculate_cvar(returns, 0.95)
            ulcer_index = self.calculate_ulcer_index(values)

            # Calculate trade metrics if available
            if trades:
                win_rate = self.calculate_win_rate(trades)
                profit_factor = self.calculate_profit_factor(trades)
                trade_averages = self.calculate_average_trade(trades)
                expectancy = self.calculate_expectancy(
                    win_rate,
                    trade_averages['avg_win'],
                    abs(trade_averages['avg_loss'])
                )

                # Calculate Kelly criterion if possible
                if trade_averages['avg_loss'] != 0:
                    win_loss_ratio = trade_averages['avg_win'] / abs(trade_averages['avg_loss'])
                    kelly = self.calculate_kelly_criterion(win_rate, win_loss_ratio)
                else:
                    win_loss_ratio = float('inf')
                    kelly = 0.0

                risk_of_ruin = self.calculate_risk_of_ruin(win_rate, 1.0/win_loss_ratio) if win_loss_ratio > 0 else 1.0
            else:
                win_rate = 0.0
                profit_factor = 0.0
                trade_averages = {'avg_trade': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0}
                expectancy = 0.0
                kelly = 0.0
                risk_of_ruin = 1.0

            # Calculate benchmark metrics if available
            if benchmark_returns and len(benchmark_returns) == len(returns):
                benchmark_metrics = self.calculate_benchmark_metrics(returns, benchmark_returns)
            else:
                benchmark_metrics = {
                    'alpha': 0.0,
                    'beta': 0.0,
                    'correlation': 0.0,
                    'r_squared': 0.0,
                    'tracking_error': 0.0,
                    'information_ratio': 0.0
                }

            # Compile all metrics
            metrics = {
                # Basic metrics
                'initial_value': initial_value,
                'current_value': current_value,
                'total_return': total_return,
                'annualized_return': annualized_return,

                # Time information
                'start_date': start_date.isoformat() if start_date else None,
                'end_date': end_date.isoformat() if end_date else None,
                'days': days,

                # Risk metrics
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'ulcer_index': ulcer_index,

                # Ratio metrics
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,

                # Trade metrics
                'num_trades': len(trades),
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_trade': trade_averages['avg_trade'],
                'avg_win': trade_averages['avg_win'],
                'avg_loss': trade_averages['avg_loss'],
                'expectancy': expectancy,
                'kelly_criterion': kelly,
                'risk_of_ruin': risk_of_ruin,

                # Benchmark metrics
                'benchmark': benchmark_metrics
            }

            return metrics

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}", exc_info=True)
            return {}

    def get_real_time_metrics(self, window_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Get real-time performance metrics from the state manager.

        Args:
            window_size: Optional window size for recent metrics calculation

        Returns:
            Dictionary with performance metrics
        """
        if not self.state_manager:
            logger.warning("No state manager available for real-time metrics")
            return {}

        try:
            # Get portfolio values from state manager
            portfolio_values = self.state_manager.get_portfolio_values(window_size)

            # Get trades from state manager
            trades = self.state_manager.get_trades(window_size)

            # Get benchmark values if available
            benchmark_values = self.state_manager.get_benchmark_values(window_size)

            # Calculate all metrics
            return self.calculate_all_metrics(portfolio_values, trades, benchmark_values)

        except Exception as e:
            logger.error(f"Error getting real-time metrics: {str(e)}", exc_info=True)
            return {}