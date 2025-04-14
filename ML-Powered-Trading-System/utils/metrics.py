"""
Financial Metrics Utility Module

This module provides reusable financial metrics calculation functions for use across
the trading system. Unlike the performance_metrics module which focuses on real-time
calculation of complex trading performance metrics, this utility module provides:

1. Core mathematical functions for basic financial calculations
2. Vectorized operations for efficient processing
3. Common financial metrics that may be needed across different components
4. Helper functions for data preparation and transformation

The module follows the system architecture principles:
- Stateless functions (no state maintained between calls)
- Strict performance budget (optimized for vectorized operations)
- Single source of truth (consistent calculation methods)
- Deterministic behavior (consistent results for same inputs)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Union, Optional, Any, Callable
from functools import lru_cache
import logging
from datetime import datetime, timedelta

# Configure logger
logger = logging.getLogger(__name__)

# Type aliases for clarity
PriceType = Union[float, np.ndarray, List[float]]
DateType = Union[str, datetime, np.datetime64, pd.Timestamp]
WindowType = Union[int, timedelta, str]


# ==================== Basic Return Calculations ====================

def simple_returns(prices: PriceType, fill_na: bool = True) -> np.ndarray:
    """
    Calculate simple period-to-period returns from a series of prices.
    
    Args:
        prices: Array of prices
        fill_na: Whether to fill NA values with zeros
        
    Returns:
        Array of simple returns
    """
    if isinstance(prices, list):
        prices = np.array(prices)
    
    if len(prices) < 2:
        return np.array([])
    
    returns = np.diff(prices) / prices[:-1]
    
    if fill_na:
        returns = np.nan_to_num(returns, nan=0.0)
        
    return returns


def log_returns(prices: PriceType, fill_na: bool = True) -> np.ndarray:
    """
    Calculate logarithmic returns from a series of prices.
    
    Args:
        prices: Array of prices
        fill_na: Whether to fill NA values with zeros
        
    Returns:
        Array of log returns
    """
    if isinstance(prices, list):
        prices = np.array(prices)
    
    if len(prices) < 2:
        return np.array([])
    
    returns = np.diff(np.log(prices))
    
    if fill_na:
        returns = np.nan_to_num(returns, nan=0.0)
        
    return returns


def total_return(start_value: float, end_value: float) -> float:
    """
    Calculate total return between two values.
    
    Args:
        start_value: Initial value
        end_value: Final value
        
    Returns:
        Total return as a decimal
    """
    if start_value <= 0:
        logger.warning("Start value is zero or negative, cannot calculate return")
        return 0.0
    
    return (end_value / start_value) - 1.0


def annualize_return(total_return_value: float, days: int, 
                    trading_days_per_year: int = 252) -> float:
    """
    Annualize a return given the period in days.
    
    Args:
        total_return_value: Total return as decimal
        days: Number of days in the period
        trading_days_per_year: Number of trading days in a year
        
    Returns:
        Annualized return as decimal
    """
    if days <= 0:
        logger.warning("Days parameter is zero or negative, cannot annualize return")
        return 0.0
    
    # Use calendar days approach
    return (1 + total_return_value) ** (trading_days_per_year / days) - 1.0


def compound_returns(returns: PriceType) -> float:
    """
    Compound a series of returns.
    
    Args:
        returns: Array of period returns (as decimals)
        
    Returns:
        Compounded total return
    """
    if isinstance(returns, list):
        returns = np.array(returns)
    
    if len(returns) == 0:
        return 0.0
    
    return np.prod(1 + returns) - 1.0


# ==================== Risk Metrics ====================

def volatility(returns: PriceType, annualize: bool = True, 
              periods_per_year: int = 252) -> float:
    """
    Calculate volatility (standard deviation of returns).
    
    Args:
        returns: Array of period returns
        annualize: Whether to annualize the result
        periods_per_year: Number of periods in a year
        
    Returns:
        Volatility as decimal
    """
    if isinstance(returns, list):
        returns = np.array(returns)
    
    if len(returns) < 2:
        logger.warning("Not enough returns to calculate volatility")
        return 0.0
    
    vol = np.std(returns, ddof=1)
    
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
        
    return vol


def downside_deviation(returns: PriceType, threshold: float = 0.0, 
                      annualize: bool = True, periods_per_year: int = 252) -> float:
    """
    Calculate downside deviation (standard deviation of returns below threshold).
    
    Args:
        returns: Array of period returns
        threshold: Minimum acceptable return
        annualize: Whether to annualize the result
        periods_per_year: Number of periods in a year
        
    Returns:
        Downside deviation as decimal
    """
    if isinstance(returns, list):
        returns = np.array(returns)
    
    if len(returns) < 2:
        logger.warning("Not enough returns to calculate downside deviation")
        return 0.0
    
    # Filter returns below threshold
    downside_returns = returns[returns < threshold]
    
    if len(downside_returns) == 0:
        logger.info("No returns below threshold, downside deviation is 0")
        return 0.0
    
    dd = np.std(downside_returns, ddof=1)
    
    if annualize:
        dd = dd * np.sqrt(periods_per_year)
        
    return dd


def max_drawdown(prices: PriceType) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown from a series of prices or values.
    
    Args:
        prices: Array of prices or values
        
    Returns:
        Tuple of (max drawdown as decimal, peak index, trough index)
    """
    if isinstance(prices, list):
        prices = np.array(prices)
    
    if len(prices) < 2:
        logger.warning("Not enough prices to calculate maximum drawdown")
        return 0.0, 0, 0
    
    # Calculate the maximum drawdown
    peak = prices[0]
    max_dd = 0.0
    peak_idx = 0
    trough_idx = 0
    
    for i, price in enumerate(prices):
        if price > peak:
            peak = price
            peak_i = i
        else:
            dd = (peak - price) / peak
            if dd > max_dd:
                max_dd = dd
                peak_idx = peak_i
                trough_idx = i
    
    return max_dd, peak_idx, trough_idx


def drawdowns(prices: PriceType) -> np.ndarray:
    """
    Calculate drawdowns at each point in a price series.
    
    Args:
        prices: Array of prices or values
        
    Returns:
        Array of drawdown values at each point
    """
    if isinstance(prices, list):
        prices = np.array(prices)
    
    if len(prices) < 2:
        return np.zeros_like(prices)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(prices)
    
    # Calculate drawdowns
    drawdown_array = (running_max - prices) / running_max
    
    return drawdown_array


def value_at_risk(returns: PriceType, confidence_level: float = 0.95, 
                 method: str = 'historical') -> float:
    """
    Calculate Value at Risk (VaR) from a series of returns.
    
    Args:
        returns: Array of returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        method: Method to use ('historical', 'parametric', or 'monte_carlo')
        
    Returns:
        Value at Risk as a positive decimal
    """
    if isinstance(returns, list):
        returns = np.array(returns)
    
    if len(returns) < 10:
        logger.warning("Not enough returns to calculate reliable VaR")
        return 0.0
    
    if method == 'historical':
        # Historical VaR - empirical quantile
        return abs(np.percentile(returns, 100 * (1 - confidence_level)))
    
    elif method == 'parametric':
        # Parametric VaR - assuming normal distribution
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        z_score = abs(np.percentile(np.random.normal(0, 1, 10000), 
                                   (1 - confidence_level) * 100))
        return abs(mean - z_score * std)
    
    elif method == 'monte_carlo':
        # Simple Monte Carlo simulation
        np.random.seed(42)  # For reproducibility
        simulations = 10000
        sampled_returns = np.random.choice(returns, size=simulations, replace=True)
        return abs(np.percentile(sampled_returns, 100 * (1 - confidence_level)))
    
    else:
        logger.warning(f"Unknown VaR method: {method}, using historical")
        return abs(np.percentile(returns, 100 * (1 - confidence_level)))


def conditional_value_at_risk(returns: PriceType, confidence_level: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) or Expected Shortfall (ES).
    
    Args:
        returns: Array of returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        Conditional Value at Risk as a positive decimal
    """
    if isinstance(returns, list):
        returns = np.array(returns)
    
    if len(returns) < 10:
        logger.warning("Not enough returns to calculate reliable CVaR")
        return 0.0
    
    # Calculate VaR first
    var = value_at_risk(returns, confidence_level, 'historical')
    
    # CVaR is the average of returns beyond VaR
    threshold = -var  # Negative since we're looking at losses
    tail_returns = returns[returns <= threshold]
    
    if len(tail_returns) == 0:
        return var  # Fallback if no returns beyond VaR
    
    return abs(np.mean(tail_returns))


# ==================== Performance Ratios ====================

def sharpe_ratio(returns: PriceType, risk_free_rate: float = 0.0, 
                annualize: bool = True, periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annualized)
        annualize: Whether to annualize the ratio
        periods_per_year: Number of periods in a year
        
    Returns:
        Sharpe ratio
    """
    if isinstance(returns, list):
        returns = np.array(returns)
    
    if len(returns) < 2:
        logger.warning("Not enough returns to calculate Sharpe ratio")
        return 0.0
    
    # Convert annualized risk-free rate to per-period
    if annualize:
        per_period_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    else:
        per_period_rf = risk_free_rate / periods_per_year
    
    excess_returns = np.mean(returns) - per_period_rf
    vol = np.std(returns, ddof=1)
    
    if vol <= 0:
        logger.warning("Zero volatility, cannot calculate Sharpe ratio")
        return 0.0
    
    sharpe = excess_returns / vol
    
    if annualize:
        sharpe = sharpe * np.sqrt(periods_per_year)
    
    return sharpe


def sortino_ratio(returns: PriceType, risk_free_rate: float = 0.0, 
                 annualize: bool = True, periods_per_year: int = 252) -> float:
    """
    Calculate Sortino ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annualized)
        annualize: Whether to annualize the ratio
        periods_per_year: Number of periods in a year
        
    Returns:
        Sortino ratio
    """
    if isinstance(returns, list):
        returns = np.array(returns)
    
    if len(returns) < 2:
        logger.warning("Not enough returns to calculate Sortino ratio")
        return 0.0
    
    # Convert annualized risk-free rate to per-period
    if annualize:
        per_period_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    else:
        per_period_rf = risk_free_rate / periods_per_year
    
    excess_returns = np.mean(returns) - per_period_rf
    dd = downside_deviation(returns, per_period_rf, False)
    
    if dd <= 0:
        logger.warning("Zero downside deviation, cannot calculate Sortino ratio")
        return 0.0 if excess_returns <= 0 else float('inf')
    
    sortino = excess_returns / dd
    
    if annualize:
        sortino = sortino * np.sqrt(periods_per_year)
    
    return sortino


def calmar_ratio(returns: PriceType, prices: Optional[PriceType] = None, 
                period_years: float = 1.0) -> float:
    """
    Calculate Calmar ratio.
    
    Args:
        returns: Array of returns
        prices: Optional array of prices (if not provided, will be derived from returns)
        period_years: Period in years for which the ratio is calculated
        
    Returns:
        Calmar ratio
    """
    if isinstance(returns, list):
        returns = np.array(returns)
    
    if len(returns) < 2:
        logger.warning("Not enough returns to calculate Calmar ratio")
        return 0.0
    
    # Annualized return
    ann_return = (np.prod(1 + returns) ** (1 / period_years)) - 1
    
    # Calculate prices if not provided
    if prices is None:
        prices = np.cumprod(1 + returns)
    elif isinstance(prices, list):
        prices = np.array(prices)
    
    # Calculate max drawdown
    max_dd, _, _ = max_drawdown(prices)
    
    if max_dd <= 0:
        logger.warning("Zero max drawdown, cannot calculate Calmar ratio")
        return 0.0 if ann_return <= 0 else float('inf')
    
    return ann_return / max_dd


def information_ratio(returns: PriceType, benchmark_returns: PriceType, 
                     annualize: bool = True, periods_per_year: int = 252) -> float:
    """
    Calculate Information Ratio.
    
    Args:
        returns: Array of returns
        benchmark_returns: Array of benchmark returns
        annualize: Whether to annualize the ratio
        periods_per_year: Number of periods in a year
        
    Returns:
        Information ratio
    """
    if isinstance(returns, list):
        returns = np.array(returns)
    
    if isinstance(benchmark_returns, list):
        benchmark_returns = np.array(benchmark_returns)
    
    if len(returns) != len(benchmark_returns) or len(returns) < 2:
        logger.warning("Return series must be the same length and have at least 2 points")
        return 0.0
    
    # Active returns (return difference)
    active_returns = returns - benchmark_returns
    
    # Tracking error (std of active returns)
    tracking_error = np.std(active_returns, ddof=1)
    
    if tracking_error <= 0:
        logger.warning("Zero tracking error, cannot calculate Information ratio")
        return 0.0
    
    # Information ratio
    ir = np.mean(active_returns) / tracking_error
    
    if annualize:
        ir = ir * np.sqrt(periods_per_year)
    
    return ir


def treynor_ratio(returns: PriceType, benchmark_returns: PriceType, 
                 risk_free_rate: float = 0.0, 
                 annualize: bool = True, periods_per_year: int = 252) -> float:
    """
    Calculate Treynor Ratio.
    
    Args:
        returns: Array of returns
        benchmark_returns: Array of benchmark returns
        risk_free_rate: Risk-free rate (annualized)
        annualize: Whether to annualize the ratio
        periods_per_year: Number of periods in a year
        
    Returns:
        Treynor ratio
    """
    if isinstance(returns, list):
        returns = np.array(returns)
    
    if isinstance(benchmark_returns, list):
        benchmark_returns = np.array(benchmark_returns)
    
    if len(returns) != len(benchmark_returns) or len(returns) < 2:
        logger.warning("Return series must be the same length and have at least 2 points")
        return 0.0
    
    # Calculate beta
    cov = np.cov(returns, benchmark_returns)[0, 1]
    var = np.var(benchmark_returns, ddof=1)
    
    if var <= 0:
        logger.warning("Zero benchmark variance, cannot calculate beta")
        return 0.0
    
    beta = cov / var
    
    if beta == 0:
        logger.warning("Zero beta, cannot calculate Treynor ratio")
        return 0.0
    
    # Convert annualized risk-free rate to per-period
    if annualize:
        per_period_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    else:
        per_period_rf = risk_free_rate / periods_per_year
    
    # Excess return
    excess_return = np.mean(returns) - per_period_rf
    
    # Treynor ratio
    treynor = excess_return / beta
    
    if annualize:
        treynor = treynor * periods_per_year
    
    return treynor


# ==================== Moving Window Calculations ====================

def rolling_returns(prices: PriceType, window: int = 252) -> np.ndarray:
    """
    Calculate rolling returns over a window.
    
    Args:
        prices: Array of prices
        window: Window size in periods
        
    Returns:
        Array of rolling returns
    """
    if isinstance(prices, list):
        prices = np.array(prices)
    
    if len(prices) < window:
        logger.warning(f"Not enough prices for window of {window}")
        return np.array([])
    
    return np.array([
        (prices[i] / prices[i - window]) - 1 
        for i in range(window, len(prices))
    ])


def rolling_volatility(returns: PriceType, window: int = 30, 
                      annualize: bool = True, periods_per_year: int = 252) -> np.ndarray:
    """
    Calculate rolling volatility.
    
    Args:
        returns: Array of returns
        window: Window size in periods
        annualize: Whether to annualize the result
        periods_per_year: Number of periods in a year
        
    Returns:
        Array of rolling volatility values
    """
    if isinstance(returns, list):
        returns = np.array(returns)
    
    if len(returns) < window:
        logger.warning(f"Not enough returns for window of {window}")
        return np.array([])
    
    # Calculate rolling standard deviation
    roll_vol = np.array([
        np.std(returns[i-window:i], ddof=1) 
        for i in range(window, len(returns) + 1)
    ])
    
    if annualize:
        roll_vol = roll_vol * np.sqrt(periods_per_year)
    
    return roll_vol


def rolling_sharpe(returns: PriceType, window: int = 63, risk_free_rate: float = 0.0,
                  annualize: bool = True, periods_per_year: int = 252) -> np.ndarray:
    """
    Calculate rolling Sharpe ratio.
    
    Args:
        returns: Array of returns
        window: Window size in periods
        risk_free_rate: Risk-free rate (annualized)
        annualize: Whether to annualize the ratio
        periods_per_year: Number of periods in a year
        
    Returns:
        Array of rolling Sharpe ratio values
    """
    if isinstance(returns, list):
        returns = np.array(returns)
    
    if len(returns) < window:
        logger.warning(f"Not enough returns for window of {window}")
        return np.array([])
    
    # Convert annualized risk-free rate to per-period
    if annualize:
        per_period_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    else:
        per_period_rf = risk_free_rate / periods_per_year
    
    # Calculate rolling Sharpe ratio
    sharpe_values = []
    for i in range(window, len(returns) + 1):
        window_returns = returns[i-window:i]
        mean_return = np.mean(window_returns)
        std_return = np.std(window_returns, ddof=1)
        
        if std_return <= 0:
            sharpe = 0.0
        else:
            sharpe = (mean_return - per_period_rf) / std_return
            if annualize:
                sharpe = sharpe * np.sqrt(periods_per_year)
        
        sharpe_values.append(sharpe)
    
    return np.array(sharpe_values)


# ==================== Trade Analysis ====================

def win_rate(pnl_values: List[float]) -> float:
    """
    Calculate win rate from a list of trade P&L values.
    
    Args:
        pnl_values: List of P&L values for trades
        
    Returns:
        Win rate as a decimal
    """
    if not pnl_values:
        return 0.0
    
    winning_trades = sum(1 for pnl in pnl_values if pnl > 0)
    return winning_trades / len(pnl_values)


def profit_factor(pnl_values: List[float]) -> float:
    """
    Calculate profit factor (gross profits / gross losses).
    
    Args:
        pnl_values: List of P&L values for trades
        
    Returns:
        Profit factor
    """
    if not pnl_values:
        return 0.0
    
    gross_profits = sum(pnl for pnl in pnl_values if pnl > 0)
    gross_losses = abs(sum(pnl for pnl in pnl_values if pnl < 0))
    
    if gross_losses == 0:
        return float('inf') if gross_profits > 0 else 0.0
    
    return gross_profits / gross_losses


def expectancy(pnl_values: List[float]) -> float:
    """
    Calculate trade expectancy (average P&L per trade).
    
    Args:
        pnl_values: List of P&L values for trades
        
    Returns:
        Expectancy value
    """
    if not pnl_values:
        return 0.0
    
    return sum(pnl_values) / len(pnl_values)


def kelly_criterion(win_rate_value: float, win_loss_ratio: float) -> float:
    """
    Calculate Kelly Criterion optimal position size.
    
    Args:
        win_rate_value: Win rate as decimal
        win_loss_ratio: Average win amount / average loss amount
        
    Returns:
        Kelly percentage (between 0 and 1)
    """
    if win_loss_ratio <= 0:
        return 0.0
    
    kelly = win_rate_value - ((1 - win_rate_value) / win_loss_ratio)
    
    # Cap at 1.0 and floor at 0.0
    return max(0.0, min(1.0, kelly))


# ==================== Risk Analysis ====================

def risk_contribution(returns: List[PriceType], weights: List[float]) -> List[float]:
    """
    Calculate risk contribution of each asset in a portfolio.
    
    Args:
        returns: List of return series for each asset
        weights: Portfolio weights
        
    Returns:
        List of risk contributions
    """
    if len(returns) != len(weights):
        raise ValueError("Number of return series must match number of weights")
    
    # Convert to numpy arrays for easier calculation
    returns_array = np.array([np.array(r) if isinstance(r, list) else r for r in returns])
    weights_array = np.array(weights)
    
    # Calculate covariance matrix
    cov_matrix = np.cov(returns_array)
    
    # Calculate portfolio volatility
    portfolio_vol = np.sqrt(weights_array.T @ cov_matrix @ weights_array)
    
    if portfolio_vol <= 0:
        logger.warning("Zero portfolio volatility, cannot calculate risk contributions")
        return [0.0] * len(weights)
    
    # Calculate marginal contribution to risk
    mcr = cov_matrix @ weights_array
    
    # Calculate risk contribution
    rc = weights_array * mcr / portfolio_vol
    
    return rc.tolist()


def portfolio_volatility(returns: List[PriceType], weights: List[float],
                        annualize: bool = True, periods_per_year: int = 252) -> float:
    """
    Calculate portfolio volatility.
    
    Args:
        returns: List of return series for each asset
        weights: Portfolio weights
        annualize: Whether to annualize the result
        periods_per_year: Number of periods in a year
        
    Returns:
        Portfolio volatility
    """
    if len(returns) != len(weights):
        raise ValueError("Number of return series must match number of weights")
    
    # Convert to numpy arrays for easier calculation
    returns_array = np.array([np.array(r) if isinstance(r, list) else r for r in returns])
    weights_array = np.array(weights)
    
    # Calculate covariance matrix
    cov_matrix = np.cov(returns_array)
    
    # Calculate portfolio volatility
    vol = np.sqrt(weights_array.T @ cov_matrix @ weights_array)
    
    if annualize:
        vol = vol * np.sqrt(periods_per_year)
    
    return vol


def portfolio_returns(returns: List[PriceType], weights: List[float]) -> np.ndarray:
    """
    Calculate portfolio returns from individual asset returns and weights.
    
    Args:
        returns: List of return series for each asset
        weights: Portfolio weights
        
    Returns:
        Portfolio return series
    """
    if len(returns) != len(weights):
        raise ValueError("Number of return series must match number of weights")
    
    # Convert to numpy arrays for easier calculation
    returns_array = np.array([np.array(r) if isinstance(r, list) else r for r in returns])
    weights_array = np.array(weights)
    
    # Calculate portfolio returns
    port_returns = np.sum(returns_array.T * weights_array, axis=1)
    
    return port_returns

def beta(returns: PriceType, benchmark_returns: PriceType) -> float:
    """
    Calculate beta (sensitivity to benchmark returns).
    
    Args:
        returns: Array of returns
        benchmark_returns: Array of benchmark returns
        
    Returns:
        Beta value
    """
    if isinstance(returns, list):
        returns = np.array(returns)
    
    if isinstance(benchmark_returns, list):
        benchmark_returns = np.array(benchmark_returns)
    
    if len(returns) != len(benchmark_returns) or len(returns) < 2:
        logger.warning("Return series must be the same length and have at least 2 points")
        return 0.0
    
    # Calculate beta using covariance / variance
    cov = np.cov(returns, benchmark_returns)[0, 1]
    var = np.var(benchmark_returns, ddof=1)
    
    if var <= 0:
        logger.warning("Zero benchmark variance, cannot calculate beta")
        return 0.0
    
    return cov / var


def alpha(returns: PriceType, benchmark_returns: PriceType, 
         risk_free_rate: float = 0.0, annualize: bool = True, 
         periods_per_year: int = 252) -> float:
    """
    Calculate Jensen's alpha.
    
    Args:
        returns: Array of returns
        benchmark_returns: Array of benchmark returns
        risk_free_rate: Risk-free rate (annualized)
        annualize: Whether to annualize the result
        periods_per_year: Number of periods in a year
        
    Returns:
        Alpha value
    """
    if isinstance(returns, list):
        returns = np.array(returns)
    
    if isinstance(benchmark_returns, list):
        benchmark_returns = np.array(benchmark_returns)
    
    if len(returns) != len(benchmark_returns) or len(returns) < 2:
        logger.warning("Return series must be the same length and have at least 2 points")
        return 0.0
    
    # Calculate beta
    b = beta(returns, benchmark_returns)
    
    # Convert annualized risk-free rate to per-period
    if annualize:
        per_period_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    else:
        per_period_rf = risk_free_rate / periods_per_year
    
    # Calculate alpha
    mean_return = np.mean(returns)
    mean_benchmark = np.mean(benchmark_returns)
    
    alpha_value = mean_return - (per_period_rf + b * (mean_benchmark - per_period_rf))
    
    if annualize:
        alpha_value = (1 + alpha_value) ** periods_per_year - 1
    
    return alpha_value


def correlation(returns: PriceType, benchmark_returns: PriceType) -> float:
    """
    Calculate correlation between return series.
    
    Args:
        returns: Array of returns
        benchmark_returns: Array of benchmark returns
        
    Returns:
        Correlation coefficient
    """
    if isinstance(returns, list):
        returns = np.array(returns)
    
    if isinstance(benchmark_returns, list):
        benchmark_returns = np.array(benchmark_returns)
    
    if len(returns) != len(benchmark_returns) or len(returns) < 2:
        logger.warning("Return series must be the same length and have at least 2 points")
        return 0.0
    
    return np.corrcoef(returns, benchmark_returns)[0, 1]


def r_squared(returns: PriceType, benchmark_returns: PriceType) -> float:
    """
    Calculate R-squared (coefficient of determination).
    
    Args:
        returns: Array of returns
        benchmark_returns: Array of benchmark returns
        
    Returns:
        R-squared value
    """
    corr = correlation(returns, benchmark_returns)
    return corr ** 2


def tracking_error(returns: PriceType, benchmark_returns: PriceType,
                  annualize: bool = True, periods_per_year: int = 252) -> float:
    """
    Calculate tracking error.
    
    Args:
        returns: Array of returns
        benchmark_returns: Array of benchmark returns
        annualize: Whether to annualize the result
        periods_per_year: Number of periods in a year
        
    Returns:
        Tracking error
    """
    if isinstance(returns, list):
        returns = np.array(returns)
    
    if isinstance(benchmark_returns, list):
        benchmark_returns = np.array(benchmark_returns)
    
    if len(returns) != len(benchmark_returns) or len(returns) < 2:
        logger.warning("Return series must be the same length and have at least 2 points")
        return 0.0
    
    # Calculate tracking difference
    tracking_diff = returns - benchmark_returns
    
    # Calculate tracking error (standard deviation of tracking difference)
    te = np.std(tracking_diff, ddof=1)
    
    if annualize:
        te = te * np.sqrt(periods_per_year)
    
    return te


# ==================== Data Processing and Transformation ====================

@lru_cache(maxsize=128)
def normalize_data(data: PriceType, method: str = 'zscore') -> np.ndarray:
    """
    Normalize data using various methods.
    
    Args:
        data: Array of values
        method: Normalization method ('zscore', 'minmax', 'robust')
        
    Returns:
        Normalized data
    """
    if isinstance(data, list):
        data = np.array(data)
    
    if len(data) == 0:
        return np.array([])
    
    if method == 'zscore':
        # Z-score normalization: (x - mean) / std
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return np.zeros_like(data)
        
        return (data - mean) / std
    
    elif method == 'minmax':
        # Min-max normalization: (x - min) / (max - min)
        min_val = np.min(data)
        max_val = np.max(data)
        
        if max_val == min_val:
            return np.zeros_like(data)
        
        return (data - min_val) / (max_val - min_val)
    
    elif method == 'robust':
        # Robust normalization: (x - median) / IQR
        median = np.median(data)
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        
        if iqr == 0:
            return np.zeros_like(data)
        
        return (data - median) / iqr
    
    else:
        logger.warning(f"Unknown normalization method: {method}, using zscore")
        return normalize_data(data, 'zscore')


def exponential_smoothing(data: PriceType, alpha: float = 0.3) -> np.ndarray:
    """
    Apply exponential smoothing to a time series.
    
    Args:
        data: Array of values
        alpha: Smoothing factor (0-1)
        
    Returns:
        Smoothed data
    """
    if isinstance(data, list):
        data = np.array(data)
    
    if len(data) == 0:
        return np.array([])
    
    if alpha < 0 or alpha > 1:
        logger.warning(f"Alpha should be between 0 and 1, got {alpha}")
        alpha = max(0, min(1, alpha))
    
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]
    
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
    
    return smoothed


def rolling_window(data: PriceType, window: int, step: int = 1) -> List[np.ndarray]:
    """
    Create rolling windows from a time series.
    
    Args:
        data: Array of values
        window: Window size
        step: Step size between windows
        
    Returns:
        List of arrays, each containing a window of data
    """
    if isinstance(data, list):
        data = np.array(data)
    
    if len(data) < window:
        return []
    
    # Create windows
    windows = []
    for i in range(0, len(data) - window + 1, step):
        windows.append(data[i:i+window])
    
    return windows


def detect_outliers(data: PriceType, method: str = 'zscore', 
                   threshold: float = 3.0) -> np.ndarray:
    """
    Detect outliers in a time series.
    
    Args:
        data: Array of values
        method: Detection method ('zscore', 'iqr', 'mad')
        threshold: Outlier threshold
        
    Returns:
        Boolean array where True indicates an outlier
    """
    if isinstance(data, list):
        data = np.array(data)
    
    if len(data) == 0:
        return np.array([])
    
    if method == 'zscore':
        # Z-score method
        z = zscore(data)
        return np.abs(z) > threshold
    
    elif method == 'iqr':
        # Interquartile range method
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        upper_bound = q75 + threshold * iqr
        lower_bound = q25 - threshold * iqr
        
        return (data > upper_bound) | (data < lower_bound)
    
    elif method == 'mad':
        # Median absolute deviation method
        median = np.median(data)
        mad = median_absolute_deviation(data)
        
        return np.abs(data - median) > threshold * mad
    
    else:
        logger.warning(f"Unknown outlier detection method: {method}, using zscore")
        return detect_outliers(data, 'zscore', threshold)


def remove_outliers(data: PriceType, outlier_mask: np.ndarray, 
                   method: str = 'mean') -> np.ndarray:
    """
    Remove outliers from a time series by replacing them.
    
    Args:
        data: Array of values
        outlier_mask: Boolean array where True indicates an outlier
        method: Replacement method ('mean', 'median', 'interpolate', 'nearest')
        
    Returns:
        Array with outliers replaced
    """
    if isinstance(data, list):
        data = np.array(data)
    
    if len(data) == 0 or np.sum(outlier_mask) == 0:
        return data
    
    # Create copy to avoid modifying input
    cleaned = data.copy()
    
    if method == 'mean':
        # Replace with mean of non-outliers
        mean_value = np.mean(data[~outlier_mask])
        cleaned[outlier_mask] = mean_value
    
    elif method == 'median':
        # Replace with median of non-outliers
        median_value = np.median(data[~outlier_mask])
        cleaned[outlier_mask] = median_value
    
    elif method == 'interpolate':
        # Linear interpolation
        x = np.arange(len(data))
        cleaned = np.interp(x, x[~outlier_mask], data[~outlier_mask])
    
    elif method == 'nearest':
        # Replace with nearest non-outlier
        for i in np.where(outlier_mask)[0]:
            # Find nearest non-outlier
            non_outlier_indices = np.where(~outlier_mask)[0]
            if len(non_outlier_indices) == 0:
                continue
                
            nearest_idx = non_outlier_indices[np.argmin(np.abs(non_outlier_indices - i))]
            cleaned[i] = data[nearest_idx]
    
    else:
        logger.warning(f"Unknown outlier replacement method: {method}, using mean")
        return remove_outliers(data, outlier_mask, 'mean')
    
    return cleaned


def moving_average_crossover(fast_ma: PriceType, slow_ma: PriceType) -> np.ndarray:
    """
    Detect moving average crossover points.
    
    Args:
        fast_ma: Fast moving average array
        slow_ma: Slow moving average array
        
    Returns:
        Array with 1 for bullish crossover, -1 for bearish crossover, 0 otherwise
    """
    if isinstance(fast_ma, list):
        fast_ma = np.array(fast_ma)
    
    if isinstance(slow_ma, list):
        slow_ma = np.array(slow_ma)
    
    if len(fast_ma) != len(slow_ma):
        raise ValueError("Moving average arrays must have the same length")
    
    if len(fast_ma) < 2:
        return np.array([])
    
    # Initialize signal array
    signals = np.zeros(len(fast_ma))
    
    # Calculate crossover points
    for i in range(1, len(fast_ma)):
        # Check for bullish crossover (fast crosses above slow)
        if fast_ma[i-1] <= slow_ma[i-1] and fast_ma[i] > slow_ma[i]:
            signals[i] = 1
        # Check for bearish crossover (fast crosses below slow)
        elif fast_ma[i-1] >= slow_ma[i-1] and fast_ma[i] < slow_ma[i]:
            signals[i] = -1
    
    return signals


def pivot_points(high: PriceType, low: PriceType, close: PriceType, 
                method: str = 'standard') -> Dict[str, float]:
    """
    Calculate pivot points for a given period.
    
    Args:
        high: High price
        low: Low price
        close: Close price
        method: Pivot calculation method ('standard', 'fibonacci', 'woodie', 'camarilla')
        
    Returns:
        Dictionary with pivot level values
    """
    if isinstance(high, list) or isinstance(high, np.ndarray):
        high = high[-1]  # Use most recent high
    
    if isinstance(low, list) or isinstance(low, np.ndarray):
        low = low[-1]  # Use most recent low
    
    if isinstance(close, list) or isinstance(close, np.ndarray):
        close = close[-1]  # Use most recent close
    
    if method == 'standard':
        # Standard pivot points
        p = (high + low + close) / 3
        s1 = (2 * p) - high
        s2 = p - (high - low)
        s3 = low - 2 * (high - p)
        r1 = (2 * p) - low
        r2 = p + (high - low)
        r3 = high + 2 * (p - low)
        
        return {
            'pivot': p,
            'support1': s1,
            'support2': s2,
            'support3': s3,
            'resistance1': r1,
            'resistance2': r2,
            'resistance3': r3
        }
    
    elif method == 'fibonacci':
        # Fibonacci pivot points
        p = (high + low + close) / 3
        r1 = p + 0.382 * (high - low)
        r2 = p + 0.618 * (high - low)
        r3 = p + 1.000 * (high - low)
        s1 = p - 0.382 * (high - low)
        s2 = p - 0.618 * (high - low)
        s3 = p - 1.000 * (high - low)
        
        return {
            'pivot': p,
            'support1': s1,
            'support2': s2,
            'support3': s3,
            'resistance1': r1,
            'resistance2': r2,
            'resistance3': r3
        }
    
    elif method == 'woodie':
        # Woodie pivot points
        p = (high + low + 2 * close) / 4
        r1 = (2 * p) - low
        r2 = p + (high - low)
        s1 = (2 * p) - high
        s2 = p - (high - low)
        
        return {
            'pivot': p,
            'support1': s1,
            'support2': s2,
            'resistance1': r1,
            'resistance2': r2
        }
    
    elif method == 'camarilla':
        # Camarilla pivot points
        r4 = close + (high - low) * 1.5
        r3 = close + (high - low) * 1.25
        r2 = close + (high - low) * 1.1
        r1 = close + (high - low) * 1.05
        s1 = close - (high - low) * 1.05
        s2 = close - (high - low) * 1.1
        s3 = close - (high - low) * 1.25
        s4 = close - (high - low) * 1.5
        
        return {
            'resistance4': r4,
            'resistance3': r3,
            'resistance2': r2,
            'resistance1': r1,
            'support1': s1,
            'support2': s2,
            'support3': s3,
            'support4': s4
        }
    
    else:
        logger.warning(f"Unknown pivot point method: {method}, using standard")
        return pivot_points(high, low, close, 'standard')

def efficient_frontier(returns: List[PriceType], min_weight: float = 0.0, 
                      max_weight: float = 1.0, points: int = 20) -> Dict[str, List]:
    """
    Calculate the efficient frontier for a portfolio.
    
    Args:
        returns: List of return series for each asset
        min_weight: Minimum weight for each asset
        max_weight: Maximum weight for each asset
        points: Number of points on the frontier
        
    Returns:
        Dictionary with volatilities, returns, and weights for frontier
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        logger.warning("scipy not installed, cannot calculate efficient frontier")
        return {'volatilities': [], 'returns': [], 'sharpe_ratios': [], 'weights': []}
    
    # Convert to numpy arrays
    returns_array = [np.array(r) if isinstance(r, list) else r for r in returns]
    n_assets = len(returns_array)
    
    if n_assets < 2:
        logger.warning("Need at least 2 assets for efficient frontier")
        return {'volatilities': [], 'returns': [], 'sharpe_ratios': [], 'weights': []}
    
    # Calculate expected returns and covariance matrix
    exp_returns = np.array([np.mean(r) for r in returns_array])
    cov_matrix = np.cov(returns_array)
    
    # Define constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum of weights = 1
    ]
    
    # Define bounds
    bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
    
    # Function to minimize portfolio volatility
    def portfolio_volatility(weights):
        return np.sqrt(weights.T @ cov_matrix @ weights)
    
    # Function to calculate portfolio return
    def portfolio_return(weights):
        return np.sum(exp_returns * weights)
    
    # Function to calculate negative Sharpe ratio (for maximization)
    def negative_sharpe(weights):
        p_ret = portfolio_return(weights)
        p_vol = portfolio_volatility(weights)
        return -p_ret / p_vol if p_vol > 0 else 0
    
    # Calculate minimum volatility portfolio
    min_vol_result = minimize(
        portfolio_volatility,
        np.ones(n_assets) / n_assets,  # Equal weights initial guess
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    min_vol_weights = min_vol_result['x']
    min_vol = portfolio_volatility(min_vol_weights)
    min_ret = portfolio_return(min_vol_weights)
    
    # Calculate maximum return portfolio
    max_ret_weights = np.zeros(n_assets)
    max_ret_weights[np.argmax(exp_returns)] = 1.0
    max_ret = portfolio_return(max_ret_weights)
    
    # Calculate maximum Sharpe ratio portfolio
    max_sharpe_result = minimize(
        negative_sharpe,
        np.ones(n_assets) / n_assets,  # Equal weights initial guess
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    max_sharpe_weights = max_sharpe_result['x']
    max_sharpe_vol = portfolio_volatility(max_sharpe_weights)
    max_sharpe_ret = portfolio_return(max_sharpe_weights)
    max_sharpe_ratio = max_sharpe_ret / max_sharpe_vol if max_sharpe_vol > 0 else 0
    
    # Generate efficient frontier
    target_returns = np.linspace(min_ret, max_ret, points)
    efficient_weights = []
    efficient_vols = []
    efficient_rets = []
    efficient_sharpes = []
    
    for target_return in target_returns:
        # Add return constraint
        return_constraint = {
            'type': 'eq',
            'fun': lambda w: portfolio_return(w) - target_return
        }
        
        # Find minimum volatility for this target return
        constraints_with_return = constraints + [return_constraint]
        
        result = minimize(
            portfolio_volatility,
            np.ones(n_assets) / n_assets,  # Equal weights initial guess
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_with_return
        )
        
        if result['success']:
            weights = result['x']
            vol = portfolio_volatility(weights)
            ret = portfolio_return(weights)
            sharpe = ret / vol if vol > 0 else 0
            
            efficient_weights.append(weights)
            efficient_vols.append(vol)
            efficient_rets.append(ret)
            efficient_sharpes.append(sharpe)
    
    return {
        'volatilities': efficient_vols,
        'returns': efficient_rets,
        'sharpe_ratios': efficient_sharpes,
        'weights': efficient_weights,
        'min_vol_weights': min_vol_weights,
        'max_sharpe_weights': max_sharpe_weights
    }


def optimal_portfolio(returns: List[PriceType], target_return: Optional[float] = None,
                     target_risk: Optional[float] = None, 
                     objective: str = 'sharpe',
                     min_weight: float = 0.0, max_weight: float = 1.0) -> Dict[str, Any]:
    """
    Find the optimal portfolio weights according to a specified objective.
    
    Args:
        returns: List of return series for each asset
        target_return: Optional target return constraint
        target_risk: Optional target risk constraint
        objective: Optimization objective ('sharpe', 'min_risk', 'max_return')
        min_weight: Minimum weight for each asset
        max_weight: Maximum weight for each asset
        
    Returns:
        Dictionary with optimal weights and portfolio metrics
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        logger.warning("scipy not installed, cannot calculate optimal portfolio")
        return {'weights': [], 'volatility': 0, 'return': 0, 'sharpe_ratio': 0}
    
    # Convert to numpy arrays
    returns_array = [np.array(r) if isinstance(r, list) else r for r in returns]
    n_assets = len(returns_array)
    
    if n_assets < 1:
        logger.warning("Need at least 1 asset for portfolio optimization")
        return {'weights': [], 'volatility': 0, 'return': 0, 'sharpe_ratio': 0}
    
    # Calculate expected returns and covariance matrix
    exp_returns = np.array([np.mean(r) for r in returns_array])
    cov_matrix = np.cov(returns_array)
    
    # Define constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum of weights = 1
    ]
    
    # Add target return constraint if specified
    if target_return is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(exp_returns * w) - target_return
        })
    
    # Add target risk constraint if specified
    if target_risk is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sqrt(w.T @ cov_matrix @ w) - target_risk
        })
    
    # Define bounds
    bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
    
    # Define objective functions
    def portfolio_volatility(weights):
        return np.sqrt(weights.T @ cov_matrix @ weights)
    
    def portfolio_return(weights):
        return np.sum(exp_returns * weights)
    
    def negative_sharpe(weights):
        p_ret = portfolio_return(weights)
        p_vol = portfolio_volatility(weights)
        return -p_ret / p_vol if p_vol > 0 else 0
    
    def negative_return(weights):
        return -portfolio_return(weights)
    
    # Select objective function
    if objective == 'sharpe':
        obj_function = negative_sharpe
    elif objective == 'min_risk':
        obj_function = portfolio_volatility
    elif objective == 'max_return':
        obj_function = negative_return
    else:
        logger.warning(f"Unknown objective: {objective}, using Sharpe ratio")
        obj_function = negative_sharpe
    
    # Perform optimization
    result = minimize(
        obj_function,
        np.ones(n_assets) / n_assets,  # Equal weights initial guess
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    if not result['success']:
        logger.warning(f"Optimization failed: {result['message']}")
        return {'weights': [], 'volatility': 0, 'return': 0, 'sharpe_ratio': 0}
    
    # Get optimal weights
    weights = result['x']
    
    # Calculate portfolio metrics
    vol = portfolio_volatility(weights)
    ret = portfolio_return(weights)
    sharpe = ret / vol if vol > 0 else 0
    
    return {
        'weights': weights.tolist(),
        'volatility': vol,
        'return': ret,
        'sharpe_ratio': sharpe
    }


def optimal_position_sizing(win_probability: float, 
                           win_loss_ratio: float,
                           risk_tolerance: float = 0.5) -> float:
    """
    Calculate optimal position size using Kelly Criterion with risk constraint.
    
    Args:
        win_probability: Probability of winning trade
        win_loss_ratio: Ratio of average win to average loss
        risk_tolerance: Risk tolerance factor (0-1, fraction of full Kelly)
        
    Returns:
        Optimal position size as fraction of capital
    """
    if win_probability <= 0 or win_probability >= 1:
        logger.warning("Win probability must be between 0 and 1")
        return 0.0
    
    if win_loss_ratio <= 0:
        logger.warning("Win/loss ratio must be positive")
        return 0.0
    
    # Calculate full Kelly fraction
    kelly = win_probability - ((1 - win_probability) / win_loss_ratio)
    
    # Apply risk tolerance
    fractional_kelly = kelly * risk_tolerance
    
    # Ensure result is between 0 and 1
    return max(0.0, min(1.0, fractional_kelly))

def zscore(values: PriceType, window: int = None) -> np.ndarray:
    """
    Calculate z-score (standardized values).
    
    Args:
        values: Array of values
        window: Optional rolling window (if None, uses full dataset)
        
    Returns:
        Array of z-scores
    """
    if isinstance(values, list):
        values = np.array(values)
    
    if window is None:
        # Calculate for the entire dataset
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        
        if std == 0:
            return np.zeros_like(values)
        
        return (values - mean) / std
    else:
        # Calculate rolling z-score
        if len(values) < window:
            logger.warning(f"Not enough values for rolling z-score with window {window}")
            return np.array([])
        
        z_scores = np.zeros_like(values)
        z_scores[:window-1] = np.nan  # First window-1 values are undefined
        
        for i in range(window-1, len(values)):
            window_values = values[i-window+1:i+1]
            mean = np.mean(window_values)
            std = np.std(window_values, ddof=1)
            
            if std == 0:
                z_scores[i] = 0
            else:
                z_scores[i] = (values[i] - mean) / std
        
        return z_scores


def median_absolute_deviation(values: PriceType, scale: float = 1.4826) -> float:
    """
    Calculate median absolute deviation (robust measure of dispersion).
    
    Args:
        values: Array of values
        scale: Scaling factor for normal distribution (1.4826 makes MAD equivalent to 
               standard deviation for normal distributions)
        
    Returns:
        Median absolute deviation
    """
    if isinstance(values, list):
        values = np.array(values)
    
    if len(values) == 0:
        return 0.0
    
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    
    return mad * scale


def percentile_rank(values: PriceType, lookback: int) -> np.ndarray:
    """
    Calculate percentile rank of values within a lookback window.
    
    Args:
        values: Array of values
        lookback: Lookback period
        
    Returns:
        Array of percentile ranks (0-100)
    """
    if isinstance(values, list):
        values = np.array(values)
    
    if len(values) < lookback:
        logger.warning(f"Not enough values for percentile rank with lookback {lookback}")
        return np.array([])
    
    ranks = np.zeros_like(values)
    ranks[:lookback-1] = np.nan  # First lookback-1 values are undefined
    
    for i in range(lookback-1, len(values)):
        window = values[i-lookback+1:i+1]
        ranks[i] = percentileofscore(window, values[i])
    
    return ranks


def percentileofscore(a: PriceType, score: float) -> float:
    """
    Calculate the percentile rank of a score within an array.
    
    Args:
        a: Array of values
        score: Value to find rank for
        
    Returns:
        Percentile rank (0-100)
    """
    if isinstance(a, list):
        a = np.array(a)
    
    n = len(a)
    if n == 0:
        return 0.0
    
    # Count values below score
    below = np.sum(a < score)
    
    # Count values equal to score
    equal = np.sum(a == score)
    
    # Calculate percentile
    if equal == 0:
        return 100.0 * below / n
    else:
        return 100.0 * (below + 0.5 * equal) / n


def rolling_correlation(x: PriceType, y: PriceType, window: int) -> np.ndarray:
    """
    Calculate rolling correlation between two series.
    
    Args:
        x: First series
        y: Second series
        window: Correlation window
        
    Returns:
        Array of rolling correlation values
    """
    if isinstance(x, list):
        x = np.array(x)
    
    if isinstance(y, list):
        y = np.array(y)
    
    if len(x) != len(y):
        raise ValueError("Series must have the same length")
    
    if len(x) < window:
        logger.warning(f"Not enough values for rolling correlation with window {window}")
        return np.array([])
    
    correlations = np.zeros(len(x))
    correlations[:window-1] = np.nan  # First window-1 values are undefined
    
    for i in range(window-1, len(x)):
        x_window = x[i-window+1:i+1]
        y_window = y[i-window+1:i+1]
        
        # Check for constant values
        if np.std(x_window) == 0 or np.std(y_window) == 0:
            correlations[i] = np.nan
        else:
            correlations[i] = np.corrcoef(x_window, y_window)[0, 1]
    
    return correlations


def autocorrelation(values: PriceType, lag: int) -> float:
    """
    Calculate autocorrelation at a specific lag.
    
    Args:
        values: Array of values
        lag: Lag period
        
    Returns:
        Autocorrelation value
    """
    if isinstance(values, list):
        values = np.array(values)
    
    if len(values) <= lag:
        logger.warning(f"Not enough values for autocorrelation with lag {lag}")
        return 0.0
    
    # Shift series
    series1 = values[:-lag]
    series2 = values[lag:]
    
    # Calculate correlation
    if np.std(series1) == 0 or np.std(series2) == 0:
        return 0.0
    
    return np.corrcoef(series1, series2)[0, 1]


def hurst_exponent(prices: PriceType, max_lag: int = 20) -> float:
    """
    Calculate Hurst exponent to measure long-term memory of a time series.
    
    Args:
        prices: Array of prices or values
        max_lag: Maximum lag for calculation
        
    Returns:
        Hurst exponent value
    """
    if isinstance(prices, list):
        prices = np.array(prices)
    
    if len(prices) < max_lag * 2:
        logger.warning(f"Not enough values for Hurst exponent with max_lag {max_lag}")
        return 0.5  # Default to random walk
    
    # Calculate returns
    returns = np.diff(np.log(prices))
    
    # Calculate range/standard deviation for different lags
    lags = range(2, max_lag)
    tau = []
    rs = []
    
    for lag in lags:
        # Calculate range
        x = np.cumsum(returns - np.mean(returns))
        r = np.max(x) - np.min(x)
        
        # Calculate standard deviation
        s = np.std(returns)
        
        if s > 0:
            rs.append(r / s)
            tau.append(lag)
    
    if len(tau) < 2:
        return 0.5  # Default to random walk
    
    # Estimate Hurst exponent using linear regression
    log_tau = np.log(tau)
    log_rs = np.log(rs)
    
    # Fit regression line
    slope, _, _, _, _ = np.polyfit(log_tau, log_rs, 1, full=True)[0]
    
    return slope


def is_stationary(values: PriceType, significance: float = 0.05) -> bool:
    """
    Test whether a time series is stationary using the Augmented Dickey-Fuller test.
    
    Args:
        values: Array of values
        significance: Significance level
        
    Returns:
        True if the series is stationary, False otherwise
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        logger.warning("statsmodels not installed, cannot perform stationarity test")
        return False
    
    if isinstance(values, list):
        values = np.array(values)
    
    if len(values) < 20:
        logger.warning("Not enough values for stationarity test")
        return False
    
    # Perform ADF test
    result = adfuller(values)
    
    # Extract p-value
    p_value = result[1]
    
    # If p-value is less than significance level, the series is stationary
    return p_value < significance


def regime_hmm(returns: PriceType, n_regimes: int = 2) -> np.ndarray:
    """
    Identify market regimes using Hidden Markov Model.
    
    Args:
        returns: Array of returns
        n_regimes: Number of regimes to identify
        
    Returns:
        Array of regime labels
    """
    try:
        from hmmlearn import hmm
    except ImportError:
        logger.warning("hmmlearn not installed, cannot perform HMM regime detection")
        return np.zeros(len(returns))
    
    if isinstance(returns, list):
        returns = np.array(returns)
    
    if len(returns) < n_regimes * 10:
        logger.warning(f"Not enough returns for HMM with {n_regimes} regimes")
        return np.zeros(len(returns))
    
    # Reshape for HMM input
    X = returns.reshape(-1, 1)
    
    # Initialize HMM
    model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="diag", n_iter=1000)
    
    try:
        # Fit model
        model.fit(X)
        
        # Predict regimes
        regimes = model.predict(X)
    except Exception as e:
        logger.error(f"HMM failed: {str(e)}")
        regimes = np.zeros(len(returns))
    
    return regimes

def exponential_moving_average(prices: PriceType, span: int) -> np.ndarray:
    """
    Calculate exponential moving average.
    
    Args:
        prices: Array of prices
        span: EMA span (roughly equivalent to period)
        
    Returns:
        Array of EMA values
    """
    if isinstance(prices, list):
        prices = np.array(prices)
    
    if len(prices) < span:
        logger.warning(f"Not enough prices for EMA with span {span}")
        return np.array([])
    
    # Calculate alpha (smoothing factor)
    alpha = 2 / (span + 1)
    
    # Calculate EMA
    ema = np.zeros_like(prices)
    ema[0] = prices[0]  # Initialize with first price
    
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    
    return ema


def simple_moving_average(prices: PriceType, window: int) -> np.ndarray:
    """
    Calculate simple moving average.
    
    Args:
        prices: Array of prices
        window: SMA window
        
    Returns:
        Array of SMA values
    """
    if isinstance(prices, list):
        prices = np.array(prices)
    
    if len(prices) < window:
        logger.warning(f"Not enough prices for SMA with window {window}")
        return np.array([])
    
    # Calculate rolling mean
    sma = np.zeros_like(prices)
    sma[:window-1] = np.nan  # First window-1 values are undefined
    
    for i in range(window-1, len(prices)):
        sma[i] = np.mean(prices[i-window+1:i+1])
    
    return sma


def relative_strength_index(prices: PriceType, window: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Array of prices
        window: RSI window
        
    Returns:
        Array of RSI values
    """
    if isinstance(prices, list):
        prices = np.array(prices)
    
    if len(prices) < window + 1:
        logger.warning(f"Not enough prices for RSI with window {window}")
        return np.array([])
    
    # Calculate price changes
    deltas = np.diff(prices)
    
    # Calculate seed values
    seed = deltas[:window]
    up = seed[seed >= 0].sum() / window
    down = -seed[seed < 0].sum() / window
    
    if down == 0:
        rs = float('inf')
    else:
        rs = up / down
    
    rsi = np.zeros_like(prices)
    rsi[0] = 100. - 100. / (1. + rs)
    
    # Calculate RSI using EMA of up and down
    for i in range(1, len(prices)):
        if i < window:
            rsi[i] = np.nan
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
            
        rsi[i] = 100. - 100. / (1. + rs)
    
    return rsi


def bollinger_bands(prices: PriceType, window: int = 20, num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: Array of prices
        window: Window for moving average
        num_std: Number of standard deviations for bands
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    if isinstance(prices, list):
        prices = np.array(prices)
    
    if len(prices) < window:
        logger.warning(f"Not enough prices for Bollinger Bands with window {window}")
        return np.array([]), np.array([]), np.array([])
    
    # Calculate middle band (SMA)
    middle_band = simple_moving_average(prices, window)
    
    # Calculate standard deviation
    rolling_std = np.zeros_like(prices)
    rolling_std[:window-1] = np.nan  # First window-1 values are undefined
    
    for i in range(window-1, len(prices)):
        rolling_std[i] = np.std(prices[i-window+1:i+1], ddof=1)
    
    # Calculate upper and lower bands
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)
    
    return upper_band, middle_band, lower_band


def macd(prices: PriceType, fast_span: int = 12, slow_span: int = 26, signal_span: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    Args:
        prices: Array of prices
        fast_span: Span for fast EMA
        slow_span: Span for slow EMA
        signal_span: Span for signal line
        
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    if isinstance(prices, list):
        prices = np.array(prices)
    
    if len(prices) < slow_span:
        logger.warning(f"Not enough prices for MACD calculation")
        return np.array([]), np.array([]), np.array([])
    
    # Calculate fast and slow EMAs
    fast_ema = exponential_moving_average(prices, fast_span)
    slow_ema = exponential_moving_average(prices, slow_span)
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line
    signal_line = exponential_moving_average(macd_line, signal_span)
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def average_true_range(high_prices: PriceType, low_prices: PriceType, 
                      close_prices: PriceType, window: int = 14) -> np.ndarray:
    """
    Calculate Average True Range (ATR).
    
    Args:
        high_prices: Array of high prices
        low_prices: Array of low prices
        close_prices: Array of closing prices
        window: Window for ATR calculation
        
    Returns:
        Array of ATR values
    """
    if isinstance(high_prices, list):
        high_prices = np.array(high_prices)
    
    if isinstance(low_prices, list):
        low_prices = np.array(low_prices)
    
    if isinstance(close_prices, list):
        close_prices = np.array(close_prices)
    
    if len(high_prices) != len(low_prices) or len(high_prices) != len(close_prices):
        raise ValueError("High, low, and close price arrays must have the same length")
    
    if len(high_prices) < window:
        logger.warning(f"Not enough prices for ATR with window {window}")
        return np.array([])
    
    # Calculate true range
    tr = np.zeros_like(high_prices)
    tr[0] = high_prices[0] - low_prices[0]  # First TR is just high - low
    
    for i in range(1, len(high_prices)):
        # Maximum of:
        # 1. Current high - current low
        # 2. Abs(current high - previous close)
        # 3. Abs(current low - previous close)
        tr[i] = max(
            high_prices[i] - low_prices[i],
            abs(high_prices[i] - close_prices[i-1]),
            abs(low_prices[i] - close_prices[i-1])
        )
    
    # Calculate ATR using EMA
    atr = np.zeros_like(high_prices)
    atr[window-1] = np.mean(tr[:window])  # Initialize with simple average
    
    for i in range(window, len(high_prices)):
        atr[i] = ((window - 1) * atr[i-1] + tr[i]) / window
    
    # Set first window-1 values to NaN
    atr[:window-1] = np.nan
    
    return atr

# ==================== Date and Time Utilities ====================

def timestamp_to_datetime(timestamp: Union[float, int, str]) -> datetime:
    """
    Convert timestamp to datetime object.
    
    Args:
        timestamp: Timestamp value (seconds or milliseconds)
        
    Returns:
        Datetime object
    """
    if isinstance(timestamp, str):
        try:
            timestamp = float(timestamp)
        except ValueError:
            raise ValueError(f"Cannot convert string '{timestamp}' to timestamp")
    
    # Check if timestamp is in milliseconds (13 digits) and convert to seconds if needed
    if timestamp > 1e12:  # Threshold for milliseconds timestamps
        timestamp = timestamp / 1000.0
        
    return datetime.fromtimestamp(timestamp)


def datetime_to_timestamp(dt: DateType, unit: str = 'seconds') -> float:
    """
    Convert datetime to timestamp.
    
    Args:
        dt: Datetime object or string
        unit: Output unit ('seconds' or 'milliseconds')
        
    Returns:
        Timestamp value
    """
    # Convert string to datetime if needed
    if isinstance(dt, str):
        dt = pd.to_datetime(dt)
    
    # Convert pd.Timestamp to datetime if needed
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    
    # Convert np.datetime64 to datetime if needed
    if isinstance(dt, np.datetime64):
        dt = pd.Timestamp(dt).to_pydatetime()
    
    # Get timestamp in seconds
    ts = dt.timestamp()
    
    # Convert to milliseconds if requested
    if unit.lower() == 'milliseconds':
        ts = ts * 1000
        
    return ts


def date_range(start_date: DateType, end_date: DateType, 
              freq: str = 'D') -> List[datetime]:
    """
    Generate a list of dates between start and end dates.
    
    Args:
        start_date: Start date
        end_date: End date
        freq: Frequency ('D' for daily, 'W' for weekly, 'M' for monthly, etc.)
        
    Returns:
        List of datetime objects
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    return [d.to_pydatetime() for d in date_range]


def resample_prices(timestamps: List[Union[float, int]], 
                   prices: List[float], 
                   freq: str = 'D', 
                   method: str = 'last') -> Tuple[List[datetime], List[float]]:
    """
    Resample time series data to a different frequency.
    
    Args:
        timestamps: List of timestamps
        prices: List of price values
        freq: Target frequency ('D', 'W', 'M', etc.)
        method: Resampling method ('last', 'first', 'mean', 'max', 'min')
        
    Returns:
        Tuple of (resampled_dates, resampled_prices)
    """
    if len(timestamps) != len(prices):
        raise ValueError("Timestamps and prices must have the same length")
    
    if len(timestamps) == 0:
        return [], []
    
    # Convert timestamps to datetime
    dates = [timestamp_to_datetime(ts) for ts in timestamps]
    
    # Create DataFrame
    df = pd.DataFrame({'price': prices}, index=dates)
    
    # Resample
    if method == 'last':
        resampled = df.resample(freq).last()
    elif method == 'first':
        resampled = df.resample(freq).first()
    elif method == 'mean':
        resampled = df.resample(freq).mean()
    elif method == 'max':
        resampled = df.resample(freq).max()
    elif method == 'min':
        resampled = df.resample(freq).min()
    else:
        raise ValueError(f"Unknown resampling method: {method}")
    
    # Fill missing values (forward fill then backward fill)
    resampled = resampled.fillna(method='ffill').fillna(method='bfill')
    
    # Convert back to lists
    resampled_dates = [d.to_pydatetime() for d in resampled.index]
    resampled_prices = resampled['price'].tolist()
    
    return resampled_dates, resampled_prices


def ohlc_resample(timestamps: List[Union[float, int]],
                 open_prices: List[float],
                 high_prices: List[float],
                 low_prices: List[float],
                 close_prices: List[float],
                 volumes: Optional[List[float]] = None,
                 freq: str = 'D') -> Dict[str, List]:
    """
    Resample OHLC data to a different frequency.
    
    Args:
        timestamps: List of timestamps
        open_prices: List of open prices
        high_prices: List of high prices
        low_prices: List of low prices
        close_prices: List of close prices
        volumes: Optional list of volumes
        freq: Target frequency ('D', 'W', 'M', etc.)
        
    Returns:
        Dictionary with resampled OHLCV data
    """
    if not (len(timestamps) == len(open_prices) == len(high_prices) == 
            len(low_prices) == len(close_prices)):
        raise ValueError("All price arrays must have the same length")
    
    if volumes is not None and len(volumes) != len(timestamps):
        raise ValueError("Volume array must have the same length as price arrays")
    
    if len(timestamps) == 0:
        return {
            'timestamps': [],
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': [] if volumes is not None else None
        }
    
    # Convert timestamps to datetime
    dates = [timestamp_to_datetime(ts) for ts in timestamps]
    
    # Create DataFrame
    data = {
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices
    }
    
    if volumes is not None:
        data['volume'] = volumes
    
    df = pd.DataFrame(data, index=dates)
    
    # Resample
    resampled = df.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum' if volumes is not None else None
    })
    
    # Fill missing values (forward fill then backward fill)
    resampled = resampled.fillna(method='ffill').fillna(method='bfill')
    
    # Convert back to lists
    result = {
        'timestamps': [d.to_pydatetime() for d in resampled.index],
        'open': resampled['open'].tolist(),
        'high': resampled['high'].tolist(),
        'low': resampled['low'].tolist(),
        'close': resampled['close'].tolist()
    }
    
    if volumes is not None:
        result['volume'] = resampled['volume'].tolist()
    
    return result


def calculate_capacity(strategy_returns: PriceType,
                       avg_trade_size: float,
                       market_impact_model: Callable,
                       threshold: float = 0.2) -> float:
    """
    Estimate strategy capacity (maximum AUM before returns degrade).

    Args:
        strategy_returns: Historical strategy returns
        avg_trade_size: Average trade size as percentage of portfolio
        market_impact_model: Function to estimate market impact
        threshold: Maximum acceptable performance degradation

    Returns:
        Estimated capacity in base currency units
    """
    if isinstance(strategy_returns, list):
        strategy_returns = np.array(strategy_returns)

    if len(strategy_returns) == 0:
        return 0.0

    # Calculate baseline Sharpe ratio
    baseline_sharpe = sharpe_ratio(strategy_returns)

    # Start with a small AUM and increase until performance degrades
    aum = 1000000  # Start with $1M
    max_aum = 1000000000000  # Upper limit at $1T to prevent infinite loop

    while aum < max_aum:
        # Estimate impact at current AUM
        impact = market_impact_model(aum, avg_trade_size)

        # Adjust returns for impact
        adjusted_returns = strategy_returns - impact

        # Calculate new Sharpe ratio
        new_sharpe = sharpe_ratio(adjusted_returns)

        # Check if performance has degraded beyond threshold
        if baseline_sharpe > 0 and new_sharpe < baseline_sharpe * (1 - threshold):
            return aum

        # Increase AUM for next iteration
        aum *= 2

    return max_aum


def transaction_cost_analysis(trade_prices: List[float],
                              execution_prices: List[float],
                              benchmark_prices: List[float],
                              trade_sizes: List[float],
                              trade_sides: List[int]) -> Dict[str, float]:
    """
    Perform transaction cost analysis on executed trades.

    Args:
        trade_prices: Decision prices when trade was initiated
        execution_prices: Actual execution prices
        benchmark_prices: Benchmark prices (e.g., VWAP)
        trade_sizes: Sizes of trades in currency units
        trade_sides: Trade directions (1 for buy, -1 for sell)

    Returns:
        Dictionary with cost analysis metrics
    """
    if not (len(trade_prices) == len(execution_prices) == len(benchmark_prices) ==
            len(trade_sizes) == len(trade_sides)):
        raise ValueError("All input lists must have the same length")

    if len(trade_prices) == 0:
        return {
            'implementation_shortfall': 0.0,
            'market_impact': 0.0,
            'timing_cost': 0.0,
            'total_cost_bps': 0.0,
            'total_cost_currency': 0.0
        }

    # Convert to numpy arrays
    trade_prices = np.array(trade_prices)
    execution_prices = np.array(execution_prices)
    benchmark_prices = np.array(benchmark_prices)
    trade_sizes = np.array(trade_sizes)
    trade_sides = np.array(trade_sides)

    # Calculate implementation shortfall
    # For buys: execution_price - trade_price
    # For sells: trade_price - execution_price
    implementation_shortfall = trade_sides * (trade_prices - execution_prices)

    # Calculate market impact
    # For buys: execution_price - benchmark_price
    # For sells: benchmark_price - execution_price
    market_impact = trade_sides * (benchmark_prices - execution_prices)

    # Calculate timing cost
    # For buys: benchmark_price - trade_price
    # For sells: trade_price - benchmark_price
    timing_cost = trade_sides * (trade_prices - benchmark_prices)

    # Calculate total cost in currency units
    total_cost_currency = implementation_shortfall * trade_sizes

    # Calculate total cost in basis points
    total_cost_bps = (implementation_shortfall / trade_prices) * 10000  # Convert to basis points

    return {
        'implementation_shortfall': np.mean(implementation_shortfall),
        'market_impact': np.mean(market_impact),
        'timing_cost': np.mean(timing_cost),
        'total_cost_bps': np.mean(total_cost_bps),
        'total_cost_currency': np.sum(total_cost_currency)
    }


def system_health_metrics(cpu_usage: float, memory_usage: float,
                          latency: float, error_rate: float) -> Dict[str, float]:
    """
    Calculate system health score based on resource usage and performance.

    Args:
        cpu_usage: CPU usage as percentage (0-100)
        memory_usage: Memory usage as percentage (0-100)
        latency: Response latency in milliseconds
        error_rate: Error rate as percentage (0-100)

    Returns:
        Dictionary with health metrics and overall score
    """
    # Normalize all metrics to 0-1 scale (higher is better)
    cpu_score = 1 - (cpu_usage / 100)
    memory_score = 1 - (memory_usage / 100)

    # For latency, use an exponential decay function
    # 0ms -> 1.0, 100ms -> 0.37, 500ms -> 0.007
    latency_score = np.exp(-latency / 100)

    # For error rate, use a steeper decay
    # 0% -> 1.0, 1% -> 0.36, 5% -> 0.007
    error_score = np.exp(-error_rate / 1)

    # Calculate weighted average (error rate and latency are more critical)
    weights = {
        'cpu_score': 1,
        'memory_score': 1,
        'latency_score': 3,
        'error_score': 5
    }

    total_weight = sum(weights.values())

    weighted_sum = (
            weights['cpu_score'] * cpu_score +
            weights['memory_score'] * memory_score +
            weights['latency_score'] * latency_score +
            weights['error_score'] * error_score
    )

    overall_score = weighted_sum / total_weight

    return {
        'cpu_score': cpu_score,
        'memory_score': memory_score,
        'latency_score': latency_score,
        'error_score': error_score,
        'overall_health': overall_score
    }


def execution_quality_score(fill_times: List[float],
                            order_sizes: List[float],
                            slippages: List[float]) -> float:
    """
    Calculate execution quality score based on fill times and slippage.

    Args:
        fill_times: Time to fill orders in milliseconds
        order_sizes: Sizes of orders in currency units
        slippages: Slippage for each order in basis points

    Returns:
        Execution quality score (0-100)
    """
    if not (len(fill_times) == len(order_sizes) == len(slippages)):
        raise ValueError("All input lists must have the same length")

    if len(fill_times) == 0:
        return 0.0

    # Convert to numpy arrays
    fill_times = np.array(fill_times)
    order_sizes = np.array(order_sizes)
    slippages = np.array(slippages)

    # Normalize fill times (lower is better)
    # Scale so that 0ms -> 1.0, 1000ms -> 0.37
    fill_time_scores = np.exp(-fill_times / 1000)

    # Normalize slippages (lower is better)
    # Scale so that 0bp -> 1.0, 10bp -> 0.37
    slippage_scores = np.exp(-slippages / 10)

    # Weight by order size
    weights = order_sizes / np.sum(order_sizes)

    # Calculate weighted scores
    weighted_fill_time_score = np.sum(fill_time_scores * weights)
    weighted_slippage_score = np.sum(slippage_scores * weights)

    # Calculate overall score (70% slippage, 30% fill time)
    overall_score = (0.7 * weighted_slippage_score + 0.3 * weighted_fill_time_score) * 100

    return overall_score


def regime_stability_score(regime_labels: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Calculate stability score for market regime classifications.

    Args:
        regime_labels: Array of regime labels
        window: Window for stability calculation

    Returns:
        Array of stability scores
    """
    if len(regime_labels) < window:
        logger.warning(f"Not enough values for regime stability with window {window}")
        return np.zeros_like(regime_labels)

    # Initialize stability scores
    stability = np.zeros_like(regime_labels, dtype=float)
    stability[:window - 1] = np.nan  # First window-1 values are undefined

    for i in range(window - 1, len(regime_labels)):
        # Get window of regime labels
        window_labels = regime_labels[i - window + 1:i + 1]

        # Count frequency of most common regime
        unique, counts = np.unique(window_labels, return_counts=True)
        most_common_count = np.max(counts)

        # Calculate stability as percentage of most common regime
        stability[i] = most_common_count / window

    return stability


def create_return_table(returns_dict: Dict[str, PriceType],
                        freq: str = 'M') -> pd.DataFrame:
    """
    Create a nicely formatted return table for multiple assets/strategies.

    Args:
        returns_dict: Dictionary of return series (key: asset name, value: returns)
        freq: Resampling frequency ('D' for daily, 'W' for weekly, 'M' for monthly, etc.)

    Returns:
        DataFrame with return table
    """
    # Convert all returns to numpy arrays
    for key in returns_dict:
        if isinstance(returns_dict[key], list):
            returns_dict[key] = np.array(returns_dict[key])

    # Create DataFrames for each asset
    dfs = {}
    for asset_name, returns in returns_dict.items():
        if len(returns) == 0:
            logger.warning(f"Empty return series for {asset_name}")
            continue

        # Create DataFrame
        df = pd.DataFrame({'return': returns})

        # Convert to log returns if needed for compounding
        df['log_return'] = np.log(1 + df['return'])

        # Add dates if they don't exist
        if not isinstance(df.index, pd.DatetimeIndex):
            # Create a date range ending today
            end_date = datetime.now()
            start_date = end_date - timedelta(days=len(returns))
            df.index = pd.date_range(start=start_date, periods=len(returns), freq='D')

        dfs[asset_name] = df

    if not dfs:
        return pd.DataFrame()

    # Combine all DataFrames
    result_df = pd.DataFrame()

    for asset_name, df in dfs.items():
        # Resample to desired frequency
        resampled = df.resample(freq).apply(lambda x: (1 + x).prod() - 1)

        # Add to result DataFrame
        result_df[asset_name] = resampled['return']

    # Calculate statistics
    stats = {}

    # Calculate annualized return
    periods_per_year = {
        'D': 252,  # Trading days
        'W': 52,  # Weeks
        'M': 12,  # Months
        'Q': 4,  # Quarters
        'Y': 1  # Years
    }

    if freq in periods_per_year:
        ann_factor = periods_per_year[freq]
        annual_returns = (1 + result_df).mean() ** ann_factor - 1
        stats['Annual Return'] = annual_returns

    # Calculate other statistics
    stats['Mean'] = result_df.mean()
    stats['Std Dev'] = result_df.std()
    stats['Min'] = result_df.min()
    stats['Max'] = result_df.max()
    stats['Win Rate'] = (result_df > 0).mean()

    # Calculate drawdowns
    for asset_name in result_df.columns:
        # Calculate cumulative returns
        cumulative = (1 + result_df[asset_name]).cumprod()
        # Calculate running maximum
        running_max = cumulative.cummax()
        # Calculate drawdowns
        drawdowns = (cumulative / running_max) - 1
        # Calculate maximum drawdown
        stats.setdefault('Max Drawdown', {})[asset_name] = drawdowns.min()

    # Combine statistics into DataFrame
    stats_df = pd.DataFrame(stats)

    # Format percentages
    stats_df = stats_df * 100

    # Transpose so assets are columns
    stats_df = stats_df.T

    # Return both the return table and statistics
    return {
        'returns': result_df,
        'statistics': stats_df
    }


def calculate_market_impact(size: float, aum: float, adv: float,
                            price: float, volatility: float,
                            model: str = 'square_root') -> float:
    """
    Estimate market impact of a trade.

    Args:
        size: Trade size in units
        aum: Assets under management
        adv: Average daily volume
        price: Current price
        volatility: Daily volatility
        model: Market impact model ('square_root', 'linear', 'power_law')

    Returns:
        Estimated market impact in percentage points
    """
    # Calculate trade value
    trade_value = size * price

    # Calculate participation rate
    participation = trade_value / adv

    # Bound participation at reasonable levels
    participation = min(0.3, participation)

    if model == 'square_root':
        # Square root model: impact = sigma * sqrt(participation)
        impact = volatility * np.sqrt(participation)

    elif model == 'linear':
        # Linear model: impact = sigma * participation
        impact = volatility * participation

    elif model == 'power_law':
        # Power law model: impact = sigma * participation^0.6
        impact = volatility * (participation ** 0.6)

    else:
        logger.warning(f"Unknown market impact model: {model}, using square root")
        impact = volatility * np.sqrt(participation)

    return impact


def estimate_slippage(volume: float, volatility: float, bid_ask: float = None) -> float:
    """
    Estimate trading slippage based on volume and volatility.

    Args:
        volume: Average daily volume
        volatility: Daily volatility (decimal)
        bid_ask: Optional bid-ask spread (decimal)

    Returns:
        Estimated slippage in basis points (1bp = 0.01%)
    """
    # Base slippage based on volatility (higher volatility = higher slippage)
    base_slippage = volatility * 10  # Convert volatility to basis points

    # Volume adjustment (higher volume = lower slippage)
    volume_factor = 1.0

    if volume > 1000000:
        volume_factor = 0.7
    elif volume > 100000:
        volume_factor = 0.85
    elif volume < 10000:
        volume_factor = 1.2

    # Add bid-ask spread if provided
    spread_component = 0
    if bid_ask is not None:
        spread_component = bid_ask * 100 / 2  # Convert to basis points and half spread

    # Calculate total slippage
    slippage = (base_slippage * volume_factor) + spread_component

    return slippage


def order_size_constraints(portfolio_value: float, price: float,
                           volatility: float, max_pct_aum: float = 0.05,
                           max_days_volume: float = 0.1,
                           adv: float = None) -> Dict[str, float]:
    """
    Calculate order size constraints based on portfolio constraints.

    Args:
        portfolio_value: Portfolio value
        price: Current price
        volatility: Daily volatility
        max_pct_aum: Maximum percentage of AUM per position
        max_days_volume: Maximum days of volume to trade
        adv: Average daily volume (if None, this constraint is ignored)

    Returns:
        Dictionary with order constraints
    """
    # AUM constraint
    aum_limit = portfolio_value * max_pct_aum
    max_shares_aum = aum_limit / price

    # Volume constraint
    max_shares_volume = float('inf')
    if adv is not None and adv > 0:
        max_shares_volume = adv * max_days_volume

    # Risk-based position sizing
    # For a 2% risk with volatility given as decimal
    risk_budget = portfolio_value * 0.02
    max_shares_risk = risk_budget / (price * volatility)

    # Return all constraints
    return {
        'aum_constraint': max_shares_aum,
        'volume_constraint': max_shares_volume,
        'risk_constraint': max_shares_risk,
        'max_shares': min(max_shares_aum, max_shares_volume, max_shares_risk)
    }


def round_order_size(size: float, min_size: float = 1.0,
                     lot_size: float = 1.0) -> float:
    """
    Round order size to valid exchange constraints.

    Args:
        size: Desired order size
        min_size: Minimum order size
        lot_size: Lot size (order sizes must be multiples of this)

    Returns:
        Rounded order size
    """
    # Check minimum size
    if size < min_size:
        return 0.0

    # Round to lot size
    rounded = np.floor(size / lot_size) * lot_size

    # Check minimum size again after rounding
    if rounded < min_size:
        return 0.0

    return rounded


def map_indicator_to_signal(indicator: PriceType, thresholds: Dict[str, float]) -> np.ndarray:
    """
    Map an indicator to trading signals based on thresholds.

    Args:
        indicator: Indicator values
        thresholds: Dictionary of thresholds (e.g., {'buy': 70, 'sell': 30})

    Returns:
        Array of signals (1 for buy, -1 for sell, 0 for no action)
    """
    if isinstance(indicator, list):
        indicator = np.array(indicator)

    if len(indicator) == 0:
        return np.array([])

    signals = np.zeros_like(indicator)

    # Apply buy threshold
    if 'buy' in thresholds:
        signals[indicator >= thresholds['buy']] = 1

    # Apply sell threshold
    if 'sell' in thresholds:
        signals[indicator <= thresholds['sell']] = -1

    return signals

