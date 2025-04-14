"""
Mathematical Utilities Module

This module provides mathematical functions and utilities for the ML-powered trading system.
It includes functions for statistical analysis, time series processing, risk calculations,
and optimization routines commonly used in quantitative finance.

Dependencies:
- numpy
- scipy
- pandas
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize, signal
from typing import Dict, List, Tuple, Union, Optional, Callable, Any
import logging
from functools import wraps
import warnings

# Set up logging
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")


def safe_computation(default_value=np.nan):
    """
    Decorator for safe computation of mathematical functions.
    Catches numerical exceptions and returns a default value.

    Args:
        default_value: Value to return if computation fails

    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (
                ValueError, TypeError, ZeroDivisionError,
                np.linalg.LinAlgError, OverflowError, RuntimeWarning
            ) as e:
                logger.debug(f"Math error in {func.__name__}: {str(e)}")
                return default_value
        return wrapper
    return decorator


# ---- Basic Statistical Functions ----

@safe_computation(default_value=np.nan)
def winsorize(x: np.ndarray, limits: Tuple[float, float] = (0.05, 0.05)) -> np.ndarray:
    """
    Winsorize a data series (limit extreme values).

    Args:
        x: Input array
        limits: Tuple of (lower, upper) percentile bounds for winsorization

    Returns:
        Winsorized array
    """
    if len(x) < 3:
        return x

    lower_limit, upper_limit = limits
    lower_bound = np.percentile(x, lower_limit * 100)
    upper_bound = np.percentile(x, (1 - upper_limit) * 100)

    return np.clip(x, lower_bound, upper_bound)


@safe_computation(default_value=np.nan)
def robust_mean(x: np.ndarray, trim_pct: float = 0.1) -> float:
    """
    Calculate robust mean by trimming outliers.

    Args:
        x: Input array
        trim_pct: Percentage of data to trim from each end

    Returns:
        Robust mean value
    """
    if len(x) < 3:
        return np.mean(x)

    return stats.trim_mean(x, trim_pct)


@safe_computation(default_value=np.nan)
def robust_std(x: np.ndarray, q_range: float = 95.0) -> float:
    """
    Calculate robust standard deviation using interquartile range.

    Args:
        x: Input array
        q_range: Quantile range to use (default: 95.0 for 2.5% to 97.5%)

    Returns:
        Robust standard deviation
    """
    if len(x) < 3:
        return np.std(x)

    q_low = (100 - q_range) / 2 / 100
    q_high = 1 - q_low
    q_delta = np.percentile(x, q_high * 100) - np.percentile(x, q_low * 100)

    # Convert quantile range to approximate standard deviation
    # For normal distribution, 95% range is approximately 2 * 1.96 * sigma
    return q_delta / (2 * 1.96)


@safe_computation(default_value=np.nan)
def modified_z_score(x: np.ndarray) -> np.ndarray:
    """
    Calculate modified Z-scores using median absolute deviation.
    More robust to outliers than traditional Z-scores.

    Args:
        x: Input array

    Returns:
        Array of modified Z-scores
    """
    if len(x) < 3:
        return np.zeros_like(x)

    median = np.median(x)
    mad = np.median(np.abs(x - median))

    # Scale factor for normal distribution
    mad_scaling = 0.6745

    # Avoid division by zero
    if mad == 0:
        return np.zeros_like(x)

    return (x - median) / (mad_scaling * mad)


@safe_computation(default_value=np.nan)
def exponential_weighted_std(x: np.ndarray, alpha: float = 0.1) -> float:
    """
    Calculate exponentially weighted standard deviation.

    Args:
        x: Input array
        alpha: Smoothing factor (higher alpha gives more weight to recent observations)

    Returns:
        Exponentially weighted standard deviation
    """
    if len(x) < 3:
        return np.std(x)

    # Convert to pandas series for EWM calculations
    return pd.Series(x).ewm(alpha=alpha).std().iloc[-1]


# ---- Returns and Volatility Calculations ----

@safe_computation(default_value=np.zeros(1))
def calculate_returns(prices: np.ndarray, method: str = 'log') -> np.ndarray:
    """
    Calculate returns from price series.

    Args:
        prices: Array of prices
        method: Return calculation method ('log', 'simple', 'pct_change')

    Returns:
        Array of returns
    """
    if len(prices) < 2:
        return np.zeros(1)

    if method == 'log':
        return np.diff(np.log(prices))
    elif method == 'simple':
        return prices[1:] / prices[:-1] - 1
    elif method == 'pct_change':
        return pd.Series(prices).pct_change().values[1:]
    else:
        raise ValueError(f"Unknown return calculation method: {method}")


@safe_computation(default_value=0.0)
def realized_volatility(returns: np.ndarray, annualization_factor: float = 252) -> float:
    """
    Calculate realized volatility from returns.

    Args:
        returns: Array of returns
        annualization_factor: Factor to annualize volatility

    Returns:
        Annualized volatility
    """
    if len(returns) < 2:
        return 0.0

    # Remove NaN values
    clean_returns = returns[~np.isnan(returns)]

    if len(clean_returns) < 2:
        return 0.0

    return np.std(clean_returns) * np.sqrt(annualization_factor)


@safe_computation(default_value=0.0)
def parkinson_volatility(high: np.ndarray, low: np.ndarray, annualization_factor: float = 252) -> float:
    """
    Calculate Parkinson volatility estimator using high-low range.

    Args:
        high: Array of high prices
        low: Array of low prices
        annualization_factor: Factor to annualize volatility

    Returns:
        Annualized Parkinson volatility estimate
    """
    if len(high) < 1 or len(low) < 1:
        return 0.0

    n = len(high)
    log_hl = np.log(high / low)
    estimator = np.sum(log_hl ** 2) / (4 * n * np.log(2))

    return np.sqrt(estimator * annualization_factor)


@safe_computation(default_value=0.0)
def garman_klass_volatility(
    open_price: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    annualization_factor: float = 252
) -> float:
    """
    Calculate Garman-Klass volatility estimator.

    Args:
        open_price: Array of opening prices
        high: Array of high prices
        low: Array of low prices
        close: Array of closing prices
        annualization_factor: Factor to annualize volatility

    Returns:
        Annualized Garman-Klass volatility estimate
    """
    if len(high) < 1 or len(low) < 1 or len(open_price) < 1 or len(close) < 1:
        return 0.0

    n = len(high)
    log_hl = 0.5 * np.log(high / low) ** 2
    log_co = (2 * np.log(2) - 1) * np.log(close / open_price) ** 2

    estimator = np.sum(log_hl - log_co) / n

    return np.sqrt(estimator * annualization_factor)


@safe_computation(default_value=0.0)
def calculate_var(returns: np.ndarray, alpha: float = 0.05, method: str = 'historical') -> float:
    """
    Calculate Value at Risk (VaR).

    Args:
        returns: Array of returns
        alpha: Significance level (e.g., 0.05 for 95% confidence)
        method: Method to calculate VaR ('historical', 'parametric', 'cornish_fisher')

    Returns:
        Value at Risk estimate
    """
    if len(returns) < 3:
        return 0.0

    # Remove NaN values
    clean_returns = returns[~np.isnan(returns)]

    if len(clean_returns) < 3:
        return 0.0

    if method == 'historical':
        return np.percentile(clean_returns, alpha * 100)

    elif method == 'parametric':
        mu = np.mean(clean_returns)
        sigma = np.std(clean_returns)
        return mu + stats.norm.ppf(alpha) * sigma

    elif method == 'cornish_fisher':
        mu = np.mean(clean_returns)
        sigma = np.std(clean_returns)
        skew = stats.skew(clean_returns)
        kurt = stats.kurtosis(clean_returns)

        z_alpha = stats.norm.ppf(alpha)
        z_alpha_cf = z_alpha + (z_alpha**2 - 1) * skew / 6 + (z_alpha**3 - 3*z_alpha) * kurt / 24

        return mu + sigma * z_alpha_cf

    else:
        raise ValueError(f"Unknown VaR calculation method: {method}")


@safe_computation(default_value=0.0)
def calculate_cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.

    Args:
        returns: Array of returns
        alpha: Significance level (e.g., 0.05 for 95% confidence)

    Returns:
        Conditional Value at Risk estimate
    """
    if len(returns) < 3:
        return 0.0

    # Remove NaN values
    clean_returns = returns[~np.isnan(returns)]

    if len(clean_returns) < 3:
        return 0.0

    var = np.percentile(clean_returns, alpha * 100)
    return np.mean(clean_returns[clean_returns <= var])


# ---- Time Series Analysis ----

@safe_computation(default_value=(None, None))
def hurst_exponent(time_series: np.ndarray, max_lag: int = 20) -> Tuple[float, float]:
    """
    Calculate Hurst exponent to determine if a time series is mean-reverting,
    random walk, or trending.

    H < 0.5: Mean-reverting
    H = 0.5: Random walk (Brownian motion)
    H > 0.5: Trending/persistent

    Args:
        time_series: Input time series
        max_lag: Maximum lag for calculation

    Returns:
        Tuple of (Hurst exponent, R-squared of fit)
    """
    if len(time_series) < max_lag * 2:
        return None, None

    # Remove NaN values
    time_series = time_series[~np.isnan(time_series)]

    if len(time_series) < max_lag * 2:
        return None, None

    # Calculate range of cumulative deviation from mean
    lags = range(2, max_lag)
    tau = []; rs = []

    for lag in lags:
        # Subseries
        n = len(time_series)
        series = time_series.copy()
        max_blocks = int(n / lag)
        blocks = max_blocks * lag
        series = series[:blocks]

        # Reshape to (max_blocks, lag)
        series = series.reshape((max_blocks, lag))

        # Calculate range/std for each block
        rs_values = []
        for block in range(max_blocks):
            block_series = series[block]
            mean = np.mean(block_series)

            # Cumulative sum of deviations from mean
            cumsum = np.cumsum(block_series - mean)

            # Range and standard deviation
            r = np.max(cumsum) - np.min(cumsum)
            s = np.std(block_series)

            if s > 0:
                rs_values.append(r / s)

        # Average R/S values across blocks
        rs.append(np.mean(rs_values))
        tau.append(lag)

    if len(tau) < 2 or len(rs) < 2:
        return None, None

    # Convert to log-log space
    log_tau = np.log10(tau)
    log_rs = np.log10(rs)

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_tau, log_rs)

    return slope, r_value**2


@safe_computation(default_value=(None, None, None))
def half_life_mean_reversion(time_series: np.ndarray) -> Tuple[float, float, float]:
    """
    Estimate the half-life of mean reversion using AR(1) model.

    Args:
        time_series: Input time series

    Returns:
        Tuple of (half-life, lambda coefficient, t-statistic)
    """
    if len(time_series) < 4:
        return None, None, None

    # Remove NaN values
    time_series = time_series[~np.isnan(time_series)]

    if len(time_series) < 4:
        return None, None, None

    # Calculate lagged values and differences
    y = time_series[1:] - time_series[:-1]  # y(t) - y(t-1)
    x = time_series[:-1]                     # y(t-1)

    # Add constant for regression
    x = np.vstack([x, np.ones(len(x))]).T

    # OLS regression
    beta = np.linalg.lstsq(x, y, rcond=None)[0]

    # Calculate half-life
    lambda_coef = beta[0]

    if lambda_coef >= 0:
        # Not mean-reverting
        return np.inf, lambda_coef, 0

    half_life = -np.log(2) / lambda_coef

    # Calculate t-statistic
    y_pred = np.dot(x, beta)
    residuals = y - y_pred
    n = len(residuals)
    k = 2  # Number of params
    sigma_squared = np.sum(residuals**2) / (n - k)
    x_transpose_x_inv = np.linalg.inv(np.dot(x.T, x))
    var_beta = sigma_squared * x_transpose_x_inv[0, 0]
    t_stat = lambda_coef / np.sqrt(var_beta)

    return half_life, lambda_coef, t_stat


def detrend_time_series(time_series: np.ndarray, method: str = 'linear') -> np.ndarray:
    """
    Detrend a time series using various methods.

    Args:
        time_series: Input time series
        method: Detrending method ('linear', 'polynomial', 'ewma')

    Returns:
        Detrended time series
    """
    if len(time_series) < 3:
        return time_series

    # Remove NaN values
    clean_series = time_series[~np.isnan(time_series)]
    if len(clean_series) < 3:
        return time_series

    # Create index array
    x = np.arange(len(clean_series))

    if method == 'linear':
        # Linear detrending
        slope, intercept = np.polyfit(x, clean_series, 1)
        trend = intercept + slope * x

    elif method.startswith('polynomial'):
        # Extract degree from method string, default to 2
        try:
            degree = int(method.split('_')[1])
        except (IndexError, ValueError):
            degree = 2

        # Polynomial detrending
        coeffs = np.polyfit(x, clean_series, degree)
        trend = np.polyval(coeffs, x)

    elif method == 'ewma':
        # Exponential weighted moving average
        alpha = 0.05  # Smoothing factor
        trend = pd.Series(clean_series).ewm(alpha=alpha).mean().values

    elif method == 'hodrick_prescott':
        # Hodrick-Prescott filter
        from statsmodels.tsa.filters.hp_filter import hpfilter
        cycle, trend = hpfilter(clean_series, lamb=1600)  # Lambda for quarterly data

    else:
        raise ValueError(f"Unknown detrending method: {method}")

    # Return detrended series
    detrended = clean_series - trend

    # Create result array with same shape as input, preserving NaNs
    result = np.full_like(time_series, np.nan)
    non_nan_idx = ~np.isnan(time_series)
    result[non_nan_idx] = detrended

    return result


@safe_computation(default_value=(None, None))
def detect_outliers(time_series: np.ndarray, method: str = 'zscore', threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect outliers in a time series.

    Args:
        time_series: Input time series
        method: Detection method ('zscore', 'modified_zscore', 'iqr')
        threshold: Threshold for outlier detection

    Returns:
        Tuple of (binary mask of outliers, test statistic values)
    """
    if len(time_series) < 3:
        return np.zeros_like(time_series, dtype=bool), np.zeros_like(time_series)

    # Handle NaN values
    clean_series = time_series.copy()
    nan_mask = np.isnan(clean_series)

    if np.all(nan_mask):
        return np.zeros_like(time_series, dtype=bool), np.zeros_like(time_series)

    if method == 'zscore':
        # Z-score method
        z = np.zeros_like(clean_series)
        z[~nan_mask] = stats.zscore(clean_series[~nan_mask])
        outliers = np.abs(z) > threshold

    elif method == 'modified_zscore':
        # Modified Z-score using MAD
        z = np.zeros_like(clean_series)
        z[~nan_mask] = modified_z_score(clean_series[~nan_mask])
        outliers = np.abs(z) > threshold

    elif method == 'iqr':
        # Interquartile range method
        q1, q3 = np.percentile(clean_series[~nan_mask], [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        z = np.zeros_like(clean_series)
        outliers = np.zeros_like(clean_series, dtype=bool)

        for i in range(len(clean_series)):
            if nan_mask[i]:
                outliers[i] = False
                z[i] = 0
            else:
                outliers[i] = (clean_series[i] < lower_bound) or (clean_series[i] > upper_bound)
                # Calculate equivalent z-score for consistency in return values
                if outliers[i]:
                    if clean_series[i] < lower_bound:
                        z[i] = (clean_series[i] - q1) / iqr * 1.35  # 1.35 is scaling for normal distribution
                    else:
                        z[i] = (clean_series[i] - q3) / iqr * 1.35
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

    return outliers, z


@safe_computation(default_value=None)
def lowpass_filter(time_series: np.ndarray, cutoff: float, fs: float = 1.0, order: int = 5) -> np.ndarray:
    """
    Apply a lowpass filter to a time series.

    Args:
        time_series: Input time series
        cutoff: Cutoff frequency
        fs: Sampling frequency
        order: Filter order

    Returns:
        Filtered time series
    """
    if len(time_series) < 2 * order:
        return time_series

    # Handle NaN values
    nan_mask = np.isnan(time_series)
    has_nans = np.any(nan_mask)

    if has_nans:
        # Interpolate NaNs for filtering
        clean_series = pd.Series(time_series).interpolate(method='linear').fillna(
            method='ffill').fillna(method='bfill').values
    else:
        clean_series = time_series

    # Calculate Nyquist frequency
    nyq = 0.5 * fs

    # Normalize cutoff frequency
    normal_cutoff = cutoff / nyq

    # Design Butterworth filter
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)

    # Apply filter (with zero-phase to avoid time shifts)
    filtered = signal.filtfilt(b, a, clean_series)

    # Restore NaN values
    if has_nans:
        filtered_with_nans = np.copy(filtered)
        filtered_with_nans[nan_mask] = np.nan
        return filtered_with_nans

    return filtered


# ---- Correlation and Covariance Functions ----

@safe_computation(default_value=None)
def robust_correlation(x: np.ndarray, y: np.ndarray, method: str = 'spearman') -> float:
    """
    Calculate robust correlation between two series.

    Args:
        x: First input array
        y: Second input array
        method: Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        Correlation coefficient
    """
    if len(x) < 3 or len(y) < 3:
        return np.nan

    # Handle NaN values
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) < 3:
        return np.nan

    x_clean = x[mask]
    y_clean = y[mask]

    if method == 'pearson':
        return stats.pearsonr(x_clean, y_clean)[0]
    elif method == 'spearman':
        return stats.spearmanr(x_clean, y_clean)[0]
    elif method == 'kendall':
        return stats.kendalltau(x_clean, y_clean)[0]
    else:
        raise ValueError(f"Unknown correlation method: {method}")


@safe_computation(default_value=None)
def correlation_significance(x: np.ndarray, y: np.ndarray, method: str = 'spearman') -> Tuple[float, float]:
    """
    Calculate correlation and its statistical significance.

    Args:
        x: First input array
        y: Second input array
        method: Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        Tuple of (correlation coefficient, p-value)
    """
    if len(x) < 3 or len(y) < 3:
        return np.nan, np.nan

    # Handle NaN values
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) < 3:
        return np.nan, np.nan

    x_clean = x[mask]
    y_clean = y[mask]

    if method == 'pearson':
        corr, p_value = stats.pearsonr(x_clean, y_clean)
    elif method == 'spearman':
        corr, p_value = stats.spearmanr(x_clean, y_clean)
    elif method == 'kendall':
        corr, p_value = stats.kendalltau(x_clean, y_clean)
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    return corr, p_value


@safe_computation(default_value=None)
def ewma_correlation(x: np.ndarray, y: np.ndarray, alpha: float = 0.1) -> float:
    """
    Calculate exponentially weighted moving average correlation.

    Args:
        x: First input array
        y: Second input array
        alpha: Smoothing factor

    Returns:
        EWMA correlation coefficient
    """
    if len(x) < 3 or len(y) < 3:
        return np.nan

    # Create pandas series for EWM calculations
    df = pd.DataFrame({'x': x, 'y': y})

    # Calculate EWMA correlation
    ewm_corr = df['x'].ewm(alpha=alpha).corr(df['y']).iloc[-1]

    return ewm_corr


@safe_computation(default_value=None)
def shrink_covariance(returns: np.ndarray, shrinkage: float = None) -> np.ndarray:
    """
    Compute a shrinkage estimate of the covariance matrix.

    Args:
        returns: Array of returns (variables as columns)
        shrinkage: Shrinkage intensity (if None, it's estimated)

    Returns:
        Shrinkage estimate of the covariance matrix
    """
    if len(returns) < 3:
        return np.cov(returns.T) if returns.ndim > 1 else np.var(returns)

    # Handle 1D arrays
    if returns.ndim == 1:
        return np.var(returns)

    n, p = returns.shape

    # Sample covariance
    sample_cov = np.cov(returns.T, bias=True)  # Use biased estimator for shrinkage

    # Target (diagonal matrix with sample variances)
    target = np.diag(np.diag(sample_cov))

    if shrinkage is not None and 0 <= shrinkage <= 1:
        # Use provided shrinkage parameter
        shrunk_cov = (1 - shrinkage) * sample_cov + shrinkage * target
        return shrunk_cov

    # Estimate optimal shrinkage (Ledoit-Wolf method simplified)
    # Computational shortcuts for Ledoit-Wolf estimate

    # Frobenius norm of the difference between sample and target
    var_target = np.sum(np.diag(sample_cov)**2)
    var_sample = np.sum(sample_cov**2)

    # Estimate optimal shrinkage intensity
    alpha = (var_target - var_sample / n) / ((n-1) * var_sample)

    # Ensure shrinkage is in [0, 1]
    shrinkage = max(0, min(1, alpha))

    # Apply shrinkage
    shrunk_cov = (1 - shrinkage) * sample_cov + shrinkage * target

    return shrunk_cov


# ---- Financial Metrics ----

@safe_computation(default_value=0.0)
def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252
) -> float:
    """
    Calculate the Sharpe ratio.

    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annualized)
        annualization_factor: Annualization factor based on return frequency

    Returns:
        Sharpe ratio
    """
    if len(returns) < 3:
        return 0.0

    # Handle NaN values
    clean_returns = returns[~np.isnan(returns)]
    if len(clean_returns) < 3:
        return 0.0

    # Convert annual risk-free rate to match return frequency
    period_risk_free = (1 + risk_free_rate) ** (1 / annualization_factor) - 1

    # Calculate excess returns
    excess_returns = clean_returns - period_risk_free

    # Calculate Sharpe ratio
    mean_excess_return = np.mean(excess_returns)
    return_volatility = np.std(clean_returns, ddof=1)

    if return_volatility == 0:
        return 0.0

    sharpe = mean_excess_return / return_volatility * np.sqrt(annualization_factor)

    return sharpe


@safe_computation(default_value=0.0)
def sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252,
    target_return: float = 0.0
) -> float:
    """
    Calculate the Sortino ratio (using downside deviation).

    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (annualized)
        annualization_factor: Annualization factor based on return frequency
        target_return: Minimum acceptable return

    Returns:
        Sortino ratio
    """
    if len(returns) < 3:
        return 0.0

    # Handle NaN values
    clean_returns = returns[~np.isnan(returns)]
    if len(clean_returns) < 3:
        return 0.0

    # Convert annual risk-free rate to match return frequency
    period_risk_free = (1 + risk_free_rate) ** (1 / annualization_factor) - 1

    # Calculate excess returns
    excess_returns = clean_returns - period_risk_free
    mean_excess_return = np.mean(excess_returns)

    # Calculate downside deviation (only negative returns relative to target)
    downside_returns = clean_returns[clean_returns < target_return]

    if len(downside_returns) == 0:
        # No downside returns, avoid division by zero
        return np.inf if mean_excess_return > 0 else 0.0

    downside_deviation = np.sqrt(np.mean((downside_returns - target_return) ** 2))

    if downside_deviation == 0:
        return np.inf if mean_excess_return > 0 else 0.0

    sortino = mean_excess_return / downside_deviation * np.sqrt(annualization_factor)

    return sortino


@safe_computation(default_value=0.0)
def calmar_ratio(returns: np.ndarray, annualization_factor: float = 252, window: int = None) -> float:
    """
    Calculate the Calmar ratio (return / maximum drawdown).

    Args:
        returns: Array of returns
        annualization_factor: Annualization factor based on return frequency
        window: Rolling window for max drawdown calculation (None = entire series)

    Returns:
        Calmar ratio
    """
    if len(returns) < 3:
        return 0.0

    # Handle NaN values
    clean_returns = returns[~np.isnan(returns)]
    if len(clean_returns) < 3:
        return 0.0

    # Calculate annualized return
    annualized_return = np.mean(clean_returns) * annualization_factor

    # Calculate maximum drawdown
    cumulative = np.cumprod(1 + clean_returns)
    if window is not None and window < len(cumulative):
        # Use rolling window
        rolling_max = pd.Series(cumulative).rolling(window, min_periods=1).max().values
    else:
        # Use expanding window
        rolling_max = np.maximum.accumulate(cumulative)

    drawdowns = (cumulative / rolling_max) - 1
    max_drawdown = abs(np.min(drawdowns))

    if max_drawdown == 0:
        return np.inf if annualized_return > 0 else 0.0

    calmar = annualized_return / max_drawdown

    return calmar


@safe_computation(default_value=(0.0, 0.0))
def calculate_drawdowns(returns: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Calculate the drawdown series and maximum drawdown.

    Args:
        returns: Array of returns

    Returns:
        Tuple of (drawdown series, maximum drawdown)
    """
    if len(returns) < 2:
        return np.zeros_like(returns), 0.0

    # Handle NaN values
    clean_returns = returns[~np.isnan(returns)]
    if len(clean_returns) < 2:
        return np.zeros_like(returns), 0.0

    # Calculate cumulative returns
    cumulative = np.cumprod(1 + clean_returns)

    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative)

    # Calculate drawdowns
    drawdowns = (cumulative / running_max) - 1
    max_drawdown = abs(np.min(drawdowns))

    # Create result array matching original shape
    result = np.zeros_like(returns)
    result[~np.isnan(returns)] = drawdowns

    return result, max_drawdown


@safe_computation(default_value=0.0)
def information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray, annualization_factor: float = 252) -> float:
    """
    Calculate the Information Ratio (active return / tracking error).

    Args:
        returns: Array of strategy returns
        benchmark_returns: Array of benchmark returns
        annualization_factor: Annualization factor based on return frequency

    Returns:
        Information ratio
    """
    if len(returns) < 3 or len(benchmark_returns) < 3:
        return 0.0

    # Align arrays
    min_length = min(len(returns), len(benchmark_returns))
    r = returns[-min_length:]
    b = benchmark_returns[-min_length:]

    # Handle NaN values
    mask = ~np.isnan(r) & ~np.isnan(b)
    if np.sum(mask) < 3:
        return 0.0

    r = r[mask]
    b = b[mask]

    # Calculate active returns
    active_returns = r - b

    # Calculate IR components
    active_return_mean = np.mean(active_returns)
    tracking_error = np.std(active_returns, ddof=1)

    if tracking_error == 0:
        return 0.0

    ir = active_return_mean / tracking_error * np.sqrt(annualization_factor)

    return ir


@safe_computation(default_value=0.0)
def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    Calculate the Omega ratio (probability weighted ratio of gains vs. losses).

    Args:
        returns: Array of returns
        threshold: Minimum acceptable return

    Returns:
        Omega ratio
    """
    if len(returns) < 3:
        return 0.0

    # Handle NaN values
    clean_returns = returns[~np.isnan(returns)]
    if len(clean_returns) < 3:
        return 0.0

    # Separate returns into gains and losses relative to threshold
    gains = clean_returns[clean_returns > threshold] - threshold
    losses = threshold - clean_returns[clean_returns < threshold]

    if len(losses) == 0 or np.sum(losses) == 0:
        return np.inf if len(gains) > 0 else 1.0

    # Calculate Omega ratio
    omega = np.sum(gains) / np.sum(losses)

    return omega


# ---- Optimization Functions ----

def minimize_portfolio_variance(
    cov_matrix: np.ndarray,
    constraints: Optional[List[Dict]] = None,
    bounds: Optional[List[Tuple[float, float]]] = None
) -> np.ndarray:
    """
    Find the minimum variance portfolio weights.

    Args:
        cov_matrix: Covariance matrix of asset returns
        constraints: List of constraint dictionaries for scipy.optimize.minimize
        bounds: List of (min, max) tuples for each asset weight

    Returns:
        Array of optimal weights
    """
    n = cov_matrix.shape[0]

    # Set default bounds if not provided
    if bounds is None:
        bounds = [(0, 1) for _ in range(n)]  # Default: long only, no leverage

    # Set default constraints if not provided
    if constraints is None:
        # Sum of weights = 1
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

    # Define the objective function (portfolio variance)
    def portfolio_variance(weights):
        return weights @ cov_matrix @ weights

    # Initial guess: equal weight
    initial_weights = np.ones(n) / n

    # Perform optimization
    result = optimize.minimize(
        portfolio_variance,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        logger.warning(f"Portfolio optimization failed: {result.message}")
        return initial_weights

    return result.x


def maximize_sharpe_ratio(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.0,
    constraints: Optional[List[Dict]] = None,
    bounds: Optional[List[Tuple[float, float]]] = None
) -> np.ndarray:
    """
    Find the portfolio weights that maximize the Sharpe ratio.

    Args:
        expected_returns: Array of expected returns for each asset
        cov_matrix: Covariance matrix of asset returns
        risk_free_rate: Risk-free rate
        constraints: List of constraint dictionaries for scipy.optimize.minimize
        bounds: List of (min, max) tuples for each asset weight

    Returns:
        Array of optimal weights
    """
    n = len(expected_returns)

    # Set default bounds if not provided
    if bounds is None:
        bounds = [(0, 1) for _ in range(n)]  # Default: long only, no leverage

    # Set default constraints if not provided
    if constraints is None:
        # Sum of weights = 1
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

    # Define the objective function (negative Sharpe ratio)
    def neg_sharpe_ratio(weights):
        port_return = np.sum(weights * expected_returns)
        port_volatility = np.sqrt(weights @ cov_matrix @ weights)

        # Handle division by zero
        if port_volatility == 0:
            return -np.sign(port_return - risk_free_rate) * np.inf

        return -(port_return - risk_free_rate) / port_volatility

    # Initial guess: equal weight
    initial_weights = np.ones(n) / n

    # Perform optimization
    result = optimize.minimize(
        neg_sharpe_ratio,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        logger.warning(f"Portfolio optimization failed: {result.message}")
        return initial_weights

    return result.x


def risk_budgeting_weights(
    cov_matrix: np.ndarray,
    risk_budget: Optional[np.ndarray] = None,
    constraints: Optional[List[Dict]] = None,
    bounds: Optional[List[Tuple[float, float]]] = None
) -> np.ndarray:
    """
    Find portfolio weights that allocate risk according to specified risk budget.

    Args:
        cov_matrix: Covariance matrix of asset returns
        risk_budget: Target risk contribution for each asset (defaults to equal)
        constraints: List of constraint dictionaries for scipy.optimize.minimize
        bounds: List of (min, max) tuples for each asset weight

    Returns:
        Array of optimal weights
    """
    n = cov_matrix.shape[0]

    # Set default risk budget if not provided (equal risk)
    if risk_budget is None:
        risk_budget = np.ones(n) / n
    else:
        # Normalize risk budget
        risk_budget = risk_budget / np.sum(risk_budget)

    # Set default bounds if not provided
    if bounds is None:
        bounds = [(0, 1) for _ in range(n)]  # Default: long only, no leverage

    # Set default constraints if not provided
    if constraints is None:
        # Sum of weights = 1
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

    # Define the objective function (risk budget error)
    def risk_budget_error(weights):
        portfolio_risk = np.sqrt(weights @ cov_matrix @ weights)
        asset_contrib = weights * (cov_matrix @ weights) / portfolio_risk

        # Calculate error vs. target risk budget
        risk_contrib = asset_contrib / np.sum(asset_contrib)
        error = np.sum((risk_contrib - risk_budget) ** 2)

        return error

    # Initial guess: inverse volatility weighting
    vol = np.sqrt(np.diag(cov_matrix))
    initial_weights = (1 / vol) / np.sum(1 / vol)

    # Perform optimization
    result = optimize.minimize(
        risk_budget_error,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        logger.warning(f"Risk budgeting optimization failed: {result.message}")
        return initial_weights

    return result.x


# ---- Machine Learning Utilities ----

def rolling_window_dataset(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int,
    forecast_horizon: int = 1,
    step: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create rolling window dataset for time series forecasting.

    Args:
        X: Input features (2D array: samples x features)
        y: Target values (1D array)
        window_size: Size of the lookback window
        forecast_horizon: How far ahead to predict
        step: Step size between windows

    Returns:
        Tuple of (windowed X data, windowed y data)
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")

    n_samples = len(X)
    n_features = X.shape[1] if X.ndim > 1 else 1

    # Calculate number of valid windows
    n_windows = max(0, (n_samples - window_size - forecast_horizon + 1) // step)

    # Create output arrays
    if X.ndim > 1:
        X_windows = np.zeros((n_windows, window_size, n_features))
    else:
        X_windows = np.zeros((n_windows, window_size))

    y_targets = np.zeros(n_windows)

    # Fill the windows
    for i in range(n_windows):
        start_idx = i * step
        end_idx = start_idx + window_size
        target_idx = end_idx + forecast_horizon - 1

        X_windows[i] = X[start_idx:end_idx]
        y_targets[i] = y[target_idx]

    return X_windows, y_targets


def train_test_split_time_series(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    validation_size: Optional[float] = None
) -> Tuple:
    """
    Split time series data into train, test, and optional validation sets.
    Respects temporal order of data.

    Args:
        X: Input features
        y: Target values
        test_size: Proportion of data for testing
        validation_size: Optional proportion of data for validation

    Returns:
        Tuple of splits (X_train, X_test, y_train, y_test) or
        (X_train, X_val, X_test, y_train, y_val, y_test) if validation_size is provided
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    n = len(X)

    if validation_size is None:
        # Split into train and test only
        test_idx = int(n * (1 - test_size))
        X_train, X_test = X[:test_idx], X[test_idx:]
        y_train, y_test = y[:test_idx], y[test_idx:]

        return X_train, X_test, y_train, y_test
    else:
        # Split into train, validation, and test
        test_idx = int(n * (1 - test_size))
        val_idx = int(test_idx * (1 - validation_size))

        X_train = X[:val_idx]
        X_val = X[val_idx:test_idx]
        X_test = X[test_idx:]

        y_train = y[:val_idx]
        y_val = y[val_idx:test_idx]
        y_test = y[test_idx:]

        return X_train, X_val, X_test, y_train, y_val, y_test


def purged_cross_validation_splits(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    embargo_size: float = 0.0,
    purge_overlap: bool = True
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create purged cross-validation splits for financial time series.

    Args:
        X: Input features
        y: Target values
        n_splits: Number of CV splits
        embargo_size: Proportion of data to embargo after test set
        purge_overlap: Whether to purge overlapping samples

    Returns:
        List of (train_idx, test_idx) tuples
    """
    n_samples = len(X)
    indices = np.arange(n_samples)

    # Calculate test size
    test_size = n_samples // n_splits
    embargo_samples = int(test_size * embargo_size)

    splits = []
    for i in range(n_splits):
        # Test indices
        test_start = i * test_size
        test_end = min((i + 1) * test_size, n_samples)
        test_idx = indices[test_start:test_end]

        # Train indices (all data except test and embargo)
        train_idx = indices[:test_start]

        # Add embargo period after test
        embargo_end = min(test_end + embargo_samples, n_samples)

        # Add remaining data after embargo
        train_idx = np.concatenate([train_idx, indices[embargo_end:]])

        splits.append((train_idx, test_idx))

    return splits


def feature_importance_permutation(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    metric: Callable = None,
    n_repeats: int = 10,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate feature importance using permutation method.

    Args:
        model: Fitted model with predict method
        X: Input features
        y: Target values
        metric: Scoring function (defaults to MSE for regression)
        n_repeats: Number of permutation rounds
        random_state: Random seed

    Returns:
        Tuple of (mean importance scores, standard deviations)
    """
    # Default metric is negative MSE for regression
    if metric is None:
        metric = lambda y_true, y_pred: -np.mean((y_true - y_pred) ** 2)

    # Get baseline score
    baseline_prediction = model.predict(X)
    baseline_score = metric(y, baseline_prediction)

    # Initialize
    rng = np.random.RandomState(random_state)
    n_features = X.shape[1]
    importance = np.zeros((n_repeats, n_features))

    # Permute each feature n_repeats times
    for r in range(n_repeats):
        for i in range(n_features):
            # Create shuffled dataset
            X_permuted = X.copy()
            X_permuted[:, i] = rng.permutation(X[:, i])

            # Score with permuted feature
            permuted_prediction = model.predict(X_permuted)
            permuted_score = metric(y, permuted_prediction)

            # Importance is decrease in performance
            importance[r, i] = baseline_score - permuted_score

    return np.mean(importance, axis=0), np.std(importance, axis=0)


# ---- Regime Change Detection ----

@safe_computation(default_value=None)
def detect_cpt_cusum(
    time_series: np.ndarray,
    threshold: float = 1.0,
    drift: float = 0.0
) -> List[int]:
    """
    Detect change points using CUSUM (cumulative sum) method.

    Args:
        time_series: Input time series
        threshold: Detection threshold
        drift: Drift parameter to prevent false alarms

    Returns:
        List of change point indices
    """
    if len(time_series) < 10:
        return []

    # Handle NaN values
    clean_series = time_series[~np.isnan(time_series)]
    if len(clean_series) < 10:
        return []

    # Standardize the time series
    ts_std = (clean_series - np.mean(clean_series)) / np.std(clean_series)

    # Initialize
    s_pos = np.zeros_like(ts_std)
    s_neg = np.zeros_like(ts_std)
    change_points = []

    # CUSUM algorithm
    for i in range(1, len(ts_std)):
        # Update CUSUM statistics
        s_pos[i] = max(0, s_pos[i-1] + ts_std[i] - drift)
        s_neg[i] = max(0, s_neg[i-1] - ts_std[i] - drift)

        # Check for change point
        if s_pos[i] > threshold or s_neg[i] > threshold:
            change_points.append(i)
            s_pos[i] = 0
            s_neg[i] = 0

    return change_points


@safe_computation(default_value=None)
def detect_cpt_pelt(
    time_series: np.ndarray,
    penalty: float = 10.0,
    min_segment_length: int = 10
) -> List[int]:
    """
    Detect change points using PELT (Pruned Exact Linear Time) method.

    Args:
        time_series: Input time series
        penalty: Penalty value for adding change points
        min_segment_length: Minimum segment length

    Returns:
        List of change point indices
    """
    try:
        from ruptures import Pelt, costs
    except ImportError:
        logger.warning("ruptures package not installed. Install with: pip install ruptures")
        return []

    if len(time_series) < 2 * min_segment_length:
        return []

    # Handle NaN values
    clean_series = np.copy(time_series)
    nan_indices = np.isnan(clean_series)

    if np.any(nan_indices):
        # Interpolate NaNs
        valid_indices = np.where(~nan_indices)[0]
        if len(valid_indices) < 2 * min_segment_length:
            return []

        nan_positions = np.where(nan_indices)[0]
        for i in nan_positions:
            # Find nearest valid values
            prev_valid = valid_indices[valid_indices < i]
            next_valid = valid_indices[valid_indices > i]

            if len(prev_valid) == 0:
                # Use next valid
                clean_series[i] = clean_series[next_valid[0]]
            elif len(next_valid) == 0:
                # Use previous valid
                clean_series[i] = clean_series[prev_valid[-1]]
            else:
                # Linear interpolation
                prev_idx = prev_valid[-1]
                next_idx = next_valid[0]
                prev_val = clean_series[prev_idx]
                next_val = clean_series[next_idx]

                weight = (i - prev_idx) / (next_idx - prev_idx)
                clean_series[i] = prev_val * (1 - weight) + next_val * weight

    # Apply PELT algorithm
    model = Pelt(model=costs.normal_mean).fit(np.array(clean_series).reshape(-1, 1))
    change_points = model.predict(pen=penalty, min_size=min_segment_length)

    # PELT returns end indices, remove the last one which is just the length
    if change_points and change_points[-1] == len(clean_series):
        change_points = change_points[:-1]

    return change_points


# ---- Directional Movement and Trend Strength ----

@safe_computation(default_value=None)
def directional_movement_index(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    window: int = 14
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Directional Movement Index (DMI): ADX, +DI, -DI.

    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        window: Calculation window

    Returns:
        Tuple of (ADX, positive DI, negative DI)
    """
    if len(high) < window + 1 or len(low) < window + 1 or len(close) < window + 1:
        return None, None, None

    # Calculate True Range
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)

    # Calculate Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]

    # Positive and Negative DM
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Smooth TR, +DM, -DM with Wilder's smoothing
    tr_smooth = np.zeros_like(tr)
    pos_dm_smooth = np.zeros_like(pos_dm)
    neg_dm_smooth = np.zeros_like(neg_dm)

    # Initial values
    tr_smooth[0] = tr[0]
    pos_dm_smooth[0] = pos_dm[0]
    neg_dm_smooth[0] = neg_dm[0]

    # Apply smoothing
    for i in range(1, len(tr)):
        tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / window) + tr[i]
        pos_dm_smooth[i] = pos_dm_smooth[i-1] - (pos_dm_smooth[i-1] / window) + pos_dm[i]
        neg_dm_smooth[i] = neg_dm_smooth[i-1] - (neg_dm_smooth[i-1] / window) + neg_dm[i]

    # Calculate +DI and -DI
    pos_di = 100 * pos_dm_smooth / tr_smooth
    neg_di = 100 * neg_dm_smooth / tr_smooth

    # Calculate DX
    dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)

    # Calculate ADX with smoothing
    adx = np.zeros_like(dx)
    adx[0] = dx[0]

    for i in range(1, len(dx)):
        adx[i] = (adx[i-1] * (window - 1) + dx[i]) / window

    # Pad beginning to match input length
    pad = np.full(1, np.nan)
    adx_result = np.concatenate([pad, adx])
    pos_di_result = np.concatenate([pad, pos_di])
    neg_di_result = np.concatenate([pad, neg_di])

    return adx_result, pos_di_result, neg_di_result


# ---- Additional Utility Functions ----

def z_score_normalization(
    data: np.ndarray,
    window: Optional[int] = None,
    min_periods: int = None,
    center: bool = True
) -> np.ndarray:
    """
    Normalize data using z-score (rolling or static).

    Args:
        data: Input data array
        window: Rolling window size (None for static normalization)
        min_periods: Minimum number of observations in window
        center: Whether to center the window

    Returns:
        Z-score normalized data
    """
    if window is None:
        # Static normalization
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std
    else:
        # Rolling normalization
        if min_periods is None:
            min_periods = max(1, window // 4)

        # Use pandas for rolling calculations
        s = pd.Series(data)
        rolling_mean = s.rolling(window=window, min_periods=min_periods, center=center).mean()
        rolling_std = s.rolling(window=window, min_periods=min_periods, center=center).std()

        # Handle zero standard deviation
        z_scores = np.zeros_like(data)
        mask = rolling_std > 0
        z_scores[mask] = (data[mask] - rolling_mean[mask]) / rolling_std[mask]

        return z_scores