"""
data_normalizer.py - Data Normalization and Validation

This module provides functions for normalizing and validating market data
from various sources into a consistent format.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Any, Tuple

logger = logging.getLogger(__name__)


def normalize_market_data(
    data: pd.DataFrame,
    symbol: str,
    timeframe: str,
    include_derived_fields: bool = True
) -> pd.DataFrame:
    """
    Normalize market data into a consistent format.

    Args:
        data: Market data DataFrame
        symbol: Symbol of the data
        timeframe: Timeframe of the data
        include_derived_fields: Whether to include derived fields

    Returns:
        Normalized DataFrame
    """
    if data.empty:
        logger.warning(f"Cannot normalize empty DataFrame for {symbol}")
        return data

    # Make a copy to avoid modifying original data
    df = data.copy()

    # Ensure index is a datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        # Check if there's a timestamp/datetime column
        datetime_cols = [c for c in df.columns if c.lower() in ('timestamp', 'datetime', 'date', 'time')]
        if datetime_cols:
            df.set_index(datetime_cols[0], inplace=True)
            df.index = pd.to_datetime(df.index)
        else:
            logger.warning(f"No datetime index found in data for {symbol}")
            # Create a dummy index if none exists
            df.index = pd.date_range(
                start=pd.Timestamp.now().floor('D'),
                periods=len(df),
                freq=timeframe
            )

    # Ensure we have OHLCV columns (use consistent lowercase names)
    required_columns = {'open', 'high', 'low', 'close', 'volume'}
    column_mapping = {}

    # Map existing columns to required columns (case-insensitive)
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in required_columns and col != col_lower:
            column_mapping[col] = col_lower

    # Rename columns if needed
    if column_mapping:
        df.rename(columns=column_mapping, inplace=True)

    # Add missing columns with reasonable defaults
    for col in required_columns:
        if col not in df.columns:
            if col == 'open' and 'close' in df.columns:
                # Use previous close as open if not available
                df['open'] = df['close'].shift(1)
            elif col == 'high' and 'close' in df.columns:
                # Use close as high if not available
                df['high'] = df['close']
            elif col == 'low' and 'close' in df.columns:
                # Use close as low if not available
                df['low'] = df['close']
            elif col == 'volume':
                # Default to zero for volume
                df['volume'] = 0
            else:
                # We need at least a close price
                logger.warning(f"Missing required column {col} in data for {symbol}")
                df[col] = np.nan

    # Fix data types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Replace inf values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill missing values
    if df.isna().any().any():
        # Forward fill missing values
        df.fillna(method='ffill', inplace=True)
        # Backward fill any remaining NaNs
        df.fillna(method='bfill', inplace=True)

    # Add security info
    df['symbol'] = symbol
    df['timeframe'] = timeframe

    # Add derived fields if requested
    if include_derived_fields:
        add_derived_fields(df)

    # Validate OHLC relationships
    validate_ohlc_relationships(df)

    return df


def add_derived_fields(df: pd.DataFrame) -> None:
    """
    Add derived technical fields to the DataFrame.

    Args:
        df: Market data DataFrame
    """
    # Add basic price fields
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        # Typical price: (high + low + close) / 3
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

        # Median price: (high + low) / 2
        df['median_price'] = (df['high'] + df['low']) / 2

        # Average price: (open + high + low + close) / 4
        df['avg_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

        # Price range of the bar
        df['range'] = df['high'] - df['low']

        # Body size (absolute value of open-close)
        df['body'] = (df['close'] - df['open']).abs()

        # Upper shadow
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)

        # Lower shadow
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']

    # Add price changes
    if 'close' in df.columns:
        # Absolute and percentage changes
        df['change'] = df['close'].diff()
        df['returns'] = df['close'].pct_change()

        # Log returns (used in many financial models)
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Add volume-based metrics
    if all(col in df.columns for col in ['volume', 'close']):
        # Volume moving average (5 periods)
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()

        # Volume ratio to 5-period average
        df['volume_ratio'] = df['volume'] / df['volume_ma5']

        # Volume * close (aka turnover or money flow)
        df['volume_price'] = df['volume'] * df['close']


def validate_ohlc_relationships(df: pd.DataFrame) -> None:
    """
    Validate and fix OHLC price relationships.

    Args:
        df: Market data DataFrame with OHLC columns
    """
    if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        return

    # High should be >= Low
    invalid_hl = (df['high'] < df['low'])
    if invalid_hl.any():
        invalid_count = invalid_hl.sum()
        logger.warning(f"Found {invalid_count} bars where high < low, fixing...")
        # Swap high and low where high < low
        df.loc[invalid_hl, ['high', 'low']] = df.loc[invalid_hl, ['low', 'high']].values

    # High should be >= Open and Close
    invalid_ho = (df['high'] < df['open'])
    invalid_hc = (df['high'] < df['close'])

    if invalid_ho.any() or invalid_hc.any():
        invalid_count = invalid_ho.sum() + invalid_hc.sum()
        logger.warning(f"Found {invalid_count} bars with invalid high values, fixing...")
        # Set high to max of open, high, close
        df['high'] = df[['open', 'high', 'close']].max(axis=1)

    # Low should be <= Open and Close
    invalid_lo = (df['low'] > df['open'])
    invalid_lc = (df['low'] > df['close'])

    if invalid_lo.any() or invalid_lc.any():
        invalid_count = invalid_lo.sum() + invalid_lc.sum()
        logger.warning(f"Found {invalid_count} bars with invalid low values, fixing...")
        # Set low to min of open, low, close
        df['low'] = df[['open', 'low', 'close']].min(axis=1)


def detect_outliers(
    df: pd.DataFrame,
    columns: List[str] = None,
    method: str = 'iqr',
    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Detect outliers in market data.

    Args:
        df: Market data DataFrame
        columns: Columns to check for outliers (default: ['open', 'high', 'low', 'close'])
        method: Method to use ('iqr' or 'zscore')
        threshold: Threshold for outlier detection

    Returns:
        DataFrame with boolean mask of outliers
    """
    if df.empty:
        return pd.DataFrame()

    # Default columns to check
    if columns is None:
        columns = ['open', 'high', 'low', 'close']
        columns = [c for c in columns if c in df.columns]

    if not columns:
        logger.warning("No valid columns for outlier detection")
        return pd.DataFrame(False, index=df.index, columns=columns)

    outliers = pd.DataFrame(False, index=df.index, columns=columns)

    for col in columns:
        if method == 'iqr':
            # IQR method
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
        elif method == 'zscore':
            # Z-score method
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:  # Avoid division by zero
                z_scores = (df[col] - mean) / std
                outliers[col] = z_scores.abs() > threshold
        else:
            logger.warning(f"Unknown outlier detection method: {method}")

    return outliers


def remove_outliers(
    df: pd.DataFrame,
    columns: List[str] = None,
    method: str = 'iqr',
    threshold: float = 3.0,
    fill_method: str = 'interpolate'
) -> pd.DataFrame:
    """
    Remove outliers from market data.

    Args:
        df: Market data DataFrame
        columns: Columns to check for outliers (default: ['open', 'high', 'low', 'close'])
        method: Method to use ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
        fill_method: Method to fill outliers ('interpolate', 'ffill', 'bfill', or 'mean')

    Returns:
        DataFrame with outliers removed
    """
    if df.empty:
        return df

    # Make a copy to avoid modifying the original
    result = df.copy()

    # Detect outliers
    outliers = detect_outliers(df, columns, method, threshold)

    # Fill outliers with appropriate values
    for col in outliers.columns:
        if outliers[col].any():
            outlier_count = outliers[col].sum()
            logger.info(f"Removing {outlier_count} outliers from {col}")

            if fill_method == 'interpolate':
                # Mark outliers as NaN then interpolate
                result.loc[outliers[col], col] = np.nan
                result[col] = result[col].interpolate(method='time')
                # Fill any remaining NaNs at edges
                result[col] = result[col].fillna(method='ffill').fillna(method='bfill')
            elif fill_method == 'ffill':
                # Mark outliers as NaN then forward fill
                result.loc[outliers[col], col] = np.nan
                result[col] = result[col].fillna(method='ffill')
            elif fill_method == 'bfill':
                # Mark outliers as NaN then backward fill
                result.loc[outliers[col], col] = np.nan
                result[col] = result[col].fillna(method='bfill')
            elif fill_method == 'mean':
                # Replace with rolling mean
                rolling_mean = df[col].rolling(window=5, center=True).mean()
                result.loc[outliers[col], col] = rolling_mean.loc[outliers[col]]
                # Fill any NaNs at edges
                result[col] = result[col].fillna(method='ffill').fillna(method='bfill')

    # Ensure OHLC relationships are still valid after removal
    validate_ohlc_relationships(result)

    return result


def resample_market_data(
    df: pd.DataFrame,
    target_timeframe: str,
    volume_weighted: bool = True
) -> pd.DataFrame:
    """
    Resample market data to a different timeframe.

    Args:
        df: Market data DataFrame with OHLCV columns
        target_timeframe: Target timeframe (e.g., '1h', '1d')
        volume_weighted: Whether to use volume-weighted averages for price

    Returns:
        Resampled DataFrame
    """
    if df.empty:
        return df

    # Make a copy to avoid modifying the original
    result = df.copy()

    # Ensure we have a datetime index
    if not isinstance(result.index, pd.DatetimeIndex):
        logger.warning("DataFrame must have a DatetimeIndex for resampling")
        return df

    # Map pandas resampling rule
    rule_map = {
        '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
        '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '8h': '8H', '12h': '12H',
        '1d': '1D', '3d': '3D', '1w': '1W', '1M': '1M'
    }

    if target_timeframe not in rule_map:
        logger.warning(f"Unsupported target timeframe: {target_timeframe}")
        return df

    rule = rule_map[target_timeframe]

    # Define resampling functions
    if volume_weighted and 'volume' in result.columns and result['volume'].sum() > 0:
        # Volume-weighted resampling for price data
        price_cols = [c for c in ['open', 'high', 'low', 'close'] if c in result.columns]

        # Calculate volume-weighted average price
        if 'close' in result.columns and 'volume' in result.columns:
            result['vwap'] = (result['close'] * result['volume']).cumsum() / result['volume'].cumsum()

        # Define aggregation functions for OHLCV data
        agg_funcs = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }

        # Add aggregation for derived fields
        for col in result.columns:
            if col not in agg_funcs and col not in ['symbol', 'timeframe']:
                if col in ['vwap', 'typical_price', 'median_price', 'avg_price']:
                    # Use last value for derived price fields
                    agg_funcs[col] = 'last'
                elif 'volume' in col:
                    # Sum for volume-related fields
                    agg_funcs[col] = 'sum'
                else:
                    # Mean for other fields
                    agg_funcs[col] = 'mean'

        # Keep symbol and timeframe
        if 'symbol' in result.columns:
            agg_funcs['symbol'] = 'first'

        # Perform resampling with the defined aggregation functions
        resampled = result.resample(rule).agg(agg_funcs)

    else:
        # Standard OHLC resampling without volume weighting
        resampled = result.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

    # Update timeframe in result
    if 'timeframe' in resampled.columns:
        resampled['timeframe'] = target_timeframe
    else:
        resampled['timeframe'] = target_timeframe

    # Recalculate derived fields for the new timeframe
    add_derived_fields(resampled)

    # Handle missing values that might have been introduced
    resampled.dropna(inplace=True)

    return resampled


def merge_market_data(dataframes: List[pd.DataFrame], fill_method: str = 'ffill') -> pd.DataFrame:
    """
    Merge multiple market data DataFrames into one.

    Args:
        dataframes: List of DataFrames to merge
        fill_method: Method to fill missing values after merging

    Returns:
        Merged DataFrame
    """
    if not dataframes:
        return pd.DataFrame()

    # Filter out empty dataframes
    valid_dfs = [df for df in dataframes if not df.empty]

    if not valid_dfs:
        return pd.DataFrame()

    if len(valid_dfs) == 1:
        return valid_dfs[0].copy()

    # Ensure all DataFrames have datetime index
    for i, df in enumerate(valid_dfs):
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning(f"DataFrame {i} does not have DatetimeIndex, skipping")
            valid_dfs[i] = None

    valid_dfs = [df for df in valid_dfs if df is not None]

    if not valid_dfs:
        return pd.DataFrame()

    # Concatenate DataFrames
    merged = pd.concat(valid_dfs)

    # Remove duplicates (keep last value for each timestamp)
    merged = merged[~merged.index.duplicated(keep='last')]

    # Sort by timestamp
    merged.sort_index(inplace=True)

    # Fill missing values
    if fill_method == 'ffill':
        merged.fillna(method='ffill', inplace=True)
    elif fill_method == 'bfill':
        merged.fillna(method='bfill', inplace=True)
    elif fill_method == 'interpolate':
        merged = merged.interpolate(method='time')
        # Fill any remaining NaNs at edges
        merged.fillna(method='ffill', inplace=True)
        merged.fillna(method='bfill', inplace=True)

    # Validate OHLC relationships
    validate_ohlc_relationships(merged)

    return merged