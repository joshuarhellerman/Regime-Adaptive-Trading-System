"""
Unit tests for data_normalizer.py module.
"""

import unittest
import pandas as pd
import numpy as np
import logging
from unittest.mock import patch, MagicMock
import datetime

# Import the module to test
from data_normalizer import (
    normalize_market_data,
    add_derived_fields,
    validate_ohlc_relationships,
    detect_outliers,
    remove_outliers,
    resample_market_data,
    merge_market_data,
)


class TestDataNormalizer(unittest.TestCase):
    """Test cases for data_normalizer.py functions."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Sample OHLCV data with datetime index
        self.sample_dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
        self.sample_data = pd.DataFrame({
            'open': [100.0, 102.0, 101.0, 103.0, 105.0],
            'high': [105.0, 107.0, 106.0, 108.0, 110.0],
            'low': [98.0, 100.0, 99.0, 101.0, 103.0],
            'close': [102.0, 103.0, 104.0, 106.0, 108.0],
            'volume': [1000, 1500, 1200, 1800, 2000]
        }, index=self.sample_dates)

        # Sample data with invalid OHLC relationships
        self.invalid_ohlc_data = pd.DataFrame({
            'open': [100.0, 102.0, 101.0, 103.0, 105.0],
            'high': [98.0, 107.0, 106.0, 102.0, 110.0],  # High < Low in index 0, High < Open in index 3
            'low': [99.0, 100.0, 107.0, 101.0, 103.0],   # Low > High in index 0, Low > Close in index 2
            'close': [102.0, 103.0, 104.0, 106.0, 108.0],
            'volume': [1000, 1500, 1200, 1800, 2000]
        }, index=self.sample_dates)

        # Sample data with outliers
        self.outlier_data = pd.DataFrame({
            'open': [100.0, 102.0, 101.0, 103.0, 105.0],
            'high': [105.0, 107.0, 106.0, 108.0, 150.0],  # Outlier in high at index 4
            'low': [98.0, 100.0, 99.0, 101.0, 103.0],
            'close': [102.0, 103.0, 104.0, 106.0, 108.0],
            'volume': [1000, 1500, 1200, 1800, 10000]    # Outlier in volume at index 4
        }, index=self.sample_dates)

        # Sample data without datetime index
        self.non_datetime_data = pd.DataFrame({
            'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
            'Open': [100.0, 102.0, 101.0, 103.0, 105.0],  # Capitals to test column name normalization
            'High': [105.0, 107.0, 106.0, 108.0, 110.0],
            'Low': [98.0, 100.0, 99.0, 101.0, 103.0],
            'Close': [102.0, 103.0, 104.0, 106.0, 108.0],
            'Volume': [1000, 1500, 1200, 1800, 2000]
        })

        # Sample data missing columns
        self.missing_columns_data = pd.DataFrame({
            'close': [102.0, 103.0, 104.0, 106.0, 108.0],
            'volume': [1000, 1500, 1200, 1800, 2000]
        }, index=self.sample_dates)

    def test_normalize_market_data_with_valid_data(self):
        """Test normalizing valid market data."""
        result = normalize_market_data(self.sample_data, 'AAPL', '1d')

        # Check that required columns exist
        for col in ['open', 'high', 'low', 'close', 'volume', 'symbol', 'timeframe']:
            self.assertIn(col, result.columns)

        # Check that symbol and timeframe are correctly set
        self.assertEqual(result['symbol'].iloc[0], 'AAPL')
        self.assertEqual(result['timeframe'].iloc[0], '1d')

        # Check that derived fields are added
        for col in ['typical_price', 'median_price', 'avg_price', 'range', 'body',
                   'upper_shadow', 'lower_shadow', 'change', 'returns', 'log_returns']:
            self.assertIn(col, result.columns)

    def test_normalize_market_data_with_empty_data(self):
        """Test normalizing empty market data."""
        empty_df = pd.DataFrame()
        result = normalize_market_data(empty_df, 'AAPL', '1d')
        self.assertTrue(result.empty)

    def test_normalize_market_data_with_non_datetime_index(self):
        """Test normalizing data without datetime index."""
        result = normalize_market_data(self.non_datetime_data, 'AAPL', '1d')

        # Check that index is now DatetimeIndex
        self.assertIsInstance(result.index, pd.DatetimeIndex)

        # Check that columns are normalized (lowercase)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            self.assertIn(col, result.columns)

    def test_normalize_market_data_without_derived_fields(self):
        """Test normalizing data without including derived fields."""
        result = normalize_market_data(self.sample_data, 'AAPL', '1d', include_derived_fields=False)

        # Check that derived fields are not added
        for col in ['typical_price', 'median_price', 'avg_price']:
            self.assertNotIn(col, result.columns)

    def test_normalize_market_data_with_missing_columns(self):
        """Test normalizing data with missing columns."""
        result = normalize_market_data(self.missing_columns_data, 'AAPL', '1d')

        # Check that missing columns were added
        for col in ['open', 'high', 'low']:
            self.assertIn(col, result.columns)

        # Check that open was set to previous close
        pd.testing.assert_series_equal(
            result['open'].iloc[1:],
            self.missing_columns_data['close'].iloc[:-1].reset_index(drop=True),
            check_names=False
        )

    def test_add_derived_fields(self):
        """Test adding derived fields to market data."""
        df = self.sample_data.copy()
        add_derived_fields(df)

        # Check that derived fields are added correctly
        self.assertAlmostEqual(
            df['typical_price'].iloc[0],
            (df['high'].iloc[0] + df['low'].iloc[0] + df['close'].iloc[0]) / 3
        )
        self.assertAlmostEqual(
            df['median_price'].iloc[0],
            (df['high'].iloc[0] + df['low'].iloc[0]) / 2
        )

        # Check that returns are calculated correctly
        self.assertAlmostEqual(
            df['returns'].iloc[1],
            (df['close'].iloc[1] / df['close'].iloc[0]) - 1
        )

    def test_validate_ohlc_relationships(self):
        """Test validating and fixing OHLC relationships."""
        df = self.invalid_ohlc_data.copy()

        # Before validation - check invalid relationships
        self.assertTrue((df['high'] < df['low']).any())
        self.assertTrue((df['high'] < df['open']).any())
        self.assertTrue((df['low'] > df['close']).any())

        validate_ohlc_relationships(df)

        # After validation - check relationships are fixed
        self.assertFalse((df['high'] < df['low']).any())
        self.assertFalse((df['high'] < df['open']).any())
        self.assertFalse((df['high'] < df['close']).any())
        self.assertFalse((df['low'] > df['open']).any())
        self.assertFalse((df['low'] > df['close']).any())

    def test_detect_outliers_iqr_method(self):
        """Test detecting outliers using IQR method."""
        outliers = detect_outliers(self.outlier_data, method='iqr', threshold=1.5)

        # Check that outliers are detected correctly
        self.assertTrue(outliers['high'].iloc[4])  # High value at index 4 is an outlier
        self.assertTrue(outliers['volume'].iloc[4])  # Volume at index 4 is an outlier

        # Check that non-outliers are not detected
        self.assertFalse(outliers['high'].iloc[0])
        self.assertFalse(outliers['close'].iloc[4])

    def test_detect_outliers_zscore_method(self):
        """Test detecting outliers using Z-score method."""
        outliers = detect_outliers(self.outlier_data, method='zscore', threshold=2.0)

        # Check that outliers are detected correctly
        self.assertTrue(outliers['high'].iloc[4])  # High value at index 4 is an outlier
        self.assertTrue(outliers['volume'].iloc[4])  # Volume at index 4 is an outlier

    def test_remove_outliers_interpolate(self):
        """Test removing outliers with interpolation."""
        result = remove_outliers(
            self.outlier_data, method='iqr', threshold=1.5, fill_method='interpolate'
        )

        # Check that outliers are removed
        self.assertNotEqual(result['high'].iloc[4], self.outlier_data['high'].iloc[4])
        self.assertNotEqual(result['volume'].iloc[4], self.outlier_data['volume'].iloc[4])

        # Check that non-outliers are preserved
        self.assertEqual(result['close'].iloc[4], self.outlier_data['close'].iloc[4])

    def test_remove_outliers_ffill(self):
        """Test removing outliers with forward fill."""
        result = remove_outliers(
            self.outlier_data, method='iqr', threshold=1.5, fill_method='ffill'
        )

        # Check that outliers are replaced with previous values
        self.assertEqual(result['high'].iloc[4], self.outlier_data['high'].iloc[3])

    def test_resample_market_data(self):
        """Test resampling market data to a different timeframe."""
        # Create hourly data
        hourly_dates = pd.date_range(start='2023-01-01', periods=24, freq='H')
        hourly_data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 24),
            'high': np.random.uniform(105, 115, 24),
            'low': np.random.uniform(95, 105, 24),
            'close': np.random.uniform(100, 110, 24),
            'volume': np.random.uniform(1000, 2000, 24)
        }, index=hourly_dates)

        # Resample to 4-hour data
        result = resample_market_data(hourly_data, '4h')

        # Check result shape (should be 24/4 = 6 rows)
        self.assertEqual(len(result), 6)

        # Check that resampling preserves OHLC relationships
        self.assertTrue((result['high'] >= result['low']).all())
        self.assertTrue((result['high'] >= result['open']).all())
        self.assertTrue((result['high'] >= result['close']).all())
        self.assertTrue((result['low'] <= result['open']).all())
        self.assertTrue((result['low'] <= result['close']).all())

    def test_resample_market_data_volume_weighted(self):
        """Test resampling market data with volume weighting."""
        # Create hourly data
        hourly_dates = pd.date_range(start='2023-01-01', periods=24, freq='H')
        hourly_data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 24),
            'high': np.random.uniform(105, 115, 24),
            'low': np.random.uniform(95, 105, 24),
            'close': np.random.uniform(100, 110, 24),
            'volume': np.random.uniform(1000, 2000, 24)
        }, index=hourly_dates)

        # Resample with volume weighting
        result = resample_market_data(hourly_data, '4h', volume_weighted=True)

        # Check result shape (should be 24/4 = 6 rows)
        self.assertEqual(len(result), 6)

        # Check that vwap column is added
        self.assertIn('vwap', result.columns)

    def test_merge_market_data(self):
        """Test merging multiple market data DataFrames."""
        # Create two DataFrames with different date ranges
        dates1 = pd.date_range(start='2023-01-01', periods=5, freq='D')
        data1 = pd.DataFrame({
            'open': [100.0, 102.0, 101.0, 103.0, 105.0],
            'high': [105.0, 107.0, 106.0, 108.0, 110.0],
            'low': [98.0, 100.0, 99.0, 101.0, 103.0],
            'close': [102.0, 103.0, 104.0, 106.0, 108.0],
            'volume': [1000, 1500, 1200, 1800, 2000]
        }, index=dates1)

        dates2 = pd.date_range(start='2023-01-04', periods=5, freq='D')
        data2 = pd.DataFrame({
            'open': [104.0, 106.0, 107.0, 108.0, 109.0],
            'high': [109.0, 111.0, 112.0, 113.0, 114.0],
            'low': [102.0, 104.0, 105.0, 106.0, 107.0],
            'close': [107.0, 109.0, 110.0, 111.0, 112.0],
            'volume': [1900, 2100, 2200, 2300, 2400]
        }, index=dates2)

        result = merge_market_data([data1, data2])

        # Check that merged data has correct length (should be 8 unique dates)
        self.assertEqual(len(result), 8)

        # Check that overlapping dates take the value from the second DataFrame
        self.assertEqual(result.loc['2023-01-04', 'close'], data2.loc['2023-01-04', 'close'])

    def test_merge_market_data_with_fill_methods(self):
        """Test merging market data with different fill methods."""
        # Create two DataFrames with gaps
        dates1 = pd.date_range(start='2023-01-01', periods=3, freq='2D')
        data1 = pd.DataFrame({
            'close': [100.0, 102.0, 104.0],
        }, index=dates1)

        dates2 = pd.date_range(start='2023-01-02', periods=3, freq='2D')
        data2 = pd.DataFrame({
            'close': [101.0, 103.0, 105.0],
        }, index=dates2)

        # Test forward fill
        result_ffill = merge_market_data([data1, data2], fill_method='ffill')
        self.assertEqual(len(result_ffill), 5)  # Should have 5 unique dates

        # Test backward fill
        result_bfill = merge_market_data([data1, data2], fill_method='bfill')
        self.assertEqual(len(result_bfill), 5)  # Should have 5 unique dates

        # Test interpolate
        result_interp = merge_market_data([data1, data2], fill_method='interpolate')
        self.assertEqual(len(result_interp), 5)  # Should have 5 unique dates

    def test_merge_market_data_with_empty_dataframes(self):
        """Test merging with empty DataFrames."""
        result = merge_market_data([self.sample_data, pd.DataFrame()])

        # Should return a copy of the non-empty DataFrame
        self.assertEqual(len(result), len(self.sample_data))
        pd.testing.assert_frame_equal(result, self.sample_data)

        # Should return empty DataFrame when all inputs are empty
        result_empty = merge_market_data([pd.DataFrame(), pd.DataFrame()])
        self.assertTrue(result_empty.empty)

    @patch('logging.Logger.warning')
    def test_logging_warnings(self, mock_warning):
        """Test that appropriate warnings are logged."""
        # Test normalize_market_data with empty data
        normalize_market_data(pd.DataFrame(), 'AAPL', '1d')
        mock_warning.assert_called_with("Cannot normalize empty DataFrame for AAPL")

        # Test normalize_market_data with non-datetime index and no timestamp column
        data_no_timestamp = pd.DataFrame({
            'close': [100.0, 101.0, 102.0]
        })
        normalize_market_data(data_no_timestamp, 'AAPL', '1d')
        mock_warning.assert_called_with("No datetime index found in data for AAPL")

        # Test validate_ohlc_relationships with invalid data
        validate_ohlc_relationships(self.invalid_ohlc_data)
        mock_warning.assert_called_with("Found 1 bars where high < low, fixing...")


if __name__ == '__main__':
    unittest.main()