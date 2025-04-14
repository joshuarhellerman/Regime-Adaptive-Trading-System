"""
tests/unit/data/historical_repository.py - Tests for Historical Data Repository

This module contains unit tests for the HistoricalRepository class which provides
access to archived historical market data.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import tempfile
import shutil

from data.fetchers.historical_repository import HistoricalRepository
from data.fetchers.exchange_connector import MarketType, DataType, DataFrequency
from data.processors.data_normalizer import DataNormalizer
from data.storage.time_series_store import TimeSeriesStore
from core.event_bus import EventBus


class TestHistoricalRepository(unittest.TestCase):
    """Test cases for the HistoricalRepository class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mocks for dependencies
        self.time_series_store = Mock(spec=TimeSeriesStore)
        self.data_normalizer = Mock(spec=DataNormalizer)
        self.event_bus = Mock(spec=EventBus)
        self.event_bus.publish = AsyncMock()

        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()

        # Configuration with test paths
        self.config = {
            'base_path': self.temp_dir,
            'sources': {
                'test_source': {'type': 'test'}
            },
            'versioning_enabled': True,
            'current_version': 'test_version'
        }

        # Create test version directory
        os.makedirs(os.path.join(self.temp_dir, 'test_version'))

        # Initialize repository
        self.repo = HistoricalRepository(
            self.time_series_store,
            self.data_normalizer,
            self.event_bus,
            self.config
        )

        # Sample data for tests
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
        self.test_df = pd.DataFrame({
            'open': np.random.rand(len(dates)) * 100,
            'high': np.random.rand(len(dates)) * 100,
            'low': np.random.rand(len(dates)) * 100,
            'close': np.random.rand(len(dates)) * 100,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        # Remove temp directory
        shutil.rmtree(self.temp_dir)

    async def test_init(self):
        """Test initialization of the repository."""
        # Verify base directory was created
        self.assertTrue(os.path.exists(self.temp_dir))

        # Verify configuration was properly loaded
        self.assertEqual(self.repo.base_path, Path(self.temp_dir))
        self.assertEqual(self.repo.sources, self.config['sources'])
        self.assertEqual(self.repo.current_version, 'test_version')
        self.assertTrue(self.repo.versioning_enabled)

    @patch('pandas.read_parquet')
    async def test_get_historical_data_single_instrument(self, mock_read_parquet):
        """Test fetching historical data for a single instrument."""
        # Setup mock return value
        mock_read_parquet.return_value = self.test_df

        # Call method
        result = await self.repo.get_historical_data(
            instruments='BTC-USD',
            frequency=DataFrequency.DAILY,
            start_time=datetime(2023, 1, 1),
            end_time=datetime(2023, 1, 10),
            data_type=DataType.OHLCV,
            market_type=MarketType.CRYPTO,
            source='test_source'
        )

        # Verify result
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.test_df))
        mock_read_parquet.assert_called_once()

    @patch('pandas.read_parquet')
    async def test_get_historical_data_multiple_instruments(self, mock_read_parquet):
        """Test fetching historical data for multiple instruments."""
        # Setup mock return value
        mock_read_parquet.return_value = self.test_df

        # Call method
        result = await self.repo.get_historical_data(
            instruments=['BTC-USD', 'ETH-USD'],
            frequency=DataFrequency.DAILY,
            start_time=datetime(2023, 1, 1),
            end_time=datetime(2023, 1, 10),
            data_type=DataType.OHLCV,
            market_type=MarketType.CRYPTO
        )

        # Verify result
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        self.assertIn('BTC-USD', result)
        self.assertIn('ETH-USD', result)
        self.assertEqual(mock_read_parquet.call_count, 2)

    @patch('pandas.read_parquet')
    async def test_get_historical_data_timestamp_conversion(self, mock_read_parquet):
        """Test timestamp conversion in get_historical_data."""
        # Setup mock return value
        mock_read_parquet.return_value = self.test_df

        # Call method with timestamps instead of datetime
        timestamp_start = int(datetime(2023, 1, 1).timestamp() * 1000)
        timestamp_end = int(datetime(2023, 1, 10).timestamp() * 1000)

        result = await self.repo.get_historical_data(
            instruments='BTC-USD',
            frequency=DataFrequency.DAILY,
            start_time=timestamp_start,
            end_time=timestamp_end,
            data_type=DataType.OHLCV
        )

        # Verify result
        self.assertIsInstance(result, pd.DataFrame)
        mock_read_parquet.assert_called_once()

    @patch('pandas.read_parquet')
    async def test_get_historical_data_error_handling(self, mock_read_parquet):
        """Test error handling in get_historical_data."""
        # Setup mock to raise exception
        mock_read_parquet.side_effect = Exception("Test error")

        # Call method and expect exception
        with self.assertRaises(Exception):
            await self.repo.get_historical_data(
                instruments='BTC-USD',
                frequency=DataFrequency.DAILY,
                start_time=datetime(2023, 1, 1),
                end_time=datetime(2023, 1, 10)
            )

        # Verify error event was published
        self.event_bus.publish.assert_called_once()

    async def test_store_historical_data_single_instrument(self):
        """Test storing historical data for a single instrument."""
        # Mock the write methods
        with patch.object(self.repo, '_write_parquet') as mock_write:
            # Call method
            result = await self.repo.store_historical_data(
                data=self.test_df,
                instruments='BTC-USD',
                frequency=DataFrequency.DAILY,
                data_type=DataType.OHLCV,
                market_type=MarketType.CRYPTO,
                source='test_source'
            )

            # Verify result
            self.assertTrue(result)
            mock_write.assert_called_once()

    async def test_store_historical_data_multiple_instruments(self):
        """Test storing historical data for multiple instruments."""
        # Prepare data dictionary
        data_dict = {
            'BTC-USD': self.test_df,
            'ETH-USD': self.test_df
        }

        # Mock the write methods
        with patch.object(self.repo, '_write_parquet') as mock_write:
            # Call method
            result = await self.repo.store_historical_data(
                data=data_dict,
                instruments=['BTC-USD', 'ETH-USD'],
                frequency=DataFrequency.DAILY,
                data_type=DataType.OHLCV
            )

            # Verify result
            self.assertTrue(result)
            self.assertEqual(mock_write.call_count, 2)

    async def test_store_historical_data_empty_data(self):
        """Test storing empty historical data."""
        # Mock the write methods
        with patch.object(self.repo, '_write_parquet') as mock_write:
            # Call method with empty DataFrame
            result = await self.repo.store_historical_data(
                data=pd.DataFrame(),
                instruments='BTC-USD',
                frequency=DataFrequency.DAILY,
                data_type=DataType.OHLCV
            )

            # Verify result
            self.assertTrue(result)
            mock_write.assert_not_called()

    async def test_store_historical_data_error_handling(self):
        """Test error handling in store_historical_data."""
        # Mock the write methods to raise exception
        with patch.object(self.repo, '_write_parquet', side_effect=Exception("Test error")):
            # Call method
            result = await self.repo.store_historical_data(
                data=self.test_df,
                instruments='BTC-USD',
                frequency=DataFrequency.DAILY,
                data_type=DataType.OHLCV
            )

            # Verify result
            self.assertFalse(result)

    async def test_list_available_data(self):
        """Test listing available data."""
        # Create test directories and files
        test_path = os.path.join(
            self.temp_dir,
            'test_version',
            'crypto',
            'ohlcv',
            'daily',
            'test_source'
        )
        os.makedirs(test_path, exist_ok=True)

        # Create test files
        with open(os.path.join(test_path, 'BTC-USD.parquet'), 'w') as f:
            f.write("test data")

        # Call method
        result = await self.repo.list_available_data(
            market_type=MarketType.CRYPTO,
            data_type=DataType.OHLCV,
            frequency=DataFrequency.DAILY,
            source='test_source'
        )

        # Verify result
        self.assertEqual(result['version'], 'test_version')
        self.assertEqual(result['count'], 1)
        self.assertIn('crypto', result['market_types'])
        self.assertIn('ohlcv', result['data_types'])
        self.assertIn('daily', result['frequencies'])
        self.assertIn('test_source', result['sources'])
        self.assertIn('BTC-USD', result['instruments'])

    async def test_list_versions(self):
        """Test listing available versions."""
        # Create additional test version
        os.makedirs(os.path.join(self.temp_dir, 'another_version'))

        # Call method
        versions = await self.repo.list_versions()

        # Verify result
        self.assertEqual(len(versions), 2)
        self.assertIn('test_version', versions)
        self.assertIn('another_version', versions)

    async def test_create_version(self):
        """Test creating a new version."""
        # Call method
        result = await self.repo.create_version(
            version='new_version',
            description='Test version',
            base_version='test_version'
        )

        # Verify result
        self.assertTrue(result)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'new_version')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'new_version', 'metadata.json')))

        # Check metadata content
        with open(os.path.join(self.temp_dir, 'new_version', 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        self.assertEqual(metadata['version'], 'new_version')
        self.assertEqual(metadata['description'], 'Test version')
        self.assertEqual(metadata['based_on'], 'test_version')

    async def test_create_version_already_exists(self):
        """Test creating a version that already exists."""
        # Call method with existing version name
        result = await self.repo.create_version(version='test_version')

        # Verify result
        self.assertFalse(result)

    async def test_create_version_invalid_base(self):
        """Test creating a version with invalid base version."""
        # Call method with non-existent base version
        result = await self.repo.create_version(
            version='new_version',
            base_version='non_existent'
        )

        # Verify result
        self.assertFalse(result)

    async def test_set_current_version(self):
        """Test setting the current version."""
        # Call method
        result = await self.repo.set_current_version('test_version')

        # Verify result
        self.assertTrue(result)
        self.assertEqual(self.repo.current_version, 'test_version')
        self.assertEqual(self.config['current_version'], 'test_version')

    async def test_set_current_version_not_found(self):
        """Test setting a non-existent version as current."""
        # Call method with non-existent version
        result = await self.repo.set_current_version('non_existent')

        # Verify result
        self.assertFalse(result)
        self.assertEqual(self.repo.current_version, 'test_version')

    def test_get_data_path(self):
        """Test building data file paths."""
        # Call method
        path = self.repo._get_data_path(
            instrument='BTC-USD',
            frequency='daily',
            data_type=DataType.OHLCV,
            market_type='crypto',
            source='test_source',
            version='test_version'
        )

        # Verify result
        expected_path = Path(os.path.join(
            self.temp_dir,
            'test_version',
            'crypto',
            'ohlcv',
            'daily',
            'test_source',
            'BTC-USD.parquet'
        ))
        self.assertEqual(path, expected_path)

    def test_get_data_path_with_defaults(self):
        """Test building data file paths with default values."""
        # Call method with minimal parameters
        path = self.repo._get_data_path(
            instrument='BTC-USD',
            frequency='daily',
            data_type=DataType.OHLCV,
            market_type=None,
            source=None,
            version='test_version'
        )

        # Verify result
        expected_path = Path(os.path.join(
            self.temp_dir,
            'test_version',
            'default',
            'ohlcv',
            'daily',
            'default',
            'BTC-USD.parquet'
        ))
        self.assertEqual(path, expected_path)

    def test_build_path_pattern(self):
        """Test building path patterns for glob."""
        # Call method
        pattern = self.repo._build_path_pattern(
            market_type='crypto',
            data_type='ohlcv',
            frequency='daily',
            source='test_source',
            version='test_version'
        )

        # Verify result
        expected_pattern = 'test_version/crypto/ohlcv/daily/test_source/*'
        self.assertEqual(pattern, expected_pattern)

    def test_build_path_pattern_with_wildcards(self):
        """Test building path patterns with wildcards."""
        # Call method with minimal parameters
        pattern = self.repo._build_path_pattern(
            market_type=None,
            data_type=None,
            frequency=None,
            source=None,
            version='test_version'
        )

        # Verify result
        expected_pattern = 'test_version/*/*/*/*/*'
        self.assertEqual(pattern, expected_pattern)

    def test_get_data_format(self):
        """Test getting data formats for different data types."""
        # Test for different data types
        self.assertEqual(self.repo._get_data_format(DataType.OHLCV), 'parquet')
        self.assertEqual(self.repo._get_data_format(DataType.DEPTH), 'hdf5')
        self.assertEqual(self.repo._get_data_format(DataType.FUNDAMENTAL), 'json')

    def test_get_file_extension(self):
        """Test getting file extensions for different data types."""
        # Test for different data types
        self.assertEqual(self.repo._get_file_extension(DataType.OHLCV), '.parquet')
        self.assertEqual(self.repo._get_file_extension(DataType.DEPTH), '.h5')
        self.assertEqual(self.repo._get_file_extension(DataType.FUNDAMENTAL), '.json')

    @patch('pandas.read_parquet')
    def test_read_parquet(self, mock_read_parquet):
        """Test reading parquet files."""
        # Setup mock
        mock_read_parquet.return_value = self.test_df

        # Call method
        result = self.repo._read_parquet(Path('test.parquet'))

        # Verify result
        self.assertEqual(result.equals(self.test_df), True)
        mock_read_parquet.assert_called_once_with(Path('test.parquet'))

    @patch('pandas.read_parquet', side_effect=Exception("Test error"))
    def test_read_parquet_error(self, mock_read_parquet):
        """Test error handling when reading parquet files."""
        # Call method
        result = self.repo._read_parquet(Path('test.parquet'))

        # Verify result is empty DataFrame on error
        self.assertTrue(result.empty)
        mock_read_parquet.assert_called_once_with(Path('test.parquet'))

    @patch('pandas.DataFrame.to_parquet')
    def test_write_parquet(self, mock_to_parquet):
        """Test writing parquet files."""
        # Call method
        self.repo._write_parquet(self.test_df, Path('test.parquet'))

        # Verify mocked method was called
        mock_to_parquet.assert_called_once_with(Path('test.parquet'))

    @patch('pandas.read_hdf')
    def test_read_hdf5(self, mock_read_hdf):
        """Test reading HDF5 files."""
        # Setup mock
        mock_read_hdf.return_value = self.test_df

        # Call method
        result = self.repo._read_hdf5(Path('test.h5'), 'test_key')

        # Verify result
        self.assertEqual(result.equals(self.test_df), True)
        mock_read_hdf.assert_called_once_with(Path('test.h5'), key='test_key')

    @patch('pandas.read_hdf', side_effect=Exception("Test error"))
    def test_read_hdf5_error(self, mock_read_hdf):
        """Test error handling when reading HDF5 files."""
        # Call method
        result = self.repo._read_hdf5(Path('test.h5'), 'test_key')

        # Verify result is empty DataFrame on error
        self.assertTrue(result.empty)
        mock_read_hdf.assert_called_once_with(Path('test.h5'), key='test_key')

    @patch('pandas.DataFrame.to_hdf')
    def test_write_hdf5(self, mock_to_hdf):
        """Test writing HDF5 files."""
        # Call method
        self.repo._write_hdf5(self.test_df, Path('test.h5'), 'test_key')

        # Verify mocked method was called
        mock_to_hdf.assert_called_once_with(Path('test.h5'), key='test_key', mode='w')


if __name__ == '__main__':
    unittest.main()