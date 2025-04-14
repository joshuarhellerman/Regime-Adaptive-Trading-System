"""
tests/unit/data/test_market_snapshot.py - Tests for Market Data Snapshot

This module contains unit tests for the MarketDataSnapshot class which provides
functionality to capture and retrieve snapshots of market data at specific moments in time.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import tempfile
import shutil
from pathlib import Path

from data.storage.market_data_snapshot import MarketDataSnapshot
from data.fetchers.exchange_connector import MarketType, DataType, DataFrequency
from data.processors.data_normalizer import DataNormalizer
from core.event_bus import EventBus


class TestMarketDataSnapshot(unittest.TestCase):
    """Test cases for the MarketDataSnapshot class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mocks for dependencies
        self.data_normalizer = Mock(spec=DataNormalizer)
        self.event_bus = Mock(spec=EventBus)
        self.event_bus.publish = AsyncMock()

        # Create a temporary directory for test data
        self.temp_dir = tempfile.mkdtemp()

        # Configuration with test paths
        self.config = {
            'snapshot_base_path': self.temp_dir,
            'retention_policy': {
                'max_snapshots': 100,
                'max_age_days': 30,
            },
            'compression': True,
            'snapshot_formats': {
                DataType.OHLCV.value: 'parquet',
                DataType.DEPTH.value: 'hdf5',
                DataType.TRADES.value: 'parquet',
                DataType.QUOTES.value: 'parquet'
            }
        }

        # Initialize snapshot manager
        self.snapshot = MarketDataSnapshot(
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

        # Sample depth data
        self.test_depth_data = pd.DataFrame({
            'price': np.linspace(19000, 21000, 20),
            'quantity': np.random.rand(20) * 10,
            'side': ['bid'] * 10 + ['ask'] * 10
        })

        # Create test directories
        os.makedirs(os.path.join(self.temp_dir, 'snapshots'), exist_ok=True)

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        # Remove temp directory
        shutil.rmtree(self.temp_dir)

    async def test_init(self):
        """Test initialization of the snapshot manager."""
        # Verify snapshot directory was created
        snapshot_path = os.path.join(self.temp_dir, 'snapshots')
        self.assertTrue(os.path.exists(snapshot_path))

        # Verify configuration was properly loaded
        self.assertEqual(self.snapshot.base_path, Path(self.temp_dir))
        self.assertEqual(self.snapshot.max_snapshots, 100)
        self.assertEqual(self.snapshot.max_age_days, 30)
        self.assertTrue(self.snapshot.compression)

    @patch('pandas.read_parquet')
    async def test_get_snapshot_by_id(self, mock_read_parquet):
        """Test retrieving a snapshot by its ID."""
        # Setup mock return value
        mock_read_parquet.return_value = self.test_df

        # Mock snapshot metadata
        snapshot_id = "snapshot_20230101_123045"
        with patch.object(self.snapshot, '_get_snapshot_metadata') as mock_metadata:
            mock_metadata.return_value = {
                'id': snapshot_id,
                'timestamp': '2023-01-01T12:30:45',
                'instruments': ['BTC-USD', 'ETH-USD'],
                'market_types': {'BTC-USD': 'CRYPTO', 'ETH-USD': 'CRYPTO'},
                'data_types': {'BTC-USD': 'OHLCV', 'ETH-USD': 'OHLCV'}
            }

            # Call method
            result = await self.snapshot.get_snapshot_by_id(snapshot_id)

            # Verify result
            self.assertIsInstance(result, dict)
            self.assertEqual(result['metadata']['id'], snapshot_id)
            self.assertIn('data', result)
            mock_read_parquet.assert_called()

    @patch('pandas.DataFrame.to_parquet')
    async def test_take_snapshot(self, mock_to_parquet):
        """Test taking a new snapshot of market data."""
        # Prepare test data dictionary
        test_data = {
            'BTC-USD': self.test_df,
            'ETH-USD': self.test_df
        }

        # Mock timestamp to ensure consistent ID generation
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 30, 45)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # Call method
            snapshot_id = await self.snapshot.take_snapshot(
                data=test_data,
                instruments=['BTC-USD', 'ETH-USD'],
                market_types={'BTC-USD': MarketType.CRYPTO, 'ETH-USD': MarketType.CRYPTO},
                data_types={'BTC-USD': DataType.OHLCV, 'ETH-USD': DataType.OHLCV},
                description="Test snapshot"
            )

            # Verify result
            self.assertEqual(snapshot_id, "snapshot_20230101_123045")
            self.assertTrue(mock_to_parquet.called)

            # Verify metadata file was created
            metadata_path = os.path.join(
                self.temp_dir,
                'snapshots',
                snapshot_id,
                'metadata.json'
            )
            self.assertTrue(os.path.exists(metadata_path))

            # Verify metadata content
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            self.assertEqual(metadata['id'], snapshot_id)
            self.assertEqual(metadata['description'], "Test snapshot")
            self.assertEqual(len(metadata['instruments']), 2)

    @patch('pandas.DataFrame.to_parquet')
    async def test_take_snapshot_with_different_data_types(self, mock_to_parquet):
        """Test taking a snapshot with different data types."""
        # Prepare test data with different types
        test_data = {
            'BTC-USD': {
                DataType.OHLCV: self.test_df,
                DataType.DEPTH: self.test_depth_data
            },
            'ETH-USD': {
                DataType.OHLCV: self.test_df
            }
        }

        # Mock HDF5 write method
        with patch.object(self.snapshot, '_write_hdf5') as mock_write_hdf5:
            # Call method
            snapshot_id = await self.snapshot.take_snapshot(
                data=test_data,
                instruments=['BTC-USD', 'ETH-USD'],
                market_types={'BTC-USD': MarketType.CRYPTO, 'ETH-USD': MarketType.CRYPTO},
                data_types={
                    'BTC-USD': [DataType.OHLCV, DataType.DEPTH],
                    'ETH-USD': [DataType.OHLCV]
                },
                description="Mixed data types"
            )

            # Verify both storage methods were called
            self.assertTrue(mock_to_parquet.called)
            self.assertTrue(mock_write_hdf5.called)

    async def test_list_snapshots(self):
        """Test listing available snapshots."""
        # Create test snapshot directories
        os.makedirs(os.path.join(self.temp_dir, 'snapshots', 'snapshot_20230101_123045'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'snapshots', 'snapshot_20230102_123045'), exist_ok=True)

        # Create metadata files
        with open(os.path.join(self.temp_dir, 'snapshots', 'snapshot_20230101_123045', 'metadata.json'), 'w') as f:
            json.dump({
                'id': 'snapshot_20230101_123045',
                'timestamp': '2023-01-01T12:30:45',
                'instruments': ['BTC-USD'],
                'description': 'First test snapshot'
            }, f)

        with open(os.path.join(self.temp_dir, 'snapshots', 'snapshot_20230102_123045', 'metadata.json'), 'w') as f:
            json.dump({
                'id': 'snapshot_20230102_123045',
                'timestamp': '2023-01-02T12:30:45',
                'instruments': ['BTC-USD', 'ETH-USD'],
                'description': 'Second test snapshot'
            }, f)

        # Call method
        snapshots = await self.snapshot.list_snapshots()

        # Verify result
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(snapshots[0]['id'], 'snapshot_20230102_123045')  # Should be in reverse chronological order
        self.assertEqual(snapshots[1]['id'], 'snapshot_20230101_123045')
        self.assertEqual(len(snapshots[0]['instruments']), 2)
        self.assertEqual(len(snapshots[1]['instruments']), 1)

    async def test_list_snapshots_with_filter(self):
        """Test listing snapshots with filtering."""
        # Create test snapshot directories with metadata
        os.makedirs(os.path.join(self.temp_dir, 'snapshots', 'snapshot_20230101_123045'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'snapshots', 'snapshot_20230102_123045'), exist_ok=True)

        with open(os.path.join(self.temp_dir, 'snapshots', 'snapshot_20230101_123045', 'metadata.json'), 'w') as f:
            json.dump({
                'id': 'snapshot_20230101_123045',
                'timestamp': '2023-01-01T12:30:45',
                'instruments': ['BTC-USD'],
                'market_types': {'BTC-USD': 'CRYPTO'},
                'data_types': {'BTC-USD': 'OHLCV'},
                'description': 'First test snapshot'
            }, f)

        with open(os.path.join(self.temp_dir, 'snapshots', 'snapshot_20230102_123045', 'metadata.json'), 'w') as f:
            json.dump({
                'id': 'snapshot_20230102_123045',
                'timestamp': '2023-01-02T12:30:45',
                'instruments': ['BTC-USD', 'ETH-USD'],
                'market_types': {'BTC-USD': 'CRYPTO', 'ETH-USD': 'CRYPTO'},
                'data_types': {'BTC-USD': 'DEPTH', 'ETH-USD': 'OHLCV'},
                'description': 'Second test snapshot'
            }, f)

        # Call method with instrument filter
        snapshots = await self.snapshot.list_snapshots(instrument='ETH-USD')

        # Verify result
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]['id'], 'snapshot_20230102_123045')

        # Call method with data type filter
        snapshots = await self.snapshot.list_snapshots(data_type=DataType.DEPTH)

        # Verify result
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]['id'], 'snapshot_20230102_123045')

        # Call method with start time filter
        snapshots = await self.snapshot.list_snapshots(
            start_time=datetime(2023, 1, 2)
        )

        # Verify result
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]['id'], 'snapshot_20230102_123045')

    async def test_delete_snapshot(self):
        """Test deleting a snapshot."""
        # Create test snapshot directory
        snapshot_id = 'snapshot_20230101_123045'
        snapshot_dir = os.path.join(self.temp_dir, 'snapshots', snapshot_id)
        os.makedirs(snapshot_dir, exist_ok=True)

        # Create metadata file
        with open(os.path.join(snapshot_dir, 'metadata.json'), 'w') as f:
            json.dump({
                'id': snapshot_id,
                'timestamp': '2023-01-01T12:30:45',
                'instruments': ['BTC-USD'],
                'description': 'Test snapshot to delete'
            }, f)

        # Create data files
        with open(os.path.join(snapshot_dir, 'BTC-USD.parquet'), 'w') as f:
            f.write("mock data")

        # Call method
        result = await self.snapshot.delete_snapshot(snapshot_id)

        # Verify result
        self.assertTrue(result)
        self.assertFalse(os.path.exists(snapshot_dir))

    async def test_delete_snapshot_not_found(self):
        """Test attempting to delete a non-existent snapshot."""
        # Call method with non-existent snapshot ID
        result = await self.snapshot.delete_snapshot('non_existent_snapshot')

        # Verify result
        self.assertFalse(result)

    @patch('shutil.rmtree')
    async def test_delete_snapshot_error_handling(self, mock_rmtree):
        """Test error handling when deleting a snapshot."""
        # Setup mock to raise exception
        mock_rmtree.side_effect = Exception("Test error")

        # Create test snapshot directory
        snapshot_id = 'snapshot_20230101_123045'
        snapshot_dir = os.path.join(self.temp_dir, 'snapshots', snapshot_id)
        os.makedirs(snapshot_dir, exist_ok=True)

        # Call method
        result = await self.snapshot.delete_snapshot(snapshot_id)

        # Verify result
        self.assertFalse(result)
        # Verify error event was published
        self.event_bus.publish.assert_called_once()

    async def test_cleanup_old_snapshots(self):
        """Test cleaning up old snapshots based on retention policy."""
        # Create test snapshot directories with different dates
        # Recent snapshots
        for i in range(1, 4):
            snapshot_id = f'snapshot_20230{i}01_123045'
            snapshot_dir = os.path.join(self.temp_dir, 'snapshots', snapshot_id)
            os.makedirs(snapshot_dir, exist_ok=True)

            with open(os.path.join(snapshot_dir, 'metadata.json'), 'w') as f:
                json.dump({
                    'id': snapshot_id,
                    'timestamp': f'2023-0{i}-01T12:30:45',
                    'instruments': ['BTC-USD'],
                    'description': f'Test snapshot {i}'
                }, f)

        # Old snapshot (beyond retention policy)
        old_date = datetime.now() - timedelta(days=40)
        old_snapshot_id = f'snapshot_{old_date.strftime("%Y%m%d")}_123045'
        old_snapshot_dir = os.path.join(self.temp_dir, 'snapshots', old_snapshot_id)
        os.makedirs(old_snapshot_dir, exist_ok=True)

        with open(os.path.join(old_snapshot_dir, 'metadata.json'), 'w') as f:
            json.dump({
                'id': old_snapshot_id,
                'timestamp': old_date.isoformat(),
                'instruments': ['BTC-USD'],
                'description': 'Old test snapshot'
            }, f)

        # Call method
        deleted_count = await self.snapshot.cleanup_old_snapshots()

        # Verify result
        self.assertEqual(deleted_count, 1)
        self.assertFalse(os.path.exists(old_snapshot_dir))
        # Verify recent snapshots still exist
        for i in range(1, 4):
            self.assertTrue(os.path.exists(os.path.join(
                self.temp_dir, 'snapshots', f'snapshot_20230{i}01_123045'
            )))

    async def test_cleanup_excess_snapshots(self):
        """Test cleaning up excess snapshots beyond the maximum limit."""
        # Override max_snapshots for this test
        self.snapshot.max_snapshots = 2

        # Create test snapshot directories
        for i in range(1, 5):  # Create 4 snapshots, with max set to 2
            snapshot_id = f'snapshot_20230{i}01_123045'
            snapshot_dir = os.path.join(self.temp_dir, 'snapshots', snapshot_id)
            os.makedirs(snapshot_dir, exist_ok=True)

            with open(os.path.join(snapshot_dir, 'metadata.json'), 'w') as f:
                json.dump({
                    'id': snapshot_id,
                    'timestamp': f'2023-0{i}-01T12:30:45',
                    'instruments': ['BTC-USD'],
                    'description': f'Test snapshot {i}'
                }, f)

        # Call method
        deleted_count = await self.snapshot.cleanup_old_snapshots()

        # Verify result
        self.assertEqual(deleted_count, 2)  # Should delete the oldest 2 snapshots

        # Verify only the 2 newest snapshots remain
        self.assertFalse(os.path.exists(os.path.join(self.temp_dir, 'snapshots', 'snapshot_20230101_123045')))
        self.assertFalse(os.path.exists(os.path.join(self.temp_dir, 'snapshots', 'snapshot_20230201_123045')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'snapshots', 'snapshot_20230301_123045')))
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, 'snapshots', 'snapshot_20230401_123045')))

    @patch('pandas.DataFrame.to_parquet')
    @patch('pandas.DataFrame.to_hdf')
    def test_data_storage_methods(self, mock_to_hdf, mock_to_parquet):
        """Test the data storage methods for different formats."""
        # Test writing parquet
        self.snapshot._write_parquet(self.test_df, Path('test.parquet'))
        mock_to_parquet.assert_called_once_with(Path('test.parquet'))

        # Test writing HDF5
        self.snapshot._write_hdf5(self.test_df, Path('test.h5'), 'test_key')
        mock_to_hdf.assert_called_once_with(Path('test.h5'), key='test_key', mode='w')

    @patch('pandas.read_parquet')
    @patch('pandas.read_hdf')
    def test_data_loading_methods(self, mock_read_hdf, mock_read_parquet):
        """Test the data loading methods for different formats."""
        # Setup mocks
        mock_read_parquet.return_value = self.test_df
        mock_read_hdf.return_value = self.test_depth_data

        # Test reading parquet
        result = self.snapshot._read_parquet(Path('test.parquet'))
        self.assertEqual(result.equals(self.test_df), True)
        mock_read_parquet.assert_called_once_with(Path('test.parquet'))

        # Test reading HDF5
        result = self.snapshot._read_hdf5(Path('test.h5'), 'test_key')
        self.assertEqual(result.equals(self.test_depth_data), True)
        mock_read_hdf.assert_called_once_with(Path('test.h5'), key='test_key')

    def test_get_snapshot_path(self):
        """Test building snapshot file paths."""
        # Call method
        path = self.snapshot._get_snapshot_path('test_snapshot', 'BTC-USD', DataType.OHLCV)

        # Verify result
        expected_path = Path(os.path.join(
            self.temp_dir,
            'snapshots',
            'test_snapshot',
            'BTC-USD.parquet'
        ))
        self.assertEqual(path, expected_path)

        # Test with DEPTH data type (should use HDF5)
        path = self.snapshot._get_snapshot_path('test_snapshot', 'BTC-USD', DataType.DEPTH)
        expected_path = Path(os.path.join(
            self.temp_dir,
            'snapshots',
            'test_snapshot',
            'BTC-USD.h5'
        ))
        self.assertEqual(path, expected_path)

    def test_get_file_extension(self):
        """Test getting file extensions for different data types."""
        # Test for different data types
        self.assertEqual(self.snapshot._get_file_extension(DataType.OHLCV), '.parquet')
        self.assertEqual(self.snapshot._get_file_extension(DataType.DEPTH), '.h5')
        self.assertEqual(self.snapshot._get_file_extension(DataType.QUOTES), '.parquet')
        self.assertEqual(self.snapshot._get_file_extension(DataType.TRADES), '.parquet')

    @patch('builtins.open', new_callable=MagicMock)
    def test_get_snapshot_metadata(self, mock_open):
        """Test retrieving snapshot metadata."""
        # Setup mock file content
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = json.dumps({
            'id': 'test_snapshot',
            'timestamp': '2023-01-01T12:30:45',
            'instruments': ['BTC-USD']
        })
        mock_open.return_value = mock_file

        # Call method
        metadata = self.snapshot._get_snapshot_metadata('test_snapshot')

        # Verify result
        self.assertEqual(metadata['id'], 'test_snapshot')
        self.assertEqual(metadata['timestamp'], '2023-01-01T12:30:45')
        self.assertEqual(metadata['instruments'], ['BTC-USD'])

    @patch('builtins.open', side_effect=Exception("File not found"))
    def test_get_snapshot_metadata_error(self, mock_open):
        """Test error handling when retrieving snapshot metadata."""
        # Call method
        metadata = self.snapshot._get_snapshot_metadata('non_existent_snapshot')

        # Verify result is empty dict on error
        self.assertEqual(metadata, {})

    def test_generate_snapshot_id(self):
        """Test generating unique snapshot IDs."""
        # Call method with fixed timestamp
        timestamp = datetime(2023, 1, 1, 12, 30, 45)
        snapshot_id = self.snapshot._generate_snapshot_id(timestamp)

        # Verify result format
        self.assertEqual(snapshot_id, "snapshot_20230101_123045")

        # Generate another ID with different timestamp
        another_id = self.snapshot._generate_snapshot_id(datetime(2023, 2, 15, 9, 5, 30))
        self.assertEqual(another_id, "snapshot_20230215_090530")

        # Verify IDs are different
        self.assertNotEqual(snapshot_id, another_id)


if __name__ == '__main__':
    unittest.main()