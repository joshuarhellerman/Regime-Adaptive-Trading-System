import unittest
import os
import shutil
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
from unittest.mock import MagicMock, patch

# Import the FeatureStore class
from data.storage.feature_store import FeatureStore
from data.storage.time_series_store import TimeSeriesStore
from data.processors.data_integrity import DataValidator


class TestFeatureStore(unittest.TestCase):
    """Unit tests for the FeatureStore class."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        
        # Mock dependencies
        self.mock_time_series_store = MagicMock(spec=TimeSeriesStore)
        self.mock_data_validator = MagicMock(spec=DataValidator)
        self.mock_data_validator.validate_dataframe.return_value = True
        
        # Initialize the feature store with test directory
        self.feature_store = FeatureStore(
            base_path=self.test_dir,
            time_series_store=self.mock_time_series_store,
            data_validator=self.mock_data_validator,
            enable_caching=True,
            cache_ttl=3600,
            enable_versioning=True,
            compression=True
        )
        
        # Sample data for testing
        dates = pd.date_range(start='2023-01-01', periods=10)
        self.sample_data = pd.DataFrame({
            'open': np.random.rand(10),
            'high': np.random.rand(10),
            'low': np.random.rand(10),
            'close': np.random.rand(10),
            'volume': np.random.randint(1000, 10000, 10),
            'feature1': np.random.rand(10),
            'feature2': np.random.rand(10),
            'feature3': np.random.rand(10)
        }, index=dates)
        
        self.symbol = 'AAPL'
        self.timeframe = '1d'
        self.feature_set = 'technical_indicators'
        self.key = f"{self.symbol}_{self.timeframe}_{self.feature_set}"
        
    def tearDown(self):
        """Clean up after each test."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)

    def test_init(self):
        """Test initialization of the FeatureStore."""
        # Check that directories are created
        for path in [
            os.path.join(self.test_dir, 'data'),
            os.path.join(self.test_dir, 'metadata'),
            os.path.join(self.test_dir, 'versions')
        ]:
            self.assertTrue(os.path.exists(path))
        
        # Check that the feature registry is initialized
        self.assertIsInstance(self.feature_store.feature_registry, dict)

    def test_store_features(self):
        """Test storing features in the feature store."""
        # Store sample data
        result = self.feature_store.store_features(
            data=self.sample_data,
            symbol=self.symbol,
            timeframe=self.timeframe,
            feature_set=self.feature_set,
            metadata={'source': 'test'}
        )
        
        # Check result
        self.assertTrue(result)
        
        # Verify time series store was called
        self.mock_time_series_store.store.assert_called_once()
        
        # Check that metadata was stored
        metadata_path = os.path.join(self.test_dir, 'metadata', f"{self.key}.json")
        self.assertTrue(os.path.exists(metadata_path))
        
        # Verify the content of metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            self.assertEqual(metadata['symbol'], self.symbol)
            self.assertEqual(metadata['timeframe'], self.timeframe)
            self.assertEqual(metadata['feature_set'], self.feature_set)
            self.assertEqual(metadata['source'], 'test')
            self.assertEqual(len(metadata['columns']), 8)
            self.assertEqual(len(metadata['features']), 3)
            
    def test_get_features(self):
        """Test retrieving features from the feature store."""
        # Setup: store sample data first
        self.feature_store.store_features(
            data=self.sample_data,
            symbol=self.symbol,
            timeframe=self.timeframe,
            feature_set=self.feature_set
        )
        
        # Set up mock to return the data
        self.mock_time_series_store.retrieve.return_value = self.sample_data
        
        # Test retrieval
        result = self.feature_store.get_features(
            symbol=self.symbol,
            timeframe=self.timeframe,
            feature_set=self.feature_set
        )
        
        # Check result
        self.assertIsNotNone(result)
        self.mock_time_series_store.retrieve.assert_called_once()
        self.assertEqual(len(result), len(self.sample_data))
        
    def test_get_features_with_filters(self):
        """Test retrieving features with time and column filters."""
        # Setup: store sample data first
        self.feature_store.store_features(
            data=self.sample_data,
            symbol=self.symbol,
            timeframe=self.timeframe,
            feature_set=self.feature_set
        )
        
        # Set up mock to return the data
        self.mock_time_series_store.retrieve.return_value = self.sample_data
        
        # Test retrieval with filters
        start_time = self.sample_data.index[2]
        end_time = self.sample_data.index[7]
        columns = ['feature1', 'feature2']
        
        result = self.feature_store.get_features(
            symbol=self.symbol,
            timeframe=self.timeframe,
            feature_set=self.feature_set,
            start_time=start_time,
            end_time=end_time,
            columns=columns
        )
        
        # Check result
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 6)  # 6 rows between indices 2 and 7, inclusive
        self.assertEqual(len(result.columns), 2)  # Only feature1 and feature2
        
    def test_update_features(self):
        """Test updating existing features."""
        # Setup: store initial data
        self.feature_store.store_features(
            data=self.sample_data,
            symbol=self.symbol,
            timeframe=self.timeframe,
            feature_set=self.feature_set
        )
        
        # New data with some overlap
        new_dates = pd.date_range(start='2023-01-05', periods=10)
        new_data = pd.DataFrame({
            'open': np.random.rand(10),
            'high': np.random.rand(10),
            'low': np.random.rand(10),
            'close': np.random.rand(10),
            'volume': np.random.randint(1000, 10000, 10),
            'feature1': np.random.rand(10),
            'feature2': np.random.rand(10),
            'feature3': np.random.rand(10),
            'feature4': np.random.rand(10)  # New feature
        }, index=new_dates)
        
        # Set up mocks
        self.mock_time_series_store.retrieve.return_value = self.sample_data
        
        # Update features
        result = self.feature_store.update_features(
            data=new_data,
            symbol=self.symbol,
            timeframe=self.timeframe,
            feature_set=self.feature_set
        )
        
        # Check result
        self.assertTrue(result)
        
        # Verify that both retrieve and store were called
        self.mock_time_series_store.retrieve.assert_called_once()
        self.assertEqual(self.mock_time_series_store.store.call_count, 2)  # Initial store + update
        
    def test_get_available_features(self):
        """Test getting available features."""
        # Setup: store sample data
        self.feature_store.store_features(
            data=self.sample_data,
            symbol=self.symbol,
            timeframe=self.timeframe,
            feature_set=self.feature_set
        )
        
        # Test getting available features
        result = self.feature_store.get_available_features()
        
        # Check result
        self.assertIn(self.key, result)
        self.assertEqual(len(result[self.key]), 3)  # feature1, feature2, feature3
        
        # Test with filter
        result = self.feature_store.get_available_features(
            symbol=self.symbol,
            timeframe=self.timeframe
        )
        
        # Check filtered result
        self.assertIn(self.key, result)
        
        # Test with non-matching filter
        result = self.feature_store.get_available_features(
            symbol='NONEXISTENT'
        )
        
        # Check empty result
        self.assertEqual(len(result), 0)

    def test_get_metadata(self):
        """Test retrieving metadata."""
        # Setup: store sample data with metadata
        metadata = {'source': 'test', 'description': 'Test data'}
        self.feature_store.store_features(
            data=self.sample_data,
            symbol=self.symbol,
            timeframe=self.timeframe,
            feature_set=self.feature_set,
            metadata=metadata
        )
        
        # Test getting metadata
        result = self.feature_store.get_metadata(
            symbol=self.symbol,
            timeframe=self.timeframe,
            feature_set=self.feature_set
        )
        
        # Check result
        self.assertIsNotNone(result)
        self.assertEqual(result['source'], 'test')
        self.assertEqual(result['description'], 'Test data')
        
    def test_update_feature_metadata(self):
        """Test updating feature metadata."""
        # Setup: store sample data
        self.feature_store.store_features(
            data=self.sample_data,
            symbol=self.symbol,
            timeframe=self.timeframe,
            feature_set=self.feature_set
        )
        
        # Test updating metadata for specific features
        new_metadata = {
            'description': 'Updated feature',
            'importance': {'feature1': 'high', 'feature2': 'medium', 'feature3': 'low'}
        }
        
        result = self.feature_store.update_feature_metadata(
            features=['feature1', 'feature2'],
            metadata=new_metadata,
            symbol=self.symbol,
            timeframe=self.timeframe,
            feature_set=self.feature_set
        )
        
        # Check result
        self.assertTrue(result)
        
        # Verify metadata was updated
        metadata = self.feature_store.get_metadata(
            symbol=self.symbol,
            timeframe=self.timeframe,
            feature_set=self.feature_set
        )
        
        self.assertIn('feature_metadata', metadata)
        self.assertIn('feature1', metadata['feature_metadata'])
        self.assertIn('feature2', metadata['feature_metadata'])
        self.assertEqual(metadata['feature_metadata']['feature1']['importance'], 'high')
        
    def test_delete_features(self):
        """Test deleting features."""
        # Setup: store sample data
        self.feature_store.store_features(
            data=self.sample_data,
            symbol=self.symbol,
            timeframe=self.timeframe,
            feature_set=self.feature_set
        )
        
        # Test deleting features
        result = self.feature_store.delete_features(
            symbol=self.symbol,
            timeframe=self.timeframe,
            feature_set=self.feature_set
        )
        
        # Check result
        self.assertTrue(result)
        
        # Verify time series store delete was called
        self.mock_time_series_store.delete.assert_called_once()
        
        # Verify metadata file was deleted
        metadata_path = os.path.join(self.test_dir, 'metadata', f"{self.key}.json")
        self.assertFalse(os.path.exists(metadata_path))
        
    def test_get_performance_stats(self):
        """Test getting performance statistics."""
        # Perform some operations to generate stats
        self.feature_store.store_features(
            data=self.sample_data,
            symbol=self.symbol,
            timeframe=self.timeframe,
            feature_set=self.feature_set
        )
        
        self.mock_time_series_store.retrieve.return_value = self.sample_data
        
        self.feature_store.get_features(
            symbol=self.symbol,
            timeframe=self.timeframe,
            feature_set=self.feature_set
        )
        
        # Get performance stats
        stats = self.feature_store.get_performance_stats()
        
        # Check stats
        self.assertIn('reads', stats)
        self.assertIn('writes', stats)
        self.assertIn('cache_hits', stats)
        self.assertIn('cache_misses', stats)
        self.assertIn('cache_size', stats)
        
    def test_clear_cache(self):
        """Test clearing the cache."""
        # Setup: store and retrieve to populate cache
        self.feature_store.store_features(
            data=self.sample_data,
            symbol=self.symbol,
            timeframe=self.timeframe,
            feature_set=self.feature_set
        )
        
        self.mock_time_series_store.retrieve.return_value = self.sample_data
        
        self.feature_store.get_features(
            symbol=self.symbol,
            timeframe=self.timeframe,
            feature_set=self.feature_set
        )
        
        # Verify cache is populated
        self.assertGreater(len(self.feature_store.feature_cache), 0)
        
        # Clear cache
        self.feature_store.clear_cache()
        
        # Verify cache is empty
        self.assertEqual(len(self.feature_store.feature_cache), 0)
        self.assertEqual(len(self.feature_store.metadata_cache), 0)
        self.assertEqual(len(self.feature_store.cache_timestamps), 0)
        
    def test_get_feature_versions(self):
        """Test getting feature versions."""
        # Setup: store data with explicit version
        version = "test_version_1"
        self.feature_store.store_features(
            data=self.sample_data,
            symbol=self.symbol,
            timeframe=self.timeframe,
            feature_set=self.feature_set,
            version=version
        )
        
        # Mock the os.listdir function to return our version files
        with patch('os.listdir', return_value=[f"{version}.pkl"]):
            # Test getting versions
            versions = self.feature_store.get_feature_versions(
                symbol=self.symbol,
                timeframe=self.timeframe,
                feature_set=self.feature_set
            )
            
            # Check result
            self.assertEqual(len(versions), 1)
            self.assertEqual(versions[0], version)
            
    def test_cache_behavior(self):
        """Test cache behavior and TTL."""
        # Setup: store data
        self.feature_store.store_features(
            data=self.sample_data,
            symbol=self.symbol,
            timeframe=self.timeframe,
            feature_set=self.feature_set
        )
        
        # Setup mock
        self.mock_time_series_store.retrieve.return_value = self.sample_data
        
        # First retrieval should miss cache
        self.feature_store.get_features(
            symbol=self.symbol,
            timeframe=self.timeframe,
            feature_set=self.feature_set
        )
        
        # Cache should be populated now
        self.assertIn(self.key, self.feature_store.feature_cache)
        
        # Second retrieval should hit cache
        self.feature_store.get_features(
            symbol=self.symbol,
            timeframe=self.timeframe,
            feature_set=self.feature_set
        )
        
        # Verify retrieve was called only once (for the first call)
        self.assertEqual(self.mock_time_series_store.retrieve.call_count, 1)
        
        # Mock time.time to simulate TTL expiry
        with patch('time.time', return_value=time.time() + 4000):  # TTL is 3600
            # This retrieval should miss cache due to TTL expiry
            self.feature_store.get_features(
                symbol=self.symbol,
                timeframe=self.timeframe,
                feature_set=self.feature_set
            )
            
            # Verify retrieve was called again
            self.assertEqual(self.mock_time_series_store.retrieve.call_count, 2)
            
    def test_data_validation(self):
        """Test data validation during store."""
        # Set validator to fail
        self.mock_data_validator.validate_dataframe.return_value = False
        
        # Attempt to store data
        result = self.feature_store.store_features(
            data=self.sample_data,
            symbol=self.symbol,
            timeframe=self.timeframe,
            feature_set=self.feature_set
        )
        
        # Check result
        self.assertFalse(result)
        
        # Verify validator was called
        self.mock_data_validator.validate_dataframe.assert_called_once()
        
        # Verify store was not called
        self.mock_time_series_store.store.assert_not_called()
        
    def test_store_empty_dataframe(self):
        """Test storing an empty DataFrame."""
        # Attempt to store empty data
        empty_data = pd.DataFrame()
        result = self.feature_store.store_features(
            data=empty_data,
            symbol=self.symbol,
            timeframe=self.timeframe,
            feature_set=self.feature_set
        )
        
        # Check result
        self.assertFalse(result)
        
        # Verify store was not called
        self.mock_time_series_store.store.assert_not_called()
        
    def test_get_nonexistent_features(self):
        """Test retrieving features that don't exist."""
        # Mock retrieve to return None (not found)
        self.mock_time_series_store.retrieve.return_value = None
        
        # Attempt to get non-existent features
        result = self.feature_store.get_features(
            symbol='NONEXISTENT',
            timeframe=self.timeframe,
            feature_set=self.feature_set
        )
        
        # Check result
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()