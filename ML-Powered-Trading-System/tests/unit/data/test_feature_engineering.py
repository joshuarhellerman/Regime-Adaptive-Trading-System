"""
Unit tests for feature_engineering.py
"""

import unittest
from unittest.mock import patch, MagicMock, Mock
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from data.processors.feature_engineering import FeatureEngineering
from data.processors.feature_generator import FeatureGenerator
from data.storage.feature_store import FeatureStore
from data.processors.data_integrity import DataIntegrityChecker
from data.processors.tick_aggregator import TickAggregator


class TestFeatureEngineering(unittest.TestCase):
    """Test suite for FeatureEngineering class"""

    def setUp(self):
        """Set up test fixtures before each test"""
        self.feature_config = {
            'default': {
                'features': ['price_sma', 'price_ema', 'volume_vwap'],
                'description': 'Default feature set'
            }
        }

        # Create mock dependencies
        self.mock_feature_store = MagicMock(spec=FeatureStore)
        self.mock_tick_aggregator = MagicMock(spec=TickAggregator)
        self.mock_data_integrity_checker = MagicMock(spec=DataIntegrityChecker)

        # Set up sample data
        self.sample_dates = pd.date_range(start='2023-01-01', periods=100, freq='H')
        self.sample_data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(100, 10000, 100)
        }, index=self.sample_dates)

        # Initialize feature engineering object
        self.fe = FeatureEngineering(
            feature_config=self.feature_config,
            use_feature_store=True,
            feature_store=self.mock_feature_store,
            data_integrity_checker=self.mock_data_integrity_checker,
            logging_level=logging.ERROR
        )

        # Replace feature generator with mock
        self.fe.feature_generator = MagicMock(spec=FeatureGenerator)

        # Configure mock method returns
        self.mock_data_integrity_checker.check_and_fix.return_value = self.sample_data
        self.fe.feature_generator.generate_features.return_value = self._create_sample_features()
        self.fe.feature_generator.update_features.return_value = self._create_sample_features()

    def _create_sample_features(self):
        """Helper method to create sample dataframe with features"""
        df = self.sample_data.copy()
        df['price_sma'] = df['close'].rolling(window=5).mean()
        df['price_ema'] = df['close'].ewm(span=5).mean()
        df['volume_vwap'] = df['close'] * df['volume'] / df['volume'].rolling(window=5).sum()
        df['price_returns'] = df['close'].pct_change()
        return df

    def test_init(self):
        """Test initialization of FeatureEngineering class"""
        fe = FeatureEngineering(feature_config=self.feature_config)
        self.assertEqual(fe.feature_config, self.feature_config)
        self.assertIsInstance(fe.feature_generator, FeatureGenerator)
        self.assertEqual(fe.active_feature_sets, set())
        self.assertEqual(fe.feature_sets, {})

    def test_process_data_empty_dataframe(self):
        """Test process_data with empty dataframe"""
        empty_df = pd.DataFrame()
        result = self.fe.process_data(empty_df, symbol='BTC-USD')
        self.assertTrue(result.empty)

    def test_process_data_new_data_no_cache(self):
        """Test process_data with new data when nothing is in cache"""
        self.mock_feature_store.get_features.return_value = None

        result = self.fe.process_data(
            self.sample_data,
            symbol='BTC-USD',
            feature_set_name='test_set',
            timeframe='1h'
        )

        # Verify feature generation was called
        self.fe.feature_generator.generate_features.assert_called_once()

        # Verify result is the DataFrame returned by feature generator
        self.assertEqual(result.equals(self._create_sample_features()), True)

        # Verify feature store interaction
        self.mock_feature_store.get_features.assert_called_once()
        self.mock_feature_store.store_features.assert_called_once()

    def test_process_data_with_cached_data_no_update_needed(self):
        """Test process_data with data already in cache and no update needed"""
        cached_data = self._create_sample_features()
        # Make cached data cover the same time period as sample data
        self.mock_feature_store.get_features.return_value = cached_data

        result = self.fe.process_data(
            self.sample_data,
            symbol='BTC-USD',
            feature_set_name='test_set',
            timeframe='1h'
        )

        # Verify feature generation was not called
        self.fe.feature_generator.generate_features.assert_not_called()

        # Verify store_features was not called (no updates)
        self.mock_feature_store.store_features.assert_not_called()

        # Verify result is cached data
        self.assertEqual(result.equals(cached_data), True)

    def test_process_data_with_cached_data_update_needed(self):
        """Test process_data with cached data that needs updating"""
        # Create cached data that's missing the last 10 rows
        cached_data = self._create_sample_features().iloc[:-10]
        self.mock_feature_store.get_features.return_value = cached_data

        result = self.fe.process_data(
            self.sample_data,
            symbol='BTC-USD',
            feature_set_name='test_set',
            timeframe='1h'
        )

        # Verify feature generation was called
        self.fe.feature_generator.generate_features.assert_called_once()

        # Verify update_features was called to merge old and new
        self.fe.feature_generator.update_features.assert_called_once()

        # Verify store_features was called with updated data
        self.mock_feature_store.store_features.assert_called_once()

    def test_process_data_without_feature_store(self):
        """Test process_data when not using feature store"""
        # Create new feature engineering object without feature store
        fe = FeatureEngineering(
            feature_config=self.feature_config,
            use_feature_store=False,
            data_integrity_checker=self.mock_data_integrity_checker,
            logging_level=logging.ERROR
        )

        # Replace feature generator with mock
        fe.feature_generator = MagicMock(spec=FeatureGenerator)
        fe.feature_generator.generate_features.return_value = self._create_sample_features()

        result = fe.process_data(
            self.sample_data,
            symbol='BTC-USD',
            feature_set_name='test_set',
            timeframe='1h'
        )

        # Verify feature generation was called
        fe.feature_generator.generate_features.assert_called_once()

        # Verify feature set was registered
        self.assertIn('BTC-USD_1h_test_set', fe.feature_sets)

    def test_generate_features(self):
        """Test generate_features method"""
        self.fe.generate_features(self.sample_data)

        # Verify data integrity check was called
        self.mock_data_integrity_checker.check_and_fix.assert_called_once_with(self.sample_data)

        # Verify feature generator was called
        self.fe.feature_generator.generate_features.assert_called_once()

    def test_generate_features_with_tick_data(self):
        """Test generate_features with tick data"""
        # Set up FeatureEngineering to use tick data
        fe = FeatureEngineering(
            feature_config=self.feature_config,
            use_tick_data=True,
            tick_aggregator=self.mock_tick_aggregator,
            data_integrity_checker=self.mock_data_integrity_checker,
            logging_level=logging.ERROR
        )

        # Replace feature generator with mock
        fe.feature_generator = MagicMock(spec=FeatureGenerator)
        fe.feature_generator.generate_features.return_value = self._create_sample_features()

        # Create sample data with tick data attribute
        sample_data_with_ticks = self.sample_data.copy()
        sample_data_with_ticks.attrs['tick_data'] = pd.DataFrame({
            'price': np.random.randn(1000).cumsum() + 100,
            'volume': np.random.randint(1, 100, 1000),
            'timestamp': pd.date_range(start='2023-01-01', periods=1000, freq='T')
        })

        # Configure mock tick aggregator
        self.mock_tick_aggregator.augment_with_tick_features.return_value = sample_data_with_ticks

        fe.generate_features(sample_data_with_ticks)

        # Verify tick aggregator was called
        self.mock_tick_aggregator.augment_with_tick_features.assert_called_once()

    def test_generate_features_with_include_exclude(self):
        """Test generate_features with include and exclude filters"""
        # Set up feature generator to return dataframe with many features
        sample_features = self._create_sample_features()
        sample_features['extra_feature_1'] = 1
        sample_features['extra_feature_2'] = 2
        self.fe.feature_generator.generate_features.return_value = sample_features

        # Call with include filter
        result = self.fe.generate_features(
            self.sample_data,
            include_features=['price_sma', 'price_ema']
        )

        # Verify feature generation was called
        self.fe.feature_generator.generate_features.assert_called()

        # Call with exclude filter
        self.fe.feature_generator.generate_features.reset_mock()
        result = self.fe.generate_features(
            self.sample_data,
            exclude_features=['extra_feature_1', 'extra_feature_2']
        )

        # Verify feature generation was called
        self.fe.feature_generator.generate_features.assert_called()

    def test_update_features(self):
        """Test update_features method"""
        existing_data = self._create_sample_features().iloc[:-10]
        new_data = self._create_sample_features().iloc[-10:]

        self.fe.update_features(existing_data, new_data)

        # Verify feature generator update_features was called
        self.fe.feature_generator.update_features.assert_called_once_with(existing_data, new_data)

    def test_process_streaming_data_with_context(self):
        """Test process_streaming_data with context data"""
        data_point = {
            'timestamp': datetime.now(),
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 1000
        }

        context_data = self._create_sample_features()

        # Configure mock for generate_features to return dataframe with single row
        single_row_result = pd.DataFrame([{
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 1000,
            'price_sma': 100.2,
            'price_ema': 100.3,
            'volume_vwap': 100.4
        }], index=[pd.Timestamp(data_point['timestamp'])])

        self.fe.generate_features = MagicMock(return_value=single_row_result)

        result = self.fe.process_streaming_data(
            data_point,
            symbol='BTC-USD',
            context_data=context_data
        )

        # Verify generate_features was called
        self.fe.generate_features.assert_called_once()

        # Verify result has expected features
        self.assertIn('price_sma', result)
        self.assertIn('price_ema', result)
        self.assertIn('volume_vwap', result)

    def test_process_streaming_data_without_context(self):
        """Test process_streaming_data without context data"""
        data_point = {
            'timestamp': datetime.now(),
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 1000
        }

        # Configure mock for get_features to return empty dataframe (no cache)
        self.mock_feature_store.get_features.return_value = pd.DataFrame()

        # Configure mock for generate_features to return dataframe with single row
        single_row_result = pd.DataFrame([{
            'open': 100.0,
            'high': 101.0,
            'low': 99.0,
            'close': 100.5,
            'volume': 1000,
            'price_sma': 100.2,
            'price_ema': 100.3,
            'volume_vwap': 100.4
        }], index=[pd.Timestamp(data_point['timestamp'])])

        self.fe.generate_features = MagicMock(return_value=single_row_result)

        result = self.fe.process_streaming_data(
            data_point,
            symbol='BTC-USD'
        )

        # Verify get_features was called (to try to get context)
        self.mock_feature_store.get_features.assert_called_once()

        # Verify generate_features was called
        self.fe.generate_features.assert_called_once()

    def test_feature_set_registry(self):
        """Test feature set registry methods"""
        # Register a feature set
        self.fe._register_feature_set(
            'test_set',
            ['price_sma', 'price_ema', 'volume_vwap'],
            'BTC-USD',
            '1h'
        )

        # Test get_feature_sets
        feature_sets = self.fe.get_feature_sets()
        self.assertIn('BTC-USD_1h_test_set', feature_sets)

        # Test get_active_feature_sets
        active_sets = self.fe.get_active_feature_sets()
        self.assertIn('BTC-USD_1h_test_set', active_sets)

        # Test get_available_features
        features = self.fe.get_available_features('BTC-USD', '1h', 'test_set')
        self.assertIn('price_sma', features)
        self.assertIn('price_ema', features)
        self.assertIn('volume_vwap', features)

    def test_create_delete_feature_set(self):
        """Test create_feature_set and delete_feature_set methods"""
        # Create a feature set
        self.fe.create_feature_set(
            'custom_set',
            ['price_sma', 'price_ema', 'custom_feature'],
            'A custom feature set'
        )

        # Verify it was added to feature_config
        self.assertIn('custom_set', self.fe.feature_config)
        self.assertEqual(
            self.fe.feature_config['custom_set']['features'],
            ['price_sma', 'price_ema', 'custom_feature']
        )

        # Delete the feature set
        result = self.fe.delete_feature_set('custom_set')

        # Verify it was removed
        self.assertTrue(result)
        self.assertNotIn('custom_set', self.fe.feature_config)

        # Try to delete non-existent set
        result = self.fe.delete_feature_set('non_existent')
        self.assertFalse(result)

    def test_get_feature_importance(self):
        """Test get_feature_importance method"""
        # Create sample data with features and returns
        feature_data = self._create_sample_features()
        feature_data['price_returns'] = feature_data['close'].pct_change()

        # Configure feature store to return the sample data
        self.mock_feature_store.get_features.return_value = feature_data

        importance = self.fe.get_feature_importance(
            feature_set='default',
            symbol='BTC-USD',
            timeframe='1h'
        )

        # Verify feature store was queried
        self.mock_feature_store.get_features.assert_called_once()

        # Verify we got some feature importance values
        self.assertIsInstance(importance, dict)

    def test_get_feature_metadata(self):
        """Test get_feature_metadata method"""
        # Configure feature generator to return list of features and groups
        self.fe.feature_generator.get_feature_list.return_value = [
            'price_sma', 'price_ema', 'volume_vwap'
        ]
        self.fe.feature_generator.get_feature_groups.return_value = {
            'price': ['price_sma', 'price_ema'],
            'volume': ['volume_vwap']
        }

        metadata = self.fe.get_feature_metadata()

        # Verify feature generator methods were called
        self.fe.feature_generator.get_feature_list.assert_called_once()
        self.fe.feature_generator.get_feature_groups.assert_called_once()

        # Verify metadata structure
        self.assertIn('price_sma', metadata)
        self.assertIn('price_ema', metadata)
        self.assertIn('volume_vwap', metadata)

        # Verify metadata content
        self.assertEqual(metadata['price_sma']['group'], 'price')
        self.assertEqual(metadata['volume_vwap']['group'], 'volume')

        # Verify description and requirements are present
        self.assertIn('description', metadata['price_sma'])
        self.assertIn('requirements', metadata['price_sma'])


if __name__ == '__main__':
    unittest.main()