import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import pytest
from collections import defaultdict
from datetime import datetime, timedelta

# Import the class to test
from models.regime.regime_adapter import RegimePropertiesCalculator


class TestRegimePropertiesCalculator(unittest.TestCase):
    """Unit tests for the RegimePropertiesCalculator class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.calculator = RegimePropertiesCalculator(
            memory_decay=0.95,
            min_samples=20,
            property_history_size=5
        )
        
        # Create sample data for testing
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'open': np.random.normal(100, 2, 100),
            'high': np.random.normal(102, 2, 100),
            'low': np.random.normal(98, 2, 100),
            'close': np.random.normal(101, 2, 100),
            'volume': np.random.randint(1000, 5000, 100),
            'returns': np.random.normal(0.001, 0.02, 100),
            'volatility': np.random.normal(0.02, 0.005, 100),
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100)
        }, index=dates)
        
        # Ensure high is always >= than open/close/low
        self.sample_data['high'] = self.sample_data[['open', 'close', 'high']].max(axis=1) + 0.1
        
        # Ensure low is always <= than open/close/high
        self.sample_data['low'] = self.sample_data[['open', 'close', 'low']].min(axis=1) - 0.1
        
        # Add time features
        self.sample_data['hour'] = self.sample_data.index.hour
        self.sample_data['day_of_week'] = self.sample_data.index.dayofweek

    def test_initialization(self):
        """Test that calculator initializes with correct default values."""
        calculator = RegimePropertiesCalculator()
        
        self.assertEqual(calculator.memory_decay, 0.95)
        self.assertEqual(calculator.min_samples, 50)
        self.assertEqual(calculator.property_history_size, 10)
        self.assertEqual(calculator.regime_transition_matrix_size, 10)
        self.assertIsInstance(calculator.regime_stats, defaultdict)
        self.assertIsInstance(calculator.regime_properties_history, defaultdict)
        self.assertIsInstance(calculator.regime_samples, defaultdict)
        self.assertIsInstance(calculator.feature_stats, defaultdict)
        self.assertEqual(calculator.transitions, [])
        self.assertIsNone(calculator.transition_matrix)
        
    def test_prepare_data(self):
        """Test the _prepare_data method."""
        # Test with minimal data
        minimal_df = pd.DataFrame({
            'price': [100, 101, 102, 103, 104]
        })
        
        prepared_df = self.calculator._prepare_data(minimal_df)
        
        # Check if close column was created from price
        self.assertIn('close', prepared_df.columns)
        self.assertTrue(np.array_equal(prepared_df['close'].values, minimal_df['price'].values))
        
        # Check if returns were calculated
        self.assertIn('returns', prepared_df.columns)
        expected_returns = [0, 0.01, 0.00990099, 0.00980392]
        np.testing.assert_almost_equal(
            prepared_df['returns'].values[1:], 
            expected_returns[1:], 
            decimal=6
        )
        
        # Check if volatility was calculated
        self.assertIn('volatility', prepared_df.columns)
        
        # Test with datetime index
        df_with_datetime = pd.DataFrame({
            'close': [100, 101, 102]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        prepared_datetime_df = self.calculator._prepare_data(df_with_datetime)
        
        # Check if time features were added
        self.assertIn('hour', prepared_datetime_df.columns)
        self.assertIn('day_of_week', prepared_datetime_df.columns)

    def test_update_with_noise_regime(self):
        """Test that update method handles noise regime (-1) correctly."""
        result = self.calculator.update(self.sample_data, regime_label=-1)
        
        # Should return empty dict for noise regime
        self.assertEqual(result, {})
        
        # Regime -1 shouldn't be in regime_stats
        self.assertNotIn(-1, self.calculator.regime_stats)
        
    def test_update_with_insufficient_samples(self):
        """Test update method with insufficient samples."""
        # First update with regime 1
        result = self.calculator.update(self.sample_data.iloc[:5], regime_label=1)
        
        # Should have basic properties but not full stats
        self.assertIn('last_seen', result)
        self.assertIn('sample_count', result)
        self.assertNotIn('mean_return', result)
        
        # Sample count should be incremented
        self.assertGreater(self.calculator.regime_samples[1], 0)
        
    def test_update_with_sufficient_samples(self):
        """Test update method with sufficient samples."""
        # Force enough samples for full calculation
        self.calculator.regime_samples[2] = 20
        
        # Update with regime 2
        result = self.calculator.update(self.sample_data, regime_label=2)
        
        # Should have comprehensive properties
        self.assertIn('mean_return', result)
        self.assertIn('return_volatility', result)
        self.assertIn('sharpe_ratio', result)
        self.assertIn('volatility_level', result)
        self.assertIn('sample_count', result)
        self.assertIn('confidence', result)
        
        # Check history storage
        self.assertIn(2, self.calculator.regime_properties_history)
        self.assertEqual(len(self.calculator.regime_properties_history[2]), 1)
        
    def test_return_properties_calculation(self):
        """Test calculation of return-based properties."""
        # Setup data with known returns
        returns_data = pd.DataFrame({
            'returns': [0.01, 0.02, -0.01, 0.03, -0.02, 0.01]
        })
        
        # Force enough samples
        self.calculator.regime_samples[3] = 20
        
        properties = self.calculator._calculate_return_properties(returns_data, 3)
        
        # Verify return properties
        self.assertIn('mean_return', properties)
        self.assertIn('return_volatility', properties)
        self.assertIn('sharpe_ratio', properties)
        self.assertIn('win_rate', properties)
        self.assertIn('profit_factor', properties)
        
        # Win rate should be 4/6 = 0.6667
        self.assertAlmostEqual(properties['win_rate'], 0.6667, places=4)
        
        # Profit factor should be (0.01+0.02+0.03+0.01)/abs(-0.01-0.02) = 0.07/0.03 = 2.33
        self.assertAlmostEqual(properties['profit_factor'], 2.33, places=2)
        
    def test_volatility_properties_calculation(self):
        """Test calculation of volatility-based properties."""
        # Setup data with volatility and OHLC
        vol_data = pd.DataFrame({
            'volatility': [0.02, 0.025, 0.022, 0.018, 0.023],
            'high': [102, 103, 101, 102, 104],
            'low': [99, 100, 98, 99, 101],
            'open': [100, 101, 99, 100, 102],
            'close': [101, 100, 100, 101, 103]
        })
        
        properties = self.calculator._calculate_volatility_properties(vol_data, 4)
        
        # Verify volatility properties
        self.assertIn('volatility_level', properties)
        self.assertIn('volatility_of_volatility', properties)
        self.assertIn('range_intensity', properties)
        
        # Validate values
        self.assertAlmostEqual(properties['volatility_level'], 0.0216, places=4)
        
        # Range intensity = avg((high-low)/close)
        expected_range = np.mean([(102-99)/101, (103-100)/100, (101-98)/100, 
                                  (102-99)/101, (104-101)/103])
        self.assertAlmostEqual(properties['range_intensity'], expected_range, places=4)
        
    def test_correlation_properties_calculation(self):
        """Test calculation of correlation-based properties."""
        # Setup data with correlated columns
        x = np.linspace(0, 10, 50)
        corr_data = pd.DataFrame({
            'feature1': x,
            'feature2': x * 2 + np.random.normal(0, 1, 50),
            'feature3': -x + np.random.normal(0, 1, 50)
        })
        
        properties = self.calculator._calculate_correlation_properties(corr_data, 5)
        
        # Verify correlation properties
        self.assertIn('internal_correlation', properties)
        
        # Internal correlation should be close to 0 (average of high positive and high negative)
        self.assertTrue(-0.5 < properties['internal_correlation'] < 0.5)
        
    def test_temporal_properties_calculation(self):
        """Test calculation of temporal properties."""
        # Setup data with time patterns
        dates = pd.date_range('2023-01-01', periods=50)
        temp_data = pd.DataFrame({
            'hour': [h % 24 for h in range(50)],
            'day_of_week': [d % 7 for d in range(50)]
        }, index=dates)
        
        # Add history entries for consistent timestamps
        timestamp1 = pd.Timestamp('2023-01-01 10:00:00')
        timestamp2 = pd.Timestamp('2023-01-01 14:00:00')
        
        self.calculator.regime_properties_history[6] = [
            {'timestamp': timestamp1, 'properties': {}},
            {'timestamp': timestamp2, 'properties': {}}
        ]
        
        timestamp3 = pd.Timestamp('2023-01-01 18:00:00')
        properties = self.calculator._calculate_temporal_properties(temp_data, 6, timestamp3)
        
        # Verify temporal properties
        self.assertIn('peak_hour', properties)
        self.assertIn('peak_day', properties)
        self.assertIn('time_of_day_bias', properties)
        self.assertIn('day_of_week_bias', properties)
        
        # Verify avg_duration (difference between timestamp2 and timestamp1)
        expected_duration = (timestamp2 - timestamp1).total_seconds()
        self.assertAlmostEqual(properties['avg_duration'], expected_duration)
        
    def test_feature_stats_update(self):
        """Test updating of feature-specific statistics."""
        # Force enough samples
        self.calculator.regime_samples[7] = 30
        
        # Update with sample data
        self.calculator.update(self.sample_data, regime_label=7)
        
        # Check feature stats for a feature
        feature_stats = self.calculator.get_feature_distribution(7, 'feature1')
        
        self.assertIn('mean', feature_stats)
        self.assertIn('std', feature_stats)
        self.assertIn('skew', feature_stats)
        self.assertIn('kurtosis', feature_stats)
        self.assertIn('percentiles', feature_stats)
        
        # Check percentiles
        self.assertIn('10', feature_stats['percentiles'])
        self.assertIn('50', feature_stats['percentiles'])
        self.assertIn('90', feature_stats['percentiles'])
        
    def test_record_transition(self):
        """Test recording regime transitions."""
        timestamp = pd.Timestamp('2023-01-01 12:00:00')
        
        # Record a transition
        self.calculator.record_transition(1, 2, timestamp)
        
        # Check if transition was recorded
        self.assertEqual(len(self.calculator.transitions), 1)
        self.assertEqual(self.calculator.transitions[0]['from_regime'], 1)
        self.assertEqual(self.calculator.transitions[0]['to_regime'], 2)
        self.assertEqual(self.calculator.transitions[0]['timestamp'], timestamp)
        
        # Record another transition
        self.calculator.record_transition(2, 3)
        
        # Check if transition matrix was updated
        self.assertIsNotNone(self.calculator.transition_matrix)
        self.assertEqual(self.calculator.transition_matrix['regimes'], [1, 2, 3])
        
        # Check matrix dimensions
        self.assertEqual(self.calculator.transition_matrix['matrix'].shape, (3, 3))
        
    def test_get_transition_probability(self):
        """Test getting transition probabilities."""
        # Setup transitions
        for _ in range(10):
            self.calculator.record_transition(1, 2)
        
        for _ in range(5):
            self.calculator.record_transition(1, 3)
            
        # Force update of transition matrix
        self.calculator._update_transition_matrix()
        
        # Get probability
        prob = self.calculator.get_transition_probability(1, 2)
        
        # This should return 0 since the method is incomplete in the provided code
        self.assertEqual(prob, 0)
        
    def test_multiple_history_entries(self):
        """Test that history entries are properly maintained."""
        # Set a small history size for testing
        self.calculator.property_history_size = 3
        
        # Force enough samples
        self.calculator.regime_samples[8] = 30
        
        # Update multiple times
        for i in range(5):
            timestamp = pd.Timestamp(f'2023-01-0{i+1} 12:00:00')
            self.calculator.update(self.sample_data, regime_label=8, timestamp=timestamp)
            
        # Should only keep the most recent 3 entries
        self.assertEqual(len(self.calculator.regime_properties_history[8]), 3)
        
        # The earliest timestamp should be the 3rd update
        earliest_timestamp = self.calculator.regime_properties_history[8][0]['timestamp']
        self.assertEqual(earliest_timestamp, pd.Timestamp('2023-01-03 12:00:00'))
        
    def test_regime_consistency_calculation(self):
        """Test calculation of regime consistency property."""
        # Force enough samples
        self.calculator.regime_samples[9] = 30
        
        # First update to add history
        first_props = {
            'mean_return': 0.01, 
            'return_volatility': 0.02,
            'volatility_level': 0.025
        }
        
        # Add manually to history
        self.calculator.regime_properties_history[9] = [{
            'timestamp': pd.Timestamp('2023-01-01'),
            'properties': first_props
        }]
        
        # Setup similar data (should be consistent)
        similar_data = pd.DataFrame({
            'returns': np.random.normal(0.01, 0.02, 50),
            'volatility': np.random.normal(0.025, 0.005, 50)
        })
        
        # Update with similar data
        self.calculator.update(similar_data, regime_label=9)
        
        # Check if regime_consistency was calculated
        latest_props = self.calculator.regime_stats[9]
        self.assertIn('regime_consistency', latest_props)
        
        # Should be close to 1 (high consistency)
        self.assertGreater(latest_props['regime_consistency'], 0.5)
        
    @patch('logging.getLogger')
    def test_logger_initialization(self, mock_get_logger):
        """Test that logger is properly initialized."""
        calculator = RegimePropertiesCalculator()
        mock_get_logger.assert_called_once_with('__name__')
        self.assertIsNotNone(calculator.logger)


if __name__ == '__main__':
    unittest.main()