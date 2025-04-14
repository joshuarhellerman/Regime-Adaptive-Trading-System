import unittest
from unittest.mock import MagicMock, patch, ANY
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import joblib
from pathlib import Path

# Import the module to test
from models.regime.regime_classifier import RegimeClassifier


class TestRegimeClassifier(unittest.TestCase):
    """Unit tests for RegimeClassifier class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for model saving/loading
        self.temp_dir = tempfile.TemporaryDirectory()
        self.models_dir = self.temp_dir.name

        # Create a basic classifier instance for tests
        self.classifier = RegimeClassifier(
            model_id="test_model",
            feature_cols=["feature1", "feature2", "feature3"],
            window_size=100,  # Smaller window for testing
            min_cluster_size=10,
            min_samples=5,
            models_dir=self.models_dir
        )

        # Create sample data
        date_range = pd.date_range(start='2023-01-01', periods=200, freq='H')
        self.sample_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 200),
            'feature2': np.random.normal(0, 1, 200),
            'feature3': np.random.normal(0, 1, 200),
            'returns': np.random.normal(0, 0.01, 200)
        }, index=date_range)

        # Sample performance metrics
        self.sample_performance = {
            'sharpe_ratio': 1.2,
            'sortino_ratio': 1.5,
            'win_rate': 0.6,
            'profit_factor': 1.3
        }

    def tearDown(self):
        """Clean up after each test method."""
        self.temp_dir.cleanup()

    def test_init(self):
        """Test initialization of the RegimeClassifier."""
        # Check default attributes
        self.assertEqual(self.classifier.model_id, "test_model")
        self.assertEqual(self.classifier.feature_cols, ["feature1", "feature2", "feature3"])
        self.assertEqual(self.classifier.current_regime, -1)
        self.assertEqual(self.classifier.samples_processed, 0)
        self.assertFalse(self.classifier.transition_in_progress)

        # Check components initialization
        self.assertIsNotNone(self.classifier.clusterer)
        self.assertIsNotNone(self.classifier.transition_detector)
        self.assertIsNotNone(self.classifier.properties_calculator)
        self.assertIsNotNone(self.classifier.regime_adaptor)

    def test_register_default_parameters(self):
        """Test that default parameters are registered."""
        # Check that some parameters exist in the regime adaptor
        params = self.classifier.regime_adaptor.get_parameter_info()
        self.assertIn('position_size_factor', params)
        self.assertIn('max_position_size', params)
        self.assertIn('profit_target_multiplier', params)
        self.assertIn('stop_loss_multiplier', params)
        self.assertIn('holding_period', params)
        self.assertIn('entry_timeout', params)
        self.assertIn('entry_threshold', params)
        self.assertIn('exit_threshold', params)

    @patch('models.regime.online_regime_clusterer.OnlineRegimeClusterer.partial_fit')
    def test_process_basic(self, mock_partial_fit):
        """Test basic processing of data without regime transitions."""
        # Mock the clusterer to return a stable regime
        mock_partial_fit.return_value = 1

        # Process some data
        result = self.classifier.process(self.sample_data)

        # Check that clusterer was called
        mock_partial_fit.assert_called_once_with(self.sample_data, feature_cols=["feature1", "feature2", "feature3"])

        # Check the result
        self.assertEqual(result['regime'], 1)
        self.assertFalse(result['transition_detected'])
        self.assertEqual(self.classifier.current_regime, 1)
        self.assertEqual(self.classifier.samples_processed, 200)

    @patch('models.regime.online_regime_clusterer.OnlineRegimeClusterer.partial_fit')
    @patch('models.regime.regime_transition_detector.RegimeTransitionDetector.update')
    def test_process_with_transition(self, mock_transition_update, mock_partial_fit):
        """Test processing data with a regime transition."""
        # Mock the clusterer to return different regimes
        mock_partial_fit.side_effect = [0, 1]  # First call returns 0, second call returns 1

        # Mock transition detector to report a transition
        mock_transition_update.return_value = (True, {"volatility_change": 2.5})

        # First call - establish initial regime
        self.classifier.process(self.sample_data.iloc[:100])
        self.assertEqual(self.classifier.current_regime, 0)

        # Second call - trigger transition
        result = self.classifier.process(self.sample_data.iloc[100:])

        # Check the result
        self.assertEqual(result['regime'], 1)
        self.assertTrue(result['transition_detected'])
        self.assertTrue(result['in_transition'])
        self.assertEqual(result['transition_info'], {"volatility_change": 2.5})
        self.assertEqual(self.classifier.current_regime, 1)
        self.assertEqual(self.classifier.last_regime, 0)

    def test_process_with_performance_metrics(self):
        """Test processing with performance feedback."""
        # Mock the regime adaptor's adapt_parameters method
        self.classifier.regime_adaptor.adapt_parameters = MagicMock(return_value={
            'position_size_factor': 0.8,
            'max_position_size': 0.05
        })

        # Set current regime
        self.classifier.current_regime = 2

        # Process with performance metrics
        result = self.classifier.process(self.sample_data, self.sample_performance)

        # Check performance metrics were stored
        self.assertEqual(self.classifier.latest_performance, self.sample_performance)

        # Check adaptor was called
        self.classifier.regime_adaptor.adapt_parameters.assert_called_once_with(
            2, self.classifier.latest_properties, self.sample_performance
        )

        # Check result contains adapted parameters
        self.assertEqual(result['adapted_parameters'], {
            'position_size_factor': 0.8,
            'max_position_size': 0.05
        })

    def test_save_and_load_model(self):
        """Test saving and loading a model."""
        # Process some data to have something to save
        self.classifier.clusterer.partial_fit = MagicMock(return_value=1)
        self.classifier.process(self.sample_data)

        # Set some state to verify after loading
        self.classifier.latest_properties = {"volatility": 0.02, "correlation": 0.3}
        self.classifier.latest_regime_params = {"position_size_factor": 0.75}
        self.classifier.latest_performance = self.sample_performance

        # Save the model
        filepath = self.classifier.save_model()
        self.assertTrue(os.path.exists(filepath))

        # Create a new classifier instance
        new_classifier = RegimeClassifier(
            model_id="test_model",
            feature_cols=["feature1", "feature2", "feature3"],
            models_dir=self.models_dir
        )

        # Load the saved model
        success = new_classifier.load_model(filepath)
        self.assertTrue(success)

        # Verify the loaded state
        self.assertEqual(new_classifier.model_id, "test_model")
        self.assertEqual(new_classifier.feature_cols, ["feature1", "feature2", "feature3"])
        self.assertEqual(new_classifier.current_regime, 1)
        self.assertEqual(new_classifier.samples_processed, 200)
        self.assertEqual(new_classifier.latest_properties, {"volatility": 0.02, "correlation": 0.3})
        self.assertEqual(new_classifier.latest_regime_params, {"position_size_factor": 0.75})
        self.assertEqual(new_classifier.latest_performance, self.sample_performance)

    def test_get_optimal_parameters(self):
        """Test getting optimal parameters for a regime."""
        # Mock the regime adaptor's get_optimal_parameters method
        self.classifier.regime_adaptor.get_optimal_parameters = MagicMock(return_value={
            'position_size_factor': 0.9,
            'max_position_size': 0.08,
            'profit_target_multiplier': 2.5
        })

        # Set current regime
        self.classifier.current_regime = 3

        # Get optimal parameters without specifying regime (should use current)
        params = self.classifier.get_optimal_parameters()
        self.classifier.regime_adaptor.get_optimal_parameters.assert_called_with(3)
        self.assertEqual(params, {
            'position_size_factor': 0.9,
            'max_position_size': 0.08,
            'profit_target_multiplier': 2.5
        })

        # Get optimal parameters for specific regime
        params = self.classifier.get_optimal_parameters(2)
        self.classifier.regime_adaptor.get_optimal_parameters.assert_called_with(2)

    def test_get_regime_properties(self):
        """Test getting properties for a regime."""
        # Setup test data
        self.classifier.properties_calculator.regime_stats = {
            1: {'volatility': 0.01, 'trend_strength': 0.7},
            2: {'volatility': 0.03, 'trend_strength': 0.2}
        }

        # Get properties for regime 1
        props = self.classifier.get_regime_properties(1)
        self.assertEqual(props, {'volatility': 0.01, 'trend_strength': 0.7})

        # Get properties for current regime
        self.classifier.current_regime = 2
        props = self.classifier.get_regime_properties()
        self.assertEqual(props, {'volatility': 0.03, 'trend_strength': 0.2})

        # Get properties for non-existent regime
        props = self.classifier.get_regime_properties(999)
        self.assertEqual(props, {})

    def test_get_all_regimes_properties(self):
        """Test getting properties for all regimes."""
        # Setup test data
        self.classifier.properties_calculator.regime_stats = {
            -1: {'noise': True},  # Noise regime should be filtered out
            0: {'volatility': 0.005, 'trend_strength': 0.9},
            1: {'volatility': 0.01, 'trend_strength': 0.7},
            2: {'volatility': 0.03, 'trend_strength': 0.2}
        }

        # Get all properties
        all_props = self.classifier.get_all_regimes_properties()
        
        # Should not include noise regime (-1)
        self.assertNotIn(-1, all_props)
        self.assertEqual(len(all_props), 3)
        self.assertEqual(all_props[0], {'volatility': 0.005, 'trend_strength': 0.9})
        self.assertEqual(all_props[1], {'volatility': 0.01, 'trend_strength': 0.7})
        self.assertEqual(all_props[2], {'volatility': 0.03, 'trend_strength': 0.2})

    def test_reset(self):
        """Test resetting the classifier state."""
        # Set some state
        self.classifier.current_regime = 2
        self.classifier.last_regime = 1
        self.classifier.regime_start_time = datetime.now()
        self.classifier.transition_in_progress = True
        self.classifier.last_transition_time = datetime.now()

        # Mock the transition detector's reset method
        self.classifier.transition_detector.reset = MagicMock()

        # Reset the classifier
        self.classifier.reset()

        # Check state was reset
        self.assertEqual(self.classifier.current_regime, -1)
        self.assertEqual(self.classifier.last_regime, -1)
        self.assertIsNone(self.classifier.regime_start_time)
        self.assertFalse(self.classifier.transition_in_progress)
        self.assertIsNone(self.classifier.last_transition_time)

        # Check transition detector was reset
        self.classifier.transition_detector.reset.assert_called_once()

    def test_feature_importance_update(self):
        """Test updating feature importance scores."""
        # Create clusterer with mock data window
        self.classifier.clusterer.data_window = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'returns': [0.01, 0.02, -0.01, -0.02, 0.01]
        })
        self.classifier.clusterer.feature_names = ['feature1', 'feature2', 'returns']

        # Mock the methods that use feature importance
        self.classifier.clusterer.update_feature_weights = MagicMock()
        self.classifier.transition_detector.update_feature_importance = MagicMock()

        # Trigger feature importance update
        self.classifier._update_feature_importance()

        # Check that feature weights were updated
        self.classifier.clusterer.update_feature_weights.assert_called_once()
        self.classifier.transition_detector.update_feature_importance.assert_called_once()

        # Check that latest_feature_importance was updated
        self.assertIn('feature1', self.classifier.latest_feature_importance)
        self.assertIn('feature2', self.classifier.latest_feature_importance)

    def test_get_transition_matrix(self):
        """Test getting the regime transition matrix."""
        # Setup mock transition data
        self.classifier.properties_calculator.transition_matrix = {
            (0, 1): 0.3,
            (1, 0): 0.2,
            (1, 2): 0.5
        }
        self.classifier.transition_detector.transition_history = [
            {'from': 0, 'to': 1, 'time': datetime.now() - timedelta(days=1)},
            {'from': 1, 'to': 2, 'time': datetime.now()}
        ]

        # Get transition matrix
        matrix_info = self.classifier.get_transition_matrix()

        # Check result structure
        self.assertIn('matrix', matrix_info)
        self.assertIn('last_transitions', matrix_info)
        self.assertEqual(matrix_info['matrix'], {
            (0, 1): 0.3,
            (1, 0): 0.2,
            (1, 2): 0.5
        })
        self.assertEqual(len(matrix_info['last_transitions']), 2)

    def test_process_invalid_data(self):
        """Test processing invalid data."""
        # Empty DataFrame
        result = self.classifier.process(pd.DataFrame())
        self.assertIn('error', result)

        # None value
        result = self.classifier.process(None)
        self.assertIn('error', result)

if __name__ == '__main__':
    unittest.main()