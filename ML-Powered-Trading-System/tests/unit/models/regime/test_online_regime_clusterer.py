import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import pytest
from datetime import datetime, timedelta

from models.regime.online_regime_clusterer import OnlineRegimeClusterer


class TestOnlineRegimeClusterer(unittest.TestCase):
    """Unit tests for the OnlineRegimeClusterer class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a basic clusterer with default parameters
        self.clusterer = OnlineRegimeClusterer(
            min_cluster_size=10,  # Smaller value for testing
            min_samples=5,  # Smaller value for testing
            window_size=100,  # Smaller window for testing
            refit_interval=50  # More frequent refits for testing
        )

        # Create some mock market data
        self.generate_test_data()

    def generate_test_data(self):
        """Generate synthetic market data for testing."""
        # Create a date range
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')

        # Generate synthetic features that will form distinct clusters
        np.random.seed(42)  # For reproducibility

        # Create three regimes with different characteristics
        regime1 = np.random.normal(0, 1, (70, 5))  # Regime 1: centered at 0
        regime2 = np.random.normal(5, 1, (70, 5))  # Regime 2: centered at 5
        regime3 = np.random.normal(-5, 1, (60, 5))  # Regime 3: centered at -5

        # Combine regimes
        features = np.vstack([regime1, regime2, regime3])

        # Create feature names
        feature_names = ['volatility', 'momentum', 'correlation', 'volume', 'trend']

        # Create DataFrame with features
        self.test_data = pd.DataFrame(
            features,
            columns=feature_names,
            index=dates
        )

        # Add a timestamp column
        self.test_data['timestamp'] = dates

    def test_initialization(self):
        """Test proper initialization of OnlineRegimeClusterer."""
        # Check default parameter initialization
        clusterer = OnlineRegimeClusterer()
        self.assertEqual(clusterer.min_cluster_size, 50)
        self.assertEqual(clusterer.min_samples, 15)
        self.assertEqual(clusterer.window_size, 500)

        # Check custom parameter initialization
        custom_clusterer = OnlineRegimeClusterer(
            min_cluster_size=20,
            min_samples=10,
            window_size=300
        )
        self.assertEqual(custom_clusterer.min_cluster_size, 20)
        self.assertEqual(custom_clusterer.min_samples, 10)
        self.assertEqual(custom_clusterer.window_size, 300)

        # Check initial state
        self.assertEqual(clusterer.samples_seen, 0)
        self.assertEqual(clusterer.current_regime, -1)
        self.assertEqual(clusterer.next_regime_id, 0)
        self.assertIsNone(clusterer.data_window)
        self.assertIsNone(clusterer.clusterer)

    def test_partial_fit_initial(self):
        """Test the first call to partial_fit."""
        # Take first batch of data
        first_batch = self.test_data.iloc[:20]

        # Call partial_fit for the first time
        regime = self.clusterer.partial_fit(first_batch)

        # Check that internal state is updated correctly
        self.assertEqual(self.clusterer.samples_seen, 20)
        self.assertIsNotNone(self.clusterer.data_window)
        self.assertEqual(len(self.clusterer.data_window), 20)

        # Initial regime should be -1 (noise) because we likely don't have enough data yet
        self.assertEqual(regime, -1)

    def test_partial_fit_subsequent(self):
        """Test subsequent calls to partial_fit."""
        # First batch
        first_batch = self.test_data.iloc[:50]
        self.clusterer.partial_fit(first_batch)

        # Second batch
        second_batch = self.test_data.iloc[50:100]
        regime = self.clusterer.partial_fit(second_batch)

        # Check that window size is maintained
        self.assertEqual(len(self.clusterer.data_window), 100)

        # Check that samples seen is updated
        self.assertEqual(self.clusterer.samples_seen, 100)

        # By now we should have identified at least one regime
        # Note: the actual regime number may vary depending on the clustering
        self.assertTrue(hasattr(self.clusterer, 'labels_'))

    def test_partial_fit_with_feature_cols(self):
        """Test partial_fit with explicit feature columns."""
        # Specify only a subset of features
        feature_cols = ['volatility', 'momentum', 'trend']

        # Create a new clusterer
        clusterer = OnlineRegimeClusterer(
            min_cluster_size=10,
            min_samples=5,
            window_size=100
        )

        # Fit with only specified features
        regime = clusterer.partial_fit(self.test_data.iloc[:50], feature_cols=feature_cols)

        # Check that only specified features are used
        self.assertEqual(clusterer.feature_names, feature_cols)
        self.assertEqual(clusterer.data_window.shape[1], len(feature_cols))

    def test_predict_regime(self):
        """Test regime prediction for new data points."""
        # First fit the model with enough data
        self.clusterer.partial_fit(self.test_data.iloc[:100])

        # Then predict regime for a new point
        new_point = self.test_data.iloc[150:151]
        regime = self.clusterer._predict_regime(new_point)

        # Check that a regime is assigned (could be -1 for noise)
        self.assertIsInstance(regime, int)

    def test_update_feature_weights(self):
        """Test updating feature weights."""
        # First fit with default weights
        self.clusterer.partial_fit(self.test_data.iloc[:100])

        # Update feature weights
        weights = {'volatility': 2.0, 'momentum': 0.5}
        self.clusterer.update_feature_weights(weights)

        # Check weights are updated
        self.assertEqual(self.clusterer.feature_weights, weights)

        # Make a prediction with updated weights
        new_point = self.test_data.iloc[150:151]
        regime = self.clusterer._predict_regime(new_point)

        # We should still get a valid regime
        self.assertIsInstance(regime, int)

    def test_get_regime_stats(self):
        """Test getting regime statistics."""
        # First fit with enough data to establish regimes
        self.clusterer.partial_fit(self.test_data)

        # Get stats
        stats = self.clusterer.get_regime_stats()

        # Check structure of stats
        self.assertIn('total_regimes', stats)
        self.assertIn('active_regimes', stats)
        self.assertIn('current_regime', stats)
        self.assertIn('regime_counts', stats)
        self.assertIn('parameters', stats)

    def test_export_import_model(self):
        """Test exporting and importing model state."""
        # First fit the model
        self.clusterer.partial_fit(self.test_data.iloc[:100])

        # Export model state
        model_data = self.clusterer.export_model()

        # Check exported data structure
        self.assertIn('parameters', model_data)
        self.assertIn('state', model_data)
        self.assertIn('performance', model_data)
        self.assertIn('metadata', model_data)

        # Create a new clusterer
        new_clusterer = OnlineRegimeClusterer()

        # Import the exported model
        success = new_clusterer.import_model(model_data)

        # Check import was successful
        self.assertTrue(success)

        # Check that parameters were properly imported
        self.assertEqual(new_clusterer.min_cluster_size, self.clusterer.min_cluster_size)
        self.assertEqual(new_clusterer.min_samples, self.clusterer.min_samples)
        self.assertEqual(new_clusterer.feature_names, self.clusterer.feature_names)

    def test_optimize_parameters(self):
        """Test parameter optimization."""
        # Mock cluster performance history to trigger optimization
        self.clusterer.cluster_performance['silhouette'] = [0.2, 0.25, 0.3, 0.35, 0.4]

        # Store original parameters
        original_min_cluster_size = self.clusterer.min_cluster_size
        original_min_samples = self.clusterer.min_samples
        original_epsilon = self.clusterer.cluster_selection_epsilon

        # Trigger optimization
        self.clusterer._optimize_parameters()

        # Check optimization history is updated
        self.assertGreater(len(self.clusterer.optimization_history), 0)

        # At least one parameter should have changed
        params_changed = (
                original_min_cluster_size != self.clusterer.min_cluster_size or
                original_min_samples != self.clusterer.min_samples or
                original_epsilon != self.clusterer.cluster_selection_epsilon
        )
        self.assertTrue(params_changed)

    def test_update_regime_tracking(self):
        """Test regime transition tracking."""
        # Initialize with a specific regime
        self.clusterer.current_regime = 1

        # Update to a new regime
        timestamp = pd.Timestamp.now()
        self.clusterer._update_regime_tracking(2, timestamp)

        # Check transition was recorded
        self.assertEqual(len(self.clusterer.transition_history), 1)
        self.assertEqual(self.clusterer.transition_history[0]['from_regime'], 1)
        self.assertEqual(self.clusterer.transition_history[0]['to_regime'], 2)

        # Check regime history was updated
        self.assertEqual(len(self.clusterer.regime_history), 1)
        self.assertEqual(self.clusterer.regime_history[0]['regime'], 2)

        # Check current regime was updated
        self.assertEqual(self.clusterer.current_regime, 2)

    @patch('hdbscan.HDBSCAN')
    def test_create_clusterer(self, mock_hdbscan):
        """Test creation of HDBSCAN clusterer."""
        # Call the method
        self.clusterer._create_clusterer()

        # Check that HDBSCAN was called with correct parameters
        mock_hdbscan.assert_called_once_with(
            min_cluster_size=self.clusterer.min_cluster_size,
            min_samples=self.clusterer.min_samples,
            cluster_selection_epsilon=self.clusterer.cluster_selection_epsilon,
            metric=self.clusterer.metric,
            prediction_data=True
        )

    def test_update_regime_mapping(self):
        """Test regime mapping consistency."""
        # Need to set up clusterer with labels
        self.clusterer.clusterer = MagicMock()
        self.clusterer.clusterer.labels_ = np.array([0, 0, 1, 1, 2, 2])
        self.clusterer.data_window = pd.DataFrame(
            np.random.normal(0, 1, (6, 5)),
            columns=self.clusterer.feature_names if self.clusterer.feature_names else ['f1', 'f2', 'f3', 'f4', 'f5']
        )

        # Call update_regime_mapping
        self.clusterer._update_regime_mapping()

        # Check regime map was created
        self.assertEqual(len(self.clusterer.regime_map), 3)  # 3 clusters
        self.assertEqual(len(self.clusterer.regime_centers), 3)  # 3 centers

    def test_evaluate_clustering(self):
        """Test clustering evaluation metrics."""
        # Need clusterer with labels
        self.clusterer.clusterer = MagicMock()
        # Create fake labels with multiple clusters
        self.clusterer.clusterer.labels_ = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, -1])

        # Fake scaled data
        X_scaled = np.random.normal(0, 1, (10, 5))

        # Mock silhouette_score and calinski_harabasz_score to avoid actual computation
        with patch('models.regime.online_regime_clusterer.silhouette_score', return_value=0.7), \
                patch('models.regime.online_regime_clusterer.calinski_harabasz_score', return_value=120.5):
            # Call evaluate_clustering
            metrics = self.clusterer._evaluate_clustering(X_scaled)

            # Check metrics
            self.assertIn('noise_ratio', metrics)
            self.assertIn('silhouette', metrics)
            self.assertIn('calinski_harabasz', metrics)

            # Check performance history was updated
            self.assertEqual(self.clusterer.cluster_performance['noise_ratio'][-1], 0.1)  # 1/10 points are noise
            self.assertEqual(self.clusterer.cluster_performance['silhouette'][-1], 0.7)
            self.assertEqual(self.clusterer.cluster_performance['calinski_harabasz'][-1], 120.5)

    def test_handle_nans(self):
        """Test handling of NaN values in input data."""
        # Create data with NaNs
        data_with_nans = self.test_data.iloc[:20].copy()
        data_with_nans.iloc[5, 0] = np.nan  # Add NaN in volatility
        data_with_nans.iloc[10, 2] = np.nan  # Add NaN in correlation

        # Should not raise an exception
        try:
            self.clusterer.partial_fit(data_with_nans)
            success = True
        except Exception as e:
            success = False
            print(f"Exception: {e}")

        self.assertTrue(success)

    def test_regime_prediction_smoothing(self):
        """Test regime prediction smoothing logic."""
        # Set up prediction buffer with some values
        self.clusterer.prediction_buffer = [1, 1, 2, 1]
        self.clusterer.min_cluster_persistence = 0.6  # 60% threshold

        # Mock the approximate_predict function
        with patch('hdbscan.approximate_predict', return_value=(np.array([1]), np.array([0.8]))):
            # Mock clusterer and regime_map
            self.clusterer.clusterer = MagicMock()
            self.clusterer.regime_map = {1: 1}

            # Create a mock point
            point = pd.DataFrame([[1, 2, 3, 4, 5]],
                                 columns=['volatility', 'momentum', 'correlation', 'volume', 'trend'])

            # Call _predict_regime
            regime = self.clusterer._predict_regime(point)

            # Should update the prediction buffer
            self.assertEqual(len(self.clusterer.prediction_buffer), 5)
            self.assertEqual(self.clusterer.prediction_buffer[-1], 1)

            # Should use smoothed regime
            self.assertEqual(self.clusterer.smoothed_regime, 1)

    def test_import_model_error_handling(self):
        """Test error handling in import_model."""
        # Create invalid model data
        invalid_model = {
            'parameters': {'invalid_param': 'value'},
            'state': 'not_a_dict'
        }

        # Should return False but not raise exception
        result = self.clusterer.import_model(invalid_model)
        self.assertFalse(result)

    def test_partial_fit_input_validation(self):
        """Test input validation in partial_fit."""
        # Test with non-DataFrame input
        with self.assertRaises(TypeError):
            self.clusterer.partial_fit(np.random.random((10, 5)))

        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        regime = self.clusterer.partial_fit(empty_df)
        self.assertEqual(regime, -1)  # Should return noise

    def test_window_management(self):
        """Test proper management of the sliding window."""
        # Create clusterer with small window
        small_window_clusterer = OnlineRegimeClusterer(window_size=50)

        # Add more data than window size
        small_window_clusterer.partial_fit(self.test_data.iloc[:100])

        # Check window size is maintained
        self.assertEqual(len(small_window_clusterer.data_window), 50)

        # Check that window contains the most recent data
        pd.testing.assert_frame_equal(
            small_window_clusterer.data_window,
            self.test_data.iloc[50:100][small_window_clusterer.feature_names]
        )


if __name__ == '__main__':
    unittest.main()