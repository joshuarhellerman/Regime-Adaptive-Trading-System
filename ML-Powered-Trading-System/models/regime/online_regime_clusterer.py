import numpy as np
import pandas as pd
import hdbscan
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import random
from datetime import datetime, timedelta


class OnlineRegimeClusterer:
    """
    Online market regime clusterer with automatic parameter optimization
    and incremental learning capabilities.
    """

    def __init__(
            self,
            min_cluster_size: int = 50,
            min_samples: int = 15,
            cluster_selection_epsilon: float = 0.3,
            metric: str = 'euclidean',
            feature_weights: Optional[Dict[str, float]] = None,
            window_size: int = 500,
            adaptation_rate: float = 0.05,
            refit_interval: int = 100,
            min_cluster_persistence: float = 0.7
    ):
        """
        Initialize the online regime clusterer.

        Args:
            min_cluster_size: Minimum points to form a cluster (default: 50)
            min_samples: Minimum samples for core point (default: 15)
            cluster_selection_epsilon: Cluster boundary looseness (default: 0.3)
            metric: Distance metric for clustering (default: 'euclidean')
            feature_weights: Dictionary of feature importance weights (default: None)
            window_size: Size of sliding window for clustering (default: 500)
            adaptation_rate: Learning rate for parameter optimization (default: 0.05)
            refit_interval: How often to retrain the clusterer (default: 100)
            min_cluster_persistence: Required stability for cluster validity (default: 0.7)
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric
        self.feature_weights = feature_weights or {}
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        self.refit_interval = refit_interval
        self.min_cluster_persistence = min_cluster_persistence

        # Predict using approximate_predict
        try:
            raw_labels, strengths = hdbscan.approximate_predict(
                self.clusterer, X_scaled
            )

            # Convert raw HDBSCAN labels to consistent regime IDs
            regimes = np.array([self.regime_map.get(label, -1) if label != -1 else -1
                                for label in raw_labels])

            # Apply confidence threshold using strengths
            confidence_threshold = 0.5
            regimes[strengths < confidence_threshold] = -1

            # Return the label of the first (and only) point
            raw_regime = regimes[0]

        except Exception as e:
            self.logger.warning(f"Prediction failed: {str(e)}")
            raw_regime = -1

        # Apply smoothing using a buffer of recent predictions
        self.prediction_buffer.append(raw_regime)
        if len(self.prediction_buffer) > 5:
            self.prediction_buffer.pop(0)

        # Find most common regime in buffer
        if len(self.prediction_buffer) >= 3:
            from collections import Counter
            counts = Counter(self.prediction_buffer)
            # Only change regime if new regime is dominant
            most_common = counts.most_common(1)[0][0]

            # Check if most common is consistent enough
            if counts[most_common] >= len(self.prediction_buffer) * self.min_cluster_persistence:
                self.smoothed_regime = most_common

        # Increment regime counter
        if self.smoothed_regime >= 0:
            self.regime_counts[self.smoothed_regime] += 1

        return self.smoothed_regime

    def _update_regime_tracking(self, new_regime: int, timestamp: Optional[pd.Timestamp] = None) -> None:
        """Track regime transitions and history."""
        # Use current time if no timestamp provided
        if timestamp is None:
            timestamp = pd.Timestamp.now()

        # Check for regime transition
        if new_regime != self.current_regime and self.current_regime != -1:
            self.transition_history.append({
                'timestamp': timestamp,
                'from_regime': self.current_regime,
                'to_regime': new_regime
            })

        # Update current regime
        self.current_regime = new_regime

        # Add to history
        self.regime_history.append({
            'timestamp': timestamp,
            'regime': new_regime
        })

        # Keep history at reasonable size
        if len(self.regime_history) > 1000:
            self.regime_history = self.regime_history[-1000:]

    def _optimize_parameters(self) -> None:
        """
        Optimize clustering parameters using grid search on recent performance.
        Uses a modified Bayesian optimization approach.
        """
        # Skip if not enough performance history
        if (len(self.cluster_performance['silhouette']) < 5 or
                all(s <= 0 for s in self.cluster_performance['silhouette'][-5:])):
            return

        # Current performance (average of last 3)
        current_perf = np.mean(self.cluster_performance['silhouette'][-3:])

        # Store current parameters
        current_params = {
            'min_cluster_size': self.min_cluster_size,
            'min_samples': self.min_samples,
            'cluster_selection_epsilon': self.cluster_selection_epsilon
        }

        # Record performance
        self.parameter_performance['min_cluster_size'].append((self.min_cluster_size, current_perf))
        self.parameter_performance['min_samples'].append((self.min_samples, current_perf))
        self.parameter_performance['cluster_selection_epsilon'].append((self.cluster_selection_epsilon, current_perf))

        # Focused exploration of one parameter at a time
        param_to_optimize = random.choice([
            'min_cluster_size',
            'min_samples',
            'cluster_selection_epsilon'
        ])

        new_params = current_params.copy()

        if param_to_optimize == 'min_cluster_size':
            # Try larger or smaller cluster size
            direction = 1 if random.random() > 0.5 else -1
            delta = max(5, int(self.min_cluster_size * 0.1))  # At least 5, or 10%
            new_value = self.min_cluster_size + direction * delta
            # Ensure reasonable bounds
            new_value = max(10, min(100, new_value))
            new_params['min_cluster_size'] = new_value

        elif param_to_optimize == 'min_samples':
            # Try larger or smaller min_samples
            direction = 1 if random.random() > 0.5 else -1
            delta = max(2, int(self.min_samples * 0.1))  # At least 2, or 10%
            new_value = self.min_samples + direction * delta
            # Ensure reasonable bounds
            new_value = max(5, min(50, new_value))
            new_params['min_samples'] = new_value

        elif param_to_optimize == 'cluster_selection_epsilon':
            # Try larger or smaller epsilon
            direction = 1 if random.random() > 0.5 else -1
            delta = 0.05  # Small step for epsilon
            new_value = self.cluster_selection_epsilon + direction * delta
            # Ensure reasonable bounds
            new_value = max(0.0, min(1.0, new_value))
            new_params['cluster_selection_epsilon'] = new_value

        # Save to optimization history
        self.optimization_history.append({
            'timestamp': pd.Timestamp.now(),
            'parameter': param_to_optimize,
            'old_value': current_params[param_to_optimize],
            'new_value': new_params[param_to_optimize],
            'current_performance': current_perf
        })

        # Apply new parameters
        self.min_cluster_size = new_params['min_cluster_size']
        self.min_samples = new_params['min_samples']
        self.cluster_selection_epsilon = new_params['cluster_selection_epsilon']

        # Log change
        self.logger.info(
            f"Optimized {param_to_optimize}: {current_params[param_to_optimize]} -> {new_params[param_to_optimize]}")

        # Force model refit with new parameters
        self._update_model()

        # Check if performance improved
        if len(self.cluster_performance['silhouette']) > 0:
            new_perf = self.cluster_performance['silhouette'][-1]

            # Update best parameters if improved
            if new_perf > current_perf:
                self.best_parameters = new_params.copy()
                self.logger.info(f"Performance improved: {current_perf:.4f} -> {new_perf:.4f}")
            elif new_perf < current_perf * 0.9:  # Significant degradation
                # Revert to previous parameters
                self.min_cluster_size = current_params['min_cluster_size']
                self.min_samples = current_params['min_samples']
                self.cluster_selection_epsilon = current_params['cluster_selection_epsilon']
                self.logger.info(f"Reverting parameters due to performance drop: {new_perf:.4f} < {current_perf:.4f}")
                self._update_model()

    def update_feature_weights(self, weights: Dict[str, float]) -> None:
        """
        Update feature importance weights.

        Args:
            weights: Dictionary mapping feature names to importance weights
        """
        self.feature_weights = weights.copy()

        # Force model refit with new weights
        self._update_model()

    def get_regime_stats(self) -> Dict:
        """
        Get statistics about detected regimes.

        Returns:
            Dictionary with regime statistics
        """
        stats = {
            'total_regimes': self.next_regime_id,
            'active_regimes': len([r for r in self.regime_counts.keys() if r >= 0]),
            'current_regime': self.current_regime,
            'regime_counts': dict(self.regime_counts),
            'parameters': {
                'min_cluster_size': self.min_cluster_size,
                'min_samples': self.min_samples,
                'cluster_selection_epsilon': self.cluster_selection_epsilon,
                'window_size': self.window_size
            }
        }

        # Calculate regime stability
        if len(self.regime_history) > 20:
            # Get regimes in last 20 points
            recent_regimes = [h['regime'] for h in self.regime_history[-20:]]
            # Count transitions
            transitions = sum(1 for i in range(1, len(recent_regimes))
                              if recent_regimes[i] != recent_regimes[i - 1])
            stats['regime_stability'] = 1 - (transitions / 19)  # 19 possible transitions in 20 points

        # Calculate transition frequencies
        if self.transition_history:
            # Count last 50 transitions
            recent_transitions = self.transition_history[-50:]
            transition_counts = defaultdict(int)

            for transition in recent_transitions:
                from_regime = transition['from_regime']
                to_regime = transition['to_regime']
                if from_regime >= 0 and to_regime >= 0:  # Skip noise
                    key = f"{from_regime}->{to_regime}"
                    transition_counts[key] += 1

            stats['transition_counts'] = dict(transition_counts)

            # Calculate average time between transitions
            if len(recent_transitions) > 1 and isinstance(recent_transitions[0]['timestamp'], pd.Timestamp):
                durations = []
                for i in range(1, len(recent_transitions)):
                    duration = (recent_transitions[i]['timestamp'] -
                                recent_transitions[i - 1]['timestamp']).total_seconds()
                    durations.append(duration)

                if durations:
                    stats['avg_transition_interval'] = np.mean(durations)

        return stats

    def export_model(self) -> Dict:
        """
        Export model data for persistence.

        Returns:
            Dictionary with model state
        """
        model_data = {
            'parameters': {
                'min_cluster_size': self.min_cluster_size,
                'min_samples': self.min_samples,
                'cluster_selection_epsilon': self.cluster_selection_epsilon,
                'metric': self.metric,
                'window_size': self.window_size,
                'feature_weights': self.feature_weights
            },
            'state': {
                'regime_map': self.regime_map,
                'next_regime_id': self.next_regime_id,
                'regime_centers': {k: v.tolist() for k, v in self.regime_centers.items()},
                'current_regime': self.current_regime,
                'samples_seen': self.samples_seen
            },
            'performance': {
                'best_parameters': self.best_parameters
            },
            'metadata': {
                'feature_names': self.feature_names,
                'timestamp': pd.Timestamp.now().isoformat()
            }
        }

        return model_data

    def import_model(self, model_data: Dict) -> bool:
        """
        Import model data for continued learning.

        Args:
            model_data: Dictionary with model state

        Returns:
            Success flag
        """
        try:
            # Import parameters
            params = model_data.get('parameters', {})
            self.min_cluster_size = params.get('min_cluster_size', self.min_cluster_size)
            self.min_samples = params.get('min_samples', self.min_samples)
            self.cluster_selection_epsilon = params.get('cluster_selection_epsilon', self.cluster_selection_epsilon)
            self.metric = params.get('metric', self.metric)
            self.window_size = params.get('window_size', self.window_size)
            self.feature_weights = params.get('feature_weights', self.feature_weights)

            # Import state
            state = model_data.get('state', {})
            self.regime_map = state.get('regime_map', {})
            self.next_regime_id = state.get('next_regime_id', 0)

            # Convert regime centers back to numpy arrays
            centers = state.get('regime_centers', {})
            self.regime_centers = {}
            for k, v in centers.items():
                try:
                    self.regime_centers[int(k)] = np.array(v)
                except (ValueError, TypeError):
                    pass

            self.current_regime = state.get('current_regime', -1)
            self.samples_seen = state.get('samples_seen', 0)

            # Import best parameters
            perf = model_data.get('performance', {})
            self.best_parameters = perf.get('best_parameters', {})

            # Import metadata
            metadata = model_data.get('metadata', {})
            if 'feature_names' in metadata:
                self.feature_names = metadata['feature_names']

            return True
        except Exception as e:
            self.logger.error(f"Failed to import model: {str(e)}")
            return False
            Core
            HDBSCAN
            model
        self.clusterer = None
        self.scaler = StandardScaler()

        # Data management
        self.data_window = None
        self.feature_names = []
        self.samples_seen = 0
        self.last_refit = 0

        # State tracking
        self.current_regime = -1
        self.labels_ = None
        self.regime_map = {}  # Maps raw cluster labels to consistent regime IDs
        self.next_regime_id = 0

        # Performance tracking
        self.cluster_performance = {
            'silhouette': [],
            'calinski_harabasz': [],
            'noise_ratio': []
        }

        # Parameter optimization state
        self.parameter_performance = defaultdict(list)
        self.optimization_history = []
        self.best_parameters = {}

        # Regime tracking
        self.regime_centers = {}
        self.regime_counts = defaultdict(int)
        self.regime_history = []
        self.transition_history = []

        # Consistency tracking
        self.prediction_buffer = []
        self.smoothed_regime = -1

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _create_clusterer(self) -> hdbscan.HDBSCAN:
        """Create a new HDBSCAN clusterer with current parameters."""
        return hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric=self.metric,
            prediction_data=True
        )

    def partial_fit(self, X: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> int:
        """
        Update the model with new data and potentially refit.

        Args:
            X: DataFrame with new market data
            feature_cols: Features to use for clustering (default: use all numeric)

        Returns:
            Current regime label
        """
        # Ensure DataFrame input
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        # Select features
        if feature_cols is None:
            feature_cols = X.select_dtypes(include=np.number).columns.tolist()
        else:
            feature_cols = [f for f in feature_cols if f in X.columns]

        # Store feature names
        if not self.feature_names:
            self.feature_names = feature_cols.copy()

        # Check for sufficient features
        if len(feature_cols) < 2:
            self.logger.warning("Insufficient features for clustering")
            return -1

        # Extract feature values and handle NaNs
        X_features = X[feature_cols].copy()
        if X_features.isna().any().any():
            X_features = X_features.fillna(method='ffill').fillna(method='bfill')
            if X_features.isna().any().any():
                X_features = X_features.fillna(0)

        # Initialize data window if first call
        if self.data_window is None:
            self.data_window = X_features.copy()
            self.feature_names = feature_cols.copy()
            # Initial fit will happen in update_model
        else:
            # Append new data to window
            self.data_window = pd.concat([self.data_window, X_features])

            # Keep window at specified size
            if len(self.data_window) > self.window_size:
                self.data_window = self.data_window.iloc[-self.window_size:]

        # Update sample counter
        self.samples_seen += len(X_features)

        # Refit model if needed
        if (self.samples_seen - self.last_refit >= self.refit_interval or
                self.clusterer is None):
            self._update_model()
            self.last_refit = self.samples_seen

        # Predict regime for the latest data point
        latest_point = X_features.iloc[-1:].copy()
        regime = self._predict_regime(latest_point)

        # Update regime transition tracking
        self._update_regime_tracking(regime, X.index[-1] if isinstance(X.index, pd.DatetimeIndex) else None)

        return regime

    def _update_model(self) -> None:
        """Refit the clustering model with current window data."""
        # Check if we have enough data
        if len(self.data_window) < self.min_cluster_size:
            self.logger.warning(f"Insufficient data for clustering: {len(self.data_window)} < {self.min_cluster_size}")
            return

        # Prepare feature matrix
        X = self.data_window[self.feature_names].values

        # Apply feature scaling
        X_scaled = self.scaler.fit_transform(X)

        # Apply feature weighting if specified
        if self.feature_weights:
            for i, feature in enumerate(self.feature_names):
                if feature in self.feature_weights:
                    X_scaled[:, i] *= self.feature_weights[feature]

        # Fit the model
        self.clusterer = self._create_clusterer()
        self.clusterer.fit(X_scaled)

        # Store labels
        self.labels_ = self.clusterer.labels_

        # Evaluate clustering performance
        self._evaluate_clustering(X_scaled)

        # Update regime mapping for consistency
        self._update_regime_mapping()

        # Periodically optimize parameters
        if self.samples_seen > 1000 and random.random() < 0.2:  # 20% chance
            self._optimize_parameters()

    def _evaluate_clustering(self, X_scaled: np.ndarray) -> Dict:
        """Evaluate clustering quality with multiple metrics."""
        metrics = {}

        # Count noise points (-1 label)
        if hasattr(self.clusterer, 'labels_'):
            noise_ratio = (self.clusterer.labels_ == -1).sum() / len(self.clusterer.labels_)
            metrics['noise_ratio'] = noise_ratio
            self.cluster_performance['noise_ratio'].append(noise_ratio)

        # Skip other metrics if all points are noise
        if hasattr(self.clusterer, 'labels_') and len(np.unique(self.clusterer.labels_)) > 1:
            # Silhouette score (expensive but useful for parameter tuning)
            try:
                silhouette = silhouette_score(X_scaled, self.clusterer.labels_)
                metrics['silhouette'] = silhouette
                self.cluster_performance['silhouette'].append(silhouette)
            except Exception as e:
                self.logger.warning(f"Silhouette score calculation failed: {str(e)}")

            # Calinski-Harabasz score (variance-based metric)
            try:
                ch_score = calinski_harabasz_score(X_scaled, self.clusterer.labels_)
                metrics['calinski_harabasz'] = ch_score
                self.cluster_performance['calinski_harabasz'].append(ch_score)
            except Exception as e:
                self.logger.warning(f"Calinski-Harabasz score calculation failed: {str(e)}")

        return metrics

    def _update_regime_mapping(self) -> None:
        """
        Map raw HDBSCAN cluster labels to consistent regime IDs.
        This ensures regimes maintain their identity across refits.
        """
        if not hasattr(self.clusterer, 'labels_') or self.clusterer is None:
            return

        # Calculate cluster centers
        new_centers = {}
        for label in np.unique(self.clusterer.labels_):
            if label == -1:  # Skip noise
                continue

            # Get points in this cluster
            mask = self.clusterer.labels_ == label
            points = self.scaler.inverse_transform(
                self.data_window[self.feature_names].values[mask]
            )

            # Calculate center
            new_centers[label] = points.mean(axis=0)

        # If first run, just assign new IDs
        if not self.regime_centers:
            for label in new_centers:
                self.regime_map[label] = self.next_regime_id
                self.regime_centers[self.next_regime_id] = new_centers[label]
                self.next_regime_id += 1
            return

        # Otherwise, map new clusters to most similar existing regimes
        for label, center in new_centers.items():
            # Find closest existing regime center
            min_dist = float('inf')
            best_match = None

            for regime_id, existing_center in self.regime_centers.items():
                # Skip if lengths don't match
                if len(center) != len(existing_center):
                    continue

                # Calculate distance
                dist = np.linalg.norm(center - existing_center)
                if dist < min_dist:
                    min_dist = dist
                    best_match = regime_id

            # If close enough to existing regime, map to it
            if best_match is not None and min_dist < 1.0:  # Threshold for similarity
                self.regime_map[label] = best_match
                # Update center with weighted average
                alpha = 0.2  # Weight for new center
                self.regime_centers[best_match] = (
                        (1 - alpha) * self.regime_centers[best_match] + alpha * center
                )
            else:
                # Otherwise create new regime
                self.regime_map[label] = self.next_regime_id
                self.regime_centers[self.next_regime_id] = center
                self.next_regime_id += 1

    def _predict_regime(self, X: pd.DataFrame) -> int:
        """
        Predict regime for new data points.

        Args:
            X: DataFrame with new data points

        Returns:
            Regime label (or -1 for noise)
        """
        if self.clusterer is None:
            return -1

        if len(X) == 0:
            return -1

        # Prepare features
        X_features = X[self.feature_names].copy()

        if X_features.isna().any().any():
            X_features = X_features.fillna(method='ffill').fillna(0)

        # Scale features
        X_scaled = self.scaler.transform(X_features.values)

        # Apply feature weighting
        if self.feature_weights:
            for i, feature in enumerate(self.feature_names):
                if feature in self.feature_weights:
                    X_scaled[:, i] *= self.feature_weights[feature]

        #