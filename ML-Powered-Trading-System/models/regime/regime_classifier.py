import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import joblib
import json
import os
from pathlib import Path
from datetime import datetime, timedelta

# Import the components we've developed
from .online_regime_clusterer import OnlineRegimeClusterer
from .regime_transition_detector import RegimeTransitionDetector
from .regime_properties import RegimePropertiesCalculator
from .regime_adaptor import RegimeAdaptor


class RegimeClassifier:
    """
    Main entry point for the market regime classification system.
    Orchestrates the regime detection, transition detection, property calculation,
    and parameter adaptation components.
    """

    def __init__(
            self,
            model_id: str = "default",
            feature_cols: Optional[List[str]] = None,
            window_size: int = 500,
            min_cluster_size: int = 50,
            min_samples: int = 15,
            cluster_selection_epsilon: float = 0.3,
            metric: str = 'euclidean',
            adaptation_rate: float = 0.05,
            prediction_smoothing: float = 0.7,
            models_dir: str = "./models/regime/saved_models",
            **kwargs
    ):
        """
        Initialize the regime classifier.

        Args:
            model_id: Unique identifier for this model instance (default: "default")
            feature_cols: Features to use for regime detection (default: None, use all)
            window_size: Size of data window for clustering (default: 500)
            min_cluster_size: Minimum points to form a cluster (default: 50)
            min_samples: Minimum samples for core point (default: 15)
            cluster_selection_epsilon: Cluster boundary looseness (default: 0.3)
            metric: Distance metric for clustering (default: 'euclidean')
            adaptation_rate: Learning rate for parameter optimization (default: 0.05)
            prediction_smoothing: Required stability for regime classification (default: 0.7)
            models_dir: Directory for saving/loading models (default: "./models/regime/saved_models")
        """
        self.model_id = model_id
        self.feature_cols = feature_cols
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.clusterer = OnlineRegimeClusterer(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric=metric,
            window_size=window_size,
            adaptation_rate=adaptation_rate,
            min_cluster_persistence=prediction_smoothing
        )

        self.transition_detector = RegimeTransitionDetector(
            window_size=20,
            threshold_multiplier=2.0,
            adaptation_rate=adaptation_rate
        )

        self.properties_calculator = RegimePropertiesCalculator(
            memory_decay=0.95,
            min_samples=50
        )

        self.regime_adaptor = RegimeAdaptor(
            learning_rate=adaptation_rate,
            exploration_rate=0.2,
            performance_metric='sharpe_ratio'
        )

        # State tracking
        self.current_regime = -1
        self.last_regime = -1
        self.regime_start_time = None
        self.samples_processed = 0
        self.last_transition_time = None
        self.transition_in_progress = False

        # Last known state
        self.latest_properties = {}
        self.latest_regime_params = {}
        self.latest_performance = {}
        self.latest_feature_importance = {}

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Register default strategy parameters
        self._register_default_parameters()

        # Try to load existing model if available
        self._try_load_model()

    def _register_default_parameters(self) -> None:
        """Register default strategy parameters for adaptation."""
        # Position sizing parameters
        self.regime_adaptor.register_parameter(
            'position_size_factor', 1.0, min_value=0.1, max_value=2.0,
            description='Base position size multiplier'
        )

        self.regime_adaptor.register_parameter(
            'max_position_size', 0.1, min_value=0.01, max_value=0.5,
            description='Maximum position size as fraction of capital'
        )

        # Entry/exit parameters
        self.regime_adaptor.register_parameter(
            'profit_target_multiplier', 2.0, min_value=0.5, max_value=5.0,
            description='Profit target as multiple of expected volatility'
        )

        self.regime_adaptor.register_parameter(
            'stop_loss_multiplier', 1.0, min_value=0.5, max_value=3.0,
            description='Stop loss as multiple of expected volatility'
        )

        # Time-based parameters
        self.regime_adaptor.register_parameter(
            'holding_period', 24, min_value=4, max_value=72, value_type='int',
            description='Target holding period in hours'
        )

        self.regime_adaptor.register_parameter(
            'entry_timeout', 4, min_value=1, max_value=12, value_type='int',
            description='Maximum time to wait for entry conditions'
        )

        # Signal threshold parameters
        self.regime_adaptor.register_parameter(
            'entry_threshold', 0.7, min_value=0.3, max_value=0.9,
            description='Minimum signal strength for entry'
        )

        self.regime_adaptor.register_parameter(
            'exit_threshold', 0.5, min_value=0.2, max_value=0.8,
            description='Minimum signal strength for exit'
        )

        # Policy function for volatility-based position sizing
        def volatility_position_policy(regime_props, performance):
            """Adjust position size based on volatility level."""
            vol_level = regime_props.get('volatility_level', 0.01)
            base_position = 0.1  # 10% base position

            # Reduce position in high volatility
            if vol_level > 0.02:  # High volatility
                return base_position * 0.5
            elif vol_level < 0.005:  # Low volatility
                return base_position * 1.5
            else:
                return base_position

        self.regime_adaptor.register_policy_function(
            'position_size_factor',
            volatility_position_policy,
            'Adjusts position size based on volatility'
        )

    def process(self, df: pd.DataFrame, performance_metrics: Optional[Dict] = None) -> Dict:
        """
        Process new market data and update regime classification.

        Args:
            df: DataFrame with market data
            performance_metrics: Optional dict with recent strategy performance metrics

        Returns:
            Dictionary with classification results and adapted parameters
        """
        # Validate input
        if not isinstance(df, pd.DataFrame) or len(df) == 0:
            return {'error': 'Invalid input data'}

        # Get timestamp
        timestamp = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else None

        # 1. Update regime clustering
        features_to_use = self.feature_cols if self.feature_cols else None
        regime = self.clusterer.partial_fit(df, feature_cols=features_to_use)

        # Track regime changes
        self.last_regime = self.current_regime
        self.current_regime = regime

        # 2. Check for regime transition
        transition_detected = False
        transition_info = {}

        if self.current_regime >= 0:  # Skip transition detection if in noise regime
            transition_detected, transition_info = self.transition_detector.update(
                df, self.current_regime
            )

            # Update transition state
            if transition_detected:
                self.transition_in_progress = True
                self.last_transition_time = timestamp

                # Record transition in properties calculator
                if self.last_regime >= 0 and self.last_regime != self.current_regime:
                    self.properties_calculator.record_transition(
                        self.last_regime, self.current_regime, timestamp
                    )
            else:
                # Reset transition flag after some time (stability period)
                if self.transition_in_progress and self.last_transition_time:
                    if timestamp and isinstance(timestamp, pd.Timestamp):
                        stability_period = timedelta(hours=4)
                        if timestamp - self.last_transition_time > stability_period:
                            self.transition_in_progress = False

        # 3. Calculate regime properties
        if self.current_regime >= 0:  # Skip for noise regime
            regime_properties = self.properties_calculator.update(
                df, self.current_regime, timestamp
            )
            self.latest_properties = regime_properties

        # 4. Adapt strategy parameters
        adapted_params = {}
        if self.current_regime >= 0:  # Skip for noise regime
            adapted_params = self.regime_adaptor.adapt_parameters(
                self.current_regime, self.latest_properties, performance_metrics
            )
            self.latest_regime_params = adapted_params

        # Store performance metrics for future reference
        if performance_metrics:
            self.latest_performance = performance_metrics

            # Update parameter learning with feedback
            if self.current_regime >= 0:
                self.regime_adaptor.update_with_feedback(
                    self.current_regime, performance_metrics, timestamp
                )

        # Calculate feature importance for clustering improvement
        if self.samples_processed % 100 == 0:  # Periodically update
            self._update_feature_importance()

        # Increment counter
        self.samples_processed += len(df)

        # Auto-save model periodically
        if self.samples_processed % 1000 == 0:
            self.save_model()

        # Prepare response
        response = {
            'regime': self.current_regime,
            'timestamp': timestamp,
            'in_transition': self.transition_in_progress,
            'transition_detected': transition_detected,
            'adapted_parameters': adapted_params,
            'regime_properties': self.latest_properties,
            'regime_stats': self.clusterer.get_regime_stats()
        }

        if transition_detected:
            response['transition_info'] = transition_info

        return response

    def _update_feature_importance(self) -> None:
        """Calculate feature importance and update clustering weights."""
        # Skip if no feature importance data
        if not hasattr(self.clusterer, 'data_window') or self.clusterer.data_window is None:
            return

        try:
            # Simple calculation based on correlation with returns
            if 'returns' in self.clusterer.data_window.columns:
                feature_corrs = {}
                for feature in self.clusterer.feature_names:
                    if feature != 'returns':
                        corr = abs(self.clusterer.data_window[feature].corr(
                            self.clusterer.data_window['returns']
                        ))
                        if np.isfinite(corr):
                            feature_corrs[feature] = max(0.1, corr)

                # Apply min/max scaling to get weights between 0.1 and 1.0
                if feature_corrs:
                    min_corr = min(feature_corrs.values())
                    max_corr = max(feature_corrs.values())
                    scaled_weights = {}

                    for feature, corr in feature_corrs.items():
                        if max_corr > min_corr:
                            weight = 0.1 + 0.9 * (corr - min_corr) / (max_corr - min_corr)
                        else:
                            weight = 0.5  # Default middle weight

                        scaled_weights[feature] = weight

                    # Update clusterer
                    self.clusterer.update_feature_weights(scaled_weights)

                    # Update transition detector
                    self.transition_detector.update_feature_importance(scaled_weights)

                    # Store for reference
                    self.latest_feature_importance = scaled_weights

        except Exception as e:
            self.logger.warning(f"Feature importance calculation failed: {str(e)}")

    def get_optimal_parameters(self, regime_label: Optional[int] = None) -> Dict:
        """
        Get optimal parameters for a specific regime.

        Args:
            regime_label: Regime to get parameters for (default: current regime)

        Returns:
            Dictionary of optimal parameters
        """
        # Default to current regime if none specified
        if regime_label is None:
            regime_label = self.current_regime

        # Return optimal parameters from regime adaptor
        return self.regime_adaptor.get_optimal_parameters(regime_label)

    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Save the model to disk.

        Args:
            filepath: Path to save model (default: auto-generated)

        Returns:
            Path to saved model
        """
        if filepath is None:
            # Generate filename based on model_id and timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.models_dir / f"{self.model_id}_{timestamp}.pkl"

        # Create model data dictionary
        model_data = {
            'model_id': self.model_id,
            'feature_cols': self.feature_cols,
            'samples_processed': self.samples_processed,
            'current_regime': self.current_regime,
            'last_regime': self.last_regime,
            'transition_in_progress': self.transition_in_progress,
            'last_transition_time': self.last_transition_time,
            'latest_properties': self.latest_properties,
            'latest_regime_params': self.latest_regime_params,
            'latest_performance': self.latest_performance,
            'latest_feature_importance': self.latest_feature_importance,
            'timestamp': datetime.now().isoformat(),

            # Component states
            'clusterer': self.clusterer.export_model(),
            'regime_adaptor': self.regime_adaptor.export_adaptation_model()
        }

        # Save to disk
        try:
            joblib.dump(model_data, filepath)
            self.logger.info(f"Model saved to {filepath}")

            # Save a reference to latest model
            latest_path = self.models_dir / f"{self.model_id}_latest.pkl"
            joblib.dump(model_data, latest_path)

            return str(filepath)
        except Exception as e:
            self.logger.error(f"Failed to save model: {str(e)}")
            return ""

    def load_model(self, filepath: str) -> bool:
        """
        Load a model from disk.

        Args:
            filepath: Path to load model from

        Returns:
            Success flag
        """
        try:
            model_data = joblib.load(filepath)

            # Update basic attributes
            self.model_id = model_data.get('model_id', self.model_id)
            self.feature_cols = model_data.get('feature_cols', self.feature_cols)
            self.samples_processed = model_data.get('samples_processed', 0)
            self.current_regime = model_data.get('current_regime', -1)
            self.last_regime = model_data.get('last_regime', -1)
            self.transition_in_progress = model_data.get('transition_in_progress', False)
            self.last_transition_time = model_data.get('last_transition_time')
            self.latest_properties = model_data.get('latest_properties', {})
            self.latest_regime_params = model_data.get('latest_regime_params', {})
            self.latest_performance = model_data.get('latest_performance', {})
            self.latest_feature_importance = model_data.get('latest_feature_importance', {})

            # Load component states
            if 'clusterer' in model_data:
                self.clusterer.import_model(model_data['clusterer'])

            if 'regime_adaptor' in model_data:
                self.regime_adaptor.import_adaptation_model(model_data['regime_adaptor'])

            self.logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            return False

    def _try_load_model(self) -> None:
        """Try to load the latest model if available."""
        latest_path = self.models_dir / f"{self.model_id}_latest.pkl"
        if latest_path.exists():
            self.load_model(str(latest_path))

    def get_transition_matrix(self) -> Dict:
        """
        Get the regime transition probability matrix.

        Returns:
            Dictionary with transition matrix data
        """
        return {
            'matrix': self.properties_calculator.transition_matrix,
            'last_transitions': self.transition_detector.transition_history[-10:]
        }

    def reset(self) -> None:
        """Reset the classifier state without losing learned parameters."""
        self.current_regime = -1
        self.last_regime = -1
        self.regime_start_time = None
        self.transition_in_progress = False
        self.last_transition_time = None

        # Reset components
        self.transition_detector.reset()

    def get_features_importance(self) -> Dict:
        """
        Get the importance scores for features used in regime detection.

        Returns:
            Dictionary mapping features to importance scores
        """
        return self.latest_feature_importance

    def get_regime_properties(self, regime_label: Optional[int] = None) -> Dict:
        """
        Get properties for a specific regime.

        Args:
            regime_label: Regime to get properties for (default: current regime)

        Returns:
            Dictionary of regime properties
        """
        # Default to current regime if none specified
        if regime_label is None:
            regime_label = self.current_regime

        if regime_label < 0:
            return {}

        # Get properties from calculator
        return self.properties_calculator.regime_stats.get(regime_label, {})

    def get_all_regimes_properties(self) -> Dict:
        """
        Get properties for all known regimes.

        Returns:
            Dictionary mapping regime labels to their properties
        """
        return {
            regime: props.copy()
            for regime, props in self.properties_calculator.regime_stats.items()
            if regime >= 0
        }