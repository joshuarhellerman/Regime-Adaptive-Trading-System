"""
Feature Engineering Module for ML-Powered Trading System.

This module transforms raw market data into feature sets for trading strategies
and machine learning models. It leverages the FeatureGenerator class to create
a comprehensive set of technical indicators and statistical features.

Dependencies:
- data.processors.tick_aggregator.py: For working with tick-level data
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Set, Tuple, Any
from datetime import datetime, timedelta
from functools import lru_cache

# Import system dependencies
from data.processors.tick_aggregator import TickAggregator
from data.storage.feature_store import FeatureStore
from data.processors.data_integrity import DataIntegrityChecker

# Import the feature generator
from data.processors.feature_generator import FeatureGenerator

logger = logging.getLogger(__name__)


class FeatureEngineering:
    """
    Transforms raw market data into feature sets for trading strategies and ML models.

    This class orchestrates the feature engineering pipeline by:
    1. Normalizing and pre-processing raw market data
    2. Generating technical indicators and statistical features
    3. Creating custom features for specific strategies
    4. Managing feature sets in the feature store
    5. Providing real-time feature updates for online trading
    """

    def __init__(self,
                feature_config: Optional[Dict] = None,
                use_feature_store: bool = True,
                use_tick_data: bool = False,
                use_parallel: bool = False,
                tick_aggregator: Optional[TickAggregator] = None,
                feature_store: Optional[FeatureStore] = None,
                data_integrity_checker: Optional[DataIntegrityChecker] = None,
                logging_level: int = logging.INFO):
        """
        Initialize the feature engineering module.

        Args:
            feature_config: Configuration dictionary for feature generation
            use_feature_store: Whether to store/retrieve features from feature store
            use_tick_data: Whether to use tick-level data for feature generation
            use_parallel: Whether to use parallel processing for feature generation
            tick_aggregator: Optional TickAggregator instance
            feature_store: Optional FeatureStore instance
            data_integrity_checker: Optional DataIntegrityChecker instance
            logging_level: Logging level to use
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging_level,
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Feature configuration
        self.feature_config = feature_config or {}

        # Initialize dependencies
        self.use_feature_store = use_feature_store
        self.use_tick_data = use_tick_data
        self.use_parallel = use_parallel

        # Initialize or use provided components
        self.tick_aggregator = tick_aggregator
        self.feature_store = feature_store if feature_store else FeatureStore() if use_feature_store else None
        self.data_integrity_checker = data_integrity_checker if data_integrity_checker else DataIntegrityChecker()

        # Initialize feature generator
        self.feature_generator = FeatureGenerator(
            config=self.feature_config,
            use_parallel=self.use_parallel,
            logging_level=logging_level
        )

        # Feature set registry
        self.feature_sets = {}
        self.active_feature_sets = set()

        # Feature computation statistics
        self.last_computation_time = None
        self.feature_computation_stats = {}

        self.logger.info("FeatureEngineering module initialized")

    def process_data(self,
                    data: pd.DataFrame,
                    symbol: str,
                    feature_set_name: str = 'default',
                    timeframe: str = '1h',
                    include_features: Optional[List[str]] = None,
                    exclude_features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Process market data and generate features.

        Args:
            data: Input DataFrame with OHLCV data
            symbol: Trading symbol (e.g., 'BTC-USD')
            feature_set_name: Name of the feature set to use
            timeframe: Timeframe of the data (e.g., '1m', '1h', '1d')
            include_features: Optional list of features to include
            exclude_features: Optional list of features to exclude

        Returns:
            DataFrame with generated features
        """
        if data.empty:
            self.logger.warning(f"Empty dataframe provided for {symbol}, returning as is")
            return data

        start_time = datetime.now()
        self.logger.info(f"Processing {len(data)} rows of {timeframe} data for {symbol}")

        # Check if data exists in feature store
        if self.use_feature_store and self.feature_store:
            cache_key = f"{symbol}_{timeframe}_{feature_set_name}"
            cached_data = self.feature_store.get_features(
                symbol=symbol,
                timeframe=timeframe,
                feature_set=feature_set_name
            )

            if cached_data is not None and not cached_data.empty:
                # Check if we need to update with new data
                last_cached_time = cached_data.index[-1]
                if data.index[-1] > last_cached_time:
                    # Get only the new data
                    new_data = data[data.index > last_cached_time]
                    self.logger.info(f"Updating features for {symbol} with {len(new_data)} new rows")

                    # Generate features for new data
                    new_data_with_features = self.generate_features(
                        new_data,
                        include_features=include_features,
                        exclude_features=exclude_features
                    )

                    # Update cache
                    updated_data = self.feature_generator.update_features(cached_data, new_data_with_features)

                    # Store in feature store
                    self.feature_store.store_features(
                        updated_data,
                        symbol=symbol,
                        timeframe=timeframe,
                        feature_set=feature_set_name
                    )

                    result_df = updated_data
                else:
                    self.logger.info(f"Using cached features for {symbol}")
                    result_df = cached_data
            else:
                # Generate features for all data
                self.logger.info(f"Generating new features for {symbol}")
                result_df = self.generate_features(
                    data,
                    include_features=include_features,
                    exclude_features=exclude_features
                )

                # Store in feature store
                if self.feature_store:
                    self.feature_store.store_features(
                        result_df,
                        symbol=symbol,
                        timeframe=timeframe,
                        feature_set=feature_set_name
                    )
        else:
            # Generate features without caching
            result_df = self.generate_features(
                data,
                include_features=include_features,
                exclude_features=exclude_features
            )

        # Register feature set
        self._register_feature_set(
            feature_set_name,
            list(set(result_df.columns) - set(data.columns)),
            symbol,
            timeframe
        )

        # Track timing statistics
        end_time = datetime.now()
        self.last_computation_time = end_time
        duration = (end_time - start_time).total_seconds()

        self.feature_computation_stats[f"{symbol}_{timeframe}_{feature_set_name}"] = {
            'duration': duration,
            'row_count': len(result_df),
            'feature_count': len(set(result_df.columns) - set(data.columns)),
            'timestamp': end_time
        }

        self.logger.info(f"Generated {len(set(result_df.columns) - set(data.columns))} "
                        f"features for {symbol} in {duration:.2f} seconds")

        return result_df

    def generate_features(self,
                        data: pd.DataFrame,
                        include_features: Optional[List[str]] = None,
                        exclude_features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate features for input data using the feature generator.

        Args:
            data: Input DataFrame with OHLCV data
            include_features: Optional list of features to include
            exclude_features: Optional list of features to exclude

        Returns:
            DataFrame with generated features
        """
        # Apply data integrity checks
        if self.data_integrity_checker:
            data = self.data_integrity_checker.check_and_fix(data)

        # Use tick-level data if available and configured
        if self.use_tick_data and self.tick_aggregator and 'tick_data' in data.attrs:
            tick_data = data.attrs['tick_data']
            self.logger.info(f"Using tick data with {len(tick_data)} ticks for feature engineering")

            # Generate tick-based features
            data = self.tick_aggregator.augment_with_tick_features(data, tick_data)

        # Use feature generator to create features
        result_df = self.feature_generator.generate_features(data)

        # Filter features if specified
        if include_features:
            # Get original columns plus requested features
            original_cols = set(data.columns)
            include_set = set(include_features)
            keep_cols = list(original_cols.union(include_set))

            # Filter columns
            available_cols = set(result_df.columns)
            filtered_cols = [col for col in keep_cols if col in available_cols]

            result_df = result_df[filtered_cols]

        if exclude_features:
            # Remove excluded features
            exclude_set = set(exclude_features)
            result_df = result_df[[col for col in result_df.columns if col not in exclude_set]]

        return result_df

    def update_features(self,
                       existing_data: pd.DataFrame,
                       new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Update existing features with new data.

        Args:
            existing_data: Existing DataFrame with features
            new_data: New data to incorporate

        Returns:
            Updated DataFrame with features
        """
        return self.feature_generator.update_features(existing_data, new_data)

    def process_streaming_data(self,
                             data_point: Dict[str, Any],
                             symbol: str,
                             feature_set_name: str = 'default',
                             timeframe: str = '1m',
                             context_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Process a single streaming data point for online feature updates.

        Args:
            data_point: Dictionary with OHLCV data
            symbol: Trading symbol
            feature_set_name: Name of the feature set to use
            timeframe: Timeframe of the data
            context_data: Optional historical context data

        Returns:
            Dictionary with data point and features
        """
        self.logger.debug(f"Processing streaming data for {symbol}")

        # Convert data point to DataFrame
        if 'timestamp' in data_point:
            timestamp = pd.to_datetime(data_point['timestamp'])
        else:
            timestamp = pd.Timestamp.now()

        # Create a single-row DataFrame
        df = pd.DataFrame([data_point], index=[timestamp])

        # Check if we need context data
        if context_data is None and self.use_feature_store and self.feature_store:
            # Retrieve context data from feature store
            lookback = 100  # Number of historical bars needed for feature calculation
            context_data = self.feature_store.get_features(
                symbol=symbol,
                timeframe=timeframe,
                feature_set=feature_set_name,
                limit=lookback
            )

        # If we have context, use it for better feature calculation
        if context_data is not None and not context_data.empty:
            # Combine context with new data
            combined_df = pd.concat([context_data, df])

            # Remove duplicates if any
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]

            # Generate features
            features_df = self.generate_features(combined_df)

            # Extract just the latest row with features
            latest_features = features_df.iloc[-1].to_dict()

            # Optionally update feature store with new data point
            if self.use_feature_store and self.feature_store:
                self.feature_store.update_features(
                    features_df.iloc[[-1]],
                    symbol=symbol,
                    timeframe=timeframe,
                    feature_set=feature_set_name
                )

            return latest_features
        else:
            # Generate features without context (less accurate)
            self.logger.warning(f"Generating features without context data for {symbol}")
            features_df = self.generate_features(df)
            return features_df.iloc[0].to_dict()

    def _register_feature_set(self,
                            name: str,
                            features: List[str],
                            symbol: str,
                            timeframe: str) -> None:
        """
        Register a feature set in the registry.

        Args:
            name: Name of the feature set
            features: List of features in the set
            symbol: Trading symbol
            timeframe: Timeframe of the data
        """
        key = f"{symbol}_{timeframe}_{name}"
        self.feature_sets[key] = {
            'name': name,
            'symbol': symbol,
            'timeframe': timeframe,
            'features': features,
            'created_at': datetime.now()
        }
        self.active_feature_sets.add(key)

        self.logger.debug(f"Registered feature set {name} for {symbol} ({timeframe}) with {len(features)} features")

    def get_feature_sets(self) -> Dict[str, Dict]:
        """
        Get information about all registered feature sets.

        Returns:
            Dictionary of feature sets
        """
        return self.feature_sets

    def get_active_feature_sets(self) -> List[str]:
        """
        Get list of active feature set names.

        Returns:
            List of active feature set names
        """
        return list(self.active_feature_sets)

    def get_available_features(self,
                              symbol: Optional[str] = None,
                              timeframe: Optional[str] = None,
                              feature_set: Optional[str] = None) -> List[str]:
        """
        Get list of available features, optionally filtered.

        Args:
            symbol: Optional filter by symbol
            timeframe: Optional filter by timeframe
            feature_set: Optional filter by feature set name

        Returns:
            List of available feature names
        """
        all_features = set()

        for key, fs in self.feature_sets.items():
            # Apply filters
            if symbol and fs['symbol'] != symbol:
                continue

            if timeframe and fs['timeframe'] != timeframe:
                continue

            if feature_set and fs['name'] != feature_set:
                continue

            all_features.update(fs['features'])

        return sorted(list(all_features))

    def get_feature_importance(self,
                             feature_set: str = 'default',
                             symbol: Optional[str] = None,
                             timeframe: Optional[str] = None,
                             top_n: int = 20) -> Dict[str, float]:
        """
        Get approximate feature importance based on statistical properties.

        Args:
            feature_set: Name of the feature set
            symbol: Optional symbol filter
            timeframe: Optional timeframe filter
            top_n: Number of top features to return

        Returns:
            Dictionary mapping feature names to importance scores
        """
        # This is a simplified implementation without machine learning
        # In a real system, you would use model-based feature importance

        # Get feature data
        feature_data = None

        if self.use_feature_store and self.feature_store:
            feature_data = self.feature_store.get_features(
                symbol=symbol,
                timeframe=timeframe,
                feature_set=feature_set
            )

        if feature_data is None or feature_data.empty:
            self.logger.warning("No feature data available for importance calculation")
            return {}

        # Calculate simplified feature importance
        target = 'price_returns'  # Use returns as proxy target
        if target not in feature_data.columns:
            target = 'close'  # Fallback

        # Get numeric features only
        numeric_features = feature_data.select_dtypes(include=[np.number]).columns
        feature_list = [f for f in numeric_features if f not in ['open', 'high', 'low', 'close', 'volume', target]]

        if not feature_list:
            return {}

        # Calculate correlation with target
        corr_with_target = {}
        for feat in feature_list:
            if feat == target:
                continue

            # Skip if all values are the same or if column has NaNs
            if feature_data[feat].nunique() <= 1 or feature_data[feat].isna().any():
                continue

            # Use absolute correlation as importance
            corr = abs(feature_data[[feat, target]].corr().iloc[0, 1])

            if np.isnan(corr):
                continue

            corr_with_target[feat] = corr

        # Sort by correlation and take top N
        sorted_features = sorted(corr_with_target.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:top_n])

    def create_feature_set(self,
                          name: str,
                          features: List[str],
                          description: Optional[str] = None) -> None:
        """
        Create a named feature set for reuse.

        Args:
            name: Name of the feature set
            features: List of features to include
            description: Optional description
        """
        self.feature_config[name] = {
            'features': features,
            'description': description or f"Feature set {name}",
            'created_at': datetime.now().isoformat()
        }

        self.logger.info(f"Created feature set {name} with {len(features)} features")

    def delete_feature_set(self, name: str) -> bool:
        """
        Delete a named feature set.

        Args:
            name: Name of the feature set to delete

        Returns:
            Boolean indicating success
        """
        if name in self.feature_config:
            del self.feature_config[name]

            # Remove from active sets
            to_remove = []
            for key in self.active_feature_sets:
                if key.endswith(f"_{name}"):
                    to_remove.append(key)

            for key in to_remove:
                self.active_feature_sets.remove(key)

            # Remove from feature set registry
            to_remove = []
            for key in self.feature_sets:
                if key.endswith(f"_{name}"):
                    to_remove.append(key)

            for key in to_remove:
                del self.feature_sets[key]

            self.logger.info(f"Deleted feature set {name}")
            return True

        self.logger.warning(f"Feature set {name} not found")
        return False

    @lru_cache(maxsize=32)
    def get_feature_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata about available features.

        Returns:
            Dictionary with feature metadata
        """
        # Get all available features from the feature generator
        all_features = self.feature_generator.get_feature_list()
        feature_groups = self.feature_generator.get_feature_groups()

        metadata = {}

        # Create metadata for each feature
        for feature in all_features:
            # Determine feature group
            group = next((g for g, features in feature_groups.items() if feature in features), "other")

            # Create metadata entry
            metadata[feature] = {
                'group': group,
                'description': self._get_feature_description(feature),
                'requirements': self._get_feature_requirements(feature)
            }

        return metadata

    def _get_feature_description(self, feature: str) -> str:
        """
        Get description for a given feature.

        Args:
            feature: Feature name

        Returns:
            Description string
        """
        # This is a simplified implementation
        # In a real system, you would maintain descriptions in a database or config

        prefix_map = {
            'price_': "Price-based feature showing ",
            'volume_': "Volume-based feature indicating ",
            'volatility_': "Volatility measure that represents ",
            'trend_': "Trend indicator that shows ",
            'momentum_': "Momentum indicator measuring ",
            'mean_rev_': "Mean reversion signal that identifies ",
            'cycle_': "Cycle detection feature that captures ",
            'pattern_': "Chart pattern feature that detects ",
            'time_': "Time-based feature related to ",
            'microstructure_': "Market microstructure feature showing ",
            'stats_': "Statistical feature calculating ",
            'adaptive_': "Adaptive feature that adjusts to market conditions for ",
            'custom_': "Custom feature designed for specific strategy needs: ",
            'interaction_': "Interaction feature combining multiple indicators to show "
        }

        for prefix, desc in prefix_map.items():
            if feature.startswith(prefix):
                return desc + " ".join(feature[len(prefix):].split('_'))

        return "Feature " + " ".join(feature.split('_'))

    def _get_feature_requirements(self, feature: str) -> List[str]:
        """
        Get data requirements for a given feature.

        Args:
            feature: Feature name

        Returns:
            List of required data fields
        """
        # Simplified implementation
        requirements = ['close']

        # Add high, low if relevant prefixes
        if any(feature.startswith(p) for p in ['volatility_', 'pattern_', 'microstructure_']):
            requirements.extend(['high', 'low'])

        # Add volume if relevant
        if feature.startswith('volume_') or 'volume' in feature:
            requirements.append('volume')

        # Add open if relevant
        if feature.startswith('pattern_') or 'open' in feature:
            requirements.append('open')

        return sorted(list(set(requirements)))