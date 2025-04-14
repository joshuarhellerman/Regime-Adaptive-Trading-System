"""
Feature Selector for ML-Powered Trading System.

This module provides feature selection algorithms optimized for financial data
to identify the most predictive features for machine learning models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Set, Tuple, Any
import logging
from collections import defaultdict, Counter
import time
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.metrics import r2_score, accuracy_score

# Import XGBoost if available, otherwise use a placeholder warning
try:
    from xgboost import XGBRegressor, XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Import system utilities
from utils.logger import get_logger


class FeatureSelector:
    """
    Feature selection for identifying optimal features for regime detection and forecasting.

    This class implements various feature selection methods tailored for financial time series:
    1. Model-based importance using XGBoost
    2. Mutual information analysis
    3. Correlation-based filtering
    4. Stability-based selection across time periods
    5. Online learning with adaptive feature sets
    """

    def __init__(self,
                 n_features: int = 20,
                 importance_threshold: float = 0.01,
                 stability_threshold: float = 0.7,
                 correlation_threshold: float = 0.85,
                 lookback_window: int = 10,
                 use_classification: bool = True,
                 use_mutual_info: bool = True,
                 logging_level: int = logging.INFO):
        """
        Initialize the feature selector.

        Args:
            n_features: Maximum number of features to select
            importance_threshold: Minimum importance score for feature selection
            stability_threshold: Minimum selection frequency for stable features
            correlation_threshold: Maximum correlation between features
            lookback_window: Window size for returns/volatility calculations
            use_classification: Whether to use classification objectives
            use_mutual_info: Whether to incorporate mutual information
            logging_level: Logging level
        """
        # Set up logging
        self.logger = get_logger("feature_selector")
        self.logger.setLevel(logging_level)

        # Configuration
        self.n_features = n_features
        self.importance_threshold = importance_threshold
        self.stability_threshold = stability_threshold
        self.correlation_threshold = correlation_threshold
        self.lookback_window = lookback_window
        self.use_classification = use_classification
        self.use_mutual_info = use_mutual_info

        # Check if XGBoost is available
        if not XGBOOST_AVAILABLE:
            self.logger.warning("XGBoost not available. Using simplified selection methods.")
            self.use_classification = False

        # Selection history for stability tracking
        self.feature_selection_history = defaultdict(list)
        self.stable_features = []
        self.feature_importances = {}

        # Initialize models if XGBoost is available
        if XGBOOST_AVAILABLE:
            # Configure regression model
            self.reg_model = XGBRegressor(
                verbosity=0,
                tree_method='hist',
                objective='reg:squarederror',
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                early_stopping_rounds=10
            )

            # Configure classification model
            if self.use_classification:
                self.cls_model = XGBClassifier(
                    verbosity=0,
                    tree_method='hist',
                    objective='binary:logistic',
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    early_stopping_rounds=10
                )

        self.logger.info(f"FeatureSelector initialized with {n_features} max features")

    def _prepare_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare target variables for feature selection.

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with added target columns
        """
        # Make a copy to avoid modifying the original
        df = df.copy()

        # Calculate returns if not present
        if 'returns' not in df.columns:
            if 'close' in df.columns:
                df['returns'] = df['close'].pct_change().fillna(0)
            else:
                self.logger.error("No 'close' column found for returns calculation")
                return df

        # Target for regression (next period return)
        df['target_returns'] = df['returns'].shift(-1)

        # Target for volatility prediction
        df['volatility'] = df['returns'].rolling(
            window=self.lookback_window, min_periods=1
        ).std().fillna(0)
        df['target_volatility'] = df['volatility'].shift(-1)

        # Target for direction prediction (classification)
        if self.use_classification:
            df['target_direction'] = np.sign(df['returns'].shift(-1))
            df['target_direction'] = (df['target_direction'] > 0).astype(int)

        return df.dropna()

    def _filter_correlated_features(self, X: pd.DataFrame) -> List[str]:
        """
        Remove highly correlated features to reduce redundancy.

        Args:
            X: DataFrame with feature columns

        Returns:
            List of selected feature names after correlation filtering
        """
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()

        # Extract upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features to drop (greedy approach)
        to_drop = set()

        # Sort correlations from highest to lowest
        sorted_corrs = upper.unstack().sort_values(ascending=False).dropna()

        for pair, corr_value in sorted_corrs.items():
            if corr_value < self.correlation_threshold:
                # Stop once we're below threshold
                break

            feat1, feat2 = pair

            # If both features are still candidates
            if feat1 not in to_drop and feat2 not in to_drop:
                # Drop the one with lower mean importance (if we have history)
                if feat1 in self.feature_importances and feat2 in self.feature_importances:
                    if self.feature_importances[feat1] < self.feature_importances[feat2]:
                        to_drop.add(feat1)
                    else:
                        to_drop.add(feat2)
                else:
                    # If no history, drop the second one arbitrarily
                    to_drop.add(feat2)

        # Return filtered feature list
        return [col for col in X.columns if col not in to_drop]

    def select_features(self,
                        df: pd.DataFrame,
                        candidate_features: List[str],
                        time_id: Optional[str] = None) -> List[str]:
        """
        Select optimal features for predictive modeling.

        Args:
            df: DataFrame with feature data
            candidate_features: List of potential features to evaluate
            time_id: Optional identifier for this selection period

        Returns:
            List of selected feature names
        """
        start_time = time.time()

        # Validate input
        if len(df) < 100:
            self.logger.warning("Too few samples for reliable selection")
            selected = candidate_features[:min(len(candidate_features), self.n_features)]
            self.logger.info(f"Selected {len(selected)} features (limited input)")
            return selected

        # Prepare targets
        df_targets = self._prepare_targets(df)

        # Ensure all candidate features exist
        valid_features = [f for f in candidate_features if f in df_targets.columns]

        if not valid_features:
            self.logger.warning("No valid features provided")
            return []

        # Prepare feature data
        X = df_targets[valid_features].copy()

        # Handle target variables
        y_ret = df_targets['target_returns']
        y_vol = df_targets['target_volatility']

        if self.use_classification:
            y_dir = df_targets['target_direction']

        # Filter correlated features
        filtered_features = self._filter_correlated_features(X)

        if len(filtered_features) < len(valid_features):
            self.logger.info(f"Removed {len(valid_features) - len(filtered_features)} correlated features")
            X = X[filtered_features]

        # Create train/validation split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_ret_train, y_ret_val = y_ret.iloc[:split_idx], y_ret.iloc[split_idx:]
        y_vol_train, y_vol_val = y_vol.iloc[:split_idx], y_vol.iloc[split_idx:]

        if self.use_classification:
            y_dir_train, y_dir_val = y_dir.iloc[:split_idx], y_dir.iloc[split_idx:]

        # Calculate feature importances
        importance_dict = {}

        # Model-based selection if XGBoost is available
        if XGBOOST_AVAILABLE:
            # Returns regression
            self.reg_model.fit(
                X_train, y_ret_train,
                eval_set=[(X_val, y_ret_val)],
                verbose=False
            )
            ret_importances = dict(zip(X.columns, self.reg_model.feature_importances_))

            # Volatility regression
            self.reg_model.fit(
                X_train, y_vol_train,
                eval_set=[(X_val, y_vol_val)],
                verbose=False
            )
            vol_importances = dict(zip(X.columns, self.reg_model.feature_importances_))

            # Direction classification if enabled
            if self.use_classification:
                self.cls_model.fit(
                    X_train, y_dir_train,
                    eval_set=[(X_val, y_dir_val)],
                    verbose=False
                )
                dir_importances = dict(zip(X.columns, self.cls_model.feature_importances_))

            # Combine importances with weighted average
            for feature in X.columns:
                if self.use_classification:
                    # Equal weight between regression and classification targets
                    importance_dict[feature] = (
                            ret_importances.get(feature, 0) * 0.4 +
                            vol_importances.get(feature, 0) * 0.3 +
                            dir_importances.get(feature, 0) * 0.3
                    )
                else:
                    # Just the regression targets
                    importance_dict[feature] = (
                            ret_importances.get(feature, 0) * 0.6 +
                            vol_importances.get(feature, 0) * 0.4
                    )
        else:
            # Fallback to simpler methods if XGBoost not available
            # Use correlation with targets
            for feature in X.columns:
                # Calculate correlation with returns and volatility
                ret_corr = abs(X[feature].corr(y_ret))
                vol_corr = abs(X[feature].corr(y_vol))

                # Combine with weighted average
                importance_dict[feature] = ret_corr * 0.6 + vol_corr * 0.4

        # Add mutual information if enabled
        if self.use_mutual_info:
            # Calculate mutual information for returns
            mi_ret = mutual_info_regression(X_train, y_ret_train)
            mi_ret_dict = dict(zip(X.columns, mi_ret / mi_ret.sum() if mi_ret.sum() > 0 else mi_ret))

            # Calculate mutual information for volatility
            mi_vol = mutual_info_regression(X_train, y_vol_train)
            mi_vol_dict = dict(zip(X.columns, mi_vol / mi_vol.sum() if mi_vol.sum() > 0 else mi_vol))

            # Update importance dictionary with MI scores
            for feature in X.columns:
                # Add MI with small weight
                importance_dict[feature] = importance_dict[feature] * 0.8 + (
                        mi_ret_dict.get(feature, 0) * 0.1 +
                        mi_vol_dict.get(feature, 0) * 0.1
                )

        # Filter by threshold
        filtered_importances = {
            k: v for k, v in importance_dict.items()
            if v > self.importance_threshold
        }

        # Select top features
        selected_features = sorted(
            filtered_importances.keys(),
            key=lambda x: filtered_importances[x],
            reverse=True
        )[:self.n_features]

        # Update history if time_id provided
        if time_id:
            for feature in X.columns:
                self.feature_selection_history[feature].append({
                    'selected': feature in selected_features,
                    'importance': importance_dict.get(feature, 0),
                    'time_id': time_id
                })

            # Update overall feature importances (exponential moving average)
            for feature, importance in importance_dict.items():
                if feature in self.feature_importances:
                    # 80% old value, 20% new value
                    self.feature_importances[feature] = (
                            self.feature_importances[feature] * 0.8 + importance * 0.2
                    )
                else:
                    self.feature_importances[feature] = importance

            # Update stable features
            self._update_stable_features()

        elapsed = time.time() - start_time
        self.logger.info(f"Selected {len(selected_features)} features in {elapsed:.2f}s")

        return selected_features

    def _update_stable_features(self) -> None:
        """Update the list of stable features based on selection history."""
        # Only update if we have enough history
        min_history_entries = 5

        # Count selections for each feature
        feature_counts = {}
        feature_windows = {}

        for feature, history in self.feature_selection_history.items():
            if not history:
                continue

            # Count selections
            selections = sum(1 for entry in history if entry['selected'])
            feature_counts[feature] = selections
            feature_windows[feature] = len(history)

        # Calculate stability scores
        stability_scores = {}
        for feature, count in feature_counts.items():
            windows = feature_windows[feature]
            if windows >= min_history_entries:
                stability_scores[feature] = count / windows

        # Update stable features
        self.stable_features = [
            feature for feature, score in stability_scores.items()
            if score >= self.stability_threshold
        ]

        self.logger.info(f"Updated stable features: {len(self.stable_features)} features")

    def get_stable_features(self) -> List[str]:
        """
        Get list of features that have been consistently selected.

        Returns:
            List of stable feature names
        """
        return self.stable_features

    def feature_importance_report(self) -> pd.DataFrame:
        """
        Generate a report of feature importances.

        Returns:
            DataFrame with feature importance metrics
        """
        # Only report if we have history
        if not self.feature_selection_history:
            return pd.DataFrame()

        # Calculate metrics for each feature
        report_data = []

        for feature, history in self.feature_selection_history.items():
            if not history:
                continue

            # Calculate selection rate
            selections = sum(1 for entry in history if entry['selected'])
            windows = len(history)
            selection_rate = selections / windows if windows > 0 else 0

            # Calculate importance stats
            importances = [entry['importance'] for entry in history]

            report_data.append({
                'feature': feature,
                'importance': self.feature_importances.get(feature, 0),
                'selection_rate': selection_rate,
                'times_selected': selections,
                'windows': windows,
                'importance_std': np.std(importances) if importances else 0,
                'is_stable': feature in self.stable_features
            })

        # Create report DataFrame
        report_df = pd.DataFrame(report_data)

        # Sort by importance
        if not report_df.empty:
            report_df = report_df.sort_values('importance', ascending=False)

        return report_df

    def update_with_online_data(self,
                                df: pd.DataFrame,
                                candidate_features: List[str],
                                time_id: str) -> List[str]:
        """
        Update feature selection with new data in an online fashion.

        Args:
            df: New data DataFrame
            candidate_features: List of candidate features
            time_id: Identifier for this time period

        Returns:
            List of selected features for this period
        """
        # Select features for this time period
        selected = self.select_features(df, candidate_features, time_id)

        # Use stable features if available (with high weight), combined with current selection
        if self.stable_features:
            # Create a counter with stable features having higher weight
            feature_counter = Counter()

            # Add stable features with weight 2
            for feature in self.stable_features:
                if feature in candidate_features:
                    feature_counter[feature] += 2

            # Add current selection with weight 1
            for feature in selected:
                feature_counter[feature] += 1

            # Take top features
            combined_features = [feat for feat, _ in feature_counter.most_common(self.n_features)]

            # Log the stability influence
            overlap = len(set(selected) & set(combined_features))
            self.logger.info(f"Online update: {overlap}/{len(selected)} features from current selection retained")

            return combined_features
        else:
            # If no stable features yet, just use current selection
            return selected

    def evaluate_feature_set(self,
                             df: pd.DataFrame,
                             feature_set: List[str]) -> Dict[str, float]:
        """
        Evaluate a set of features for predictive power.

        Args:
            df: DataFrame with feature data
            feature_set: List of features to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        # Check if XGBoost is available for evaluation
        if not XGBOOST_AVAILABLE:
            self.logger.warning("XGBoost not available for feature evaluation")
            return {
                'error': 'XGBoost not available for evaluation',
                'feature_count': len(feature_set)
            }

        # Prepare targets
        df_targets = self._prepare_targets(df)

        # Ensure all features exist
        valid_features = [f for f in feature_set if f in df_targets.columns]
        if not valid_features:
            return {'error': 'No valid features'}

        # Prepare data
        X = df_targets[valid_features].copy()
        y_ret = df_targets['target_returns']
        y_vol = df_targets['target_volatility']

        if self.use_classification:
            y_dir = df_targets['target_direction']

        # Create train/validation split
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_ret_train, y_ret_val = y_ret.iloc[:split_idx], y_ret.iloc[split_idx:]
        y_vol_train, y_vol_val = y_vol.iloc[:split_idx], y_vol.iloc[split_idx:]

        if self.use_classification:
            y_dir_train, y_dir_val = y_dir.iloc[:split_idx], y_dir.iloc[split_idx:]

        # Train and evaluate returns model
        self.reg_model.fit(X_train, y_ret_train, verbose=False)
        ret_preds = self.reg_model.predict(X_val)
        ret_r2 = r2_score(y_ret_val, ret_preds)

        # Train and evaluate volatility model
        self.reg_model.fit(X_train, y_vol_train, verbose=False)
        vol_preds = self.reg_model.predict(X_val)
        vol_r2 = r2_score(y_vol_val, vol_preds)

        # Train and evaluate direction model if enabled
        if self.use_classification:
            self.cls_model.fit(X_train, y_dir_train, verbose=False)
            dir_preds = self.cls_model.predict(X_val)
            dir_acc = accuracy_score(y_dir_val, dir_preds)

            return {
                'returns_r2': ret_r2,
                'volatility_r2': vol_r2,
                'direction_accuracy': dir_acc,
                'feature_count': len(valid_features)
            }
        else:
            return {
                'returns_r2': ret_r2,
                'volatility_r2': vol_r2,
                'feature_count': len(valid_features)
            }

    def save_state(self) -> Dict[str, Any]:
        """
        Save the current state for persistence.

        Returns:
            Dictionary with state data
        """
        return {
            'feature_selection_history': dict(self.feature_selection_history),
            'feature_importances': self.feature_importances,
            'stable_features': self.stable_features,
            'n_features': self.n_features,
            'importance_threshold': self.importance_threshold,
            'stability_threshold': self.stability_threshold,
            'correlation_threshold': self.correlation_threshold,
            'lookback_window': self.lookback_window,
            'use_classification': self.use_classification,
            'use_mutual_info': self.use_mutual_info
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load state from saved data.

        Args:
            state: Dictionary with state data
        """
        if 'feature_selection_history' in state:
            self.feature_selection_history = defaultdict(list)
            for feature, history in state['feature_selection_history'].items():
                self.feature_selection_history[feature] = history

        if 'feature_importances' in state:
            self.feature_importances = state['feature_importances']

        if 'stable_features' in state:
            self.stable_features = state['stable_features']

        # Load configuration if provided
        if 'n_features' in state:
            self.n_features = state['n_features']

        if 'importance_threshold' in state:
            self.importance_threshold = state['importance_threshold']

        if 'stability_threshold' in state:
            self.stability_threshold = state['stability_threshold']

        if 'correlation_threshold' in state:
            self.correlation_threshold = state['correlation_threshold']

        if 'lookback_window' in state:
            self.lookback_window = state['lookback_window']

        if 'use_classification' in state:
            self.use_classification = state['use_classification']

        if 'use_mutual_info' in state:
            self.use_mutual_info = state['use_mutual_info']

        self.logger.info("Loaded feature selector state")