"""
feature_analyzer.py - Feature Analysis and Selection Framework

This module provides tools for analyzing features, calculating feature importance,
feature selection, and feature engineering for ML models in the trading system.
"""

import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union, Set, Callable
from enum import Enum
from dataclasses import dataclass, field
import warnings
import os
import json
from datetime import datetime, timedelta
import time

# Try to import optional dependencies
try:
    from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif
    from sklearn.feature_selection import RFE, RFECV
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from data.feature_store import FeatureStore

logger = logging.getLogger(__name__)


class FeatureImportanceMethod(Enum):
    """Methods for calculating feature importance"""
    PERMUTATION = "permutation"          # Permutation importance
    SHAP = "shap"                        # SHAP values
    MUTUAL_INFO = "mutual_info"          # Mutual information
    MODEL_SPECIFIC = "model_specific"    # Model-specific importance (e.g., Random Forest)
    CORRELATION = "correlation"          # Correlation with target
    RFE = "rfe"                          # Recursive Feature Elimination
    VARIANCE = "variance"                # Variance threshold
    PCA = "pca"                          # Principal Component Analysis


class FeatureSelectionMethod(Enum):
    """Methods for feature selection"""
    THRESHOLD = "threshold"              # Select features above importance threshold
    TOP_K = "top_k"                      # Select top K features
    CUMULATIVE = "cumulative"            # Select features up to cumulative importance
    FORWARD = "forward"                  # Forward feature selection
    BACKWARD = "backward"                # Backward feature elimination
    RFE = "rfe"                          # Recursive Feature Elimination
    HYBRID = "hybrid"                    # Hybrid approach combining multiple methods


@dataclass
class FeatureImportance:
    """Feature importance results"""
    method: FeatureImportanceMethod
    importances: Dict[str, float]
    ranked_features: List[str]  # Features sorted by importance
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureSelectionResult:
    """Feature selection results"""
    method: FeatureSelectionMethod
    selected_features: List[str]
    importance_scores: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeatureSummary:
    """Summary statistics and properties of a feature"""

    def __init__(self,
                 name: str,
                 data: Union[pd.Series, np.ndarray],
                 feature_type: Optional[str] = None):
        """
        Initialize feature summary.

        Args:
            name: Feature name
            data: Feature data
            feature_type: Optional type override
        """
        self.name = name

        # Convert to pandas Series if numpy array
        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        self.data = data

        # Infer feature type if not provided
        if feature_type:
            self.feature_type = feature_type
        elif pd.api.types.is_numeric_dtype(data):
            self.feature_type = "numeric"
        elif pd.api.types.is_categorical_dtype(data) or data.nunique() < 10:
            self.feature_type = "categorical"
        elif pd.api.types.is_datetime64_dtype(data):
            self.feature_type = "datetime"
        else:
            self.feature_type = "other"

        # Calculate summary statistics
        self.summary = self._calculate_summary()

    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics for the feature"""
        summary = {
            "name": self.name,
            "type": self.feature_type,
            "count": len(self.data),
            "missing": self.data.isna().sum()
        }

        # Add type-specific statistics
        if self.feature_type == "numeric":
            summary.update({
                "mean": float(self.data.mean()),
                "std": float(self.data.std()),
                "min": float(self.data.min()),
                "25%": float(self.data.quantile(0.25)),
                "50%": float(self.data.median()),
                "75%": float(self.data.quantile(0.75)),
                "max": float(self.data.max()),
                "skew": float(self.data.skew()),
                "kurtosis": float(self.data.kurtosis()),
                "zero_count": int((self.data == 0).sum()),
                "positive_count": int((self.data > 0).sum()),
                "negative_count": int((self.data < 0).sum())
            })

            # Add trading-specific metrics for numeric features
            if summary["count"] > 1:
                try:
                    # Calculate returns-like metrics for financial analysis
                    summary["sharpe"] = float(self.data.mean() / (self.data.std() + 1e-8))
                    summary["abs_mean"] = float(abs(self.data).mean())

                    # Calculate serial correlation (auto-correlation)
                    summary["autocorr_1"] = float(self.data.autocorr(1)) if hasattr(self.data, 'autocorr') else None

                    # Calculate positive ratio (% of positive values)
                    summary["positive_ratio"] = float(summary["positive_count"] / summary["count"])
                except Exception as e:
                    logger.warning(f"Error calculating trading metrics for {self.name}: {str(e)}")

        elif self.feature_type == "categorical":
            # Get value counts
            value_counts = self.data.value_counts()

            summary.update({
                "unique_count": self.data.nunique(),
                "most_common": value_counts.index[0] if not value_counts.empty else None,
                "most_common_count": int(value_counts.iloc[0]) if not value_counts.empty else 0,
                "entropy": float(self._calculate_entropy(value_counts))
            })

            # Add value distribution for low-cardinality categoricals
            if summary["unique_count"] <= 10:
                summary["value_distribution"] = {
                    str(k): int(v) for k, v in value_counts.items()
                }

        elif self.feature_type == "datetime":
            summary.update({
                "min": self.data.min(),
                "max": self.data.max(),
                "range_days": (self.data.max() - self.data.min()).days if isinstance(self.data.max(), pd.Timestamp) else None
            })

        return summary

    def _calculate_entropy(self, value_counts: pd.Series) -> float:
        """Calculate entropy of a categorical feature"""
        probabilities = value_counts / value_counts.sum()
        return float(-np.sum(probabilities * np.log2(probabilities)))

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary"""
        return self.summary


class FeatureAnalyzer:
    """
    Framework for analyzing features, calculating importance,
    selecting features, and engineering new features.
    """

    def __init__(self,
                 feature_store: Optional[FeatureStore] = None,
                 importance_methods: List[FeatureImportanceMethod] = None,
                 selection_methods: List[FeatureSelectionMethod] = None,
                 output_dir: str = "data/feature_analysis",
                 random_state: Optional[int] = None):
        """
        Initialize feature analyzer.

        Args:
            feature_store: Optional feature store for retrieving and storing features
            importance_methods: List of feature importance calculation methods
            selection_methods: List of feature selection methods
            output_dir: Directory for storing analysis results
            random_state: Random seed for reproducibility
        """
        self.feature_store = feature_store
        self.importance_methods = importance_methods or [
            FeatureImportanceMethod.PERMUTATION,
            FeatureImportanceMethod.CORRELATION,
            FeatureImportanceMethod.MUTUAL_INFO
        ]
        self.selection_methods = selection_methods or [
            FeatureSelectionMethod.TOP_K,
            FeatureSelectionMethod.THRESHOLD
        ]
        self.output_dir = output_dir
        self.random_state = random_state

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)

        logger.info(f"FeatureAnalyzer initialized with {len(self.importance_methods)} importance methods "
                   f"and {len(self.selection_methods)} selection methods")

    def analyze_features(self,
                        data: Union[pd.DataFrame, Dict[str, np.ndarray]],
                        target: Optional[Union[pd.Series, np.ndarray]] = None,
                        feature_subset: Optional[List[str]] = None) -> Dict[str, FeatureSummary]:
        """
        Analyze features in a dataset.

        Args:
            data: DataFrame or dictionary of features
            target: Optional target variable for correlation analysis
            feature_subset: Optional subset of features to analyze

        Returns:
            Dictionary mapping feature names to FeatureSummary objects
        """
        # Convert dictionary to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame(data)

        # Filter to feature subset if specified
        if feature_subset:
            data = data[feature_subset]

        # Analyze each feature
        feature_summaries = {}

        for column in data.columns:
            feature_data = data[column]
            feature_summary = FeatureSummary(column, feature_data)
            feature_summaries[column] = feature_summary

        # Add target correlation if target is provided
        if target is not None:
            self._add_target_correlations(feature_summaries, data, target)

        logger.info(f"Analyzed {len(feature_summaries)} features")

        return feature_summaries

    def _add_target_correlations(self,
                                feature_summaries: Dict[str, FeatureSummary],
                                data: pd.DataFrame,
                                target: Union[pd.Series, np.ndarray]):
        """Add target correlation to feature summaries"""
        try:
            # Calculate correlations for numeric features
            numeric_cols = data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                # Convert target to Series if ndarray
                if isinstance(target, np.ndarray):
                    target = pd.Series(target)

                # Calculate Pearson correlation
                correlations = pd.DataFrame({
                    'feature': numeric_cols,
                    'correlation': [data[col].corr(target) for col in numeric_cols]
                })

                # Add to feature summaries
                for _, row in correlations.iterrows():
                    feature = row['feature']
                    if feature in feature_summaries:
                        feature_summaries[feature].summary['target_correlation'] = float(row['correlation'])

        except Exception as e:
            logger.warning(f"Error calculating target correlations: {str(e)}")

    def calculate_feature_importance(self,
                                   X: Union[pd.DataFrame, np.ndarray],
                                   y: Union[pd.Series, np.ndarray],
                                   model: Any,
                                   method: FeatureImportanceMethod = FeatureImportanceMethod.PERMUTATION,
                                   n_repeats: int = 10,
                                   feature_names: Optional[List[str]] = None) -> FeatureImportance:
        """
        Calculate feature importance.

        Args:
            X: Feature data
            y: Target data
            model: Trained model
            method: Method for calculating importance
            n_repeats: Number of repetitions for permutation importance
            feature_names: Optional feature names for numpy arrays

        Returns:
            FeatureImportance object
        """
        # Get feature names from DataFrame or use provided names
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        elif feature_names is None:
            # Create default feature names
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        if method == FeatureImportanceMethod.PERMUTATION:
            return self._calculate_permutation_importance(X, y, model, feature_names, n_repeats)
        elif method == FeatureImportanceMethod.SHAP:
            return self._calculate_shap_importance(X, y, model, feature_names)
        elif method == FeatureImportanceMethod.MUTUAL_INFO:
            return self._calculate_mutual_info_importance(X, y, feature_names)
        elif method == FeatureImportanceMethod.MODEL_SPECIFIC:
            return self._calculate_model_specific_importance(X, y, model, feature_names)
        elif method == FeatureImportanceMethod.CORRELATION:
            return self._calculate_correlation_importance(X, y, feature_names)
        elif method == FeatureImportanceMethod.RFE:
            return self._calculate_rfe_importance(X, y, model, feature_names)
        elif method == FeatureImportanceMethod.VARIANCE:
            return self._calculate_variance_importance(X, feature_names)
        elif method == FeatureImportanceMethod.PCA:
            return self._calculate_pca_importance(X, feature_names)
        else:
            raise ValueError(f"Unsupported importance method: {method}")

    def _calculate_permutation_importance(self,
                                        X: Union[pd.DataFrame, np.ndarray],
                                        y: Union[pd.Series, np.ndarray],
                                        model: Any,
                                        feature_names: List[str],
                                        n_repeats: int) -> FeatureImportance:
        """Calculate permutation importance"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for permutation importance")

        from sklearn.inspection import permutation_importance

        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=self.random_state
        )

        # Get mean importance scores
        importance_scores = perm_importance.importances_mean

        # Create importance dictionary
        importances = dict(zip(feature_names, importance_scores))

        # Rank features by importance
        ranked_features = [feature_names[i] for i in importance_scores.argsort()[::-1]]

        return FeatureImportance(
            method=FeatureImportanceMethod.PERMUTATION,
            importances=importances,
            ranked_features=ranked_features,
            metadata={
                "importance_std": dict(zip(feature_names, perm_importance.importances_std)),
                "n_repeats": n_repeats
            }
        )

    def _calculate_shap_importance(self,
                                  X: Union[pd.DataFrame, np.ndarray],
                                  y: Union[pd.Series, np.ndarray],
                                  model: Any,
                                  feature_names: List[str]) -> FeatureImportance:
        """Calculate SHAP importance"""
        if not SHAP_AVAILABLE:
            raise ImportError("shap is required for SHAP importance")

        # Check if model is compatible with SHAP
        try:
            # Create explainer
            explainer = shap.Explainer(model, X)

            # Calculate SHAP values
            shap_values = explainer(X)

            # Get mean absolute SHAP values per feature
            if hasattr(shap_values, 'abs'):
                importance_scores = shap_values.abs.mean(axis=0).values
            else:
                importance_scores = np.abs(shap_values.values).mean(axis=0)

            # Create importance dictionary
            importances = dict(zip(feature_names, importance_scores))

            # Rank features by importance
            ranked_features = [feature_names[i] for i in importance_scores.argsort()[::-1]]

            return FeatureImportance(
                method=FeatureImportanceMethod.SHAP,
                importances=importances,
                ranked_features=ranked_features,
                metadata={
                    "shap_values": shap_values
                }
            )

        except Exception as e:
            logger.error(f"Error calculating SHAP importance: {str(e)}")
            raise

    def _calculate_mutual_info_importance(self,
                                        X: Union[pd.DataFrame, np.ndarray],
                                        y: Union[pd.Series, np.ndarray],
                                        feature_names: List[str]) -> FeatureImportance:
        """Calculate mutual information importance"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for mutual information importance")

        # Determine if classification or regression
        is_classification = False
        if hasattr(y, 'dtype'):
            is_classification = pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y)
        else:
            # Try to infer from number of unique values
            is_classification = len(np.unique(y)) < 10  # Heuristic

        # Select appropriate mutual information function
        if is_classification:
            mi_func = mutual_info_classif
        else:
            mi_func = mutual_info_regression

        # Calculate mutual information
        importance_scores = mi_func(X, y, random_state=self.random_state)

        # Create importance dictionary
        importances = dict(zip(feature_names, importance_scores))

        # Rank features by importance
        ranked_features = [feature_names[i] for i in importance_scores.argsort()[::-1]]

        return FeatureImportance(
            method=FeatureImportanceMethod.MUTUAL_INFO,
            importances=importances,
            ranked_features=ranked_features,
            metadata={
                "is_classification": is_classification
            }
        )

    def _calculate_model_specific_importance(self,
                                           X: Union[pd.DataFrame, np.ndarray],
                                           y: Union[pd.Series, np.ndarray],
                                           model: Any,
                                           feature_names: List[str]) -> FeatureImportance:
        """Calculate model-specific importance"""
        # Check if model has feature_importances_ attribute (e.g., RandomForest)
        if hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
        # Check if model has coef_ attribute (e.g., Linear models)
        elif hasattr(model, 'coef_'):
            importance_scores = np.abs(model.coef_)
            # Handle multi-class classification
            if len(importance_scores.shape) > 1:
                importance_scores = np.mean(np.abs(importance_scores), axis=0)
        else:
            raise ValueError("Model does not support intrinsic feature importance")

        # Create importance dictionary
        importances = dict(zip(feature_names, importance_scores))

        # Rank features by importance
        ranked_features = [feature_names[i] for i in importance_scores.argsort()[::-1]]

        return FeatureImportance(
            method=FeatureImportanceMethod.MODEL_SPECIFIC,
            importances=importances,
            ranked_features=ranked_features,
            metadata={
                "model_type": type(model).__name__
            }
        )

    def _calculate_correlation_importance(self,
                                        X: Union[pd.DataFrame, np.ndarray],
                                        y: Union[pd.Series, np.ndarray],
                                        feature_names: List[str]) -> FeatureImportance:
        """Calculate correlation-based importance"""
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feature_names)

        # Convert target to Series if needed
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # Calculate correlation with target
        correlations = []
        for col in X.columns:
            corr = X[col].corr(y)
            correlations.append(abs(corr))  # Use absolute correlation as importance

        # Create importance dictionary
        importances = dict(zip(feature_names, correlations))

        # Rank features by importance
        ranked_features = [feature_names[i] for i in np.argsort(correlations)[::-1]]

        return FeatureImportance(
            method=FeatureImportanceMethod.CORRELATION,
            importances=importances,
            ranked_features=ranked_features,
            metadata={}
        )

    def _calculate_rfe_importance(self,
                                X: Union[pd.DataFrame, np.ndarray],
                                y: Union[pd.Series, np.ndarray],
                                model: Any,
                                feature_names: List[str]) -> FeatureImportance:
        """Calculate importance using Recursive Feature Elimination"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for RFE importance")

        # Check if model implements required methods for RFE
        if not (hasattr(model, 'fit') and (hasattr(model, 'coef_') or hasattr(model, 'feature_importances_'))):
            raise ValueError("Model must implement fit and have coef_ or feature_importances_ attribute for RFE")

        # Create RFE estimator
        rfe = RFE(model, n_features_to_select=1, step=1)

        # Fit RFE
        rfe.fit(X, y)

        # Get ranking (lower is better)
        rankings = rfe.ranking_

        # Convert to importance (higher is better)
        max_rank = np.max(rankings)
        importance_scores = max_rank - rankings + 1

        # Create importance dictionary
        importances = dict(zip(feature_names, importance_scores))

        # Rank features by importance
        ranked_features = [feature_names[i] for i in importance_scores.argsort()[::-1]]

        return FeatureImportance(
            method=FeatureImportanceMethod.RFE,
            importances=importances,
            ranked_features=ranked_features,
            metadata={
                "rankings": dict(zip(feature_names, rankings)),
                "support": dict(zip(feature_names, rfe.support_))
            }
        )

    def _calculate_variance_importance(self,
                                     X: Union[pd.DataFrame, np.ndarray],
                                     feature_names: List[str]) -> FeatureImportance:
        """Calculate importance based on feature variance"""
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feature_names)

        # Calculate variance for each feature
        variances = X.var().values

        # Create importance dictionary
        importances = dict(zip(feature_names, variances))

        # Rank features by importance
        ranked_features = [feature_names[i] for i in variances.argsort()[::-1]]

        return FeatureImportance(
            method=FeatureImportanceMethod.VARIANCE,
            importances=importances,
            ranked_features=ranked_features,
            metadata={}
        )

    def _calculate_pca_importance(self,
                                X: Union[pd.DataFrame, np.ndarray],
                                feature_names: List[str]) -> FeatureImportance:
        """Calculate importance based on PCA components"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for PCA importance")

        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply PCA
        pca = PCA()
        pca.fit(X_scaled)

        # Calculate feature importance from component loadings
        n_components = min(5, len(feature_names))  # Use top 5 components or fewer
        loadings = np.abs(pca.components_[:n_components, :])
        importance_scores = np.sum(loadings * pca.explained_variance_ratio_[:n_components, np.newaxis], axis=0)

        # Create importance dictionary
        importances = dict(zip(feature_names, importance_scores))

        # Rank features by importance
        ranked_features = [feature_names[i] for i in importance_scores.argsort()[::-1]]

        return FeatureImportance(
            method=FeatureImportanceMethod.PCA,
            importances=importances,
            ranked_features=ranked_features,
            metadata={
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "n_components": n_components
            }
        )

    def select_features(self,
                       importance_result: FeatureImportance,
                       method: FeatureSelectionMethod = FeatureSelectionMethod.TOP_K,
                       k: int = 10,
                       threshold: float = 0.01,
                       cumulative_importance: float = 0.95) -> FeatureSelectionResult:
        """
        Select features based on importance scores.

        Args:
            importance_result: Feature importance calculation result
            method: Feature selection method
            k: Number of features to select for TOP_K method
            threshold: Importance threshold for THRESHOLD method
            cumulative_importance: Cumulative importance threshold for CUMULATIVE method

        Returns:
            FeatureSelectionResult object
        """
        if method == FeatureSelectionMethod.TOP_K:
            # Select top k features
            selected_features = importance_result.ranked_features[:k]

            return FeatureSelectionResult(
                method=method,
                selected_features=selected_features,
                importance_scores={f: importance_result.importances[f] for f in selected_features},
                metadata={
                    "k": k,
                    "total_features": len(importance_result.ranked_features)
                }
            )

        elif method == FeatureSelectionMethod.THRESHOLD:
            # Select features above threshold
            selected_features = [
                f for f in importance_result.ranked_features
                if importance_result.importances[f] >= threshold
            ]

            return FeatureSelectionResult(
                method=method,
                selected_features=selected_features,
                importance_scores={f: importance_result.importances[f] for f in selected_features},
                metadata={
                    "threshold": threshold,
                    "total_features": len(importance_result.ranked_features)
                }
            )

        elif method == FeatureSelectionMethod.CUMULATIVE:
            # Select features up to cumulative importance threshold
            importances = [importance_result.importances[f] for f in importance_result.ranked_features]

            # Normalize importances if they don't sum to 1
            total_importance = sum(importances)
            if total_importance <= 0:
                logger.warning("Total importance is zero or negative, using equal weights")
                normalized_importances = [1/len(importances)] * len(importances)
            else:
                normalized_importances = [imp / total_importance for imp in importances]

            # Calculate cumulative importance
            cumulative_importances = np.cumsum(normalized_importances)

            # Find cutoff index
            cutoff_idx = np.argmax(cumulative_importances >= cumulative_importance)

            # Select features up to cutoff
            selected_features = importance_result.ranked_features[:cutoff_idx + 1]

            return FeatureSelectionResult(
                method=method,
                selected_features=selected_features,
                importance_scores={f: importance_result.importances[f] for f in selected_features},
                metadata={
                    "cumulative_importance": cumulative_importance,
                    "actual_cumulative": cumulative_importances[cutoff_idx],
                    "total_features": len(importance_result.ranked_features)
                }
            )

        else:
            raise ValueError(f"Unsupported selection method: {method}")

    def generate_feature_report(self,
                              feature_summaries: Dict[str, FeatureSummary],
                              importance_result: Optional[FeatureImportance] = None,
                              selection_result: Optional[FeatureSelectionResult] = None,
                              output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive feature analysis report.

        Args:
            feature_summaries: Dictionary of feature summaries
            importance_result: Optional feature importance result
            selection_result: Optional feature selection result
            output_file: Optional file path to write report

        Returns:
            Dictionary with report data
        """
        # Create report structure
        report = {
            "report_time": datetime.now().isoformat(),
            "feature_count": len(feature_summaries),
            "feature_summaries": {name: summary.to_dict() for name, summary in feature_summaries.items()},
            "feature_types": {
                "numeric": [name for name, summary in feature_summaries.items() if summary.feature_type == "numeric"],
                "categorical": [name for name, summary in feature_summaries.items() if summary.feature_type == "categorical"],
                "datetime": [name for name, summary in feature_summaries.items() if summary.feature_type == "datetime"],
                "other": [name for name, summary in feature_summaries.items() if summary.feature_type == "other"]
            }
        }

        # Add importance information if available
        if importance_result:
            report["importance"] = {
                "method": importance_result.method.value,
                "scores": importance_result.importances,
                "ranked_features": importance_result.ranked_features
            }

        # Add selection information if available
        if selection_result:
            report["selection"] = {
                "method": selection_result.method.value,
                "selected_features": selection_result.selected_features,
                "selected_count": len(selection_result.selected_features),
                "importance_scores": selection_result.importance_scores,
                "metadata": selection_result.metadata
            }

        # Add correlation matrix if available
        if any(summary.feature_type == "numeric" for summary in feature_summaries.values()):
            numeric_features = [name for name, summary in feature_summaries.items() if summary.feature_type == "numeric"]

            # Check if we have feature data
            if all(hasattr(feature_summaries[name], 'data') for name in numeric_features):
                # Create correlation matrix
                correlation_data = pd.DataFrame({
                    name: feature_summaries[name].data for name in numeric_features
                })

                correlation_matrix = correlation_data.corr().to_dict()

                report["correlations"] = correlation_matrix

        # Write report to file if specified
        if output_file:
            output_path = os.path.join(self.output_dir, output_file)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=4)

        return report

    def plot_feature_importance(self,
                              importance_result: FeatureImportance,
                              top_n: int = 20,
                              figsize: Tuple[int, int] = (12, 8),
                              output_file: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance scores.

        Args:
            importance_result: Feature importance result
            top_n: Number of top features to display
            figsize: Figure size
            output_file: Optional file to save the plot

        Returns:
            Matplotlib figure
        """
        # Get top N features
        top_features = importance_result.ranked_features[:top_n]
        top_importances = [importance_result.importances[f] for f in top_features]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot horizontal bar chart
        bars = ax.barh(range(len(top_features)), top_importances, align='center')

        # Add feature names as labels
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)

        # Add importance values as text
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_x_pos = width * 1.01
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.4f}',
                   va='center')

        # Add titles and labels
        ax.set_title(f'Feature Importance ({importance_result.method.value})', fontsize=14)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)

        # Invert y-axis to show most important features at the top
        ax.invert_yaxis()

        # Add grid lines
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)

        # Tight layout
        plt.tight_layout()

        # Save figure if output file is specified
        if output_file:
            output_path = os.path.join(self.output_dir, output_file)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_correlation_matrix(self,
                              feature_summaries: Dict[str, FeatureSummary],
                              figsize: Tuple[int, int] = (12, 10),
                              output_file: Optional[str] = None) -> plt.Figure:
        """
        Plot correlation matrix for numeric features.

        Args:
            feature_summaries: Dictionary of feature summaries
            figsize: Figure size
            output_file: Optional file to save the plot

        Returns:
            Matplotlib figure
        """
        # Get numeric features
        numeric_features = [name for name, summary in feature_summaries.items() if summary.feature_type == "numeric"]

        # Create DataFrame with numeric features
        data = pd.DataFrame({
            name: feature_summaries[name].data for name in numeric_features
        })

        # Calculate correlation matrix
        corr_matrix = data.corr()

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                   annot=True, fmt=".2f", square=True, linewidths=.5, ax=ax)

        # Set title
        ax.set_title('Feature Correlation Matrix', fontsize=14)

        # Tight layout
        plt.tight_layout()

        # Save figure if output file is specified
        if output_file:
            output_path = os.path.join(self.output_dir, output_file)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_feature_distributions(self,
                                 feature_summaries: Dict[str, FeatureSummary],
                                 feature_subset: Optional[List[str]] = None,
                                 ncols: int = 3,
                                 figsize: Tuple[int, int] = (15, 12),
                                 output_file: Optional[str] = None) -> plt.Figure:
        """
        Plot distributions of features.

        Args:
            feature_summaries: Dictionary of feature summaries
            feature_subset: Optional subset of features to plot
            ncols: Number of columns in the grid
            figsize: Figure size
            output_file: Optional file to save the plot

        Returns:
            Matplotlib figure
        """
        # Select features to plot
        if feature_subset:
            features_to_plot = [f for f in feature_subset if f in feature_summaries]
        else:
            # Default to top 12 features or all if fewer
            features_to_plot = list(feature_summaries.keys())[:12]

        # Calculate grid dimensions
        n_features = len(features_to_plot)
        nrows = (n_features + ncols - 1) // ncols

        # Create figure
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = axes.flatten() if n_features > 1 else [axes]

        # Plot each feature
        for i, feature_name in enumerate(features_to_plot):
            if i < len(axes):
                ax = axes[i]
                feature = feature_summaries[feature_name]

                if feature.feature_type == "numeric":
                    # Histogram for numeric features
                    ax.hist(feature.data.dropna(), bins=30, alpha=0.7, edgecolor='black')

                    # Add mean and median lines
                    if 'mean' in feature.summary:
                        ax.axvline(feature.summary['mean'], color='red', linestyle='--', label=f"Mean: {feature.summary['mean']:.2f}")
                    if '50%' in feature.summary:
                        ax.axvline(feature.summary['50%'], color='green', linestyle='--', label=f"Median: {feature.summary['50%']:.2f}")

                    ax.legend(fontsize=8)

                elif feature.feature_type == "categorical":
                    # Bar chart for categorical features
                    value_counts = feature.data.value_counts().sort_values(ascending=False).head(10)
                    value_counts.plot(kind='bar', ax=ax)

                    # Rotate x-axis labels
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

                # Set title and labels
                ax.set_title(feature_name)
                ax.set_xlabel('')
                ax.set_ylabel('Count')

                # Add grid
                ax.grid(True, linestyle='--', alpha=0.6)

        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        # Add overall title
        plt.suptitle('Feature Distributions', fontsize=16, y=1.02)

        # Tight layout
        plt.tight_layout()

        # Save figure if output file is specified
        if output_file:
            output_path = os.path.join(self.output_dir, output_file)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        return fig

    def engineer_features(self,
                        data: pd.DataFrame,
                        target: Optional[pd.Series] = None,
                        time_column: Optional[str] = None,
                        numeric_transformations: List[str] = None,
                        interaction_terms: bool = True,
                        temporal_features: bool = True,
                        polynomial_features: bool = False,
                        polynomial_degree: int = 2) -> pd.DataFrame:
        """
        Engineer new features from existing ones.

        Args:
            data: DataFrame with features
            target: Optional target variable
            time_column: Optional time column name
            numeric_transformations: Transformations for numeric features
            interaction_terms: Whether to create interaction terms
            temporal_features: Whether to create temporal features
            polynomial_features: Whether to create polynomial features
            polynomial_degree: Degree for polynomial features

        Returns:
            DataFrame with engineered features
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for feature engineering")

        # Clone data to avoid modifying the original
        result = data.copy()

        # Default transformations
        if numeric_transformations is None:
            numeric_transformations = ['log', 'sqrt', 'square']

        # Get numeric columns
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        # Apply transformations to numeric features
        for col in numeric_cols:
            feature_data = data[col]

            # Skip columns with zeros/negatives for certain transformations
            has_nonpositive = (feature_data <= 0).any()

            for transform in numeric_transformations:
                if transform == 'log' and not has_nonpositive:
                    result[f'{col}_log'] = np.log(feature_data)
                elif transform == 'sqrt' and not has_nonpositive:
                    result[f'{col}_sqrt'] = np.sqrt(feature_data)
                elif transform == 'square':
                    result[f'{col}_squared'] = feature_data ** 2
                elif transform == 'cube':
                    result[f'{col}_cubed'] = feature_data ** 3
                elif transform == 'reciprocal' and not (feature_data == 0).any():
                    result[f'{col}_reciprocal'] = 1 / feature_data
                elif transform == 'zscore':
                    mean = feature_data.mean()
                    std = feature_data.std()
                    if std > 0:
                        result[f'{col}_zscore'] = (feature_data - mean) / std

        # Create interaction terms between numeric features
        if interaction_terms and len(numeric_cols) > 1:
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    # Multiplication interaction
                    result[f'{col1}_times_{col2}'] = data[col1] * data[col2]

                    # Ratio interaction (avoid division by zero)
                    if not (data[col2] == 0).any():
                        result[f'{col1}_div_{col2}'] = data[col1] / data[col2]

                    if not (data[col1] == 0).any():
                        result[f'{col2}_div_{col1}'] = data[col2] / data[col1]

        # Create temporal features if time column is provided
        if temporal_features and time_column and time_column in data.columns:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_dtype(data[time_column]):
                time_data = pd.to_datetime(data[time_column], errors='coerce')
            else:
                time_data = data[time_column]

            # Extract datetime components
            result['hour_of_day'] = time_data.dt.hour
            result['day_of_week'] = time_data.dt.dayofweek
            result['day_of_month'] = time_data.dt.day
            result['week_of_year'] = time_data.dt.isocalendar().week
            result['month'] = time_data.dt.month
            result['quarter'] = time_data.dt.quarter
            result['year'] = time_data.dt.year

            # Create cyclical features for periodic variables
            result['hour_sin'] = np.sin(2 * np.pi * result['hour_of_day'] / 24)
            result['hour_cos'] = np.cos(2 * np.pi * result['hour_of_day'] / 24)
            result['day_of_week_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
            result['day_of_week_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
            result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
            result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)

            # Create lag features if target is provided
            if target is not None:
                # Add lag 1, 7, and 30 (common for time series)
                for lag in [1, 7, 30]:
                    if len(target) > lag:
                        result[f'target_lag_{lag}'] = target.shift(lag)

        # Create polynomial features
        if polynomial_features and polynomial_degree > 1:
            from sklearn.preprocessing import PolynomialFeatures

            # Select subset of numeric features to avoid explosion of features
            if len(numeric_cols) > 5:
                # Use top 5 features if available
                if target is not None:
                    # Calculate correlation with target
                    correlations = [(col, abs(data[col].corr(target))) for col in numeric_cols]
                    sorted_cols = sorted(correlations, key=lambda x: x[1], reverse=True)
                    poly_cols = [col for col, _ in sorted_cols[:5]]
                else:
                    # Just use first 5 features
                    poly_cols = numeric_cols[:5]
            else:
                poly_cols = numeric_cols

            # Generate polynomial features
            poly = PolynomialFeatures(degree=polynomial_degree, include_bias=False, interaction_only=False)
            poly_features = poly.fit_transform(data[poly_cols])

            # Get feature names
            poly_feature_names = poly.get_feature_names_out(poly_cols)

            # Add to result DataFrame
            for i, name in enumerate(poly_feature_names):
                # Skip original features
                if name in poly_cols:
                    continue

                # Replace ^ with _ for better readability
                readable_name = name.replace("^", "_pow_").replace(" ", "")
                result[f'poly_{readable_name}'] = poly_features[:, i]

        # Remove constant features
        for col in result.columns:
            if result[col].nunique() <= 1:
                result.drop(col, axis=1, inplace=True)
                logger.debug(f"Dropped constant feature: {col}")

        logger.info(f"Engineered {len(result.columns) - len(data.columns)} new features")

        return result

    def save_feature_metadata(self,
                            feature_metadata: Dict[str, Any],
                            feature_set_name: str) -> bool:
        """
        Save feature metadata to feature store.

        Args:
            feature_metadata: Dictionary with feature metadata
            feature_set_name: Name of the feature set

        Returns:
            Success boolean
        """
        if self.feature_store is None:
            logger.warning("Feature store not available, metadata not saved")
            return False

        try:
            # Extract feature names
            if "feature_summaries" in feature_metadata:
                features = list(feature_metadata["feature_summaries"].keys())
            else:
                features = feature_metadata.get("features", [])

            # Create metadata for feature store
            metadata = {
                "created_at": datetime.now().isoformat(),
                "feature_count": len(features),
                "features": features,
                "analysis": feature_metadata
            }

            # Store in feature store
            self.feature_store.update_feature_metadata(
                features=features,
                metadata=metadata,
                feature_set=feature_set_name
            )

            logger.info(f"Saved metadata for {len(features)} features in feature set '{feature_set_name}'")
            return True

        except Exception as e:
            logger.error(f"Error saving feature metadata: {str(e)}")
            return False


# Factory function for creating feature analyzers
def create_feature_analyzer(feature_store: Optional[FeatureStore] = None, **kwargs) -> FeatureAnalyzer:
    """
    Create a feature analyzer.

    Args:
        feature_store: Optional feature store
        **kwargs: Additional parameters for the analyzer

    Returns:
        FeatureAnalyzer instance
    """
    return FeatureAnalyzer(feature_store=feature_store, **kwargs)