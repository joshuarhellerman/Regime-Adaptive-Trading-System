"""
Unit tests for the feature_analyzer module.
"""

import pytest
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from unittest.mock import MagicMock, patch

from models.research.feature_analyzer import (
    FeatureAnalyzer,
    FeatureImportanceMethod,
    FeatureSelectionMethod,
    FeatureSummary,
    FeatureImportance,
    FeatureSelectionResult,
    create_feature_analyzer
)

class TestFeatureSummary:
    """Tests for the FeatureSummary class"""

    def test_numeric_feature_summary(self):
        """Test summarizing a numeric feature"""
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        summary = FeatureSummary("test_numeric", data)

        assert summary.name == "test_numeric"
        assert summary.feature_type == "numeric"
        assert summary.summary["count"] == 5
        assert summary.summary["mean"] == 3.0
        assert summary.summary["min"] == 1.0
        assert summary.summary["max"] == 5.0
        assert summary.summary["positive_count"] == 5
        assert "sharpe" in summary.summary

    def test_categorical_feature_summary(self):
        """Test summarizing a categorical feature"""
        data = pd.Series(["A", "B", "A", "C", "B"])
        summary = FeatureSummary("test_categorical", data)

        assert summary.name == "test_categorical"
        assert summary.feature_type == "categorical"
        assert summary.summary["count"] == 5
        assert summary.summary["unique_count"] == 3
        assert "value_distribution" in summary.summary
        assert summary.summary["value_distribution"]["A"] == 2

    def test_datetime_feature_summary(self):
        """Test summarizing a datetime feature"""
        dates = pd.date_range(start='2020-01-01', periods=5)
        data = pd.Series(dates)
        summary = FeatureSummary("test_datetime", data)

        assert summary.name == "test_datetime"
        assert summary.feature_type == "datetime"
        assert summary.summary["count"] == 5
        assert "range_days" in summary.summary

    def test_missing_values_handling(self):
        """Test handling of missing values in feature summary"""
        data = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
        summary = FeatureSummary("test_with_missing", data)

        assert summary.summary["count"] == 5
        assert summary.summary["missing"] == 1
        assert summary.summary["mean"] < 3.0  # Mean should exclude the NaN


class TestFeatureAnalyzer:
    """Tests for the FeatureAnalyzer class"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        n_samples = 100

        # Create simple feature matrix
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(5, 2, n_samples),
            'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
            'feature4': pd.date_range(start='2020-01-01', periods=n_samples),
            'feature5': np.random.normal(0, 1, n_samples) * 5  # Correlated with feature1
        })

        # Create target variable (correlated with feature1 and feature5)
        y = X['feature1'] * 2 + X['feature5'] * 0.5 + np.random.normal(0, 1, n_samples)

        return X, y

    @pytest.fixture
    def feature_analyzer(self):
        """Create a feature analyzer instance for testing"""
        return FeatureAnalyzer(
            output_dir="./test_output",
            random_state=42
        )

    def test_initialization(self, feature_analyzer):
        """Test initialization of FeatureAnalyzer"""
        assert feature_analyzer.random_state == 42
        assert feature_analyzer.output_dir == "./test_output"
        assert len(feature_analyzer.importance_methods) > 0
        assert len(feature_analyzer.selection_methods) > 0

    def test_analyze_features(self, feature_analyzer, sample_data):
        """Test feature analysis"""
        X, y = sample_data
        feature_summaries = feature_analyzer.analyze_features(X, y)

        assert len(feature_summaries) == 5
        assert all(isinstance(summary, FeatureSummary) for summary in feature_summaries.values())
        assert feature_summaries['feature1'].feature_type == "numeric"
        assert feature_summaries['feature3'].feature_type == "categorical"
        assert feature_summaries['feature4'].feature_type == "datetime"

    def test_analyze_features_with_subset(self, feature_analyzer, sample_data):
        """Test feature analysis with feature subset"""
        X, y = sample_data
        feature_subset = ['feature1', 'feature2']
        feature_summaries = feature_analyzer.analyze_features(X, y, feature_subset=feature_subset)

        assert len(feature_summaries) == 2
        assert set(feature_summaries.keys()) == set(feature_subset)

    def test_target_correlations(self, feature_analyzer, sample_data):
        """Test target correlation calculation"""
        X, y = sample_data
        feature_summaries = feature_analyzer.analyze_features(X, y)

        # Check that target correlation is calculated for numeric features
        assert 'target_correlation' in feature_summaries['feature1'].summary
        assert 'target_correlation' in feature_summaries['feature2'].summary
        assert 'target_correlation' in feature_summaries['feature5'].summary

        # Correlation should be higher for feature1 and feature5 which were used to generate y
        assert abs(feature_summaries['feature1'].summary['target_correlation']) > abs(feature_summaries['feature2'].summary['target_correlation'])
        assert abs(feature_summaries['feature5'].summary['target_correlation']) > abs(feature_summaries['feature2'].summary['target_correlation'])

    @pytest.mark.skipif(not hasattr(pd.Series, 'autocorr'), reason="pandas version doesn't support autocorr")
    def test_numeric_feature_trading_metrics(self, feature_analyzer, sample_data):
        """Test trading-specific metrics for numeric features"""
        X, y = sample_data
        feature_summaries = feature_analyzer.analyze_features(X)

        # Check for trading metrics in numeric features
        assert 'sharpe' in feature_summaries['feature1'].summary
        assert 'abs_mean' in feature_summaries['feature1'].summary
        assert 'positive_ratio' in feature_summaries['feature1'].summary

    @patch('matplotlib.pyplot.savefig')
    def test_plot_feature_distributions(self, mock_savefig, feature_analyzer, sample_data):
        """Test plotting feature distributions"""
        X, _ = sample_data
        feature_summaries = feature_analyzer.analyze_features(X)

        # Test with default parameters
        fig = feature_analyzer.plot_feature_distributions(feature_summaries)
        assert isinstance(fig, plt.Figure)

        # Test with feature subset
        fig = feature_analyzer.plot_feature_distributions(
            feature_summaries,
            feature_subset=['feature1', 'feature2'],
            output_file='test_distributions.png'
        )
        assert isinstance(fig, plt.Figure)
        mock_savefig.assert_called_once()

    def test_calculate_correlation_importance(self, feature_analyzer, sample_data):
        """Test correlation importance calculation"""
        X, y = sample_data

        # Keep only numeric features
        X_numeric = X.select_dtypes(include=['number'])

        importance = feature_analyzer._calculate_correlation_importance(X_numeric, y, X_numeric.columns.tolist())

        assert isinstance(importance, FeatureImportance)
        assert importance.method == FeatureImportanceMethod.CORRELATION
        assert len(importance.importances) == X_numeric.shape[1]
        assert len(importance.ranked_features) == X_numeric.shape[1]

        # Feature1 and feature5 should have higher importance
        assert 'feature1' in importance.ranked_features[:2]
        assert 'feature5' in importance.ranked_features[:2]

    def test_variance_importance(self, feature_analyzer, sample_data):
        """Test variance-based feature importance"""
        X, _ = sample_data

        # Keep only numeric features
        X_numeric = X.select_dtypes(include=['number'])

        importance = feature_analyzer._calculate_variance_importance(X_numeric, X_numeric.columns.tolist())

        assert isinstance(importance, FeatureImportance)
        assert importance.method == FeatureImportanceMethod.VARIANCE
        assert len(importance.importances) == X_numeric.shape[1]

        # Feature2 should have highest variance (normal with std=2)
        assert importance.ranked_features[0] == 'feature2'

    @pytest.mark.skipif(not pytest.importorskip("sklearn", reason="sklearn not installed"), reason="sklearn not available")
    def test_mutual_info_importance(self, feature_analyzer, sample_data):
        """Test mutual information importance calculation"""
        X, y = sample_data

        # Keep only numeric features
        X_numeric = X.select_dtypes(include=['number'])

        importance = feature_analyzer._calculate_mutual_info_importance(X_numeric, y, X_numeric.columns.tolist())

        assert isinstance(importance, FeatureImportance)
        assert importance.method == FeatureImportanceMethod.MUTUAL_INFO
        assert len(importance.importances) == X_numeric.shape[1]
        assert 'is_classification' in importance.metadata

    def test_top_k_feature_selection(self, feature_analyzer, sample_data):
        """Test top-k feature selection"""
        X, y = sample_data

        # Calculate correlation importance
        X_numeric = X.select_dtypes(include=['number'])
        importance = feature_analyzer._calculate_correlation_importance(X_numeric, y, X_numeric.columns.tolist())

        # Select top 2 features
        selection_result = feature_analyzer.select_features(
            importance_result=importance,
            method=FeatureSelectionMethod.TOP_K,
            k=2
        )

        assert isinstance(selection_result, FeatureSelectionResult)
        assert selection_result.method == FeatureSelectionMethod.TOP_K
        assert len(selection_result.selected_features) == 2

        # Feature1 and feature5 should be selected
        assert set(selection_result.selected_features) == {'feature1', 'feature5'}

    def test_threshold_feature_selection(self, feature_analyzer, sample_data):
        """Test threshold-based feature selection"""
        X, y = sample_data

        # Calculate correlation importance
        X_numeric = X.select_dtypes(include=['number'])
        importance = feature_analyzer._calculate_correlation_importance(X_numeric, y, X_numeric.columns.tolist())

        # Find a threshold that selects the top 2 features
        threshold = sorted(importance.importances.values(), reverse=True)[1] - 0.01

        selection_result = feature_analyzer.select_features(
            importance_result=importance,
            method=FeatureSelectionMethod.THRESHOLD,
            threshold=threshold
        )

        assert isinstance(selection_result, FeatureSelectionResult)
        assert selection_result.method == FeatureSelectionMethod.THRESHOLD
        assert len(selection_result.selected_features) == 2

    def test_cumulative_importance_selection(self, feature_analyzer, sample_data):
        """Test cumulative importance feature selection"""
        X, y = sample_data

        # Calculate correlation importance
        X_numeric = X.select_dtypes(include=['number'])
        importance = feature_analyzer._calculate_correlation_importance(X_numeric, y, X_numeric.columns.tolist())

        selection_result = feature_analyzer.select_features(
            importance_result=importance,
            method=FeatureSelectionMethod.CUMULATIVE,
            cumulative_importance=0.95
        )

        assert isinstance(selection_result, FeatureSelectionResult)
        assert selection_result.method == FeatureSelectionMethod.CUMULATIVE
        assert len(selection_result.selected_features) >= 1
        assert 'actual_cumulative' in selection_result.metadata

    def test_generate_feature_report(self, feature_analyzer, sample_data):
        """Test generating feature report"""
        X, y = sample_data

        # Analyze features
        feature_summaries = feature_analyzer.analyze_features(X, y)

        # Calculate importance
        X_numeric = X.select_dtypes(include=['number'])
        importance_result = feature_analyzer._calculate_correlation_importance(X_numeric, y, X_numeric.columns.tolist())

        # Select features
        selection_result = feature_analyzer.select_features(
            importance_result=importance_result,
            method=FeatureSelectionMethod.TOP_K,
            k=2
        )

        # Generate report
        report = feature_analyzer.generate_feature_report(
            feature_summaries=feature_summaries,
            importance_result=importance_result,
            selection_result=selection_result,
            output_file="test_report.json"
        )

        assert isinstance(report, dict)
        assert "feature_count" in report
        assert "feature_summaries" in report
        assert "feature_types" in report
        assert "importance" in report
        assert "selection" in report
        assert os.path.exists(os.path.join(feature_analyzer.output_dir, "test_report.json"))

    def test_engineer_features(self, feature_analyzer, sample_data):
        """Test feature engineering"""
        X, y = sample_data

        # Engineer features
        X_engineered = feature_analyzer.engineer_features(
            data=X,
            target=y,
            time_column='feature4',
            numeric_transformations=['log', 'sqrt', 'square'],
            interaction_terms=True,
            temporal_features=True
        )

        assert isinstance(X_engineered, pd.DataFrame)
        assert X_engineered.shape[1] > X.shape[1]
        assert 'feature1_squared' in X_engineered.columns
        assert 'feature1_times_feature2' in X_engineered.columns
        assert 'hour_of_day' in X_engineered.columns

    @patch('matplotlib.pyplot.savefig')
    def test_plot_feature_importance(self, mock_savefig, feature_analyzer, sample_data):
        """Test plotting feature importance"""
        X, y = sample_data

        # Calculate importance
        X_numeric = X.select_dtypes(include=['number'])
        importance_result = feature_analyzer._calculate_correlation_importance(X_numeric, y, X_numeric.columns.tolist())

        # Plot feature importance
        fig = feature_analyzer.plot_feature_importance(
            importance_result=importance_result,
            top_n=3,
            output_file="importance_plot.png"
        )

        assert isinstance(fig, plt.Figure)
        mock_savefig.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    def test_plot_correlation_matrix(self, mock_savefig, feature_analyzer, sample_data):
        """Test plotting correlation matrix"""
        X, _ = sample_data

        # Analyze features
        feature_summaries = feature_analyzer.analyze_features(X)

        # Plot correlation matrix
        fig = feature_analyzer.plot_correlation_matrix(
            feature_summaries=feature_summaries,
            output_file="correlation_matrix.png"
        )

        assert isinstance(fig, plt.Figure)
        mock_savefig.assert_called_once()


class TestFactoryFunction:
    """Tests for the create_feature_analyzer factory function"""

    def test_create_feature_analyzer(self):
        """Test creating feature analyzer with factory function"""
        # Create mock feature store
        mock_feature_store = MagicMock()

        # Create analyzer with factory function
        analyzer = create_feature_analyzer(
            feature_store=mock_feature_store,
            random_state=42,
            output_dir="custom_dir"
        )

        assert isinstance(analyzer, FeatureAnalyzer)
        assert analyzer.feature_store == mock_feature_store
        assert analyzer.random_state == 42
        assert analyzer.output_dir == "custom_dir"

    def test_create_feature_analyzer_default(self):
        """Test creating feature analyzer with defaults"""
        analyzer = create_feature_analyzer()

        assert isinstance(analyzer, FeatureAnalyzer)
        assert analyzer.feature_store is None
        assert len(analyzer.importance_methods) > 0
        assert analyzer.output_dir == "data/feature_analysis"


if __name__ == "__main__":
    pytest.main(["-v", "test_feature_analyzer.py"])