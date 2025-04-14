import unittest
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import os
import tempfile
from pathlib import Path

# Import the modules to test
from models.research.research_environment import (
    ResearchEnvironment,
    ResearchConfig,
    TimeSeriesSplit,
    DataSplitMode,
    BacktestMode
)
from data.market_data_service import MarketDataService, DataTimeframe, DataSource
from data.feature_store import FeatureStore
from models.research.feature_analyzer import FeatureAnalyzer
from models.research.model_validator import ModelValidator
from models.research.hyperparameter_optimization import HyperparameterOptimizer


class TestResearchConfig(unittest.TestCase):
    """Test cases for ResearchConfig dataclass"""

    def test_default_values(self):
        """Test that default values are set correctly"""
        config = ResearchConfig(
            symbols=["AAPL"],
            timeframes=[DataTimeframe.DAILY],
            start_date="2020-01-01",
            end_date="2020-12-31"
        )

        # Check default values
        self.assertEqual(config.data_source, DataSource.AUTO)
        self.assertEqual(config.data_split_mode, DataSplitMode.TRAIN_VALIDATION_TEST)
        self.assertEqual(config.train_ratio, 0.7)
        self.assertEqual(config.validation_ratio, 0.15)
        self.assertEqual(config.test_ratio, 0.15)
        self.assertEqual(config.backtest_mode, BacktestMode.VECTORIZED)
        self.assertEqual(config.initial_capital, 100000.0)
        self.assertEqual(config.output_dir, "data/research")
        self.assertEqual(config.random_state, None)
        self.assertTrue(config.use_feature_store)
        self.assertTrue(config.auto_feature_engineering)
        self.assertEqual(config.name, "research_environment")
        self.assertEqual(config.description, "")
        self.assertEqual(config.tags, [])

    def test_manual_values(self):
        """Test that manual values are set correctly"""
        config = ResearchConfig(
            symbols=["AAPL", "MSFT"],
            timeframes=[DataTimeframe.DAILY, "1H"],
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            data_source=DataSource.CSV,
            data_split_mode=DataSplitMode.CROSS_VALIDATION,
            train_ratio=0.8,
            validation_ratio=0.1,
            test_ratio=0.1,
            backtest_mode=BacktestMode.EVENT_DRIVEN,
            initial_capital=50000.0,
            output_dir="custom/output",
            random_state=42,
            use_feature_store=False,
            auto_feature_engineering=False,
            name="custom_research",
            description="Custom research environment",
            tags=["stocks", "daily"]
        )

        # Check custom values
        self.assertEqual(config.symbols, ["AAPL", "MSFT"])
        self.assertEqual(config.timeframes, [DataTimeframe.DAILY, "1H"])
        self.assertEqual(config.start_date, datetime(2020, 1, 1))
        self.assertEqual(config.end_date, datetime(2020, 12, 31))
        self.assertEqual(config.data_source, DataSource.CSV)
        self.assertEqual(config.data_split_mode, DataSplitMode.CROSS_VALIDATION)
        self.assertEqual(config.train_ratio, 0.8)
        self.assertEqual(config.validation_ratio, 0.1)
        self.assertEqual(config.test_ratio, 0.1)
        self.assertEqual(config.backtest_mode, BacktestMode.EVENT_DRIVEN)
        self.assertEqual(config.initial_capital, 50000.0)
        self.assertEqual(config.output_dir, "custom/output")
        self.assertEqual(config.random_state, 42)
        self.assertFalse(config.use_feature_store)
        self.assertFalse(config.auto_feature_engineering)
        self.assertEqual(config.name, "custom_research")
        self.assertEqual(config.description, "Custom research environment")
        self.assertEqual(config.tags, ["stocks", "daily"])


class TestTimeSeriesSplit(unittest.TestCase):
    """Test cases for TimeSeriesSplit class"""

    def setUp(self):
        """Set up test data"""
        # Create sample time series data
        dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
        np.random.seed(42)  # For reproducible tests
        prices = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
        
        self.test_data = pd.DataFrame({
            "close": prices,
            "volume": np.random.randint(1000, 10000, len(dates)),
            "feature1": np.random.random(len(dates)),
            "feature2": np.random.random(len(dates))
        }, index=dates)

    def test_train_test_split(self):
        """Test train/test split mode"""
        # Create split
        split = TimeSeriesSplit(
            data=self.test_data,
            mode=DataSplitMode.TRAIN_TEST,
            train_ratio=0.8,
            test_ratio=0.2
        )

        # Check number of splits
        self.assertEqual(len(split.splits), 1)

        # Check split data
        split_data = split.get_split(0)
        self.assertIn("train", split_data)
        self.assertIn("test", split_data)
        self.assertNotIn("validation", split_data)

        # Check split sizes
        total_rows = len(self.test_data)
        expected_train_size = int(total_rows * 0.8)
        expected_test_size = total_rows - expected_train_size

        self.assertEqual(len(split_data["train"]), expected_train_size)
        self.assertEqual(len(split_data["test"]), expected_test_size)

        # Check that split maintains chronological order
        self.assertTrue(split_data["train"].index.max() < split_data["test"].index.min())

    def test_train_validation_test_split(self):
        """Test train/validation/test split mode"""
        # Create split
        split = TimeSeriesSplit(
            data=self.test_data,
            mode=DataSplitMode.TRAIN_VALIDATION_TEST,
            train_ratio=0.7,
            validation_ratio=0.15,
            test_ratio=0.15
        )

        # Check number of splits
        self.assertEqual(len(split.splits), 1)

        # Check split data
        split_data = split.get_split(0)
        self.assertIn("train", split_data)
        self.assertIn("validation", split_data)
        self.assertIn("test", split_data)

        # Check split sizes
        total_rows = len(self.test_data)
        expected_train_size = int(total_rows * 0.7)
        expected_val_size = int(total_rows * 0.15)
        expected_test_size = total_rows - expected_train_size - expected_val_size

        self.assertEqual(len(split_data["train"]), expected_train_size)
        self.assertEqual(len(split_data["validation"]), expected_val_size)
        self.assertEqual(len(split_data["test"]), expected_test_size)

        # Check that split maintains chronological order
        self.assertTrue(split_data["train"].index.max() < split_data["validation"].index.min())
        self.assertTrue(split_data["validation"].index.max() < split_data["test"].index.min())

    def test_cross_validation_split(self):
        """Test cross-validation split mode"""
        # Create split
        split = TimeSeriesSplit(
            data=self.test_data,
            mode=DataSplitMode.CROSS_VALIDATION,
            train_ratio=0.7,
            n_splits=3
        )

        # Check number of splits
        self.assertEqual(len(split.splits), 3)

        # Check each split
        for i in range(3):
            split_data = split.get_split(i)
            self.assertIn("train", split_data)
            self.assertIn("test", split_data)
            
            # Check that split maintains chronological order
            self.assertTrue(split_data["train"].index.max() <= split_data["test"].index.min())

    def test_invalid_mode(self):
        """Test invalid split mode"""
        with self.assertRaises(ValueError):
            TimeSeriesSplit(
                data=self.test_data,
                mode="invalid_mode"
            )

    def test_get_invalid_split(self):
        """Test getting an invalid split index"""
        split = TimeSeriesSplit(
            data=self.test_data,
            mode=DataSplitMode.TRAIN_TEST
        )

        with self.assertRaises(ValueError):
            split.get_split(1)  # Only have 1 split (index 0)

    def test_get_all_splits(self):
        """Test getting all splits"""
        split = TimeSeriesSplit(
            data=self.test_data,
            mode=DataSplitMode.CROSS_VALIDATION,
            n_splits=5
        )

        all_splits = split.get_all_splits()
        self.assertEqual(len(all_splits), 5)


class TestResearchEnvironment(unittest.TestCase):
    """Test cases for ResearchEnvironment class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create sample data
        dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
        np.random.seed(42)  # For reproducibility
        
        # Create OHLCV data
        close_prices = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
        self.test_data = pd.DataFrame({
            "open": close_prices * (1 + np.random.normal(0, 0.01, len(dates))),
            "high": close_prices * (1 + np.random.normal(0, 0.02, len(dates))),
            "low": close_prices * (1 - np.random.normal(0, 0.02, len(dates))),
            "close": close_prices,
            "volume": np.random.randint(1000, 10000, len(dates)),
        }, index=dates)
        
        # Create mocks
        self.mock_market_data_service = MagicMock(spec=MarketDataService)
        self.mock_feature_store = MagicMock(spec=FeatureStore)
        self.mock_feature_analyzer = MagicMock(spec=FeatureAnalyzer)
        self.mock_model_validator = MagicMock(spec=ModelValidator)
        self.mock_hyperparameter_optimizer = MagicMock(spec=HyperparameterOptimizer)
        
        # Set up mock behavior
        self.mock_market_data_service.get_data.return_value = self.test_data
        
        # Create temp directory for output
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create config
        self.config = ResearchConfig(
            symbols=["AAPL"],
            timeframes=[DataTimeframe.DAILY],
            start_date="2020-01-01",
            end_date="2020-12-31",
            output_dir=self.temp_dir.name,
            random_state=42
        )

    def tearDown(self):
        """Clean up after tests"""
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test environment initialization"""
        env = ResearchEnvironment(
            config=self.config,
            market_data_service=self.mock_market_data_service,
            feature_store=self.mock_feature_store,
            feature_analyzer=self.mock_feature_analyzer,
            model_validator=self.mock_model_validator,
            hyperparameter_optimizer=self.mock_hyperparameter_optimizer
        )

        # Check that data was loaded
        self.mock_market_data_service.get_data.assert_called_with(
            symbol="AAPL",
            timeframe=DataTimeframe.DAILY.value,
            start_date="2020-01-01",
            end_date="2020-12-31",
            source=DataSource.AUTO
        )

    def test_initialization_without_market_data(self):
        """Test initialization without market data service"""
        env = ResearchEnvironment(
            config=self.config,
            feature_store=self.mock_feature_store,
            feature_analyzer=self.mock_feature_analyzer
        )

        # Check that data was not loaded
        self.assertEqual(len(env.data), 0)

    def test_create_splits(self):
        """Test creating data splits"""
        # Create environment
        env = ResearchEnvironment(
            config=self.config,
            market_data_service=self.mock_market_data_service
        )

        # Manually set data
        env.data = {
            "AAPL": {
                DataTimeframe.DAILY.value: self.test_data
            }
        }

        # Create splits
        splits = env.create_splits(
            symbol="AAPL",
            timeframe=DataTimeframe.DAILY,
            feature_columns=["open", "high", "low", "close", "volume"],
            target_column="close",
            target_transformation="returns"
        )

        # Check that splits were created
        self.assertIsInstance(splits, TimeSeriesSplit)
        self.assertEqual(len(splits.splits), 1)  # Default is train/val/test

        # Check that splits are stored
        self.assertIn("AAPL_" + DataTimeframe.DAILY.value, env.splits)

    def test_create_splits_with_cross_validation(self):
        """Test creating splits with cross-validation"""
        # Create config with cross-validation
        cv_config = ResearchConfig(
            symbols=["AAPL"],
            timeframes=[DataTimeframe.DAILY],
            start_date="2020-01-01",
            end_date="2020-12-31",
            data_split_mode=DataSplitMode.CROSS_VALIDATION,
            output_dir=self.temp_dir.name,
            random_state=42
        )

        # Create environment
        env = ResearchEnvironment(
            config=cv_config,
            market_data_service=self.mock_market_data_service
        )

        # Manually set data
        env.data = {
            "AAPL": {
                DataTimeframe.DAILY.value: self.test_data
            }
        }

        # Create splits
        splits = env.create_splits(
            symbol="AAPL",
            timeframe=DataTimeframe.DAILY
        )

        # Check that multiple splits were created
        self.assertIsInstance(splits, TimeSeriesSplit)
        self.assertTrue(len(splits.splits) > 1)

    @patch("models.research.research_environment.FeatureAnalyzer")
    def test_engineer_features(self, mock_feature_analyzer_class):
        """Test feature engineering"""
        # Set up mock
        mock_feature_analyzer = mock_feature_analyzer_class.return_value
        mock_feature_analyzer.engineer_features.return_value = pd.DataFrame({
            "feature1": np.random.random(len(self.test_data)),
            "feature2": np.random.random(len(self.test_data)),
            "feature3": np.random.random(len(self.test_data))
        }, index=self.test_data.index)

        # Create environment
        env = ResearchEnvironment(
            config=self.config,
            market_data_service=self.mock_market_data_service,
            feature_store=self.mock_feature_store
        )

        # Manually set data
        env.data = {
            "AAPL": {
                DataTimeframe.DAILY.value: self.test_data
            }
        }

        # Engineer features
        features = env.engineer_features(
            symbol="AAPL",
            timeframe=DataTimeframe.DAILY,
            feature_set_name="test_features"
        )

        # Check that features were created
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), len(self.test_data))
        self.assertEqual(len(features.columns), 3)

        # Check that features are stored
        self.assertIn("test_features", env.feature_data)

        # Check that feature store was used
        self.mock_feature_store.get_features.assert_called_once()
        self.mock_feature_store.store_features.assert_called_once()

    def test_select_features(self):
        """Test feature selection"""
        # Create environment
        env = ResearchEnvironment(
            config=self.config,
            market_data_service=self.mock_market_data_service,
            feature_analyzer=self.mock_feature_analyzer
        )

        # Create test feature data
        feature_data = pd.DataFrame({
            "feature1": np.random.random(100),
            "feature2": np.random.random(100),
            "feature3": np.random.random(100),
            "feature4": np.random.random(100),
            "feature5": np.random.random(100),
            "target": np.random.random(100)
        })

        # Set up mock behavior
        from models.research.feature_analyzer import FeatureSelectionResult
        self.mock_feature_analyzer.calculate_feature_importance.return_value = {
            "feature1": 0.5,
            "feature2": 0.3,
            "feature3": 0.2,
            "feature4": 0.1,
            "feature5": 0.05
        }
        self.mock_feature_analyzer.select_features.return_value = FeatureSelectionResult(
            selected_features=["feature1", "feature2"],
            importance={"feature1": 0.5, "feature2": 0.3},
            method="test"
        )

        # Select features
        selected_features = env.select_features(
            feature_data=feature_data,
            target="target",
            n_features=2,
            method="mutual_info"
        )

        # Check selected features
        self.assertEqual(len(selected_features), 2)
        self.assertEqual(selected_features, ["feature1", "feature2"])

        # Check that methods were called
        self.mock_feature_analyzer.calculate_feature_importance.assert_called_once()
        self.mock_feature_analyzer.select_features.assert_called_once()

    def test_train_model(self):
        """Test model training"""
        # Create environment
        env = ResearchEnvironment(
            config=self.config
        )

        # Create test data
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        
        X = pd.DataFrame({
            "feature1": np.random.random(100),
            "feature2": np.random.random(100),
            "feature3": np.random.random(100)
        })
        y = np.random.random(100)

        # Train model
        trained_model, metrics = env.train_model(
            model=model,
            feature_data=X,
            target=y,
            test_size=0.2,
            val_size=0.1
        )

        # Check trained model
        self.assertIsInstance(trained_model, RandomForestRegressor)
        
        # Check metrics
        self.assertIn("train", metrics)
        self.assertIn("validation", metrics)
        self.assertIn("test", metrics)
        self.assertIn("train_time", metrics)
        self.assertIn("data_sizes", metrics)
        self.assertIn("model_info", metrics)

        # Check metric contents
        for split in ["train", "validation", "test"]:
            self.assertIn("mse", metrics[split])
            self.assertIn("rmse", metrics[split])
            self.assertIn("mae", metrics[split])
            self.assertIn("r2", metrics[split])

    def test_calculate_metrics(self):
        """Test metric calculation"""
        # Create environment
        env = ResearchEnvironment(
            config=self.config
        )

        # Create test data
        np.random.seed(42)
        y_true = np.random.random(100)
        y_pred = y_true + np.random.normal(0, 0.1, 100)

        # Calculate metrics
        metrics = env._calculate_metrics(y_true, y_pred)

        # Check regression metrics
        self.assertIn("mse", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("r2", metrics)
        self.assertIn("spearman", metrics)

        # Check values are reasonable
        self.assertTrue(0 <= metrics["mse"] <= 1)
        self.assertTrue(0 <= metrics["rmse"] <= 1)
        self.assertTrue(0 <= metrics["mae"] <= 1)
        self.assertTrue(0 <= metrics["r2"] <= 1)
        self.assertTrue(-1 <= metrics["spearman"] <= 1)

    def test_calculate_metrics_classification(self):
        """Test metric calculation for classification"""
        # Create environment
        env = ResearchEnvironment(
            config=self.config
        )

        # Create test data for binary classification
        np.random.seed(42)
        y_true = np.random.choice([0, 1], size=100)
        y_pred = np.random.random(100)  # Continuous predictions

        # Calculate metrics
        metrics = env._calculate_metrics(y_true, y_pred)

        # Check classification metrics
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)

        # Check values are reasonable
        self.assertTrue(0 <= metrics["accuracy"] <= 1)
        self.assertTrue(0 <= metrics["precision"] <= 1)
        self.assertTrue(0 <= metrics["recall"] <= 1)
        self.assertTrue(0 <= metrics["f1"] <= 1)

    @patch("models.research.research_environment.HyperparameterOptimizer")
    def test_optimize_hyperparameters(self, mock_optimizer_class):
        """Test hyperparameter optimization"""
        from dataclasses import dataclass
        from enum import Enum
        
        # Create mock result
        class MockOptimizationMethod(Enum):
            BAYESIAN = "bayesian"
            
        @dataclass
        class MockOptimizationResult:
            best_params: dict
            best_score: float
            method: MockOptimizationMethod
            cv_scores: list
            all_results: list
            
        # Set up mock
        mock_optimizer = mock_optimizer_class.return_value
        mock_optimizer.optimize.return_value = MockOptimizationResult(
            best_params={"n_estimators": 100, "max_depth": 10},
            best_score=0.85,
            method=MockOptimizationMethod.BAYESIAN,
            cv_scores=[0.83, 0.84, 0.86, 0.85, 0.87],
            all_results=[
                {"params": {"n_estimators": 100, "max_depth": 10}, "mean_test_score": 0.85},
                {"params": {"n_estimators": 50, "max_depth": 5}, "mean_test_score": 0.80}
            ]
        )

        # Create environment
        env = ResearchEnvironment(
            config=self.config,
            hyperparameter_optimizer=mock_optimizer
        )

        # Create test data
        from sklearn.ensemble import RandomForestRegressor
        X = pd.DataFrame({
            "feature1": np.random.random(100),
            "feature2": np.random.random(100)
        })
        y = np.random.random(100)
        
        param_grid = {
            "n_estimators": [10, 50, 100],
            "max_depth": [5, 10, None]
        }

        # Optimize hyperparameters
        best_params, results = env.optimize_hyperparameters(
            model_class=RandomForestRegressor,
            feature_data=X,
            target=y,
            param_grid=param_grid
        )

        # Check results
        self.assertEqual(best_params, {"n_estimators": 100, "max_depth": 10})
        self.assertEqual(results["best_score"], 0.85)
        self.assertEqual(results["method"], "bayesian")
        self.assertEqual(len(results["all_results"]), 2)

    def test_run_backtest(self):
        """Test running a backtest"""
        # Create environment
        env = ResearchEnvironment(
            config=self.config
        )

        # Create mock model
        class MockModel:
            def predict(self, X):
                # Simple momentum strategy: predict based on last 5 days
                return np.sign(X.mean(axis=1))

        model = MockModel()

        # Create test data
        # Use more realistic price data
        dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
        np.random.seed(42)
        close_prices = 100 * (1 + np.cumsum(np.random.normal(0.0005, 0.01, len(dates))))
        
        data = pd.DataFrame({
            "open": close_prices * (1 + np.random.normal(0, 0.005, len(dates))),
            "high": close_prices * (1 + np.random.normal(0.005, 0.005, len(dates))),
            "low": close_prices * (1 - np.random.normal(0.005, 0.005, len(dates))),
            "close": close_prices,
            "volume": np.random.randint(1000, 10000, len(dates)),
            "sma5": pd.Series(close_prices).rolling(5).mean().values,
            "sma10": pd.Series(close_prices).rolling(10).mean().values,
            "sma20": pd.Series(close_prices).rolling(20).mean().values,
            "target": pd.Series(close_prices).pct_change().values
        }, index=dates)
        
        # Fill NaN values
        data = data.fillna(method='bfill')

        # Run backtest
        backtest_results = env.run_backtest(
            model=model,
            data=data,
            features=["sma5", "sma10", "sma20"],
            target="target",
            initial_capital=10000.0,
            transaction_costs=0.001
        )

        # Check results
        self.assertIn("data", backtest_results)
        self.assertIn("trade_stats", backtest_results)
        self.assertIn("performance_metrics", backtest_results)
        self.assertIn("rolling_metrics", backtest_results)
        self.assertIn("initial_capital", backtest_results)
        self.assertIn("final_equity", backtest_results)
        self.assertIn("total_return", backtest_results)
        self.assertIn("max_drawdown", backtest_results)

        # Check backtest data columns
        backtest_data = backtest_results["data"]
        self.assertIn("prediction", backtest_data)
        self.assertIn("signal", backtest_data)
        self.assertIn("position", backtest_data)
        self.assertIn("costs", backtest_data)
        self.assertIn("market_return", backtest_data)
        self.assertIn("strategy_return", backtest_data)
        self.assertIn("market_equity", backtest_data)
        self.assertIn("strategy_equity", backtest_data)
        self.assertIn("market_dd", backtest_data)
        self.assertIn("strategy_dd", backtest_data)

        # Check trade stats
        trade_stats = backtest_results["trade_stats"]
        self.assertIn("n_trades", trade_stats)
        self.assertIn("win_rate", trade_stats)

        # Check performance metrics
        performance_metrics = backtest_results["performance_metrics"]
        self.assertIn("total_return", performance_metrics)
        self.assertIn("market_return", performance_metrics)
        self.assertIn("excess_return", performance_metrics)
        self.assertIn("sharpe_ratio", performance_metrics)
        self.assertIn("max_drawdown", performance_metrics)

    def test_calculate_performance_metrics(self):
        """Test calculation of performance metrics"""
        # Create environment
        env = ResearchEnvironment(
            config=self.config
        )

        # Create test backtest data
        dates = pd.date_range(start="2020-01-01", periods=252, freq="D")
        
        # Create realistic equity curves
        np.random.seed(42)
        market_returns = np.random.normal(0.0005, 0.01, len(dates))
        strategy_returns = market_returns + np.random.normal(0.0002, 0.005, len(dates))
        
        market_equity = 10000 * np.cumprod(1 + market_returns)
        strategy_equity = 10000 * np.cumprod(1 + strategy_returns)
        
        # Calculate drawdowns
        market_dd = 1 - market_equity / np.maximum.accumulate(market_equity)
        strategy_dd = 1 - strategy_equity / np.maximum.accumulate(strategy_equity)
        
        backtest_data = pd.DataFrame({
            "market_return": market_returns,
            "strategy_return": strategy_returns,
            "market_equity": market_equity,
            "strategy_equity": strategy_equity,
            "market_dd": market_dd,
            "strategy_dd": strategy_dd
        }, index=dates)

        # Calculate metrics
        metrics = env._calculate_performance_metrics(backtest_data)

        # Check metrics
        self.assertIn("total_return", metrics)
        self.assertIn("market_return", metrics)
        self.assertIn("excess_return", metrics)
        self.assertIn("cagr", metrics)
        self.assertIn("volatility", metrics)
        self.assertIn("max_drawdown", metrics)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("sortino_ratio", metrics)
        self.assertIn("calmar_ratio", metrics)
        self.assertIn("win_rate", metrics)
        self.assertIn("profit_factor", metrics)

        # Check values
        self.assertTrue(metrics["total_return"] > 0)
        self.assertTrue(metrics["sharpe_ratio"] > 0)
        self.assertTrue(0 <= metrics["win_rate"] <= 1)
        self.assertTrue(metrics["max_drawdown"] >= 0)

    def test_calculate_rolling_metrics(self):
        """Test calculation of rolling metrics"""
        # Create environment
        env = ResearchEnvironment(
            config=self.config
        )

        # Create test data
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        np.random.seed(42)
        
        # Create synthetic returns
        strategy_returns = np.random.normal(0.001, 0.01, len(dates))
        
        # Create DataFrame
        backtest_data = pd.DataFrame({
            "strategy_return": strategy_returns
        }, index=dates)

        # Calculate rolling metrics with window=20
        rolling_metrics = env._calculate_rolling_metrics(backtest_data, window=20)

        # Check keys
        self.assertIn("rolling_return", rolling_metrics)
        self.assertIn("rolling_volatility", rolling_metrics)
        self.assertIn("rolling_sharpe", rolling_metrics)
        self.assertIn("rolling_max_drawdown", rolling_metrics)
        self.assertIn("rolling_win_rate", rolling_metrics)
        self.assertIn("rolling_profit_factor", rolling_metrics)

        # Check sizes (should be same as input with NaNs at beginning)
        self.assertEqual(len(rolling_metrics["rolling_return"]), len(backtest_data))
        
        # First values should be NaN because of window
        self.assertTrue(np.isnan(rolling_metrics["rolling_return"].iloc[0]))
        
        # Values after window should be valid
        self.assertFalse(np.isnan(rolling_metrics["rolling_return"].iloc[20]))

    def test_calculate_trade_stats(self):
        """Test calculation of trade statistics"""
        # Create environment
        env = ResearchEnvironment(
            config=self.config
        )

        # Create test data
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        np.random.seed(42)
        
        # Create positions and position changes
        positions = np.zeros(len(dates))
        positions[10:30] = 1  # Long position
        positions[40:60] = -1  # Short position
        positions[70:90] = 1  # Long position
        
        position_changes = np.zeros(len(dates))
        position_changes[10] = 1  # Enter long
        position_changes[30] = -1  # Exit long
        position_changes[40] = -1  # Enter short
        position_changes[60] = 1  # Exit short
        position_changes[70] = 1  # Enter long
        position_changes[90] = -1  # Exit long
        
        # Create returns and equity
        base_returns = np.random.normal(0.001, 0.01, len(dates))
        strategy_returns = base_returns * positions[:-1]  # Using previous position for returns
        strategy_returns = np.append([0], strategy_returns)  # Add initial zero
        
        strategy_equity = 10000 * np.cumprod(1 + strategy_returns)
        
        # Create backtest data
        backtest_data = pd.DataFrame({
            "position": positions,
            "position_change": position_changes,
            "strategy_return": strategy_returns,
            "strategy_equity": strategy_equity
        }, index=dates)

        # Set up global variable required by the function
        global initial_equity
        initial_equity = 10000

        # Calculate trade stats
        trade_stats = env._calculate_trade_stats(backtest_data)

        # Check keys
        self.assertIn("n_trades", trade_stats)
        self.assertIn("win_rate", trade_stats)
        self.assertIn("profit_factor", trade_stats)
        self.assertIn("avg_profit_per_trade", trade_stats)
        self.assertIn("avg_trade_length", trade_stats)
        self.assertIn("avg_win", trade_stats)
        self.assertIn("avg_loss", trade_stats)
        self.assertIn("max_win", trade_stats)
        self.assertIn("max_loss", trade_stats)
        self.assertIn("consecutive_wins", trade_stats)
        self.assertIn("consecutive_losses", trade_stats)

        # Check number of trades
        self.assertEqual(trade_stats["n_trades"], 6)  # 3 entries and 3 exits


if __name__ == "__main__":
    unittest.main()