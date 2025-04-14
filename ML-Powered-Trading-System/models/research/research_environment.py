"""
research_environment.py - Isolated Backtesting Environment

This module provides an isolated environment for research and backtesting
of trading strategies, with proper time-series cross-validation, and
protection against lookahead bias.
"""

import numpy as np
import pandas as pd
import logging
import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
import pickle
import hashlib
import uuid
from pathlib import Path
import multiprocessing
from scipy import stats

# Internal imports
from data.market_data_service import MarketDataService, DataTimeframe, DataSource
from data.feature_store import FeatureStore
from models.research.feature_analyzer import FeatureAnalyzer
from models.research.cross_validation import TimeSeriesCV, CVMethod
from models.research.model_validator import ModelValidator
from models.research.hyperparameter_optimization import HyperparameterOptimizer, OptimizationMethod

logger = logging.getLogger(__name__)


class DataSplitMode(Enum):
    """Methods for splitting data in research environment"""
    TRAIN_TEST = "train_test"                # Simple train/test split
    TRAIN_VALIDATION_TEST = "train_val_test" # Train, validation, and test split
    CROSS_VALIDATION = "cross_validation"    # Cross-validation splits
    WALK_FORWARD = "walk_forward"            # Walk-forward validation
    EXPANDING_WINDOW = "expanding_window"    # Expanding window validation
    ANCHORED = "anchored"                    # Anchored validation


class BacktestMode(Enum):
    """Modes for backtesting trading strategies"""
    VECTORIZED = "vectorized"      # Vectorized backtesting (faster)
    EVENT_DRIVEN = "event_driven"  # Event-driven backtesting (more realistic)
    TICK_BY_TICK = "tick_by_tick"  # Tick-by-tick backtesting (most realistic)


@dataclass
class ResearchConfig:
    """Configuration for research environment"""
    # Data configuration
    symbols: List[str]
    timeframes: List[Union[DataTimeframe, str]]
    start_date: Union[str, datetime]
    end_date: Union[str, datetime]
    data_source: DataSource = DataSource.AUTO

    # Data splitting configuration
    data_split_mode: DataSplitMode = DataSplitMode.TRAIN_VALIDATION_TEST
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15

    # Backtesting configuration
    backtest_mode: BacktestMode = BacktestMode.VECTORIZED
    initial_capital: float = 100000.0

    # Environment configuration
    output_dir: str = "data/research"
    random_state: Optional[int] = None

    # Feature configuration
    use_feature_store: bool = True
    auto_feature_engineering: bool = True

    # Metadata
    name: str = "research_environment"
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class ResearchResult:
    """Results from a research experiment"""
    config: ResearchConfig
    performance_metrics: Dict[str, Any]
    model_metrics: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]] = None
    model_path: Optional[str] = None
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)


class TimeSeriesSplit:
    """Time series data split with proper train, validation, and test sets"""

    def __init__(self,
                 data: pd.DataFrame,
                 mode: DataSplitMode,
                 train_ratio: float = 0.7,
                 validation_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 n_splits: int = 5,
                 gap: int = 0):
        """
        Initialize time series data split.

        Args:
            data: Data to split
            mode: Data split mode
            train_ratio: Ratio of data for training
            validation_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
            n_splits: Number of splits for cross-validation modes
            gap: Gap between train and validation/test sets
        """
        self.data = data
        self.mode = mode
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.n_splits = n_splits
        self.gap = gap

        # Total rows
        self.total_rows = len(data)

        # Create splits based on mode
        if mode == DataSplitMode.TRAIN_TEST:
            self.splits = self._create_train_test_split()
        elif mode == DataSplitMode.TRAIN_VALIDATION_TEST:
            self.splits = self._create_train_validation_test_split()
        elif mode in [DataSplitMode.CROSS_VALIDATION, DataSplitMode.WALK_FORWARD,
                      DataSplitMode.EXPANDING_WINDOW, DataSplitMode.ANCHORED]:
            self.splits = self._create_cv_splits()
        else:
            raise ValueError(f"Unsupported split mode: {mode}")

    def _create_train_test_split(self) -> List[Dict[str, pd.DataFrame]]:
        """Create simple train/test split"""
        split_idx = int(self.total_rows * self.train_ratio)

        train_data = self.data.iloc[:split_idx]
        test_data = self.data.iloc[split_idx + self.gap:]

        return [{
            "train": train_data,
            "test": test_data
        }]

    def _create_train_validation_test_split(self) -> List[Dict[str, pd.DataFrame]]:
        """Create train/validation/test split"""
        train_idx = int(self.total_rows * self.train_ratio)
        val_idx = train_idx + int(self.total_rows * self.validation_ratio)

        train_data = self.data.iloc[:train_idx]
        val_data = self.data.iloc[train_idx + self.gap:val_idx]
        test_data = self.data.iloc[val_idx + self.gap:]

        return [{
            "train": train_data,
            "validation": val_data,
            "test": test_data
        }]

    def _create_cv_splits(self) -> List[Dict[str, pd.DataFrame]]:
        """Create cross-validation splits"""
        # Map data split mode to CV method
        cv_method_map = {
            DataSplitMode.CROSS_VALIDATION: CVMethod.PURGED,
            DataSplitMode.WALK_FORWARD: CVMethod.ROLLING_ORIGIN,
            DataSplitMode.EXPANDING_WINDOW: CVMethod.EXPANDING_WINDOW,
            DataSplitMode.ANCHORED: CVMethod.ANCHORED
        }

        cv_method = cv_method_map[self.mode]

        # Create CV splitter
        cv = TimeSeriesCV(
            method=cv_method,
            n_splits=self.n_splits,
            train_pct=self.train_ratio,
            gap=self.gap,
            allow_overlap=True
        )

        # Get CV splits
        cv_splits = cv.split(self.data)

        # Convert to dictionary format
        result = []
        for i, split in enumerate(cv_splits):
            split_data = {
                "train": self.data.loc[self.data.index[
                    (self.data.index >= split.train_start) &
                    (self.data.index <= split.train_end)
                ]],
                "test": self.data.loc[self.data.index[
                    (self.data.index >= split.test_start) &
                    (self.data.index <= split.test_end)
                ]]
            }

            # Add validation set if available
            if self.mode == DataSplitMode.WALK_FORWARD and i < len(cv_splits) - 1:
                # Use next test set as validation
                next_split = cv_splits[i+1]
                split_data["validation"] = self.data.loc[self.data.index[
                    (self.data.index >= next_split.test_start) &
                    (self.data.index <= next_split.test_end)
                ]]

            result.append(split_data)

        return result

    def get_split(self, index: int = 0) -> Dict[str, pd.DataFrame]:
        """
        Get a specific split.

        Args:
            index: Split index

        Returns:
            Dictionary with data splits
        """
        if index < 0 or index >= len(self.splits):
            raise ValueError(f"Split index {index} out of range (0-{len(self.splits)-1})")

        return self.splits[index]

    def get_all_splits(self) -> List[Dict[str, pd.DataFrame]]:
        """
        Get all splits.

        Returns:
            List of data splits
        """
        return self.splits


class ResearchEnvironment:
    """
    Isolated environment for research and backtesting with protection
    against data leakage and lookahead bias.
    """

    def __init__(self,
                 config: ResearchConfig,
                 market_data_service: Optional[MarketDataService] = None,
                 feature_store: Optional[FeatureStore] = None,
                 feature_analyzer: Optional[FeatureAnalyzer] = None,
                 model_validator: Optional[ModelValidator] = None,
                 hyperparameter_optimizer: Optional[HyperparameterOptimizer] = None):
        """
        Initialize research environment.

        Args:
            config: Research configuration
            market_data_service: Optional market data service
            feature_store: Optional feature store
            feature_analyzer: Optional feature analyzer
            model_validator: Optional model validator
            hyperparameter_optimizer: Optional hyperparameter optimizer
        """
        self.config = config
        self.market_data_service = market_data_service
        self.feature_store = feature_store if config.use_feature_store else None
        self.feature_analyzer = feature_analyzer or FeatureAnalyzer(
            feature_store=self.feature_store,
            output_dir=os.path.join(config.output_dir, "feature_analysis"),
            random_state=config.random_state
        )
        self.model_validator = model_validator or ModelValidator()
        self.hyperparameter_optimizer = hyperparameter_optimizer

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)

        # Set random seed
        if config.random_state is not None:
            np.random.seed(config.random_state)

        # Initialize data storage
        self.data = {}
        self.feature_data = {}
        self.splits = {}
        self.models = {}
        self.results = {}

        # Load data if market data service is available
        if self.market_data_service:
            self._load_data()

        logger.info(f"Research environment '{config.name}' initialized")

    def _load_data(self):
        """Load data based on configuration"""
        # Convert timeframes to strings if needed
        timeframes = [
            tf.value if isinstance(tf, DataTimeframe) else tf
            for tf in self.config.timeframes
        ]

        # Load data for each symbol and timeframe
        for symbol in self.config.symbols:
            self.data[symbol] = {}

            for timeframe in timeframes:
                try:
                    # Load data from market data service
                    data = self.market_data_service.get_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=self.config.start_date,
                        end_date=self.config.end_date,
                        source=self.config.data_source
                    )

                    if data is not None and not data.empty:
                        self.data[symbol][timeframe] = data
                        logger.info(f"Loaded {len(data)} rows for {symbol}/{timeframe}")
                    else:
                        logger.warning(f"No data available for {symbol}/{timeframe}")

                except Exception as e:
                    logger.error(f"Error loading data for {symbol}/{timeframe}: {str(e)}")

        # Log summary
        total_symbols = sum(1 for s in self.data if self.data[s])
        total_timeframes = sum(len(self.data[s]) for s in self.data)
        logger.info(f"Loaded data for {total_symbols} symbols and {total_timeframes} symbol-timeframe combinations")

    def create_splits(self,
                     symbol: str,
                     timeframe: Union[str, DataTimeframe],
                     feature_columns: Optional[List[str]] = None,
                     target_column: str = 'close',
                     target_transformation: Optional[str] = 'returns') -> TimeSeriesSplit:
        """
        Create data splits for a symbol and timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            feature_columns: Optional list of feature columns to include
            target_column: Target column for prediction
            target_transformation: Optional transformation for target

        Returns:
            TimeSeriesSplit object
        """
        # Convert timeframe to string if needed
        tf = timeframe.value if isinstance(timeframe, DataTimeframe) else timeframe

        # Check if data is available
        if symbol not in self.data or tf not in self.data[symbol]:
            raise ValueError(f"No data available for {symbol}/{tf}")

        # Get data
        data = self.data[symbol][tf].copy()

        # Create target based on transformation
        if target_transformation:
            if target_transformation == 'returns':
                data['target'] = data[target_column].pct_change()
            elif target_transformation == 'log_returns':
                data['target'] = np.log(data[target_column] / data[target_column].shift(1))
            elif target_transformation == 'direction':
                data['target'] = np.sign(data[target_column].pct_change())
            else:
                raise ValueError(f"Unsupported target transformation: {target_transformation}")
        else:
            data['target'] = data[target_column]

        # Drop NaN values
        data = data.dropna()

        # Filter to feature columns if specified
        if feature_columns:
            all_columns = feature_columns + ['target']
            data = data[all_columns]

        # Create splits
        splits = TimeSeriesSplit(
            data=data,
            mode=self.config.data_split_mode,
            train_ratio=self.config.train_ratio,
            validation_ratio=self.config.validation_ratio,
            test_ratio=self.config.test_ratio,
            n_splits=5,
            gap=0
        )

        # Store splits
        split_key = f"{symbol}_{tf}"
        self.splits[split_key] = splits

        logger.info(f"Created {len(splits.splits)} splits for {symbol}/{tf}")

        return splits

    def engineer_features(self,
                         symbol: str,
                         timeframe: Union[str, DataTimeframe],
                         feature_set_name: Optional[str] = None) -> pd.DataFrame:
        """
        Engineer features for a symbol and timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            feature_set_name: Optional name for the feature set

        Returns:
            DataFrame with engineered features
        """
        # Convert timeframe to string if needed
        tf = timeframe.value if isinstance(timeframe, DataTimeframe) else timeframe

        # Check if data is available
        if symbol not in self.data or tf not in self.data[symbol]:
            raise ValueError(f"No data available for {symbol}/{tf}")

        # Get data
        data = self.data[symbol][tf].copy()

        # Set default feature set name if not provided
        if feature_set_name is None:
            feature_set_name = f"{symbol}_{tf}_features"

        # Check if features are already in feature store
        if self.feature_store and self.config.use_feature_store:
            features = self.feature_store.get_features(
                symbol=symbol,
                timeframe=tf,
                feature_set=feature_set_name
            )

            if features is not None:
                logger.info(f"Loaded {len(features.columns)} features from feature store for {symbol}/{tf}")
                self.feature_data[feature_set_name] = features
                return features

        # Engineer features
        features = self.feature_analyzer.engineer_features(
            data=data,
            time_column=data.index.name or 'index',
            temporal_features=True,
            interaction_terms=True
        )

        # Store features
        self.feature_data[feature_set_name] = features

        # Save to feature store if available
        if self.feature_store and self.config.use_feature_store:
            self.feature_store.store_features(
                data=features,
                symbol=symbol,
                timeframe=tf,
                feature_set=feature_set_name
            )

            # Save feature metadata
            feature_summaries = self.feature_analyzer.analyze_features(features)
            feature_metadata = self.feature_analyzer.generate_feature_report(feature_summaries)
            self.feature_analyzer.save_feature_metadata(feature_metadata, feature_set_name)

        logger.info(f"Engineered {len(features.columns)} features for {symbol}/{tf}")

        return features

    def select_features(self,
                       feature_data: pd.DataFrame,
                       target: Union[str, pd.Series],
                       n_features: int = 20,
                       method: str = 'mutual_info') -> List[str]:
        """
        Select top features for modeling.

        Args:
            feature_data: DataFrame with features
            target: Target column name or Series
            n_features: Number of features to select
            method: Feature selection method

        Returns:
            List of selected feature names
        """
        from models.research.feature_analyzer import FeatureImportanceMethod, FeatureSelectionMethod

        # Get target data
        if isinstance(target, str):
            if target in feature_data.columns:
                target_data = feature_data[target]
            else:
                raise ValueError(f"Target column '{target}' not found in feature data")
        else:
            target_data = target

        # Remove target from features
        if isinstance(target, str) and target in feature_data.columns:
            feature_data = feature_data.drop(columns=[target])

        # Map method string to enum
        if method == 'mutual_info':
            importance_method = FeatureImportanceMethod.MUTUAL_INFO
        elif method == 'correlation':
            importance_method = FeatureImportanceMethod.CORRELATION
        elif method == 'variance':
            importance_method = FeatureImportanceMethod.VARIANCE
        else:
            importance_method = FeatureImportanceMethod.MUTUAL_INFO

        # Create a dummy model for importance calculation if needed
        dummy_model = None
        if importance_method in [FeatureImportanceMethod.MODEL_SPECIFIC, FeatureImportanceMethod.PERMUTATION]:
            from sklearn.ensemble import RandomForestRegressor
            dummy_model = RandomForestRegressor(n_estimators=10, random_state=self.config.random_state)
            dummy_model.fit(feature_data, target_data)

        # Calculate feature importance
        importance_result = self.feature_analyzer.calculate_feature_importance(
            X=feature_data,
            y=target_data,
            model=dummy_model,
            method=importance_method
        )

        # Select top features
        selection_result = self.feature_analyzer.select_features(
            importance_result=importance_result,
            method=FeatureSelectionMethod.TOP_K,
            k=n_features
        )

        logger.info(f"Selected {len(selection_result.selected_features)} features using {method}")

        return selection_result.selected_features

    def train_model(self,
                   model: Any,
                   feature_data: pd.DataFrame,
                   target: Union[str, pd.Series],
                   test_size: float = 0.2,
                   val_size: float = 0.0,
                   random_state: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Train a model with proper train/test splitting.

        Args:
            model: Model to train
            feature_data: DataFrame with features
            target: Target column name or Series
            test_size: Fraction of data to use for testing
            val_size: Fraction of data to use for validation
            random_state: Random state for reproducibility

        Returns:
            Tuple of (trained model, metrics)
        """
        from sklearn.model_selection import train_test_split

        # Set default random state if not provided
        if random_state is None:
            random_state = self.config.random_state

        # Get target data
        if isinstance(target, str):
            if target in feature_data.columns:
                target_data = feature_data[target]
                feature_data = feature_data.drop(columns=[target])
            else:
                raise ValueError(f"Target column '{target}' not found in feature data")
        else:
            target_data = target

        # Train/test split
        if val_size > 0:
            # Three-way split
            X_train, X_temp, y_train, y_temp = train_test_split(
                feature_data, target_data, test_size=test_size + val_size, random_state=random_state
            )

            # Split temp into validation and test
            val_ratio = val_size / (test_size + val_size)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=1-val_ratio, random_state=random_state
            )
        else:
            # Two-way split
            X_train, X_test, y_train, y_test = train_test_split(
                feature_data, target_data, test_size=test_size, random_state=random_state
            )
            X_val, y_val = None, None

        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Evaluate model
        metrics = {}

        # Training metrics
        y_train_pred = model.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, y_train_pred)
        metrics["train"] = train_metrics

        # Validation metrics
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, y_val_pred)
            metrics["validation"] = val_metrics

        # Test metrics
        y_test_pred = model.predict(X_test)
        test_metrics = self._calculate_metrics(y_test, y_test_pred)
        metrics["test"] = test_metrics

        # Add training time
        metrics["train_time"] = train_time

        # Add data sizes
        metrics["data_sizes"] = {
            "train": len(X_train),
            "validation": len(X_val) if X_val is not None else 0,
            "test": len(X_test)
        }

        # Add model info
        metrics["model_info"] = {
            "type": type(model).__name__
        }

        # Add feature importance if available
        if hasattr(model, "feature_importances_"):
            feature_names = feature_data.columns
            feature_importances = dict(zip(feature_names, model.feature_importances_))
            metrics["feature_importances"] = feature_importances

        logger.info(f"Trained {type(model).__name__} model with "
                   f"train score: {train_metrics.get('r2', 0):.4f}, "
                   f"test score: {test_metrics.get('r2', 0):.4f}")

        return model, metrics

    def _calculate_metrics(self, y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        Calculate performance metrics for model evaluation.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        # Convert to numpy arrays
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)

        metrics = {}

        # Determine task type (regression or classification)
        unique_values = np.unique(y_true_np)
        if len(unique_values) <= 5 and np.all(np.isin(unique_values, [-1, 0, 1])):
            # Classification metrics for binary/ternary classification
            metrics["accuracy"] = np.mean(y_true_np == np.sign(y_pred_np))

            # For binary classification (if applicable)
            if len(unique_values) <= 2:
                # Only calculate precision/recall/f1 if we have positive samples
                if np.any(y_true_np > 0):
                    # Convert to binary
                    y_true_binary = y_true_np > 0
                    y_pred_binary = y_pred_np > 0

                    # Calculate metrics
                    tp = np.sum((y_true_binary) & (y_pred_binary))
                    fp = np.sum((~y_true_binary) & (y_pred_binary))
                    fn = np.sum((y_true_binary) & (~y_pred_binary))

                    metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
                    metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0
                    metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"]) if (metrics["precision"] + metrics["recall"]) > 0 else 0

        # Regression metrics (calculate these regardless of task type)
        metrics["mse"] = np.mean((y_true_np - y_pred_np) ** 2)
        metrics["rmse"] = np.sqrt(metrics["mse"])
        metrics["mae"] = np.mean(np.abs(y_true_np - y_pred_np))

        # Calculate r-squared
        if np.var(y_true_np) > 0:
            metrics["r2"] = 1 - np.var(y_true_np - y_pred_np) / np.var(y_true_np)
        else:
            metrics["r2"] = 0

        # Calculate spearman correlation
        try:
            metrics["spearman"] = stats.spearmanr(y_true_np, y_pred_np)[0]
        except:
            metrics["spearman"] = 0

        # Trading-specific metrics
        if len(y_true_np) > 1:
            # Calculate directional accuracy
            if np.any(np.diff(y_true_np) != 0):  # Only if there are actual changes
                true_direction = np.sign(np.diff(y_true_np))
                pred_direction = np.sign(np.diff(y_pred_np))
                metrics["direction_accuracy"] = np.mean(true_direction == pred_direction)

            # Calculate profitability (assuming y_true is returns and y_pred is signals)
            strategy_returns = y_true_np[1:] * np.sign(y_pred_np[:-1])
            if len(strategy_returns) > 0:
                metrics["win_rate"] = np.mean(strategy_returns > 0) if len(strategy_returns) > 0 else 0

                # Calculate Sharpe ratio (annualized assuming daily data)
                if np.std(strategy_returns) > 0:
                    metrics["sharpe_ratio"] = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
                else:
                    metrics["sharpe_ratio"] = 0

        return metrics

    def _calculate_trade_stats(self, backtest_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate trading statistics from backtest data.

        Args:
            backtest_data: DataFrame with backtest results

        Returns:
            Dictionary of trade statistics
        """
        # Extract relevant data
        position_changes = backtest_data['position_change']
        strategy_returns = backtest_data['strategy_return']
        positions = backtest_data['position']

        # Identify trades (entry and exit points)
        trade_entries = position_changes != 0

        # Count trades
        n_trades = trade_entries.sum()

        # Calculate trade stats
        stats = {
            "n_trades": int(n_trades),
            "avg_trade_length": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "avg_profit_per_trade": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "max_win": 0,
            "max_loss": 0,
            "consecutive_wins": 0,
            "consecutive_losses": 0
        }

        if n_trades > 0:
            # Get entry and exit indices
            entry_indices = backtest_data.index[trade_entries]

            # Calculate trade lengths
            if len(entry_indices) > 1:
                trade_lengths = []
                for i in range(len(entry_indices) - 1):
                    if position_changes.loc[entry_indices[i]] != 0 and position_changes.loc[entry_indices[i+1]] != 0:
                        trade_lengths.append((entry_indices[i+1] - entry_indices[i]).days)

                if trade_lengths:
                    stats["avg_trade_length"] = np.mean(trade_lengths)

            # Calculate trade returns
            trade_returns = []
            current_position = 0
            trade_start_equity = initial_equity = backtest_data['strategy_equity'].iloc[0]

            for i, row in backtest_data.iterrows():
                if row['position_change'] != 0:
                    # If we're changing position, calculate the return from the previous trade
                    if current_position != 0:
                        trade_end_equity = row['strategy_equity']
                        trade_return = (trade_end_equity / trade_start_equity) - 1
                        trade_returns.append(trade_return)

                    # Update for new trade
                    current_position = row['position']
                    trade_start_equity = row['strategy_equity']

            # Add the final trade if there's an open position
            if current_position != 0 and len(backtest_data) > 0:
                trade_end_equity = backtest_data['strategy_equity'].iloc[-1]
                trade_return = (trade_end_equity / trade_start_equity) - 1
                trade_returns.append(trade_return)

            # Calculate trade statistics
            if trade_returns:
                winning_trades = [r for r in trade_returns if r > 0]
                losing_trades = [r for r in trade_returns if r <= 0]

                stats["win_rate"] = len(winning_trades) / len(trade_returns) if trade_returns else 0
                stats["avg_profit_per_trade"] = np.mean(trade_returns) if trade_returns else 0
                stats["avg_win"] = np.mean(winning_trades) if winning_trades else 0
                stats["avg_loss"] = np.mean(losing_trades) if losing_trades else 0
                stats["max_win"] = max(winning_trades) if winning_trades else 0
                stats["max_loss"] = min(losing_trades) if losing_trades else 0

                # Calculate profit factor
                gross_profits = sum(r for r in trade_returns if r > 0)
                gross_losses = abs(sum(r for r in trade_returns if r < 0))
                stats["profit_factor"] = gross_profits / gross_losses if gross_losses > 0 else float('inf')

                # Calculate max consecutive wins/losses
                win_loss_sequence = [1 if r > 0 else -1 for r in trade_returns]

                current_streak = 0
                max_win_streak = 0
                max_loss_streak = 0

                for wl in win_loss_sequence:
                    if wl > 0:
                        # Win
                        if current_streak > 0:
                            current_streak += 1
                        else:
                            current_streak = 1
                        max_win_streak = max(max_win_streak, current_streak)
                    else:
                        # Loss
                        if current_streak < 0:
                            current_streak -= 1
                        else:
                            current_streak = -1
                        max_loss_streak = min(max_loss_streak, current_streak)

                stats["consecutive_wins"] = max_win_streak
                stats["consecutive_losses"] = abs(max_loss_streak)

        return stats

    def _calculate_performance_metrics(self, backtest_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics for a backtest.

        Args:
            backtest_data: DataFrame with backtest results

        Returns:
            Dictionary of performance metrics
        """
        # Extract relevant data
        market_returns = backtest_data['market_return']
        strategy_returns = backtest_data['strategy_return']
        market_equity = backtest_data['market_equity']
        strategy_equity = backtest_data['strategy_equity']
        market_dd = backtest_data['market_dd']
        strategy_dd = backtest_data['strategy_dd']

        # Create metrics dictionary
        metrics = {}

        # Calculate returns
        metrics["total_return"] = (strategy_equity.iloc[-1] / strategy_equity.iloc[0]) - 1
        metrics["market_return"] = (market_equity.iloc[-1] / market_equity.iloc[0]) - 1
        metrics["excess_return"] = metrics["total_return"] - metrics["market_return"]

        # Calculate annualized returns (assuming daily data)
        n_days = (backtest_data.index[-1] - backtest_data.index[0]).days
        n_years = n_days / 365.25

        if n_years > 0:
            metrics["cagr"] = (strategy_equity.iloc[-1] / strategy_equity.iloc[0]) ** (1 / n_years) - 1
            metrics["market_cagr"] = (market_equity.iloc[-1] / market_equity.iloc[0]) ** (1 / n_years) - 1
        else:
            metrics["cagr"] = metrics["total_return"]
            metrics["market_cagr"] = metrics["market_return"]

        # Calculate volatility
        metrics["volatility"] = strategy_returns.std() * np.sqrt(252)  # Annualized
        metrics["market_volatility"] = market_returns.std() * np.sqrt(252)

        # Calculate drawdowns
        metrics["max_drawdown"] = strategy_dd.max()
        metrics["market_max_drawdown"] = market_dd.max()

        # Calculate risk metrics
        if metrics["volatility"] > 0:
            metrics["sharpe_ratio"] = metrics["cagr"] / metrics["volatility"]
        else:
            metrics["sharpe_ratio"] = 0

        # Calculate Sortino ratio (downside risk)
        downside_returns = strategy_returns[strategy_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            metrics["sortino_ratio"] = metrics["cagr"] / (downside_returns.std() * np.sqrt(252))
        else:
            metrics["sortino_ratio"] = float('inf') if metrics["cagr"] > 0 else 0

        # Calculate Calmar ratio
        if metrics["max_drawdown"] > 0:
            metrics["calmar_ratio"] = metrics["cagr"] / metrics["max_drawdown"]
        else:
            metrics["calmar_ratio"] = float('inf') if metrics["cagr"] > 0 else 0

        # Calculate win rate
        metrics["win_rate"] = (strategy_returns > 0).mean()

        # Calculate profit factor
        gross_profits = strategy_returns[strategy_returns > 0].sum()
        gross_losses = abs(strategy_returns[strategy_returns < 0].sum())

        if gross_losses > 0:
            metrics["profit_factor"] = gross_profits / gross_losses
        else:
            metrics["profit_factor"] = float('inf') if gross_profits > 0 else 0

        # Calculate recovery time
        if metrics["max_drawdown"] > 0:
            max_dd_idx = strategy_dd.idxmax()
            recovery_idx = None

            for i in range(backtest_data.index.get_loc(max_dd_idx) + 1, len(backtest_data)):
                if strategy_equity.iloc[i] >= strategy_equity.iloc[:max_dd_idx].max():
                    recovery_idx = backtest_data.index[i]
                    break

            if recovery_idx:
                metrics["recovery_days"] = (recovery_idx - max_dd_idx).days
            else:
                metrics["recovery_days"] = float('inf')  # No recovery yet
        else:
            metrics["recovery_days"] = 0

        return metrics

    def _calculate_rolling_metrics(self, backtest_data: pd.DataFrame, window: int = 63) -> Dict[str, pd.Series]:
        """
        Calculate rolling metrics for backtest analysis.

        Args:
            backtest_data: DataFrame with backtest results
            window: Rolling window size (default: 63 days = 3 months)

        Returns:
            Dictionary of rolling metrics
        """
        # Extract relevant data
        strategy_returns = backtest_data['strategy_return']

        # Create metrics dictionary
        rolling_metrics = {}

        # Calculate rolling returns
        rolling_metrics["rolling_return"] = strategy_returns.rolling(window).sum()

        # Calculate rolling volatility
        rolling_metrics["rolling_volatility"] = strategy_returns.rolling(window).std() * np.sqrt(252)

        # Calculate rolling Sharpe ratio
        rolling_metrics["rolling_sharpe"] = (
            strategy_returns.rolling(window).mean() /
            strategy_returns.rolling(window).std()
        ) * np.sqrt(252)

        # Calculate rolling maximum drawdown
        def rolling_max_drawdown(returns):
            equity_curve = (1 + returns).cumprod()
            running_max = equity_curve.cummax()
            drawdown = (equity_curve / running_max) - 1
            return drawdown.min()

        rolling_metrics["rolling_max_drawdown"] = strategy_returns.rolling(window).apply(
            rolling_max_drawdown, raw=False
        )

        # Calculate rolling win rate
        rolling_metrics["rolling_win_rate"] = strategy_returns.rolling(window).apply(
            lambda x: (x > 0).mean()
        )

        # Calculate rolling profit factor
        def profit_factor(returns):
            gross_profits = returns[returns > 0].sum()
            gross_losses = abs(returns[returns < 0].sum())
            return gross_profits / gross_losses if gross_losses > 0 else float('inf')

        rolling_metrics["rolling_profit_factor"] = strategy_returns.rolling(window).apply(
            profit_factor, raw=False
        )

        return rolling_metrics

    def optimize_hyperparameters(self,
                               model_class: Any,
                               feature_data: pd.DataFrame,
                               target: Union[str, pd.Series],
                               param_grid: Dict[str, List[Any]],
                               cv: int = 5,
                               n_iter: int = 50,
                               method: str = 'bayesian',
                               scoring: str = 'neg_mean_squared_error',
                               n_jobs: int = -1) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Optimize model hyperparameters.

        Args:
            model_class: Model class to optimize
            feature_data: DataFrame with features
            target: Target column name or Series
            param_grid: Parameter grid to search
            cv: Number of cross-validation folds
            n_iter: Number of iterations for random/bayesian search
            method: Optimization method
            scoring: Scoring metric
            n_jobs: Number of parallel jobs

        Returns:
            Tuple of (best parameters, optimization results)
        """
        # Get target data
        if isinstance(target, str):
            if target in feature_data.columns:
                target_data = feature_data[target]
                feature_data = feature_data.drop(columns=[target])
            else:
                raise ValueError(f"Target column '{target}' not found in feature data")
        else:
            target_data = target

        # Create base model
        base_model = model_class()

        # Create hyperparameter optimizer if needed
        if self.hyperparameter_optimizer is None:
            from models.research.hyperparameter_optimization import create_optimizer
            self.hyperparameter_optimizer = create_optimizer(
                method=method,
                n_iterations=n_iter,
                random_state=self.config.random_state,
                n_jobs=n_jobs,
                scoring=scoring
            )

        # Create parameter definitions
        from models.research.hyperparameter_optimization import create_parameter
        parameters = []

        for param_name, param_values in param_grid.items():
            # Infer parameter type
            if all(isinstance(v, (int, np.integer)) for v in param_values):
                param_type = "int"
                param_range = (min(param_values), max(param_values))
                log_scale = max(param_values) / min(param_values) > 10 if min(param_values) > 0 else False
            elif all(isinstance(v, (float, np.floating)) for v in param_values):
                param_type = "float"
                param_range = (min(param_values), max(param_values))
                log_scale = max(param_values) / min(param_values) > 10 if min(param_values) > 0 else False
            elif all(isinstance(v, bool) for v in param_values):
                param_type = "boolean"
                param_range = None
            else:
                param_type = "categorical"
                param_range = None

            # Create parameter definition
            if param_type in ["int", "float"]:
                parameters.append(create_parameter(
                    name=param_name,
                    type=param_type,
                    values=param_range,
                    log_scale=log_scale
                ))
            else:
                parameters.append(create_parameter(
                    name=param_name,
                    type=param_type,
                    values=param_values
                ))

        # Run optimization
        start_time = time.time()
        result = self.hyperparameter_optimizer.optimize(
            estimator=base_model,
            parameters=parameters,
            X=feature_data,
            y=target_data,
            model_type=type(base_model).__name__
        )

        # Create results dictionary
        optimization_time = time.time() - start_time
        optimization_results = {
            "best_params": result.best_params,
            "best_score": result.best_score,
            "cv_scores": result.cv_scores,
            "optimization_time": optimization_time,
            "method": result.method.value,
            "iterations": len(result.all_results),
            "all_results": [
                {"params": r["params"], "score": r["mean_test_score"]}
                for r in result.all_results[:10]  # Top 10 results
            ]
        }

        logger.info(f"Optimized hyperparameters for {type(base_model).__name__} "
                   f"with best score: {result.best_score:.4f}, "
                   f"time: {optimization_time:.2f}s")

        return result.best_params, optimization_results

    def run_backtest(self,
                    model: Any,
                    data: pd.DataFrame,
                    features: List[str],
                    target: str = 'target',
                    position_sizing: str = 'fixed',
                    initial_capital: Optional[float] = None,
                    transaction_costs: float = 0.001,
                    slippage: float = 0.001) -> Dict[str, Any]:
        """
        Run a vectorized backtest of a trading strategy.

        Args:
            model: Trained model for signal generation
            data: Market data with features
            features: List of features to use
            target: Target column name
            position_sizing: Position sizing method
            initial_capital: Initial capital (uses config if None)
            transaction_costs: Transaction costs as fraction of trade value
            slippage: Slippage as fraction of price

        Returns:
            Dictionary with backtest results
        """
        # Check if features are available
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")

        # Check if target is available
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in data")

        # Set default initial capital
        if initial_capital is None:
            initial_capital = self.config.initial_capital

        # Clone data to avoid modifying original
        backtest_data = data.copy()

        # Generate predictions
        feature_data = backtest_data[features]
        predictions = model.predict(feature_data)

        # Add predictions to data
        backtest_data['prediction'] = predictions

        # Generate trading signals (buy/sell/hold)
        backtest_data['signal'] = np.sign(backtest_data['prediction'])

        # Calculate position sizes
        if position_sizing == 'fixed':
            # Fixed position size (1 unit)
            backtest_data['position'] = backtest_data['signal']
        elif position_sizing == 'percent':
            # Percent of capital (based on prediction strength)
            backtest_data['position'] = backtest_data['signal'] * np.abs(backtest_data['prediction'])
        else:
            # Default to fixed
            backtest_data['position'] = backtest_data['signal']

        # Calculate position changes
        backtest_data['position_change'] = backtest_data['position'].diff().fillna(0)

        # Calculate transaction costs
        backtest_data['costs'] = np.abs(backtest_data['position_change']) * transaction_costs * backtest_data['close']

        # Apply slippage to entry and exit prices
        backtest_data['entry_price'] = backtest_data['close'] * (1 + slippage * np.sign(backtest_data['position_change']))

        # Calculate returns
        # - Market returns (no strategy)
        backtest_data['market_return'] = backtest_data['close'].pct_change().fillna(0)

        # - Strategy returns (with position sizing, costs, and slippage)
        backtest_data['strategy_return'] = backtest_data['position'].shift(1) * backtest_data['market_return'] - backtest_data['costs']

        # Calculate equity curves
        backtest_data['market_equity'] = (1 + backtest_data['market_return']).cumprod() * initial_capital
        backtest_data['strategy_equity'] = (1 + backtest_data['strategy_return']).cumprod() * initial_capital

        # Calculate drawdowns
        backtest_data['market_dd'] = 1 - backtest_data['market_equity'] / backtest_data['market_equity'].cummax()
        backtest_data['strategy_dd'] = 1 - backtest_data['strategy_equity'] / backtest_data['strategy_equity'].cummax()

        # Calculate trade statistics
        trade_stats = self._calculate_trade_stats(backtest_data)

        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(backtest_data)

        # Add rolling risk metrics
        rolling_metrics = self._calculate_rolling_metrics(backtest_data)

        # Combine results
        results = {
            "data": backtest_data,
            "trade_stats": trade_stats,
            "performance_metrics": performance_metrics,
            "rolling_metrics": rolling_metrics,
            "initial_capital": initial_capital,
            "final_equity": backtest_data['strategy_equity'].iloc[-1],
            "total_return": (backtest_data['strategy_equity'].iloc[-1] / initial_capital) - 1,
            "max_drawdown": backtest_data['strategy_dd'].max(),
            "transaction_costs": transaction_costs,
            "slippage": slippage
        }

        logger.info(f"Backtest completed with total return: {results['total_return']:.2%}, "
                   f"max drawdown: {results['max_drawdown']:.2%}")

        return results