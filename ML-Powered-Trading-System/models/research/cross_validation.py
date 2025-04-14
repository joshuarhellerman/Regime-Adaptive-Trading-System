"""
cross_validation.py - Time Series Cross-Validation Framework

This module provides time series cross-validation utilities for backtesting
and evaluating machine learning models in trading systems.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Iterator
from enum import Enum
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from itertools import product
import time

logger = logging.getLogger(__name__)


class CVMethod(Enum):
    """Methods for time series cross-validation"""
    EXPANDING_WINDOW = "expanding_window"  # Expanding training window
    SLIDING_WINDOW = "sliding_window"      # Fixed-size sliding window
    ANCHORED = "anchored"                  # Fixed start point, expanding end
    ROLLING_ORIGIN = "rolling_origin"      # Rolling origin forecasting
    PURGED = "purged"                      # Time series purging to avoid leakage
    EMBARGO = "embargo"                    # Time series embargo for validation


@dataclass
class CVSplit:
    """A single cross-validation split"""
    train_start: Union[int, float, pd.Timestamp]
    train_end: Union[int, float, pd.Timestamp]
    test_start: Union[int, float, pd.Timestamp]
    test_end: Union[int, float, pd.Timestamp]
    split_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TimeSeriesCV:
    """
    Time series cross-validation with considerations for financial data.
    Handles expanding windows, sliding windows, embargos, and purging.
    """

    def __init__(self,
                 method: CVMethod = CVMethod.EXPANDING_WINDOW,
                 n_splits: int = 5,
                 train_pct: float = 0.7,
                 test_size: Optional[int] = None,
                 gap: int = 0,
                 embargo: int = 0,
                 min_train_size: Optional[int] = None,
                 allow_overlap: bool = False):
        """
        Initialize time series cross-validation.

        Args:
            method: Cross-validation method
            n_splits: Number of train/test splits
            train_pct: Percentage of data for training (0.0-1.0)
            test_size: Size of test set in number of samples
            gap: Number of samples between train and test sets
            embargo: Number of samples to embargo after each test set
            min_train_size: Minimum size of training set
            allow_overlap: Whether to allow overlapping test sets
        """
        self.method = method
        self.n_splits = n_splits
        self.train_pct = train_pct
        self.test_size = test_size
        self.gap = gap
        self.embargo = embargo
        self.min_train_size = min_train_size
        self.allow_overlap = allow_overlap

        logger.info(f"TimeSeriesCV initialized with method {method.value}, {n_splits} splits")

    def split(self,
              X: Union[pd.DataFrame, np.ndarray, pd.Series],
              y: Optional[Union[pd.Series, np.ndarray]] = None) -> List[CVSplit]:
        """
        Generate cross-validation splits.

        Args:
            X: Feature data with time index
            y: Optional target data

        Returns:
            List of CVSplit objects
        """
        # Get index if available
        if hasattr(X, 'index'):
            index = X.index
        else:
            index = np.arange(len(X))

        n_samples = len(X)

        # Set default test size if not provided
        if self.test_size is None:
            # Use ~20% of data for test by default, but ensure at least 1 sample
            self.test_size = max(1, int(n_samples * 0.2))

        # Set default minimum train size if not provided
        if self.min_train_size is None:
            self.min_train_size = max(1, int(n_samples * 0.1))

        if self.method == CVMethod.EXPANDING_WINDOW:
            return self._expanding_window_split(index, n_samples)
        elif self.method == CVMethod.SLIDING_WINDOW:
            return self._sliding_window_split(index, n_samples)
        elif self.method == CVMethod.ANCHORED:
            return self._anchored_split(index, n_samples)
        elif self.method == CVMethod.ROLLING_ORIGIN:
            return self._rolling_origin_split(index, n_samples)
        elif self.method == CVMethod.PURGED:
            return self._purged_split(index, n_samples, X, y)
        elif self.method == CVMethod.EMBARGO:
            return self._embargo_split(index, n_samples, X, y)
        else:
            raise ValueError(f"Unknown cross-validation method: {self.method}")

    def _expanding_window_split(self, index, n_samples) -> List[CVSplit]:
        """
        Generate expanding window splits.

        Args:
            index: Data index
            n_samples: Number of samples

        Returns:
            List of CVSplit objects
        """
        # Check if we can create the requested number of splits
        max_splits = (n_samples - self.min_train_size - self.gap) // self.test_size
        if not self.allow_overlap:
            max_splits = min(max_splits, (n_samples - self.min_train_size - self.gap) // (self.test_size + self.embargo))

        n_splits = min(self.n_splits, max_splits)
        if n_splits < self.n_splits:
            logger.warning(f"Requested {self.n_splits} splits but can only create {n_splits} with current settings")

        if n_splits <= 0:
            raise ValueError(f"Cannot create any splits with current settings. Adjust parameters.")

        # Calculate initial train size
        if self.train_pct > 0:
            initial_train_size = max(self.min_train_size, int(n_samples * self.train_pct))
        else:
            initial_train_size = self.min_train_size

        # Generate splits
        splits = []
        test_end_idx = n_samples

        for i in range(n_splits):
            # Calculate test start/end
            test_start_idx = test_end_idx - self.test_size

            # Ensure test start is valid
            if test_start_idx < initial_train_size + self.gap:
                break

            # Calculate train start/end
            train_start_idx = 0
            train_end_idx = test_start_idx - self.gap

            # Create split
            split = CVSplit(
                train_start=index[train_start_idx],
                train_end=index[train_end_idx - 1],
                test_start=index[test_start_idx],
                test_end=index[test_end_idx - 1],
                split_index=i
            )

            splits.append(split)

            # Update test end index for next iteration
            if self.allow_overlap:
                test_end_idx = test_start_idx + max(1, (n_samples - initial_train_size - self.gap - self.test_size) // n_splits)
            else:
                test_end_idx = test_start_idx

        # Reverse so earliest split has index 0
        return list(reversed(splits))

    def _sliding_window_split(self, index, n_samples) -> List[CVSplit]:
        """
        Generate sliding window splits.

        Args:
            index: Data index
            n_samples: Number of samples

        Returns:
            List of CVSplit objects
        """
        # Calculate train window size
        if self.train_pct > 0:
            train_size = max(self.min_train_size, int(n_samples * self.train_pct))
        else:
            train_size = self.min_train_size

        # Check if we can create the requested number of splits
        max_splits = (n_samples - train_size - self.gap - self.test_size) + 1
        if not self.allow_overlap:
            max_splits = min(max_splits, (n_samples - train_size - self.gap) // (self.test_size + self.embargo))

        n_splits = min(self.n_splits, max_splits)
        if n_splits < self.n_splits:
            logger.warning(f"Requested {self.n_splits} splits but can only create {n_splits} with current settings")

        if n_splits <= 0:
            raise ValueError(f"Cannot create any splits with current settings. Adjust parameters.")

        # Generate splits
        splits = []

        # Calculate step size
        step_size = max(1, (n_samples - train_size - self.gap - self.test_size) // (n_splits - 1)) if n_splits > 1 else 1

        for i in range(n_splits):
            train_start_idx = i * step_size
            train_end_idx = train_start_idx + train_size

            test_start_idx = train_end_idx + self.gap
            test_end_idx = test_start_idx + self.test_size

            # Check if we've gone past the end of the data
            if test_end_idx > n_samples:
                break

            # Create split
            split = CVSplit(
                train_start=index[train_start_idx],
                train_end=index[train_end_idx - 1],
                test_start=index[test_start_idx],
                test_end=index[test_end_idx - 1],
                split_index=i
            )

            splits.append(split)

        return splits

    def _anchored_split(self, index, n_samples) -> List[CVSplit]:
        """
        Generate anchored splits (fixed start point, expanding end).

        Args:
            index: Data index
            n_samples: Number of samples

        Returns:
            List of CVSplit objects
        """
        # Check if we can create the requested number of splits
        max_splits = (n_samples - self.min_train_size - self.gap) // self.test_size
        if not self.allow_overlap:
            max_splits = min(max_splits, (n_samples - self.min_train_size - self.gap) // (self.test_size + self.embargo))

        n_splits = min(self.n_splits, max_splits)
        if n_splits < self.n_splits:
            logger.warning(f"Requested {self.n_splits} splits but can only create {n_splits} with current settings")

        if n_splits <= 0:
            raise ValueError(f"Cannot create any splits with current settings. Adjust parameters.")

        # Generate splits
        splits = []

        # Fixed train start
        train_start_idx = 0

        # Calculate step size to spread test sets evenly
        step_size = max(1, (n_samples - self.min_train_size - self.gap - self.test_size) // (n_splits - 1)) if n_splits > 1 else 1

        for i in range(n_splits):
            train_end_idx = self.min_train_size + i * step_size
            test_start_idx = train_end_idx + self.gap
            test_end_idx = test_start_idx + self.test_size

            # Check if we've gone past the end of the data
            if test_end_idx > n_samples:
                break

            # Create split
            split = CVSplit(
                train_start=index[train_start_idx],
                train_end=index[train_end_idx - 1],
                test_start=index[test_start_idx],
                test_end=index[test_end_idx - 1],
                split_index=i
            )

            splits.append(split)

        return splits

    def _rolling_origin_split(self, index, n_samples) -> List[CVSplit]:
        """
        Generate rolling origin forecasting splits.

        Args:
            index: Data index
            n_samples: Number of samples

        Returns:
            List of CVSplit objects
        """
        # This is similar to sliding window but with more carefully chosen test periods
        # for time series forecasting

        # Calculate train window size
        if self.train_pct > 0:
            train_size = max(self.min_train_size, int(n_samples * self.train_pct))
        else:
            train_size = self.min_train_size

        # Check if we can create the requested number of splits
        max_splits = (n_samples - train_size - self.gap) // self.test_size

        n_splits = min(self.n_splits, max_splits)
        if n_splits < self.n_splits:
            logger.warning(f"Requested {self.n_splits} splits but can only create {n_splits} with current settings")

        if n_splits <= 0:
            raise ValueError(f"Cannot create any splits with current settings. Adjust parameters.")

        # Generate splits
        splits = []

        for i in range(n_splits):
            train_start_idx = 0
            train_end_idx = train_size + i * self.test_size

            test_start_idx = train_end_idx + self.gap
            test_end_idx = test_start_idx + self.test_size

            # Check if we've gone past the end of the data
            if test_end_idx > n_samples:
                break

            # Create split
            split = CVSplit(
                train_start=index[train_start_idx],
                train_end=index[train_end_idx - 1],
                test_start=index[test_start_idx],
                test_end=index[test_end_idx - 1],
                split_index=i
            )

            splits.append(split)

        return splits

    def _purged_split(self, index, n_samples, X, y) -> List[CVSplit]:
        """
        Generate splits with purging to prevent data leakage.

        Args:
            index: Data index
            n_samples: Number of samples
            X: Feature data with time index
            y: Target data

        Returns:
            List of CVSplit objects
        """
        # This is a simplified version of purged cross-validation
        # For a full implementation, we would need labels with timestamps

        # Start with expanding window splits
        splits = self._expanding_window_split(index, n_samples)

        # For each split, mark the "purged" regions where data would be leaked
        for i, split in enumerate(splits):
            # In financial data, purging often involves removing samples around
            # test period edges to avoid information leakage
            purge_before = max(0, index.get_loc(split.test_start) - self.gap)
            purge_after = min(n_samples - 1, index.get_loc(split.test_end) + self.gap)

            split.metadata["purged_indices"] = list(range(purge_before, purge_after + 1))

        return splits

    def _embargo_split(self, index, n_samples, X, y) -> List[CVSplit]:
        """
        Generate splits with embargo periods after test sets.

        Args:
            index: Data index
            n_samples: Number of samples
            X: Feature data with time index
            y: Target data

        Returns:
            List of CVSplit objects
        """
        # Start with expanding window splits
        splits = self._expanding_window_split(index, n_samples)

        # Apply embargo periods
        for i, split in enumerate(splits):
            if i < len(splits) - 1:  # Skip last split
                test_end_idx = index.get_loc(split.test_end)
                embargo_end_idx = min(test_end_idx + self.embargo, n_samples - 1)

                # Store embargo period
                split.metadata["embargo_start"] = split.test_end
                split.metadata["embargo_end"] = index[embargo_end_idx]
                split.metadata["embargo_indices"] = list(range(test_end_idx + 1, embargo_end_idx + 1))

        return splits

    def get_indices(self,
                   split: CVSplit,
                   index: Union[pd.Index, np.ndarray, List]) -> Dict[str, List[int]]:
        """
        Get indices for a split.

        Args:
            split: Cross-validation split
            index: Data index

        Returns:
            Dictionary with train and test indices
        """
        if isinstance(index, pd.Index):
            train_start_idx = index.get_loc(split.train_start)
            train_end_idx = index.get_loc(split.train_end)
            test_start_idx = index.get_loc(split.test_start)
            test_end_idx = index.get_loc(split.test_end)
        else:
            # Find indices by value
            train_start_idx = np.where(index == split.train_start)[0][0]
            train_end_idx = np.where(index == split.train_end)[0][0]
            test_start_idx = np.where(index == split.test_start)[0][0]
            test_end_idx = np.where(index == split.test_end)[0][0]

        train_indices = list(range(train_start_idx, train_end_idx + 1))
        test_indices = list(range(test_start_idx, test_end_idx + 1))

        # Apply purging if present in metadata
        if "purged_indices" in split.metadata:
            train_indices = [i for i in train_indices if i not in split.metadata["purged_indices"]]

        return {
            "train": train_indices,
            "test": test_indices
        }

    def get_train_test_data(self,
                           split: CVSplit,
                           X: Union[pd.DataFrame, np.ndarray],
                           y: Optional[Union[pd.Series, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Get train/test data for a split.

        Args:
            split: Cross-validation split
            X: Feature data
            y: Optional target data

        Returns:
            Dictionary with train and test data
        """
        if hasattr(X, 'index'):
            index = X.index

            train_mask = (index >= split.train_start) & (index <= split.train_end)
            test_mask = (index >= split.test_start) & (index <= split.test_end)

            X_train = X.loc[train_mask]
            X_test = X.loc[test_mask]

            if y is not None:
                y_train = y.loc[train_mask]
                y_test = y.loc[test_mask]
            else:
                y_train = None
                y_test = None
        else:
            # Use positional indices
            indices = self.get_indices(split, np.arange(len(X)))

            X_train = X[indices["train"]]
            X_test = X[indices["test"]]

            if y is not None:
                y_train = y[indices["train"]]
                y_test = y[indices["test"]]
            else:
                y_train = None
                y_test = None

        # Apply purging if present in metadata
        if "purged_indices" in split.metadata and hasattr(X_train, 'iloc'):
            purged_indices = split.metadata["purged_indices"]
            keep_indices = [i for i in range(len(X_train)) if i not in purged_indices]
            X_train = X_train.iloc[keep_indices]
            if y_train is not None:
                y_train = y_train.iloc[keep_indices]

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }

    def plot_splits(self, splits: List[CVSplit], figsize: Tuple[int, int] = (12, 8)):
        """
        Visualize the cross-validation splits.

        Args:
            splits: List of CV splits
            figsize: Figure size
        """
        plt.figure(figsize=figsize)

        # Sort splits by index for cleaner visualization
        splits = sorted(splits, key=lambda s: s.split_index)

        for i, split in enumerate(splits):
            train_start = pd.to_datetime(split.train_start) if isinstance(split.train_start, (str, pd.Timestamp)) else split.train_start
            train_end = pd.to_datetime(split.train_end) if isinstance(split.train_end, (str, pd.Timestamp)) else split.train_end
            test_start = pd.to_datetime(split.test_start) if isinstance(split.test_start, (str, pd.Timestamp)) else test_start
            test_end = pd.to_datetime(split.test_end) if isinstance(split.test_end, (str, pd.Timestamp)) else test_end

            # Plot train period
            plt.plot([train_start, train_end], [i, i], 'b-', linewidth=2, label='Train' if i == 0 else "")
            plt.plot([train_start, train_start], [i-0.1, i+0.1], 'b|')
            plt.plot([train_end, train_end], [i-0.1, i+0.1], 'b|')

            # Plot gap if exists
            if split.gap:
                plt.plot([train_end, test_start], [i, i], 'k--', alpha=0.3)

            # Plot test period
            plt.plot([test_start, test_end], [i, i], 'r-', linewidth=2, label='Test' if i == 0 else "")
            plt.plot([test_start, test_start], [i-0.1, i+0.1], 'r|')
            plt.plot([test_end, test_end], [i-0.1, i+0.1], 'r|')

            # Plot embargo period if exists
            if "embargo_start" in split.metadata and "embargo_end" in split.metadata:
                embargo_start = pd.to_datetime(split.metadata["embargo_start"]) if isinstance(split.metadata["embargo_start"], (str, pd.Timestamp)) else split.metadata["embargo_start"]
                embargo_end = pd.to_datetime(split.metadata["embargo_end"]) if isinstance(split.metadata["embargo_end"], (str, pd.Timestamp)) else split.metadata["embargo_end"]

                plt.plot([embargo_start, embargo_end], [i, i], 'g-', alpha=0.5, linewidth=2, label='Embargo' if i == 0 else "")

        plt.yticks(range(len(splits)), [f"Split {i}" for i in range(len(splits))])
        plt.xlabel('Time')
        plt.ylabel('CV Splits')
        plt.title(f'Time Series {self.method.value} Cross-Validation')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.grid(True, alpha=0.3)

        return plt.gcf()


class ParameterGrid:
    """Grid of parameters for hyperparameter optimization with time series considerations"""

    def __init__(self, param_grid: Dict[str, List[Any]]):
        """
        Initialize parameter grid.

        Args:
            param_grid: Dictionary mapping parameter names to possible values
        """
        self.param_grid = param_grid
        self.param_names = list(param_grid.keys())
        self.param_values = list(param_grid.values())

        # Calculate total number of combinations
        self.n_combinations = 1
        for values in self.param_values:
            self.n_combinations *= len(values)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate through parameter combinations"""
        for combination in product(*self.param_values):
            yield {name: value for name, value in zip(self.param_names, combination)}

    def __len__(self) -> int:
        """Get number of parameter combinations"""
        return self.n_combinations


class TimeSeriesGridSearchCV:
    """Grid search with time series cross-validation for trading models"""

    def __init__(self,
                 estimator: Any,
                 param_grid: Dict[str, List[Any]],
                 cv: TimeSeriesCV,
                 scoring: Union[str, Callable] = 'neg_mean_squared_error',
                 n_jobs: int = 1,
                 verbose: int = 0,
                 refit: bool = True,
                 return_train_score: bool = False):
        """
        Initialize time series grid search.

        Args:
            estimator: The model to optimize
            param_grid: Parameter grid
            cv: TimeSeriesCV instance
            scoring: Scoring method or function(y_true, y_pred) -> float
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
            refit: Whether to refit the model on the best parameters
            return_train_score: Whether to compute train scores
        """
        self.estimator = estimator
        self.param_grid = ParameterGrid(param_grid)
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.refit = refit
        self.return_train_score = return_train_score

        # Results
        self.cv_results_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.best_index_ = None
        self.best_estimator_ = None

        logger.info(f"TimeSeriesGridSearchCV initialized with {len(self.param_grid)} parameter combinations")

    def fit(self, X, y=None):
        """
        Fit the grid search.

        Args:
            X: Feature data
            y: Target data

        Returns:
            Self
        """
        # Generate cross-validation splits
        cv_splits = self.cv.split(X, y)

        if len(cv_splits) == 0:
            raise ValueError("No CV splits generated. Check CV parameters.")

        if self.verbose > 0:
            logger.info(f"Performing grid search with {len(cv_splits)} CV splits and {len(self.param_grid)} parameter combinations")

        # Initialize results
        self.cv_results_ = {
            "params": [],
            "mean_test_score": [],
            "std_test_score": [],
            "split_test_scores": [],
            "mean_fit_time": [],
            "mean_score_time": []
        }

        if self.return_train_score:
            self.cv_results_["mean_train_score"] = []
            self.cv_results_["std_train_score"] = []
            self.cv_results_["split_train_scores"] = []

        # Start grid search
        start_time = time.time()

        for params in self.param_grid:
            if self.verbose > 0:
                logger.info(f"Evaluating parameters: {params}")

            # Track scores for this parameter set
            test_scores = []
            train_scores = []
            fit_times = []
            score_times = []

            # Cross-validation
            for split in cv_splits:
                # Get train/test data
                split_data = self.cv.get_train_test_data(split, X, y)
                X_train, X_test = split_data["X_train"], split_data["X_test"]
                y_train, y_test = split_data["y_train"], split_data["y_test"]

                # Clone estimator
                estimator = self._clone_estimator()

                # Set parameters
                for param, value in params.items():
                    setattr(estimator, param, value)

                # Fit model
                fit_start = time.time()
                estimator.fit(X_train, y_train)
                fit_end = time.time()
                fit_times.append(fit_end - fit_start)

                # Score model
                score_start = time.time()
                test_score = self._compute_score(estimator, X_test, y_test)
                test_scores.append(test_score)

                if self.return_train_score:
                    train_score = self._compute_score(estimator, X_train, y_train)
                    train_scores.append(train_score)

                score_end = time.time()
                score_times.append(score_end - score_start)

            # Compute mean and std of scores
            mean_test_score = np.mean(test_scores)
            std_test_score = np.std(test_scores)
            mean_fit_time = np.mean(fit_times)
            mean_score_time = np.mean(score_times)

            # Add to results
            self.cv_results_["params"].append(params)
            self.cv_results_["mean_test_score"].append(mean_test_score)
            self.cv_results_["std_test_score"].append(std_test_score)
            self.cv_results_["split_test_scores"].append(test_scores)
            self.cv_results_["mean_fit_time"].append(mean_fit_time)
            self.cv_results_["mean_score_time"].append(mean_score_time)

            if self.return_train_score:
                mean_train_score = np.mean(train_scores)
                std_train_score = np.std(train_scores)
                self.cv_results_["mean_train_score"].append(mean_train_score)
                self.cv_results_["std_train_score"].append(std_train_score)
                self.cv_results_["split_train_scores"].append(train_scores)

            if self.verbose > 0:
                logger.info(f"Mean CV score: {mean_test_score:.4f} (±{std_test_score:.4f})")

        # Find best parameters
        best_index = np.argmax(self.cv_results_["mean_test_score"])
        self.best_index_ = best_index
        self.best_params_ = self.cv_results_["params"][best_index]
        self.best_score_ = self.cv_results_["mean_test_score"][best_index]

        # Refit on full dataset if requested
        if self.refit:
            if self.verbose > 0:
                logger.info(f"Refitting model with best parameters: {self.best_params_}")

            self.best_estimator_ = self._clone_estimator()

            # Set best parameters
            for param, value in self.best_params_.items():
                setattr(self.best_estimator_, param, value)

            # Fit on full dataset
            self.best_estimator_.fit(X, y)

        total_time = time.time() - start_time
        if self.verbose > 0:
            logger.info(f"Grid search completed in {total_time:.2f}s")
            logger.info(f"Best parameters: {self.best_params_}")
            logger.info(f"Best CV score: {self.best_score_:.4f}")

        return self

    def predict(self, X):
        """
        Make predictions with the best estimator.

        Args:
            X: Feature data

        Returns:
            Predictions
        """
        if self.best_estimator_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.best_estimator_.predict(X)

    def _clone_estimator(self):
        """Clone the estimator"""
        try:
            from sklearn.base import clone
            return clone(self.estimator)
        except ImportError:
            # If sklearn is not available, try a simple copy
            import copy
            return copy.deepcopy(self.estimator)

    def _compute_score(self, estimator, X, y):
        """
        Compute score for a fitted estimator.

        Args:
            estimator: Fitted estimator
            X: Feature data
            y: Target data

        Returns:
            Score value
        """
        if callable(self.scoring):
            # Custom scoring function
            y_pred = estimator.predict(X)
            return self.scoring(y, y_pred)
        elif isinstance(self.scoring, str):
            # Handle common scoring metrics
            if self.scoring == 'neg_mean_squared_error':
                y_pred = estimator.predict(X)
                return -np.mean((y - y_pred) ** 2)
            elif self.scoring == 'neg_mean_absolute_error':
                y_pred = estimator.predict(X)
                return -np.mean(np.abs(y - y_pred))
            elif self.scoring == 'r2':
                y_pred = estimator.predict(X)
                u = ((y - y_pred) ** 2).sum()
                v = ((y - y.mean()) ** 2).sum()
                return 1 - (u / v) if v > 0 else 0
            elif self.scoring == 'accuracy':
                y_pred = estimator.predict(X)
                return np.mean(y == y_pred)
            else:
                raise ValueError(f"Unknown scoring method: {self.scoring}")
        else:
            raise ValueError("scoring should be a callable or a string")


# Factory function for creating cross-validators
def create_time_series_cv(method: str = "expanding_window", **kwargs) -> TimeSeriesCV:
    """
    Create a time series cross-validator.

    Args:
        method: Cross-validation method name
        **kwargs: Additional parameters for the cross-validator

    Returns:
        TimeSeriesCV instance
    """
    try:
        cv_method = CVMethod(method)
    except ValueError:
        raise ValueError(f"Unknown cross-validation method: {method}. "
                         f"Available methods: {[m.value for m in CVMethod]}")

    return TimeSeriesCV(method=cv_method, **kwargs)r