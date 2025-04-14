import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Import the module to test
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from models.research.cross_validation import (
    TimeSeriesCV,
    CVMethod,
    CVSplit,
    TimeSeriesGridSearchCV,
    ParameterGrid,
    create_time_series_cv
)


class TestCVSplit(unittest.TestCase):
    """Tests for the CVSplit class."""
    
    def test_init_with_defaults(self):
        """Test initialization with default values."""
        split = CVSplit(
            train_start=pd.Timestamp('2020-01-01'),
            train_end=pd.Timestamp('2020-03-31'),
            test_start=pd.Timestamp('2020-04-01'),
            test_end=pd.Timestamp('2020-04-30')
        )
        
        self.assertEqual(split.train_start, pd.Timestamp('2020-01-01'))
        self.assertEqual(split.train_end, pd.Timestamp('2020-03-31'))
        self.assertEqual(split.test_start, pd.Timestamp('2020-04-01'))
        self.assertEqual(split.test_end, pd.Timestamp('2020-04-30'))
        self.assertEqual(split.split_index, 0)
        self.assertEqual(split.metadata, {})
    
    def test_init_with_all_params(self):
        """Test initialization with all parameters."""
        metadata = {"source": "test"}
        split = CVSplit(
            train_start=pd.Timestamp('2020-01-01'),
            train_end=pd.Timestamp('2020-03-31'),
            test_start=pd.Timestamp('2020-04-01'),
            test_end=pd.Timestamp('2020-04-30'),
            split_index=5,
            metadata=metadata
        )
        
        self.assertEqual(split.split_index, 5)
        self.assertEqual(split.metadata, metadata)
    
    def test_with_integer_indices(self):
        """Test initialization with integer indices."""
        split = CVSplit(
            train_start=0,
            train_end=99,
            test_start=100,
            test_end=149,
            split_index=2
        )
        
        self.assertEqual(split.train_start, 0)
        self.assertEqual(split.train_end, 99)
        self.assertEqual(split.test_start, 100)
        self.assertEqual(split.test_end, 149)


class TestCVMethod(unittest.TestCase):
    """Tests for the CVMethod enum."""
    
    def test_enum_values(self):
        """Test that the enum has the expected values."""
        self.assertEqual(CVMethod.EXPANDING_WINDOW.value, "expanding_window")
        self.assertEqual(CVMethod.SLIDING_WINDOW.value, "sliding_window")
        self.assertEqual(CVMethod.ANCHORED.value, "anchored")
        self.assertEqual(CVMethod.ROLLING_ORIGIN.value, "rolling_origin")
        self.assertEqual(CVMethod.PURGED.value, "purged")
        self.assertEqual(CVMethod.EMBARGO.value, "embargo")
    
    def test_creating_from_string(self):
        """Test creating enum from string value."""
        self.assertEqual(CVMethod("expanding_window"), CVMethod.EXPANDING_WINDOW)
        self.assertEqual(CVMethod("sliding_window"), CVMethod.SLIDING_WINDOW)
        self.assertEqual(CVMethod("anchored"), CVMethod.ANCHORED)
        self.assertEqual(CVMethod("rolling_origin"), CVMethod.ROLLING_ORIGIN)
        self.assertEqual(CVMethod("purged"), CVMethod.PURGED)
        self.assertEqual(CVMethod("embargo"), CVMethod.EMBARGO)
    
    def test_invalid_enum_value(self):
        """Test that invalid enum values raise ValueError."""
        with self.assertRaises(ValueError):
            CVMethod("invalid_method")


class TestTimeSeriesCV(unittest.TestCase):
    """Tests for TimeSeriesCV class."""
    
    def setUp(self):
        """Set up test data."""
        # Create daily data for 1 year
        self.dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
        self.X = pd.DataFrame({
            'feature1': np.random.randn(365),
            'feature2': np.random.randn(365)
        }, index=self.dates)
        self.y = pd.Series(np.random.randn(365), index=self.dates)
        
        # Create numpy arrays for testing without index
        self.X_array = np.random.randn(365, 2)
        self.y_array = np.random.randn(365)
    
    def test_init_with_defaults(self):
        """Test initialization with default values."""
        cv = TimeSeriesCV()
        
        self.assertEqual(cv.method, CVMethod.EXPANDING_WINDOW)
        self.assertEqual(cv.n_splits, 5)
        self.assertEqual(cv.train_pct, 0.7)
        self.assertIsNone(cv.test_size)
        self.assertEqual(cv.gap, 0)
        self.assertEqual(cv.embargo, 0)
        self.assertIsNone(cv.min_train_size)
        self.assertFalse(cv.allow_overlap)
    
    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        cv = TimeSeriesCV(
            method=CVMethod.SLIDING_WINDOW,
            n_splits=3,
            train_pct=0.6,
            test_size=30,
            gap=5,
            embargo=10,
            min_train_size=60,
            allow_overlap=True
        )
        
        self.assertEqual(cv.method, CVMethod.SLIDING_WINDOW)
        self.assertEqual(cv.n_splits, 3)
        self.assertEqual(cv.train_pct, 0.6)
        self.assertEqual(cv.test_size, 30)
        self.assertEqual(cv.gap, 5)
        self.assertEqual(cv.embargo, 10)
        self.assertEqual(cv.min_train_size, 60)
        self.assertTrue(cv.allow_overlap)
    
    def test_expanding_window_split(self):
        """Test expanding window split method."""
        cv = TimeSeriesCV(
            method=CVMethod.EXPANDING_WINDOW,
            n_splits=3,
            test_size=30,
            gap=5
        )
        
        splits = cv.split(self.X)
        
        # Check number of splits
        self.assertEqual(len(splits), 3)
        
        # Check that splits are ordered correctly
        for i in range(len(splits)):
            self.assertEqual(splits[i].split_index, i)
        
        # Check that test periods don't overlap
        for i in range(len(splits) - 1):
            self.assertLess(
                self.dates.get_loc(splits[i].test_end),
                self.dates.get_loc(splits[i+1].test_start)
            )
        
        # Check that training periods are expanding
        for i in range(len(splits) - 1):
            self.assertEqual(splits[i].train_start, splits[i+1].train_start)
            self.assertLess(
                self.dates.get_loc(splits[i].train_end),
                self.dates.get_loc(splits[i+1].train_end)
            )
    
    def test_sliding_window_split(self):
        """Test sliding window split method."""
        cv = TimeSeriesCV(
            method=CVMethod.SLIDING_WINDOW,
            n_splits=3,
            test_size=30,
            train_pct=0.2,  # Use a fixed percentage for training
            gap=5
        )
        
        splits = cv.split(self.X)
        
        # Check number of splits
        self.assertEqual(len(splits), 3)
        
        # Check that the train window size is consistent
        train_sizes = [
            self.dates.get_loc(s.train_end) - self.dates.get_loc(s.train_start) + 1
            for s in splits
        ]
        self.assertTrue(all(size == train_sizes[0] for size in train_sizes))
        
        # Check that test periods don't overlap
        for i in range(len(splits) - 1):
            self.assertLess(
                self.dates.get_loc(splits[i].test_end),
                self.dates.get_loc(splits[i+1].test_start)
            )
    
    def test_anchored_split(self):
        """Test anchored split method."""
        cv = TimeSeriesCV(
            method=CVMethod.ANCHORED,
            n_splits=3,
            test_size=30,
            gap=5,
            min_train_size=60
        )
        
        splits = cv.split(self.X)
        
        # Check number of splits
        self.assertEqual(len(splits), 3)
        
        # Check that all splits have the same train start
        train_starts = [s.train_start for s in splits]
        self.assertTrue(all(start == train_starts[0] for start in train_starts))
        
        # Check that train end is increasing
        train_ends = [self.dates.get_loc(s.train_end) for s in splits]
        self.assertTrue(all(train_ends[i] < train_ends[i+1] for i in range(len(train_ends)-1)))
    
    def test_rolling_origin_split(self):
        """Test rolling origin split method."""
        cv = TimeSeriesCV(
            method=CVMethod.ROLLING_ORIGIN,
            n_splits=3,
            test_size=30,
            gap=5,
            train_pct=0.2
        )
        
        splits = cv.split(self.X)
        
        # Check number of splits
        self.assertEqual(len(splits), 3)
        
        # Check that all splits have the same train start
        train_starts = [s.train_start for s in splits]
        self.assertTrue(all(start == train_starts[0] for start in train_starts))
        
        # Check that test periods are consecutive
        for i in range(len(splits) - 1):
            expected_next_test_start = self.dates[
                self.dates.get_loc(splits[i].test_end) + 1
            ]
            self.assertEqual(splits[i+1].test_start, expected_next_test_start)
    
    def test_purged_split(self):
        """Test purged split method."""
        cv = TimeSeriesCV(
            method=CVMethod.PURGED,
            n_splits=3,
            test_size=30,
            gap=5
        )
        
        splits = cv.split(self.X, self.y)
        
        # Check number of splits
        self.assertEqual(len(splits), 3)
        
        # Check that purged indices are in metadata
        for split in splits:
            self.assertIn("purged_indices", split.metadata)
            self.assertTrue(isinstance(split.metadata["purged_indices"], list))
    
    def test_embargo_split(self):
        """Test embargo split method."""
        cv = TimeSeriesCV(
            method=CVMethod.EMBARGO,
            n_splits=3,
            test_size=30,
            embargo=10
        )
        
        splits = cv.split(self.X, self.y)
        
        # Check number of splits
        self.assertEqual(len(splits), 3)
        
        # Check that embargo info is in metadata (except for last split)
        for i, split in enumerate(splits[:-1]):
            self.assertIn("embargo_start", split.metadata)
            self.assertIn("embargo_end", split.metadata)
            self.assertIn("embargo_indices", split.metadata)
            
            # Check that embargo period is correct length
            embargo_indices = split.metadata["embargo_indices"]
            self.assertEqual(len(embargo_indices), cv.embargo)
    
    def test_get_indices(self):
        """Test get_indices method."""
        cv = TimeSeriesCV(test_size=30)
        splits = cv.split(self.X)
        
        for split in splits:
            indices = cv.get_indices(split, self.X.index)
            
            # Check that indices are lists
            self.assertTrue(isinstance(indices["train"], list))
            self.assertTrue(isinstance(indices["test"], list))
            
            # Check that indices are within bounds
            self.assertTrue(all(0 <= i < len(self.X) for i in indices["train"]))
            self.assertTrue(all(0 <= i < len(self.X) for i in indices["test"]))
            
            # Check that indices correspond to the right dates
            train_start_idx = self.dates.get_loc(split.train_start)
            train_end_idx = self.dates.get_loc(split.train_end)
            test_start_idx = self.dates.get_loc(split.test_start)
            test_end_idx = self.dates.get_loc(split.test_end)
            
            self.assertEqual(indices["train"][0], train_start_idx)
            self.assertEqual(indices["train"][-1], train_end_idx)
            self.assertEqual(indices["test"][0], test_start_idx)
            self.assertEqual(indices["test"][-1], test_end_idx)
    
    def test_get_train_test_data(self):
        """Test get_train_test_data method."""
        cv = TimeSeriesCV(test_size=30)
        splits = cv.split(self.X)
        
        for split in splits:
            data = cv.get_train_test_data(split, self.X, self.y)
            
            # Check that all parts are returned
            self.assertIn("X_train", data)
            self.assertIn("X_test", data)
            self.assertIn("y_train", data)
            self.assertIn("y_test", data)
            
            # Check shapes
            self.assertEqual(data["X_train"].shape[0], data["y_train"].shape[0])
            self.assertEqual(data["X_test"].shape[0], data["y_test"].shape[0])
            
            # Check that data matches the split dates
            self.assertEqual(data["X_train"].index[0], split.train_start)
            self.assertEqual(data["X_train"].index[-1], split.train_end)
            self.assertEqual(data["X_test"].index[0], split.test_start)
            self.assertEqual(data["X_test"].index[-1], split.test_end)
    
    def test_with_numpy_arrays(self):
        """Test that the CV works with numpy arrays."""
        cv = TimeSeriesCV(test_size=30)
        
        # This should work without errors
        splits = cv.split(self.X_array)
        
        # Get data for first split
        split = splits[0]
        data = cv.get_train_test_data(split, self.X_array, self.y_array)
        
        # Check shape of returned arrays
        self.assertEqual(len(data["X_train"].shape), 2)
        self.assertEqual(len(data["X_test"].shape), 2)
        self.assertEqual(len(data["y_train"].shape), 1)
        self.assertEqual(len(data["y_test"].shape), 1)
    
    def test_invalid_method(self):
        """Test that invalid method raises ValueError."""
        with self.assertRaises(ValueError):
            cv = TimeSeriesCV(method="invalid_method")
    
    def test_plot_splits(self):
        """Test plot_splits method."""
        cv = TimeSeriesCV(n_splits=3, test_size=30)
        splits = cv.split(self.X)
        
        # Should return a figure
        fig = cv.plot_splits(splits)
        self.assertIsNotNone(fig)
        plt.close(fig)


class TestParameterGrid(unittest.TestCase):
    """Tests for ParameterGrid class."""
    
    def test_init(self):
        """Test initialization."""
        param_grid = {
            'param1': [1, 2, 3],
            'param2': ['a', 'b']
        }
        grid = ParameterGrid(param_grid)
        
        self.assertEqual(grid.param_grid, param_grid)
        self.assertEqual(set(grid.param_names), {'param1', 'param2'})
        self.assertEqual(grid.n_combinations, 6)  # 3 * 2
    
    def test_iter(self):
        """Test iteration over parameters."""
        param_grid = {
            'param1': [1, 2],
            'param2': ['a', 'b']
        }
        grid = ParameterGrid(param_grid)
        
        combinations = list(grid)
        self.assertEqual(len(combinations), 4)
        
        # Check all combinations are present
        expected = [
            {'param1': 1, 'param2': 'a'},
            {'param1': 1, 'param2': 'b'},
            {'param1': 2, 'param2': 'a'},
            {'param1': 2, 'param2': 'b'}
        ]
        
        for combo in expected:
            self.assertIn(combo, combinations)
    
    def test_len(self):
        """Test length of parameter grid."""
        param_grid = {
            'param1': [1, 2, 3],
            'param2': ['a', 'b'],
            'param3': [True, False]
        }
        grid = ParameterGrid(param_grid)
        
        self.assertEqual(len(grid), 12)  # 3 * 2 * 2


class TestTimeSeriesGridSearchCV(unittest.TestCase):
    """Tests for TimeSeriesGridSearchCV class."""
    
    def setUp(self):
        """Set up test data and mock estimator."""
        # Create daily data for 1 year
        self.dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
        self.X = pd.DataFrame({
            'feature1': np.random.randn(365),
            'feature2': np.random.randn(365)
        }, index=self.dates)
        self.y = pd.Series(np.random.randn(365), index=self.dates)
        
        # Create a mock estimator
        self.estimator = MagicMock()
        self.estimator.fit.return_value = self.estimator
        self.estimator.predict.return_value = np.random.randn(30)  # For test set
        
        # Create CV
        self.cv = TimeSeriesCV(method=CVMethod.EXPANDING_WINDOW, n_splits=3, test_size=30)
        
        # Parameter grid
        self.param_grid = {
            'alpha': [0.1, 0.5],
            'fit_intercept': [True, False]
        }
    
    def test_init(self):
        """Test initialization."""
        grid_search = TimeSeriesGridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            cv=self.cv
        )
        
        self.assertEqual(grid_search.estimator, self.estimator)
        self.assertEqual(len(grid_search.param_grid), 4)  # 2 * 2
        self.assertEqual(grid_search.cv, self.cv)
        self.assertEqual(grid_search.scoring, 'neg_mean_squared_error')
        self.assertEqual(grid_search.n_jobs, 1)
        self.assertEqual(grid_search.verbose, 0)
        self.assertTrue(grid_search.refit)
        self.assertFalse(grid_search.return_train_score)
    
    def test_fit(self):
        """Test fit method."""
        grid_search = TimeSeriesGridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            cv=self.cv,
            verbose=1
        )
        
        # This should not raise errors
        result = grid_search.fit(self.X, self.y)
        
        # Check that result is self
        self.assertEqual(result, grid_search)
        
        # Check that results are populated
        self.assertIsNotNone(grid_search.cv_results_)
        self.assertIsNotNone(grid_search.best_params_)
        self.assertIsNotNone(grid_search.best_score_)
        self.assertIsNotNone(grid_search.best_index_)
        self.assertIsNotNone(grid_search.best_estimator_)
        
        # Check results structure
        self.assertIn("params", grid_search.cv_results_)
        self.assertIn("mean_test_score", grid_search.cv_results_)
        self.assertIn("std_test_score", grid_search.cv_results_)
        
        # Check that best_params_ is one of the param combinations
        self.assertIn(grid_search.best_params_, grid_search.cv_results_["params"])
    
    def test_with_return_train_score(self):
        """Test with return_train_score=True."""
        grid_search = TimeSeriesGridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            cv=self.cv,
            return_train_score=True
        )
        
        grid_search.fit(self.X, self.y)
        
        # Check that train scores are in results
        self.assertIn("mean_train_score", grid_search.cv_results_)
        self.assertIn("std_train_score", grid_search.cv_results_)
    
    def test_predict(self):
        """Test predict method."""
        grid_search = TimeSeriesGridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            cv=self.cv
        )
        
        grid_search.fit(self.X, self.y)
        
        # This should call predict on best_estimator_
        X_new = pd.DataFrame({
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10)
        })
        predictions = grid_search.predict(X_new)
        
        # Check that predictions have the right shape
        self.assertEqual(len(predictions), 10)
        
        # Check that predict was called on best_estimator_
        grid_search.best_estimator_.predict.assert_called_once()
    
    def test_custom_scoring(self):
        """Test with custom scoring function."""
        def custom_scorer(y_true, y_pred):
            return -np.mean(np.abs(y_true - y_pred))
        
        grid_search = TimeSeriesGridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            cv=self.cv,
            scoring=custom_scorer
        )
        
        # This should work without errors
        grid_search.fit(self.X, self.y)
    
    def test_no_refit(self):
        """Test with refit=False."""
        grid_search = TimeSeriesGridSearchCV(
            estimator=self.estimator,
            param_grid=self.param_grid,
            cv=self.cv,
            refit=False
        )
        
        grid_search.fit(self.X, self.y)
        
        # best_estimator_ should be None
        self.assertIsNone(grid_search.best_estimator_)
        
        # predict should raise ValueError
        with self.assertRaises(ValueError):
            grid_search.predict(self.X)


class TestCreateTimeSeriesCV(unittest.TestCase):
    """Tests for create_time_series_cv factory function."""
    
    def test_valid_method(self):
        """Test with valid method."""
        cv = create_time_series_cv("expanding_window", n_splits=3, test_size=30)
        
        self.assertIsInstance(cv, TimeSeriesCV)
        self.assertEqual(cv.method, CVMethod.EXPANDING_WINDOW)
        self.assertEqual(cv.n_splits, 3)
        self.assertEqual(cv.test_size, 30)
    
    def test_invalid_method(self):
        """Test with invalid method."""
        with self.assertRaises(ValueError):
            create_time_series_cv("invalid_method")
    
    def test_with_kwargs(self):
        """Test with additional kwargs."""
        cv = create_time_series_cv(
            "sliding_window",
            n_splits=3,
            test_size=30,
            gap=5,
            embargo=10,
            allow_overlap=True
        )
        
        self.assertEqual(cv.method, CVMethod.SLIDING_WINDOW)
        self.assertEqual(cv.gap, 5)
        self.assertEqual(cv.embargo, 10)
        self.assertTrue(cv.allow_overlap)


if __name__ == "__main__":
    unittest.main()