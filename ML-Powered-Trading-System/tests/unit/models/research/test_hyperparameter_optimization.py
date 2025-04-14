import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
import pytest

from models.research.hyperparameter_optimization import (
    HyperparameterOptimizer,
    OptimizationMethod,
    Parameter,
    OptimizationResult,
    create_optimizer,
    create_parameter
)
from models.research.cross_validation import TimeSeriesCV

class TestHyperparameterOptimization(unittest.TestCase):
    """Test cases for hyperparameter optimization module."""

    def setUp(self):
        """Set up test data and mocks."""
        # Create simple test data
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.random.randn(100)
        
        # Create a mock estimator class
        class MockEstimator:
            def __init__(self):
                self.param1 = 1
                self.param2 = 0.1
                self.fit_called = False
                self.predict_called = False
            
            def fit(self, X, y):
                self.fit_called = True
                return self
            
            def predict(self, X):
                self.predict_called = True
                return np.zeros(len(X))
        
        self.MockEstimator = MockEstimator
        
        # Create parameters for testing
        self.parameters = [
            Parameter(name="param1", type="int", values=(1, 10)),
            Parameter(name="param2", type="float", values=(0.1, 1.0), log_scale=True)
        ]
        
        # Create a mock cross-validator
        self.mock_cv = MagicMock(spec=TimeSeriesCV)
        self.mock_cv.split.return_value = [
            (np.arange(80), np.arange(80, 100)),
            (np.arange(60), np.arange(60, 80))
        ]
        self.mock_cv.get_train_test_data.side_effect = lambda split, X, y: {
            "X_train": X[split[0]], 
            "X_test": X[split[1]],
            "y_train": y[split[0]], 
            "y_test": y[split[1]]
        }

    def test_create_optimizer(self):
        """Test creation of optimizer with different methods."""
        # Test valid methods
        for method in [m.value for m in OptimizationMethod]:
            optimizer = create_optimizer(method=method, n_iterations=50)
            self.assertIsInstance(optimizer, HyperparameterOptimizer)
            self.assertEqual(optimizer.method, OptimizationMethod(method))
            self.assertEqual(optimizer.n_iterations, 50)
        
        # Test invalid method
        with self.assertRaises(ValueError):
            create_optimizer(method="invalid_method")

    def test_create_parameter(self):
        """Test creation of parameter objects."""
        # Test categorical parameter
        cat_param = create_parameter(
            name="cat_param", 
            type="categorical", 
            values=["option1", "option2", "option3"]
        )
        self.assertEqual(cat_param.name, "cat_param")
        self.assertEqual(cat_param.type, "categorical")
        self.assertEqual(cat_param.values, ["option1", "option2", "option3"])
        
        # Test numerical parameter with log scale
        num_param = create_parameter(
            name="num_param", 
            type="float", 
            values=(0.1, 10.0),
            log_scale=True,
            step=0.1
        )
        self.assertEqual(num_param.name, "num_param")
        self.assertEqual(num_param.type, "float")
        self.assertEqual(num_param.values, (0.1, 10.0))
        self.assertTrue(num_param.log_scale)
        self.assertEqual(num_param.step, 0.1)
        
        # Test boolean parameter with dependencies
        bool_param = create_parameter(
            name="bool_param", 
            type="boolean", 
            values=[True, False],
            depends_on={"other_param": "value"}
        )
        self.assertEqual(bool_param.name, "bool_param")
        self.assertEqual(bool_param.type, "boolean")
        self.assertEqual(bool_param.values, [True, False])
        self.assertEqual(bool_param.depends_on, {"other_param": "value"})

    @patch("models.research.hyperparameter_optimization.SKOPT_AVAILABLE", False)
    def test_unavailable_methods_fallback(self):
        """Test fallback when selected method is not available."""
        optimizer = HyperparameterOptimizer(
            method=OptimizationMethod.BAYESIAN_OPTIMIZATION,
            n_iterations=10
        )
        # Should fall back to random search
        self.assertEqual(optimizer.method, OptimizationMethod.RANDOM_SEARCH)

    def test_format_parameters(self):
        """Test parameter formatting for optimization methods."""
        optimizer = HyperparameterOptimizer(
            method=OptimizationMethod.RANDOM_SEARCH,
            n_iterations=10
        )
        
        parameters = [
            Parameter(name="p1", type="int", values=(1, 10)),
            Parameter(name="p2", type="float", values=(0.1, 1.0), log_scale=True),
            Parameter(name="p3", type="categorical", values=["a", "b", "c"]),
            Parameter(name="p4", type="boolean", values=None)
        ]
        
        params_dict = optimizer._format_parameters(parameters)
        
        self.assertEqual(len(params_dict), 4)
        self.assertEqual(params_dict["p1"]["type"], "int")
        self.assertEqual(params_dict["p1"]["range"], (1, 10))
        self.assertEqual(params_dict["p2"]["log_scale"], True)
        self.assertEqual(params_dict["p3"]["values"], ["a", "b", "c"])
        self.assertEqual(params_dict["p4"]["values"], [True, False])

    @patch("models.research.hyperparameter_optimization.ParameterSampler")
    def test_random_search(self, mock_sampler):
        """Test random search optimization."""
        # Set up the mock parameter sampler
        mock_sampler.return_value = [
            {"param1": 5, "param2": 0.5},
            {"param1": 7, "param2": 0.2}
        ]
        
        optimizer = HyperparameterOptimizer(
            method=OptimizationMethod.RANDOM_SEARCH,
            cv=self.mock_cv,
            n_iterations=2,
            verbose=0
        )
        
        # Mock the _compute_score method
        optimizer._compute_score = MagicMock(return_value=0.85)
        
        # Mock the _clone_estimator method
        optimizer._clone_estimator = MagicMock(return_value=self.MockEstimator())
        
        # Run optimization
        result = optimizer._random_search(
            self.MockEstimator(), 
            {
                "param1": {"type": "int", "range": (1, 10)},
                "param2": {"type": "float", "range": (0.1, 1.0), "log_scale": True}
            },
            self.X, 
            self.y
        )
        
        # Verify results
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.method, OptimizationMethod.RANDOM_SEARCH)
        self.assertEqual(len(result.all_results), 2)
        self.assertTrue(0.8 <= result.best_score <= 0.9)  # Since we mocked _compute_score to return 0.85

    @patch("models.research.hyperparameter_optimization.OPTUNA_AVAILABLE", True)
    @patch("models.research.hyperparameter_optimization.optuna")
    def test_tpe_optimization(self, mock_optuna):
        """Test TPE optimization."""
        # Set up mock for optuna
        mock_study = MagicMock()
        mock_study.best_trial.params = {"param1": 5, "param2": 0.5}
        mock_study.best_trial.value = 0.9
        mock_study.trials = [
            MagicMock(state="COMPLETE", params={"param1": 5, "param2": 0.5}, value=0.9),
            MagicMock(state="COMPLETE", params={"param1": 3, "param2": 0.3}, value=0.8)
        ]
        mock_optuna.trial.TrialState.COMPLETE = "COMPLETE"
        mock_optuna.create_study.return_value = mock_study
        
        optimizer = HyperparameterOptimizer(
            method=OptimizationMethod.TPE,
            cv=self.mock_cv,
            n_iterations=2,
            verbose=0
        )
        
        # Mock the _compute_score method
        optimizer._compute_score = MagicMock(return_value=0.85)
        
        # Mock the _clone_estimator method
        optimizer._clone_estimator = MagicMock(return_value=self.MockEstimator())
        
        # Run optimization
        result = optimizer._tpe_optimization(
            self.MockEstimator(), 
            {
                "param1": {"type": "int", "range": (1, 10)},
                "param2": {"type": "float", "range": (0.1, 1.0), "log_scale": True}
            },
            self.X, 
            self.y
        )
        
        # Verify results
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.method, OptimizationMethod.TPE)
        self.assertEqual(result.best_params, {"param1": 5, "param2": 0.5})
        self.assertEqual(result.best_score, 0.9)

    @patch("models.research.hyperparameter_optimization.SKOPT_AVAILABLE", True)
    @patch("models.research.hyperparameter_optimization.BayesSearchCV")
    def test_bayesian_optimization(self, mock_bayes_cv):
        """Test Bayesian optimization."""
        # Set up mock for BayesSearchCV
        mock_bayes = MagicMock()
        mock_bayes.best_params_ = {"param1": 5, "param2": 0.5}
        mock_bayes.best_score_ = 0.9
        mock_bayes.best_index_ = 0
        mock_bayes.cv_results_ = {
            "params": [
                {"param1": 5, "param2": 0.5},
                {"param1": 3, "param2": 0.3}
            ],
            "mean_test_score": [0.9, 0.8],
            "std_test_score": [0.05, 0.06],
            "split_test_scores": [[0.85, 0.95], [0.75, 0.85]]
        }
        mock_bayes_cv.return_value = mock_bayes
        
        optimizer = HyperparameterOptimizer(
            method=OptimizationMethod.BAYESIAN_OPTIMIZATION,
            cv=self.mock_cv,
            n_iterations=2,
            verbose=0
        )
        
        # Run optimization
        result = optimizer._bayesian_optimization(
            self.MockEstimator(), 
            {
                "param1": {"type": "int", "range": (1, 10)},
                "param2": {"type": "float", "range": (0.1, 1.0), "log_scale": True}
            },
            self.X, 
            self.y
        )
        
        # Verify results
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.method, OptimizationMethod.BAYESIAN_OPTIMIZATION)
        self.assertEqual(result.best_params, {"param1": 5, "param2": 0.5})
        self.assertEqual(result.best_score, 0.9)
        self.assertEqual(result.cv_scores, [0.85, 0.95])

    @patch("concurrent.futures.ProcessPoolExecutor")
    def test_parallel_execution(self, mock_executor):
        """Test parallel execution in random search."""
        # Set up the mock executor
        mock_future = MagicMock()
        mock_future.map.return_value = [
            {"params": {"param1": 5, "param2": 0.5}, "mean_test_score": 0.9, "std_test_score": 0.05, "scores": [0.85, 0.95]},
            {"params": {"param1": 3, "param2": 0.3}, "mean_test_score": 0.8, "std_test_score": 0.06, "scores": [0.75, 0.85]}
        ]
        mock_executor.return_value.__enter__.return_value = mock_future
        
        optimizer = HyperparameterOptimizer(
            method=OptimizationMethod.RANDOM_SEARCH,
            cv=self.mock_cv,
            n_iterations=2,
            n_jobs=2,
            verbose=0
        )
        
        # Set up parameter sampler mock
        with patch("models.research.hyperparameter_optimization.ParameterSampler") as mock_sampler:
            mock_sampler.return_value = [
                {"param1": 5, "param2": 0.5},
                {"param1": 3, "param2": 0.3}
            ]
            
            # Run optimization
            result = optimizer._random_search(
                self.MockEstimator(), 
                {
                    "param1": {"type": "int", "range": (1, 10)},
                    "param2": {"type": "float", "range": (0.1, 1.0), "log_scale": True}
                },
                self.X, 
                self.y
            )
        
        # Verify results
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.method, OptimizationMethod.RANDOM_SEARCH)
        self.assertEqual(len(result.all_results), 2)
        self.assertEqual(result.best_params, {"param1": 5, "param2": 0.5})
        self.assertEqual(result.best_score, 0.9)

    def test_integrate_with_progress_callback(self):
        """Test integration with progress callback."""
        # Create a mock progress callback
        mock_callback = MagicMock()
        
        optimizer = HyperparameterOptimizer(
            method=OptimizationMethod.RANDOM_SEARCH,
            cv=self.mock_cv,
            n_iterations=2,
            verbose=0,
            progress_callback=mock_callback
        )
        
        # Set up parameter sampler mock
        with patch("models.research.hyperparameter_optimization.ParameterSampler") as mock_sampler:
            mock_sampler.return_value = [
                {"param1": 5, "param2": 0.5},
                {"param1": 3, "param2": 0.3}
            ]
            
            # Mock the _compute_score method
            optimizer._compute_score = MagicMock(return_value=0.85)
            
            # Mock the _clone_estimator method
            optimizer._clone_estimator = MagicMock(return_value=self.MockEstimator())
            
            # Run optimization
            result = optimizer._random_search(
                self.MockEstimator(), 
                {
                    "param1": {"type": "int", "range": (1, 10)},
                    "param2": {"type": "float", "range": (0.1, 1.0), "log_scale": True}
                },
                self.X, 
                self.y
            )
        
        # Verify callback was called with progress updates
        self.assertTrue(mock_callback.called)
        # Progress callback should be called twice (once for each parameter set)
        self.assertEqual(mock_callback.call_count, 2)

    def test_optimize_method(self):
        """Test the main optimize method."""
        optimizer = HyperparameterOptimizer(
            method=OptimizationMethod.RANDOM_SEARCH,
            cv=self.mock_cv,
            n_iterations=2,
            verbose=0
        )
        
        # Mock the optimization method
        optimizer._random_search = MagicMock(return_value=OptimizationResult(
            best_params={"param1": 5, "param2": 0.5},
            best_score=0.9,
            all_results=[
                {"params": {"param1": 5, "param2": 0.5}, "mean_test_score": 0.9, "std_test_score": 0.05, "rank": 1},
                {"params": {"param1": 3, "param2": 0.3}, "mean_test_score": 0.8, "std_test_score": 0.06, "rank": 2}
            ],
            optimization_time=0,
            method=OptimizationMethod.RANDOM_SEARCH,
            model_type="MockEstimator",
            cv_scores=[0.85, 0.95]
        ))
        
        # Mock the clone estimator
        optimizer._clone_estimator = MagicMock(return_value=self.MockEstimator())
        
        # Run optimization
        result = optimizer.optimize(
            self.MockEstimator(),
            self.parameters,
            self.X,
            self.y
        )
        
        # Verify results
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.best_params, {"param1": 5, "param2": 0.5})
        self.assertEqual(result.best_score, 0.9)
        self.assertGreater(result.optimization_time, 0)  # Time should be updated

    @patch("models.research.hyperparameter_optimization.OPTUNA_AVAILABLE", True)
    @patch("models.research.hyperparameter_optimization.optuna")
    def test_hyperband_optimization(self, mock_optuna):
        """Test Hyperband optimization."""
        # Set up mock for optuna
        mock_study = MagicMock()
        mock_study.best_trial.params = {"param1": 5, "param2": 0.5}
        mock_study.best_trial.value = 0.9
        mock_study.trials = [
            MagicMock(state="COMPLETE", params={"param1": 5, "param2": 0.5}, value=0.9),
            MagicMock(state="COMPLETE", params={"param1": 3, "param2": 0.3}, value=0.8)
        ]
        mock_optuna.trial.TrialState.COMPLETE = "COMPLETE"
        mock_optuna.create_study.return_value = mock_study
        
        optimizer = HyperparameterOptimizer(
            method=OptimizationMethod.HYPERBAND,
            cv=self.mock_cv,
            n_iterations=2,
            verbose=0
        )
        
        # Mock the _compute_score method
        optimizer._compute_score = MagicMock(return_value=0.85)
        
        # Mock the _clone_estimator method
        optimizer._clone_estimator = MagicMock(return_value=self.MockEstimator())
        
        # Run optimization
        result = optimizer._hyperband_optimization(
            self.MockEstimator(), 
            {
                "param1": {"type": "int", "range": (1, 10)},
                "param2": {"type": "float", "range": (0.1, 1.0), "log_scale": True}
            },
            self.X, 
            self.y
        )
        
        # Verify results
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.method, OptimizationMethod.HYPERBAND)
        self.assertEqual(result.best_params, {"param1": 5, "param2": 0.5})
        self.assertEqual(result.best_score, 0.9)

    @pytest.mark.skipif("'deap' not in sys.modules")
    def test_evolutionary_optimization(self):
        """Test evolutionary optimization (skipped if DEAP not available)."""
        optimizer = HyperparameterOptimizer(
            method=OptimizationMethod.EVOLUTIONARY,
            cv=self.mock_cv,
            n_iterations=10,
            verbose=0
        )
        
        # Only test if DEAP is available
        try:
            import deap
            # Mock the _compute_score method
            optimizer._compute_score = MagicMock(return_value=0.85)
            
            # Mock the _clone_estimator method
            optimizer._clone_estimator = MagicMock(return_value=self.MockEstimator())
            
            # Run optimization
            result = optimizer._evolutionary_optimization(
                self.MockEstimator(), 
                {
                    "param1": {"type": "int", "range": (1, 10)},
                    "param2": {"type": "float", "range": (0.1, 1.0), "log_scale": True}
                },
                self.X, 
                self.y
            )
            
            # Verify results
            self.assertIsInstance(result, OptimizationResult)
            self.assertEqual(result.method, OptimizationMethod.EVOLUTIONARY)
        except ImportError:
            # Skip test if DEAP not available
            self.skipTest("DEAP library not available")

    def test_compute_score(self):
        """Test score computation for different metrics."""
        optimizer = HyperparameterOptimizer(
            method=OptimizationMethod.RANDOM_SEARCH,
            cv=self.mock_cv,
            n_iterations=2
        )
        
        # Create a mock model that always predicts zeros
        mock_model = MagicMock()
        mock_model.predict.return_value = np.zeros(10)
        
        # Test different scoring metrics
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        
        # MSE
        optimizer.scoring = 'neg_mean_squared_error'
        score = optimizer._compute_score(mock_model, None, y_true)
        self.assertEqual(score, -0.5)  # Mean squared error of (0-0)^2, (1-0)^2, ...
        
        # MAE
        optimizer.scoring = 'neg_mean_absolute_error'
        score = optimizer._compute_score(mock_model, None, y_true)
        self.assertEqual(score, -0.5)  # Mean absolute error of |0-0|, |1-0|, ...
        
        # Accuracy
        optimizer.scoring = 'accuracy'
        score = optimizer._compute_score(mock_model, None, y_true)
        self.assertEqual(score, 0.5)  # 5 out of 10 correct predictions
        
        # Custom function
        optimizer.scoring = lambda y_true, y_pred: np.sum(y_true == y_pred) / len(y_true)
        score = optimizer._compute_score(mock_model, None, y_true)
        self.assertEqual(score, 0.5)
        
        # Invalid scoring
        optimizer.scoring = 'invalid_scoring'
        with self.assertRaises(ValueError):
            optimizer._compute_score(mock_model, None, y_true)

if __name__ == '__main__':
    unittest.main()