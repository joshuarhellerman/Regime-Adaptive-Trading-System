"""
model_validator.py - Model Validation Framework

This module provides comprehensive validation for ML models before deployment,
including performance validation, stability testing, and compliance checks.
"""

import logging
import numpy as np
import pandas as pd
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ValidationMetric(Enum):
    """Standard metrics for model validation"""
    MSE = "mse"                  # Mean Squared Error
    RMSE = "rmse"                # Root Mean Squared Error
    MAE = "mae"                  # Mean Absolute Error
    R2 = "r2"                    # R-squared
    ACCURACY = "accuracy"        # Classification Accuracy
    PRECISION = "precision"      # Precision
    RECALL = "recall"            # Recall
    F1 = "f1"                    # F1 Score
    ROC_AUC = "roc_auc"          # ROC AUC
    PR_AUC = "pr_auc"            # Precision-Recall AUC
    SHARPE = "sharpe"            # Sharpe Ratio
    SORTINO = "sortino"          # Sortino Ratio
    MAX_DRAWDOWN = "max_drawdown"  # Maximum Drawdown
    CALMAR = "calmar"            # Calmar Ratio
    STABILITY = "stability"      # Model Stability
    LATENCY = "latency"          # Inference Latency


class ValidationType(Enum):
    """Types of model validation"""
    PERFORMANCE = "performance"    # Performance metrics validation
    STABILITY = "stability"        # Stability and robustness validation
    COMPLIANCE = "compliance"      # Regulatory and policy compliance
    BIAS = "bias"                  # Bias and fairness validation
    INFERENCE = "inference"        # Inference performance validation


@dataclass
class ValidationResult:
    """Result of a model validation"""
    validation_type: ValidationType
    passed: bool
    metrics: Dict[str, float]
    threshold_metrics: Dict[str, float]
    details: Dict[str, Any] = field(default_factory=dict)
    message: str = ""


class ModelValidator:
    """
    Comprehensive model validation framework for ensuring models meet
    performance, stability, and compliance standards before deployment.
    """

    def __init__(self,
                 config_path: Optional[str] = None,
                 validation_data_path: Optional[str] = None,
                 performance_thresholds: Optional[Dict[str, float]] = None,
                 stability_thresholds: Optional[Dict[str, float]] = None,
                 latency_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize the model validator.

        Args:
            config_path: Path to validation configuration file
            validation_data_path: Path to validation datasets
            performance_thresholds: Default performance metric thresholds
            stability_thresholds: Default stability metric thresholds
            latency_thresholds: Default latency metric thresholds
        """
        self.config_path = config_path
        self.validation_data_path = validation_data_path

        # Default thresholds if not provided
        self.performance_thresholds = performance_thresholds or {
            # Regression metrics
            "mse": 0.1,
            "rmse": 0.3,
            "mae": 0.2,
            "r2": 0.7,

            # Classification metrics
            "accuracy": 0.7,
            "precision": 0.7,
            "recall": 0.7,
            "f1": 0.7,
            "roc_auc": 0.75,
            "pr_auc": 0.7,

            # Trading metrics
            "sharpe": 1.0,
            "sortino": 1.0,
            "max_drawdown": 0.2,  # 20% maximum drawdown
            "calmar": 0.5
        }

        self.stability_thresholds = stability_thresholds or {
            "feature_importance_shift": 0.3,  # Max allowed change in feature importance
            "prediction_shift": 0.15,          # Max allowed shift in prediction distribution
            "coef_of_variation": 0.2,         # Max coefficient of variation across runs
            "cross_validation_std": 0.1       # Max standard deviation in cross-validation
        }

        self.latency_thresholds = latency_thresholds or {
            "mean_latency_ms": 50.0,          # Mean latency in milliseconds
            "p95_latency_ms": 100.0,          # 95th percentile latency in milliseconds
            "p99_latency_ms": 200.0,          # 99th percentile latency in milliseconds
            "max_latency_ms": 500.0           # Maximum allowed latency in milliseconds
        }

        # Model type specific configurations
        self.model_type_configs = {}

        # Load configuration if provided
        if config_path:
            self._load_config()

        logger.info("Model validator initialized")

    def _load_config(self):
        """Load configuration from file"""
        if not os.path.exists(self.config_path):
            logger.warning(f"Configuration file not found: {self.config_path}")
            return

        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            # Update default thresholds
            if "performance_thresholds" in config:
                self.performance_thresholds.update(config["performance_thresholds"])

            if "stability_thresholds" in config:
                self.stability_thresholds.update(config["stability_thresholds"])

            if "latency_thresholds" in config:
                self.latency_thresholds.update(config["latency_thresholds"])

            # Load model type specific configurations
            if "model_types" in config:
                self.model_type_configs = config["model_types"]

            logger.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")

    def validate_model(self,
                      model: Any,
                      model_type: str,
                      validation_data: Optional[Any] = None,
                      metadata: Optional[Dict[str, Any]] = None,
                      validation_types: Optional[List[ValidationType]] = None) -> Dict[str, Any]:
        """
        Validate a model before deployment.

        Args:
            model: Model object to validate
            model_type: Type of the model (e.g., 'regime', 'alpha', 'portfolio')
            validation_data: Data for validation, or None to use default
            metadata: Additional metadata including performance metrics
            validation_types: Types of validation to perform

        Returns:
            Dictionary with validation results
        """
        # Default to all validation types if not specified
        if validation_types is None:
            validation_types = [
                ValidationType.PERFORMANCE,
                ValidationType.STABILITY,
                ValidationType.INFERENCE
            ]

        # Get model type specific configuration
        model_config = self.model_type_configs.get(model_type, {})

        # Initialize results
        results = {}
        all_passed = True

        # Performance validation
        if ValidationType.PERFORMANCE in validation_types:
            performance_result = self._validate_performance(
                model, model_type, validation_data, metadata, model_config
            )
            results["performance"] = performance_result
            all_passed = all_passed and performance_result.passed

        # Stability validation
        if ValidationType.STABILITY in validation_types:
            stability_result = self._validate_stability(
                model, model_type, validation_data, metadata, model_config
            )
            results["stability"] = stability_result
            all_passed = all_passed and stability_result.passed

        # Inference performance validation
        if ValidationType.INFERENCE in validation_types:
            inference_result = self._validate_inference(
                model, model_type, validation_data, metadata, model_config
            )
            results["inference"] = inference_result
            all_passed = all_passed and inference_result.passed

        # Bias and fairness validation (if supported)
        if ValidationType.BIAS in validation_types and hasattr(self, '_validate_bias'):
            bias_result = self._validate_bias(
                model, model_type, validation_data, metadata, model_config
            )
            results["bias"] = bias_result
            all_passed = all_passed and bias_result.passed

        # Compliance validation (if supported)
        if ValidationType.COMPLIANCE in validation_types and hasattr(self, '_validate_compliance'):
            compliance_result = self._validate_compliance(
                model, model_type, metadata, model_config
            )
            results["compliance"] = compliance_result
            all_passed = all_passed and compliance_result.passed

        # Generate final result
        final_result = {
            "passed": all_passed,
            "results": {k: v.__dict__ for k, v in results.items()},
            "model_type": model_type,
            "timestamp": time.time(),
            "message": "All validations passed" if all_passed else "One or more validations failed"
        }

        logger.info(f"Model validation for {model_type}: {'PASSED' if all_passed else 'FAILED'}")
        return final_result

    def _validate_performance(self,
                             model: Any,
                             model_type: str,
                             validation_data: Optional[Any],
                             metadata: Optional[Dict[str, Any]],
                             model_config: Dict[str, Any]) -> ValidationResult:
        """
        Validate model performance metrics.

        Args:
            model: Model object
            model_type: Model type
            validation_data: Validation data
            metadata: Model metadata
            model_config: Model type specific configuration

        Returns:
            Validation result
        """
        # Get performance metrics from metadata if provided
        if metadata and "performance" in metadata:
            performance_metrics = metadata["performance"]
        elif metadata and "validation_results" in metadata and "metrics" in metadata["validation_results"]:
            performance_metrics = metadata["validation_results"]["metrics"]
        else:
            performance_metrics = {}

            # If no metrics provided and we have validation data, calculate metrics
            if validation_data:
                try:
                    # Try to extract features and targets from validation data
                    if isinstance(validation_data, tuple) and len(validation_data) == 2:
                        X_val, y_val = validation_data
                    elif hasattr(validation_data, 'features') and hasattr(validation_data, 'target'):
                        X_val = validation_data.features
                        y_val = validation_data.target
                    else:
                        X_val = validation_data
                        y_val = None

                    # Make predictions
                    if hasattr(model, 'predict'):
                        y_pred = model.predict(X_val)

                        # Calculate metrics if target is available
                        if y_val is not None:
                            # Calculate basic metrics
                            if len(y_val.shape) == 1 or y_val.shape[1] == 1:  # Regression or binary classification
                                performance_metrics.update(self._calculate_regression_metrics(y_val, y_pred))
                            else:  # Multi-class classification
                                performance_metrics.update(self._calculate_classification_metrics(y_val, y_pred))
                except Exception as e:
                    logger.error(f"Error calculating performance metrics: {str(e)}")

        # Get thresholds
        thresholds = {}

        # Start with default thresholds
        for metric in self.performance_thresholds:
            if metric in performance_metrics:
                thresholds[metric] = self.performance_thresholds[metric]

        # Override with model type specific thresholds
        if "performance_thresholds" in model_config:
            for metric, threshold in model_config["performance_thresholds"].items():
                if metric in performance_metrics:
                    thresholds[metric] = threshold

        # Check if metrics meet thresholds
        failed_metrics = {}

        for metric, threshold in thresholds.items():
            if metric in performance_metrics:
                metric_value = performance_metrics[metric]

                # Handle metrics where lower is better
                if metric in ["mse", "rmse", "mae", "max_drawdown"]:
                    if metric_value > threshold:
                        failed_metrics[metric] = (metric_value, threshold)
                # Handle metrics where higher is better
                else:
                    if metric_value < threshold:
                        failed_metrics[metric] = (metric_value, threshold)

        # Generate result
        passed = len(failed_metrics) == 0

        # Generate message
        if passed:
            message = "Performance validation passed"
        else:
            message = f"Performance validation failed: {len(failed_metrics)} metrics below threshold"

        return ValidationResult(
            validation_type=ValidationType.PERFORMANCE,
            passed=passed,
            metrics=performance_metrics,
            threshold_metrics=thresholds,
            details={"failed_metrics": failed_metrics},
            message=message
        )

    def _validate_stability(self,
                           model: Any,
                           model_type: str,
                           validation_data: Optional[Any],
                           metadata: Optional[Dict[str, Any]],
                           model_config: Dict[str, Any]) -> ValidationResult:
        """
        Validate model stability.

        Args:
            model: Model object
            model_type: Model type
            validation_data: Validation data
            metadata: Model metadata
            model_config: Model type specific configuration

        Returns:
            Validation result
        """
        # Initialize stability metrics
        stability_metrics = {}

        # Get stability metrics from metadata if provided
        if metadata and "stability" in metadata:
            stability_metrics = metadata["stability"]
        elif metadata and "validation_results" in metadata and "stability" in metadata["validation_results"]:
            stability_metrics = metadata["validation_results"]["stability"]

        # If validation data is provided, calculate additional stability metrics
        if validation_data and len(stability_metrics) == 0:
            try:
                # Try to extract features from validation data
                if isinstance(validation_data, tuple) and len(validation_data) == 2:
                    X_val, _ = validation_data
                elif hasattr(validation_data, 'features'):
                    X_val = validation_data.features
                else:
                    X_val = validation_data

                # Check if model has feature importances
                if hasattr(model, 'feature_importances_'):
                    stability_metrics["has_feature_importances"] = True

                # Make predictions multiple times to check stability
                if hasattr(model, 'predict'):
                    predictions = []
                    latencies = []

                    # Run multiple predictions to measure stability
                    for _ in range(5):
                        start_time = time.time()
                        pred = model.predict(X_val)
                        end_time = time.time()

                        predictions.append(pred)
                        latencies.append((end_time - start_time) * 1000)  # Convert to ms

                    # Calculate prediction stability
                    if len(predictions) > 1:
                        # Calculate coefficient of variation across runs
                        prediction_means = np.mean(predictions, axis=1)
                        prediction_std = np.std(prediction_means)
                        prediction_mean = np.mean(prediction_means)

                        if prediction_mean > 0:
                            stability_metrics["coef_of_variation"] = prediction_std / prediction_mean

                        # Calculate latency statistics
                        stability_metrics["mean_latency_ms"] = np.mean(latencies)
                        stability_metrics["std_latency_ms"] = np.std(latencies)
                        stability_metrics["min_latency_ms"] = np.min(latencies)
                        stability_metrics["max_latency_ms"] = np.max(latencies)
            except Exception as e:
                logger.error(f"Error calculating stability metrics: {str(e)}")

        # Get thresholds
        thresholds = {}

        # Start with default thresholds
        for metric in self.stability_thresholds:
            if metric in stability_metrics:
                thresholds[metric] = self.stability_thresholds[metric]

        # Override with model type specific thresholds
        if "stability_thresholds" in model_config:
            for metric, threshold in model_config["stability_thresholds"].items():
                if metric in stability_metrics:
                    thresholds[metric] = threshold

        # Check if metrics meet thresholds
        failed_metrics = {}

        for metric, threshold in thresholds.items():
            if metric in stability_metrics:
                metric_value = stability_metrics[metric]

                # For all stability metrics, lower is better
                if metric_value > threshold:
                    failed_metrics[metric] = (metric_value, threshold)

        # Generate result
        passed = len(failed_metrics) == 0

        # Generate message
        if passed:
            message = "Stability validation passed"
        else:
            message = f"Stability validation failed: {len(failed_metrics)} metrics below threshold"

        return ValidationResult(
            validation_type=ValidationType.STABILITY,
            passed=passed,
            metrics=stability_metrics,
            threshold_metrics=thresholds,
            details={"failed_metrics": failed_metrics},
            message=message
        )

    def _validate_inference(self,
                           model: Any,
                           model_type: str,
                           validation_data: Optional[Any],
                           metadata: Optional[Dict[str, Any]],
                           model_config: Dict[str, Any]) -> ValidationResult:
        """
        Validate model inference performance.

        Args:
            model: Model object
            model_type: Model type
            validation_data: Validation data
            metadata: Model metadata
            model_config: Model type specific configuration

        Returns:
            Validation result
        """
        # Initialize inference metrics
        inference_metrics = {}

        # Get inference metrics from metadata if provided
        if metadata and "inference" in metadata:
            inference_metrics = metadata["inference"]
        elif metadata and "validation_results" in metadata and "inference" in metadata["validation_results"]:
            inference_metrics = metadata["validation_results"]["inference"]

        # If validation data is provided, calculate additional inference metrics
        if validation_data and len(inference_metrics) == 0:
            try:
                # Try to extract features from validation data
                if isinstance(validation_data, tuple) and len(validation_data) == 2:
                    X_val, _ = validation_data
                elif hasattr(validation_data, 'features'):
                    X_val = validation_data.features
                else:
                    X_val = validation_data

                # Make predictions to measure latency
                if hasattr(model, 'predict'):
                    latencies = []
                    batch_latencies = []

                    # Measure single instance prediction latency
                    single_instance = X_val[0:1]
                    for _ in range(20):  # Multiple runs for reliable measurement
                        start_time = time.time()
                        _ = model.predict(single_instance)
                        end_time = time.time()

                        latencies.append((end_time - start_time) * 1000)  # Convert to ms

                    # Measure batch prediction latency
                    batch_size = min(100, len(X_val))
                    batch = X_val[:batch_size]
                    for _ in range(5):  # Fewer runs for batch
                        start_time = time.time()
                        _ = model.predict(batch)
                        end_time = time.time()

                        batch_latencies.append((end_time - start_time) * 1000)  # Convert to ms

                    # Calculate latency statistics
                    inference_metrics["mean_latency_ms"] = np.mean(latencies)
                    inference_metrics["p95_latency_ms"] = np.percentile(latencies, 95)
                    inference_metrics["p99_latency_ms"] = np.percentile(latencies, 99)
                    inference_metrics["max_latency_ms"] = np.max(latencies)

                    # Batch metrics
                    inference_metrics["batch_mean_latency_ms"] = np.mean(batch_latencies)
                    inference_metrics["batch_size"] = batch_size
                    inference_metrics["latency_per_instance_ms"] = inference_metrics["batch_mean_latency_ms"] / batch_size
            except Exception as e:
                logger.error(f"Error calculating inference metrics: {str(e)}")

        # Get thresholds
        thresholds = {}

        # Start with default thresholds
        for metric in self.latency_thresholds:
            if metric in inference_metrics:
                thresholds[metric] = self.latency_thresholds[metric]

        # Override with model type specific thresholds
        if "latency_thresholds" in model_config:
            for metric, threshold in model_config["latency_thresholds"].items():
                if metric in inference_metrics:
                    thresholds[metric] = threshold

        # Check if metrics meet thresholds
        failed_metrics = {}

        for metric, threshold in thresholds.items():
            if metric in inference_metrics:
                metric_value = inference_metrics[metric]

                # For all latency metrics, lower is better
                if metric_value > threshold:
                    failed_metrics[metric] = (metric_value, threshold)

        # Generate result
        passed = len(failed_metrics) == 0

        # Generate message
        if passed:
            message = "Inference validation passed"
        else:
            message = f"Inference validation failed: {len(failed_metrics)} metrics exceed threshold"

        return ValidationResult(
            validation_type=ValidationType.INFERENCE,
            passed=passed,
            metrics=inference_metrics,
            threshold_metrics=thresholds,
            details={"failed_metrics": failed_metrics},
            message=message
        )

    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate metrics for regression problems.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        metrics = {}

        # Basic regression metrics
        metrics["mse"] = float(mean_squared_error(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))

        return metrics

    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate metrics for classification problems.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        metrics = {}

        # For binary classification
        if len(np.unique(y_true)) == 2:
            # Basic classification metrics
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            metrics["precision"] = float(precision_score(y_true, y_pred))
            metrics["recall"] = float(recall_score(y_true, y_pred))
            metrics["f1"] = float(f1_score(y_true, y_pred))

            # Try to get probabilities for ROC AUC
            try:
                if hasattr(y_pred, 'predict_proba'):
                    y_proba = y_pred.predict_proba(y_true)[:, 1]
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            except:
                pass
        else:
            # Multi-class metrics
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

            # Macro-averaged metrics
            metrics["precision_macro"] = float(precision_score(y_true, y_pred, average='macro'))
            metrics["recall_macro"] = float(recall_score(y_true, y_pred, average='macro'))
            metrics["f1_macro"] = float(f1_score(y_true, y_pred, average='macro'))

            # Weighted-averaged metrics
            metrics["precision_weighted"] = float(precision_score(y_true, y_pred, average='weighted'))
            metrics["recall_weighted"] = float(recall_score(y_true, y_pred, average='weighted'))
            metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average='weighted'))

        return metrics


# Factory function for creating model validators
def create_model_validator(config_path: Optional[str] = None) -> ModelValidator:
    """
    Create a model validator instance.

    Args:
        config_path: Path to validation configuration file

    Returns:
        ModelValidator instance
    """
    return ModelValidator(config_path=config_path)