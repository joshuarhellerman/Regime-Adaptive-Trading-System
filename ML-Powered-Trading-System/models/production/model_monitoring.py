"""
model_monitoring.py - Drift Detection and Alerting

This module provides monitoring for ML models in production, including
feature drift detection, prediction drift detection, and model performance monitoring.
"""

import logging
import numpy as np
import pandas as pd
import threading
import time
import json
import os
import pickle
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import queue
import statistics
from collections import defaultdict, deque
import scipy.stats as stats

from core.event_bus import EventTopics, Event, get_event_bus, create_event, EventPriority
from core.health_monitor import HealthMonitor, HealthStatus, AlertLevel, AlertCategory
from core.state_manager import StateManager, StateScope
from models.production.model_deployer import ModelDeployer, DeploymentType

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of drift that can be detected"""
    FEATURE_DRIFT = "feature_drift"      # Drift in input features
    PREDICTION_DRIFT = "prediction_drift"  # Drift in model predictions
    PERFORMANCE_DRIFT = "performance_drift"  # Drift in model performance
    CONCEPT_DRIFT = "concept_drift"      # Drift in relationship between features and target
    DATA_INTEGRITY = "data_integrity"    # Issues with data quality or integrity


class DriftSeverity(Enum):
    """Severity levels for drift alerts"""
    NONE = "none"            # No significant drift detected
    LOW = "low"              # Minor drift, worth monitoring
    MEDIUM = "medium"        # Moderate drift, requires attention
    HIGH = "high"            # Severe drift, immediate action needed
    CRITICAL = "critical"    # Critical drift, model may be unreliable


@dataclass
class DriftAlert:
    """Represents a drift alert"""
    model_id: str
    model_type: str
    drift_type: DriftType
    severity: DriftSeverity
    timestamp: float
    metrics: Dict[str, float]
    feature_importances: Optional[Dict[str, float]] = None
    recommendation: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    model_id: str
    model_type: str
    timestamp: float
    metrics: Dict[str, float]
    feature_values: Optional[Dict[str, List[float]]] = None
    prediction_values: Optional[List[float]] = None
    ground_truth_values: Optional[List[float]] = None


class DriftDetectionMethod(Enum):
    """Statistical methods for detecting drift"""
    KS_TEST = "ks_test"                # Kolmogorov-Smirnov test
    CHI_SQUARED = "chi_squared"        # Chi-squared test
    PSI = "psi"                        # Population Stability Index
    JS_DIVERGENCE = "js_divergence"    # Jensen-Shannon divergence
    WASSERSTEIN = "wasserstein"        # Wasserstein distance (earth mover's)
    FEATURE_IMPORTANCE = "feature_importance"  # Monitoring feature importances


class ModelMonitor:
    """
    Monitors ML models in production for feature drift, prediction drift,
    and performance degradation with alerting capabilities.
    """

    def __init__(self,
                 model_deployer: Optional[ModelDeployer] = None,
                 health_monitor: Optional[HealthMonitor] = None,
                 state_manager: Optional[StateManager] = None,
                 event_bus=None,
                 reference_data_dir: str = "data/reference",
                 monitoring_interval: int = 3600,  # 1 hour
                 feature_drift_threshold: float = 0.1,
                 prediction_drift_threshold: float = 0.15,
                 performance_drift_threshold: float = 0.2,
                 min_samples_for_drift: int = 100,
                 default_detection_method: DriftDetectionMethod = DriftDetectionMethod.KS_TEST,
                 enable_auto_retraining: bool = False,
                 max_history_size: int = 10000):
        """
        Initialize the model monitor.

        Args:
            model_deployer: Model deployer instance
            health_monitor: Health monitor instance
            state_manager: State manager instance
            event_bus: Event bus instance
            reference_data_dir: Directory for storing reference data
            monitoring_interval: Monitoring interval in seconds
            feature_drift_threshold: Threshold for feature drift detection
            prediction_drift_threshold: Threshold for prediction drift detection
            performance_drift_threshold: Threshold for performance drift detection
            min_samples_for_drift: Minimum samples required for drift detection
            default_detection_method: Default statistical method for drift detection
            enable_auto_retraining: Whether to trigger automatic retraining on drift
            max_history_size: Maximum number of historical samples to keep
        """
        self.model_deployer = model_deployer
        self.health_monitor = health_monitor
        self.state_manager = state_manager
        self.event_bus = event_bus or get_event_bus()

        # Configuration
        self.reference_data_dir = reference_data_dir
        self.monitoring_interval = monitoring_interval
        self.feature_drift_threshold = feature_drift_threshold
        self.prediction_drift_threshold = prediction_drift_threshold
        self.performance_drift_threshold = performance_drift_threshold
        self.min_samples_for_drift = min_samples_for_drift
        self.default_detection_method = default_detection_method
        self.enable_auto_retraining = enable_auto_retraining
        self.max_history_size = max_history_size

        # Create reference data directory if needed
        os.makedirs(reference_data_dir, exist_ok=True)

        # Model reference data
        self.reference_data = {}

        # Current production data
        self.production_data = {}

        # Historical metrics
        self.historical_metrics = defaultdict(lambda: deque(maxlen=max_history_size))

        # Configuration for each model
        self.model_configs = {}

        # Running state
        self._running = False
        self._lock = threading.RLock()
        self._monitoring_thread = None

        # Register with event bus
        self.event_bus.subscribe(
            EventTopics.MODEL_PREDICTION,
            self._handle_prediction_event
        )

        self.event_bus.subscribe(
            EventTopics.MODEL_DEPLOYED,
            self._handle_model_deployed
        )

        # Register with health monitor
        if self.health_monitor:
            self.health_monitor.register_component("model_monitor", "monitoring")

        # Load existing reference data
        self._load_reference_data()

        logger.info("Model monitor initialized")

    def start(self):
        """Start the model monitor"""
        with self._lock:
            if self._running:
                logger.warning("Model monitor is already running")
                return

            self._running = True

            # Start monitoring thread
            self._monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                name="ModelMonitoringThread",
                daemon=True
            )
            self._monitoring_thread.start()

            logger.info("Model monitor started")

            # Report initial health status
            if self.health_monitor:
                self.health_monitor.update_component_health(
                    "model_monitor",
                    status=HealthStatus.HEALTHY,
                    metrics={
                        "start_time": time.time(),
                        "monitoring_interval": self.monitoring_interval,
                        "monitored_models": len(self.model_configs)
                    }
                )

    def stop(self):
        """Stop the model monitor"""
        with self._lock:
            if not self._running:
                logger.warning("Model monitor is not running")
                return

            self._running = False

            # Wait for monitoring thread to finish
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=5.0)

            logger.info("Model monitor stopped")

    def add_model_reference_data(self,
                                model_id: str,
                                model_type: str,
                                features: Optional[pd.DataFrame] = None,
                                predictions: Optional[np.ndarray] = None,
                                ground_truth: Optional[np.ndarray] = None,
                                metrics: Optional[Dict[str, float]] = None,
                                feature_importances: Optional[Dict[str, float]] = None,
                                detection_config: Optional[Dict[str, Any]] = None):
        """
        Add reference data for a model.

        Args:
            model_id: Model ID
            model_type: Model type
            features: Reference feature values
            predictions: Reference prediction values
            ground_truth: Reference ground truth values
            metrics: Reference performance metrics
            feature_importances: Feature importance values
            detection_config: Configuration for drift detection
        """
        with self._lock:
            # Create reference data entry
            reference_data = {
                "model_id": model_id,
                "model_type": model_type,
                "timestamp": time.time(),
                "feature_stats": self._compute_stats(features) if features is not None else None,
                "prediction_stats": self._compute_stats(predictions) if predictions is not None else None,
                "ground_truth_stats": self._compute_stats(ground_truth) if ground_truth is not None else None,
                "metrics": metrics or {},
                "feature_importances": feature_importances or {},
                "sample_size": len(features) if features is not None else (len(predictions) if predictions is not None else 0)
            }

            # Store reference data
            self.reference_data[model_id] = reference_data

            # Store model configuration
            self.model_configs[model_id] = {
                "model_type": model_type,
                "detection_method": detection_config.get("detection_method", self.default_detection_method) if detection_config else self.default_detection_method,
                "feature_drift_threshold": detection_config.get("feature_drift_threshold", self.feature_drift_threshold) if detection_config else self.feature_drift_threshold,
                "prediction_drift_threshold": detection_config.get("prediction_drift_threshold", self.prediction_drift_threshold) if detection_config else self.prediction_drift_threshold,
                "performance_drift_threshold": detection_config.get("performance_drift_threshold", self.performance_drift_threshold) if detection_config else self.performance_drift_threshold,
                "monitored_features": detection_config.get("monitored_features", []) if detection_config else [],
                "min_samples_for_drift": detection_config.get("min_samples_for_drift", self.min_samples_for_drift) if detection_config else self.min_samples_for_drift
            }

            # Save reference data to disk
            self._save_reference_data(model_id)

            logger.info(f"Added reference data for model {model_id}")

            # Initialize production data structure
            self.production_data[model_id] = {
                "features": [],
                "predictions": [],
                "ground_truth": [],
                "metrics": defaultdict(list),
                "timestamps": [],
                "last_checked": time.time()
            }

    def update_production_data(self,
                              model_id: str,
                              features: Optional[Any] = None,
                              prediction: Optional[Any] = None,
                              ground_truth: Optional[Any] = None,
                              metrics: Optional[Dict[str, float]] = None):
        """
        Update production data for a model.

        Args:
            model_id: Model ID
            features: Input features
            prediction: Model prediction
            ground_truth: Actual ground truth value
            metrics: Performance metrics
        """
        with self._lock:
            if model_id not in self.production_data:
                # Initialize if not exists
                self.production_data[model_id] = {
                    "features": [],
                    "predictions": [],
                    "ground_truth": [],
                    "metrics": defaultdict(list),
                    "timestamps": [],
                    "last_checked": time.time()
                }

            # Add data to production tracking
            current_time = time.time()

            # Add features if provided
            if features is not None:
                self.production_data[model_id]["features"].append(features)

                # Limit size if needed
                if len(self.production_data[model_id]["features"]) > self.max_history_size:
                    self.production_data[model_id]["features"] = self.production_data[model_id]["features"][-self.max_history_size:]

            # Add prediction if provided
            if prediction is not None:
                self.production_data[model_id]["predictions"].append(prediction)

                # Limit size if needed
                if len(self.production_data[model_id]["predictions"]) > self.max_history_size:
                    self.production_data[model_id]["predictions"] = self.production_data[model_id]["predictions"][-self.max_history_size:]

            # Add ground truth if provided
            if ground_truth is not None:
                self.production_data[model_id]["ground_truth"].append(ground_truth)

                # Limit size if needed
                if len(self.production_data[model_id]["ground_truth"]) > self.max_history_size:
                    self.production_data[model_id]["ground_truth"] = self.production_data[model_id]["ground_truth"][-self.max_history_size:]

            # Add metrics if provided
            if metrics:
                for key, value in metrics.items():
                    self.production_data[model_id]["metrics"][key].append(value)

                    # Limit size if needed
                    if len(self.production_data[model_id]["metrics"][key]) > self.max_history_size:
                        self.production_data[model_id]["metrics"][key] = self.production_data[model_id]["metrics"][key][-self.max_history_size:]

            # Add timestamp
            self.production_data[model_id]["timestamps"].append(current_time)

            # Limit size if needed
            if len(self.production_data[model_id]["timestamps"]) > self.max_history_size:
                self.production_data[model_id]["timestamps"] = self.production_data[model_id]["timestamps"][-self.max_history_size:]

            # Update historical metrics
            if features is not None or prediction is not None or ground_truth is not None or metrics:
                metric_entry = ModelMetrics(
                    model_id=model_id,
                    model_type=self.model_configs.get(model_id, {}).get("model_type", "unknown"),
                    timestamp=current_time,
                    metrics=metrics or {},
                    feature_values=features if isinstance(features, dict) else None,
                    prediction_values=[prediction] if prediction is not None else None,
                    ground_truth_values=[ground_truth] if ground_truth is not None else None
                )
                self.historical_metrics[model_id].append(metric_entry)

    def check_for_drift(self, model_id: str, force: bool = False) -> Optional[List[DriftAlert]]:
        """
        Check for drift in a model.

        Args:
            model_id: Model ID
            force: Whether to force a check regardless of sample size

        Returns:
            List of drift alerts or None if not enough data
        """
        with self._lock:
            if model_id not in self.reference_data:
                logger.warning(f"No reference data for model {model_id}")
                return None

            if model_id not in self.production_data:
                logger.warning(f"No production data for model {model_id}")
                return None

            # Get model configuration
            config = self.model_configs.get(model_id, {})
            detection_method = config.get("detection_method", self.default_detection_method)
            feature_drift_threshold = config.get("feature_drift_threshold", self.feature_drift_threshold)
            prediction_drift_threshold = config.get("prediction_drift_threshold", self.prediction_drift_threshold)
            performance_drift_threshold = config.get("performance_drift_threshold", self.performance_drift_threshold)
            min_samples = config.get("min_samples_for_drift", self.min_samples_for_drift)

            # Get reference and production data
            reference = self.reference_data[model_id]
            production = self.production_data[model_id]

            # Check if we have enough data
            if not force and (
                (len(production["features"]) < min_samples if production["features"] else True) and
                (len(production["predictions"]) < min_samples if production["predictions"] else True) and
                (len(production["ground_truth"]) < min_samples if production["ground_truth"] else True)
            ):
                logger.debug(f"Not enough data to check drift for model {model_id}")
                return None

            alerts = []

            # Check for feature drift
            if reference["feature_stats"] and production["features"]:
                try:
                    feature_drift = self._detect_feature_drift(
                        reference["feature_stats"],
                        production["features"],
                        detection_method,
                        feature_drift_threshold,
                        config.get("monitored_features", [])
                    )

                    if feature_drift["severity"] != DriftSeverity.NONE:
                        alerts.append(DriftAlert(
                            model_id=model_id,
                            model_type=reference["model_type"],
                            drift_type=DriftType.FEATURE_DRIFT,
                            severity=feature_drift["severity"],
                            timestamp=time.time(),
                            metrics=feature_drift["metrics"],
                            feature_importances=reference.get("feature_importances"),
                            recommendation=feature_drift["recommendation"],
                            details=feature_drift["details"]
                        ))
                except Exception as e:
                    logger.error(f"Error checking feature drift for model {model_id}: {str(e)}", exc_info=True)

            # Check for prediction drift
            if reference["prediction_stats"] and production["predictions"]:
                try:
                    prediction_drift = self._detect_prediction_drift(
                        reference["prediction_stats"],
                        production["predictions"],
                        detection_method,
                        prediction_drift_threshold
                    )

                    if prediction_drift["severity"] != DriftSeverity.NONE:
                        alerts.append(DriftAlert(
                            model_id=model_id,
                            model_type=reference["model_type"],
                            drift_type=DriftType.PREDICTION_DRIFT,
                            severity=prediction_drift["severity"],
                            timestamp=time.time(),
                            metrics=prediction_drift["metrics"],
                            recommendation=prediction_drift["recommendation"],
                            details=prediction_drift["details"]
                        ))
                except Exception as e:
                    logger.error(f"Error checking prediction drift for model {model_id}: {str(e)}", exc_info=True)

            # Check for performance drift
            if reference["metrics"] and production["metrics"]:
                try:
                    performance_drift = self._detect_performance_drift(
                        reference["metrics"],
                        production["metrics"],
                        performance_drift_threshold
                    )

                    if performance_drift["severity"] != DriftSeverity.NONE:
                        alerts.append(DriftAlert(
                            model_id=model_id,
                            model_type=reference["model_type"],
                            drift_type=DriftType.PERFORMANCE_DRIFT,
                            severity=performance_drift["severity"],
                            timestamp=time.time(),
                            metrics=performance_drift["metrics"],
                            recommendation=performance_drift["recommendation"],
                            details=performance_drift["details"]
                        ))
                except Exception as e:
                    logger.error(f"Error checking performance drift for model {model_id}: {str(e)}", exc_info=True)

            # Check for concept drift if we have ground truth
            if (reference["feature_stats"] and reference["ground_truth_stats"] and
                production["features"] and production["ground_truth"]):
                try:
                    concept_drift = self._detect_concept_drift(
                        reference["feature_stats"],
                        reference["ground_truth_stats"],
                        production["features"],
                        production["ground_truth"],
                        detection_method,
                        prediction_drift_threshold  # Use prediction threshold for concept drift too
                    )

                    if concept_drift["severity"] != DriftSeverity.NONE:
                        alerts.append(DriftAlert(
                            model_id=model_id,
                            model_type=reference["model_type"],
                            drift_type=DriftType.CONCEPT_DRIFT,
                            severity=concept_drift["severity"],
                            timestamp=time.time(),
                            metrics=concept_drift["metrics"],
                            feature_importances=reference.get("feature_importances"),
                            recommendation=concept_drift["recommendation"],
                            details=concept_drift["details"]
                        ))
                except Exception as e:
                    logger.error(f"Error checking concept drift for model {model_id}: {str(e)}", exc_info=True)

            # Update last checked timestamp
            self.production_data[model_id]["last_checked"] = time.time()

            return alerts if alerts else None

    def check_all_models(self, force: bool = False) -> Dict[str, List[DriftAlert]]:
        """
        Check all models for drift.

        Args:
            force: Whether to force a check regardless of sample size

        Returns:
            Dictionary mapping model IDs to lists of drift alerts
        """
        results = {}

        for model_id in self.reference_data:
            alerts = self.check_for_drift(model_id, force)
            if alerts:
                results[model_id] = alerts

                # Process alerts
                self._process_alerts(model_id, alerts)

        return results

    def get_model_health(self, model_id: str) -> Dict[str, Any]:
        """
        Get health status of a model.

        Args:
            model_id: Model ID

        Returns:
            Dictionary with model health information
        """
        with self._lock:
            if model_id not in self.reference_data:
                return {"status": "unknown", "reason": "No reference data available"}

            if model_id not in self.production_data:
                return {"status": "unknown", "reason": "No production data available"}

            # Get reference and production data
            reference = self.reference_data[model_id]
            production = self.production_data[model_id]

            # Calculate basic stats
            stats = {
                "model_id": model_id,
                "model_type": reference["model_type"],
                "reference_timestamp": reference["timestamp"],
                "last_checked": production["last_checked"],
                "production_samples": {
                    "features": len(production["features"]),
                    "predictions": len(production["predictions"]),
                    "ground_truth": len(production["ground_truth"])
                },
                "metrics": {}
            }

            # Add metrics if available
            for metric_name, values in production["metrics"].items():
                if values:
                    stats["metrics"][metric_name] = {
                        "current": values[-1],
                        "mean": statistics.mean(values),
                        "min": min(values),
                        "max": max(values),
                        "reference": reference["metrics"].get(metric_name)
                    }

            # Check for drift
            alerts = self.check_for_drift(model_id)
            if alerts:
                # Find highest severity
                highest_severity = max(alert.severity for alert in alerts)

                if highest_severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
                    status = "unhealthy"
                elif highest_severity == DriftSeverity.MEDIUM:
                    status = "degraded"
                else:
                    status = "healthy"

                stats["status"] = status
                stats["alerts"] = [
                    {
                        "type": alert.drift_type.value,
                        "severity": alert.severity.value,
                        "metrics": alert.metrics,
                        "recommendation": alert.recommendation
                    }
                    for alert in alerts
                ]
            else:
                stats["status"] = "healthy"
                stats["alerts"] = []

            return stats

    def get_all_models_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Get health status of all models.

        Returns:
            Dictionary mapping model IDs to health information
        """
        results = {}

        for model_id in self.reference_data:
            results[model_id] = self.get_model_health(model_id)

        return results

    def update_model_config(self, model_id: str, config: Dict[str, Any]) -> bool:
        """
        Update configuration for a model.

        Args:
            model_id: Model ID
            config: Configuration dictionary

        Returns:
            Success boolean
        """
        with self._lock:
            if model_id not in self.model_configs:
                logger.warning(f"No configuration found for model {model_id}")
                return False

            # Update configuration
            for key, value in config.items():
                if key in self.model_configs[model_id]:
                    self.model_configs[model_id][key] = value

            logger.info(f"Updated configuration for model {model_id}")
            return True

    def get_historical_metrics(self,
                             model_id: str,
                             metric_name: Optional[str] = None,
                             start_time: Optional[float] = None,
                             end_time: Optional[float] = None,
                             limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Get historical metrics for a model.

        Args:
            model_id: Model ID
            metric_name: Optional specific metric name
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of entries to return

        Returns:
            List of metric entries
        """
        with self._lock:
            if model_id not in self.historical_metrics:
                return []

            # Apply filters
            results = []

            for entry in self.historical_metrics[model_id]:
                # Apply time filters
                if start_time and entry.timestamp < start_time:
                    continue

                if end_time and entry.timestamp > end_time:
                    continue

                # Apply metric filter
                if metric_name:
                    if metric_name not in entry.metrics:
                        continue

                    # Create simplified entry with just the requested metric
                    results.append({
                        "timestamp": entry.timestamp,
                        "model_id": entry.model_id,
                        "model_type": entry.model_type,
                        "metric": {
                            metric_name: entry.metrics[metric_name]
                        }
                    })
                else:
                    # Include all metrics
                    results.append({
                        "timestamp": entry.timestamp,
                        "model_id": entry.model_id,
                        "model_type": entry.model_type,
                        "metrics": entry.metrics
                    })

            # Apply limit
            return results[-limit:] if len(results) > limit else results

    def trigger_retraining(self, model_id: str, reason: str) -> bool:
        """
        Trigger retraining for a model.

        Args:
            model_id: Model ID
            reason: Reason for retraining

        Returns:
            Success boolean
        """
        if not self.enable_auto_retraining:
            logger.info(f"Auto-retraining is disabled, skipping for model {model_id}")
            return False

        # Publish retraining event
        event = create_event(
            EventTopics.MODEL_UPDATED,
            {
                "model_id": model_id,
                "action": "retrain",
                "reason": reason,
                "timestamp": time.time()
            },
            priority=EventPriority.HIGH,
            source="model_monitor"
        )

        self.event_bus.publish(event)

        logger.info(f"Triggered retraining for model {model_id}: {reason}")
        return True

    def _handle_prediction_event(self, event: Event):
        """
        Handle prediction event.

        Args:
            event: Prediction event
        """
        if not event.data:
            return

        # Extract data from event
        model_id = event.data.get("model_id")
        if not model_id:
            return

        features = event.data.get("features")
        prediction = event.data.get("prediction")
        ground_truth = event.data.get("ground_truth")
        metrics = event.data.get("metrics")

        # Update production data
        self.update_production_data(
            model_id=model_id,
            features=features,
            prediction=prediction,
            ground_truth=ground_truth,
            metrics=metrics
        )

    def _handle_model_deployed(self, event: Event):
        """
        Handle model deployed event.

        Args:
            event: Model deployed event
        """
        if not event.data:
            return

        # Extract data from event
        model_id = event.data.get("model_id")
        model_type = event.data.get("model_type")

        if not model_id or not model_type:
            return

        # Check if this is a new model
        if model_id not in self.reference_data:
            logger.info(f"New model deployed: {model_id}, type: {model_type}")

            # Try to get reference data from model deployer
            if self.model_deployer:
                metadata = self.model_deployer.get_model_metadata(model_id)
                if metadata:
                    # Extract reference data from metadata
                    metrics = metadata.get("validation_results", {}).get("metrics", {})
                    feature_importances = metadata.get("validation_results", {}).get("feature_importances", {})

                    # Add reference data
                    self.add_model_reference_data(
                        model_id=model_id,
                        model_type=model_type,
                        metrics=metrics,
                        feature_importances=feature_importances
                    )

    def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.debug("Model monitoring loop started")

        while self._running:
            try:
                # Check all models for drift
                self.check_all_models()

                # Sleep until next interval
                for _ in range(self.monitoring_interval):
                    if not self._running:
                        break
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}", exc_info=True)
                time.sleep(60)  # Sleep on error to avoid tight loops

    def _process_alerts(self, model_id: str, alerts: List[DriftAlert]):
        """
        Process drift alerts.

        Args:
            model_id: Model ID
            alerts: List of drift alerts
        """
        for alert in alerts:
            # Log alert
            log_level = logging.WARNING
            if alert.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
                log_level = logging.ERROR

            logger.log(log_level, f"Drift alert for model {model_id}: {alert.drift_type.value} drift with {alert.severity.value} severity")

            # Send to health monitor
            if self.health_monitor:
                # Determine health status
                if alert.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
                    status = HealthStatus.CRITICAL
                elif alert.severity == DriftSeverity.MEDIUM:
                    status = HealthStatus.WARNING
                else:
                    status = HealthStatus.HEALTHY

                # Update model health
                self.health_monitor.update_component_health(
                    component_id=f"model_{model_id}",
                    status=status,
                    metrics={
                        "drift_type": alert.drift_type.value,
                        "drift_severity": alert.severity.value,
                        "drift_timestamp": alert.timestamp,
                        **alert.metrics
                    }
                )

                # Send alert
                alert_level = AlertLevel.WARNING
                if alert.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
                    alert_level = AlertLevel.CRITICAL

                alert_data = {
                    "model_id": model_id,
                    "model_type": alert.model_type,
                    "drift_type": alert.drift_type.value,
                    "severity": alert.severity.value,
                    "metrics": alert.metrics,
                    "recommendation": alert.recommendation,
                    "timestamp": alert.timestamp
                }

                self.health_monitor.update_component_health(
                    "model_monitor",
                    metrics={
                        f"last_alert_{model_id}": time.time(),
                        f"last_alert_type_{model_id}": alert.drift_type.value,
                        f"last_alert_severity_{model_id}": alert.severity.value
                    }
                )

            # Publish alert event
            event = create_event(
                EventTopics.MODEL_UPDATED,
                {
                    "model_id": model_id,
                    "alert_type": alert.drift_type.value,
                    "severity": alert.severity.value,
                    "metrics": alert.metrics,
                    "recommendation": alert.recommendation,
                    "timestamp": alert.timestamp
                },
                priority=EventPriority.HIGH if alert.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL] else EventPriority.NORMAL,
                source="model_monitor"
            )

            self.event_bus.publish(event)

            # Trigger retraining for high severity alerts
            if alert.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]:
                reason = f"{alert.drift_type.value} drift detected with {alert.severity.value} severity"
                self.trigger_retraining(model_id, reason)

            # Store alert in state manager if available
            if self.state_manager:
                alerts_path = f"models.alerts.{model_id}"
                existing_alerts = self.state_manager.get(alerts_path, [])

                # Add new alert
                alert_dict = {
                    "drift_type": alert.drift_type.value,
                    "severity": alert.severity.value,
                    "timestamp": alert.timestamp,
                    "metrics": alert.metrics,
                    "recommendation": alert.recommendation
                }

                existing_alerts.append(alert_dict)

                # Keep only recent alerts (last 100)
                if len(existing_alerts) > 100:
                    existing_alerts = existing_alerts[-100:]

                # Update state
                self.state_manager.set(alerts_path, existing_alerts, scope=StateScope.PERSISTENT)

    def _load_reference_data(self):
        """Load reference data from disk"""
        if not os.path.exists(self.reference_data_dir):
            return

        for filename in os.listdir(self.reference_data_dir):
            if filename.endswith(".json"):
                model_id = filename[:-5]  # Remove .json extension
                file_path = os.path.join(self.reference_data_dir, filename)

                try:
                    with open(file_path, 'r') as f:
                        reference_data = json.load(f)

                    self.reference_data[model_id] = reference_data

                    # Load configuration
                    if "config" in reference_data:
                        self.model_configs[model_id] = reference_data["config"]
                    else:
                        # Create default configuration
                        self.model_configs[model_id] = {
                            "model_type": reference_data["model_type"],
                            "detection_method": self.default_detection_method,
                            "feature_drift_threshold": self.feature_drift_threshold,
                            "prediction_drift_threshold": self.prediction_drift_threshold,
                            "performance_drift_threshold": self.performance_drift_threshold,
                            "monitored_features": [],
                            "min_samples_for_drift": self.min_samples_for_drift
                        }

                    # Initialize production data
                    self.production_data[model_id] = {
                        "features": [],
                        "predictions": [],
                        "ground_truth": [],
                        "metrics": defaultdict(list),
                        "timestamps": [],
                        "last_checked": time.time()
                    }

                    logger.info(f"Loaded reference data for model {model_id}")
                except Exception as e:
                    logger.error(f"Error loading reference data for model {model_id}: {str(e)}")

    def _save_reference_data(self, model_id: str):
        """Save reference data to disk"""
        if model_id not in self.reference_data:
            return

        file_path = os.path.join(self.reference_data_dir, f"{model_id}.json")

        try:
            # Add configuration to reference data
            reference_data = self.reference_data[model_id].copy()
            if model_id in self.model_configs:
                reference_data["config"] = self.model_configs[model_id]

            with open(file_path, 'w') as f:
                json.dump(reference_data, f, indent=2)

            logger.debug(f"Saved reference data for model {model_id}")
        except Exception as e:
            logger.error(f"Error saving reference data for model {model_id}: {str(e)}")