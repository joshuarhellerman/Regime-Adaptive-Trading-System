"""
Unit tests for the model_monitoring module.

This test suite verifies functionality of the ModelMonitor class, including:
- Initialization and configuration
- Drift detection (feature, prediction, performance, and concept drift)
- Alert generation and processing
- Historical metrics tracking
- Integration with health monitoring
"""

import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np
import pandas as pd
import time
import json
import os
import pickle
from datetime import datetime, timedelta
import threading
import queue
import tempfile
import shutil

from models.production.model_monitoring import (
    ModelMonitor, DriftType, DriftSeverity, DriftAlert,
    ModelMetrics, DriftDetectionMethod
)
from core.event_bus import EventTopics, Event, create_event, EventPriority
from core.health_monitor import HealthMonitor, HealthStatus, AlertLevel
from core.state_manager import StateManager, StateScope
from models.production.model_deployer import ModelDeployer


class TestModelMonitor(unittest.TestCase):
    """Test suite for ModelMonitor class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock dependencies
        self.model_deployer = MagicMock(spec=ModelDeployer)
        self.health_monitor = MagicMock(spec=HealthMonitor)
        self.state_manager = MagicMock(spec=StateManager)
        self.event_bus = MagicMock()
        self.event_bus.subscribe = MagicMock()
        self.event_bus.publish = MagicMock()

        # Create temporary directory for reference data
        self.temp_dir = tempfile.mkdtemp()

        # Initialize ModelMonitor with faster interval for testing
        self.monitor = ModelMonitor(
            model_deployer=self.model_deployer,
            health_monitor=self.health_monitor,
            state_manager=self.state_manager,
            event_bus=self.event_bus,
            reference_data_dir=self.temp_dir,
            monitoring_interval=1  # 1 second for faster testing
        )

        # Sample data for testing
        self.model_id = "test_model_123"
        self.model_type = "classification"

        # Sample features
        self.sample_features = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [0.1, 0.2, 0.3, 0.4, 0.5]
        })

        # Sample predictions
        self.sample_predictions = np.array([0, 1, 0, 1, 0])

        # Sample ground truth
        self.sample_ground_truth = np.array([0, 1, 0, 0, 1])

        # Sample metrics
        self.sample_metrics = {
            "accuracy": 0.8,
            "precision": 0.75,
            "recall": 0.85,
            "f1_score": 0.79
        }

        # Sample feature importances
        self.sample_feature_importances = {
            "feature1": 0.7,
            "feature2": 0.3
        }

    def tearDown(self):
        """Clean up after tests"""
        # Stop the monitoring thread if running
        if hasattr(self, 'monitor'):
            self.monitor.stop()

        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """Test ModelMonitor initialization"""
        # Verify that the monitor was initialized correctly
        self.assertEqual(self.monitor.monitoring_interval, 1)
        self.assertEqual(self.monitor.reference_data_dir, self.temp_dir)

        # Verify event bus subscriptions
        self.event_bus.subscribe.assert_any_call(
            EventTopics.MODEL_PREDICTION,
            self.monitor._handle_prediction_event
        )

        self.event_bus.subscribe.assert_any_call(
            EventTopics.MODEL_DEPLOYED,
            self.monitor._handle_model_deployed
        )

        # Verify health monitor registration
        self.health_monitor.register_component.assert_called_with(
            "model_monitor", "monitoring"
        )

    def test_add_model_reference_data(self):
        """Test adding reference data for a model"""
        # Add reference data
        self.monitor.add_model_reference_data(
            model_id=self.model_id,
            model_type=self.model_type,
            features=self.sample_features,
            predictions=self.sample_predictions,
            ground_truth=self.sample_ground_truth,
            metrics=self.sample_metrics,
            feature_importances=self.sample_feature_importances
        )

        # Verify reference data was stored
        self.assertIn(self.model_id, self.monitor.reference_data)
        self.assertEqual(self.monitor.reference_data[self.model_id]["model_type"], self.model_type)
        self.assertEqual(self.monitor.reference_data[self.model_id]["metrics"], self.sample_metrics)
        self.assertEqual(self.monitor.reference_data[self.model_id]["feature_importances"], self.sample_feature_importances)

        # Verify model configuration was created
        self.assertIn(self.model_id, self.monitor.model_configs)
        self.assertEqual(self.monitor.model_configs[self.model_id]["model_type"], self.model_type)

        # Verify production data structure was initialized
        self.assertIn(self.model_id, self.monitor.production_data)
        self.assertEqual(len(self.monitor.production_data[self.model_id]["features"]), 0)
        self.assertEqual(len(self.monitor.production_data[self.model_id]["predictions"]), 0)

        # Verify reference data file was created
        file_path = os.path.join(self.temp_dir, f"{self.model_id}.json")
        self.assertTrue(os.path.exists(file_path))

        # Verify file contents
        with open(file_path, 'r') as f:
            data = json.load(f)
            self.assertEqual(data["model_id"], self.model_id)
            self.assertEqual(data["model_type"], self.model_type)
            self.assertEqual(data["metrics"], self.sample_metrics)

    def test_update_production_data(self):
        """Test updating production data"""
        # First, add reference data
        self.monitor.add_model_reference_data(
            model_id=self.model_id,
            model_type=self.model_type
        )

        # Update with new production data
        features = {"feature1": 1.5, "feature2": 0.3}
        prediction = 1
        ground_truth = 1
        metrics = {"accuracy": 0.9}

        self.monitor.update_production_data(
            model_id=self.model_id,
            features=features,
            prediction=prediction,
            ground_truth=ground_truth,
            metrics=metrics
        )

        # Verify data was stored
        self.assertEqual(len(self.monitor.production_data[self.model_id]["features"]), 1)
        self.assertEqual(len(self.monitor.production_data[self.model_id]["predictions"]), 1)
        self.assertEqual(len(self.monitor.production_data[self.model_id]["ground_truth"]), 1)
        self.assertEqual(len(self.monitor.production_data[self.model_id]["metrics"]["accuracy"]), 1)

        # Verify values
        self.assertEqual(self.monitor.production_data[self.model_id]["features"][0], features)
        self.assertEqual(self.monitor.production_data[self.model_id]["predictions"][0], prediction)
        self.assertEqual(self.monitor.production_data[self.model_id]["ground_truth"][0], ground_truth)
        self.assertEqual(self.monitor.production_data[self.model_id]["metrics"]["accuracy"][0], metrics["accuracy"])

        # Verify historical metrics
        self.assertEqual(len(self.monitor.historical_metrics[self.model_id]), 1)
        metric_entry = self.monitor.historical_metrics[self.model_id][0]
        self.assertEqual(metric_entry.model_id, self.model_id)
        self.assertEqual(metric_entry.model_type, self.model_type)
        self.assertEqual(metric_entry.metrics, metrics)

    def test_update_production_data_handles_max_history(self):
        """Test that production data respects max history size"""
        # Set a small max history size for testing
        self.monitor.max_history_size = 3

        # Add reference data
        self.monitor.add_model_reference_data(
            model_id=self.model_id,
            model_type=self.model_type
        )

        # Add more data than the max history size
        for i in range(5):
            self.monitor.update_production_data(
                model_id=self.model_id,
                prediction=i,
                metrics={"accuracy": 0.8 + i * 0.02}
            )

        # Verify only the most recent entries are kept
        self.assertEqual(len(self.monitor.production_data[self.model_id]["predictions"]), 3)
        self.assertEqual(len(self.monitor.production_data[self.model_id]["metrics"]["accuracy"]), 3)

        # Verify the values are the most recent ones (2, 3, 4)
        self.assertEqual(self.monitor.production_data[self.model_id]["predictions"], [2, 3, 4])

    @patch('models.production.model_monitoring.ModelMonitor._detect_feature_drift')
    def test_check_for_drift_feature_drift(self, mock_detect_feature_drift):
        """Test checking for feature drift"""
        # Mock the drift detection to return a drift alert
        mock_detect_feature_drift.return_value = {
            "severity": DriftSeverity.HIGH,
            "metrics": {"drift_score": 0.8},
            "recommendation": "Retrain model",
            "details": {"affected_features": ["feature1"]}
        }

        # Add reference data with features
        self.monitor.add_model_reference_data(
            model_id=self.model_id,
            model_type=self.model_type,
            features=self.sample_features
        )

        # Add production feature data
        for i in range(self.monitor.min_samples_for_drift + 1):
            self.monitor.update_production_data(
                model_id=self.model_id,
                features={"feature1": 5.0 + i, "feature2": 0.5 + i * 0.1}
            )

        # Check for drift
        alerts = self.monitor.check_for_drift(self.model_id)

        # Verify drift detection was called
        mock_detect_feature_drift.assert_called_once()

        # Verify alert was generated
        self.assertIsNotNone(alerts)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].drift_type, DriftType.FEATURE_DRIFT)
        self.assertEqual(alerts[0].severity, DriftSeverity.HIGH)
        self.assertEqual(alerts[0].model_id, self.model_id)

    @patch('models.production.model_monitoring.ModelMonitor._detect_prediction_drift')
    def test_check_for_drift_prediction_drift(self, mock_detect_prediction_drift):
        """Test checking for prediction drift"""
        # Mock the drift detection to return a drift alert
        mock_detect_prediction_drift.return_value = {
            "severity": DriftSeverity.MEDIUM,
            "metrics": {"drift_score": 0.6},
            "recommendation": "Monitor closely",
            "details": {}
        }

        # Add reference data with predictions
        self.monitor.add_model_reference_data(
            model_id=self.model_id,
            model_type=self.model_type,
            predictions=self.sample_predictions
        )

        # Add production prediction data
        for i in range(self.monitor.min_samples_for_drift + 1):
            self.monitor.update_production_data(
                model_id=self.model_id,
                prediction=1  # Always predict 1 to create drift
            )

        # Check for drift
        alerts = self.monitor.check_for_drift(self.model_id)

        # Verify drift detection was called
        mock_detect_prediction_drift.assert_called_once()

        # Verify alert was generated
        self.assertIsNotNone(alerts)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].drift_type, DriftType.PREDICTION_DRIFT)
        self.assertEqual(alerts[0].severity, DriftSeverity.MEDIUM)

    @patch('models.production.model_monitoring.ModelMonitor._detect_performance_drift')
    def test_check_for_drift_performance_drift(self, mock_detect_performance_drift):
        """Test checking for performance drift"""
        # Mock the drift detection to return a drift alert
        mock_detect_performance_drift.return_value = {
            "severity": DriftSeverity.CRITICAL,
            "metrics": {"accuracy_drop": 0.3},
            "recommendation": "Retrain immediately",
            "details": {"metrics_affected": ["accuracy", "f1_score"]}
        }

        # Add reference data with metrics
        self.monitor.add_model_reference_data(
            model_id=self.model_id,
            model_type=self.model_type,
            metrics=self.sample_metrics
        )

        # Add production metrics data
        for i in range(10):  # Add some metrics
            self.monitor.update_production_data(
                model_id=self.model_id,
                metrics={"accuracy": 0.5, "precision": 0.4}  # Lower metrics to trigger drift
            )

        # Check for drift
        alerts = self.monitor.check_for_drift(self.model_id)

        # Verify drift detection was called
        mock_detect_performance_drift.assert_called_once()

        # Verify alert was generated
        self.assertIsNotNone(alerts)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].drift_type, DriftType.PERFORMANCE_DRIFT)
        self.assertEqual(alerts[0].severity, DriftSeverity.CRITICAL)

    @patch('models.production.model_monitoring.ModelMonitor._process_alerts')
    def test_check_all_models(self, mock_process_alerts):
        """Test checking all models for drift"""
        # Add multiple models
        model_id1 = "model1"
        model_id2 = "model2"

        # Add reference data for two models
        self.monitor.add_model_reference_data(
            model_id=model_id1,
            model_type="classification",
            predictions=self.sample_predictions
        )

        self.monitor.add_model_reference_data(
            model_id=model_id2,
            model_type="regression",
            predictions=np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        )

        # Mock check_for_drift to return alerts for one model but not the other
        original_check_for_drift = self.monitor.check_for_drift

        def mock_check_for_drift(model_id, force=False):
            if model_id == model_id1:
                return [DriftAlert(
                    model_id=model_id1,
                    model_type="classification",
                    drift_type=DriftType.PREDICTION_DRIFT,
                    severity=DriftSeverity.MEDIUM,
                    timestamp=time.time(),
                    metrics={"drift_score": 0.6}
                )]
            return None

        self.monitor.check_for_drift = mock_check_for_drift

        # Check all models
        results = self.monitor.check_all_models()

        # Restore original method
        self.monitor.check_for_drift = original_check_for_drift

        # Verify results
        self.assertIn(model_id1, results)
        self.assertNotIn(model_id2, results)
        self.assertEqual(len(results[model_id1]), 1)

        # Verify process_alerts was called
        mock_process_alerts.assert_called_once_with(model_id1, results[model_id1])

    def test_get_model_health(self):
        """Test getting model health information"""
        # Add reference data with metrics
        self.monitor.add_model_reference_data(
            model_id=self.model_id,
            model_type=self.model_type,
            metrics=self.sample_metrics
        )

        # Add production metrics data
        for i in range(5):
            self.monitor.update_production_data(
                model_id=self.model_id,
                metrics={"accuracy": 0.75 + i * 0.05}
            )

        # Mock check_for_drift to return no alerts (healthy)
        original_check_for_drift = self.monitor.check_for_drift

        def mock_check_for_drift(model_id, force=False):
            return None

        self.monitor.check_for_drift = mock_check_for_drift

        # Get model health
        health = self.monitor.get_model_health(self.model_id)

        # Restore original method
        self.monitor.check_for_drift = original_check_for_drift

        # Verify health information
        self.assertEqual(health["model_id"], self.model_id)
        self.assertEqual(health["model_type"], self.model_type)
        self.assertEqual(health["status"], "healthy")
        self.assertIn("metrics", health)
        self.assertIn("accuracy", health["metrics"])
        self.assertEqual(health["metrics"]["accuracy"]["current"], 0.95)  # Last value
        self.assertAlmostEqual(health["metrics"]["accuracy"]["mean"], 0.85, places=2)

    def test_update_model_config(self):
        """Test updating model configuration"""
        # Add reference data with default configuration
        self.monitor.add_model_reference_data(
            model_id=self.model_id,
            model_type=self.model_type
        )

        # Initial configuration values
        initial_threshold = self.monitor.model_configs[self.model_id]["feature_drift_threshold"]

        # Update configuration
        new_config = {
            "feature_drift_threshold": 0.2,
            "detection_method": DriftDetectionMethod.PSI
        }

        result = self.monitor.update_model_config(self.model_id, new_config)

        # Verify update was successful
        self.assertTrue(result)

        # Verify configuration was updated
        self.assertEqual(self.monitor.model_configs[self.model_id]["feature_drift_threshold"], 0.2)
        self.assertEqual(self.monitor.model_configs[self.model_id]["detection_method"], DriftDetectionMethod.PSI)

    def test_get_historical_metrics(self):
        """Test retrieving historical metrics"""
        # Add reference data
        self.monitor.add_model_reference_data(
            model_id=self.model_id,
            model_type=self.model_type
        )

        # Add metrics over time
        current_time = time.time()

        # Create metric entries with different timestamps
        for i in range(5):
            metric_entry = ModelMetrics(
                model_id=self.model_id,
                model_type=self.model_type,
                timestamp=current_time - (5-i) * 3600,  # Hours ago
                metrics={"accuracy": 0.7 + i * 0.05, "precision": 0.8}
            )
            self.monitor.historical_metrics[self.model_id].append(metric_entry)

        # Get all historical metrics
        all_metrics = self.monitor.get_historical_metrics(self.model_id)

        # Verify all metrics were returned
        self.assertEqual(len(all_metrics), 5)

        # Get metrics for a specific time range
        start_time = current_time - 3 * 3600
        filtered_metrics = self.monitor.get_historical_metrics(
            model_id=self.model_id,
            start_time=start_time
        )

        # Verify filtering by time
        self.assertEqual(len(filtered_metrics), 3)  # Last 3 entries

        # Get metrics for a specific metric name
        specific_metric = self.monitor.get_historical_metrics(
            model_id=self.model_id,
            metric_name="accuracy"
        )

        # Verify metric filtering
        self.assertEqual(len(specific_metric), 5)
        self.assertIn("metric", specific_metric[0])
        self.assertIn("accuracy", specific_metric[0]["metric"])

    @patch('models.production.model_monitoring.ModelMonitor.trigger_retraining')
    def test_process_alerts(self, mock_trigger_retraining):
        """Test processing drift alerts"""
        # Create alerts with different severities
        alerts = [
            DriftAlert(
                model_id=self.model_id,
                model_type=self.model_type,
                drift_type=DriftType.FEATURE_DRIFT,
                severity=DriftSeverity.LOW,
                timestamp=time.time(),
                metrics={"drift_score": 0.3}
            ),
            DriftAlert(
                model_id=self.model_id,
                model_type=self.model_type,
                drift_type=DriftType.PREDICTION_DRIFT,
                severity=DriftSeverity.HIGH,
                timestamp=time.time(),
                metrics={"drift_score": 0.8},
                recommendation="Retrain model"
            )
        ]

        # Process alerts
        self.monitor._process_alerts(self.model_id, alerts)

        # Verify health updates
        self.health_monitor.update_component_health.assert_any_call(
            component_id=f"model_{self.model_id}",
            status=HealthStatus.CRITICAL,  # For HIGH severity
            metrics={"drift_type": "prediction_drift", "drift_severity": "high", "drift_timestamp": alerts[1].timestamp, "drift_score": 0.8}
        )

        # Verify events were published
        self.assertEqual(self.event_bus.publish.call_count, 2)

        # Verify retraining was triggered for HIGH severity alert only
        mock_trigger_retraining.assert_called_once()
        mock_trigger_retraining.assert_called_with(
            self.model_id,
            "prediction_drift drift detected with high severity"
        )

        # Verify alert was stored in state manager
        self.state_manager.set.assert_called_once()

    def test_trigger_retraining(self):
        """Test triggering model retraining"""
        # Enable auto-retraining
        self.monitor.enable_auto_retraining = True

        # Trigger retraining
        result = self.monitor.trigger_retraining(
            model_id=self.model_id,
            reason="High drift detected"
        )

        # Verify result
        self.assertTrue(result)

        # Verify event was published
        self.event_bus.publish.assert_called_once()

        # Get the published event
        published_event = self.event_bus.publish.call_args[0][0]

        # Verify event properties
        self.assertEqual(published_event.topic, EventTopics.MODEL_UPDATED)
        self.assertEqual(published_event.data["model_id"], self.model_id)
        self.assertEqual(published_event.data["action"], "retrain")
        self.assertEqual(published_event.data["reason"], "High drift detected")
        self.assertEqual(published_event.priority, EventPriority.HIGH)

    def test_handle_prediction_event(self):
        """Test handling prediction events"""
        # Create a prediction event
        event = Event(
            topic=EventTopics.MODEL_PREDICTION,
            data={
                "model_id": self.model_id,
                "features": {"feature1": 1.0, "feature2": 0.5},
                "prediction": 1,
                "ground_truth": 1,
                "metrics": {"accuracy": 1.0}
            }
        )

        # Add reference data first
        self.monitor.add_model_reference_data(
            model_id=self.model_id,
            model_type=self.model_type
        )

        # Handle event
        self.monitor._handle_prediction_event(event)

        # Verify production data was updated
        self.assertEqual(len(self.monitor.production_data[self.model_id]["features"]), 1)
        self.assertEqual(len(self.monitor.production_data[self.model_id]["predictions"]), 1)
        self.assertEqual(len(self.monitor.production_data[self.model_id]["ground_truth"]), 1)
        self.assertEqual(len(self.monitor.production_data[self.model_id]["metrics"]["accuracy"]), 1)

    def test_handle_model_deployed(self):
        """Test handling model deployed events"""
        # Set up mock model deployer to return metadata
        metadata = {
            "validation_results": {
                "metrics": {"accuracy": 0.9, "f1_score": 0.85},
                "feature_importances": {"feature1": 0.7, "feature2": 0.3}
            }
        }
        self.model_deployer.get_model_metadata.return_value = metadata

        # Create a model deployed event
        event = Event(
            topic=EventTopics.MODEL_DEPLOYED,
            data={
                "model_id": self.model_id,
                "model_type": self.model_type
            }
        )

        # Handle event
        self.monitor._handle_model_deployed(event)

        # Verify model deployer was called
        self.model_deployer.get_model_metadata.assert_called_with(self.model_id)

        # Verify reference data was added
        self.assertIn(self.model_id, self.monitor.reference_data)
        self.assertEqual(self.monitor.reference_data[self.model_id]["model_type"], self.model_type)
        self.assertEqual(self.monitor.reference_data[self.model_id]["metrics"], metadata["validation_results"]["metrics"])
        self.assertEqual(self.monitor.reference_data[self.model_id]["feature_importances"], metadata["validation_results"]["feature_importances"])


if __name__ == '__main__':
    unittest.main()