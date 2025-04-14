"""
Unit tests for the ModelDeployer class.

This test suite verifies that the ModelDeployer properly handles model deployment,
including verification of model integrity, environment setup, performance tracking,
and failover mechanisms.
"""

import os
import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np
import datetime
import tempfile
import json
import logging

# Import the module to test
from models.production.model_deployer import ModelDeployer, DeploymentError
from models.production.deployment_config import DeploymentConfig

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestModelDeployer(unittest.TestCase):
    """Test suite for the ModelDeployer class."""

    def setUp(self):
        """Set up test environment before each test case."""
        # Create mock dependencies
        self.mock_state_manager = Mock()
        self.mock_performance_tracker = Mock()
        self.mock_env_validator = Mock()
        self.mock_model_validator = Mock()

        # Create a temporary directory for test models
        self.test_dir = tempfile.mkdtemp()

        # Create a basic deployment config
        self.config = DeploymentConfig(
            model_path="test_model.pkl",
            environment="production",
            performance_threshold=0.5,
            max_memory_usage=1024,
            auto_failover=True,
            version="1.0.0"
        )

        # Create the ModelDeployer instance
        self.deployer = ModelDeployer(
            state_manager=self.mock_state_manager,
            performance_tracker=self.mock_performance_tracker,
            env_validator=self.mock_env_validator,
            model_validator=self.mock_model_validator
        )

        # Create a mock model for testing
        self.mock_model = Mock()
        self.mock_model.predict = Mock(return_value=np.array([0.1, 0.2, 0.3]))

        # Set up successful validations by default
        self.mock_env_validator.validate_environment.return_value = True
        self.mock_model_validator.validate_model.return_value = True

    def tearDown(self):
        """Clean up test environment after each test case."""
        # Remove temporary test directory
        import shutil
        shutil.rmtree(self.test_dir)

    def test_init(self):
        """Test initialization of ModelDeployer."""
        self.assertEqual(self.deployer.state_manager, self.mock_state_manager)
        self.assertEqual(self.deployer.performance_tracker, self.mock_performance_tracker)
        self.assertEqual(self.deployer.env_validator, self.mock_env_validator)
        self.assertEqual(self.deployer.model_validator, self.mock_model_validator)
        self.assertFalse(self.deployer.is_deployed)
        self.assertIsNone(self.deployer.active_model)
        self.assertIsNone(self.deployer.active_config)
        self.assertEqual(self.deployer.deployment_history, [])

    def test_deploy_model_success(self):
        """Test successful model deployment."""
        # Arrange
        self.mock_model_validator.load_model.return_value = self.mock_model

        # Act
        result = self.deployer.deploy_model(self.config)

        # Assert
        self.assertTrue(result)
        self.assertTrue(self.deployer.is_deployed)
        self.assertEqual(self.deployer.active_model, self.mock_model)
        self.assertEqual(self.deployer.active_config, self.config)
        self.assertEqual(len(self.deployer.deployment_history), 1)
        self.mock_env_validator.validate_environment.assert_called_once_with(self.config.environment)
        self.mock_model_validator.validate_model.assert_called_once()
        self.mock_model_validator.load_model.assert_called_once_with(self.config.model_path)

    def test_deploy_model_env_validation_failure(self):
        """Test deployment failure due to environment validation."""
        # Arrange
        self.mock_env_validator.validate_environment.return_value = False

        # Act & Assert
        with self.assertRaises(DeploymentError):
            self.deployer.deploy_model(self.config)

        self.assertFalse(self.deployer.is_deployed)
        self.assertIsNone(self.deployer.active_model)
        self.mock_model_validator.load_model.assert_not_called()

    def test_deploy_model_model_validation_failure(self):
        """Test deployment failure due to model validation."""
        # Arrange
        self.mock_model_validator.validate_model.return_value = False

        # Act & Assert
        with self.assertRaises(DeploymentError):
            self.deployer.deploy_model(self.config)

        self.assertFalse(self.deployer.is_deployed)
        self.assertIsNone(self.deployer.active_model)
        self.mock_model_validator.load_model.assert_not_called()

    def test_deploy_model_load_failure(self):
        """Test deployment failure due to model loading error."""
        # Arrange
        self.mock_model_validator.load_model.side_effect = Exception("Failed to load model")

        # Act & Assert
        with self.assertRaises(DeploymentError):
            self.deployer.deploy_model(self.config)

        self.assertFalse(self.deployer.is_deployed)
        self.assertIsNone(self.deployer.active_model)

    def test_undeploy_model(self):
        """Test undeploying a model."""
        # Arrange - first deploy a model
        self.mock_model_validator.load_model.return_value = self.mock_model
        self.deployer.deploy_model(self.config)

        # Act
        self.deployer.undeploy_model()

        # Assert
        self.assertFalse(self.deployer.is_deployed)
        self.assertIsNone(self.deployer.active_model)
        self.assertIsNone(self.deployer.active_config)
        # Deployment history should still contain the record
        self.assertEqual(len(self.deployer.deployment_history), 1)

    def test_redeploy_model(self):
        """Test redeploying a model."""
        # Arrange - first deploy a model
        self.mock_model_validator.load_model.return_value = self.mock_model
        self.deployer.deploy_model(self.config)

        # Create a new config for redeployment
        new_config = DeploymentConfig(
            model_path="new_model.pkl",
            environment="production",
            performance_threshold=0.6,
            max_memory_usage=2048,
            auto_failover=True,
            version="2.0.0"
        )

        # Act
        result = self.deployer.deploy_model(new_config)

        # Assert
        self.assertTrue(result)
        self.assertTrue(self.deployer.is_deployed)
        self.assertEqual(self.deployer.active_config, new_config)
        self.assertEqual(len(self.deployer.deployment_history), 2)

    @patch("models.production.model_deployer.time")
    def test_deployment_timestamps(self, mock_time):
        """Test that deployment history records timestamps correctly."""
        # Arrange
        mock_time.time.return_value = 1000.0
        self.mock_model_validator.load_model.return_value = self.mock_model

        # Act
        self.deployer.deploy_model(self.config)

        # Assert
        self.assertEqual(len(self.deployer.deployment_history), 1)
        self.assertEqual(self.deployer.deployment_history[0]["timestamp"], 1000.0)
        self.assertEqual(self.deployer.deployment_history[0]["config"].version, "1.0.0")
        self.assertEqual(self.deployer.deployment_history[0]["status"], "success")

    def test_get_model_performance(self):
        """Test getting performance metrics for the deployed model."""
        # Arrange
        self.mock_model_validator.load_model.return_value = self.mock_model
        self.deployer.deploy_model(self.config)

        # Mock performance metrics
        mock_metrics = {
            "accuracy": 0.95,
            "sharpe_ratio": 1.8,
            "max_drawdown": 0.15
        }
        self.mock_performance_tracker.get_model_metrics.return_value = mock_metrics

        # Act
        metrics = self.deployer.get_model_performance()

        # Assert
        self.assertEqual(metrics, mock_metrics)
        self.mock_performance_tracker.get_model_metrics.assert_called_once_with(self.mock_model)

    def test_get_model_performance_not_deployed(self):
        """Test getting performance when no model is deployed."""
        # Act & Assert
        with self.assertRaises(DeploymentError):
            self.deployer.get_model_performance()

    def test_predict_with_model(self):
        """Test making predictions with the deployed model."""
        # Arrange
        self.mock_model_validator.load_model.return_value = self.mock_model
        self.deployer.deploy_model(self.config)

        test_data = np.array([[1, 2, 3], [4, 5, 6]])

        # Act
        predictions = self.deployer.predict(test_data)

        # Assert
        np.testing.assert_array_equal(predictions, np.array([0.1, 0.2, 0.3]))
        self.mock_model.predict.assert_called_once_with(test_data)

    def test_predict_not_deployed(self):
        """Test making predictions when no model is deployed."""
        # Arrange
        test_data = np.array([[1, 2, 3], [4, 5, 6]])

        # Act & Assert
        with self.assertRaises(DeploymentError):
            self.deployer.predict(test_data)

    def test_deployment_metrics_tracking(self):
        """Test that deployment metrics are properly tracked."""
        # Arrange
        self.mock_model_validator.load_model.return_value = self.mock_model

        # Act
        self.deployer.deploy_model(self.config)

        # Assert
        self.mock_state_manager.record_deployment.assert_called_once()
        call_args = self.mock_state_manager.record_deployment.call_args[0]
        self.assertEqual(call_args[0], self.config.model_path)
        self.assertEqual(call_args[1], self.config.version)
        self.assertEqual(call_args[2], self.config.environment)

    @patch("models.production.model_deployer.ModelDeployer._check_performance")
    def test_automatic_performance_check(self, mock_check_performance):
        """Test that performance is automatically checked during deployment."""
        # Arrange
        self.mock_model_validator.load_model.return_value = self.mock_model
        mock_check_performance.return_value = True

        # Act
        self.deployer.deploy_model(self.config)

        # Assert
        mock_check_performance.assert_called_once_with(self.mock_model, self.config.performance_threshold)

    @patch("models.production.model_deployer.ModelDeployer._check_performance")
    def test_performance_check_failure(self, mock_check_performance):
        """Test handling of performance check failure."""
        # Arrange
        self.mock_model_validator.load_model.return_value = self.mock_model
        mock_check_performance.return_value = False

        # Act & Assert
        with self.assertRaises(DeploymentError):
            self.deployer.deploy_model(self.config)

        self.assertFalse(self.deployer.is_deployed)
        mock_check_performance.assert_called_once_with(self.mock_model, self.config.performance_threshold)

    def test_list_deployments(self):
        """Test listing all past deployments."""
        # Arrange - deploy multiple models
        self.mock_model_validator.load_model.return_value = self.mock_model

        self.deployer.deploy_model(self.config)

        new_config = DeploymentConfig(
            model_path="new_model.pkl",
            environment="production",
            performance_threshold=0.6,
            max_memory_usage=2048,
            auto_failover=True,
            version="2.0.0"
        )

        self.deployer.deploy_model(new_config)

        # Act
        deployments = self.deployer.list_deployments()

        # Assert
        self.assertEqual(len(deployments), 2)
        self.assertEqual(deployments[0]["config"].version, "1.0.0")
        self.assertEqual(deployments[1]["config"].version, "2.0.0")

    def test_rollback_deployment(self):
        """Test rolling back to a previous deployment."""
        # Arrange - deploy multiple models
        self.mock_model_validator.load_model.return_value = self.mock_model

        # First deployment
        self.deployer.deploy_model(self.config)

        # Second deployment with new config
        new_config = DeploymentConfig(
            model_path="new_model.pkl",
            environment="production",
            performance_threshold=0.6,
            max_memory_usage=2048,
            auto_failover=True,
            version="2.0.0"
        )

        self.deployer.deploy_model(new_config)

        # Act - rollback to first deployment
        result = self.deployer.rollback_deployment()

        # Assert
        self.assertTrue(result)
        self.assertEqual(self.deployer.active_config.version, "1.0.0")
        self.assertEqual(len(self.deployer.deployment_history), 3)  # Original 2 plus rollback
        self.assertEqual(self.deployer.deployment_history[-1]["action"], "rollback")

    def test_rollback_no_previous_deployment(self):
        """Test rollback when there's no previous deployment."""
        # Arrange - only one deployment
        self.mock_model_validator.load_model.return_value = self.mock_model
        self.deployer.deploy_model(self.config)

        # Act & Assert
        with self.assertRaises(DeploymentError):
            self.deployer.rollback_deployment()

    def test_rollback_not_deployed(self):
        """Test rollback when no model is currently deployed."""
        # Act & Assert
        with self.assertRaises(DeploymentError):
            self.deployer.rollback_deployment()

    @patch("models.production.model_deployer.os")
    def test_export_deployment_config(self, mock_os):
        """Test exporting deployment configuration."""
        # Arrange
        self.mock_model_validator.load_model.return_value = self.mock_model
        self.deployer.deploy_model(self.config)

        mock_open = mock_os.path.exists.return_value = False

        # Mock open function
        m = unittest.mock.mock_open()
        with patch("builtins.open", m):
            # Act
            result = self.deployer.export_deployment_config("test_export.json")

            # Assert
            self.assertTrue(result)
            m.assert_called_once_with("test_export.json", "w")
            handle = m()
            self.assertEqual(handle.write.call_count, 1)

    def test_export_config_not_deployed(self):
        """Test exporting config when no model is deployed."""
        # Act & Assert
        with self.assertRaises(DeploymentError):
            self.deployer.export_deployment_config("test_export.json")

    @patch("models.production.model_deployer.json")
    def test_import_deployment_config(self, mock_json):
        """Test importing deployment configuration."""
        # Arrange
        mock_json.load.return_value = {
            "model_path": "imported_model.pkl",
            "environment": "staging",
            "performance_threshold": 0.7,
            "max_memory_usage": 1500,
            "auto_failover": False,
            "version": "3.0.0"
        }

        # Mock open function
        m = unittest.mock.mock_open()
        with patch("builtins.open", m):
            # Act
            config = self.deployer.import_deployment_config("test_import.json")

            # Assert
            self.assertEqual(config.model_path, "imported_model.pkl")
            self.assertEqual(config.environment, "staging")
            self.assertEqual(config.performance_threshold, 0.7)
            self.assertEqual(config.max_memory_usage, 1500)
            self.assertFalse(config.auto_failover)
            self.assertEqual(config.version, "3.0.0")

    @patch("builtins.open")
    def test_import_config_file_not_found(self, mock_open):
        """Test handling of file not found during import."""
        # Arrange
        mock_open.side_effect = FileNotFoundError("File not found")

        # Act & Assert
        with self.assertRaises(DeploymentError):
            self.deployer.import_deployment_config("nonexistent.json")

    @patch("models.production.model_deployer.json")
    @patch("builtins.open")
    def test_import_config_invalid_json(self, mock_open, mock_json):
        """Test handling of invalid JSON during import."""
        # Arrange
        mock_json.load.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

        # Act & Assert
        with self.assertRaises(DeploymentError):
            self.deployer.import_deployment_config("invalid.json")

    def test_get_active_model_info(self):
        """Test getting information about the active model."""
        # Arrange
        self.mock_model_validator.load_model.return_value = self.mock_model
        self.deployer.deploy_model(self.config)

        # Act
        info = self.deployer.get_active_model_info()

        # Assert
        self.assertEqual(info["model_path"], self.config.model_path)
        self.assertEqual(info["version"], self.config.version)
        self.assertEqual(info["environment"], self.config.environment)
        self.assertTrue("deployment_time" in info)
        self.assertTrue(info["is_deployed"])

    def test_get_model_info_not_deployed(self):
        """Test getting model info when no model is deployed."""
        # Act
        info = self.deployer.get_active_model_info()

        # Assert
        self.assertFalse(info["is_deployed"])
        self.assertIsNone(info.get("model_path"))
        self.assertIsNone(info.get("version"))

    @patch("models.production.model_deployer.ModelDeployer._check_health")
    def test_health_check_success(self, mock_check_health):
        """Test successful health check."""
        # Arrange
        self.mock_model_validator.load_model.return_value = self.mock_model
        self.deployer.deploy_model(self.config)
        mock_check_health.return_value = {"status": "healthy", "memory_usage": 512, "response_time": 0.05}

        # Act
        health = self.deployer.check_deployment_health()

        # Assert
        self.assertEqual(health["status"], "healthy")
        mock_check_health.assert_called_once_with(self.mock_model)

    def test_health_check_not_deployed(self):
        """Test health check when no model is deployed."""
        # Act & Assert
        with self.assertRaises(DeploymentError):
            self.deployer.check_deployment_health()

    @patch("models.production.model_deployer.ModelDeployer._check_health")
    def test_auto_failover(self, mock_check_health):
        """Test automatic failover when health check fails."""
        # Arrange
        self.mock_model_validator.load_model.return_value = self.mock_model
        self.deployer.deploy_model(self.config)

        # Mock backup model
        backup_config = DeploymentConfig(
            model_path="backup_model.pkl",
            environment="production",
            performance_threshold=0.4,
            max_memory_usage=512,
            auto_failover=True,
            version="0.9.0"
        )
        self.deployer.backup_config = backup_config
        mock_backup_model = Mock()
        self.mock_model_validator.load_model.return_value = mock_backup_model

        # Setup health check to fail
        mock_check_health.return_value = {"status": "unhealthy", "memory_usage": 2048, "error": "Memory limit exceeded"}

        # Act
        with patch.object(self.deployer, '_trigger_failover', wraps=self.deployer._trigger_failover) as mock_failover:
            health = self.deployer.check_deployment_health()

            # Assert
            mock_failover.assert_called_once()
            self.assertEqual(self.deployer.active_model, mock_backup_model)
            self.assertEqual(self.deployer.active_config, backup_config)
            self.assertEqual(len(self.deployer.deployment_history), 2)
            self.assertEqual(self.deployer.deployment_history[-1]["action"], "failover")

    def test_disable_auto_failover(self):
        """Test disabling auto-failover."""
        # Arrange
        self.mock_model_validator.load_model.return_value = self.mock_model
        config_no_failover = DeploymentConfig(
            model_path="test_model.pkl",
            environment="production",
            performance_threshold=0.5,
            max_memory_usage=1024,
            auto_failover=False,  # Auto-failover disabled
            version="1.0.0"
        )
        self.deployer.deploy_model(config_no_failover)

        # Act & Assert
        with patch.object(self.deployer, '_check_health') as mock_check_health:
            mock_check_health.return_value = {"status": "unhealthy", "error": "Test error"}

            with patch.object(self.deployer, '_trigger_failover') as mock_failover:
                health = self.deployer.check_deployment_health()

                # Failover should not be triggered
                mock_failover.assert_not_called()


if __name__ == '__main__':
    unittest.main()