"""
Unit tests for the AlphaRegistry module.

This module contains tests for the AlphaRegistry functionality including:
- Model registration
- Model retrieval
- Version management
- Evaluator functionality
- Finding models for instruments
"""

import unittest
from unittest.mock import MagicMock, patch
import time

from models.alpha.alpha_registry import AlphaRegistry
from models.alpha.alpha_model_interface import AlphaModel

class MockAlphaModel(AlphaModel):
    """Mock implementation of AlphaModel for testing."""

    def __init__(self, model_id=None):
        self.model_id = model_id

    def generate_signals(self, *args, **kwargs):
        return {"mock": "signals"}

class MockAlphaModelV2(AlphaModel):
    """Another mock implementation for testing versioning."""

    def __init__(self, model_id=None):
        self.model_id = model_id

    def generate_signals(self, *args, **kwargs):
        return {"mock": "signals_v2"}

class TestAlphaRegistry(unittest.TestCase):
    """Test cases for the AlphaRegistry class."""

    def setUp(self):
        """Set up the test environment."""
        # Clear the registry before each test
        AlphaRegistry._models = {}
        AlphaRegistry._versions = {}
        AlphaRegistry._evaluators = {}

    def test_register_model(self):
        """Test model registration."""
        # Register a model
        AlphaRegistry.register_model(
            MockAlphaModel,
            "test_model",
            "1.0.0",
            "Test model description",
            {"stable": True, "supported_instruments": ["BTC", "ETH"]}
        )

        # Check if model was registered
        self.assertIn("test_model", AlphaRegistry._models)
        self.assertIn("1.0.0", AlphaRegistry._models["test_model"])

        # Check model info
        model_info = AlphaRegistry._models["test_model"]["1.0.0"]
        self.assertEqual(model_info["class"], MockAlphaModel)
        self.assertEqual(model_info["description"], "Test model description")
        self.assertEqual(model_info["metadata"]["stable"], True)
        self.assertEqual(model_info["metadata"]["supported_instruments"], ["BTC", "ETH"])

        # Check version info
        self.assertEqual(AlphaRegistry._versions["test_model"]["latest"], "1.0.0")
        self.assertEqual(AlphaRegistry._versions["test_model"]["stable"], "1.0.0")

    def test_register_multiple_versions(self):
        """Test registering multiple versions of the same model."""
        # Register first version
        AlphaRegistry.register_model(
            MockAlphaModel,
            "test_model",
            "1.0.0",
            "V1 description",
            {"stable": True}
        )

        # Register second version
        AlphaRegistry.register_model(
            MockAlphaModelV2,
            "test_model",
            "2.0.0",
            "V2 description",
            {}
        )

        # Check if both versions are registered
        self.assertIn("1.0.0", AlphaRegistry._models["test_model"])
        self.assertIn("2.0.0", AlphaRegistry._models["test_model"])

        # Check latest version is updated
        self.assertEqual(AlphaRegistry._versions["test_model"]["latest"], "2.0.0")

        # Check stable version remains unchanged
        self.assertEqual(AlphaRegistry._versions["test_model"]["stable"], "1.0.0")

    def test_register_version_overwrite(self):
        """Test overwriting an existing version."""
        # Register initial version
        AlphaRegistry.register_model(
            MockAlphaModel,
            "test_model",
            "1.0.0",
            "Original description"
        )

        # Overwrite with new info
        AlphaRegistry.register_model(
            MockAlphaModelV2,
            "test_model",
            "1.0.0",
            "Updated description"
        )

        # Check if version was overwritten
        model_info = AlphaRegistry._models["test_model"]["1.0.0"]
        self.assertEqual(model_info["class"], MockAlphaModelV2)
        self.assertEqual(model_info["description"], "Updated description")

    def test_get_model_class(self):
        """Test retrieving model class."""
        # Register a model
        AlphaRegistry.register_model(
            MockAlphaModel,
            "test_model",
            "1.0.0"
        )

        # Get the model class
        model_class = AlphaRegistry.get_model_class("test_model")
        self.assertEqual(model_class, MockAlphaModel)

        # Get non-existent model
        model_class = AlphaRegistry.get_model_class("non_existent_model")
        self.assertIsNone(model_class)

        # Get non-existent version
        model_class = AlphaRegistry.get_model_class("test_model", "9.9.9")
        self.assertIsNone(model_class)

    def test_get_model_info(self):
        """Test retrieving model info."""
        # Register a model
        AlphaRegistry.register_model(
            MockAlphaModel,
            "test_model",
            "1.0.0",
            "Test description",
            {"key": "value"}
        )

        # Get model info
        model_info = AlphaRegistry.get_model_info("test_model")

        # Check info contents
        self.assertEqual(model_info["description"], "Test description")
        self.assertEqual(model_info["metadata"]["key"], "value")

        # Check class reference is not included
        self.assertNotIn("class", model_info)

        # Test non-existent model
        self.assertIsNone(AlphaRegistry.get_model_info("non_existent"))

    def test_get_available_models(self):
        """Test retrieving all available models."""
        # Register multiple models
        AlphaRegistry.register_model(
            MockAlphaModel,
            "model1",
            "1.0.0",
            "Model 1",
            {"stable": True}
        )

        AlphaRegistry.register_model(
            MockAlphaModelV2,
            "model2",
            "2.0.0",
            "Model 2"
        )

        # Add another version to model1
        AlphaRegistry.register_model(
            MockAlphaModelV2,
            "model1",
            "1.1.0",
            "Model 1 update"
        )

        # Get all models
        models = AlphaRegistry.get_available_models()

        # Check models
        self.assertIn("model1", models)
        self.assertIn("model2", models)

        # Check model1 details
        self.assertEqual(sorted(models["model1"]["versions"]), ["1.0.0", "1.1.0"])
        self.assertEqual(models["model1"]["latest"], "1.1.0")
        self.assertEqual(models["model1"]["stable"], "1.0.0")
        self.assertEqual(models["model1"]["description"], "Model 1 update")

        # Check model2 details
        self.assertEqual(models["model2"]["versions"], ["2.0.0"])
        self.assertEqual(models["model2"]["latest"], "2.0.0")
        self.assertNotIn("stable", models["model2"])

    def test_get_model_versions(self):
        """Test retrieving all versions of a model."""
        # Register multiple versions
        AlphaRegistry.register_model(MockAlphaModel, "test_model", "1.0.0")
        AlphaRegistry.register_model(MockAlphaModel, "test_model", "1.2.0")
        AlphaRegistry.register_model(MockAlphaModel, "test_model", "1.10.0")
        AlphaRegistry.register_model(MockAlphaModel, "test_model", "2.0.0")

        # Get versions
        versions = AlphaRegistry.get_model_versions("test_model")

        # Check versions are sorted correctly
        self.assertEqual(versions, ["1.0.0", "1.2.0", "1.10.0", "2.0.0"])

        # Check non-existent model
        self.assertEqual(AlphaRegistry.get_model_versions("non_existent"), [])

    def test_unregister_model_version(self):
        """Test unregistering a specific model version."""
        # Register multiple versions
        AlphaRegistry.register_model(
            MockAlphaModel,
            "test_model",
            "1.0.0",
            metadata={"stable": True}
        )
        AlphaRegistry.register_model(MockAlphaModelV2, "test_model", "2.0.0")

        # Unregister the latest version
        result = AlphaRegistry.unregister_model("test_model", "2.0.0")
        self.assertTrue(result)

        # Check version was removed
        self.assertNotIn("2.0.0", AlphaRegistry._models["test_model"])

        # Check latest version was updated
        self.assertEqual(AlphaRegistry._versions["test_model"]["latest"], "1.0.0")

        # Unregister the stable version
        result = AlphaRegistry.unregister_model("test_model", "1.0.0")
        self.assertTrue(result)

        # Check model was removed completely (no versions left)
        self.assertNotIn("test_model", AlphaRegistry._models)
        self.assertNotIn("test_model", AlphaRegistry._versions)

        # Try to unregister non-existent model
        result = AlphaRegistry.unregister_model("non_existent", "1.0.0")
        self.assertFalse(result)

    def test_unregister_entire_model(self):
        """Test unregistering all versions of a model."""
        # Register multiple versions
        AlphaRegistry.register_model(MockAlphaModel, "test_model", "1.0.0")
        AlphaRegistry.register_model(MockAlphaModelV2, "test_model", "2.0.0")

        # Unregister all versions
        result = AlphaRegistry.unregister_model("test_model")
        self.assertTrue(result)

        # Check model was removed completely
        self.assertNotIn("test_model", AlphaRegistry._models)
        self.assertNotIn("test_model", AlphaRegistry._versions)

    def test_register_evaluator(self):
        """Test registering an evaluator function."""
        # Create mock evaluator
        mock_evaluator = MagicMock(return_value={"accuracy": 0.85})

        # Register evaluator
        AlphaRegistry.register_evaluator("test_evaluator", mock_evaluator)

        # Check evaluator was registered
        self.assertIn("test_evaluator", AlphaRegistry._evaluators)
        self.assertEqual(AlphaRegistry._evaluators["test_evaluator"], mock_evaluator)

    def test_evaluate_model(self):
        """Test evaluating a model with a registered evaluator."""
        # Register model
        AlphaRegistry.register_model(MockAlphaModel, "test_model", "1.0.0")

        # Create and register mock evaluator
        mock_evaluator = MagicMock(return_value={"accuracy": 0.85})
        AlphaRegistry.register_evaluator("test_evaluator", mock_evaluator)

        # Evaluate model
        result = AlphaRegistry.evaluate_model(
            "test_model",
            "test_evaluator",
            {"test_data": "data"}
        )

        # Check evaluator was called with correct args
        mock_evaluator.assert_called_once()
        model_arg = mock_evaluator.call_args[0][0]
        self.assertIsInstance(model_arg, MockAlphaModel)
        self.assertEqual(mock_evaluator.call_args[0][1], {"test_data": "data"})

        # Check result
        self.assertEqual(result["accuracy"], 0.85)
        self.assertEqual(result["model_id"], "test_model")
        self.assertEqual(result["version"], "1.0.0")

        # Test evaluating non-existent model
        result = AlphaRegistry.evaluate_model(
            "non_existent",
            "test_evaluator",
            {"test_data": "data"}
        )
        self.assertIsNone(result)

        # Test evaluating with non-existent evaluator
        result = AlphaRegistry.evaluate_model(
            "test_model",
            "non_existent_evaluator",
            {"test_data": "data"}
        )
        self.assertIsNone(result)

    def test_find_models_for_instruments(self):
        """Test finding models for specific instruments."""
        # Register models with different supported instruments
        AlphaRegistry.register_model(
            MockAlphaModel,
            "model1",
            "1.0.0",
            metadata={"supported_instruments": ["BTC", "ETH"]}
        )

        AlphaRegistry.register_model(
            MockAlphaModelV2,
            "model2",
            "1.0.0",
            metadata={"supported_instruments": ["ETH", "LTC"]}
        )

        AlphaRegistry.register_model(
            MockAlphaModel,
            "model3",
            "1.0.0",
            metadata={}  # No supported instruments specified (supports all)
        )

        # Find models for specific instruments
        result = AlphaRegistry.find_models_for_instruments({"BTC", "ETH"})

        # Check results
        self.assertEqual(sorted(result["BTC"]), ["model1", "model3"])
        self.assertEqual(sorted(result["ETH"]), ["model1", "model2", "model3"])

    def test_version_comparison(self):
        """Test version comparison logic."""
        # Test basic version comparison
        self.assertEqual(AlphaRegistry._compare_versions("1.0.0", "1.0.0"), 0)
        self.assertEqual(AlphaRegistry._compare_versions("1.0.0", "1.0.1"), -1)
        self.assertEqual(AlphaRegistry._compare_versions("1.1.0", "1.0.0"), 1)
        self.assertEqual(AlphaRegistry._compare_versions("2.0.0", "1.9.9"), 1)

        # Test non-numeric version parts
        self.assertEqual(AlphaRegistry._compare_versions("1.0.0-alpha", "1.0.0"), -1)
        self.assertEqual(AlphaRegistry._compare_versions("1.0.0", "1.0.0-beta"), 1)

    def test_parse_version(self):
        """Test version parsing logic."""
        # Test basic version parsing
        self.assertEqual(AlphaRegistry._parse_version("1.0.0"), (1, 0, 0))
        self.assertEqual(AlphaRegistry._parse_version("2.10.5"), (2, 10, 5))

        # Test version with non-numeric parts
        self.assertEqual(AlphaRegistry._parse_version("1.0.0-alpha"), (1, 0, 0, "-alpha"))
        self.assertEqual(AlphaRegistry._parse_version("1.0.beta"), (1, 0, "beta"))

if __name__ == "__main__":
    unittest.main()