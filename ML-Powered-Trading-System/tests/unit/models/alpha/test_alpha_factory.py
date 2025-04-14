"""
Unit tests for the AlphaFactory class.

This module contains tests that verify the functionality of the AlphaFactory class,
including model registration, creation, lifecycle management, and serialization.
"""

import unittest
from unittest import mock
import time
from typing import Dict, Any, Optional, List

from models.alpha.alpha_factory import AlphaFactory
from models.alpha.alpha_model_interface import AlphaModel, AlphaSignal
from core.event_bus import EventBus


class MockAlphaModel(AlphaModel):
    """Mock implementation of AlphaModel for testing."""

    MODEL_TYPE = "mock_alpha"
    DESCRIPTION = "Mock alpha model for testing"
    METADATA = {"version": "1.0.0", "author": "Test"}

    def __init__(self, model_id: str = None, parameters: Dict[str, Any] = None):
        super().__init__(model_id, parameters)
        self.state_data = {}

    def get_state(self) -> Dict[str, Any]:
        return self.state_data

    def set_state(self, state: Dict[str, Any]) -> None:
        self.state_data = state

    def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock implementation of evaluate."""
        return {"result": "mock_result"}


class AnotherMockAlphaModel(AlphaModel):
    """Another mock implementation for testing multiple models."""

    MODEL_TYPE = "another_mock_alpha"
    DESCRIPTION = "Another mock alpha model"

    def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock implementation of evaluate."""
        return {"result": "another_mock_result"}


class TestAlphaFactory(unittest.TestCase):
    """Test cases for the AlphaFactory class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Clear factory state before each test
        AlphaFactory._model_classes = {}
        AlphaFactory._model_instances = {}
        AlphaFactory._model_metadata = {}

        # Register mock models for testing
        AlphaFactory.register_model(
            MockAlphaModel,
            MockAlphaModel.MODEL_TYPE,
            MockAlphaModel.DESCRIPTION,
            MockAlphaModel.METADATA
        )

    def tearDown(self):
        """Clean up after each test method."""
        # Clear factory state after each test
        AlphaFactory._model_classes = {}
        AlphaFactory._model_instances = {}
        AlphaFactory._model_metadata = {}

    def test_register_model(self):
        """Test model registration functionality."""
        # Test that the model was registered in setUp
        self.assertIn(MockAlphaModel.MODEL_TYPE, AlphaFactory._model_classes)
        self.assertEqual(AlphaFactory._model_classes[MockAlphaModel.MODEL_TYPE], MockAlphaModel)

        # Test metadata is stored correctly
        metadata = AlphaFactory._model_metadata[MockAlphaModel.MODEL_TYPE]
        self.assertEqual(metadata['description'], MockAlphaModel.DESCRIPTION)
        self.assertEqual(metadata['class_name'], MockAlphaModel.__name__)
        self.assertEqual(metadata['metadata'], MockAlphaModel.METADATA)

        # Test registering a second model
        AlphaFactory.register_model(
            AnotherMockAlphaModel,
            AnotherMockAlphaModel.MODEL_TYPE,
            AnotherMockAlphaModel.DESCRIPTION
        )
        self.assertIn(AnotherMockAlphaModel.MODEL_TYPE, AlphaFactory._model_classes)

        # Test overwriting an existing registration
        class OverwriteMockModel(AlphaModel):
            MODEL_TYPE = MockAlphaModel.MODEL_TYPE

            def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
                return {"result": "overwrite"}

        with mock.patch('logging.Logger.warning') as mock_warning:
            AlphaFactory.register_model(OverwriteMockModel, OverwriteMockModel.MODEL_TYPE)
            mock_warning.assert_called_once()
            self.assertEqual(AlphaFactory._model_classes[MockAlphaModel.MODEL_TYPE], OverwriteMockModel)

    def test_create_model(self):
        """Test model creation functionality."""
        # Test creating a model with default id
        model = AlphaFactory.create_model(MockAlphaModel.MODEL_TYPE)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, MockAlphaModel)
        self.assertIn(model.model_id, AlphaFactory._model_instances)

        # Test creating a model with specific id
        test_id = "test_model_id"
        model = AlphaFactory.create_model(MockAlphaModel.MODEL_TYPE, model_id=test_id)
        self.assertEqual(model.model_id, test_id)
        self.assertIn(test_id, AlphaFactory._model_instances)

        # Test creating a model with parameters
        params = {"param1": "value1", "param2": 42}
        model = AlphaFactory.create_model(MockAlphaModel.MODEL_TYPE, parameters=params)
        self.assertEqual(model.parameters, params)

        # Test creating a model with unknown type
        with mock.patch('logging.Logger.error') as mock_error:
            model = AlphaFactory.create_model("unknown_type")
            mock_error.assert_called_once()
            self.assertIsNone(model)

        # Test exception during model creation
        with mock.patch.object(MockAlphaModel, '__init__', side_effect=Exception("Test error")):
            with mock.patch('logging.Logger.error') as mock_error:
                model = AlphaFactory.create_model(MockAlphaModel.MODEL_TYPE)
                mock_error.assert_called_once()
                self.assertIsNone(model)

    def test_event_emission_on_create(self):
        """Test that events are emitted when models are created."""
        with mock.patch.object(EventBus, 'emit') as mock_emit:
            model = AlphaFactory.create_model(MockAlphaModel.MODEL_TYPE, model_id="test_id")

            # Check that the event was emitted with correct data
            mock_emit.assert_called_once()
            event_name, event_data = mock_emit.call_args[0]
            self.assertEqual(event_name, "alpha.created")
            self.assertEqual(event_data['model_id'], "test_id")
            self.assertEqual(event_data['model_type'], MockAlphaModel.MODEL_TYPE)
            self.assertIn('timestamp', event_data)

    def test_get_model(self):
        """Test retrieving models by ID."""
        # Create a model first
        model_id = "test_retrieve_id"
        original_model = AlphaFactory.create_model(MockAlphaModel.MODEL_TYPE, model_id=model_id)

        # Test successful retrieval
        retrieved_model = AlphaFactory.get_model(model_id)
        self.assertEqual(retrieved_model, original_model)

        # Test retrieval of non-existent model
        with mock.patch('logging.Logger.warning') as mock_warning:
            model = AlphaFactory.get_model("non_existent_id")
            mock_warning.assert_called_once()
            self.assertIsNone(model)

    def test_destroy_model(self):
        """Test destroying model instances."""
        # Create a model first
        model_id = "test_destroy_id"
        model = AlphaFactory.create_model(MockAlphaModel.MODEL_TYPE, model_id=model_id)

        # Test successful destruction
        with mock.patch.object(EventBus, 'emit') as mock_emit:
            result = AlphaFactory.destroy_model(model_id)
            self.assertTrue(result)
            self.assertNotIn(model_id, AlphaFactory._model_instances)

            # Check that the event was emitted with correct data
            mock_emit.assert_called_once()
            event_name, event_data = mock_emit.call_args[0]
            self.assertEqual(event_name, "alpha.destroyed")
            self.assertEqual(event_data['model_id'], model_id)
            self.assertEqual(event_data['model_type'], MockAlphaModel.MODEL_TYPE)
            self.assertIn('timestamp', event_data)

        # Test destruction of non-existent model
        with mock.patch('logging.Logger.warning') as mock_warning:
            result = AlphaFactory.destroy_model("non_existent_id")
            mock_warning.assert_called_once()
            self.assertFalse(result)

    def test_get_registered_models(self):
        """Test retrieving information about registered models."""
        # Register another model type
        AlphaFactory.register_model(
            AnotherMockAlphaModel,
            AnotherMockAlphaModel.MODEL_TYPE,
            AnotherMockAlphaModel.DESCRIPTION
        )

        # Get registered models
        registered_models = AlphaFactory.get_registered_models()

        # Test that both models are in the result
        self.assertIn(MockAlphaModel.MODEL_TYPE, registered_models)
        self.assertIn(AnotherMockAlphaModel.MODEL_TYPE, registered_models)

        # Test that the returned data is a deep copy
        registered_models[MockAlphaModel.MODEL_TYPE]['description'] = "Modified"
        self.assertNotEqual(
            registered_models[MockAlphaModel.MODEL_TYPE]['description'],
            AlphaFactory._model_metadata[MockAlphaModel.MODEL_TYPE]['description']
        )

    def test_get_active_instances(self):
        """Test retrieving information about active model instances."""
        # Create models
        model_id1 = "test_instance_1"
        model_id2 = "test_instance_2"
        model1 = AlphaFactory.create_model(MockAlphaModel.MODEL_TYPE, model_id=model_id1)
        model2 = AlphaFactory.create_model(MockAlphaModel.MODEL_TYPE, model_id=model_id2)

        # Test that active instances are retrieved correctly
        active_instances = AlphaFactory.get_active_instances()
        self.assertIn(model_id1, active_instances)
        self.assertIn(model_id2, active_instances)

        # Test that the info is correct
        self.assertEqual(active_instances[model_id1], model1.get_model_info())

    @mock.patch('importlib.import_module')
    @mock.patch('pkgutil.iter_modules')
    @mock.patch('inspect.getmembers')
    def test_discover_models(self, mock_getmembers, mock_iter_modules, mock_import_module):
        """Test automatic discovery and registration of models."""
        # Set up mocks
        mock_package = mock.MagicMock()
        mock_package.__path__ = ["mock_path"]
        mock_import_module.return_value = mock_package

        mock_iter_modules.return_value = [
            (None, "package.module1", None),
            (None, "package.module2", None)
        ]

        class TestModel1(AlphaModel):
            MODEL_TYPE = "test_model_1"
            DESCRIPTION = "Test model 1"

            def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
                return {}

        class TestModel2(AlphaModel):
            MODEL_TYPE = "test_model_2"

            def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
                return {}

        # First module has two alpha models
        mock_module1 = mock.MagicMock()
        mock_getmembers.side_effect = [
            [
                ("TestModel1", TestModel1),
                ("NotAnAlphaModel", object),
                ("TestModel2", TestModel2)
            ],
            []  # Second module has no alpha models
        ]

        mock_import_module.side_effect = [mock_package, mock_module1, mock.MagicMock()]

        # Test model discovery
        count = AlphaFactory.discover_models("test.package")
        self.assertEqual(count, 2)
        self.assertIn(TestModel1.MODEL_TYPE, AlphaFactory._model_classes)
        self.assertIn(TestModel2.MODEL_TYPE, AlphaFactory._model_classes)

    def test_serialize_deserialize_model(self):
        """Test serialization and deserialization of models."""
        # Create a model with specific state and signals
        model_id = "test_serialize_id"
        parameters = {"param1": "value1", "param2": 42}
        model = AlphaFactory.create_model(MockAlphaModel.MODEL_TYPE, model_id=model_id, parameters=parameters)

        # Set up some state
        model.state_data = {"state_key": "state_value", "counter": 123}

        # Add some signals
        signal = AlphaSignal(
            symbol="TEST",
            direction="LONG",
            strength=0.75,
            metadata={"source": "test"}
        )
        model.signals[signal.signal_id] = signal

        # Serialize the model
        serialized = AlphaFactory.serialize_model(model_id)

        # Clear the factory state
        AlphaFactory._model_instances = {}

        # Deserialize the model
        deserialized_model = AlphaFactory.deserialize_model(serialized)

        # Test that the model was recreated correctly
        self.assertIsNotNone(deserialized_model)
        self.assertEqual(deserialized_model.model_id, model_id)
        self.assertEqual(deserialized_model.parameters, parameters)
        self.assertEqual(deserialized_model.state_data, model.state_data)

        # Test that signals were restored
        self.assertEqual(len(deserialized_model.signals), 1)
        restored_signal = list(deserialized_model.signals.values())[0]
        self.assertEqual(restored_signal.symbol, signal.symbol)
        self.assertEqual(restored_signal.direction, signal.direction)
        self.assertEqual(restored_signal.strength, signal.strength)
        self.assertEqual(restored_signal.metadata, signal.metadata)

    def test_serialize_non_existent_model(self):
        """Test serialization of a non-existent model."""
        serialized = AlphaFactory.serialize_model("non_existent_id")
        self.assertIsNone(serialized)

    def test_deserialize_invalid_data(self):
        """Test deserialization with invalid data."""
        # Missing model_id
        invalid_data1 = {
            'model_type': MockAlphaModel.MODEL_TYPE,
            'parameters': {}
        }
        with mock.patch('logging.Logger.error') as mock_error:
            model = AlphaFactory.deserialize_model(invalid_data1)
            mock_error.assert_called_once()
            self.assertIsNone(model)

        # Missing model_type
        invalid_data2 = {
            'model_id': 'test_id',
            'parameters': {}
        }
        with mock.patch('logging.Logger.error') as mock_error:
            model = AlphaFactory.deserialize_model(invalid_data2)
            mock_error.assert_called_once()
            self.assertIsNone(model)

        # Invalid model_type
        invalid_data3 = {
            'model_id': 'test_id',
            'model_type': 'unknown_type',
            'parameters': {}
        }
        with mock.patch('logging.Logger.error') as mock_error:
            model = AlphaFactory.deserialize_model(invalid_data3)
            mock_error.assert_called_once()
            self.assertIsNone(model)

    def test_deserialize_with_error_in_set_state(self):
        """Test deserialization when set_state raises an exception."""
        # Create a model with specific state and signals
        model_id = "test_serialize_id"
        model = AlphaFactory.create_model(MockAlphaModel.MODEL_TYPE, model_id=model_id)

        # Serialize the model
        serialized = AlphaFactory.serialize_model(model_id)

        # Clear the factory state
        AlphaFactory._model_instances = {}

        # Patch set_state to raise an exception
        with mock.patch.object(MockAlphaModel, 'set_state', side_effect=Exception("Test error")):
            with mock.patch('logging.Logger.error') as mock_error:
                model = AlphaFactory.deserialize_model(serialized)
                mock_error.assert_called_once()
                self.assertIsNotNone(model)  # Model should still be created

    def test_deserialize_with_error_in_signal_restoration(self):
        """Test deserialization when signal restoration raises an exception."""
        # Create a model with specific state and signals
        model_id = "test_serialize_id"
        model = AlphaFactory.create_model(MockAlphaModel.MODEL_TYPE, model_id=model_id)

        # Add a signal
        signal = AlphaSignal(
            symbol="TEST",
            direction="LONG",
            strength=0.75
        )
        model.signals[signal.signal_id] = signal

        # Serialize the model
        serialized = AlphaFactory.serialize_model(model_id)

        # Clear the factory state
        AlphaFactory._model_instances = {}

        # Patch AlphaSignal.from_dict to raise an exception
        with mock.patch.object(AlphaSignal, 'from_dict', side_effect=Exception("Test error")):
            with mock.patch('logging.Logger.error') as mock_error:
                model = AlphaFactory.deserialize_model(serialized)
                mock_error.assert_called_once()
                self.assertIsNotNone(model)  # Model should still be created
                self.assertEqual(len(model.signals), 0)  # But no signals should be restored


if __name__ == '__main__':
    unittest.main()