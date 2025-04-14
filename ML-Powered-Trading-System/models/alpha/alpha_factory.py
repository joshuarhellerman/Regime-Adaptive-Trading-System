"""
Alpha Factory Module

This module provides factory methods for creating and managing alpha model instances.
It handles model registration, instantiation, dependency injection, and lifecycle management.
"""

import logging
import importlib
import inspect
import copy
import time
from typing import Dict, List, Any, Optional, Type, Set, Tuple

from models.alpha.alpha_model_interface import AlphaModel
from core.event_bus import EventBus

# Configure logger
logger = logging.getLogger(__name__)

class AlphaFactory:
    """
    Factory class for alpha models.

    This class handles the creation, registration, and management of alpha models,
    including dependency injection and lifecycle management.
    """

    # Registry of alpha model classes
    _model_classes: Dict[str, Type[AlphaModel]] = {}

    # Active model instances
    _model_instances: Dict[str, AlphaModel] = {}

    # Model metadata
    _model_metadata: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register_model(cls, model_class: Type[AlphaModel], model_type: str,
                      description: str = "", metadata: Dict[str, Any] = None) -> None:
        """
        Register an alpha model class.

        Args:
            model_class: The model class to register
            model_type: Type identifier for the model
            description: Description of the model
            metadata: Additional metadata about the model
        """
        if model_type in cls._model_classes:
            logger.warning(f"Alpha model type '{model_type}' already registered. Overwriting.")

        cls._model_classes[model_type] = model_class
        cls._model_metadata[model_type] = {
            'description': description,
            'class_name': model_class.__name__,
            'module': model_class.__module__,
            'metadata': metadata or {},
            'registered_at': time.time()
        }

        logger.info(f"Registered alpha model '{model_type}' from class {model_class.__name__}")

    @classmethod
    def create_model(cls, model_type: str, model_id: str = None,
                   parameters: Dict[str, Any] = None) -> Optional[AlphaModel]:
        """
        Create an instance of an alpha model.

        Args:
            model_type: Type of model to create
            model_id: Optional specific ID for the model
            parameters: Parameters to initialize the model with

        Returns:
            Alpha model instance or None if creation failed
        """
        if model_type not in cls._model_classes:
            logger.error(f"Alpha model type '{model_type}' not registered")
            return None

        model_class = cls._model_classes[model_type]

        try:
            # Create instance
            model = model_class(model_id=model_id, parameters=parameters or {})

            # Store in active instances
            cls._model_instances[model.model_id] = model

            # Emit creation event
            EventBus.emit("alpha.created", {
                'model_id': model.model_id,
                'model_type': model_type,
                'timestamp': time.time()
            })

            logger.info(f"Created alpha model instance '{model.model_id}' of type '{model_type}'")
            return model
        except Exception as e:
            logger.error(f"Error creating alpha model '{model_type}': {str(e)}")
            return None

    @classmethod
    def get_model(cls, model_id: str) -> Optional[AlphaModel]:
        """
        Get an alpha model instance by ID.

        Args:
            model_id: ID of the model to retrieve

        Returns:
            Alpha model instance or None if not found
        """
        if model_id not in cls._model_instances:
            logger.warning(f"Alpha model '{model_id}' not found")
            return None

        return cls._model_instances[model_id]

    @classmethod
    def destroy_model(cls, model_id: str) -> bool:
        """
        Destroy an alpha model instance.

        Args:
            model_id: ID of the model to destroy

        Returns:
            bool: True if successful, False otherwise
        """
        if model_id not in cls._model_instances:
            logger.warning(f"Alpha model '{model_id}' not found for destruction")
            return False

        # Get model type before removing
        model = cls._model_instances[model_id]
        model_type = None

        for type_id, metadata in cls._model_metadata.items():
            if metadata['class_name'] == model.__class__.__name__:
                model_type = type_id
                break

        # Remove from instances
        del cls._model_instances[model_id]

        # Emit destroyed event
        EventBus.emit("alpha.destroyed", {
            'model_id': model_id,
            'model_type': model_type,
            'timestamp': time.time()
        })

        logger.info(f"Destroyed alpha model instance '{model_id}'")
        return True

    @classmethod
    def get_registered_models(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered model types.

        Returns:
            Dict mapping model types to their metadata
        """
        return copy.deepcopy(cls._model_metadata)

    @classmethod
    def get_active_instances(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all active model instances.

        Returns:
            Dict mapping model IDs to their info
        """
        return {
            model_id: model.get_model_info()
            for model_id, model in cls._model_instances.items()
        }

    @classmethod
    def discover_models(cls, package_name: str = "models.alpha.implementations") -> int:
        """
        Automatically discover and register alpha models in a package.

        Args:
            package_name: Name of the package to search

        Returns:
            int: Number of models discovered and registered
        """
        try:
            # Import the package
            package = importlib.import_module(package_name)

            # Find all modules in the package
            modules = []
            if hasattr(package, "__path__"):
                import pkgutil
                for _, name, _ in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
                    modules.append(name)

            count = 0

            # Import each module and look for alpha model classes
            for module_name in modules:
                try:
                    module = importlib.import_module(module_name)

                    # Find classes that inherit from AlphaModel
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (
                            issubclass(obj, AlphaModel) and
                            obj != AlphaModel and
                            hasattr(obj, 'MODEL_TYPE')
                        ):
                            # Register the model
                            cls.register_model(
                                model_class=obj,
                                model_type=getattr(obj, 'MODEL_TYPE'),
                                description=getattr(obj, 'DESCRIPTION', ''),
                                metadata=getattr(obj, 'METADATA', {})
                            )
                            count += 1
                except Exception as e:
                    logger.error(f"Error importing module '{module_name}': {str(e)}")

            logger.info(f"Discovered and registered {count} alpha models from package '{package_name}'")
            return count
        except Exception as e:
            logger.error(f"Error discovering alpha models: {str(e)}")
            return 0

    @classmethod
    def serialize_model(cls, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Serialize an alpha model for persistence.

        Args:
            model_id: ID of the model to serialize

        Returns:
            Dict with serialized model data or None if model not found
        """
        model = cls.get_model(model_id)
        if not model:
            return None

        # Get model type
        model_type = None
        for type_id, metadata in cls._model_metadata.items():
            if metadata['class_name'] == model.__class__.__name__:
                model_type = type_id
                break

        if not model_type:
            logger.error(f"Could not determine type for model '{model_id}'")
            return None

        # Get state if the model supports it
        state = {}
        if hasattr(model, 'get_state') and callable(getattr(model, 'get_state')):
            state = model.get_state()

        # Get active signals
        signals = [signal.to_dict() for signal in model.get_active_signals()]

        return {
            'model_id': model.model_id,
            'model_type': model_type,
            'parameters': copy.deepcopy(model.parameters),
            'state': state,
            'signals': signals,
            'serialized_at': time.time()
        }

    @classmethod
    def deserialize_model(cls, data: Dict[str, Any]) -> Optional[AlphaModel]:
        """
        Create an alpha model from serialized data.

        Args:
            data: Serialized model data

        Returns:
            Alpha model instance or None if creation failed
        """
        model_id = data.get('model_id')
        model_type = data.get('model_type')
        parameters = data.get('parameters', {})
        state = data.get('state', {})
        signals = data.get('signals', [])

        if not model_id or not model_type:
            logger.error("Serialized data missing model_id or model_type")
            return None

        # Create model instance
        model = cls.create_model(model_type, model_id, parameters)

        if not model:
            return None

        # Restore state if the model supports it
        if state and hasattr(model, 'set_state') and callable(getattr(model, 'set_state')):
            try:
                model.set_state(state)
                logger.info(f"Restored state for alpha model '{model_id}'")
            except Exception as e:
                logger.error(f"Error restoring state for alpha model '{model_id}': {str(e)}")

        # Restore signals
        if signals:
            from models.alpha.alpha_model_interface import AlphaSignal
            try:
                for signal_data in signals:
                    signal = AlphaSignal.from_dict(signal_data)
                    model.signals[signal.signal_id] = signal
                logger.info(f"Restored {len(signals)} signals for alpha model '{model_id}'")
            except Exception as e:
                logger.error(f"Error restoring signals for alpha model '{model_id}': {str(e)}")

        return model