"""
Alpha Registry Module

This module provides a central registry for alpha models, allowing for discovery,
metadata management, and versioning of different model implementations.
"""

import logging
import time
from typing import Dict, List, Set, Optional, Type, Any, Callable

from models.alpha.alpha_model_interface import AlphaModel

# Configure logger
logger = logging.getLogger(__name__)

class AlphaRegistry:
    """
    Central registry for alpha models.

    This class maintains a registry of all available alpha model types,
    their metadata, and version information. It provides methods for
    registration, discovery, and selection of appropriate models.
    """

    # Registry of model classes
    _models: Dict[str, Dict[str, Any]] = {}

    # Registry of model versions
    _versions: Dict[str, Dict[str, str]] = {}

    # Registry of model evaluators
    _evaluators: Dict[str, Callable] = {}

    @classmethod
    def register_model(cls, model_class: Type[AlphaModel], model_id: str,
                     version: str, description: str = "",
                     metadata: Dict[str, Any] = None) -> None:
        """
        Register an alpha model class.

        Args:
            model_class: The alpha model class
            model_id: Unique identifier for the model type
            version: Version string (e.g., '1.0.0')
            description: Description of the model
            metadata: Additional metadata
        """
        if model_id not in cls._models:
            cls._models[model_id] = {}
            cls._versions[model_id] = {}

        if version in cls._models[model_id]:
            logger.warning(f"Alpha model '{model_id}' version '{version}' is already registered. Overwriting.")

        # Store model info
        cls._models[model_id][version] = {
            'class': model_class,
            'description': description,
            'metadata': metadata or {},
            'registered_at': time.time()
        }

        # Update latest version
        current_latest = cls._versions[model_id].get('latest')
        if not current_latest or cls._compare_versions(version, current_latest) > 0:
            cls._versions[model_id]['latest'] = version

        # Stable version remains unchanged unless explicitly specified
        if metadata and metadata.get('stable', False):
            cls._versions[model_id]['stable'] = version

        logger.info(f"Registered alpha model '{model_id}' version '{version}'")

    @classmethod
    def get_model_class(cls, model_id: str, version: str = None) -> Optional[Type[AlphaModel]]:
        """
        Get the class for a registered alpha model.

        Args:
            model_id: ID of the model type
            version: Specific version to retrieve (default: latest)

        Returns:
            The model class or None if not found
        """
        if model_id not in cls._models:
            logger.warning(f"Alpha model '{model_id}' not found in registry")
            return None

        # Determine version to use
        if not version:
            version = cls._versions[model_id].get('latest')
            if not version:
                logger.warning(f"No latest version found for alpha model '{model_id}'")
                return None

        if version not in cls._models[model_id]:
            logger.warning(f"Version '{version}' of alpha model '{model_id}' not found")
            return None

        return cls._models[model_id][version]['class']

    @classmethod
    def get_model_info(cls, model_id: str, version: str = None) -> Optional[Dict[str, Any]]:
        """
        Get information about a registered alpha model.

        Args:
            model_id: ID of the model type
            version: Specific version to retrieve (default: latest)

        Returns:
            Dict with model info or None if not found
        """
        if model_id not in cls._models:
            return None

        # Determine version to use
        if not version:
            version = cls._versions[model_id].get('latest')
            if not version:
                return None

        if version not in cls._models[model_id]:
            return None

        # Return a copy without the class reference
        model_info = dict(cls._models[model_id][version])
        del model_info['class']
        return model_info

    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered models.

        Returns:
            Dict with model IDs mapping to their version info
        """
        result = {}

        for model_id in cls._models:
            # Get available versions
            versions = list(cls._models[model_id].keys())

            # Get latest and stable versions
            latest = cls._versions[model_id].get('latest')
            stable = cls._versions[model_id].get('stable')

            # Get descriptions from latest version
            description = ""
            metadata = {}
            if latest:
                description = cls._models[model_id][latest].get('description', '')
                metadata = cls._models[model_id][latest].get('metadata', {})

            result[model_id] = {
                'versions': versions,
                'latest': latest,
                'stable': stable,
                'description': description,
                'metadata': metadata
            }

        return result

    @classmethod
    def get_model_versions(cls, model_id: str) -> List[str]:
        """
        Get all available versions of a model.

        Args:
            model_id: ID of the model type

        Returns:
            List of available versions (sorted by version number)
        """
        if model_id not in cls._models:
            return []

        versions = list(cls._models[model_id].keys())
        versions.sort(key=lambda v: cls._parse_version(v))
        return versions

    @classmethod
    def unregister_model(cls, model_id: str, version: str = None) -> bool:
        """
        Unregister a model from the registry.

        Args:
            model_id: ID of the model type
            version: Specific version to unregister (None for all versions)

        Returns:
            bool: True if successful, False otherwise
        """
        if model_id not in cls._models:
            return False

        if version:
            # Unregister specific version
            if version not in cls._models[model_id]:
                return False

            del cls._models[model_id][version]

            # Update latest and stable versions if needed
            if cls._versions[model_id].get('latest') == version:
                # Find new latest version
                versions = cls.get_model_versions(model_id)
                if versions:
                    cls._versions[model_id]['latest'] = versions[-1]
                else:
                    del cls._versions[model_id]['latest']

            if cls._versions[model_id].get('stable') == version:
                # Remove stable marker
                del cls._versions[model_id]['stable']

            logger.info(f"Unregistered version '{version}' of alpha model '{model_id}'")

            # Remove model entirely if no versions left
            if not cls._models[model_id]:
                del cls._models[model_id]
                del cls._versions[model_id]
        else:
            # Unregister all versions
            del cls._models[model_id]
            del cls._versions[model_id]
            logger.info(f"Unregistered all versions of alpha model '{model_id}'")

        return True

    @classmethod
    def register_evaluator(cls, name: str, evaluator: Callable) -> None:
        """
        Register an evaluation function for alpha models.

        Args:
            name: Name of the evaluator
            evaluator: Evaluation function
        """
        cls._evaluators[name] = evaluator
        logger.info(f"Registered alpha model evaluator '{name}'")

    @classmethod
    def evaluate_model(cls, model_id: str, evaluator_name: str,
                     data: Any, version: str = None) -> Optional[Dict[str, Any]]:
        """
        Evaluate an alpha model using a registered evaluator.

        Args:
            model_id: ID of the model type
            evaluator_name: Name of the evaluator to use
            data: Data to pass to the evaluator
            version: Specific version to evaluate (default: latest)

        Returns:
            Dict with evaluation results or None if evaluation failed
        """
        if evaluator_name not in cls._evaluators:
            logger.error(f"Evaluator '{evaluator_name}' not found")
            return None

        model_class = cls.get_model_class(model_id, version)
        if not model_class:
            return None

        try:
            # Create temporary instance for evaluation
            model = model_class(model_id=f"{model_id}-eval")

            # Run evaluation
            result = cls._evaluators[evaluator_name](model, data)

            # Add model info to result
            result['model_id'] = model_id
            result['version'] = version or cls._versions[model_id].get('latest', 'unknown')

            return result
        except Exception as e:
            logger.error(f"Error evaluating model '{model_id}': {str(e)}")
            return None

    @classmethod
    def find_models_for_instruments(cls, instruments: Set[str]) -> Dict[str, List[str]]:
        """
        Find models that can generate signals for the given instruments.

        Args:
            instruments: Set of instrument IDs

        Returns:
            Dict mapping instrument IDs to lists of suitable model IDs
        """
        result = {instr: [] for instr in instruments}

        for model_id, versions in cls._models.items():
            latest = cls._versions[model_id].get('latest')
            if not latest:
                continue

            model_info = versions[latest]
            metadata = model_info.get('metadata', {})

            # Check if model supports these instruments
            supported_instruments = metadata.get('supported_instruments', [])

            if not supported_instruments:  # Assume supports all if not specified
                for instr in instruments:
                    result[instr].append(model_id)
            else:
                for instr in instruments:
                    if instr in supported_instruments:
                        result[instr].append(model_id)

        return result

    @classmethod
    def _compare_versions(cls, version1: str, version2: str) -> int:
        """
        Compare two version strings.

        Args:
            version1: First version string
            version2: Second version string

        Returns:
            -1 if version1 < version2, 0 if equal, 1 if version1 > version2
        """
        v1 = cls._parse_version(version1)
        v2 = cls._parse_version(version2)

        if v1 < v2:
            return -1
        elif v1 > v2:
            return 1
        else:
            return 0

    @staticmethod
    def _parse_version(version: str) -> tuple:
        """
        Parse version string into a tuple for comparison.

        Args:
            version: Version string (e.g., '1.0.0')

        Returns:
            Tuple of version components
        """
        # Split by dots and convert to integers if possible
        parts = []
        for part in version.split('.'):
            try:
                parts.append(int(part))
            except ValueError:
                parts.append(part)

        return tuple(parts)