"""
serializers.py - Data serialization utilities

This module provides utilities for serializing and deserializing various data types
used throughout the trading system. It ensures consistent JSON serialization of
custom objects, enums, datetimes, and other complex types.
"""

import json
import uuid
import datetime
import enum
import dataclasses
from typing import Any, Dict, List, Optional, Type, Union, TypeVar, get_type_hints
import inspect
import logging
from decimal import Decimal

logger = logging.getLogger(__name__)

T = TypeVar('T')


class JSONEncoder(json.JSONEncoder):
    """
    Extended JSON encoder that handles special types:
    - Enum: converted to their string value
    - datetime: converted to ISO format string
    - UUID: converted to string
    - Dataclass: converted to dict
    - Decimal: converted to float
    """

    def default(self, obj: Any) -> Any:
        # Handle Enum
        if isinstance(obj, enum.Enum):
            return obj.value

        # Handle datetime
        if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
            return obj.isoformat()

        # Handle UUID
        if isinstance(obj, uuid.UUID):
            return str(obj)

        # Handle dataclass
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)

        # Handle Decimal
        if isinstance(obj, Decimal):
            return float(obj)

        # Let the parent class handle the rest
        return super().default(obj)


def to_json(obj: Any, pretty: bool = False) -> str:
    """
    Convert an object to a JSON string.

    Args:
        obj: The object to convert
        pretty: Whether to format the JSON with indentation

    Returns:
        JSON string representation of the object
    """
    indent = 2 if pretty else None
    return json.dumps(obj, cls=JSONEncoder, indent=indent)


def from_json(json_str: str) -> Any:
    """
    Convert a JSON string to a Python object.

    Args:
        json_str: JSON string to parse

    Returns:
        Parsed Python object
    """
    return json.loads(json_str)


def serialize(obj: Any) -> Dict[str, Any]:
    """
    Serialize an object to a dictionary suitable for JSON serialization.

    Args:
        obj: Object to serialize

    Returns:
        Dictionary representation of the object
    """
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)

    # Handle basic types
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Handle collections
    if isinstance(obj, list):
        return [serialize(item) for item in obj]

    if isinstance(obj, dict):
        return {str(key): serialize(value) for key, value in obj.items()}

    # Handle enums
    if isinstance(obj, enum.Enum):
        return obj.value

    # Handle datetime
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()

    # Handle UUID
    if isinstance(obj, uuid.UUID):
        return str(obj)

    # Handle Decimal
    if isinstance(obj, Decimal):
        return float(obj)

    # For other objects, try to convert to dictionary
    try:
        return {key: serialize(value) for key, value in obj.__dict__.items()
                if not key.startswith('_')}
    except AttributeError:
        logger.warning(f"Could not serialize object of type {type(obj)}")
        return str(obj)


def deserialize(data: Any, cls: Optional[Type[T]] = None) -> Union[T, Any]:
    """
    Deserialize data into an object of the specified class.

    Args:
        data: Data to deserialize
        cls: Optional class to deserialize into

    Returns:
        Deserialized object
    """
    if cls is None:
        return data

    # Handle None
    if data is None:
        return None

    # Handle basic types
    if cls in (str, int, float, bool) and isinstance(data, cls):
        return data

    # Handle enums
    if issubclass(cls, enum.Enum):
        return cls(data)

    # Handle datetime
    if cls is datetime.datetime and isinstance(data, str):
        return datetime.datetime.fromisoformat(data)

    if cls is datetime.date and isinstance(data, str):
        return datetime.date.fromisoformat(data)

    # Handle UUID
    if cls is uuid.UUID and isinstance(data, str):
        return uuid.UUID(data)

    # Handle Decimal
    if cls is Decimal and (isinstance(data, (int, float, str))):
        return Decimal(data)

    # Handle lists and dictionaries
    if cls is list and isinstance(data, list):
        return data

    if cls is dict and isinstance(data, dict):
        return data

    # Check for from_dict class method
    if hasattr(cls, "from_dict") and callable(cls.from_dict):
        return cls.from_dict(data)

    # Create instance for dataclasses
    if dataclasses.is_dataclass(cls):
        field_types = get_type_hints(cls)
        kwargs = {}

        for field in dataclasses.fields(cls):
            field_name = field.name
            if field_name in data:
                field_type = field_types.get(field_name, Any)
                kwargs[field_name] = deserialize(data[field_name], field_type)

        return cls(**kwargs)

    # For other classes, try to create an instance
    try:
        instance = cls()
        if isinstance(data, dict):
            for key, value in data.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
        return instance
    except Exception as e:
        logger.warning(f"Could not deserialize data to {cls.__name__}: {e}")
        return data


class SerializableMixin:
    """
    Mixin class providing serialization capabilities to any class.

    Any class that inherits from this mixin will gain:
    - to_dict() method to convert to dictionary
    - to_json() method to convert to JSON string
    - from_dict() class method to create instance from dictionary
    - from_json() class method to create instance from JSON string
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to dictionary"""
        return serialize(self)

    def to_json(self, pretty: bool = False) -> str:
        """Convert instance to JSON string"""
        return to_json(self.to_dict(), pretty)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create instance from dictionary"""
        return deserialize(data, cls)

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """Create instance from JSON string"""
        data = from_json(json_str)
        return cls.from_dict(data)


def load_json_file(file_path: str) -> Any:
    """
    Load and parse a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Parsed content of the JSON file
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        raise


def save_json_file(data: Any, file_path: str, pretty: bool = True) -> bool:
    """
    Save data to a JSON file.

    Args:
        data: Data to save
        file_path: Path to the JSON file
        pretty: Whether to format the JSON with indentation

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(file_path, 'w') as f:
            f.write(to_json(data, pretty))
        return True
    except Exception as e:
        logger.error(f"Error saving JSON file {file_path}: {e}")
        return False