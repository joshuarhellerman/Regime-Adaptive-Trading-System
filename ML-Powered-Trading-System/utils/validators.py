"""
validators.py - Data validation utilities

This module provides general-purpose validation functions for ensuring data 
integrity, type safety, and constraint satisfaction throughout the trading system.

The module implements common validation patterns used across the system, including:
- Type validation
- Range/boundary validation
- Schema validation
- Format validation
- Business rule validation

These utilities are used by other system components such as data_integrity.py,
model_validator.py, and pre_trade_validator.py.
"""

import logging
import json
import re
import inspect
import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Union, Optional, Set, Tuple, Callable, Type, TypeVar, Generic
from dataclasses import is_dataclass, asdict
from pathlib import Path

# Configure logger
logger = logging.getLogger(__name__)

# Type variable for generic validation
T = TypeVar('T')


class ValidationLevel(Enum):
    """Severity levels for validation errors."""
    INFO = "info"  # Informational validation message
    WARNING = "warning"  # Warning but allows operation to continue
    ERROR = "error"  # Validation error that should block operation


class ValidationError(Exception):
    """
    Exception raised for validation errors.

    Attributes:
        message: Error message
        field: Field that failed validation
        level: Severity level of the validation error
        details: Additional details about the error
    """

    def __init__(self,
                 message: str,
                 field: Optional[str] = None,
                 level: ValidationLevel = ValidationLevel.ERROR,
                 details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.field = field
        self.level = level
        self.details = details or {}
        super().__init__(self.message)


class ValidationResult:
    """
    Result of a validation operation.

    Attributes:
        valid: Whether validation passed
        errors: List of validation errors
        warnings: List of validation warnings
        info: List of validation info messages
    """

    def __init__(self):
        self.valid = True
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.info: List[Dict[str, Any]] = []

    def add_error(self, message: str, field: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> None:
        """Add an error to the validation result."""
        self.valid = False
        self.errors.append({
            "message": message,
            "field": field,
            "details": details or {}
        })

    def add_warning(self, message: str, field: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> None:
        """Add a warning to the validation result."""
        self.warnings.append({
            "message": message,
            "field": field,
            "details": details or {}
        })

    def add_info(self, message: str, field: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> None:
        """Add an info message to the validation result."""
        self.info.append({
            "message": message,
            "field": field,
            "details": details or {}
        })

    def add_result(self, other: 'ValidationResult') -> None:
        """Merge another validation result into this one."""
        if not other.valid:
            self.valid = False

        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.extend(other.info)

    def __bool__(self) -> bool:
        """Return True if validation passed, False otherwise."""
        return self.valid

    def to_dict(self) -> Dict[str, Any]:
        """Convert the validation result to a dictionary."""
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ValidationResult':
        """Create a validation result from a dictionary."""
        result = ValidationResult()
        result.valid = data.get("valid", True)
        result.errors = data.get("errors", [])
        result.warnings = data.get("warnings", [])
        result.info = data.get("info", [])
        return result


class Validator:
    """
    Base class for validators.

    This class provides common functionality for validators, including
    the ability to register validation rules and execute them.
    """

    def __init__(self):
        self._rules: List[Tuple[Callable, Dict[str, Any]]] = []

    def add_rule(self, rule: Callable, **kwargs) -> None:
        """
        Add a validation rule.

        Args:
            rule: Validation function that accepts a value and returns a ValidationResult
            **kwargs: Additional keyword arguments to pass to the rule function
        """
        self._rules.append((rule, kwargs))

    def validate(self, value: Any) -> ValidationResult:
        """
        Validate a value against all registered rules.

        Args:
            value: Value to validate

        Returns:
            ValidationResult with the outcome of all rule checks
        """
        result = ValidationResult()

        for rule, kwargs in self._rules:
            try:
                rule_result = rule(value, **kwargs)
                if isinstance(rule_result, ValidationResult):
                    result.add_result(rule_result)
                elif isinstance(rule_result, bool) and not rule_result:
                    # If the rule returns False, treat it as an error
                    rule_name = getattr(rule, "__name__", "unknown_rule")
                    result.add_error(f"Validation failed for rule: {rule_name}")
            except ValidationError as e:
                result.add_error(e.message, e.field, e.details)
            except Exception as e:
                logger.error(f"Error in validation rule {getattr(rule, '__name__', 'unknown')}: {str(e)}")
                result.add_error(f"Validation error: {str(e)}")

        return result


# Type checking validation functions

def validate_type(value: Any, expected_type: Type[T]) -> T:
    """
    Validate that a value is of the expected type.

    Args:
        value: Value to validate
        expected_type: Expected type of the value

    Returns:
        The original value if it's of the correct type

    Raises:
        ValidationError: If the value is not of the expected type
    """
    if not isinstance(value, expected_type):
        raise ValidationError(
            message=f"Expected type {expected_type.__name__}, got {type(value).__name__}",
            details={"expected_type": expected_type.__name__, "actual_type": type(value).__name__}
        )
    return value


def validate_instance(value: Any, cls: Type[T]) -> T:
    """
    Validate that a value is an instance of the specified class.

    Args:
        value: Value to validate
        cls: Class that the value should be an instance of

    Returns:
        The original value if it's an instance of the class

    Raises:
        ValidationError: If the value is not an instance of the class
    """
    if not isinstance(value, cls):
        raise ValidationError(
            message=f"Expected instance of {cls.__name__}, got {type(value).__name__}",
            details={"expected_class": cls.__name__, "actual_type": type(value).__name__}
        )
    return value


def validate_subclass(cls: Type, parent_cls: Type) -> Type:
    """
    Validate that a class is a subclass of the specified parent class.

    Args:
        cls: Class to validate
        parent_cls: Parent class that cls should inherit from

    Returns:
        The original class if it's a subclass of the parent class

    Raises:
        ValidationError: If the class is not a subclass of the parent class
    """
    if not issubclass(cls, parent_cls):
        raise ValidationError(
            message=f"Expected subclass of {parent_cls.__name__}, got {cls.__name__}",
            details={"expected_parent": parent_cls.__name__, "actual_class": cls.__name__}
        )
    return cls


# Numeric validation functions

def validate_range(value: Union[int, float, Decimal],
                   min_value: Optional[Union[int, float, Decimal]] = None,
                   max_value: Optional[Union[int, float, Decimal]] = None,
                   inclusive_min: bool = True,
                   inclusive_max: bool = True) -> Union[int, float, Decimal]:
    """
    Validate that a numeric value is within the specified range.

    Args:
        value: Numeric value to validate
        min_value: Minimum allowed value (if None, no minimum)
        max_value: Maximum allowed value (if None, no maximum)
        inclusive_min: Whether the minimum value is inclusive
        inclusive_max: Whether the maximum value is inclusive

    Returns:
        The original value if it's within the range

    Raises:
        ValidationError: If the value is outside the range
    """
    # Check minimum value
    if min_value is not None:
        if inclusive_min and value < min_value:
            raise ValidationError(
                message=f"Value {value} is less than minimum {min_value}",
                details={"value": value, "min_value": min_value, "inclusive": inclusive_min}
            )
        elif not inclusive_min and value <= min_value:
            raise ValidationError(
                message=f"Value {value} is less than or equal to minimum {min_value}",
                details={"value": value, "min_value": min_value, "inclusive": inclusive_min}
            )

    # Check maximum value
    if max_value is not None:
        if inclusive_max and value > max_value:
            raise ValidationError(
                message=f"Value {value} is greater than maximum {max_value}",
                details={"value": value, "max_value": max_value, "inclusive": inclusive_max}
            )
        elif not inclusive_max and value >= max_value:
            raise ValidationError(
                message=f"Value {value} is greater than or equal to maximum {max_value}",
                details={"value": value, "max_value": max_value, "inclusive": inclusive_max}
            )

    return value


def validate_positive(value: Union[int, float, Decimal]) -> Union[int, float, Decimal]:
    """
    Validate that a numeric value is positive (greater than zero).

    Args:
        value: Numeric value to validate

    Returns:
        The original value if it's positive

    Raises:
        ValidationError: If the value is not positive
    """
    if value <= 0:
        raise ValidationError(
            message=f"Value {value} is not positive",
            details={"value": value}
        )
    return value


def validate_non_negative(value: Union[int, float, Decimal]) -> Union[int, float, Decimal]:
    """
    Validate that a numeric value is non-negative (greater than or equal to zero).

    Args:
        value: Numeric value to validate

    Returns:
        The original value if it's non-negative

    Raises:
        ValidationError: If the value is negative
    """
    if value < 0:
        raise ValidationError(
            message=f"Value {value} is negative",
            details={"value": value}
        )
    return value


def validate_percentage(value: Union[float, Decimal], allow_zero: bool = True) -> Union[float, Decimal]:
    """
    Validate that a value is a valid percentage (0-1 range).

    Args:
        value: Value to validate
        allow_zero: Whether zero is allowed

    Returns:
        The original value if it's a valid percentage

    Raises:
        ValidationError: If the value is not a valid percentage
    """
    if not allow_zero and value == 0:
        raise ValidationError(
            message="Zero percentage is not allowed",
            details={"value": value, "allow_zero": allow_zero}
        )

    if value < 0 or value > 1:
        raise ValidationError(
            message=f"Value {value} is not a valid percentage (must be between 0 and 1)",
            details={"value": value}
        )

    return value


# String validation functions

def validate_non_empty_string(value: str) -> str:
    """
    Validate that a string is not empty.

    Args:
        value: String to validate

    Returns:
        The original string if it's not empty

    Raises:
        ValidationError: If the string is empty
    """
    if not value:
        raise ValidationError(
            message="String is empty",
            details={"value": value}
        )
    return value


def validate_string_length(value: str, min_length: Optional[int] = None, max_length: Optional[int] = None) -> str:
    """
    Validate that a string's length is within the specified range.

    Args:
        value: String to validate
        min_length: Minimum allowed length (if None, no minimum)
        max_length: Maximum allowed length (if None, no maximum)

    Returns:
        The original string if its length is within the range

    Raises:
        ValidationError: If the string's length is outside the range
    """
    length = len(value)

    if min_length is not None and length < min_length:
        raise ValidationError(
            message=f"String length {length} is less than minimum {min_length}",
            details={"value": value, "length": length, "min_length": min_length}
        )

    if max_length is not None and length > max_length:
        raise ValidationError(
            message=f"String length {length} is greater than maximum {max_length}",
            details={"value": value, "length": length, "max_length": max_length}
        )

    return value


def validate_regex(value: str, pattern: str, description: Optional[str] = None) -> str:
    """
    Validate that a string matches a regular expression pattern.

    Args:
        value: String to validate
        pattern: Regular expression pattern to match
        description: Description of the pattern (for error messages)

    Returns:
        The original string if it matches the pattern

    Raises:
        ValidationError: If the string doesn't match the pattern
    """
    if not re.match(pattern, value):
        pattern_desc = description or f"pattern '{pattern}'"
        raise ValidationError(
            message=f"String '{value}' does not match {pattern_desc}",
            details={"value": value, "pattern": pattern, "description": description}
        )
    return value


def validate_one_of(value: Any, valid_values: List[Any]) -> Any:
    """
    Validate that a value is one of a set of valid values.

    Args:
        value: Value to validate
        valid_values: List of valid values

    Returns:
        The original value if it's in the valid values

    Raises:
        ValidationError: If the value is not in the valid values
    """
    if value not in valid_values:
        raise ValidationError(
            message=f"Value '{value}' is not one of {valid_values}",
            details={"value": value, "valid_values": valid_values}
        )
    return value


def validate_enum(value: Any, enum_class: Type[Enum]) -> Any:
    """
    Validate that a value is a valid enum member.

    Args:
        value: Value to validate
        enum_class: Enum class to check against

    Returns:
        The original value if it's a valid enum member

    Raises:
        ValidationError: If the value is not a valid enum member
    """
    if isinstance(value, enum_class):
        return value

    try:
        # Try to convert string to enum
        if isinstance(value, str):
            return enum_class(value)

        # Try to get enum by value
        return enum_class(value)
    except (ValueError, KeyError):
        valid_values = [e.value for e in enum_class]
        raise ValidationError(
            message=f"Value '{value}' is not a valid {enum_class.__name__} enum value. Valid values: {valid_values}",
            details={"value": value, "enum_class": enum_class.__name__, "valid_values": valid_values}
        )


# Collection validation functions

def validate_length(value: Union[List, Dict, Set, str],
                    min_length: Optional[int] = None,
                    max_length: Optional[int] = None) -> Union[List, Dict, Set, str]:
    """
    Validate that a collection's length is within the specified range.

    Args:
        value: Collection to validate
        min_length: Minimum allowed length (if None, no minimum)
        max_length: Maximum allowed length (if None, no maximum)

    Returns:
        The original collection if its length is within the range

    Raises:
        ValidationError: If the collection's length is outside the range
    """
    length = len(value)

    if min_length is not None and length < min_length:
        raise ValidationError(
            message=f"Collection length {length} is less than minimum {min_length}",
            details={"length": length, "min_length": min_length}
        )

    if max_length is not None and length > max_length:
        raise ValidationError(
            message=f"Collection length {length} is greater than maximum {max_length}",
            details={"length": length, "max_length": max_length}
        )

    return value


def validate_non_empty(value: Union[List, Dict, Set]) -> Union[List, Dict, Set]:
    """
    Validate that a collection is not empty.

    Args:
        value: Collection to validate

    Returns:
        The original collection if it's not empty

    Raises:
        ValidationError: If the collection is empty
    """
    if not value:
        raise ValidationError(
            message="Collection is empty",
            details={"type": type(value).__name__}
        )
    return value


def validate_unique(values: List[Any]) -> List[Any]:
    """
    Validate that a list contains unique values.

    Args:
        values: List to validate

    Returns:
        The original list if all values are unique

    Raises:
        ValidationError: If there are duplicate values
    """
    seen = set()
    duplicates = set()

    for value in values:
        if value in seen:
            duplicates.add(value)
        else:
            seen.add(value)

    if duplicates:
        raise ValidationError(
            message=f"List contains duplicate values: {duplicates}",
            details={"duplicates": list(duplicates)}
        )

    return values


def validate_all_of_type(values: List[Any], expected_type: Type) -> List[Any]:
    """
    Validate that all items in a list are of the expected type.

    Args:
        values: List to validate
        expected_type: Expected type of all items

    Returns:
        The original list if all items are of the correct type

    Raises:
        ValidationError: If any item is not of the expected type
    """
    invalid_items = [(i, type(item).__name__) for i, item in enumerate(values) if not isinstance(item, expected_type)]

    if invalid_items:
        raise ValidationError(
            message=f"List contains items that are not of type {expected_type.__name__}",
            details={"expected_type": expected_type.__name__, "invalid_items": invalid_items}
        )

    return values


# Date and time validation functions

def validate_datetime(value: datetime.datetime) -> datetime.datetime:
    """
    Validate that a value is a valid datetime.

    Args:
        value: Value to validate

    Returns:
        The original value if it's a valid datetime

    Raises:
        ValidationError: If the value is not a valid datetime
    """
    if not isinstance(value, datetime.datetime):
        raise ValidationError(
            message=f"Expected datetime, got {type(value).__name__}",
            details={"type": type(value).__name__}
        )
    return value


def validate_datetime_range(value: datetime.datetime,
                            min_datetime: Optional[datetime.datetime] = None,
                            max_datetime: Optional[datetime.datetime] = None) -> datetime.datetime:
    """
    Validate that a datetime is within the specified range.

    Args:
        value: Datetime to validate
        min_datetime: Minimum allowed datetime (if None, no minimum)
        max_datetime: Maximum allowed datetime (if None, no maximum)

    Returns:
        The original datetime if it's within the range

    Raises:
        ValidationError: If the datetime is outside the range
    """
    validate_datetime(value)

    if min_datetime is not None and value < min_datetime:
        raise ValidationError(
            message=f"Datetime {value} is before minimum {min_datetime}",
            details={"value": value.isoformat(), "min_datetime": min_datetime.isoformat()}
        )

    if max_datetime is not None and value > max_datetime:
        raise ValidationError(
            message=f"Datetime {value} is after maximum {max_datetime}",
            details={"value": value.isoformat(), "max_datetime": max_datetime.isoformat()}
        )

    return value


def validate_future_datetime(value: datetime.datetime) -> datetime.datetime:
    """
    Validate that a datetime is in the future.

    Args:
        value: Datetime to validate

    Returns:
        The original datetime if it's in the future

    Raises:
        ValidationError: If the datetime is not in the future
    """
    validate_datetime(value)

    now = datetime.datetime.now(value.tzinfo)
    if value <= now:
        raise ValidationError(
            message=f"Datetime {value} is not in the future",
            details={"value": value.isoformat(), "now": now.isoformat()}
        )

    return value


def validate_past_datetime(value: datetime.datetime) -> datetime.datetime:
    """
    Validate that a datetime is in the past.

    Args:
        value: Datetime to validate

    Returns:
        The original datetime if it's in the past

    Raises:
        ValidationError: If the datetime is not in the past
    """
    validate_datetime(value)

    now = datetime.datetime.now(value.tzinfo)
    if value >= now:
        raise ValidationError(
            message=f"Datetime {value} is not in the past",
            details={"value": value.isoformat(), "now": now.isoformat()}
        )

    return value


# File validation functions

def validate_file_exists(file_path: Union[str, Path]) -> Path:
    """
    Validate that a file exists.

    Args:
        file_path: Path to the file

    Returns:
        The file path as a Path object if the file exists

    Raises:
        ValidationError: If the file does not exist
    """
    path = Path(file_path)
    if not path.exists():
        raise ValidationError(
            message=f"File {path} does not exist",
            details={"file_path": str(path)}
        )

    if not path.is_file():
        raise ValidationError(
            message=f"{path} is not a file",
            details={"file_path": str(path)}
        )

    return path


def validate_directory_exists(dir_path: Union[str, Path]) -> Path:
    """
    Validate that a directory exists.

    Args:
        dir_path: Path to the directory

    Returns:
        The directory path as a Path object if the directory exists

    Raises:
        ValidationError: If the directory does not exist
    """
    path = Path(dir_path)
    if not path.exists():
        raise ValidationError(
            message=f"Directory {path} does not exist",
            details={"dir_path": str(path)}
        )

    if not path.is_dir():
        raise ValidationError(
            message=f"{path} is not a directory",
            details={"dir_path": str(path)}
        )

    return path


def validate_file_extension(file_path: Union[str, Path], valid_extensions: List[str]) -> Path:
    """
    Validate that a file has one of the valid extensions.

    Args:
        file_path: Path to the file
        valid_extensions: List of valid file extensions (e.g., ['.txt', '.csv'])

    Returns:
        The file path as a Path object if the file has a valid extension

    Raises:
        ValidationError: If the file does not have a valid extension
    """
    path = Path(file_path)

    # Ensure extensions start with a dot
    valid_exts = [ext if ext.startswith('.') else f'.{ext}' for ext in valid_extensions]

    if path.suffix.lower() not in [ext.lower() for ext in valid_exts]:
        raise ValidationError(
            message=f"File {path} does not have a valid extension. Expected: {valid_extensions}",
            details={"file_path": str(path), "extension": path.suffix, "valid_extensions": valid_extensions}
        )

    return path


# Schema validation functions

def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a dictionary against a schema.

    The schema is a dictionary where keys are field names and values are
    dictionaries defining the field's type, requirements, and constraints.

    Schema format:
    {
        "field_name": {
            "type": type or list of types,
            "required": bool,
            "default": default value if field is missing,
            "validator": function to validate the field value,
            "schema": nested schema for dict values
        }
    }

    Args:
        data: Dictionary to validate
        schema: Schema definition

    Returns:
        The original data if it conforms to the schema

    Raises:
        ValidationError: If the data does not conform to the schema
    """
    result = ValidationResult()
    validated_data = {}

    # Check for unknown fields
    unknown_fields = set(data.keys()) - set(schema.keys())
    if unknown_fields:
        for field in unknown_fields:
            result.add_warning(
                message=f"Unknown field: {field}",
                field=field
            )

    # Validate each field in the schema
    for field_name, field_schema in schema.items():
        field_type = field_schema.get("type")
        required = field_schema.get("required", False)
        default = field_schema.get("default")
        validator = field_schema.get("validator")
        nested_schema = field_schema.get("schema")

        # Check if field exists
        if field_name not in data:
            if required:
                result.add_error(
                    message=f"Required field '{field_name}' is missing",
                    field=field_name
                )
                continue
            elif default is not None:
                validated_data[field_name] = default
                continue
            else:
                # Field is not required and has no default, skip validation
                continue

        # Get field value
        value = data[field_name]

        # Validate type
        if field_type is not None:
            if isinstance(field_type, (list, tuple)):
                # Multiple allowed types
                if not any(isinstance(value, t) for t in field_type):
                    type_names = [t.__name__ for t in field_type]
                    result.add_error(
                        message=f"Field '{field_name}' must be one of these types: {type_names}",
                        field=field_name,
                        details={"value": value, "expected_types": type_names, "actual_type": type(value).__name__}
                    )
                    continue
            elif not isinstance(value, field_type):
                result.add_error(
                    message=f"Field '{field_name}' must be of type {field_type.__name__}",
                    field=field_name,
                    details={"value": value, "expected_type": field_type.__name__, "actual_type": type(value).__name__}
                )
                continue

        # Apply custom validator
        if validator is not None:
            try:
                value = validator(value)
            except ValidationError as e:
                result.add_error(
                    message=f"Validation error for field '{field_name}': {e.message}",
                    field=field_name,
                    details=e.details
                )
                continue
            except Exception as e:
                result.add_error(
                    message=f"Validation error for field '{field_name}': {str(e)}",
                    field=field_name
                )
                continue

        # Validate nested schema for dict values
        if nested_schema is not None and isinstance(value, dict):
            nested_result = validate_schema(value, nested_schema)
            if not nested_result.valid:
                # Prefix nested field names with the current field name
                for error in nested_result.errors:
                    nested_field = error["field"]
                    error["field"] = f"{field_name}.{nested_field}" if nested_field else field_name
                    result.add_error(
                        message=error["message"],
                        field=error["field"],
                        details=error["details"]
                    )
                continue

        # Field is valid
        validated_data[field_name] = value

    # Check overall validity
    if not result.valid:
        field_errors = ", ".join(f"'{error['field']}'" for error in result.errors if error['field'])
        raise ValidationError(
            message=f"Schema validation failed for fields: {field_errors}",
            details={"errors": result.errors}
        )

    return validated_data


def validate_json(json_str: str) -> Dict[str, Any]:
    """
    Validate that a string is valid JSON and parse it.

    Args:
        json_str: JSON string to validate

    Returns:
        Parsed JSON data

    Raises:
        ValidationError: If the string is not valid JSON
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValidationError(
            message=f"Invalid JSON: {str(e)}",
            details={"error": str(e), "position": e.pos, "line": e.lineno, "column": e.colno}
        )


def validate_dataclass(value: Any, dataclass_type: Type[T]) -> T:
    """
    Validate that a value is an instance of the specified dataclass.

    Args:
        value: Value to validate
        dataclass_type: Expected dataclass type

    Returns:
        The original value if it's an instance of the dataclass

    Raises:
        ValidationError: If the value is not an instance of the dataclass
    """
    if not is_dataclass(dataclass_type):
        raise ValidationError(
            message=f"{dataclass_type.__name__} is not a dataclass",
            details={"type": dataclass_type.__name__}
        )

    if isinstance(value, dict):
        # Try to convert dict to dataclass
        try:
            # Get dataclass fields
            fields = [field.name for field in dataclass_type.__dataclass_fields__.values()]

            # Filter out unknown fields
            known_data = {k: v for k, v in value.items() if k in fields}

            # Create dataclass instance
            return dataclass_type(**known_data)
        except Exception as e:
            raise ValidationError(
                message=f"Could not convert dict to dataclass {dataclass_type.__name__}: {str(e)}",
                details={"error": str(e), "dataclass_type": dataclass_type.__name__}
            )
    elif not isinstance(value, dataclass_type):
        raise ValidationError(
            message=f"Expected dataclass {dataclass_type.__name__}, got {type(value).__name__}",
            details={"expected_type": dataclass_type.__name__, "actual_type": type(value).__name__}
        )

    return value


def validate_dataclass_fields(value: Any) -> Any:
    """
    Validate a dataclass instance by calling validate() method on fields
    that have such a method.

    Args:
        value: Dataclass instance to validate

    Returns:
        The original value if validation passes

    Raises:
        ValidationError: If validation fails for any field
    """
    if not is_dataclass(value):
        raise ValidationError(
            message=f"Value is not a dataclass instance: {type(value).__name__}",
            details={"type": type(value).__name__}
        )

    result = ValidationResult()

    for field_name, field_value in asdict(value).items():
        # Check if field has a validate method
        if hasattr(field_value, "validate") and callable(getattr(field_value, "validate")):
            try:
                getattr(field_value, "validate")()
            except ValidationError as e:
                result.add_error(
                    message=f"Validation error for field '{field_name}': {e.message}",
                    field=field_name,
                    details=e.details
                )
            except Exception as e:
                result.add_error(
                    message=f"Validation error for field '{field_name}': {str(e)}",
                    field=field_name
                )

    if not result.valid:
        field_errors = ", ".join(f"'{error['field']}'" for error in result.errors if error['field'])
        raise ValidationError(
            message=f"Dataclass validation failed for fields: {field_errors}",
            details={"errors": result.errors}
        )

    return value


# Trading system specific validation functions

def validate_symbol(symbol: str) -> str:
    """
    Validate that a string is a valid trading symbol.

    Args:
        symbol: Trading symbol to validate

    Returns:
        The original symbol if it's valid

    Raises:
        ValidationError: If the symbol is not valid
    """
    validate_non_empty_string(symbol)

    # Basic symbol validation - should be customized based on specific exchange requirements
    pattern = r'^[A-Za-z0-9._-]+
    validate_regex(symbol, pattern, "valid trading symbol format")

    return symbol


def validate_price(price: float, allow_zero: bool = False) -> float:
    """
    Validate that a value is a valid price.

    Args:
        price: Price to validate
        allow_zero: Whether zero is allowed

    Returns:
        The original price if it's valid

    Raises:
        ValidationError: If the price is not valid
    """
    if allow_zero:
        validate_non_negative(price)
    else:
        validate_positive(price)

    return price


def validate_quantity(quantity: float, allow_zero: bool = False) -> float:
    """
    Validate that a value is a valid quantity.

    Args:
        quantity: Quantity to validate
        allow_zero: Whether zero is allowed

    Returns:
        The original quantity if it's valid

    Raises:
        ValidationError: If the quantity is not valid
    """
    if allow_zero:
        validate_non_negative(quantity)
    else:
        validate_positive(quantity)

    return quantity


def validate_timeframe(timeframe: str) -> str:
    """
    Validate that a string is a valid trading timeframe.

    Args:
        timeframe: Timeframe to validate

    Returns:
        The original timeframe if it's valid

    Raises:
        ValidationError: If the timeframe is not valid
    """
    validate_non_empty_string(timeframe)

    # Basic timeframe validation - formats like "1m", "15m", "1h", "4h", "1d"
    pattern = r'^(\d+)([mhdwM])
    match = re.match(pattern, timeframe)

    if not match:
        raise ValidationError(
            message=f"Invalid timeframe format: {timeframe}",
            details={"timeframe": timeframe, "expected_format": "e.g., '1m', '15m', '1h', '4h', '1d'"}
        )

    value, unit = match.groups()
    value = int(value)

    # Validate common timeframe values based on unit
    valid_values = {
        'm': [1, 3, 5, 15, 30],  # minutes
        'h': [1, 2, 4, 6, 8, 12],  # hours
        'd': [1, 3, 7],  # days
        'w': [1, 2, 4],  # weeks
        'M': [1, 3, 6]  # months
    }

    if unit in valid_values and value not in valid_values[unit]:
        raise ValidationError(
            message=f"Uncommon timeframe value: {timeframe}",
            details={"timeframe": timeframe, "common_values": [f"{v}{unit}" for v in valid_values[unit]]},
            level=ValidationLevel.WARNING
        )

    return timeframe


def validate_feature_name(name: str) -> str:
    """
    Validate that a string is a valid feature name.

    Args:
        name: Feature name to validate

    Returns:
        The original name if it's valid

    Raises:
        ValidationError: If the name is not valid
    """
    validate_non_empty_string(name)

    # Feature names should be snake_case
    pattern = r'^[a-z][a-z0-9_]*
    validate_regex(name, pattern, "snake_case format (lowercase with underscores)")

    return name


def validate_model_name(name: str) -> str:
    """
    Validate that a string is a valid model name.

    Args:
        name: Model name to validate

    Returns:
        The original name if it's valid

    Raises:
        ValidationError: If the name is not valid
    """
    validate_non_empty_string(name)

    # Model names should have a specific format
    pattern = r'^[a-zA-Z][a-zA-Z0-9_-]*
    validate_regex(name, pattern, "valid model name format (alphanumeric with - and _)")

    return name


def validate_strategy_name(name: str) -> str:
    """
    Validate that a string is a valid strategy name.

    Args:
        name: Strategy name to validate

    Returns:
        The original name if it's valid

    Raises:
        ValidationError: If the name is not valid
    """
    validate_non_empty_string(name)

    # Strategy names should have a specific format
    pattern = r'^[a-zA-Z][a-zA-Z0-9_]*
    validate_regex(name, pattern, "valid strategy name format (alphanumeric with _)")

    return name


# Utility functions

def create_validator_from_schema(schema: Dict[str, Any]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Create a validator function from a schema definition.

    Args:
        schema: Schema definition

    Returns:
        Validator function that takes a dictionary and returns the validated dictionary
    """

    def validator(data: Dict[str, Any]) -> Dict[str, Any]:
        return validate_schema(data, schema)

    return validator


def validate_with_defaults(data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a dictionary against a schema and fill in default values for missing fields.

    Args:
        data: Dictionary to validate
        schema: Schema definition

    Returns:
        Validated dictionary with default values filled in
    """
    result = {}

    # Add default values for missing fields
    for field_name, field_schema in schema.items():
        if field_name in data:
            result[field_name] = data[field_name]
        elif "default" in field_schema:
            result[field_name] = field_schema["default"]

    # Validate the resulting dictionary
    return validate_schema(result, schema)


def chain_validators(*validators: Callable[[T], T]) -> Callable[[T], T]:
    """
    Chain multiple validators together.

    Args:
        *validators: Validator functions to chain

    Returns:
        Chained validator function
    """

    def chained_validator(value: T) -> T:
        result = value
        for validator in validators:
            result = validator(result)
        return result

    return chained_validator


def validate_config(config_obj: Any) -> ValidationResult:
    """
    Validate a configuration object by calling its validate method.

    Args:
        config_obj: Configuration object to validate

    Returns:
        ValidationResult with the outcome of validation
    """
    result = ValidationResult()

    if not hasattr(config_obj, "validate") or not callable(getattr(config_obj, "validate")):
        result.add_error(
            message="Config object does not have a validate method",
            details={"type": type(config_obj).__name__}
        )
        return result

    try:
        errors = config_obj.validate()

        if errors:
            for error in errors:
                result.add_error(message=error)

        return result
    except Exception as e:
        result.add_error(
            message=f"Validation error: {str(e)}",
            details={"error": str(e)}
        )
        return result


def validate_callable_signature(func: Callable, expected_params: List[str],
                                required_params: Optional[List[str]] = None) -> bool:
    """
    Validate that a callable has the expected parameters.

    Args:
        func: Callable to validate
        expected_params: Expected parameter names
        required_params: Required parameter names (subset of expected_params)

    Returns:
        True if the callable has the expected signature, False otherwise
    """
    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())

        # Check if all expected parameters are present
        for param in expected_params:
            if param not in params:
                return False

        # Check if all required parameters are present
        if required_params:
            for param in required_params:
                if param not in params:
                    return False

                # Check that required params don't have default values
                if sig.parameters[param].default != inspect.Parameter.empty:
                    return False

        return True
    except (TypeError, ValueError):
        return False


# Financial data validation functions

def validate_ohlcv_data(data: Dict[str, Union[List, np.ndarray]]) -> Dict[str, Union[List, np.ndarray]]:
    """
    Validate OHLCV (Open, High, Low, Close, Volume) data.

    Args:
        data: Dictionary containing OHLCV data

    Returns:
        Original data if valid

    Raises:
        ValidationError: If the data is not valid OHLCV data
    """
    # Check required columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    optional_columns = ['timestamp', 'date', 'time']

    for column in required_columns:
        if column not in data:
            raise ValidationError(
                message=f"Missing required OHLCV column: {column}",
                details={"missing_column": column, "available_columns": list(data.keys())}
            )

    # Check lengths match
    lengths = [len(data[column]) for column in data]
    if len(set(lengths)) > 1:
        raise ValidationError(
            message="OHLCV columns have inconsistent lengths",
            details={"column_lengths": {column: len(data[column]) for column in data}}
        )

    # Check high >= low
    try:
        highs = data['high']
        lows = data['low']

        invalid_indices = [i for i, (h, l) in enumerate(zip(highs, lows)) if h < l]
        if invalid_indices:
            raise ValidationError(
                message="Found high values less than low values",
                details={"invalid_indices": invalid_indices}
            )

        # Check open is within high-low range
        opens = data['open']
        invalid_indices = [i for i, (o, h, l) in enumerate(zip(opens, highs, lows)) if o > h or o < l]
        if invalid_indices:
            raise ValidationError(
                message="Found open values outside high-low range",
                details={"invalid_indices": invalid_indices}
            )

        # Check close is within high-low range
        closes = data['close']
        invalid_indices = [i for i, (c, h, l) in enumerate(zip(closes, highs, lows)) if c > h or c < l]
        if invalid_indices:
            raise ValidationError(
                message="Found close values outside high-low range",
                details={"invalid_indices": invalid_indices}
            )
    except Exception as e:
        raise ValidationError(
            message=f"Error validating OHLCV data: {str(e)}",
            details={"error": str(e)}
        )

    return data


def validate_market_data_schema(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate market data schema.

    Args:
        data: Market data dictionary

    Returns:
        Original data if valid

    Raises:
        ValidationError: If the data does not conform to the market data schema
    """
    schema = {
        "symbol": {"type": str, "required": True, "validator": validate_symbol},
        "timeframe": {"type": str, "required": True, "validator": validate_timeframe},
        "timestamp": {"type": (int, float), "required": True},
        "data": {
            "type": dict,
            "required": True,
            "validator": validate_ohlcv_data
        },
        "metadata": {"type": dict, "required": False}
    }

    return validate_schema(data, schema)


# Function to validate trade data

def validate_trade_data(trade: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate trade data.

    Args:
        trade: Trade dictionary

    Returns:
        Original trade if valid

    Raises:
        ValidationError: If the trade is not valid
    """
    schema = {
        "trade_id": {"type": str, "required": True},
        "symbol": {"type": str, "required": True, "validator": validate_symbol},
        "side": {"type": str, "required": True, "validator": lambda x: validate_one_of(x, ["buy", "sell"])},
        "quantity": {"type": (int, float), "required": True, "validator": validate_positive},
        "price": {"type": (int, float), "required": True, "validator": validate_positive},
        "timestamp": {"type": (int, float), "required": True},
        "exchange": {"type": str, "required": True},
        "order_id": {"type": str, "required": False},
        "commission": {"type": (int, float), "required": False, "validator": validate_non_negative},
        "commission_asset": {"type": str, "required": False},
        "is_maker": {"type": bool, "required": False},
        "metadata": {"type": dict, "required": False}
    }

    return validate_schema(trade, schema)


# Module initialization function

def validate_system_components(components: Dict[str, Any]) -> ValidationResult:
    """
    Validate the core system components against their required interfaces.

    Args:
        components: Dictionary of system components by name

    Returns:
        ValidationResult with validation outcome
    """
    result = ValidationResult()

    # Define required methods for each component type
    required_interfaces = {
        "event_bus": [
            "subscribe",
            "publish",
            "unsubscribe"
        ],
        "scheduler": [
            "schedule",
            "cancel",
            "get_scheduled_tasks"
        ],
        "state_manager": [
            "get_state",
            "set_state",
            "update_state",
            "delete_state"
        ],
        "risk_manager": [
            "check_risk",
            "get_risk_limits",
            "update_risk_limits"
        ],
        "data_service": [
            "get_data",
            "store_data",
            "get_latest",
            "get_history"
        ],
        "model_service": [
            "get_model",
            "predict",
            "train",
            "validate"
        ],
        "execution_service": [
            "submit_order",
            "cancel_order",
            "get_order_status",
            "get_orders"
        ]
    }

    # Validate each component
    for component_name, required_methods in required_interfaces.items():
        if component_name not in components:
            result.add_error(
                message=f"Missing required component: {component_name}",
                details={"component": component_name}
            )
            continue

        component = components[component_name]

        for method_name in required_methods:
            if not hasattr(component, method_name) or not callable(getattr(component, method_name)):
                result.add_error(
                    message=f"Component {component_name} is missing required method: {method_name}",
                    details={"component": component_name, "method": method_name}
                )

    return result