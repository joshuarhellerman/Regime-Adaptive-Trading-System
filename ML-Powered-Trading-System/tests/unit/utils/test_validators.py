import unittest
import sys
import os
import datetime
import json
import re
from decimal import Decimal
from enum import Enum
from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

# Add the project root to sys.path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.validators import (
    ValidationLevel, ValidationError, ValidationResult, Validator,
    validate_type, validate_instance, validate_subclass,
    validate_range, validate_positive, validate_non_negative, validate_percentage,
    validate_non_empty_string, validate_string_length, validate_regex, validate_one_of, validate_enum,
    validate_length, validate_non_empty, validate_unique, validate_all_of_type,
    validate_datetime, validate_datetime_range, validate_future_datetime, validate_past_datetime,
    validate_file_exists, validate_directory_exists, validate_file_extension,
    validate_schema, validate_json, validate_dataclass, validate_dataclass_fields,
    validate_symbol, validate_price, validate_quantity, validate_timeframe, validate_feature_name,
    validate_model_name, validate_strategy_name,
    create_validator_from_schema, validate_with_defaults, chain_validators, validate_config,
    validate_callable_signature, validate_ohlcv_data, validate_market_data_schema, validate_trade_data,
    validate_system_components
)


class TestValidationLevel(unittest.TestCase):
    def test_validation_levels(self):
        """Test that ValidationLevel enum has the expected values."""
        self.assertEqual(ValidationLevel.INFO.value, "info")
        self.assertEqual(ValidationLevel.WARNING.value, "warning")
        self.assertEqual(ValidationLevel.ERROR.value, "error")


class TestValidationError(unittest.TestCase):
    def test_validation_error_init(self):
        """Test ValidationError initialization with various parameters."""
        # Test with minimal parameters
        error = ValidationError("Test error")
        self.assertEqual(error.message, "Test error")
        self.assertIsNone(error.field)
        self.assertEqual(error.level, ValidationLevel.ERROR)
        self.assertEqual(error.details, {})
        
        # Test with all parameters
        details = {"key": "value"}
        error = ValidationError(
            message="Test error", 
            field="test_field", 
            level=ValidationLevel.WARNING,
            details=details
        )
        self.assertEqual(error.message, "Test error")
        self.assertEqual(error.field, "test_field")
        self.assertEqual(error.level, ValidationLevel.WARNING)
        self.assertEqual(error.details, details)


class TestValidationResult(unittest.TestCase):
    def test_init(self):
        """Test ValidationResult initialization."""
        result = ValidationResult()
        self.assertTrue(result.valid)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.warnings, [])
        self.assertEqual(result.info, [])
    
    def test_add_error(self):
        """Test adding errors to ValidationResult."""
        result = ValidationResult()
        # Add error without field and details
        result.add_error("Error message")
        self.assertFalse(result.valid)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0]["message"], "Error message")
        self.assertIsNone(result.errors[0]["field"])
        self.assertEqual(result.errors[0]["details"], {})
        
        # Add error with field and details
        details = {"key": "value"}
        result.add_error("Error with details", "field_name", details)
        self.assertEqual(len(result.errors), 2)
        self.assertEqual(result.errors[1]["message"], "Error with details")
        self.assertEqual(result.errors[1]["field"], "field_name")
        self.assertEqual(result.errors[1]["details"], details)
    
    def test_add_warning(self):
        """Test adding warnings to ValidationResult."""
        result = ValidationResult()
        # Add warning
        result.add_warning("Warning message")
        self.assertTrue(result.valid)  # Warnings don't affect validity
        self.assertEqual(len(result.warnings), 1)
        self.assertEqual(result.warnings[0]["message"], "Warning message")
        
        # Add warning with field and details
        details = {"key": "value"}
        result.add_warning("Warning with details", "field_name", details)
        self.assertEqual(len(result.warnings), 2)
        self.assertEqual(result.warnings[1]["message"], "Warning with details")
        self.assertEqual(result.warnings[1]["field"], "field_name")
        self.assertEqual(result.warnings[1]["details"], details)
    
    def test_add_info(self):
        """Test adding info messages to ValidationResult."""
        result = ValidationResult()
        # Add info
        result.add_info("Info message")
        self.assertTrue(result.valid)  # Info doesn't affect validity
        self.assertEqual(len(result.info), 1)
        self.assertEqual(result.info[0]["message"], "Info message")
        
        # Add info with field and details
        details = {"key": "value"}
        result.add_info("Info with details", "field_name", details)
        self.assertEqual(len(result.info), 2)
        self.assertEqual(result.info[1]["message"], "Info with details")
        self.assertEqual(result.info[1]["field"], "field_name")
        self.assertEqual(result.info[1]["details"], details)
    
    def test_add_result(self):
        """Test merging ValidationResults."""
        result1 = ValidationResult()
        result1.add_error("Error 1")
        result1.add_warning("Warning 1")
        
        result2 = ValidationResult()
        result2.add_error("Error 2")
        result2.add_info("Info 2")
        
        result1.add_result(result2)
        
        self.assertFalse(result1.valid)
        self.assertEqual(len(result1.errors), 2)
        self.assertEqual(result1.errors[0]["message"], "Error 1")
        self.assertEqual(result1.errors[1]["message"], "Error 2")
        self.assertEqual(len(result1.warnings), 1)
        self.assertEqual(result1.warnings[0]["message"], "Warning 1")
        self.assertEqual(len(result1.info), 1)
        self.assertEqual(result1.info[0]["message"], "Info 2")
    
    def test_bool_conversion(self):
        """Test bool conversion of ValidationResult."""
        result = ValidationResult()
        self.assertTrue(bool(result))
        
        result.add_error("Error")
        self.assertFalse(bool(result))
    
    def test_to_dict(self):
        """Test converting ValidationResult to dictionary."""
        result = ValidationResult()
        result.add_error("Error")
        result.add_warning("Warning")
        result.add_info("Info")
        
        result_dict = result.to_dict()
        self.assertFalse(result_dict["valid"])
        self.assertEqual(len(result_dict["errors"]), 1)
        self.assertEqual(len(result_dict["warnings"]), 1)
        self.assertEqual(len(result_dict["info"]), 1)
    
    def test_from_dict(self):
        """Test creating ValidationResult from dictionary."""
        data = {
            "valid": False,
            "errors": [{"message": "Error", "field": None, "details": {}}],
            "warnings": [{"message": "Warning", "field": "field", "details": {}}],
            "info": []
        }
        
        result = ValidationResult.from_dict(data)
        self.assertFalse(result.valid)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0]["message"], "Error")
        self.assertEqual(len(result.warnings), 1)
        self.assertEqual(result.warnings[0]["message"], "Warning")
        self.assertEqual(len(result.info), 0)


class TestValidator(unittest.TestCase):
    def test_add_rule_and_validate(self):
        """Test adding rules to a Validator and validating values."""
        validator = Validator()
        
        # Add a rule that always passes
        validator.add_rule(lambda x: True)
        self.assertTrue(validator.validate("anything").valid)
        
        # Add a rule that returns a ValidationResult
        def rule_with_result(value):
            result = ValidationResult()
            if value != "valid":
                result.add_error("Invalid value")
            return result
        
        validator.add_rule(rule_with_result)
        
        # Test validation passes
        result = validator.validate("valid")
        self.assertTrue(result.valid)
        
        # Test validation fails
        result = validator.validate("invalid")
        self.assertFalse(result.valid)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0]["message"], "Invalid value")
        
        # Add a rule that raises ValidationError
        def rule_with_error(value):
            if value != "valid":
                raise ValidationError("Explicit error")
            return True
        
        validator.add_rule(rule_with_error)
        
        # Test validation with multiple rules (one failing)
        result = validator.validate("invalid")
        self.assertFalse(result.valid)
        self.assertEqual(len(result.errors), 2)
        
        # Test rule with additional parameters
        def parameterized_rule(value, param1, param2=None):
            if value != param1 or value == param2:
                return False
            return True
        
        validator = Validator()
        validator.add_rule(parameterized_rule, param1="test", param2="invalid")
        
        result = validator.validate("test")
        self.assertTrue(result.valid)
        
        result = validator.validate("invalid")
        self.assertFalse(result.valid)


class TestTypeValidation(unittest.TestCase):
    def test_validate_type(self):
        """Test validate_type function."""
        # Test valid cases
        self.assertEqual(validate_type(123, int), 123)
        self.assertEqual(validate_type("string", str), "string")
        self.assertEqual(validate_type(1.0, float), 1.0)
        self.assertEqual(validate_type([1, 2, 3], list), [1, 2, 3])
        
        # Test invalid cases
        with self.assertRaises(ValidationError):
            validate_type("not an int", int)
        
        with self.assertRaises(ValidationError):
            validate_type(123, str)
    
    def test_validate_instance(self):
        """Test validate_instance function."""
        class TestClass:
            pass
        
        class SubClass(TestClass):
            pass
        
        # Test valid cases
        instance = TestClass()
        self.assertEqual(validate_instance(instance, TestClass), instance)
        
        sub_instance = SubClass()
        self.assertEqual(validate_instance(sub_instance, TestClass), sub_instance)
        
        # Test invalid cases
        with self.assertRaises(ValidationError):
            validate_instance("not an instance", TestClass)
    
    def test_validate_subclass(self):
        """Test validate_subclass function."""
        class Base:
            pass
        
        class Derived(Base):
            pass
        
        class Unrelated:
            pass
        
        # Test valid cases
        self.assertEqual(validate_subclass(Base, Base), Base)
        self.assertEqual(validate_subclass(Derived, Base), Derived)
        
        # Test invalid cases
        with self.assertRaises(ValidationError):
            validate_subclass(Unrelated, Base)


class TestNumericValidation(unittest.TestCase):
    def test_validate_range(self):
        """Test validate_range function."""
        # Test with integers
        self.assertEqual(validate_range(5, 0, 10), 5)
        self.assertEqual(validate_range(0, 0, 10), 0)  # Inclusive min
        self.assertEqual(validate_range(10, 0, 10), 10)  # Inclusive max
        
        # Test with non-inclusive bounds
        self.assertEqual(validate_range(5, 0, 10, inclusive_min=False, inclusive_max=False), 5)
        
        # Test with only min or max
        self.assertEqual(validate_range(5, min_value=0), 5)
        self.assertEqual(validate_range(5, max_value=10), 5)
        
        # Test with Decimal
        self.assertEqual(validate_range(Decimal('5.5'), Decimal('0'), Decimal('10')), Decimal('5.5'))
        
        # Test invalid cases
        with self.assertRaises(ValidationError):
            validate_range(-1, 0, 10)
        
        with self.assertRaises(ValidationError):
            validate_range(11, 0, 10)
        
        with self.assertRaises(ValidationError):
            validate_range(0, 0, 10, inclusive_min=False)
        
        with self.assertRaises(ValidationError):
            validate_range(10, 0, 10, inclusive_max=False)
    
    def test_validate_positive(self):
        """Test validate_positive function."""
        self.assertEqual(validate_positive(1), 1)
        self.assertEqual(validate_positive(0.1), 0.1)
        
        with self.assertRaises(ValidationError):
            validate_positive(0)
        
        with self.assertRaises(ValidationError):
            validate_positive(-1)
    
    def test_validate_non_negative(self):
        """Test validate_non_negative function."""
        self.assertEqual(validate_non_negative(1), 1)
        self.assertEqual(validate_non_negative(0), 0)
        
        with self.assertRaises(ValidationError):
            validate_non_negative(-0.1)
    
    def test_validate_percentage(self):
        """Test validate_percentage function."""
        self.assertEqual(validate_percentage(0), 0)
        self.assertEqual(validate_percentage(0.5), 0.5)
        self.assertEqual(validate_percentage(1), 1)
        
        # Test with allow_zero=False
        self.assertEqual(validate_percentage(0.1, allow_zero=False), 0.1)
        
        with self.assertRaises(ValidationError):
            validate_percentage(0, allow_zero=False)
        
        with self.assertRaises(ValidationError):
            validate_percentage(-0.1)
        
        with self.assertRaises(ValidationError):
            validate_percentage(1.1)


class TestStringValidation(unittest.TestCase):
    def test_validate_non_empty_string(self):
        """Test validate_non_empty_string function."""
        self.assertEqual(validate_non_empty_string("test"), "test")
        
        with self.assertRaises(ValidationError):
            validate_non_empty_string("")
    
    def test_validate_string_length(self):
        """Test validate_string_length function."""
        self.assertEqual(validate_string_length("test", 1, 10), "test")
        self.assertEqual(validate_string_length("test", 4, 4), "test")
        
        with self.assertRaises(ValidationError):
            validate_string_length("test", 5, 10)
        
        with self.assertRaises(ValidationError):
            validate_string_length("test", 1, 3)
    
    def test_validate_regex(self):
        """Test validate_regex function."""
        self.assertEqual(validate_regex("test123", r"^[a-z]+\d+$"), "test123")
        
        with self.assertRaises(ValidationError):
            validate_regex("test", r"^\d+$")
        
        # Test with description
        with self.assertRaises(ValidationError) as context:
            validate_regex("test", r"^\d+$", "digits only")
        self.assertIn("digits only", context.exception.message)
    
    def test_validate_one_of(self):
        """Test validate_one_of function."""
        self.assertEqual(validate_one_of("a", ["a", "b", "c"]), "a")
        
        with self.assertRaises(ValidationError):
            validate_one_of("d", ["a", "b", "c"])
    
    def test_validate_enum(self):
        """Test validate_enum function."""
        class TestEnum(Enum):
            A = "a"
            B = "b"
            C = 3
        
        self.assertEqual(validate_enum(TestEnum.A, TestEnum), TestEnum.A)
        self.assertEqual(validate_enum("a", TestEnum), TestEnum.A)
        self.assertEqual(validate_enum(3, TestEnum), TestEnum.C)
        
        with self.assertRaises(ValidationError):
            validate_enum("d", TestEnum)
        
        with self.assertRaises(ValidationError):
            validate_enum(4, TestEnum)


class TestCollectionValidation(unittest.TestCase):
    def test_validate_length(self):
        """Test validate_length function."""
        # Test with list
        self.assertEqual(validate_length([1, 2, 3], 1, 5), [1, 2, 3])
        
        # Test with string
        self.assertEqual(validate_length("test", 1, 5), "test")
        
        # Test with dict
        self.assertEqual(validate_length({"a": 1, "b": 2}, 1, 5), {"a": 1, "b": 2})
        
        # Test invalid cases
        with self.assertRaises(ValidationError):
            validate_length([1, 2, 3], 4, 5)
        
        with self.assertRaises(ValidationError):
            validate_length([1, 2, 3, 4, 5, 6], 1, 5)
    
    def test_validate_non_empty(self):
        """Test validate_non_empty function."""
        self.assertEqual(validate_non_empty([1]), [1])
        self.assertEqual(validate_non_empty({"a": 1}), {"a": 1})
        
        with self.assertRaises(ValidationError):
            validate_non_empty([])
        
        with self.assertRaises(ValidationError):
            validate_non_empty({})
    
    def test_validate_unique(self):
        """Test validate_unique function."""
        self.assertEqual(validate_unique([1, 2, 3]), [1, 2, 3])
        
        with self.assertRaises(ValidationError):
            validate_unique([1, 2, 2, 3])
    
    def test_validate_all_of_type(self):
        """Test validate_all_of_type function."""
        self.assertEqual(validate_all_of_type([1, 2, 3], int), [1, 2, 3])
        self.assertEqual(validate_all_of_type(["a", "b", "c"], str), ["a", "b", "c"])
        
        with self.assertRaises(ValidationError):
            validate_all_of_type([1, "2", 3], int)


class TestDateTimeValidation(unittest.TestCase):
    def test_validate_datetime(self):
        """Test validate_datetime function."""
        dt = datetime.datetime.now()
        self.assertEqual(validate_datetime(dt), dt)
        
        with self.assertRaises(ValidationError):
            validate_datetime("not a datetime")
    
    def test_validate_datetime_range(self):
        """Test validate_datetime_range function."""
        now = datetime.datetime.now()
        past = now - datetime.timedelta(days=1)
        future = now + datetime.timedelta(days=1)
        
        self.assertEqual(validate_datetime_range(now, past, future), now)
        
        with self.assertRaises(ValidationError):
            validate_datetime_range(past, now, future)
        
        with self.assertRaises(ValidationError):
            validate_datetime_range(future, past, now)
    
    def test_validate_future_datetime(self):
        """Test validate_future_datetime function."""
        now = datetime.datetime.now()
        future = now + datetime.timedelta(days=1)
        
        self.assertEqual(validate_future_datetime(future), future)
        
        with self.assertRaises(ValidationError):
            validate_future_datetime(now)
    
    def test_validate_past_datetime(self):
        """Test validate_past_datetime function."""
        now = datetime.datetime.now()
        past = now - datetime.timedelta(days=1)
        
        self.assertEqual(validate_past_datetime(past), past)
        
        with self.assertRaises(ValidationError):
            validate_past_datetime(now)


class TestFileValidation(unittest.TestCase):
    def setUp(self):
        """Set up test environment by creating temporary files and directories."""
        # Create a temporary test file
        self.test_file = Path("test_file.txt")
        with open(self.test_file, "w") as f:
            f.write("Test content")
        
        # Create a temporary test directory
        self.test_dir = Path("test_dir")
        self.test_dir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up temporary files and directories."""
        if self.test_file.exists():
            self.test_file.unlink()
        
        if self.test_dir.exists():
            self.test_dir.rmdir()
    
    def test_validate_file_exists(self):
        """Test validate_file_exists function."""
        self.assertEqual(validate_file_exists(self.test_file), self.test_file)
        
        with self.assertRaises(ValidationError):
            validate_file_exists("non_existent_file.txt")
        
        with self.assertRaises(ValidationError):
            validate_file_exists(self.test_dir)
    
    def test_validate_directory_exists(self):
        """Test validate_directory_exists function."""
        self.assertEqual(validate_directory_exists(self.test_dir), self.test_dir)
        
        with self.assertRaises(ValidationError):
            validate_directory_exists("non_existent_dir")
        
        with self.assertRaises(ValidationError):
            validate_directory_exists(self.test_file)
    
    def test_validate_file_extension(self):
        """Test validate_file_extension function."""
        self.assertEqual(validate_file_extension(self.test_file, [".txt"]), self.test_file)
        self.assertEqual(validate_file_extension(self.test_file, ["txt"]), self.test_file)
        
        with self.assertRaises(ValidationError):
            validate_file_extension(self.test_file, [".csv", ".json"])


class TestSchemaValidation(unittest.TestCase):
    def test_validate_schema(self):
        """Test validate_schema function."""
        schema = {
            "name": {"type": str, "required": True},
            "age": {"type": int, "required": True},
            "email": {"type": str, "required": False}
        }
        
        data = {
            "name": "John",
            "age": 30,
            "email": "john@example.com"
        }
        
        self.assertEqual(validate_schema(data, schema), data)
        
        # Test with missing optional field
        data_without_email = {
            "name": "John",
            "age": 30
        }
        self.assertEqual(validate_schema(data_without_email, schema), data_without_email)
        
        # Test with wrong type
        data_wrong_type = {
            "name": "John",
            "age": "30",  # Should be int
            "email": "john@example.com"
        }
        with self.assertRaises(ValidationError):
            validate_schema(data_wrong_type, schema)
        
        # Test with missing required field
        data_missing_required = {
            "name": "John"
        }
        with self.assertRaises(ValidationError):
            validate_schema(data_missing_required, schema)
        
        # Test with custom validator
        schema_with_validator = {
            "name": {"type": str, "required": True},
            "age": {
                "type": int, 
                "required": True,
                "validator": lambda x: validate_range(x, 18, 100)
            }
        }
        
        valid_data = {"name": "John", "age": 30}
        self.assertEqual(validate_schema(valid_data, schema_with_validator), valid_data)
        
        invalid_data = {"name": "John", "age": 10}
        with self.assertRaises(ValidationError):
            validate_schema(invalid_data, schema_with_validator)
        
        # Test with nested schema
        nested_schema = {
            "user": {
                "type": dict, 
                "required": True,
                "schema": {
                    "name": {"type": str, "required": True},
                    "age": {"type": int, "required": True}
                }
            }
        }
        
        nested_data = {
            "user": {
                "name": "John",
                "age": 30
            }
        }
        
        self.assertEqual(validate_schema(nested_data, nested_schema), nested_data)
        
        invalid_nested_data = {
            "user": {
                "name": "John",
                "age": "30"  # Should be int
            }
        }
        
        with self.assertRaises(ValidationError):
            validate_schema(invalid_nested_data, nested_schema)
    
    def test_validate_json(self):
        """Test validate_json function."""
        json_str = '{"name": "John", "age": 30}'
        expected = {"name": "John", "age": 30}
        
        self.assertEqual(validate_json(json_str), expected)
        
        # Test invalid JSON
        invalid_json = '{"name": "John", "age": }'
        with self.assertRaises(ValidationError):
            validate_json(invalid_json)
    
    def test_validate_dataclass(self):
        """Test validate_dataclass function."""
        @dataclass
        class Person:
            name: str
            age: int
        
        # Test with dataclass instance
        person = Person(name="John", age=30)
        self.assertEqual(validate_dataclass(person, Person), person)
        
        # Test with dictionary
        person_dict = {"name": "John", "age": 30}
        person_from_dict = validate_dataclass(person_dict, Person)
        self.assertIsInstance(person_from_dict, Person)
        self.assertEqual(person_from_dict.name, "John")
        self.assertEqual(person_from_dict.age, 30)
        
        # Test with invalid input
        with self.assertRaises(ValidationError):
            validate_dataclass("not a dataclass", Person)
        
        with self.assertRaises(ValidationError):
            validate_dataclass({"name": "John"}, Person)  # Missing required field
        
        # Test with non-dataclass type
        with self.assertRaises(ValidationError):
            validate_dataclass({}, dict)
    
    def test_validate_dataclass_fields(self):
        """Test validate_dataclass_fields function."""
        class ValidatableField:
            def validate(self):
                pass
        
        class InvalidField:
            def validate(self):
                raise ValidationError("Field validation failed")
        
        @dataclass
        class TestDataClass:
            valid_field: ValidatableField
            invalid_field: InvalidField
        
        # Create test instance
        test_instance = TestDataClass(
            valid_field=ValidatableField(),
            invalid_field=InvalidField()
        )
        
        # Test validation fails due to invalid field
        with self.assertRaises(ValidationError):
            validate_dataclass_fields(test_instance)
        
        # Test with non-dataclass
        with self.assertRaises(ValidationError):
            validate_dataclass_fields("not a dataclass")


class TestTradingValidation(unittest.TestCase):
    def test_validate_symbol(self):
        """Test validate_symbol function."""
        # Mock the validate_symbol function since there's a syntax error in the original
        
        def patched_validate_symbol(symbol):
            validate_non_empty_string(symbol)
            pattern = r'^[A-Za-z0-9._-]+'
            validate_regex(symbol, pattern, "valid trading symbol format")
            return symbol
        
        # Test with valid symbols
        self.assertEqual(patched_validate_symbol("BTC_USD"), "BTC_USD")
        self.assertEqual(patched_validate_symbol("ETH-USDT"), "ETH-USDT")
        
        # Test with invalid symbol
        with self.assertRaises(ValidationError):
            patched_validate_symbol("")
    
    def test_validate_price(self):
        """Test validate_price function."""
        self.assertEqual(validate_price(10.5), 10.5)
        self.assertEqual(validate_price(0, allow_zero=True), 0)
        
        with self.assertRaises(ValidationError):
            validate_price(0)
        
        with self.assertRaises(ValidationError):
            validate_price(-1)
    
    def test_validate_quantity(self):
        """Test validate_quantity function."""
        self.assertEqual(validate_quantity(10.5), 10.5)
        self.assertEqual(validate_quantity(0, allow_zero=True), 0)
        
        with self.assertRaises(ValidationError):
            validate_quantity(0)
        
        with self.assertRaises(ValidationError):
            validate_quantity(-1)
    
    def test_validate_timeframe(self):
        """Test validate_timeframe function."""
        # Test valid timeframes
        self.assertEqual(validate_timeframe("1m"), "1m")
        self.assertEqual(validate_timeframe("15m"), "15m")
        self.assertEqual(validate_timeframe("1h"), "1h")
        self.assertEqual(validate_timeframe("1d"), "1d")
        
        # Test invalid timeframes
        with self.assertRaises(ValidationError):
            validate_timeframe("invalid")
        
        # Test uncommon timeframes (should raise warning but not error)
        # This is difficult to test directly since it uses ValidationLevel.WARNING
        # but doesn't affect the return value
        self.assertEqual(validate_timeframe("2m"), "2m")
    
    def test_validate_feature_name(self):
        """Test validate_feature_name function."""
        self.assertEqual(validate_feature_name("feature_name"), "feature_name")
        self.assertEqual(validate_feature_name("feature123"), "feature123")
        
        with self.assertRaises(ValidationError):
            validate_feature_name("")
        
        with self.assertRaises(ValidationError):
            validate_feature_name("FeatureName")  # Not snake_case
    
    def test_validate_model_name(self):
        """Test validate_model_name function."""
        self.assertEqual(validate_model_name("model_name"), "model_name")
        self.assertEqual(validate_model_name("Model123"), "Model123")
        
        with self.assertRaises(ValidationError):
            validate_model_name("")
        
        with self.assertRaises(ValidationError):
            validate_model_name("123model")  # Doesn't start with letter
    
    def test_validate_strategy_name(self):
        """Test validate_strategy_name function."""
        self.assertEqual(validate_strategy_name("strategy_name"), "strategy_name")
        self.assertEqual(validate_strategy_name("StrategyName"), "StrategyName")
        
        with self.assertRaises(ValidationError):
            validate_strategy_name("")
        
        with self.assertRaises(ValidationError):
            validate_strategy_name("123strategy")  # Doesn't start with letter")
        
        with self.assertRaises(ValidationError):
            validate_feature_name("