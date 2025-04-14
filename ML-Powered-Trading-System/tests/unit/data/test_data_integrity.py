import datetime
import json
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, Field, ValidationError

from core.data.data_integrity import (
    DataCategory,
    DataFrameIntegrityRule,
    DataFrameNullCheckRule,
    DataIntegrityFactory,
    DataIntegrityService,
    DataQualityScorer,
    IntegrityLevel,
    IntegrityStatus,
    MarketDataIntegrityRule,
    NumericRangeRule,
    PydanticSchemaAdapter,
    RuleRegistry,
    SchemaRegistry,
    TimeSeriesIntegrityRule,
    ValidationContext,
    ValidationResult,
    ValidationRule,
    ValidationRuleFactory,
    create_context
)


class TestPydanticSchemaAdapter(unittest.TestCase):
    """Test the PydanticSchemaAdapter class."""

    def test_create_schema_class(self):
        """Test creating a Pydantic schema class dynamically."""
        # Define fields for the schema
        fields = {
            "name": (str, {"description": "Name field"}),
            "age": (int, {"ge": 0, "description": "Age field"}),
            "is_active": (bool, {"description": "Activity status"})
        }

        # Create schema class
        schema_class = PydanticSchemaAdapter.create_schema_class(
            fields=fields,
            class_name="PersonSchema"
        )

        # Verify the schema class
        self.assertEqual(schema_class.__name__, "PersonSchema")
        self.assertTrue(issubclass(schema_class, BaseModel))

        # Test field annotations
        self.assertEqual(schema_class.__annotations__["name"], str)
        self.assertEqual(schema_class.__annotations__["age"], int)
        self.assertEqual(schema_class.__annotations__["is_active"], bool)

        # Test validation with the schema
        valid_data = {"name": "John", "age": 30, "is_active": True}
        instance = schema_class(**valid_data)
        self.assertEqual(instance.name, "John")
        self.assertEqual(instance.age, 30)
        self.assertEqual(instance.is_active, True)

        # Test validation failure
        invalid_data = {"name": "John", "age": -5, "is_active": True}
        with self.assertRaises(ValidationError):
            schema_class(**invalid_data)

    def test_validate_method(self):
        """Test the validate method of PydanticSchemaAdapter."""
        # Create a simple Pydantic model
        class TestModel(BaseModel):
            name: str
            age: int = Field(ge=0)

        # Test validation success
        valid_data = {"name": "John", "age": 30}
        is_valid, errors = PydanticSchemaAdapter.validate(TestModel, valid_data)

        self.assertTrue(is_valid)
        self.assertIsNone(errors)

        # Test validation failure
        invalid_data = {"name": "John", "age": -5}
        is_valid, errors = PydanticSchemaAdapter.validate(TestModel, invalid_data)

        self.assertFalse(is_valid)
        self.assertIsNotNone(errors)
        self.assertIn("age", errors)


class TestValidationContext(unittest.TestCase):
    """Test the ValidationContext class."""

    def test_create_validation_context(self):
        """Test creating a validation context."""
        context = ValidationContext(
            source="test",
            category=DataCategory.MARKET_DATA,
            level=IntegrityLevel.STRICT,
            reference_id="test-123",
            metadata={"test": "value"}
        )

        self.assertEqual(context.source, "test")
        self.assertEqual(context.category, DataCategory.MARKET_DATA)
        self.assertEqual(context.level, IntegrityLevel.STRICT)
        self.assertEqual(context.reference_id, "test-123")
        self.assertEqual(context.metadata, {"test": "value"})
        self.assertIsInstance(context.timestamp, datetime.datetime)

    def test_create_context_helper(self):
        """Test the create_context helper function."""
        context = create_context(
            source="test_source",
            category=DataCategory.ORDER,
            level=IntegrityLevel.WARNING
        )

        self.assertEqual(context.source, "test_source")
        self.assertEqual(context.category, DataCategory.ORDER)
        self.assertEqual(context.level, IntegrityLevel.WARNING)
        self.assertIsNone(context.reference_id)
        self.assertEqual(context.metadata, {})


class TestValidationResult(unittest.TestCase):
    """Test the ValidationResult class."""

    def setUp(self):
        self.context = ValidationContext(
            source="test",
            category=DataCategory.MARKET_DATA
        )
        self.result = ValidationResult(self.context)

    def test_initial_state(self):
        """Test the initial state of a validation result."""
        self.assertEqual(self.result.status, IntegrityStatus.UNKNOWN)
        self.assertFalse(self.result.is_valid)
        self.assertEqual(len(self.result.errors), 0)
        self.assertEqual(len(self.result.warnings), 0)
        self.assertEqual(len(self.result.metrics), 0)

    def test_add_error(self):
        """Test adding an error to the validation result."""
        self.result.add_error("test_field", "Test error message", "test_value")

        self.assertEqual(len(self.result.errors), 1)
        self.assertEqual(self.result.errors[0]["field"], "test_field")
        self.assertEqual(self.result.errors[0]["message"], "Test error message")
        self.assertEqual(self.result.errors[0]["value"], "test_value")

    def test_add_warning(self):
        """Test adding a warning to the validation result."""
        self.result.add_warning("test_field", "Test warning message", "test_value")

        self.assertEqual(len(self.result.warnings), 1)
        self.assertEqual(self.result.warnings[0]["field"], "test_field")
        self.assertEqual(self.result.warnings[0]["message"], "Test warning message")
        self.assertEqual(self.result.warnings[0]["value"], "test_value")

    def test_add_metric(self):
        """Test adding a metric to the validation result."""
        self.result.add_metric("test_metric", 0.95)

        self.assertEqual(len(self.result.metrics), 1)
        self.assertEqual(self.result.metrics["test_metric"], 0.95)

    def test_update_status_no_errors_no_warnings(self):
        """Test updating status with no errors or warnings."""
        self.result.update_status()

        self.assertEqual(self.result.status, IntegrityStatus.PASSED)
        self.assertTrue(self.result.is_valid)

    def test_update_status_with_warnings(self):
        """Test updating status with warnings but no errors."""
        self.result.add_warning("test_field", "Test warning")
        self.result.update_status()

        self.assertEqual(self.result.status, IntegrityStatus.PARTIAL)
        self.assertTrue(self.result.is_valid)  # Warnings don't invalidate data

    def test_update_status_with_errors(self):
        """Test updating status with errors."""
        self.result.add_error("test_field", "Test error")
        self.result.update_status()

        self.assertEqual(self.result.status, IntegrityStatus.FAILED)
        self.assertFalse(self.result.is_valid)

    def test_update_status_with_errors_and_warnings(self):
        """Test updating status with both errors and warnings."""
        self.result.add_error("test_field", "Test error")
        self.result.add_warning("test_field", "Test warning")
        self.result.update_status()

        self.assertEqual(self.result.status, IntegrityStatus.FAILED)
        self.assertFalse(self.result.is_valid)

    def test_to_dict(self):
        """Test converting the validation result to a dictionary."""
        self.result.add_error("field1", "Error 1")
        self.result.add_warning("field2", "Warning 1")
        self.result.add_metric("metric1", 0.85)
        self.result.update_status()

        result_dict = self.result.to_dict()

        self.assertIn("context", result_dict)
        self.assertEqual(result_dict["status"], "failed")
        self.assertFalse(result_dict["is_valid"])
        self.assertIn("timestamp", result_dict)
        self.assertEqual(len(result_dict["errors"]), 1)
        self.assertEqual(len(result_dict["warnings"]), 1)
        self.assertEqual(len(result_dict["metrics"]), 1)

    def test_to_json(self):
        """Test converting the validation result to JSON."""
        self.result.add_error("field1", "Error 1")
        self.result.update_status()

        json_str = self.result.to_json()
        parsed_json = json.loads(json_str)

        self.assertIn("context", parsed_json)
        self.assertEqual(parsed_json["status"], "failed")
        self.assertFalse(parsed_json["is_valid"])

    def test_bool_representation(self):
        """Test the boolean representation of the validation result."""
        # Initially invalid
        self.assertFalse(bool(self.result))

        # Add error, still invalid
        self.result.add_error("field", "error")
        self.result.update_status()
        self.assertFalse(bool(self.result))

        # Create a new valid result
        valid_result = ValidationResult(self.context)
        valid_result.update_status()
        self.assertTrue(bool(valid_result))


class TestSchemaRegistry(unittest.TestCase):
    """Test the SchemaRegistry class."""

    def setUp(self):
        self.registry = SchemaRegistry()

        # Define a test schema
        class TestSchema(BaseModel):
            field1: str
            field2: int

        self.test_schema = TestSchema

    def test_register_schema(self):
        """Test registering a schema."""
        self.registry.register_schema(
            category=DataCategory.MARKET_DATA,
            schema=self.test_schema,
            version="v1"
        )

        # Verify schema was registered
        schema = self.registry.get_schema(DataCategory.MARKET_DATA, "v1")
        self.assertEqual(schema, self.test_schema)

    def test_register_schema_sets_default_version(self):
        """Test that registering the first schema for a category sets it as default."""
        self.registry.register_schema(
            category=DataCategory.MARKET_DATA,
            schema=self.test_schema,
            version="v1"
        )

        # Verify default version was set
        default_version = self.registry.get_default_version(DataCategory.MARKET_DATA)
        self.assertEqual(default_version, "v1")

    def test_set_default_version(self):
        """Test setting the default version for a category."""
        # Register multiple schemas
        self.registry.register_schema(
            category=DataCategory.MARKET_DATA,
            schema=self.test_schema,
            version="v1"
        )

        self.registry.register_schema(
            category=DataCategory.MARKET_DATA,
            schema=self.test_schema,
            version="v2"
        )

        # Set default version
        self.registry.set_default_version(DataCategory.MARKET_DATA, "v2")

        # Verify default version was set
        default_version = self.registry.get_default_version(DataCategory.MARKET_DATA)
        self.assertEqual(default_version, "v2")

    def test_set_default_version_invalid(self):
        """Test setting an invalid default version raises an error."""
        with self.assertRaises(ValueError):
            self.registry.set_default_version(DataCategory.MARKET_DATA, "non_existent")

    def test_get_schema_with_version(self):
        """Test getting a schema with a specific version."""
        # Register schema
        self.registry.register_schema(
            category=DataCategory.MARKET_DATA,
            schema=self.test_schema,
            version="v1"
        )

        # Get schema with version
        schema = self.registry.get_schema(DataCategory.MARKET_DATA, "v1")
        self.assertEqual(schema, self.test_schema)

    def test_get_schema_default_version(self):
        """Test getting a schema with the default version."""
        # Register schema
        self.registry.register_schema(
            category=DataCategory.MARKET_DATA,
            schema=self.test_schema,
            version="v1"
        )

        # Get schema with default version
        schema = self.registry.get_schema(DataCategory.MARKET_DATA)
        self.assertEqual(schema, self.test_schema)

    def test_get_schema_no_default_version(self):
        """Test getting a schema with no default version raises an error."""
        # Create a category with no schemas
        with self.assertRaises(ValueError):
            self.registry.get_schema(DataCategory.ORDER)

    def test_get_schema_invalid_version(self):
        """Test getting a schema with an invalid version raises an error."""
        # Register schema
        self.registry.register_schema(
            category=DataCategory.MARKET_DATA,
            schema=self.test_schema,
            version="v1"
        )

        # Try to get schema with invalid version
        with self.assertRaises(ValueError):
            self.registry.get_schema(DataCategory.MARKET_DATA, "non_existent")

    def test_get_versions(self):
        """Test getting all versions for a category."""
        # Register multiple schemas
        self.registry.register_schema(
            category=DataCategory.MARKET_DATA,
            schema=self.test_schema,
            version="v1"
        )

        self.registry.register_schema(
            category=DataCategory.MARKET_DATA,
            schema=self.test_schema,
            version="v2"
        )

        # Get versions
        versions = self.registry.get_versions(DataCategory.MARKET_DATA)
        self.assertEqual(set(versions), {"v1", "v2"})


class TestRuleRegistry(unittest.TestCase):
    """Test the RuleRegistry class."""

    def setUp(self):
        self.registry = RuleRegistry()

        # Define a test rule
        self.test_rule = NumericRangeRule(
            field_name="test_field",
            min_value=0,
            max_value=100
        )

    def test_register_rule(self):
        """Test registering a rule."""
        self.registry.register_rule(
            category=DataCategory.MARKET_DATA,
            rule=self.test_rule
        )

        # Verify rule was registered
        rules = self.registry.get_rules(DataCategory.MARKET_DATA)
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0], self.test_rule)

    def test_register_multiple_rules(self):
        """Test registering multiple rules for a category."""
        # Register first rule
        self.registry.register_rule(
            category=DataCategory.MARKET_DATA,
            rule=self.test_rule
        )

        # Register second rule
        second_rule = ValidationRule(name="second_rule")
        self.registry.register_rule(
            category=DataCategory.MARKET_DATA,
            rule=second_rule
        )

        # Verify both rules were registered
        rules = self.registry.get_rules(DataCategory.MARKET_DATA)
        self.assertEqual(len(rules), 2)
        self.assertEqual(rules[0], self.test_rule)
        self.assertEqual(rules[1], second_rule)

    def test_get_rules_empty(self):
        """Test getting rules for a category with no rules."""
        rules = self.registry.get_rules(DataCategory.MARKET_DATA)
        self.assertEqual(len(rules), 0)


class TestNumericRangeRule(unittest.TestCase):
    """Test the NumericRangeRule class."""

    def setUp(self):
        self.context = ValidationContext(
            source="test",
            category=DataCategory.MARKET_DATA
        )

        self.rule = NumericRangeRule(
            field_name="price",
            min_value=0,
            max_value=100,
            name="price_range_check",
            description="Check price is between 0 and 100",
            severity=IntegrityLevel.WARNING
        )

    def test_validate_valid_data(self):
        """Test validating data that meets the range requirement."""
        data = {"price": 50}
        result = self.rule.validate(data, self.context)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.status, IntegrityStatus.PASSED)
        self.assertEqual(len(result.errors), 0)

    def test_validate_below_min(self):
        """Test validating data below the minimum value."""
        data = {"price": -10}
        result = self.rule.validate(data, self.context)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.status, IntegrityStatus.FAILED)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0]["field"], "price")
        self.assertIn("below minimum", result.errors[0]["message"])

    def test_validate_above_max(self):
        """Test validating data above the maximum value."""
        data = {"price": 150}
        result = self.rule.validate(data, self.context)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.status, IntegrityStatus.FAILED)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0]["field"], "price")
        self.assertIn("above maximum", result.errors[0]["message"])

    def test_validate_non_numeric(self):
        """Test validating non-numeric data."""
        data = {"price": "not a number"}
        result = self.rule.validate(data, self.context)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.status, IntegrityStatus.FAILED)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0]["field"], "price")
        self.assertIn("must be numeric", result.errors[0]["message"])

    def test_validate_field_not_found(self):
        """Test validating data where the field is not found."""
        data = {"not_price": 50}
        result = self.rule.validate(data, self.context)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.status, IntegrityStatus.FAILED)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0]["field"], "price")
        self.assertIn("not found", result.errors[0]["message"])

    def test_validate_nested_field(self):
        """Test validating a nested field using dot notation."""
        # Create a rule for a nested field
        nested_rule = NumericRangeRule(
            field_name="data.price",
            min_value=0,
            max_value=100
        )

        # Test with valid nested data
        data = {"data": {"price": 50}}
        result = nested_rule.validate(data, self.context)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.status, IntegrityStatus.PASSED)

        # Test with invalid nested data
        data = {"data": {"price": 150}}
        result = nested_rule.validate(data, self.context)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.status, IntegrityStatus.FAILED)
        self.assertEqual(len(result.errors), 1)
        self.assertEqual(result.errors[0]["field"], "data.price")
        self.assertIn("above maximum", result.errors[0]["message"])


class TestDataFrameIntegrityRule(unittest.TestCase):
    """Test the DataFrameIntegrityRule class."""

    def setUp(self):
        self.context = ValidationContext(
            source="test",
            category=DataCategory.MARKET_DATA
        )

        self.rule = DataFrameIntegrityRule(
            name="dataframe_test",
            required_columns=["timestamp", "price"],
            min_rows=2
        )

        # Create test dataframe
        self.df = pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=5),
            "price": [100, 101, 102, 103, 104],
            "volume": [1000, 1100, 1200, 1300, 1400]
        })

    def test_validate_valid_dataframe(self):
        """Test validating a valid dataframe."""
        result = self.rule.validate(self.df, self.context)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.status, IntegrityStatus.PASSED)
        self.assertEqual(len(result.errors), 0)

        # Check metrics
        self.assertEqual(result.metrics["row_count"], 5)
        self.assertEqual(result.metrics["column_count"], 3)
        self.assertEqual(result.metrics["missing_value_percentage"], 0.0)

    def test_validate_not_dataframe(self):
        """Test validating data that is not a dataframe."""
        data = {"not": "a dataframe"}
        result = self.rule.validate(data, self.context)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.status, IntegrityStatus.FAILED)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("Expected pandas DataFrame", result.errors[0]["message"])

    def test_validate_missing_columns(self):
        """Test validating a dataframe with missing required columns."""
        # Create dataframe with missing column
        df_missing = self.df.drop(columns=["price"])
        result = self.rule.validate(df_missing, self.context)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.status, IntegrityStatus.FAILED)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("Missing required columns", result.errors[0]["message"])
        self.assertIn("price", result.errors[0]["message"])

    def test_validate_too_few_rows(self):
        """Test validating a dataframe with too few rows."""
        # Create dataframe with fewer rows than required
        df_small = self.df.iloc[0:1].copy()
        result = self.rule.validate(df_small, self.context)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.status, IntegrityStatus.FAILED)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("minimum required", result.errors[0]["message"])

    def test_validate_with_missing_values(self):
        """Test validating a dataframe with missing values."""
        # Create dataframe with missing values
        df_na = self.df.copy()
        df_na.loc[0, "price"] = np.nan

        result = self.rule.validate(df_na, self.context)

        self.assertTrue(result.is_valid)  # Base rule doesn't fail on NAs
        self.assertGreater(result.metrics["missing_value_percentage"], 0.0)


class TestDataFrameOutlierRule(unittest.TestCase):
    """Test the DataFrameOutlierRule class."""

    def setUp(self):
        self.context = ValidationContext(
            source="test",
            category=DataCategory.FEATURE
        )

        self.rule = DataFrameOutlierRule(
            columns_to_check=["value"],
            std_dev_threshold=2.0
        )

        # Create test dataframe with normally distributed data
        np.random.seed(42)  # For reproducibility
        self.df = pd.DataFrame({
            "value": np.random.normal(100, 10, 100),  # mean=100, std=10, n=100
            "category": np.random.choice(["A", "B", "C"], 100)
        })

    def test_validate_normal_distribution(self):
        """Test validating a dataframe with normally distributed values."""
        result = self.rule.validate(self.df, self.context)

        self.assertTrue(result.is_valid)  # Warnings don't invalidate
        self.assertIn("outlier_count_value", result.metrics)
        # With normal distribution and 2 std threshold, expect ~5% outliers
        expected_outliers = int(0.05 * len(self.df))
        self.assertAlmostEqual(
            result.metrics["outlier_count_value"],
            expected_outliers,
            delta=3  # Allow some variation due to random sampling
        )

    def test_validate_with_extreme_outliers(self):
        """Test validating a dataframe with extreme outliers."""
        # Add extreme outliers
        df_outliers = self.df.copy()
        df_outliers.loc[0, "value"] = 500  # 40 std deviations away
        df_outliers.loc[1, "value"] = -300  # 40 std deviations away

        result = self.rule.validate(df_outliers, self.context)

        self.assertTrue(result.is_valid)  # Warnings don't invalidate
        self.assertGreaterEqual(len(result.warnings), 1)
        self.assertIn("outliers", result.warnings[0]["message"])
        # Previous outliers plus the 2 extreme ones
        self.assertGreaterEqual(result.metrics["outlier_count_value"], 2)

    def test_validate_non_numeric_columns(self):
        """Test that the rule only validates numeric columns."""
        result = self.rule.validate(self.df, self.context)

        # Should not have metrics for the non-numeric 'category' column
        self.assertNotIn("outlier_count_category", result.metrics)
        self.assertNotIn("outlier_percentage_category", result.metrics)

    def test_validate_specific_columns(self):
        """Test validating only specific columns."""
        # Create a dataframe with multiple numeric columns
        df_multi = self.df.copy()
        df_multi["value2"] = df_multi["value"] * 2

        # Create a rule that only checks one column
        rule_specific = DataFrameOutlierRule(
            columns_to_check=["value"],
            std_dev_threshold=2.0
        )

        result = rule_specific.validate(df_multi, self.context)

        # Should have metrics for 'value' but not for 'value2'
        self.assertIn("outlier_count_value", result.metrics)
        self.assertNotIn("outlier_count_value2", result.metrics)


class TestDataFrameNullCheckRule(unittest.TestCase):
    """Test the DataFrameNullCheckRule class."""

    def setUp(self):
        self.context = ValidationContext(
            source="test",
            category=DataCategory.MARKET_DATA
        )

        self.rule = DataFrameNullCheckRule(
            columns_to_check=["price", "volume"],
            max_null_percentage=0.0,  # No nulls allowed
            severity=IntegrityLevel.WARNING
        )

        # Create test dataframe
        self.df = pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=5),
            "price": [100, 101, 102, 103, 104],
            "volume": [1000, 1100, 1200, 1300, 1400]
        })

    def test_validate_no_nulls(self):
        """Test validating a dataframe with no null values."""
        result = self.rule.validate(self.df, self.context)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.status, IntegrityStatus.PASSED)
        self.assertEqual(len(result.errors), 0)

        # Check metrics
        self.assertEqual(result.metrics["null_percentage_price"], 0.0)
        self.assertEqual(result.metrics["null_percentage_volume"], 0.0)

    def test_validate_with_nulls(self):
        """Test validating a dataframe with null values."""
        # Create dataframe with null values
        df_na = self.df.copy()
        df_na.loc[0, "price"] = np.nan
        df_na.loc[1, "volume"] = np.nan

        result = self.rule.validate(df_na, self.context)

        self.assertFalse(result.is_valid)
        self.assertEqual(result.status, IntegrityStatus.FAILED)
        self.assertEqual(len(result.errors), 2)  # One for each column with nulls

        # Check metrics
        self.assertEqual(result.metrics["null_percentage_price"], 20.0)  # 1/5 = 20%
        self.assertEqual(result.metrics["null_percentage_volume"], 20.0)  # 1/5 = 20%

    def test_validate_with_tolerance(self):
        """Test validating a dataframe with a null tolerance."""
        # Create rule with 25% null tolerance
        rule_with_tolerance = DataFrameNullCheckRule(
            columns_to_check=["price"],
            max_null_percentage=25.0  # Allow up to 25% nulls
        )

        # Create dataframe with 20% nulls (1/5)
        df_na = self.df.copy()
        df_na.loc[0, "price"] = np.nan

        result = rule_with_tolerance.validate(df_na, self.context)

        self.assertTrue(result.is_valid)  # Should pass with 20% nulls
        self.assertEqual(result.status, IntegrityStatus.PASSED)
        self.assertEqual(len(result.errors), 0)

        # Create dataframe with 40% nulls (2/5)
        df_na.loc[1, "price"] = np.nan

        result = rule_with_tolerance.validate(df_na, self.context)

        self.assertFalse(result.is_valid)  # Should fail with 40% nulls
        self.assertEqual(result.status, IntegrityStatus.FAILED)
        self.assertEqual(len(result.errors), 1)


class TimeSeriesIntegrityRuleTest(unittest.TestCase):
    """Test the TimeSeriesIntegrityRule class."""

    def setUp(self):
        self.context = ValidationContext(
            source="test",
            category=DataCategory.MARKET_DATA
        )

        self.rule = TimeSeriesIntegrityRule(
            timestamp_column="timestamp",
            expected_frequency="1H"
        )

        # Create a valid test dataframe with hourly data
        self.df = pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1H"),
            "value": [100, 101, 102, 103, 104]
        })

    def test_validate_valid_timeseries(self):
        """Test validating a valid time series with consistent frequency."""
        result = self.rule.validate(self.df, self.context)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.status, IntegrityStatus.PASSED)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.warnings), 0)

    def test_validate_frequency_mismatch(self):
        """Test validating a time series with frequency mismatches."""
        # Create dataframe with irregular timestamps
        df_irregular = self.df.copy()
        # Change one timestamp to create a gap
        df_irregular.loc[2, "timestamp"] = df_irregular.loc[1, "timestamp"] + pd.Timedelta(hours=3)

        result = self.rule.validate(df_irregular, self.context)

        self.assertTrue(result.is_valid)  # Warnings don't invalidate
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("frequency mismatches", result.warnings[0]["message"])
        self.assertIn("frequency_mismatch_count", result.metrics)

    def test_validate_with_gaps(self):
        """Test validating a time series with gaps but allowing gaps."""
        # Create rule that allows gaps
        rule_allow_gaps = TimeSeriesIntegrityRule(
            timestamp_column="timestamp",
            expected_frequency="1H",
            allow_gaps=True
        )

        # Create dataframe with a gap
        df_gap = self.df.copy()
        df_gap.loc[2, "timestamp"] = df_gap.loc[1, "timestamp"] + pd.Timedelta(hours=5)

        result = rule_allow_gaps.validate(df_gap, self.context)

        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.warnings), 0)  # No warnings when gaps are allowed

    def test_validate_with_max_gap_tolerance(self):
        """Test validating a time series with a maximum gap tolerance."""
        # Create rule with gap tolerance
        rule_with_tolerance = TimeSeriesIntegrityRule(
            timestamp_column="timestamp",
            expected_frequency="1H",
            allow_gaps=False,
            max_gap_tolerance=pd.Timedelta(hours=3)
        )

        # Create dataframe with gaps of different sizes
        df_gaps = self.df.copy()
        # Small gap (2 hours) - within tolerance
        df_gaps.loc[1, "timestamp"] = df_gaps.loc[0, "timestamp"] + pd.Timedelta(hours=2)
        # Large gap (4 hours) - exceeds tolerance
        df_gaps.loc[3, "timestamp"] = df_gaps.loc[2, "timestamp"] + pd.Timedelta(hours=4)

        result = rule_with_tolerance.validate(df_gaps, self.context)

        self.assertTrue(result.is_valid)  # Warnings don't invalidate
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("gaps larger than", result.warnings[0]["message"])
        self.assertEqual(result.metrics["gap_count"], 1)  # Only the large gap counts


class TestMarketDataIntegrityRule(unittest.TestCase):
    """Test the MarketDataIntegrityRule class."""

    def setUp(self):
        self.context = ValidationContext(
            source="test",
            category=DataCategory.MARKET_DATA
        )

        self.rule = MarketDataIntegrityRule(
            price_column="price",
            timestamp_column="timestamp",
            volume_column="volume",
            max_price_gap_percentage=5.0
        )

        # Create a valid test dataframe
        self.df = pd.DataFrame({
            "timestamp": pd.date_range(start="2023-01-01", periods=5, freq="1H"),
            "price": [100, 101, 102, 103, 104],  # Small increments, no gaps
            "volume": [1000, 1100, 1200, 1300, 1400]
        })

    def test_validate_valid_market_data(self):
        """Test validating valid market data."""
        result = self.rule.validate(self.df, self.context)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.status, IntegrityStatus.PASSED)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.warnings), 0)

    def test_validate_not_dataframe(self):
        """Test validating data that is not a dataframe."""
        data = {"not": "a dataframe"}
        result = self.rule.validate(data, self.context)

        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("Expected pandas DataFrame", result.errors[0]["message"])

    def test_validate_missing_columns(self):
        """Test validating market data with missing required columns."""
        # Create dataframe with missing price column
        df_missing = self.df.drop(columns=["price"])
        result = self.rule.validate(df_missing, self.context)

        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("Missing required columns", result.errors[0]["message"])

    def test_validate_price_gaps(self):
        """Test validating market data with price gaps."""
        # Create dataframe with a large price gap
        df_gap = self.df.copy()
        df_gap.loc[2, "price"] = 150  # Create a ~47% jump

        result = self.rule.validate(df_gap, self.context)

        self.assertTrue(result.is_valid)  # Warnings don't invalidate
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("price gaps larger than", result.warnings[0]["message"])
        self.assertIn("large_price_gap_count", result.metrics)
        self.assertIn("max_price_gap_percentage", result.metrics)

    def test_validate_negative_volumes(self):
        """Test validating market data with negative volumes."""
        # Create dataframe with negative volume
        df_neg_vol = self.df.copy()
        df_neg_vol.loc[2, "volume"] = -100

        result = self.rule.validate(df_neg_vol, self.context)

        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("negative volume", result.errors[0]["message"])
        self.assertEqual(result.metrics["negative_volume_count"], 1)

    def test_validate_zero_volumes(self):
        """Test validating market data with zero volumes."""
        # Create dataframe with zero volume
        df_zero_vol = self.df.copy()
        df_zero_vol.loc[2, "volume"] = 0

        result = self.rule.validate(df_zero_vol, self.context)

        self.assertTrue(result.is_valid)  # Warnings don't invalidate
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("zero volume", result.warnings[0]["message"])
        self.assertEqual(result.metrics["zero_volume_count"], 1)

    def test_validate_volume_spikes(self):
        """Test validating market data with volume spikes."""
        # Create dataframe with volume spike
        df_spike = self.df.copy()
        df_spike.loc[2, "volume"] = 10000  # Much higher than others

        result = self.rule.validate(df_spike, self.context)

        self.assertTrue(result.is_valid)  # Warnings don't invalidate
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("volume spikes", result.warnings[0]["message"])
        self.assertEqual(result.metrics["volume_spike_count"], 1)

    def test_validate_non_increasing_timestamps(self):
        """Test validating market data with non-increasing timestamps."""
        # Create dataframe with out-of-order timestamps
        df_unordered = self.df.copy()
        df_unordered.loc[2, "timestamp"] = df_unordered.loc[0, "timestamp"]  # Duplicate timestamp

        result = self.rule.validate(df_unordered, self.context)

        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.errors), 1)
        self.assertIn("not monotonically increasing", result.errors[0]["message"])

    def test_validate_duplicate_timestamps(self):
        """Test validating market data with duplicate timestamps."""
        # Create dataframe with duplicate timestamps
        df_dup = self.df.copy()
        df_dup.loc[1, "timestamp"] = df_dup.loc[0, "timestamp"]

        result = self.rule.validate(df_dup, self.context)

        self.assertTrue(result.is_valid)  # Warnings don't invalidate
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("duplicate timestamps", result.warnings[0]["message"])
        self.assertEqual(result.metrics["duplicate_timestamp_count"], 1)