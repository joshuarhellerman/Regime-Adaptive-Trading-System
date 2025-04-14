"""
Data Integrity Module

This module provides tools for validating data integrity throughout the trading system.
It ensures data follows expected schemas, meets quality standards, and maintains consistency.

Key features:
- Schema validation with flexible enforcement levels
- Data quality scoring and anomaly detection
- Integrity checks with customizable rules
- Logging and alerting for data issues
- History tracking of data quality metrics

Dependencies:
- core.event_bus for publishing data integrity events
- core.state_manager for tracking integrity metrics
- utils.logger for logging validation issues
- utils.serializers for schema serialization/deserialization
- utils.validators for basic validation utilities
"""

import dataclasses
import datetime
import enum
import json
import threading
import time
import uuid
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, validator

from core.event_bus import Event, EventPriority, EventTopics, create_event, get_event_bus
from core.state_manager import StateManager, StateScope
from utils.logger import get_logger


class IntegrityLevel(enum.Enum):
    """Levels of data integrity enforcement."""
    STRICT = "strict"  # Reject data that fails validation
    WARNING = "warning"  # Accept but log warnings for failed validation
    INFO = "info"  # Log information without affecting data flow
    AUDIT = "audit"  # Record for audit purposes only


class IntegrityStatus(enum.Enum):
    """Status of data integrity validation."""
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some checks passed, some failed
    UNKNOWN = "unknown"  # Status could not be determined


class DataCategory(enum.Enum):
    """Categories of data for appropriate schema and rule selection."""
    MARKET_DATA = "market_data"
    ORDER = "order"
    TRADE = "trade"
    POSITION = "position"
    FEATURE = "feature"
    SIGNAL = "signal"
    SYSTEM = "system"
    CONFIGURATION = "configuration"
    MODEL_OUTPUT = "model_output"
    ALTERNATIVE_DATA = "alternative_data"


@dataclasses.dataclass
class ValidationContext:
    """Context for validation operations."""
    timestamp: datetime.datetime = dataclasses.field(default_factory=datetime.datetime.utcnow)
    source: str = "unknown"
    category: DataCategory = DataCategory.SYSTEM
    level: IntegrityLevel = IntegrityLevel.STRICT
    reference_id: Optional[str] = None
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


class ValidationResult:
    """Results of a validation operation."""
    
    def __init__(self, 
                 context: ValidationContext,
                 status: IntegrityStatus = IntegrityStatus.UNKNOWN,
                 is_valid: bool = False):
        self.context = context
        self.status = status
        self.is_valid = is_valid
        self.timestamp = datetime.datetime.utcnow()
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.metrics: Dict[str, float] = {}
        
    def add_error(self, field: str, message: str, value: Any = None):
        """Add a validation error."""
        self.errors.append({
            "field": field,
            "message": message,
            "value": str(value) if value is not None else None
        })
        return self
    
    def add_warning(self, field: str, message: str, value: Any = None):
        """Add a validation warning."""
        self.warnings.append({
            "field": field,
            "message": message,
            "value": str(value) if value is not None else None
        })
        return self
    
    def add_metric(self, name: str, value: float):
        """Add a validation metric."""
        self.metrics[name] = value
        return self
    
    def update_status(self):
        """Update the validation status based on errors and warnings."""
        if len(self.errors) == 0:
            if len(self.warnings) == 0:
                self.status = IntegrityStatus.PASSED
                self.is_valid = True
            else:
                self.status = IntegrityStatus.PARTIAL
                self.is_valid = True  # Warnings don't invalidate data
        else:
            self.status = IntegrityStatus.FAILED
            self.is_valid = False
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "context": {
                "timestamp": self.context.timestamp.isoformat(),
                "source": self.context.source,
                "category": self.context.category.value,
                "level": self.context.level.value,
                "reference_id": self.context.reference_id,
                "metadata": self.context.metadata
            },
            "status": self.status.value,
            "is_valid": self.is_valid,
            "timestamp": self.timestamp.isoformat(),
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": self.metrics
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    def __bool__(self) -> bool:
        """Boolean representation of validation result."""
        return self.is_valid


class SchemaRegistry:
    """
    Registry for data schemas with versioning support.
    
    Allows storage and retrieval of schema definitions by category and version.
    """
    
    def __init__(self):
        self._schemas: Dict[Tuple[DataCategory, str], Any] = {}
        self._default_versions: Dict[DataCategory, str] = {}
        self._logger = get_logger("SchemaRegistry")
    
    def register_schema(self, category: DataCategory, schema: Any, version: str = "v1"):
        """Register a schema for a specific category and version."""
        key = (category, version)
        self._schemas[key] = schema
        if category not in self._default_versions:
            self._default_versions[category] = version
        self._logger.info(f"Registered schema for {category.value} version {version}")
        
    def set_default_version(self, category: DataCategory, version: str):
        """Set the default version for a category."""
        key = (category, version)
        if key not in self._schemas:
            raise ValueError(f"No schema registered for {category.value} version {version}")
        self._default_versions[category] = version
        
    def get_schema(self, category: DataCategory, version: Optional[str] = None) -> Any:
        """Get a schema by category and version."""
        if version is None:
            version = self._default_versions.get(category)
            if version is None:
                raise ValueError(f"No default version set for {category.value}")
        
        key = (category, version)
        if key not in self._schemas:
            raise ValueError(f"No schema registered for {category.value} version {version}")
        
        return self._schemas[key]
    
    def get_versions(self, category: DataCategory) -> List[str]:
        """Get all versions available for a category."""
        return [v for (c, v) in self._schemas.keys() if c == category]
    
    def get_default_version(self, category: DataCategory) -> Optional[str]:
        """Get the default version for a category."""
        return self._default_versions.get(category)


class ValidationRule:
    """Base class for validation rules."""
    
    def __init__(self, 
                 name: str, 
                 description: str = "", 
                 severity: IntegrityLevel = IntegrityLevel.WARNING):
        self.name = name
        self.description = description
        self.severity = severity
    
    def validate(self, data: Any, context: ValidationContext) -> ValidationResult:
        """Validate data against this rule."""
        result = ValidationResult(context)
        return result


class RuleRegistry:
    """Registry for validation rules."""
    
    def __init__(self):
        self._rules: Dict[DataCategory, List[ValidationRule]] = {}
        self._logger = get_logger("RuleRegistry")
    
    def register_rule(self, category: DataCategory, rule: ValidationRule):
        """Register a rule for a specific category."""
        if category not in self._rules:
            self._rules[category] = []
        self._rules[category].append(rule)
        self._logger.info(f"Registered rule '{rule.name}' for {category.value}")
    
    def get_rules(self, category: DataCategory) -> List[ValidationRule]:
        """Get all rules for a category."""
        return self._rules.get(category, [])


class DataIntegrityService:
    """
    Main service for data integrity validation.
    
    Combines schema validation and rule-based validation to ensure data quality.
    """
    
    def __init__(self, 
                 schema_registry: SchemaRegistry,
                 rule_registry: RuleRegistry,
                 state_manager: StateManager):
        self._schema_registry = schema_registry
        self._rule_registry = rule_registry
        self._state_manager = state_manager
        self._event_bus = get_event_bus()
        self._logger = get_logger("DataIntegrityService")
        self._lock = threading.RLock()
        
        # Initialize state for tracking metrics
        self._init_state()
    
    def _init_state(self):
        """Initialize state for tracking metrics in state manager."""
        state_path = "data.integrity.metrics"
        if not self._state_manager.get(state_path):
            # Create metrics storage for each category
            metrics = {}
            for category in DataCategory:
                metrics[category.value] = []
            
            self._state_manager.set(
                state_path, 
                metrics, 
                StateScope.PERSISTENT
            )
            
            # Set up historical metrics path
            history_path = "data.integrity.history"
            if not self._state_manager.get(history_path):
                self._state_manager.set(
                    history_path, 
                    {}, 
                    StateScope.HISTORICAL
                )
                
        self._logger.info("Initialized state for data integrity metrics")
    
    def validate_schema(self, 
                      data: Any, 
                      context: ValidationContext, 
                      schema_version: Optional[str] = None) -> ValidationResult:
        """Validate data against registered schema for its category."""
        result = ValidationResult(context)
        
        try:
            schema = self._schema_registry.get_schema(context.category, schema_version)
            
            # Handle different schema types (Pydantic, marshmallow, etc.)
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                # Pydantic validation
                try:
                    validated_data = schema.parse_obj(data)
                    result.is_valid = True
                    result.status = IntegrityStatus.PASSED
                except ValidationError as e:
                    for error in e.errors():
                        result.add_error(
                            '.'.join(str(loc) for loc in error['loc']), 
                            error['msg'],
                            error.get('ctx', {}).get('value')
                        )
            elif callable(schema):
                # Function-based validation
                try:
                    schema_result = schema(data)
                    if isinstance(schema_result, bool):
                        result.is_valid = schema_result
                        if not result.is_valid:
                            result.add_error("schema", "Failed schema validation")
                    elif isinstance(schema_result, tuple) and len(schema_result) == 2:
                        result.is_valid, errors = schema_result
                        if errors:
                            for field, msg in errors.items():
                                result.add_error(field, msg)
                except Exception as e:
                    result.add_error("schema", f"Schema validation error: {str(e)}")
            else:
                # Unsupported schema type
                result.add_error("schema", f"Unsupported schema type: {type(schema)}")
        
        except ValueError as e:
            result.add_error("schema", str(e))
        
        result.update_status()
        return result
    
    def validate_rules(self, 
                     data: Any, 
                     context: ValidationContext) -> ValidationResult:
        """Validate data against rules for its category."""
        result = ValidationResult(context)
        
        rules = self._rule_registry.get_rules(context.category)
        if not rules:
            result.add_warning("rules", f"No rules defined for category {context.category.value}")
            result.update_status()
            return result
        
        error_count = 0
        warning_count = 0
        
        for rule in rules:
            rule_result = rule.validate(data, context)
            
            # Merge rule result into overall result
            result.errors.extend(rule_result.errors)
            result.warnings.extend(rule_result.warnings)
            
            error_count += len(rule_result.errors)
            warning_count += len(rule_result.warnings)
            
            # Merge metrics
            for metric_name, metric_value in rule_result.metrics.items():
                result.add_metric(f"{rule.name}.{metric_name}", metric_value)
        
        result.add_metric("rule_count", len(rules))
        result.add_metric("error_count", error_count)
        result.add_metric("warning_count", warning_count)
        
        result.update_status()
        return result
    
    def validate(self, 
               data: Any, 
               context: ValidationContext,
               schema_version: Optional[str] = None) -> ValidationResult:
        """
        Complete validation of data using both schema and rules.
        
        Args:
            data: The data to validate
            context: Validation context information
            schema_version: Optional specific schema version to use
            
        Returns:
            ValidationResult containing validation status and details
        """
        # Schema validation
        schema_result = self.validate_schema(data, context, schema_version)
        
        # Rule validation (only if schema validation passed or we're in non-strict mode)
        if schema_result.is_valid or context.level != IntegrityLevel.STRICT:
            rule_result = self.validate_rules(data, context)
            
            # Combine results
            combined_result = ValidationResult(context)
            combined_result.errors = schema_result.errors + rule_result.errors
            combined_result.warnings = schema_result.warnings + rule_result.warnings
            
            # Merge metrics
            for metrics_dict in [schema_result.metrics, rule_result.metrics]:
                for metric_name, metric_value in metrics_dict.items():
                    combined_result.add_metric(metric_name, metric_value)
            
            combined_result.update_status()
        else:
            # Schema validation failed in strict mode, don't proceed to rule validation
            combined_result = schema_result
        
        # Store metrics for this validation
        self._store_validation_metrics(context.category, combined_result)
        
        # Handle validation results based on integrity level
        if not combined_result.is_valid:
            if context.level == IntegrityLevel.STRICT:
                self._logger.error(
                    f"Data validation failed for {context.category.value}: " +
                    f"{len(combined_result.errors)} errors"
                )
                for error in combined_result.errors:
                    self._logger.error(f"Field '{error['field']}': {error['message']}")
                
                # Publish validation failure event
                self._publish_validation_event(combined_result, "validation_failed")
                
            elif context.level == IntegrityLevel.WARNING:
                self._logger.warning(
                    f"Data validation issues for {context.category.value}: " +
                    f"{len(combined_result.errors)} errors, {len(combined_result.warnings)} warnings"
                )
                for error in combined_result.errors:
                    self._logger.warning(f"Field '{error['field']}': {error['message']}")
                
                # Publish validation warning event
                self._publish_validation_event(combined_result, "validation_warning")
        else:
            # Validation passed
            self._logger.debug(
                f"Data validation passed for {context.category.value} " +
                f"with {len(combined_result.warnings)} warnings"
            )
            if combined_result.warnings:
                for warning in combined_result.warnings:
                    self._logger.debug(f"Field '{warning['field']}': {warning['message']}")
            
            # Publish validation success event
            self._publish_validation_event(combined_result, "validation_passed")
        
        return combined_result
    
    def _publish_validation_event(self, result: ValidationResult, event_type: str):
        """Publish a validation event to the event bus."""
        try:
            event = create_event(
                topic=f"data.integrity.{event_type}",
                data={"validation_result": result.to_dict()},
                source="data_integrity",
                priority=EventPriority.NORMAL
            )
            self._event_bus.publish(event)
        except Exception as e:
            self._logger.error(f"Failed to publish validation event: {str(e)}")
    
    def _store_validation_metrics(self, category: DataCategory, result: ValidationResult):
        """Store validation metrics for historical tracking."""
        with self._lock:
            metrics_entry = {
                "timestamp": result.timestamp.timestamp(),
                "error_count": len(result.errors),
                "warning_count": len(result.warnings),
                "is_valid": result.is_valid,
                **result.metrics
            }
            
            # Get current metrics
            state_path = "data.integrity.metrics"
            current_metrics = self._state_manager.get(state_path, {})
            
            # Update metrics
            category_metrics = current_metrics.get(category.value, [])
            category_metrics.append(metrics_entry)
            
            # Limit the size of stored metrics (keep last 1000 per category)
            if len(category_metrics) > 1000:
                category_metrics = category_metrics[-1000:]
            
            current_metrics[category.value] = category_metrics
            
            # Update state
            self._state_manager.set(state_path, current_metrics)
            
            # Also store in historical metrics if appropriate
            history_path = f"data.integrity.history.{category.value}"
            
            # We store a limited set of metrics in the historical record
            historical_entry = {
                "valid_rate": 1.0 if result.is_valid else 0.0,
                "error_count": len(result.errors),
                "warning_count": len(result.warnings)
            }
            
            # Use state manager's historical storage
            self._state_manager.set(
                history_path,
                historical_entry,
                StateScope.HISTORICAL
            )
    
    def get_metrics_history(self, 
                          category: DataCategory, 
                          start_time: Optional[datetime.datetime] = None,
                          end_time: Optional[datetime.datetime] = None) -> List[Dict[str, Any]]:
        """Get historical validation metrics for a category."""
        with self._lock:
            state_path = "data.integrity.metrics"
            current_metrics = self._state_manager.get(state_path, {})
            
            metrics = current_metrics.get(category.value, [])
            
            if start_time:
                start_ts = start_time.timestamp()
                metrics = [m for m in metrics if m["timestamp"] >= start_ts]
            
            if end_time:
                end_ts = end_time.timestamp()
                metrics = [m for m in metrics if m["timestamp"] <= end_ts]
                
            return metrics
    
    def get_metrics_summary(self, 
                          category: DataCategory,
                          window: Optional[datetime.timedelta] = None) -> Dict[str, Any]:
        """Get summary statistics for validation metrics."""
        now = datetime.datetime.utcnow()
        if window:
            start_time = now - window
        else:
            start_time = now - datetime.timedelta(hours=24)  # Default to last 24 hours
        
        metrics = self.get_metrics_history(category, start_time, now)
        
        if not metrics:
            return {
                "category": category.value,
                "start_time": start_time.isoformat(),
                "end_time": now.isoformat(),
                "count": 0,
                "valid_rate": None,
                "error_rate": None,
                "validation_count": 0
            }
        
        valid_count = sum(1 for m in metrics if m["is_valid"])
        total_errors = sum(m["error_count"] for m in metrics)
        total_warnings = sum(m["warning_count"] for m in metrics)
        
        return {
            "category": category.value,
            "start_time": start_time.isoformat(),
            "end_time": now.isoformat(),
            "count": len(metrics),
            "valid_rate": valid_count / len(metrics) if metrics else None,
            "error_rate": total_errors / len(metrics) if metrics else None,
            "warning_rate": total_warnings / len(metrics) if metrics else None,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "validation_count": len(metrics)
        }):
        self._schema_registry = schema_registry
        self._rule_registry = rule_registry
        self._state_manager = state_manager
        self._event_bus = event_bus_instance or event_bus
        self._logger = logger.get_logger("DataIntegrityService")
        self._validation_metrics: Dict[DataCategory, List[Dict[str, Any]]] = {}
        
        # Initialize metrics for each category
        for category in DataCategory:
            self._validation_metrics[category] = []
    
    def validate_schema(self, 
                      data: Any, 
                      context: ValidationContext, 
                      schema_version: Optional[str] = None) -> ValidationResult:
        """Validate data against registered schema for its category."""
        result = ValidationResult(context)
        
        try:
            schema = self._schema_registry.get_schema(context.category, schema_version)
            
            # Handle different schema types (Pydantic, marshmallow, etc.)
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                # Pydantic validation
                try:
                    validated_data = schema.parse_obj(data)
                    result.is_valid = True
                    result.status = IntegrityStatus.PASSED
                except ValidationError as e:
                    for error in e.errors():
                        result.add_error(
                            '.'.join(str(loc) for loc in error['loc']), 
                            error['msg'],
                            error.get('ctx', {}).get('value')
                        )
            elif callable(schema):
                # Function-based validation
                try:
                    schema_result = schema(data)
                    if isinstance(schema_result, bool):
                        result.is_valid = schema_result
                        if not result.is_valid:
                            result.add_error("schema", "Failed schema validation")
                    elif isinstance(schema_result, tuple) and len(schema_result) == 2:
                        result.is_valid, errors = schema_result
                        if errors:
                            for field, msg in errors.items():
                                result.add_error(field, msg)
                except Exception as e:
                    result.add_error("schema", f"Schema validation error: {str(e)}")
            else:
                # Unsupported schema type
                result.add_error("schema", f"Unsupported schema type: {type(schema)}")
        
        except ValueError as e:
            result.add_error("schema", str(e))
        
        result.update_status()
        return result
    
    def validate_rules(self, 
                     data: Any, 
                     context: ValidationContext) -> ValidationResult:
        """Validate data against rules for its category."""
        result = ValidationResult(context)
        
        rules = self._rule_registry.get_rules(context.category)
        if not rules:
            result.add_warning("rules", f"No rules defined for category {context.category.value}")
            result.update_status()
            return result
        
        error_count = 0
        warning_count = 0
        
        for rule in rules:
            rule_result = rule.validate(data, context)
            
            # Merge rule result into overall result
            result.errors.extend(rule_result.errors)
            result.warnings.extend(rule_result.warnings)
            
            error_count += len(rule_result.errors)
            warning_count += len(rule_result.warnings)
            
            # Merge metrics
            for metric_name, metric_value in rule_result.metrics.items():
                result.add_metric(f"{rule.name}.{metric_name}", metric_value)
        
        result.add_metric("rule_count", len(rules))
        result.add_metric("error_count", error_count)
        result.add_metric("warning_count", warning_count)
        
        result.update_status()
        return result
    
    def validate(self, 
               data: Any, 
               context: ValidationContext,
               schema_version: Optional[str] = None) -> ValidationResult:
        """
        Complete validation of data using both schema and rules.
        
        Args:
            data: The data to validate
            context: Validation context information
            schema_version: Optional specific schema version to use
            
        Returns:
            ValidationResult containing validation status and details
        """
        # Schema validation
        schema_result = self.validate_schema(data, context, schema_version)
        
        # Rule validation (only if schema validation passed or we're in non-strict mode)
        if schema_result.is_valid or context.level != IntegrityLevel.STRICT:
            rule_result = self.validate_rules(data, context)
            
            # Combine results
            combined_result = ValidationResult(context)
            combined_result.errors = schema_result.errors + rule_result.errors
            combined_result.warnings = schema_result.warnings + rule_result.warnings
            
            # Merge metrics
            for metrics_dict in [schema_result.metrics, rule_result.metrics]:
                for metric_name, metric_value in metrics_dict.items():
                    combined_result.add_metric(metric_name, metric_value)
            
            combined_result.update_status()
        else:
            # Schema validation failed in strict mode, don't proceed to rule validation
            combined_result = schema_result
        
        # Store metrics for this validation
        self._store_validation_metrics(context.category, combined_result)
        
        # Handle validation results based on integrity level
        if not combined_result.is_valid:
            if context.level == IntegrityLevel.STRICT:
                self._logger.error(
                    f"Data validation failed for {context.category.value}: " +
                    f"{len(combined_result.errors)} errors"
                )
                for error in combined_result.errors:
                    self._logger.error(f"Field '{error['field']}': {error['message']}")
                
                # Publish validation failure event
                self._publish_validation_event(combined_result, "validation_failed")
                
            elif context.level == IntegrityLevel.WARNING:
                self._logger.warning(
                    f"Data validation issues for {context.category.value}: " +
                    f"{len(combined_result.errors)} errors, {len(combined_result.warnings)} warnings"
                )
                for error in combined_result.errors:
                    self._logger.warning(f"Field '{error['field']}': {error['message']}")
                
                # Publish validation warning event
                self._publish_validation_event(combined_result, "validation_warning")
        else:
            # Validation passed
            self._logger.debug(
                f"Data validation passed for {context.category.value} " +
                f"with {len(combined_result.warnings)} warnings"
            )
            if combined_result.warnings:
                for warning in combined_result.warnings:
                    self._logger.debug(f"Field '{warning['field']}': {warning['message']}")
            
            # Publish validation success event
            self._publish_validation_event(combined_result, "validation_passed")
        
        return combined_result
    
    def _publish_validation_event(self, result: ValidationResult, event_type: str):
        """Publish a validation event to the event bus."""
        try:
            self._event_bus.publish(
                event_type=event_type,
                data={
                    "validation_result": result.to_dict()
                },
                source="data_integrity",
                priority=1  # Medium priority
            )
        except Exception as e:
            self._logger.error(f"Failed to publish validation event: {str(e)}")
    
    def _store_validation_metrics(self, category: DataCategory, result: ValidationResult):
        """Store validation metrics for historical tracking."""
        metrics_entry = {
            "timestamp": result.timestamp,
            "error_count": len(result.errors),
            "warning_count": len(result.warnings),
            "is_valid": result.is_valid,
            **result.metrics
        }
        
        self._validation_metrics[category].append(metrics_entry)
        
        # Limit the size of stored metrics (keep last 1000 per category)
        if len(self._validation_metrics[category]) > 1000:
            self._validation_metrics[category] = self._validation_metrics[category][-1000:]
    
    def get_metrics_history(self, 
                          category: DataCategory, 
                          start_time: Optional[datetime.datetime] = None,
                          end_time: Optional[datetime.datetime] = None) -> List[Dict[str, Any]]:
        """Get historical validation metrics for a category."""
        metrics = self._validation_metrics.get(category, [])
        
        if start_time:
            metrics = [m for m in metrics if m["timestamp"] >= start_time]
        
        if end_time:
            metrics = [m for m in metrics if m["timestamp"] <= end_time]
            
        return metrics
    
    def get_metrics_summary(self, 
                          category: DataCategory,
                          window: Optional[datetime.timedelta] = None) -> Dict[str, Any]:
        """Get summary statistics for validation metrics."""
        now = datetime.datetime.utcnow()
        if window:
            start_time = now - window
        else:
            start_time = now - datetime.timedelta(hours=24)  # Default to last 24 hours
        
        metrics = self.get_metrics_history(category, start_time, now)
        
        if not metrics:
            return {
                "category": category.value,
                "start_time": start_time.isoformat(),
                "end_time": now.isoformat(),
                "count": 0,
                "valid_rate": None,
                "error_rate": None,
                "validation_count": 0
            }
        
        valid_count = sum(1 for m in metrics if m["is_valid"])
        total_errors = sum(m["error_count"] for m in metrics)
        total_warnings = sum(m["warning_count"] for m in metrics)
        
        return {
            "category": category.value,
            "start_time": start_time.isoformat(),
            "end_time": now.isoformat(),
            "count": len(metrics),
            "valid_rate": valid_count / len(metrics) if metrics else None,
            "error_rate": total_errors / len(metrics) if metrics else None,
            "warning_rate": total_warnings / len(metrics) if metrics else None,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "validation_count": len(metrics)
        }


# Common validation rules
class NumericRangeRule(ValidationRule):
    """Rule to validate a numeric field is within a specified range."""
    
    def __init__(self, 
                 field_name: str,
                 min_value: Optional[float] = None,
                 max_value: Optional[float] = None,
                 name: Optional[str] = None,
                 description: str = "",
                 severity: IntegrityLevel = IntegrityLevel.WARNING):
        name = name or f"{field_name}_range_check"
        super().__init__(name, description, severity)
        self.field_name = field_name
        self.min_value = min_value
        self.max_value = max_value
    
    def validate(self, data: Any, context: ValidationContext) -> ValidationResult:
        result = ValidationResult(context)
        
        # Extract field value using dot notation
        field_value = data
        for part in self.field_name.split('.'):
            if isinstance(field_value, dict) and part in field_value:
                field_value = field_value[part]
            elif hasattr(field_value, part):
                field_value = getattr(field_value, part)
            else:
                result.add_error(
                    self.field_name, 
                    f"Field '{part}' not found in data"
                )
                result.update_status()
                return result
        
        # Check if value is numeric
        if not isinstance(field_value, (int, float, np.number)):
            result.add_error(
                self.field_name,
                f"Value must be numeric, got {type(field_value).__name__}"
            )
            result.update_status()
            return result
        
        # Check range
        if self.min_value is not None and field_value < self.min_value:
            result.add_error(
                self.field_name,
                f"Value {field_value} is below minimum {self.min_value}",
                field_value
            )
        
        if self.max_value is not None and field_value > self.max_value:
            result.add_error(
                self.field_name,
                f"Value {field_value} is above maximum {self.max_value}",
                field_value
            )
        
        result.update_status()
        return result


class DataFrameIntegrityRule(ValidationRule):
    """Base rule for validating pandas DataFrames."""
    
    def __init__(self,
                 name: str,
                 required_columns: Optional[List[str]] = None,
                 min_rows: int = 0,
                 description: str = "",
                 severity: IntegrityLevel = IntegrityLevel.WARNING):
        super().__init__(name, description, severity)
        self.required_columns = required_columns or []
        self.min_rows = min_rows
    
    def validate(self, data: Any, context: ValidationContext) -> ValidationResult:
        result = ValidationResult(context)
        
        # Check if data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            result.add_error("data", f"Expected pandas DataFrame, got {type(data).__name__}")
            result.update_status()
            return result
        
        # Check for required columns
        if self.required_columns:
            missing_columns = [col for col in self.required_columns if col not in data.columns]
            if missing_columns:
                result.add_error(
                    "columns", 
                    f"Missing required columns: {', '.join(missing_columns)}"
                )
        
        # Check for minimum rows
        if len(data) < self.min_rows:
            result.add_error(
                "rows",
                f"DataFrame has {len(data)} rows, minimum required is {self.min_rows}"
            )
        
        # Add metrics
        result.add_metric("row_count", len(data))
        result.add_metric("column_count", len(data.columns))
        result.add_metric("missing_value_percentage", data.isna().mean().mean() * 100)
        
        result.update_status()
        return result


class DataFrameNullCheckRule(DataFrameIntegrityRule):
    """Rule to check for null values in specific columns of a DataFrame."""
    
    def __init__(self,
                 columns_to_check: Optional[List[str]] = None,
                 max_null_percentage: float = 0.0,
                 name: str = "dataframe_null_check",
                 description: str = "Checks for null values in specific DataFrame columns",
                 severity: IntegrityLevel = IntegrityLevel.WARNING):
        super().__init__(name, [], 0, description, severity)
        self.columns_to_check = columns_to_check
        self.max_null_percentage = max_null_percentage
    
    def validate(self, data: Any, context: ValidationContext) -> ValidationResult:
        result = super().validate(data, context)
        
        if not result.is_valid:
            return result
        
        columns = self.columns_to_check or data.columns
        columns = [col for col in columns if col in data.columns]
        
        for column in columns:
            null_count = data[column].isna().sum()
            null_percentage = null_count / len(data) if len(data) > 0 else 0.0
            
            result.add_metric(f"null_percentage_{column}", null_percentage * 100)
            
            if null_percentage > self.max_null_percentage:
                result.add_error(
                    column,
                    f"Column has {null_percentage:.2%} null values, max allowed is {self.max_null_percentage:.2%}",
                    null_count
                )
        
        result.update_status()
        return result


class MarketDataIntegrityRule(ValidationRule):
    """
    Specialized rule for market data integrity checking.
    
    Validates common issues in market data like price gaps, 
    timestamp irregularities, and volume anomalies.
    """
    
    def __init__(self,
                 price_column: str = "price",
                 timestamp_column: str = "timestamp",
                 volume_column: Optional[str] = "volume",
                 max_price_gap_percentage: float = 5.0,
                 check_increasing_timestamps: bool = True,
                 name: str = "market_data_integrity",
                 description: str = "Checks market data for common issues",
                 severity: IntegrityLevel = IntegrityLevel.WARNING):
        super().__init__(name, description, severity)
        self.price_column = price_column
        self.timestamp_column = timestamp_column
        self.volume_column = volume_column
        self.max_price_gap_percentage = max_price_gap_percentage
        self.check_increasing_timestamps = check_increasing_timestamps
    
    def validate(self, data: Any, context: ValidationContext) -> ValidationResult:
        result = ValidationResult(context)
        
        # Check if data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            result.add_error("data", f"Expected pandas DataFrame, got {type(data).__name__}")
            result.update_status()
            return result
        
        # Check required columns exist
        required_columns = [self.price_column, self.timestamp_column]
        if self.volume_column:
            required_columns.append(self.volume_column)
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            result.add_error(
                "columns",
                f"Missing required columns: {', '.join(missing_columns)}"
            )
            result.update_status()
            return result
        
        # Ensure data is sorted by timestamp
        if self.check_increasing_timestamps:
            data = data.sort_values(by=self.timestamp_column)
        
        # Check for decreasing timestamps
        if self.check_increasing_timestamps:
            if not data[self.timestamp_column].is_monotonic_increasing:
                result.add_error(
                    self.timestamp_column,
                    "Timestamps are not monotonically increasing"
                )
        
        # Check for duplicate timestamps
        duplicate_timestamps = data[self.timestamp_column].duplicated().sum()
        if duplicate_timestamps > 0:
            result.add_warning(
                self.timestamp_column,
                f"Found {duplicate_timestamps} duplicate timestamps"
            )
            result.add_metric("duplicate_timestamp_count", duplicate_timestamps)
        
        # Check for price gaps
        if len(data) > 1:
            data['price_change_pct'] = data[self.price_column].pct_change().abs() * 100
            large_gaps = data[data['price_change_pct'] > self.max_price_gap_percentage]
            
            if not large_gaps.empty:
                result.add_warning(
                    self.price_column,
                    f"Found {len(large_gaps)} price gaps larger than {self.max_price_gap_percentage}%"
                )
                result.add_metric("large_price_gap_count", len(large_gaps))
                result.add_metric("max_price_gap_percentage", data['price_change_pct'].max())
        
        # Check for volume anomalies if volume column specified
        if self.volume_column and self.volume_column in data.columns:
            # Check for negative volumes
            negative_volumes = (data[self.volume_column] < 0).sum()
            if negative_volumes > 0:
                result.add_error(
                    self.volume_column,
                    f"Found {negative_volumes} negative volume values"
                )
                result.add_metric("negative_volume_count", negative_volumes)
            
            # Check for zero volumes (might be valid but worth flagging)
            zero_volumes = (data[self.volume_column] == 0).sum()
            if zero_volumes > 0:
                result.add_warning(
                    self.volume_column,
                    f"Found {zero_volumes} zero volume values"
                )
                result.add_metric("zero_volume_count", zero_volumes)
            
            # Check for volume spikes (3 std dev above mean)
            volume_mean = data[self.volume_column].mean()
            volume_std = data[self.volume_column].std()
            volume_threshold = volume_mean + (3 * volume_std)
            volume_spikes = (data[self.volume_column] > volume_threshold).sum()
            
            if volume_spikes > 0:
                result.add_warning(
                    self.volume_column,
                    f"Found {volume_spikes} volume spikes (> 3 std dev)"
                )
                result.add_metric("volume_spike_count", volume_spikes)
        
        result.update_status()
        return result


# Schema adapters for different types of schemas
class PydanticSchemaAdapter:
    """Adapter for Pydantic schemas to use with the schema registry."""
    
    @staticmethod
    def create_schema_class(fields: Dict[str, Any], 
                          class_name: str = "DynamicSchema",
                          base_class: type = BaseModel) -> type:
        """
        Create a Pydantic model class dynamically from field definitions.
        
        Args:
            fields: Dictionary mapping field names to field types/definitions
            class_name: Name for the generated class
            base_class: Base class to inherit from (default: Pydantic BaseModel)
            
        Returns:
            A new Pydantic model class
        """
        namespace = {
            '__annotations__': {name: field_type for name, (field_type, _) in fields.items()},
            **{name: Field(**field_kwargs) for name, (_, field_kwargs) in fields.items()}
        }
        
        return type(class_name, (base_class,), namespace)
    
    @staticmethod
    def validate(schema: type, data: Any) -> Tuple[bool, Optional[Dict[str, str]]]:
        """
        Validate data against a Pydantic schema.
        
        Args:
            schema: Pydantic model class
            data: Data to validate
            
        Returns:
            Tuple of (is_valid, errors)
        """
        try:
            schema.parse_obj(data)
            return True, None
        except ValidationError as e:
            errors = {}
            for error in e.errors():
                field_path = '.'.join(str(loc) for loc in error['loc'])
                errors[field_path] = error['msg']
            return False, errors


# Factory to create default validation rules for different data categories
class ValidationRuleFactory:
    """Factory for creating standard validation rules for different data types."""
    
    @staticmethod
    def create_market_data_rules() -> List[ValidationRule]:
        """Create standard rules for market data validation."""
        return [
            MarketDataIntegrityRule(
                price_column="price",
                timestamp_column="timestamp",
                volume_column="volume",
                max_price_gap_percentage=5.0
            ),
            TimeSeriesIntegrityRule(
                timestamp_column="timestamp",
                expected_frequency="1min"
            ),
            DataFrameNullCheckRule(
                columns_to_check=["price", "timestamp"],
                max_null_percentage=0.0,
                severity=IntegrityLevel.STRICT
            )
        ]
    
    @staticmethod
    def create_feature_rules() -> List[ValidationRule]:
        """Create standard rules for feature data validation."""
        return [
            DataFrameNullCheckRule(
                max_null_percentage=0.01,
                severity=IntegrityLevel.WARNING
            ),
            DataFrameOutlierRule(
                std_dev_threshold=4.0
            )
        ]
    
    @staticmethod
    def create_signal_rules() -> List[ValidationRule]:
        """Create standard rules for signal data validation."""
        return [
            NumericRangeRule(
                field_name="confidence",
                min_value=0.0,
                max_value=1.0,
                severity=IntegrityLevel.STRICT
            ),
            NumericRangeRule(
                field_name="signal_value",
                min_value=-1.0,
                max_value=1.0,
                severity=IntegrityLevel.STRICT
            )
        ])
            result.update_status()
            return result
        
        # Check range
        if self.min_value is not None and field_value < self.min_value:
            result.add_error(
                self.field_name,
                f"Value {field_value} is below minimum {self.min_value}",
                field_value
            )
        
        if self.max_value is not None and field_value > self.max_value:
            result.add_error(
                self.field_name,
                f"Value {field_value} is above maximum {self.max_value}",
                field_value
            )
        
        result.update_status()
        return result


class DataFrameIntegrityRule(ValidationRule):
    """Base rule for validating pandas DataFrames."""
    
    def __init__(self,
                 name: str,
                 required_columns: Optional[List[str]] = None,
                 min_rows: int = 0,
                 description: str = "",
                 severity: IntegrityLevel = IntegrityLevel.WARNING):
        super().__init__(name, description, severity)
        self.required_columns = required_columns or []
        self.min_rows = min_rows
    
    def validate(self, data: Any, context: ValidationContext) -> ValidationResult:
        result = ValidationResult(context)
        
        # Check if data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            result.add_error("data", f"Expected pandas DataFrame, got {type(data).__name__}")
            result.update_status()
            return result
        
        # Check for required columns
        if self.required_columns:
            missing_columns = [col for col in self.required_columns if col not in data.columns]
            if missing_columns:
                result.add_error(
                    "columns", 
                    f"Missing required columns: {', '.join(missing_columns)}"
                )
        
        # Check for minimum rows
        if len(data) < self.min_rows:
            result.add_error(
                "rows",
                f"DataFrame has {len(data)} rows, minimum required is {self.min_rows}"
            )
        
        # Add metrics
        result.add_metric("row_count", len(data))
        result.add_metric("column_count", len(data.columns))
        result.add_metric("missing_value_percentage", data.isna().mean().mean() * 100)
        
        result.update_status()
        return result


class DataFrameNullCheckRule(DataFrameIntegrityRule):
    """Rule to check for null values in specific columns of a DataFrame."""
    
    def __init__(self,
                 columns_to_check: Optional[List[str]] = None,
                 max_null_percentage: float = 0.0,
                 name: str = "dataframe_null_check",
                 description: str = "Checks for null values in specific DataFrame columns",
                 severity: IntegrityLevel = IntegrityLevel.WARNING):
        super().__init__(name, [], 0, description, severity)
        self.columns_to_check = columns_to_check
        self.max_null_percentage = max_null_percentage
    
    def validate(self, data: Any, context: ValidationContext) -> ValidationResult:
        result = super().validate(data, context)
        
        if not result.is_valid:
            return result
        
        columns = self.columns_to_check or data.columns
        columns = [col for col in columns if col in data.columns]
        
        for column in columns:
            null_count = data[column].isna().sum()
            null_percentage = null_count / len(data) if len(data) > 0 else 0.0
            
            result.add_metric(f"null_percentage_{column}", null_percentage * 100)
            
            if null_percentage > self.max_null_percentage:
                result.add_error(
                    column,
                    f"Column has {null_percentage:.2%} null values, max allowed is {self.max_null_percentage:.2%}",
                    null_count
                )
        
        result.update_status()
        return result


class DataFrameOutlierRule(DataFrameIntegrityRule):
    """Rule to check for outliers in numeric columns of a DataFrame."""
    
    def __init__(self,
                 columns_to_check: Optional[List[str]] = None,
                 std_dev_threshold: float = 3.0,
                 name: str = "dataframe_outlier_check",
                 description: str = "Checks for outliers in numeric DataFrame columns",
                 severity: IntegrityLevel = IntegrityLevel.WARNING):
        super().__init__(name, [], 0, description, severity)
        self.columns_to_check = columns_to_check
        self.std_dev_threshold = std_dev_threshold
    
    def validate(self, data: Any, context: ValidationContext) -> ValidationResult:
        result = super().validate(data, context)
        
        if not result.is_valid:
            return result
        
        # Identify numeric columns
        numeric_columns = data.select_dtypes(include=['number']).columns
        
        # Filter to specified columns if provided
        if self.columns_to_check:
            columns = [col for col in self.columns_to_check if col in numeric_columns]
        else:
            columns = numeric_columns
        
        for column in columns:
            # Calculate outliers using standard deviation
            mean = data[column].mean()
            std = data[column].std()
            threshold = self.std_dev_threshold * std
            
            outliers = data[(data[column] < mean - threshold) | (data[column] > mean + threshold)]
            outlier_count = len(outliers)
            outlier_percentage = outlier_count / len(data) if len(data) > 0 else 0.0
            
            result.add_metric(f"outlier_percentage_{column}", outlier_percentage * 100)
            result.add_metric(f"outlier_count_{column}", outlier_count)
            
            if outlier_count > 0:
                result.add_warning(
                    column,
                    f"Column has {outlier_count} outliers ({outlier_percentage:.2%} of data)"
                )
        
        result.update_status()
        return result


class MarketDataIntegrityRule(ValidationRule):
    """
    Specialized rule for market data integrity checking.
    
    Validates common issues in market data like price gaps, 
    timestamp irregularities, and volume anomalies.
    """
    
    def __init__(self,
                 price_column: str = "price",
                 timestamp_column: str = "timestamp",
                 volume_column: Optional[str] = "volume",
                 max_price_gap_percentage: float = 5.0,
                 check_increasing_timestamps: bool = True,
                 name: str = "market_data_integrity",
                 description: str = "Checks market data for common issues",
                 severity: IntegrityLevel = IntegrityLevel.WARNING):
        super().__init__(name, description, severity)
        self.price_column = price_column
        self.timestamp_column = timestamp_column
        self.volume_column = volume_column
        self.max_price_gap_percentage = max_price_gap_percentage
        self.check_increasing_timestamps = check_increasing_timestamps
    
    def validate(self, data: Any, context: ValidationContext) -> ValidationResult:
        result = ValidationResult(context)
        
        # Check if data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            result.add_error("data", f"Expected pandas DataFrame, got {type(data).__name__}")
            result.update_status()
            return result
        
        # Check required columns exist
        required_columns = [self.price_column, self.timestamp_column]
        if self.volume_column:
            required_columns.append(self.volume_column)
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            result.add_error(
                "columns",
                f"Missing required columns: {', '.join(missing_columns)}"
            )
            result.update_status()
            return result
        
        # Ensure data is sorted by timestamp
        if self.check_increasing_timestamps:
            data = data.sort_values(by=self.timestamp_column)
        
        # Check for decreasing timestamps
        if self.check_increasing_timestamps:
            if not data[self.timestamp_column].is_monotonic_increasing:
                result.add_error(
                    self.timestamp_column,
                    "Timestamps are not monotonically increasing"
                )
        
        # Check for duplicate timestamps
        duplicate_timestamps = data[self.timestamp_column].duplicated().sum()
        if duplicate_timestamps > 0:
            result.add_warning(
                self.timestamp_column,
                f"Found {duplicate_timestamps} duplicate timestamps"
            )
            result.add_metric("duplicate_timestamp_count", duplicate_timestamps)
        
        # Check for price gaps
        if len(data) > 1:
            data['price_change_pct'] = data[self.price_column].pct_change().abs() * 100
            large_gaps = data[data['price_change_pct'] > self.max_price_gap_percentage]
            
            if not large_gaps.empty:
                result.add_warning(
                    self.price_column,
                    f"Found {len(large_gaps)} price gaps larger than {self.max_price_gap_percentage}%"
                )
                result.add_metric("large_price_gap_count", len(large_gaps))
                result.add_metric("max_price_gap_percentage", data['price_change_pct'].max())
        
        # Check for volume anomalies if volume column specified
        if self.volume_column and self.volume_column in data.columns:
            # Check for negative volumes
            negative_volumes = (data[self.volume_column] < 0).sum()
            if negative_volumes > 0:
                result.add_error(
                    self.volume_column,
                    f"Found {negative_volumes} negative volume values"
                )
                result.add_metric("negative_volume_count", negative_volumes)
            
            # Check for zero volumes (might be valid but worth flagging)
            zero_volumes = (data[self.volume_column] == 0).sum()
            if zero_volumes > 0:
                result.add_warning(
                    self.volume_column,
                    f"Found {zero_volumes} zero volume values"
                )
                result.add_metric("zero_volume_count", zero_volumes)
            
            # Check for volume spikes (3 std dev above mean)
            volume_mean = data[self.volume_column].mean()
            volume_std = data[self.volume_column].std()
            volume_threshold = volume_mean + (3 * volume_std)
            volume_spikes = (data[self.volume_column] > volume_threshold).sum()
            
            if volume_spikes > 0:
                result.add_warning(
                    self.volume_column,
                    f"Found {volume_spikes} volume spikes (> 3 std dev)"
                )
                result.add_metric("volume_spike_count", volume_spikes)
        
        result.update_status()
        return result


class TimeSeriesIntegrityRule(ValidationRule):
    """
    Rule to validate time series data integrity.
    
    Checks for common issues in time series data like gaps, frequency 
    consistency, and seasonality disruptions.
    """
    
    def __init__(self,
                 timestamp_column: str = "timestamp",
                 expected_frequency: Optional[str] = None,  # pandas frequency string like 'D', 'H', '1min'
                 allow_gaps: bool = False,
                 max_gap_tolerance: Optional[pd.Timedelta] = None,
                 name: str = "time_series_integrity",
                 description: str = "Checks time series data for integrity issues",
                 severity: IntegrityLevel = IntegrityLevel.WARNING):
        super().__init__(name, description, severity)
        self.timestamp_column = timestamp_column
        self.expected_frequency = expected_frequency
        self.allow_gaps = allow_gaps
        self.max_gap_tolerance = max_gap_tolerance
    
    def validate(self, data: Any, context: ValidationContext) -> ValidationResult:
        result = ValidationResult(context)
        
        # Check if data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            result.add_error("data", f"Expected pandas DataFrame, got {type(data).__name__}")
            result.update_status()
            return result
        
        # Check if timestamp column exists
        if self.timestamp_column not in data.columns:
            result.add_error(
                "columns",
                f"Missing required timestamp column: {self.timestamp_column}"
            )
            result.update_status()
            return result
        
        # Ensure timestamp column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(data[self.timestamp_column]):
            try:
                data[self.timestamp_column] = pd.to_datetime(data[self.timestamp_column])
            except Exception as e:
                result.add_error(
                    self.timestamp_column,
                    f"Failed to convert to datetime: {str(e)}"
                )
                result.update_status()
                return result
        
        # Sort by timestamp for analysis
        data = data.sort_values(by=self.timestamp_column)
        
        # Check for timestamp uniqueness
        duplicate_count = data[self.timestamp_column].duplicated().sum()
        if duplicate_count > 0:
            result.add_warning(
                self.timestamp_column,
                f"Found {duplicate_count} duplicate timestamps"
            )
            result.add_metric("duplicate_timestamp_count", duplicate_count)
        
        # Skip empty or single-row datasets
        if len(data) <= 1:
            result.add_warning("data", "Dataset has too few rows for time series validation")
            result.update_status()
            return result
        
        # Calculate time differences between consecutive rows
        data['time_diff'] = data[self.timestamp_column].diff()
        
        # Check for expected frequency if specified
        if self.expected_frequency:
            expected_delta = pd.Timedelta(self.expected_frequency)
            
            # Tolerance for frequency validation (allow minor deviations)
            tolerance = pd.Timedelta(seconds=1)  # 1 second tolerance by default
            
            # Check if time differences match expected frequency
            freq_mismatches = data[
                (data['time_diff'] > expected_delta + tolerance) | 
                (data['time_diff'] < expected_delta - tolerance)
            ]
            
            mismatch_count = len(freq_mismatches)
            if mismatch_count > 0 and not self.allow_gaps:
                result.add_warning(
                    self.timestamp_column,
                    f"Found {mismatch_count} frequency mismatches, expected {self.expected_frequency}"
                )
                result.add_metric("frequency_mismatch_count", mismatch_count)
                
                # Calculate average and max frequency deviations
                deviations = abs(freq_mismatches['time_diff'] - expected_delta)
                result.add_metric("avg_frequency_deviation_seconds", deviations.mean().total_seconds())
                result.add_metric("max_frequency_deviation_seconds", deviations.max().total_seconds())
        
        # Check for gaps in time series
        if not self.allow_gaps:
            # Identify gaps based on max_gap_tolerance if specified
            gap_threshold = self.max_gap_tolerance or (expected_delta * 2 if self.expected_frequency else None)
            
            if gap_threshold:
                gaps = data[data['time_diff'] > gap_threshold]
                gap_count = len(gaps)
                
                if gap_count > 0:
                    result.add_warning(
                        self.timestamp_column,
                        f"Found {gap_count} gaps larger than {gap_threshold}"
                    )
                    result.add_metric("gap_count", gap_count)
                    
                    # Calculate total gap duration
                    total_gap_duration = gaps['time_diff'].sum() - (gap_threshold * gap_count)
                    result.add_metric("total_gap_duration_seconds", total_gap_duration.total_seconds())
                    result.add_metric("max_gap_duration_seconds", gaps['time_diff'].max().total_seconds())
        
        result.update_status()
        return result


# Schema adapters for different types of schemas
class PydanticSchemaAdapter:
    """Adapter for Pydantic schemas to use with the schema registry."""
    
    @staticmethod
    def create_schema_class(fields: Dict[str, Any], 
                          class_name: str = "DynamicSchema",
                          base_class: type = BaseModel) -> type:
        """
        Create a Pydantic model class dynamically from field definitions.
        
        Args:
            fields: Dictionary mapping field names to field types/definitions
            class_name: Name for the generated class
            base_class: Base class to inherit from (default: Pydantic BaseModel)
            
        Returns:
            A new Pydantic model class
        """
        namespace = {
            '__annotations__': {name: field_type for name, (field_type, _) in fields.items()},
            **{name: Field(**field_kwargs) for name, (_, field_kwargs) in fields.items()}
        }
        
        return type(class_name, (base_class,), namespace)
    
    @staticmethod
    def validate(schema: type, data: Any) -> Tuple[bool, Optional[Dict[str, str]]]:
        """
        Validate data against a Pydantic schema.
        
        Args:
            schema: Pydantic model class
            data: Data to validate
            
        Returns:
            Tuple of (is_valid, errors)
        """
        try:
            schema.parse_obj(data)
            return True, None
        except ValidationError as e:
            errors = {}
            for error in e.errors():
                field_path = '.'.join(str(loc) for loc in error['loc'])
                errors[field_path] = error['msg']
            return False, errors


# Factory to create default validation rules for different data categories
class ValidationRuleFactory:
    """Factory for creating standard validation rules for different data types."""
    
    @staticmethod
    def create_market_data_rules() -> List[ValidationRule]:
        """Create standard rules for market data validation."""
        return [
            MarketDataIntegrityRule(
                price_column="price",
                timestamp_column="timestamp",
                volume_column="volume",
                max_price_gap_percentage=5.0
            ),
            TimeSeriesIntegrityRule(
                timestamp_column="timestamp",
                expected_frequency="1min"
            ),
            DataFrameNullCheckRule(
                columns_to_check=["price", "timestamp"],
                max_null_percentage=0.0,
                severity=IntegrityLevel.STRICT
            )
        ]
    
    @staticmethod
    def create_feature_rules() -> List[ValidationRule]:
        """Create standard rules for feature data validation."""
        return [
            DataFrameNullCheckRule(
                max_null_percentage=0.01,
                severity=IntegrityLevel.WARNING
            ),
            DataFrameOutlierRule(
                std_dev_threshold=4.0
            )
        ]
    
    @staticmethod
    def create_signal_rules() -> List[ValidationRule]:
        """Create standard rules for signal data validation."""
        return [
            NumericRangeRule(
                field_name="confidence",
                min_value=0.0,
                max_value=1.0,
                severity=IntegrityLevel.STRICT
            ),
            NumericRangeRule(
                field_name="signal_value",
                min_value=-1.0,
                max_value=1.0,
                severity=IntegrityLevel.STRICT
            )
        ]


# Data quality score calculator
class DataQualityScorer:
    """
    Calculates data quality scores based on validation results.
    
    Provides a standardized way to quantify data quality across the system.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize with optional custom weights for different quality factors.
        
        Args:
            weights: Dictionary mapping quality factors to their weights
        """
        # Default weights for different quality factors
        self._weights = weights or {
            "completeness": 0.3,   # How complete the data is (lack of missing values)
            "accuracy": 0.3,       # How accurate the data appears to be (lack of errors)
            "consistency": 0.2,    # How consistent the data is (follows expected patterns)
            "timeliness": 0.1,     # How timely the data is (freshness)
            "validity": 0.1        # How valid the data is (conforms to schema)
        }
    
    def calculate_score(self, result: ValidationResult) -> float:
        """
        Calculate a quality score from a validation result.
        
        Args:
            result: ValidationResult from validation process
            
        Returns:
            Quality score between 0.0 (worst) and 1.0 (best)
        """
        # Base score starts at 1.0 (perfect)
        score = 1.0
        
        # Factor penalties
        factors = {
            "completeness": self._calculate_completeness_factor(result),
            "accuracy": self._calculate_accuracy_factor(result),
            "consistency": self._calculate_consistency_factor(result),
            "timeliness": self._calculate_timeliness_factor(result),
            "validity": 1.0 if result.is_valid else 0.0
        }
        
        # Calculate weighted score
        weighted_score = sum(
            factor * self._weights[name] 
            for name, factor in factors.items()
        )
        
        return max(0.0, min(1.0, weighted_score))
    
    def _calculate_completeness_factor(self, result: ValidationResult) -> float:
        """Calculate completeness factor based on null value metrics."""
        # Look for null percentage metrics in the result
        null_metrics = {
            k: v for k, v in result.metrics.items() 
            if k.startswith("null_percentage_")
        }
        
        if not null_metrics:
            return 1.0  # No null metrics, assume complete
        
        # Average null percentage
        avg_null_pct = sum(null_metrics.values()) / len(null_metrics)
        
        # Convert to completeness factor (0-100% null becomes 1.0-0.0 factor)
        return 1.0 - (avg_null_pct / 100.0)
    
    def _calculate_accuracy_factor(self, result: ValidationResult) -> float:
        """Calculate accuracy factor based on error metrics."""
        # Calculate based on error count
        error_count = len(result.errors)
        
        if error_count == 0:
            return 1.0
        
        # Exponential penalty for errors
        return max(0.0, np.exp(-0.5 * error_count))
    
    def _calculate_consistency_factor(self, result: ValidationResult) -> float:
        """Calculate consistency factor based on warning metrics."""
        # Look for outlier and anomaly metrics
        outlier_metrics = {
            k: v for k, v in result.metrics.items()
            if k.startswith(("outlier_", "gap_", "duplicate_", "frequency_mismatch_"))
        }
        
        if not outlier_metrics:
            return 1.0  # No consistency metrics, assume consistent
        
        # Count total consistency issues
        total_issues = sum(outlier_metrics.values())
        
        # Penalize based on number of issues
        return max(0.0, 1.0 - (0.1 * total_issues))
    
    def _calculate_timeliness_factor(self, result: ValidationResult) -> float:
        """Calculate timeliness factor based on data freshness."""
        # By default, if we have no specific timeliness info, assume it's current
        return 1.0


# Create a factory to set up the data integrity service with default schemas and rules
class DataIntegrityFactory:
    """
    Factory for creating and configuring the data integrity service.
    
    Provides standard configurations for different use cases and
    offers convenience methods for initializing the service.
    """
    
    @staticmethod
    def create_service(state_manager: StateManager, event_bus_instance=None) -> DataIntegrityService:
        """
        Create a data integrity service with standard configuration.
        
        Args:
            state_manager: System state manager
            event_bus_instance: Optional event bus instance
            
        Returns:
            Configured DataIntegrityService
        """
        schema_registry = SchemaRegistry()
        rule_registry = RuleRegistry()
        
        # Initialize with default schemas and rules
        DataIntegrityFactory._register_default_schemas(schema_registry)
        DataIntegrityFactory._register_default_rules(rule_registry)
        
        return DataIntegrityService(
            schema_registry=schema_registry,
            rule_registry=rule_registry,
            state_manager=state_manager,
            event_bus_instance=event_bus_instance
        )
    
    @staticmethod
    def _register_default_schemas(registry: SchemaRegistry):
        """Register default schemas for common data categories."""
        # Market data schema
        class MarketDataSchema(BaseModel):
            symbol: str
            timestamp: datetime.datetime
            price: float
            volume: Optional[float] = None
            open: Optional[float] = None
            high: Optional[float] = None
            low: Optional[float] = None
            close: Optional[float] = None
        
        # Order schema
        class OrderSchema(BaseModel):
            order_id: str
            symbol: str
            side: str  # buy or sell
            quantity: float
            price: Optional[float] = None  # None for market orders
            order_type: str  # market, limit, etc.
            timestamp: datetime.datetime
            status: str = "new"
        
        # Trade schema
        class TradeSchema(BaseModel):
            trade_id: str
            order_id: str
            symbol: str
            side: str
            quantity: float
            price: float
            timestamp: datetime.datetime
            fee: Optional[float] = None
        
        # Signal schema
        class SignalSchema(BaseModel):
            symbol: str
            timestamp: datetime.datetime
            signal_value: float  # -1.0 to 1.0
            confidence: float  # 0.0 to 1.0
            strategy_id: str
            metadata: Dict[str, Any] = Field(default_factory=dict)
        
        # Register schemas
        registry.register_schema(DataCategory.MARKET_DATA, MarketDataSchema)
        registry.register_schema(DataCategory.ORDER, OrderSchema)
        registry.register_schema(DataCategory.TRADE, TradeSchema)
        registry.register_schema(DataCategory.SIGNAL, SignalSchema)
    
    @staticmethod
    def _register_default_rules(registry: RuleRegistry):
        """Register default validation rules for common data categories."""
        # Market data rules
        for rule in ValidationRuleFactory.create_market_data_rules():
            registry.register_rule(DataCategory.MARKET_DATA, rule)
        
        # Feature rules
        for rule in ValidationRuleFactory.create_feature_rules():
            registry.register_rule(DataCategory.FEATURE, rule)
        
        # Signal rules
        for rule in ValidationRuleFactory.create_signal_rules():
            registry.register_rule(DataCategory.SIGNAL, rule)


# Function to create a validation context
def create_context(source: str, 
                category: DataCategory, 
                level: IntegrityLevel = IntegrityLevel.STRICT,
                reference_id: Optional[str] = None,
                metadata: Optional[Dict[str, Any]] = None) -> ValidationContext:
    """
    Create a validation context for data validation.
    
    Args:
        source: Source of the data
        category: Data category
        level: Integrity enforcement level
        reference_id: Optional reference ID for the data
        metadata: Optional additional metadata
        
    Returns:
        ValidationContext object
    """
    return ValidationContext(
        timestamp=datetime.datetime.utcnow(),
        source=source,
        category=category,
        level=level,
        reference_id=reference_id,
        metadata=metadata or {}
    )
