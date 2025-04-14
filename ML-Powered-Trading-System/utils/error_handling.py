"""
error_handling.py - Error handling utilities for ML-powered trading system.

This module provides centralized error handling capabilities with custom exception
classes, error classification, retry mechanisms, graceful degradation patterns,
and integration with system components (circuit breaker, health monitor, logger,
disaster recovery).

Key features:
- Custom exception hierarchy for trading-specific errors
- Error classification and categorization
- Retry mechanisms with exponential backoff and jitter
- Error context capture and preservation
- Circuit breaker integration for failure isolation
- Health monitoring integration for system diagnostics
- Comprehensive error reporting and logging
- Graceful degradation patterns
"""

import builtins
import functools
import inspect
import logging
import random
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

# Try to import system components
try:
    from core.circuit_breaker import (
        CircuitBreaker, CircuitState, CircuitType, TriggerType,
        get_circuit_breaker_registry, create_circuit_breaker
    )
    CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    CIRCUIT_BREAKER_AVAILABLE = False

try:
    from core.health_monitor import HealthMonitor, HealthStatus, AlertLevel
    HEALTH_MONITOR_AVAILABLE = True
except ImportError:
    HEALTH_MONITOR_AVAILABLE = False

try:
    from core.event_bus import get_event_bus, Event, EventPriority, EventTopics
    EVENT_BUS_AVAILABLE = True
except ImportError:
    EVENT_BUS_AVAILABLE = False

try:
    from core.disaster_recovery import get_disaster_recovery, RecoveryTrigger
    DISASTER_RECOVERY_AVAILABLE = True
except ImportError:
    DISASTER_RECOVERY_AVAILABLE = False

# Import logger from utils
from utils.logger import get_logger


# Global error handler instance
_error_handler = None

def get_error_handler() -> ErrorHandler:
    """
    Get the global error handler instance.

    Returns:
        ErrorHandler instance
    """
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling."""
    DEBUG = "debug"          # Minor issues, no impact on functionality
    INFO = "info"            # Non-critical issues, minimal impact
    WARNING = "warning"      # Issues that might affect system but not critical
    ERROR = "error"          # Serious issues that affect functionality
    CRITICAL = "critical"    # Severe issues requiring immediate attention
    FATAL = "fatal"          # Catastrophic errors that prevent system operation


class ErrorCategory(Enum):
    """Categories of errors for classification."""
    SYSTEM = "system"              # System-level errors
    NETWORK = "network"            # Network-related errors
    DATABASE = "database"          # Database errors
    API = "api"                    # API-related errors
    MARKET_DATA = "market_data"    # Market data issues
    EXECUTION = "execution"        # Order execution issues
    MODEL = "model"                # ML model issues
    VALIDATION = "validation"      # Data validation issues
    CONFIGURATION = "configuration" # Configuration errors
    AUTHENTICATION = "authentication" # Auth issues
    PERMISSION = "permission"      # Permission issues
    RATE_LIMIT = "rate_limit"      # Rate limiting issues
    TIMEOUT = "timeout"            # Timeout errors
    RESOURCE = "resource"          # Resource-related errors
    DEPENDENCY = "dependency"      # Dependency issues
    UNKNOWN = "unknown"            # Unclassified errors


@dataclass
class ErrorContext:
    """Contextual information about an error."""
    timestamp: float = field(default_factory=time.time)
    component_id: Optional[str] = None
    operation: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    error_id: str = field(default_factory=lambda: f"err_{int(time.time() * 1000)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert error context to dictionary."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp,
            "component_id": self.component_id,
            "operation": self.operation,
            "inputs": {k: str(v)[:200] for k, v in self.inputs.items()},  # Truncate long inputs
            "details": self.details,
            "stack_trace": self.stack_trace
        }


class TradingSystemError(Exception):
    """Base exception class for all trading system errors."""

    def __init__(self, message: str,
                 category: ErrorCategory = ErrorCategory.UNKNOWN,
                 severity: ErrorSeverity = ErrorSeverity.ERROR,
                 context: Optional[ErrorContext] = None,
                 cause: Optional[Exception] = None):
        """
        Initialize trading system error.

        Args:
            message: Error message
            category: Error category
            severity: Error severity
            context: Error context information
            cause: Original exception that caused this error
        """
        self.message = message
        self.category = category
        self.severity = severity
        self.cause = cause

        # Create context if not provided
        if context is None:
            self.context = ErrorContext()

            # Capture stack trace
            if sys.exc_info()[0] is not None:
                self.context.stack_trace = "".join(traceback.format_exception(*sys.exc_info()))
        else:
            self.context = context

            # Add stack trace if not in context
            if self.context.stack_trace is None and sys.exc_info()[0] is not None:
                self.context.stack_trace = "".join(traceback.format_exception(*sys.exc_info()))

        # Call parent constructor
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization and logging."""
        result = {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context.to_dict()
        }

        # Add cause information if available
        if self.cause:
            result["cause"] = {
                "type": type(self.cause).__name__,
                "message": str(self.cause)
            }

        return result

    def __str__(self) -> str:
        """String representation of the error."""
        if self.context and self.context.component_id and self.context.operation:
            return f"{self.severity.value.upper()} [{self.category.value}] in {self.context.component_id}.{self.context.operation}: {self.message}"
        else:
            return f"{self.severity.value.upper()} [{self.category.value}]: {self.message}"


# Context manager for error handling

@contextmanager
def error_context(component_id: str, operation: str, inputs: Optional[Dict[str, Any]] = None,
                 details: Optional[Dict[str, Any]] = None, reraise: bool = True):
    """
    Context manager for error handling with proper context.

    Args:
        component_id: Component ID
        operation: Operation name
        inputs: Input data (optional)
        details: Additional details (optional)
        reraise: Whether to reraise the exception

    Yields:
        Error context for use in the with block
    """
    error_handler = ErrorHandler()
    context = ErrorContext(
        component_id=component_id,
        operation=operation,
        inputs=inputs or {},
        details=details or {}
    )

    try:
        yield context
    except Exception as e:
        # Add any additional details that were set during the with block
        error_handler.handle_exception(
            exception=e,
            component_id=component_id,
            operation=operation,
            inputs=context.inputs,
            details=context.details,
            reraise=reraise
        )


# Graceful degradation patterns

def fallback(fallback_function: Callable, exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
             log_failure: bool = True, logger_name: Optional[str] = None):
    """
    Decorator that provides fallback behavior when a function fails.

    Args:
        fallback_function: Function to call if the primary function fails
        exceptions: Exception types to catch
        log_failure: Whether to log the failure
        logger_name: Logger name to use

    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__)

            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if log_failure:
                    logger.warning(f"Function {func.__name__} failed with {type(e).__name__}: {str(e)}. Using fallback.")
                return fallback_function(*args, **kwargs)
        return wrapper
    return decorator


def circuit_protected(circuit_id: str = None, circuit_type: CircuitType = CircuitType.SYSTEM,
                     scope: str = None, exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
                     on_open: Optional[Callable] = None, logger_name: Optional[str] = None):
    """
    Decorator that integrates function with circuit breaker protection.

    Args:
        circuit_id: Circuit breaker ID, or None to generate from function name
        circuit_type: Type of circuit breaker
        scope: Scope for the circuit breaker
        exceptions: Exceptions to catch and report to circuit breaker
        on_open: Function to call when circuit opens
        logger_name: Logger name to use

    Returns:
        Decorated function
    """
    def decorator(func):
        nonlocal circuit_id, scope

        # Generate circuit ID and scope from function if not provided
        if circuit_id is None:
            circuit_id = f"circuit_{func.__module__}_{func.__name__}"

        if scope is None:
            scope = func.__module__

        # Create or get circuit breaker
        registry = get_circuit_breaker_registry() if CIRCUIT_BREAKER_AVAILABLE else None
        if registry:
            # Configure trigger conditions
            trigger_conditions = {
                TriggerType.ERROR_RATE: {
                    "threshold": 0.3,  # 30% error rate
                    "error_window": 10
                }
            }

            # Register circuit breaker
            try:
                circuit_breaker = create_circuit_breaker(
                    name=func.__name__,
                    circuit_type=circuit_type,
                    scope=scope,
                    trigger_conditions=trigger_conditions
                )
            except Exception as e:
                logger = get_logger(logger_name or func.__module__)
                logger.warning(f"Failed to create circuit breaker: {e}")
                circuit_breaker = None
        else:
            circuit_breaker = None

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__)

            # Check if circuit breaker is open
            if circuit_breaker and circuit_breaker.get_state() != CircuitState.CLOSED:
                if on_open:
                    return on_open(*args, **kwargs)
                else:
                    raise SystemError(f"Circuit breaker {circuit_id} is open")

            try:
                # Call the function
                return func(*args, **kwargs)
            except exceptions as e:
                # Report error to circuit breaker
                if circuit_breaker:
                    circuit_breaker.test_condition(TriggerType.ERROR_RATE, {"error": str(e)})

                # Reraise
                raise

        return wrapper

    return decorator


# Timeout handling

# Rate limiting utilities

class RateLimiter:
    """
    Rate limiter to prevent exceeding API or resource limits.
    """

    def __init__(self, max_calls: int, time_window: float, error_threshold: Optional[int] = None):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls in time window
            time_window: Time window in seconds
            error_threshold: Optional threshold for consecutive errors that triggers circuit breaking
        """
        self._max_calls = max_calls
        self._time_window = time_window
        self._calls = []
        self._lock = threading.RLock()
        self._error_threshold = error_threshold
        self._consecutive_errors = 0
        self._circuit_open = False
        self._last_circuit_check = 0
        self._circuit_reset_time = 60  # Reset circuit after 60 seconds by default

    def __call__(self, func):
        """Decorator implementation."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.check_rate_limit()
            try:
                result = func(*args, **kwargs)
                with self._lock:
                    self._consecutive_errors = 0
                return result
            except Exception as e:
                with self._lock:
                    self._consecutive_errors += 1
                    if (self._error_threshold and
                        self._consecutive_errors >= self._error_threshold and
                        not self._circuit_open):
                        self._circuit_open = True
                        self._last_circuit_check = time.time()
                        logger = get_logger(func.__module__)
                        logger.warning(f"Circuit opened for {func.__name__} due to {self._consecutive_errors} consecutive errors")
                raise

        return wrapper

    def check_rate_limit(self):
        """
        Check if rate limit is exceeded and wait if necessary.

        Raises:
            RateLimitError: If circuit is open
        """
        with self._lock:
            # Check if circuit is open
            if self._circuit_open:
                current_time = time.time()
                if current_time - self._last_circuit_check > self._circuit_reset_time:
                    # Reset circuit after specified time
                    self._circuit_open = False
                    self._consecutive_errors = 0
                    self._last_circuit_check = current_time
                else:
                    raise RateLimitError(f"Circuit is open due to consecutive errors")

            # Clean up old calls
            current_time = time.time()
            self._calls = [t for t in self._calls if current_time - t < self._time_window]

            # Check if we've hit the rate limit
            if len(self._calls) >= self._max_calls:
                # Calculate time to wait
                oldest_call = min(self._calls)
                wait_time = self._time_window - (current_time - oldest_call)

                if wait_time > 0:
                    time.sleep(wait_time)
                    # Update current time after sleeping
                    current_time = time.time()

            # Record this call
            self._calls.append(current_time)


def timeout(seconds: float, error_message: str = "Operation timed out", use_signals: bool = False):
    """
    Decorator that applies a timeout to a function.

    Args:
        seconds: Timeout in seconds
        error_message: Error message when timeout occurs
        use_signals: Whether to use signals (only works in main thread)

    Returns:
        Decorated function that will raise TimeoutError if it takes too long
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if use_signals:
                # Signal-based implementation (only works in main thread)
                import signal

                def handler(signum, frame):
                    raise TimeoutError(error_message)

                # Set the timeout handler
                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, handler)
                signal.setitimer(signal.ITIMER_REAL, seconds)

                try:
                    result = func(*args, **kwargs)
                finally:
                    # Restore the old handler
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old_handler)

                return result
            else:
                # Thread-based implementation (works in any thread)
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(func, *args, **kwargs)
                    try:
                        return future.result(timeout=seconds)
                    except concurrent.futures.TimeoutError:
                        raise TimeoutError(error_message)

        return wrapper

    return decorator


# Define specific exception types for different error categories

class SystemError(TradingSystemError):
    """System-level errors."""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR, **kwargs):
        super().__init__(message, category=ErrorCategory.SYSTEM, severity=severity, **kwargs)


# Safe execution utility functions

# Validation utilities

def validate_or_raise(condition: bool, error_message: str, exception_type: Type[TradingSystemError] = ValidationError,
                    component_id: Optional[str] = None, operation: Optional[str] = None,
                    details: Optional[Dict[str, Any]] = None):
    """
    Validate a condition and raise an exception if it fails.

    Args:
        condition: Condition to validate
        error_message: Error message if validation fails
        exception_type: Type of exception to raise
        component_id: Component ID for error context
        operation: Operation name for error context
        details: Additional error details

    Raises:
        Exception of specified type if condition is False
    """
    if not condition:
        # Create context
        context = ErrorContext(
            component_id=component_id,
            operation=operation,
            details=details or {}
        )

        # Add stack trace
        if sys.exc_info()[0] is not None:
            context.stack_trace = "".join(traceback.format_exception(*sys.exc_info()))

        # Raise appropriate exception
        raise exception_type(error_message, context=context)


def assert_trading(condition: bool, error_message: str,
                  component_id: Optional[str] = None,
                  operation: Optional[str] = None,
                  category: ErrorCategory = ErrorCategory.VALIDATION,
                  severity: ErrorSeverity = ErrorSeverity.ERROR):
    """
    Assert a condition for trading system.

    Similar to standard assert but with better error handling and reporting.

    Args:
        condition: Condition to assert
        error_message: Error message if assertion fails
        component_id: Component ID for error context
        operation: Operation name for error context
        category: Error category
        severity: Error severity

    Raises:
        TradingSystemError if condition is False
    """
    if not condition:
        # Create context
        context = ErrorContext(
            component_id=component_id,
            operation=operation
        )

        # Add stack trace with full traceback
        frame = inspect.currentframe().f_back
        tb = traceback.format_stack()
        # Remove this function's frame from traceback for cleaner output
        tb = tb[:-1]
        context.stack_trace = "".join(tb)

        # Find correct error class based on category
        error_classes = {
            ErrorCategory.SYSTEM: SystemError,
            ErrorCategory.NETWORK: NetworkError,
            ErrorCategory.DATABASE: DatabaseError,
            ErrorCategory.API: ApiError,
            ErrorCategory.MARKET_DATA: MarketDataError,
            ErrorCategory.EXECUTION: ExecutionError,
            ErrorCategory.MODEL: ModelError,
            ErrorCategory.VALIDATION: ValidationError,
            ErrorCategory.CONFIGURATION: ConfigurationError,
            ErrorCategory.AUTHENTICATION: AuthenticationError,
            ErrorCategory.PERMISSION: PermissionError,
            ErrorCategory.RATE_LIMIT: RateLimitError,
            ErrorCategory.TIMEOUT: TimeoutError,
            ErrorCategory.RESOURCE: ResourceError,
            ErrorCategory.DEPENDENCY: DependencyError
        }

        error_class = error_classes.get(category, TradingSystemError)

        # Raise appropriate exception
        raise error_class(error_message, severity=severity, context=context)


def safe_execute(func: Callable, *args, component_id: Optional[str] = None,
                operation: Optional[str] = None, default_value: Any = None,
                log_level: int = logging.ERROR, capture_inputs: bool = True, **kwargs) -> Any:
    """
    Execute a function safely, catching and handling exceptions.

    Args:
        func: Function to execute
        *args: Positional arguments for the function
        component_id: Component ID for error context
        operation: Operation name for error context
        default_value: Value to return if an exception occurs
        log_level: Log level for errors
        **kwargs: Keyword arguments for the function

    Returns:
        Function result or default_value if an exception occurs
    """
    error_handler = ErrorHandler()

    # Determine operation name if not provided
    if operation is None and callable(func):
        operation = getattr(func, "__name__", "unknown_operation")

    # Determine component ID if not provided
    if component_id is None and args and hasattr(args[0], "__class__"):
        component_id = args[0].__class__.__name__

    # Capture inputs if requested
    inputs = {}
    if capture_inputs:
        try:
            # Get argument names from function signature
            sig = inspect.signature(func)
            params = list(sig.parameters.keys())

            # Match positional args to parameter names
            for i, arg in enumerate(args):
                if i < len(params):
                    # Convert simple values to string, truncate if needed
                    if isinstance(arg, (str, int, float, bool)):
                        inputs[params[i]] = str(arg)[:100]

            # Add keyword args (simple values only)
            for key, value in kwargs.items():
                if isinstance(value, (str, int, float, bool)):
                    inputs[key] = str(value)[:100]
        except Exception:
            # Don't fail if we can't extract inputs
            pass

    try:
        return func(*args, **kwargs)
    except Exception as e:
        # Handle exception but don't reraise
        error_handler.handle_exception(
            exception=e,
            component_id=component_id,
            operation=operation,
            inputs=inputs,
            reraise=False,
            log_level=log_level
        )
        return default_value


class NetworkError(TradingSystemError):
    """Network-related errors."""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR, **kwargs):
        super().__init__(message, category=ErrorCategory.NETWORK, severity=severity, **kwargs)


class DatabaseError(TradingSystemError):
    """Database-related errors."""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR, **kwargs):
        super().__init__(message, category=ErrorCategory.DATABASE, severity=severity, **kwargs)


class ApiError(TradingSystemError):
    """API-related errors."""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR, **kwargs):
        super().__init__(message, category=ErrorCategory.API, severity=severity, **kwargs)


class MarketDataError(TradingSystemError):
    """Market data-related errors."""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR, **kwargs):
        super().__init__(message, category=ErrorCategory.MARKET_DATA, severity=severity, **kwargs)


class ExecutionError(TradingSystemError):
    """Order execution errors."""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR, **kwargs):
        super().__init__(message, category=ErrorCategory.EXECUTION, severity=severity, **kwargs)


class ModelError(TradingSystemError):
    """ML model-related errors."""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR, **kwargs):
        super().__init__(message, category=ErrorCategory.MODEL, severity=severity, **kwargs)


class ValidationError(TradingSystemError):
    """Data validation errors."""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.WARNING, **kwargs):
        super().__init__(message, category=ErrorCategory.VALIDATION, severity=severity, **kwargs)


class ConfigurationError(TradingSystemError):
    """Configuration-related errors."""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR, **kwargs):
        super().__init__(message, category=ErrorCategory.CONFIGURATION, severity=severity, **kwargs)


class AuthenticationError(TradingSystemError):
    """Authentication-related errors."""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR, **kwargs):
        super().__init__(message, category=ErrorCategory.AUTHENTICATION, severity=severity, **kwargs)


class PermissionError(TradingSystemError):
    """Permission-related errors."""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR, **kwargs):
        super().__init__(message, category=ErrorCategory.PERMISSION, severity=severity, **kwargs)


class RateLimitError(TradingSystemError):
    """Rate limiting errors."""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.WARNING, **kwargs):
        super().__init__(message, category=ErrorCategory.RATE_LIMIT, severity=severity, **kwargs)


class TimeoutError(TradingSystemError):
    """Timeout errors."""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.WARNING, **kwargs):
        super().__init__(message, category=ErrorCategory.TIMEOUT, severity=severity, **kwargs)


class ResourceError(TradingSystemError):
    """Resource-related errors."""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR, **kwargs):
        super().__init__(message, category=ErrorCategory.RESOURCE, severity=severity, **kwargs)


class DependencyError(TradingSystemError):
    """Dependency-related errors."""
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.ERROR, **kwargs):
        super().__init__(message, category=ErrorCategory.DEPENDENCY, severity=severity, **kwargs)


# Error classification and mapping

ERROR_MAPPING = {
    # Network errors
    "ConnectionError": (NetworkError, ErrorSeverity.ERROR),
    "ConnectionRefusedError": (NetworkError, ErrorSeverity.ERROR),
    "ConnectionResetError": (NetworkError, ErrorSeverity.ERROR),
    "ConnectionAbortedError": (NetworkError, ErrorSeverity.ERROR),
    "HTTPError": (NetworkError, ErrorSeverity.ERROR),
    "URLError": (NetworkError, ErrorSeverity.ERROR),
    "SSLError": (NetworkError, ErrorSeverity.ERROR),
    "ConnectTimeout": (NetworkError, ErrorSeverity.ERROR),
    "ReadTimeout": (TimeoutError, ErrorSeverity.WARNING),

    # Database errors
    "DatabaseError": (DatabaseError, ErrorSeverity.ERROR),
    "OperationalError": (DatabaseError, ErrorSeverity.ERROR),
    "IntegrityError": (DatabaseError, ErrorSeverity.ERROR),
    "ProgrammingError": (DatabaseError, ErrorSeverity.ERROR),
    "NotSupportedError": (DatabaseError, ErrorSeverity.ERROR),

    # API errors
    "InvalidApiKeyError": (ApiError, ErrorSeverity.CRITICAL),
    "ApiRateLimitError": (RateLimitError, ErrorSeverity.WARNING),
    "ApiResponseError": (ApiError, ErrorSeverity.ERROR),

    # Authentication errors
    "AuthenticationError": (AuthenticationError, ErrorSeverity.ERROR),
    "TokenExpiredError": (AuthenticationError, ErrorSeverity.WARNING),
    "PermissionDeniedError": (PermissionError, ErrorSeverity.ERROR),

    # System errors
    "MemoryError": (ResourceError, ErrorSeverity.CRITICAL),
    "OSError": (SystemError, ErrorSeverity.ERROR),
    "IOError": (SystemError, ErrorSeverity.ERROR),

    # Format and validation errors
    "ValueError": (ValidationError, ErrorSeverity.WARNING),
    "TypeError": (ValidationError, ErrorSeverity.WARNING),
    "KeyError": (ValidationError, ErrorSeverity.WARNING),
    "IndexError": (ValidationError, ErrorSeverity.WARNING),
    "AttributeError": (ValidationError, ErrorSeverity.WARNING),

    # Configuration errors
    "ConfigurationError": (ConfigurationError, ErrorSeverity.ERROR),
    "ImportError": (DependencyError, ErrorSeverity.ERROR),
    "ModuleNotFoundError": (DependencyError, ErrorSeverity.ERROR),
}


class ErrorHandler:
    """
    Central error handler with integrations to system components.

    This class provides methods for handling errors, including classification,
    logging, reporting, circuit breaker integration, and health monitoring.
    """

    _instance = None
    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        """Ensure singleton behavior."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ErrorHandler, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, logger=None, circuit_breaker=None, health_monitor=None, event_bus=None):
        """
        Initialize the error handler.

        Args:
            logger: Logger instance, or None to create one
            circuit_breaker: CircuitBreaker instance, or None to try getting from registry
            health_monitor: HealthMonitor instance
            event_bus: Event bus instance
        """
        with self._lock:
            if self._initialized:
                return

            # Initialize components
            self._logger = logger or get_logger(__name__)
            self._circuit_breaker = circuit_breaker
            self._health_monitor = health_monitor
            self._event_bus = event_bus

            # Try to get circuit breaker registry if available
            if CIRCUIT_BREAKER_AVAILABLE and self._circuit_breaker is None:
                try:
                    self._circuit_breaker_registry = get_circuit_breaker_registry()
                except Exception as e:
                    self._logger.warning(f"Failed to get circuit breaker registry: {e}")
                    self._circuit_breaker_registry = None
            else:
                self._circuit_breaker_registry = None

            # Try to get event bus if available
            if EVENT_BUS_AVAILABLE and self._event_bus is None:
                try:
                    self._event_bus = get_event_bus()
                except Exception as e:
                    self._logger.warning(f"Failed to get event bus: {e}")
                    self._event_bus = None

            # Try to get disaster recovery if available
            if DISASTER_RECOVERY_AVAILABLE:
                try:
                    self._disaster_recovery = get_disaster_recovery()
                except Exception as e:
                    self._logger.warning(f"Failed to get disaster recovery: {e}")
                    self._disaster_recovery = None
            else:
                self._disaster_recovery = None

            # Error counts for tracking
            self._error_counts = {}
            self._error_categories = {}
            self._error_components = {}

        # Install global exception hook if not already installed
        self._install_exception_hook()

            # Marked as initialized
            self._initialized = True
            self._logger.info("Error handler initialized")

    def handle_exception(self, exception: Exception, component_id: Optional[str] = None,
                        operation: Optional[str] = None, details: Optional[Dict[str, Any]] = None,
                        inputs: Optional[Dict[str, Any]] = None, reraise: bool = True,
                        log_level: Optional[int] = None) -> Optional[TradingSystemError]:
        """
        Handle an exception and perform appropriate logging and reporting.

        Args:
            exception: Exception to handle
            component_id: ID of component where error occurred
            operation: Operation where error occurred
            details: Additional error details
            inputs: Input values related to the error
            reraise: Whether to reraise the exception
            log_level: Optional log level override

        Returns:
            TradingSystemError if converted, or None if reraise is False and exception isn't converted
        """
        # Create error context
        context = ErrorContext(
            component_id=component_id,
            operation=operation,
            details=details or {},
            inputs=inputs or {}
        )

        # Capture stack trace
        context.stack_trace = "".join(traceback.format_exception(*sys.exc_info()))

        # Convert to TradingSystemError if not already
        error = self._convert_exception(exception, context)

        # Determine log level based on severity
        if log_level is None:
            if error.severity == ErrorSeverity.FATAL:
                log_level = logging.CRITICAL
            elif error.severity == ErrorSeverity.CRITICAL:
                log_level = logging.CRITICAL
            elif error.severity == ErrorSeverity.ERROR:
                log_level = logging.ERROR
            elif error.severity == ErrorSeverity.WARNING:
                log_level = logging.WARNING
            elif error.severity == ErrorSeverity.INFO:
                log_level = logging.INFO
            else:
                log_level = logging.DEBUG

        # Log the error
        self._log_error(error, log_level)

        # Update error tracking
        self._track_error(error)

        # Integrate with system components
        self._report_to_circuit_breaker(error)
        self._report_to_health_monitor(error)
        self._report_to_event_bus(error)
        self._report_to_disaster_recovery(error)

        # Reraise if requested
        if reraise:
            raise error

        return error

    def _convert_exception(self, exception: Exception, context: ErrorContext) -> TradingSystemError:
        """
        Convert standard exceptions to appropriate TradingSystemError subtypes.

        Args:
            exception: Exception to convert
            context: Error context

        Returns:
            Converted TradingSystemError
        """
        # If already a TradingSystemError, update context if needed and return
        if isinstance(exception, TradingSystemError):
            # Update context if provided
            if context and (not exception.context or not exception.context.component_id):
                exception.context = context
            return exception

        # Get exception type name
        exception_type = type(exception).__name__

        # Try to map to appropriate error type
        error_class, severity = ERROR_MAPPING.get(exception_type, (TradingSystemError, ErrorSeverity.ERROR))

        # Create appropriate error instance
        if error_class == TradingSystemError:
            # Use generic error with unknown category
            return TradingSystemError(
                message=str(exception),
                severity=severity,
                context=context,
                cause=exception
            )
        else:
            # Use mapped error class
            return error_class(
                message=str(exception),
                severity=severity,
                context=context,
                cause=exception
            )

    def _log_error(self, error: TradingSystemError, log_level: int):
        """
        Log an error with appropriate level and context.

        Args:
            error: Error to log
            log_level: Log level
        """
        # Create component-specific logger if component_id is available
        logger = self._logger
        if error.context and error.context.component_id:
            logger = get_logger(error.context.component_id)

        # Create detailed error message
        if error.context and error.context.operation:
            operation_info = f" during {error.context.operation}"
        else:
            operation_info = ""

        error_message = f"{error.severity.value.upper()} [{error.category.value}]{operation_info}: {error.message}"

        # Log with appropriate level
        logger.log(log_level, error_message, exc_info=True)

        # Log additional details at debug level
        if error.context and (error.context.details or error.context.inputs):
            details_str = ", ".join(f"{k}={v}" for k, v in error.context.details.items())
            inputs_str = ", ".join(f"{k}={v}" for k, v in error.context.inputs.items())

            if details_str and inputs_str:
                logger.debug(f"Error details: {details_str} | Inputs: {inputs_str}")
            elif details_str:
                logger.debug(f"Error details: {details_str}")
            elif inputs_str:
                logger.debug(f"Error inputs: {inputs_str}")

    def _track_error(self, error: TradingSystemError):
        """
        Track error for monitoring and trending.

        Args:
            error: Error to track
        """
        # Get error type
        error_type = type(error).__name__

        # Update error counts
        with self._lock:
            # Overall count
            self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1

            # Category count
            category = error.category.value
            self._error_categories[category] = self._error_categories.get(category, 0) + 1

            # Component count
            if error.context and error.context.component_id:
                component_id = error.context.component_id
                if component_id not in self._error_components:
                    self._error_components[component_id] = {}
                self._error_components[component_id][error_type] = (
                    self._error_components[component_id].get(error_type, 0) + 1
                )

    def _report_to_circuit_breaker(self, error: TradingSystemError):
        """
        Report error to circuit breaker for potential tripping.

        Args:
            error: Error to report
        """
        if not CIRCUIT_BREAKER_AVAILABLE:
            return

        # Only report critical and fatal errors to circuit breaker
        if error.severity not in [ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
            return

        try:
            # Get appropriate circuit breaker
            circuit_breaker = None
            component_id = error.context.component_id if error.context else None

            # If we have a specific circuit breaker, use it
            if self._circuit_breaker:
                circuit_breaker = self._circuit_breaker

            # Otherwise try to get from registry
            elif self._circuit_breaker_registry and component_id:
                # Determine appropriate circuit type based on error category
                if error.category == ErrorCategory.MARKET_DATA:
                    circuit_type = CircuitType.MARKET
                elif error.category == ErrorCategory.MODEL:
                    circuit_type = CircuitType.MODEL
                elif error.category == ErrorCategory.EXECUTION:
                    circuit_type = CircuitType.EXECUTION
                else:
                    circuit_type = CircuitType.SYSTEM

                # Try to get matching circuit breaker
                circuit_breakers = self._circuit_breaker_registry.get_circuit_breakers(
                    circuit_type=circuit_type,
                    scope=component_id
                )

                if circuit_breakers:
                    circuit_breaker = circuit_breakers[0]

            # Trip circuit breaker if found
            if circuit_breaker:
                circuit_breaker.trip(
                    trigger_type=TriggerType.ERROR_RATE,
                    trigger_value=error.to_dict(),
                    reason=f"{error.severity.value.upper()} {error.category.value} error: {error.message}"
                )
        except Exception as e:
            self._logger.warning(f"Failed to report to circuit breaker: {e}")

    def _report_to_health_monitor(self, error: TradingSystemError):
        """
        Report error to health monitor for system diagnostics.

        Args:
            error: Error to report
        """
        if not HEALTH_MONITOR_AVAILABLE or not self._health_monitor:
            return

        try:
            component_id = error.context.component_id if error.context else "system"

            # Log error to health monitor
            self._health_monitor.log_error(
                component_id=component_id,
                error_type=f"{type(error).__name__}",
                error_message=error.message
            )
        except Exception as e:
            self._logger.warning(f"Failed to report to health monitor: {e}")

    def _report_to_event_bus(self, error: TradingSystemError):
        """
        Report error to event bus for system-wide notification.

        Args:
            error: Error to report
        """
        if not EVENT_BUS_AVAILABLE or not self._event_bus:
            return

        try:
            # Determine event priority based on severity
            if error.severity in [ErrorSeverity.FATAL, ErrorSeverity.CRITICAL]:
                priority = EventPriority.CRITICAL
            elif error.severity == ErrorSeverity.ERROR:
                priority = EventPriority.HIGH
            else:
                priority = EventPriority.NORMAL

            # Create and publish event
            event = Event(
                topic=EventTopics.SYSTEM_ERROR,
                data=error.to_dict(),
                priority=priority,
                source="error_handler"
            )

            self._event_bus.publish(event)
        except Exception as e:
            self._logger.warning(f"Failed to report to event bus: {e}")

    def _report_to_disaster_recovery(self, error: TradingSystemError):
        """
        Report critical errors to disaster recovery for potential snapshot.

        Args:
            error: Error to report
        """
        if not DISASTER_RECOVERY_AVAILABLE or not self._disaster_recovery:
            return

        # Only report fatal errors for disaster recovery
        if error.severity != ErrorSeverity.FATAL:
            return

        try:
            # Create snapshot if configured
            if hasattr(self._disaster_recovery, 'config') and getattr(self._disaster_recovery.config, 'snapshot_on_error', False):
                self._disaster_recovery.create_snapshot(
                    trigger=RecoveryTrigger.AUTOMATIC,
                    metadata={
                        "error_type": type(error).__name__,
                        "error_category": error.category.value,
                        "error_message": error.message,
                        "component_id": error.context.component_id if error.context else None
                    }
                )
        except Exception as e:
            self._logger.warning(f"Failed to report to disaster recovery: {e}")

    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get error statistics for monitoring.

        Returns:
            Dictionary with error statistics
        """
        with self._lock:
            return {
                "total_errors": sum(self._error_counts.values()),
                "error_types": self._error_counts.copy(),
                "error_categories": self._error_categories.copy(),
                "error_components": {k: dict(v) for k, v in self._error_components.items()}
            }

    def reset_error_stats(self):
        """Reset error statistics."""
        with self._lock:
            self._error_counts = {}
            self._error_categories = {}
            self._error_components = {}

    def _install_exception_hook(self):
        """Install global exception hook to catch unhandled exceptions."""
        # Store original exception hook
        self._original_excepthook = sys.excepthook

        def global_exception_handler(exc_type, exc_value, exc_traceback):
            """Global exception handler for unhandled exceptions."""
            # Skip KeyboardInterrupt
            if issubclass(exc_type, KeyboardInterrupt):
                self._original_excepthook(exc_type, exc_value, exc_traceback)
                return

            # Create error context
            context = ErrorContext()
            context.stack_trace = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))

            # Try to determine component and operation from traceback
            try:
                tb = traceback.extract_tb(exc_traceback)
                if tb:
                    frame = tb[-1]  # Last frame in traceback
                    context.component_id = frame.filename.split('/')[-1].split('\\')[-1].split('.')[0]
                    context.operation = frame.name
            except Exception:
                pass

            # Convert to trading system error
            error = self._convert_exception(exc_value, context)

            # Log the error
            self._log_error(error, logging.CRITICAL)

            # Integrate with system components
            self._report_to_circuit_breaker(error)
            self._report_to_health_monitor(error)
            self._report_to_event_bus(error)
            self._report_to_disaster_recovery(error)

            # Call original exception hook
            self._original_excepthook(exc_type, exc_value, exc_traceback)

        # Install custom exception hook
        sys.excepthook = global_exception_handler


# Retry mechanism with exponential backoff

def retry(max_tries: int = 3, delay: float = 1.0, backoff: float = 2.0,
         jitter: float = 0.1, exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
         circuit_breaker: Optional[CircuitBreaker] = None, logger_name: Optional[str] = None,
         reraise: bool = True, on_retry: Optional[Callable[[Exception, int], None]] = None):
    """
    Retry decorator with exponential backoff.

    Args:
        max_tries: Maximum number of tries
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier
        jitter: Jitter factor to randomize delay
        exceptions: Exception(s) to catch and retry
        circuit_breaker: Optional circuit breaker to check and trigger
        logger_name: Logger name to use, or None for default
        reraise: Whether to reraise the last exception
        on_retry: Optional callback function called before each retry

    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger
            logger = get_logger(logger_name or func.__module__)
            error_handler = ErrorHandler()

            # Determine component ID and operation name
            component_id = None
            if args and hasattr(args[0], '__class__'):
                component_id = args[0].__class__.__name__
            operation = func.__name__

            # Prepare input data to capture context
            input_data = {}
            try:
                # Get argument names from function signature
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())

                # Match positional args to parameter names
                for i, arg in enumerate(args):
                    if i < len(params):
                        # Convert simple values to string, truncate if needed
                        if isinstance(arg, (str, int, float, bool)):
                            input_data[params[i]] = str(arg)[:100]

                # Add keyword args (simple values only)
                for key, value in kwargs.items():
                    if isinstance(value, (str, int, float, bool)):
                        input_data[key] = str(value)[:100]
            except Exception:
                # Don't fail the retry mechanism if we can't extract inputs
                pass

            tries = 0
            while True:
                try:
                    # Check circuit breaker if provided
                    if circuit_breaker and circuit_breaker.get_state() != CircuitState.CLOSED:
                        raise SystemError(f"Circuit breaker is open: {circuit_breaker.circuit_id}")

                    # Attempt to call the function
                    return func(*args, **kwargs)

                except exceptions as e:
                    tries += 1

                    # If this was the last try, handle error and optionally reraise
                    if tries >= max_tries:
                        # Handle the exception
                        error_handler.handle_exception(
                            exception=e,
                            component_id=component_id,
                            operation=operation,
                            details={"max_tries": max_tries, "tries": tries},
                            inputs=input_data,
                            reraise=reraise
                        )

                        # If we're still here, the exception wasn't reraised
                        return None

                    # Calculate retry delay with exponential backoff and jitter
                    retry_delay = delay * (backoff ** (tries - 1))
                    if jitter:
                        retry_delay = retry_delay + random.uniform(-jitter * retry_delay, jitter * retry_delay)

                    # Log retry attempt
                    logger.warning(
                        f"Retry {tries}/{max_tries} for {operation} after error: {str(e)}. "
                        f"Retrying in {retry_delay:.2f}s"
                    )

                    # Call on_retry callback if provided
                    if on_retry:
                        try:
                            on_retry(e, tries)
                        except Exception as cb_error:
                            logger.error(f"Error in retry callback: {str(cb_error)}")

                    # Wait before retrying
                    time.sleep(retry_delay)