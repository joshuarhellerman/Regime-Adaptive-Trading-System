"""
logger.py - Logging utilities for the ML-powered trading system.

This module provides advanced logging capabilities specifically designed for
algorithmic trading systems, including performance tracking, structured logging,
and integration with the system's event bus and health monitoring.
"""

import logging
import os
import sys
import time
import json
import threading
import functools
import inspect
import traceback
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Set, Tuple
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import importlib
# Need logging.config for dictConfig
import logging.config

# Import configuration from config module
from config.logging_config import LogLevel, LogFormat, LogDestination, LoggingConfig, get_logging_config

# Import ConfigManager from base_config
from config.base_config import ConfigManager

# Try to import event bus for integration
try:
    from core.event_bus import get_event_bus, Event, EventPriority, EventTopics
    EVENT_BUS_AVAILABLE = True
except ImportError:
    EVENT_BUS_AVAILABLE = False

# Try to import health monitor for integration
try:
    from core.health_monitor import HealthMonitor, HealthStatus
    HEALTH_MONITOR_AVAILABLE = True
except ImportError:
    HEALTH_MONITOR_AVAILABLE = False

# Get a basic logger for early messages during setup
_initial_logger = logging.getLogger(__name__)


class LoggerManager:
    """
    Centralizes logger configuration and management.
    """
    _instance = None
    _initialized = False
    _lock = threading.RLock()
    _loggers = {}
    _config = None
    _event_bus = None
    _health_monitor = None
    _root_logger = None
    _performance_trackers = {}

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    # Initialize _initialized here to avoid issues in concurrent access
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            # Initial configuration setup
            try:
                self.config_manager = ConfigManager()
                config_dict = self.config_manager.load_config("logging")

                if config_dict:
                    # Assuming LoggingConfig has a from_dict or similar method
                    # If not, use the dict directly or adapt as needed
                    if hasattr(LoggingConfig, 'from_dict'):
                         self._config = LoggingConfig.from_dict(config_dict)
                    else:
                         # Fallback: Create instance directly (might need adjustments)
                         self._config = LoggingConfig(**config_dict) # Adjust based on LoggingConfig structure
                else:
                    _initial_logger.warning("No logging configuration found, using defaults.")
                    self._config = LoggingConfig() # Use default LoggingConfig
                    self.save_config(self._config) # Save default if none exists

                # Use _configure_from_settings as it contains the setup logic
                self._configure_from_settings()
                self._initialized = True

            except Exception as e:
                 # Critical error during init, log and fallback
                 sys.stderr.write(f"CRITICAL ERROR initializing LoggerManager: {e}\n{traceback.format_exc()}")
                 # Basic fallback config
                 logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
                 self._root_logger = logging.getLogger() # Ensure root logger is set for fallback

    def _configure_from_settings(self):
        """Load and apply logging configuration."""
        try:
            # Config should already be loaded in __init__
            if not self._config:
                 _initial_logger.error("Logging configuration missing during _configure_from_settings.")
                 # Attempt to load defaults again
                 self._config = LoggingConfig()

            # Ensure log directory exists
            log_dir = Path(self._config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            # Set up Python's logging system
            logging_config = self._config.get_python_logging_config()

            # Dynamically add handlers/formatters if needed (e.g., DatabaseLogHandler)
            if 'handlers' in logging_config and 'db_handler' in logging_config['handlers']:
                if '()' not in logging_config['handlers']['db_handler']:
                    logging_config['handlers']['db_handler']['()'] = DatabaseLogHandler
            if 'formatters' in logging_config and 'structured' in logging_config['formatters']:
                 if '()' not in logging_config['formatters']['structured']:
                     logging_config['formatters']['structured']['()'] = StructuredFormatter

            # Apply the configuration
            logging.config.dictConfig(logging_config)

            # Store the root logger
            self._root_logger = logging.getLogger()

            # Try to get event bus instance
            if EVENT_BUS_AVAILABLE:
                try:
                    self._event_bus = get_event_bus()
                except Exception as e:
                    self._root_logger.warning(f"Unable to initialize event bus integration: {e}")

            # Try to get health monitor instance
            if HEALTH_MONITOR_AVAILABLE:
                try:
                    # How to get the HM instance depends on the application structure
                    # For now, allow setting it externally via set_health_monitor
                    self._health_monitor = None # Placeholder, needs actual instance
                    # Example: self._health_monitor = get_health_monitor() # if available globally
                    self._root_logger.info("Health monitor integration is available.")
                except Exception as e:
                    self._root_logger.warning(f"Unable to initialize health monitor integration: {e}")

            # Set up special loggers
            self._setup_audit_logger()

            # Enable warnings capture if configured
            if self._config.capture_warnings:
                logging.captureWarnings(True)
                self._root_logger.info("Logging capture of warnings enabled.")

            # Set up exception hook if configured
            if self._config.log_exceptions:
                self._setup_exception_hook()
                self._root_logger.info("Global exception hook enabled.")

            # Log startup message using the configured root logger
            self._root_logger.info(
                f"Logging system initialized with root level {self._config.root_level.name}. Log dir: {self._config.log_dir}"
            )

        except Exception as e:
            # Use sys.stderr for critical setup failures before logging is fully working
            sys.stderr.write(f"Error during logging configuration: {e}\n{traceback.format_exc()}")
            # Set up a basic console logger as fallback
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                stream=sys.stdout
            )
            self._root_logger = logging.getLogger()
            self._root_logger.error(f"Failed to fully initialize logging system: {e}", exc_info=True)

    def _setup_audit_logger(self):
        """Set up the audit logger for compliance and record-keeping."""
        if not hasattr(self._config, 'audit') or not self._config.audit.enable_audit_logging:
            self._root_logger.debug("Audit logging is disabled in configuration.")
            return

        try:
            audit_logger_name = self._config.audit.audit_logger_name
            audit_logger = logging.getLogger(audit_logger_name)

            # Ensure audit logger has appropriate handlers if not already configured by dictConfig
            if not audit_logger.handlers:
                self._root_logger.warning(f"Audit logger '{audit_logger_name}' has no handlers defined in config. Adding default file handler.")
                # Create a dedicated file handler for audit logs
                audit_file = Path(self._config.log_dir) / "audit.log"

                # Ensure log directory exists
                audit_file.parent.mkdir(parents=True, exist_ok=True)

                # Use TimedRotatingFileHandler by default for audit
                handler = TimedRotatingFileHandler(
                    filename=str(audit_file),
                    when=self._config.audit.get('rotation_interval', 'd')[:1], # Default daily if not specified
                    backupCount=self._config.audit.get('backup_count', 7), # Default 7 backups
                    encoding='utf-8' # Specify encoding
                )

                # Use a secure format for audit logs
                formatter = logging.Formatter(
                     self._config.audit.get('audit_log_format',
                     '%(asctime)s [AUDIT] [%(levelname)s] %(message)s - %(pathname)s:%(lineno)d')
                )
                handler.setFormatter(formatter)
                audit_logger.addHandler(handler)

                # Ensure audit logger level is appropriate
                audit_logger.setLevel(self._config.audit.get('audit_log_level', 'INFO').upper())
                audit_logger.propagate = False # Prevent duplication if root logger has handlers
                self._root_logger.info(f"Audit logger '{audit_logger_name}' configured with default file handler.")
            else:
                 self._root_logger.info(f"Audit logger '{audit_logger_name}' configured via dictConfig.")

        except Exception as e:
            self._root_logger.error(f"Failed to setup audit logger '{self._config.audit.audit_logger_name}': {e}", exc_info=True)


    def _setup_exception_hook(self):
        """Set up a global exception hook to log unhandled exceptions."""
        original_hook = sys.excepthook

        def exception_handler(exc_type, exc_value, exc_traceback):
            """Handle uncaught exceptions by logging them."""
            # Ensure root logger is available, even if init failed partially
            logger_to_use = self._root_logger if self._root_logger else logging.getLogger()

            if issubclass(exc_type, KeyboardInterrupt):
                # Call original handler for KeyboardInterrupt
                original_hook(exc_type, exc_value, exc_traceback)
                return

            # Log the exception
            logger_to_use.critical(
                "Uncaught exception",
                exc_info=(exc_type, exc_value, exc_traceback)
            )

            # Also log to event bus if available
            if self._event_bus and EVENT_BUS_AVAILABLE:
                try:
                    # Use traceback.format_exception for a more complete string
                    tb_list = traceback.format_exception(exc_type, exc_value, exc_traceback)
                    tb_string = "".join(tb_list)

                    error_data = {
                        "exception_type": exc_type.__name__,
                        "exception_message": str(exc_value),
                        "traceback": tb_string, # Use formatted string
                        "timestamp": time.time()
                    }

                    event = Event(
                        topic=EventTopics.SYSTEM_ERROR,
                        data=error_data,
                        priority=EventPriority.CRITICAL, # Use CRITICAL for uncaught exceptions
                        source="exception_hook"
                    )

                    self._event_bus.publish(event)
                except Exception as e:
                    logger_to_use.error(f"Failed to publish uncaught exception to event bus: {e}")

            # Call the original exception handler
            original_hook(exc_type, exc_value, exc_traceback)

        # Set the custom exception handler
        sys.excepthook = exception_handler

    def get_logger(self, name: str) -> 'TradingLogger':
        """
        Get a logger instance with the specified name.

        Args:
            name: Logger name, typically the module name

        Returns:
            TradingLogger instance
        """
        if not self._initialized:
            # Avoid errors if accessed before full initialization
            # Return a basic logger wrapper in this case
            return TradingLogger(name, logging.getLogger(name), self)

        with self._lock:
            if name in self._loggers:
                return self._loggers[name]

            # Create a new logger using Python's logging system
            # This ensures it inherits config from dictConfig
            python_logger = logging.getLogger(name)

            # Create our wrapper
            trading_logger = TradingLogger(name, python_logger, self)

            # Store it for reuse
            self._loggers[name] = trading_logger

            return trading_logger

    def get_audit_logger(self) -> 'TradingLogger':
        """
        Get the audit logger for compliance logging.

        Returns:
            TradingLogger instance configured for audit logging
        """
        if not hasattr(self._config, 'audit') or not self._config.audit.enable_audit_logging:
            # Return a standard logger if audit logging is disabled
            self._root_logger.warning("Audit logging disabled, returning standard logger for audit requests.")
            return self.get_logger("audit_disabled")

        audit_logger_name = self._config.audit.audit_logger_name
        # Use get_logger to ensure proper setup and caching
        return self.get_logger(audit_logger_name)

    def update_configuration(self, new_config_dict: Optional[Dict[str, Any]] = None):
        """
        Update the logging configuration from a dictionary.

        Args:
            new_config_dict: New logging configuration dictionary, or None to reload from source.
        """
        with self._lock:
            _initial_logger.info("Attempting to update logging configuration...")
            # Save the old loggers' names
            old_logger_names = list(self._loggers.keys())

            # Reset internal state (but keep the instance)
            self._loggers = {}
            self._initialized = False # Mark as re-initializing
            # Close existing handlers managed by logging.config
            logging.shutdown()

            try:
                # Load new config dict
                if new_config_dict:
                    # --- FIX: Use instance to save ---
                    try:
                        self.config_manager.save_config(new_config_dict, config_name="logging")
                        _initial_logger.info("New configuration dictionary saved.")
                    except Exception as e:
                         _initial_logger.error(f"Failed to save new logging configuration: {e}")
                         # Continue with the provided dict anyway? Or raise? For now, continue.

                    # Convert dict to LoggingConfig object if possible
                    if hasattr(LoggingConfig, 'from_dict'):
                         self._config = LoggingConfig.from_dict(new_config_dict)
                    else:
                         self._config = LoggingConfig(**new_config_dict) # Adjust as needed
                else:
                    # Reload configuration from source
                    _initial_logger.info("Reloading logging configuration from source.")
                    config_dict = self.config_manager.load_config("logging")
                    if config_dict:
                        if hasattr(LoggingConfig, 'from_dict'):
                            self._config = LoggingConfig.from_dict(config_dict)
                        else:
                            self._config = LoggingConfig(**config_dict) # Adjust as needed
                    else:
                        _initial_logger.warning("Reload failed: No logging configuration found. Using defaults.")
                        self._config = LoggingConfig() # Fallback to defaults on reload failure

                # Reconfigure using the new settings
                self._configure_from_settings()

                # Update references in existing TradingLogger wrappers if they were accessed
                # during the update process (though ideally they shouldn't be)
                for name in old_logger_names:
                    if name in self._loggers: # Check if it was re-created
                        python_logger = logging.getLogger(name)
                        # Find if any old wrapper instance exists (less ideal)
                        # A better approach might be to invalidate old wrappers,
                        # but for now, just update the underlying logger.
                        # existing_wrapper = find_old_wrapper(name) # Needs a mechanism
                        # if existing_wrapper:
                        #    existing_wrapper._update_logger(python_logger)
                _initial_logger.info("Logging configuration update complete.")
                self._initialized = True # Mark initialization complete

            except Exception as e:
                _initial_logger.error(f"CRITICAL ERROR updating logging configuration: {e}", exc_info=True)
                # Attempt to restore basic logging
                logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
                self._root_logger = logging.getLogger()
                self._root_logger.error("Fell back to basic logging configuration after update failure.")
                # Should probably mark as not initialized or in a failed state
                self._initialized = False

    def save_config(self, config_obj: LoggingConfig, config_name: str = "logging"):
         """Saves the current logging configuration object."""
         try:
             # Convert config object to dictionary if necessary
             if hasattr(config_obj, 'to_dict'):
                 config_dict = config_obj.to_dict()
             elif hasattr(config_obj, '__dict__'):
                  # Basic conversion, might need refinement for nested objects/enums
                  config_dict = config_obj.__dict__
             else:
                 _initial_logger.error("Cannot convert config object to dictionary for saving.")
                 return

             # --- FIX: Use instance to save ---
             self.config_manager.save_config(config_dict, config_name=config_name)
             # --- FIX END ---
             _initial_logger.info(f"Logging configuration '{config_name}' saved.")
         except Exception as e:
             _initial_logger.error(f"Failed to save logging configuration '{config_name}': {e}")


    def start_performance_tracker(self, context_id: str, component_id: str = None) -> 'PerformanceTracker':
        """
        Start tracking performance for a specific context.

        Args:
            context_id: Identifier for this performance tracking context
            component_id: Component identifier for health monitoring integration

        Returns:
            PerformanceTracker instance
        """
        if not self._initialized:
             _initial_logger.warning("Logger not fully initialized, performance tracking might be limited.")
        tracker = PerformanceTracker(context_id, component_id, self)
        with self._lock:
             self._performance_trackers[context_id] = tracker
        return tracker

    def get_performance_tracker(self, context_id: str) -> Optional['PerformanceTracker']:
        """
        Get an existing performance tracker by ID.

        Args:
            context_id: Identifier for the performance tracking context

        Returns:
            PerformanceTracker instance or None if not found
        """
        with self._lock:
            return self._performance_trackers.get(context_id)

    def get_event_bus(self):
        """Get the event bus instance."""
        return self._event_bus

    def get_health_monitor(self):
        """Get the health monitor instance."""
        # Return the stored instance, which might be None if not available/set
        return self._health_monitor

    def set_health_monitor(self, health_monitor):
        """Set the health monitor instance (for external dependency injection)."""
        with self._lock:
            if HEALTH_MONITOR_AVAILABLE:
                 self._health_monitor = health_monitor
                 self._root_logger.info("Health monitor instance set.")
            else:
                 self._root_logger.warning("Attempted to set health monitor, but integration is not available.")

    def get_logging_config(self) -> Optional[LoggingConfig]:
        """Get the current logging configuration object."""
        return self._config


class TradingLogger:
    """
    Enhanced logger for trading applications with additional context capabilities
    and integration with event bus and health monitoring.
    """

    def __init__(self, name: str, python_logger: logging.Logger, manager: LoggerManager):
        """
        Initialize the trading logger.

        Args:
            name: Logger name
            python_logger: Python logger instance
            manager: LoggerManager instance
        """
        self._name = name
        self._python_logger = python_logger
        self._manager = manager
        # Use a thread-local storage for context to avoid race conditions
        # across threads using the same logger instance.
        self._local_context = threading.local()
        # Initialize context for the current thread
        self._local_context.data = {}
        # Lock for context modification within the same thread (might be overkill)
        self._context_lock = threading.RLock()


    def _update_logger(self, python_logger: logging.Logger):
        """
        Update the underlying Python logger instance. (Use with caution)
        This is primarily for internal use by LoggerManager during config updates.

        Args:
            python_logger: New Python logger instance
        """
        self._python_logger = python_logger

    @property
    def context(self) -> Dict[str, Any]:
        """Get the current context for this thread."""
        if not hasattr(self._local_context, 'data'):
            self._local_context.data = {} # Initialize if missing for thread
        return self._local_context.data

    def set_context(self, **context):
        """
        Set the entire context for the current thread, replacing existing context.

        Args:
            **context: Context key-value pairs
        """
        with self._context_lock:
            self._local_context.data = context

    def update_context(self, **context):
         """
         Update the context for the current thread with new key-value pairs.

         Args:
             **context: Context key-value pairs to add or update
         """
         with self._context_lock:
            if not hasattr(self._local_context, 'data'):
                 self._local_context.data = {}
            self._local_context.data.update(context)

    def clear_context(self, keys: Optional[List[str]] = None):
         """
         Clear specific keys or the entire context for the current thread.

         Args:
             keys: List of keys to remove. If None, clears all context.
         """
         with self._context_lock:
            if not hasattr(self._local_context, 'data'):
                 return # Nothing to clear
            if keys is None:
                 self._local_context.data = {}
            else:
                 for key in keys:
                     self._local_context.data.pop(key, None)

    # --- Context Management with 'with' statement ---
    class ContextManager:
        def __init__(self, logger, context_updates):
            self.logger = logger
            self.context_updates = context_updates
            self.original_context = None

        def __enter__(self):
            # Store original context and update
            self.original_context = self.logger.context.copy()
            self.logger.update_context(**self.context_updates)
            return self.logger # Allows 'as logger:' syntax

        def __exit__(self, exc_type, exc_val, exc_tb):
            # Restore original context
            if self.original_context is not None:
                self.logger.set_context(**self.original_context)
            # Don't suppress exceptions
            return False

    def context_scope(self, **context_updates) -> ContextManager:
        """
        Provides a context manager ('with' statement) to temporarily add context.

        Example:
            with logger.context_scope(trade_id="T123", user="algo"):
                logger.info("Processing trade") # Log will include trade_id and user
            logger.info("Outside scope") # Log will not include trade_id and user (unless set previously)

        Args:
            **context_updates: Context key-value pairs to add for the duration of the block.

        Returns:
            A context manager instance.
        """
        return self.ContextManager(self, context_updates)


    # --- Standard Logging Methods ---

    def debug(self, msg: str, *args, **kwargs):
        """Log a debug message with context."""
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        """Log an info message with context."""
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Log a warning message with context."""
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """Log an error message with context."""
        self._log(logging.ERROR, msg, *args, **kwargs)

        # Report to health monitor if available
        component_id = self.context.get('component_id')
        health_monitor = self._manager.get_health_monitor()
        if component_id and health_monitor and HEALTH_MONITOR_AVAILABLE:
            try:
                error_type = 'unknown'
                exc_info = kwargs.get('exc_info')
                if exc_info:
                    if exc_info is True:
                        exc_info = sys.exc_info()
                    if exc_info and len(exc_info) > 0 and exc_info[0] is not None:
                         if isinstance(exc_info[0], type):
                            error_type = exc_info[0].__name__
                    elif isinstance(exc_info, tuple) and len(exc_info) > 0:
                        if isinstance(exc_info[0], type):
                             error_type = exc_info[0].__name__
                else:
                     error_type = 'logged_error' # Generic type if no exception info

                # Ensure health_monitor has log_error method
                if hasattr(health_monitor, 'log_error'):
                    health_monitor.log_error(
                        component_id=str(component_id), # Ensure string
                        error_type=str(error_type),
                        error_message=str(msg)
                    )
                else:
                    self._python_logger.debug(f"Health monitor instance ({type(health_monitor)}) lacks log_error method.")

            except Exception as e:
                # Don't fail if health monitor integration fails
                self._python_logger.warning(f"Failed to log error to health monitor: {e}", exc_info=True)

    def critical(self, msg: str, *args, **kwargs):
        """Log a critical message with context."""
        self._log(logging.CRITICAL, msg, *args, **kwargs)

        # Always try to publish critical errors to the event bus if available
        event_bus = self._manager.get_event_bus()
        if event_bus and EVENT_BUS_AVAILABLE:
            try:
                error_data = {
                    "message": str(msg),
                    "logger_name": self._name,
                    "context": self.context, # Use thread-local context
                    "timestamp": time.time()
                }

                # Add exception info if available
                exc_info = kwargs.get('exc_info')
                if exc_info:
                    if exc_info is True:
                        exc_info = sys.exc_info()
                    # Ensure exc_info is a valid tuple (type, value, traceback)
                    if isinstance(exc_info, tuple) and len(exc_info) == 3 and exc_info[0] is not None:
                        error_data["exception_type"] = exc_info[0].__name__
                        error_data["exception_message"] = str(exc_info[1])
                        # Format traceback robustly
                        try:
                            tb_list = traceback.format_exception(exc_info[0], exc_info[1], exc_info[2])
                            error_data["traceback"] = "".join(tb_list)
                        except Exception:
                            error_data["traceback"] = "Error formatting traceback"
                    else:
                        error_data["exception_info"] = str(exc_info) # Fallback


                event = Event(
                    topic=EventTopics.SYSTEM_ERROR,
                    data=error_data,
                    priority=EventPriority.CRITICAL, # Use CRITICAL
                    source=self._name
                )

                event_bus.publish(event)
            except Exception as e:
                # Don't fail if event bus integration fails
                self._python_logger.warning(f"Failed to publish critical error to event bus: {e}", exc_info=True)

        # Report to health monitor if available
        component_id = self.context.get('component_id')
        health_monitor = self._manager.get_health_monitor()
        if component_id and health_monitor and HEALTH_MONITOR_AVAILABLE:
            try:
                error_type = 'unknown_critical'
                exc_info = kwargs.get('exc_info')
                if exc_info:
                     if exc_info is True:
                         exc_info = sys.exc_info()
                     if exc_info and len(exc_info) > 0 and exc_info[0] is not None:
                          if isinstance(exc_info[0], type):
                              error_type = exc_info[0].__name__
                     elif isinstance(exc_info, tuple) and len(exc_info) > 0:
                         if isinstance(exc_info[0], type):
                              error_type = exc_info[0].__name__
                else:
                    error_type = 'logged_critical'

                # Ensure health_monitor has log_error method
                if hasattr(health_monitor, 'log_error'):
                    health_monitor.log_error(
                        component_id=str(component_id),
                        error_type=str(error_type),
                        error_message=str(msg)
                        # Consider adding severity=CRITICAL if HM supports it
                    )
                else:
                     self._python_logger.debug(f"Health monitor instance ({type(health_monitor)}) lacks log_error method.")

            except Exception as e:
                # Don't fail if health monitor integration fails
                self._python_logger.warning(f"Failed to log critical error to health monitor: {e}", exc_info=True)

    def exception(self, msg: str, *args, **kwargs):
        """Log an exception with context."""
        kwargs.setdefault('exc_info', True)
        self.error(msg, *args, **kwargs)

    # --- Specialized Logging Methods ---

    def audit(self, msg: str, *args, **kwargs):
        """
        Log an audit message for compliance and record-keeping.

        Args:
            msg: Message to log
            *args: Arguments for message formatting
            **kwargs: Additional logging parameters (e.g., extra={'user': 'x'})
                      Common audit fields like 'user', 'operation', 'target_resource'
                      can be passed in 'extra'.
        """
        # Get the audit logger (handles disabled case)
        audit_logger = self._manager.get_audit_logger()
        # If audit is disabled, audit_logger might be a standard logger; check name
        if audit_logger._name == "audit_disabled":
            return # Do nothing if audit logging is off

        # Combine current context with any specific extra context
        combined_context = self.context.copy() # Start with thread context
        extra = kwargs.pop('extra', {})
        combined_context.update(extra) # Add specific extra info

        # Add standard audit fields if present in extra
        audit_fields = {
            'user': combined_context.get('user', 'unknown'),
            'operation': combined_context.get('operation', 'unknown'),
            'target_resource': combined_context.get('target_resource'),
            'status': combined_context.get('status', 'success') # Default to success
        }
        # Remove None values
        audit_fields = {k: v for k, v in audit_fields.items() if v is not None}

        # Pass audit fields and the rest of combined context within 'extra'
        log_extra = {'audit_fields': audit_fields, 'full_context': combined_context}
        kwargs['extra'] = log_extra

        # Log using the audit logger instance at INFO level
        audit_logger._log(
            logging.INFO, # Audit messages are typically INFO level
            msg,
            *args,
            **kwargs
        )

        # Publish to event bus if configured and available
        event_bus = self._manager.get_event_bus()
        audit_config = self._manager.get_logging_config().audit
        if (event_bus and EVENT_BUS_AVAILABLE and
            audit_config and audit_config.enable_audit_logging and audit_config.publish_audit_events):
            try:
                audit_data = {
                    "message": str(msg % args if args else msg), # Format message
                    "logger_name": self._name, # Originating logger
                    "audit_fields": audit_fields,
                    "full_context": combined_context,
                    "timestamp": time.time()
                }

                event = Event(
                    topic=audit_config.audit_event_topic, # Use configured topic
                    data=audit_data,
                    priority=EventPriority.NORMAL, # Audit events usually normal priority
                    source=self._name
                )

                event_bus.publish(event)
            except Exception as e:
                # Don't fail if event bus integration fails
                self._python_logger.warning(f"Failed to publish audit event: {e}", exc_info=True)

    def metric(self, name: str, value: Union[float, int], unit: str = None, component_id: str = None, tags: Dict[str, str] = None):
        """
        Log a metric value for monitoring and trending.

        Args:
            name: Metric name (e.g., 'orders_processed', 'queue_size')
            value: Metric value (numeric)
            unit: Unit of measurement (e.g., 'count', 'ms', 'bytes')
            component_id: Component identifier for health monitoring. Overrides context.
            tags: Optional dictionary of key-value tags for categorization.
        """
        # Use component_id from argument, then context, then logger name
        effective_component_id = component_id or self.context.get('component_id') or self._name

        # Log message with metric info
        log_msg = f"Metric {name}: {value}"
        if unit:
            log_msg += f" {unit}"
        if tags:
            tags_str = ", ".join(f"{k}={v}" for k, v in tags.items())
            log_msg += f" (Tags: {tags_str})"

        # Log locally (consider making this configurable, maybe DEBUG level?)
        self.info(log_msg) # Log metrics at INFO level by default

        # Report to health monitor if available
        health_monitor = self._manager.get_health_monitor()
        if health_monitor and HEALTH_MONITOR_AVAILABLE:
            # Check if HM supports tags or just metrics dict
            if hasattr(health_monitor, 'log_metric'): # Preferred method if exists
                 try:
                     health_monitor.log_metric(
                         component_id=effective_component_id,
                         metric_name=name,
                         value=value,
                         unit=unit,
                         tags=tags
                     )
                 except Exception as e:
                     self._python_logger.warning(f"Failed to log metric (log_metric) to health monitor: {e}", exc_info=True)

            elif hasattr(health_monitor, 'update_component_health'): # Fallback
                try:
                    # Simplistic approach: pass metric as dict, tags lost
                    metrics_dict = {name: value}
                    health_monitor.update_component_health(
                        component_id=effective_component_id,
                        metrics=metrics_dict
                        # Optionally add status if relevant, e.g., HEALTHY
                    )
                except Exception as e:
                    self._python_logger.warning(f"Failed to log metric (update_component_health) to health monitor: {e}", exc_info=True)
            else:
                 self._python_logger.debug(f"Health monitor instance ({type(health_monitor)}) lacks log_metric or update_component_health.")


        # Publish to event bus if configured and available
        event_bus = self._manager.get_event_bus()
        metrics_config = self._manager.get_logging_config().metrics
        if (event_bus and EVENT_BUS_AVAILABLE and
            metrics_config and metrics_config.publish_metrics_events):
            try:
                metric_data = {
                    "name": name,
                    "value": value,
                    "unit": unit,
                    "tags": tags,
                    "component_id": effective_component_id,
                    "timestamp": time.time(),
                    "logger_name": self._name,
                    "context": self.context # Include logger context
                }

                # Use configured topic format
                topic = metrics_config.metrics_event_topic_format.format(
                    component=effective_component_id,
                    metric=name
                )

                event = Event(
                    topic=topic,
                    data=metric_data,
                    priority=EventPriority.LOW, # Metrics are typically low priority
                    source=self._name
                )

                event_bus.publish(event)
            except Exception as e:
                # Don't fail if event bus integration fails
                self._python_logger.warning(f"Failed to publish metric event: {e}", exc_info=True)

    def log_latency(self, operation: str, duration_ms: float, success: bool = True,
                   component_id: str = None, tags: Dict[str, str] = None):
        """
        Log operation latency for performance monitoring.

        Args:
            operation: Operation name (e.g., 'database_query', 'order_submit')
            duration_ms: Duration in milliseconds
            success: Whether the operation succeeded
            component_id: Component identifier. Overrides context.
            tags: Optional dictionary of key-value tags.
        """
        # Use component_id from argument, then context, then logger name
        effective_component_id = component_id or self.context.get('component_id') or self._name

        # Log message with latency info
        status_str = "completed" if success else "failed"
        log_msg = f"Operation {operation} {status_str} in {duration_ms:.2f} ms"
        if tags:
            tags_str = ", ".join(f"{k}={v}" for k, v in tags.items())
            log_msg += f" (Tags: {tags_str})"

        # Log locally (consider making level configurable based on duration/success)
        log_level = logging.INFO if success else logging.WARNING
        self.log(log_level, log_msg)

        # Report to health monitor if available
        health_monitor = self._manager.get_health_monitor()
        if health_monitor and HEALTH_MONITOR_AVAILABLE:
             if hasattr(health_monitor, 'log_operation'):
                 try:
                     health_monitor.log_operation(
                         component_id=effective_component_id,
                         operation_type=operation,
                         duration_ms=duration_ms,
                         success=success,
                         tags=tags
                     )
                 except Exception as e:
                     self._python_logger.warning(f"Failed to log latency (log_operation) to health monitor: {e}", exc_info=True)
             else:
                  self._python_logger.debug(f"Health monitor instance ({type(health_monitor)}) lacks log_operation method.")


        # Publish to event bus if configured and available
        event_bus = self._manager.get_event_bus()
        latency_config = self._manager.get_logging_config().latency
        if (event_bus and EVENT_BUS_AVAILABLE and
            latency_config and latency_config.publish_latency_events):
            try:
                latency_data = {
                    "operation": operation,
                    "duration_ms": duration_ms,
                    "success": success,
                    "tags": tags,
                    "component_id": effective_component_id,
                    "timestamp": time.time(),
                    "logger_name": self._name,
                    "context": self.context # Include logger context
                }

                 # Use configured topic format
                topic = latency_config.latency_event_topic_format.format(
                    component=effective_component_id,
                    operation=operation
                )

                event = Event(
                    topic=topic,
                    data=latency_data,
                    priority=EventPriority.LOW, # Latency often low priority
                    source=self._name
                )

                event_bus.publish(event)
            except Exception as e:
                # Don't fail if event bus integration fails
                self._python_logger.warning(f"Failed to publish latency event: {e}", exc_info=True)

    # --- Basic Logger Passthrough Methods ---

    def isEnabledFor(self, level):
        """Check if this logger is enabled for the specified level."""
        return self._python_logger.isEnabledFor(level)

    def setLevel(self, level):
        """Set the logging level for this logger."""
        self._python_logger.setLevel(level)

    def log(self, level, msg, *args, **kwargs):
        """Log 'msg % args' with the integer severity 'level'."""
        self._log(level, msg, *args, **kwargs)

    def _log(self, level, msg, *args, **kwargs):
        """
        Internal log method that adds context before passing to Python logger.

        Args:
            level: Logging level
            msg: Message to log
            *args: Arguments for message formatting
            **kwargs: Additional logging parameters
        """
        # Skip if not enabled for this level
        if not self._python_logger.isEnabledFor(level):
            return

        # Add thread-local context to extra if it exists
        current_context = self.context # Gets thread-local context
        if current_context:
            # Ensure 'extra' exists and merge context into it
            extra = kwargs.get('extra', {})
            # Avoid overwriting existing keys in extra with context keys
            # by prioritizing keys already in extra.
            merged_extra = current_context.copy()
            merged_extra.update(extra)
            kwargs['extra'] = merged_extra
        elif 'extra' not in kwargs:
             # Ensure 'extra' exists even if context is empty, simplifies formatter access
             kwargs['extra'] = {}

        # Convert structured data to JSON string if 'data' kwarg provided
        data = kwargs.pop('data', None)
        if data:
            try:
                # Attempt to serialize; handle potential complex objects
                data_str = json.dumps(data, default=str) # Use default=str for safety
                msg = f"{msg} | Data: {data_str}"
            except (TypeError, ValueError) as json_err:
                # If data can't be serialized, include its string representation
                msg = f"{msg} | Data (Serialization Error: {json_err}): {str(data)}"


        # Log the message using the underlying Python logger
        # Pass stacklevel=2 so file/line info refers to caller of debug/info etc.
        kwargs.setdefault('stacklevel', 2)
        self._python_logger.log(level, msg, *args, **kwargs)


class PerformanceTracker:
    """
    Tracks and logs performance metrics for operations using context managers.
    """
    def __init__(self, context_id: str, component_id: Optional[str], manager: LoggerManager):
        """
        Initialize the performance tracker.

        Args:
            context_id: Identifier for this performance context (e.g., request_id, task_name).
            component_id: Component identifier for health monitoring.
            manager: LoggerManager instance.
        """
        self._context_id = context_id
        self._component_id = component_id
        self._manager = manager
        # Use a specific logger for performance messages, often useful to filter/direct them
        self._logger = manager.get_logger(f"performance.{context_id}")
        self._start_time = time.perf_counter() # Use monotonic clock for timing
        self._checkpoints = {}
        self._lock = threading.RLock() # Protect access to checkpoints

        # Log start event using the performance logger
        self._logger.debug(f"Performance tracking started", extra={'perf_context': context_id, 'component_id': component_id})

    def checkpoint(self, name: str):
        """
        Record a time checkpoint relative to the tracker's start.

        Args:
            name: Descriptive name for the checkpoint.
        """
        with self._lock:
            now = time.perf_counter()
            elapsed_ms = (now - self._start_time) * 1000
            self._checkpoints[name] = elapsed_ms
            self._logger.debug(f"Checkpoint '{name}': {elapsed_ms:.2f} ms", extra={'perf_context': self._context_id})

    # --- Context Manager for Timing Operations ---
    class OperationTimer:
        def __init__(self, tracker, operation_name, log_level_success, log_level_failure, tags):
            self.tracker = tracker
            self.operation_name = operation_name
            self.log_level_success = log_level_success
            self.log_level_failure = log_level_failure
            self.tags = tags
            self.start_time = None

        def __enter__(self):
            self.start_time = time.perf_counter()
            # Optionally log timer start at DEBUG level
            # self.tracker._logger.debug(f"Starting timed operation: {self.operation_name}", extra={'perf_context': self.tracker._context_id})
            return self # Allows access to timer instance if needed

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.start_time is None:
                return # Should not happen if __enter__ was called

            end_time = time.perf_counter()
            duration_ms = (end_time - self.start_time) * 1000
            success = exc_type is None # Operation succeeded if no exception occurred

            log_level = self.log_level_success if success else self.log_level_failure

            # Log using the tracker's logger and its specialized method
            self.tracker._logger.log_latency(
                operation=self.operation_name,
                duration_ms=duration_ms,
                success=success,
                component_id=self.tracker._component_id,
                tags=self.tags
            )

            # Optionally log directly at the specified level too (might be redundant)
            # status_str = "completed" if success else "failed"
            # log_msg = f"Timed operation '{self.operation_name}' {status_str} in {duration_ms:.2f} ms"
            # self.tracker._logger.log(log_level, log_msg, exc_info=(exc_type, exc_val, exc_tb) if not success else None, extra={'perf_context': self.tracker._context_id})

            # Don't suppress exceptions
            return False

    def time_operation(self, operation_name: str,
                      log_level_success: int = logging.DEBUG,
                      log_level_failure: int = logging.WARNING,
                      tags: Optional[Dict[str, str]] = None) -> OperationTimer:
        """
        Provides a context manager ('with' statement) to time a block of code.

        Example:
            perf_tracker = logger.manager.start_performance_tracker("request_123")
            with perf_tracker.time_operation("database_query"):
                # Code to time
                db.execute(...)
            perf_tracker.stop()

        Args:
            operation_name: Name of the operation being timed.
            log_level_success: Log level for successful completion.
            log_level_failure: Log level for completion with an exception.
            tags: Optional tags for the latency measurement.

        Returns:
            An OperationTimer context manager instance.
        """
        return self.OperationTimer(self, operation_name, log_level_success, log_level_failure, tags)


    def get_elapsed_ms(self) -> float:
        """Get total elapsed time since the tracker started in milliseconds."""
        return (time.perf_counter() - self._start_time) * 1000

    def get_checkpoint_times_ms(self) -> Dict[str, float]:
        """Get all recorded checkpoint times in milliseconds relative to start."""
        with self._lock:
            return self._checkpoints.copy()

    def summary(self) -> Dict[str, Any]:
        """Generate a summary dictionary of the tracked performance."""
        with self._lock:
            total_elapsed_ms = self.get_elapsed_ms()
            return {
                "context_id": self._context_id,
                "component_id": self._component_id,
                "start_timestamp": self._start_time, # Perf counter value, not wall clock
                "total_elapsed_ms": total_elapsed_ms,
                "checkpoints_ms": self._checkpoints.copy(),
            }

    def log_summary(self, level: int = logging.INFO):
        """Log the performance summary."""
        with self._lock:
            summary_data = self.summary()
            checkpoints_str = ", ".join([f"'{k}': {v:.2f}ms" for k, v in summary_data["checkpoints_ms"].items()])
            if not checkpoints_str:
                checkpoints_str = "None"

            log_msg = (f"Performance Summary for Context '{self._context_id}': "
                       f"Total Time: {summary_data['total_elapsed_ms']:.2f} ms, "
                       f"Checkpoints: [{checkpoints_str}]")

            # Log using the performance logger
            self._logger.log(level, log_msg, extra={'perf_context': self._context_id, 'summary': summary_data})

    def stop(self, log_summary: bool = True, summary_level: int = logging.INFO):
        """
        Stop performance tracking and optionally log a summary.

        Args:
            log_summary: Whether to log a summary upon stopping.
            summary_level: Log level for the summary message.
        """
        with self._lock:
             if log_summary:
                 self.log_summary(level=summary_level)

             # Remove from manager's active trackers
             self._manager._performance_trackers.pop(self._context_id, None)

             # Log stop event
             total_elapsed = self.get_elapsed_ms()
             self._logger.debug(f"Performance tracking stopped ({total_elapsed:.2f} ms elapsed)", extra={'perf_context': self._context_id})



class ModuleFilter(logging.Filter):
    """
    Filter logs based on logger name (module) patterns.
    """
    def __init__(self, include_patterns=None, exclude_patterns=None):
        """
        Initialize the module filter.

        Args:
            include_patterns: List or set of module name patterns to include.
                              Supports exact match and simple prefix wildcard (e.g., 'core.*').
            exclude_patterns: List or set of module name patterns to exclude.
                              Supports exact match and simple prefix wildcard.
        """
        super().__init__()
        self.include_patterns = set(include_patterns) if include_patterns else set()
        self.exclude_patterns = set(exclude_patterns) if exclude_patterns else set()

    def filter(self, record):
        """
        Determine if the log record should be logged.

        Args:
            record: The log record.

        Returns:
            True if the record should be logged, False otherwise.
        """
        logger_name = record.name

        # Exclusion takes priority
        for pattern in self.exclude_patterns:
            if self._matches(logger_name, pattern):
                return False # Exclude if matches exclude pattern

        # If include patterns exist, it MUST match one
        if self.include_patterns:
            for pattern in self.include_patterns:
                if self._matches(logger_name, pattern):
                    return True # Include if matches include pattern
            return False # Exclude if no include pattern matched

        # If no include patterns, default to include (unless excluded above)
        return True

    def _matches(self, name, pattern):
        """Check if a logger name matches a pattern."""
        if pattern.endswith(".*"):
            # Prefix match (e.g., 'core.*' matches 'core.strategy')
            return name.startswith(pattern[:-1]) # Match 'core.'
        else:
            # Exact match
            return name == pattern


class DatabaseLogHandler(logging.Handler):
    """
    Custom handler for logging to a database (SQLite example).
    Needs refinement for production (connection pooling, error handling, async).
    """
    def __init__(self, db_path="logs/log_db.sqlite", table_name="logs"):
        """
        Initialize the database log handler.

        Args:
            db_path: Path to the SQLite database file.
            table_name: Name of the table to store logs.
        """
        super().__init__()
        self.db_path = Path(db_path)
        self.table_name = table_name
        self._connection = None
        self._connect_lock = threading.Lock()
        self._ensure_connection() # Create table on init
        self._queue = [] # Simple buffer for batching (optional)
        self._buffer_limit = 50 # Example buffer size
        self._flush_interval = 5 # Flush every 5 seconds (example)
        self._last_flush_time = time.time()
        # Consider using Queue for thread safety if handler shared across threads
        # import queue
        # self._queue = queue.Queue()


    def _ensure_connection(self):
        """Ensure database connection exists and table is created."""
        if self._connection:
            return
        with self._connect_lock:
            if self._connection: # Double check after acquiring lock
                return
            try:
                import sqlite3
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                # Use WAL mode for better concurrency with SQLite
                self._connection = sqlite3.connect(str(self.db_path), timeout=10.0, check_same_thread=False) # Allow sharing across threads
                self._connection.execute("PRAGMA journal_mode=WAL;")

                cursor = self._connection.cursor()
                cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS "{self.table_name}" (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    logger_name TEXT,
                    level_name TEXT,
                    level_no INTEGER,
                    message TEXT,
                    module TEXT,
                    function TEXT,
                    line INTEGER,
                    thread_id INTEGER,
                    thread_name TEXT,
                    process_id INTEGER,
                    context TEXT,
                    exception TEXT,
                    audit_user TEXT,
                    audit_operation TEXT,
                    audit_target TEXT,
                    audit_status TEXT
                )
                """)
                # Consider adding indexes for faster queries
                cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{self.table_name}_timestamp ON "{self.table_name}" (timestamp);')
                cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{self.table_name}_logger_name ON "{self.table_name}" (logger_name);')
                cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{self.table_name}_level_name ON "{self.table_name}" (level_name);')
                self._connection.commit()
            except Exception as e:
                sys.stderr.write(f"FATAL: Error setting up database log handler '{self.db_path}': {e}\n")
                self._connection = None # Ensure connection is None on failure
                # Potentially raise or switch to a fallback handler

    def _prepare_record_data(self, record):
        """Extract data from log record for database insertion."""
        context_str = None
        if hasattr(record, 'extra') and 'full_context' in record.extra:
            try:
                 context_str = json.dumps(record.extra['full_context'], default=str)
            except:
                 context_str = str(record.extra['full_context'])
        elif hasattr(record, 'context'): # Fallback for older context passing
            try:
                 context_str = json.dumps(record.context, default=str)
            except:
                 context_str = str(record.context)

        exception_str = None
        if record.exc_info:
            try:
                # Use traceback.format_exception for detailed info
                exception_str = "".join(traceback.format_exception(*record.exc_info))
            except:
                # Fallback if formatting fails
                exception_str = str(record.exc_info)


        # Extract audit fields if present
        audit_user = None
        audit_operation = None
        audit_target = None
        audit_status = None
        if hasattr(record, 'extra') and 'audit_fields' in record.extra:
             audit_fields = record.extra['audit_fields']
             audit_user = audit_fields.get('user')
             audit_operation = audit_fields.get('operation')
             audit_target = audit_fields.get('target_resource')
             audit_status = audit_fields.get('status')


        # Prepare tuple for insertion
        return (
            record.created, record.name, record.levelname, record.levelno,
            record.getMessage(), # Get formatted message
            record.module, record.funcName, record.lineno, record.thread,
            record.threadName, record.process, context_str, exception_str,
            audit_user, audit_operation, audit_target, audit_status
        )

    def emit(self, record):
        """
        Queue a log record for database insertion.

        Args:
            record: Log record to emit.
        """
        try:
            record_data = self._prepare_record_data(record)
            with self._connect_lock: # Lock for queue access
                self._queue.append(record_data)

            # Check if buffer limit or time interval reached
            should_flush = False
            with self._connect_lock:
                if len(self._queue) >= self._buffer_limit:
                    should_flush = True
                elif time.time() - self._last_flush_time >= self._flush_interval:
                    should_flush = True

            if should_flush:
                self.flush()

        except Exception:
            self.handleError(record) # Use standard handler error logging


    def flush(self):
        """Write buffered records to the database."""
        records_to_flush = []
        with self._connect_lock:
            if not self._queue:
                return
            records_to_flush = self._queue
            self._queue = [] # Clear buffer immediately
            self._last_flush_time = time.time()

        if not records_to_flush:
            return

        self._ensure_connection()
        if not self._connection:
             sys.stderr.write(f"Cannot flush DB logs: No connection to {self.db_path}\n")
             # Optionally requeue records_to_flush? Needs careful handling
             return

        try:
            cursor = self._connection.cursor()
            sql = (f'INSERT INTO "{self.table_name}" '
                   f'(timestamp, logger_name, level_name, level_no, message, '
                   f'module, function, line, thread_id, thread_name, process_id, '
                   f'context, exception, audit_user, audit_operation, audit_target, audit_status) '
                   f'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)')
            cursor.executemany(sql, records_to_flush)
            self._connection.commit()
        except Exception as e:
            sys.stderr.write(f"Error flushing logs to database '{self.db_path}': {e}\n")
            # How to handle failed records? Log to stderr, try again later?
            # For simplicity, they are lost here. Production needs robustness.
            # self.handleError(...) # Can't call with multiple records easily

    def close(self):
        """Flush buffer and close the database connection."""
        try:
            self.flush() # Flush any remaining records
        finally:
             with self._connect_lock:
                if self._connection:
                    try:
                        self._connection.close()
                    except Exception as e:
                         sys.stderr.write(f"Error closing DB connection '{self.db_path}': {e}\n")
                    finally:
                         self._connection = None
             super().close()


class StructuredFormatter(logging.Formatter):
    """
    Formatter that produces structured log messages, typically in JSON format.
    Handles context and audit fields.
    """
    def __init__(self, fmt=None, datefmt=None, style='%',
                 include_fields: Optional[List[str]] = None,
                 exclude_fields: Optional[List[str]] = None,
                 json_ensure_ascii=True, json_default=str):
        """
        Initialize the structured formatter.

        Args:
            fmt: Format string (ignored in favor of structured output).
            datefmt: Date format string for the 'timestamp_iso' field.
            style: Style character (ignored).
            include_fields: List of top-level fields to include (None for defaults).
            exclude_fields: List of top-level fields to explicitly exclude.
            json_ensure_ascii: Passed to json.dumps ensure_ascii.
            json_default: Passed to json.dumps default serializer.
        """
        # We don't use the standard fmt string, but call super for datefmt handling
        super().__init__(fmt=None, datefmt=datefmt, style=style)

        self.json_ensure_ascii = json_ensure_ascii
        self.json_default = json_default # Function to handle non-serializable types

        # Define default fields if include_fields is None
        default_fields = [
            'timestamp_unix', 'timestamp_iso', 'level', 'logger', 'message',
            'module', 'function', 'line', 'thread_id', 'thread_name', 'process_id',
            'context', 'audit', 'exception' # Add standard fields
        ]
        self.include_fields = set(include_fields) if include_fields else set(default_fields)
        self.exclude_fields = set(exclude_fields) if exclude_fields else set()


    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as a JSON string.

        Args:
            record: The log record instance.

        Returns:
            A JSON string representing the log record.
        """
        log_entry = {}

        # Basic LogRecord attributes
        log_entry['timestamp_unix'] = record.created
        log_entry['timestamp_iso'] = self.formatTime(record, self.datefmt)
        log_entry['level'] = record.levelname
        log_entry['level_no'] = record.levelno
        log_entry['logger'] = record.name
        log_entry['message'] = record.getMessage() # Formatted message
        log_entry['module'] = record.pathname # Use pathname for full path
        log_entry['function'] = record.funcName
        log_entry['line'] = record.lineno
        log_entry['thread_id'] = record.thread
        log_entry['thread_name'] = record.threadName
        log_entry['process_id'] = record.process

        # Add context if available in 'extra'
        if hasattr(record, 'extra'):
            # Look for 'full_context' first, then 'context'
            if 'full_context' in record.extra:
                log_entry['context'] = record.extra['full_context']
            elif 'context' in record.extra:
                log_entry['context'] = record.extra['context']

        # Add audit fields if available in 'extra'
        if hasattr(record, 'extra') and 'audit_fields' in record.extra:
            log_entry['audit'] = record.extra['audit_fields']

        # Add exception info if present
        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            try:
                 # Format traceback into a list of strings or a single string
                 tb_list = traceback.format_exception(exc_type, exc_value, exc_tb)
                 log_entry['exception'] = {
                      'type': exc_type.__name__ if exc_type else None,
                      'message': str(exc_value),
                      'traceback': "".join(tb_list) # Single string traceback
                 }
            except Exception:
                 # Fallback if formatting fails
                 log_entry['exception'] = {
                      'type': str(exc_type),
                      'message': str(exc_value),
                      'traceback': 'Error formatting traceback'
                 }


        # Filter fields: Apply include/exclude logic
        # Start with all generated fields
        final_log_entry = {}
        for key, value in log_entry.items():
             # Check if included (if include list exists)
             include_check = key in self.include_fields if self.include_fields else True
             # Check if excluded
             exclude_check = key in self.exclude_fields

             if include_check and not exclude_check:
                 final_log_entry[key] = value

        # Serialize to JSON
        try:
            return json.dumps(final_log_entry, ensure_ascii=self.json_ensure_ascii, default=self.json_default)
        except Exception as e:
            # Fallback in case of unexpected serialization errors
            try:
                return json.dumps({
                    "timestamp_iso": self.formatTime(record, self.datefmt),
                    "level": "ERROR",
                    "logger": "StructuredFormatter",
                    "message": "Failed to serialize log record",
                    "original_record": str(final_log_entry), # Try string representation
                    "error": str(e)
                })
            except Exception:
                return '{"level": "ERROR", "logger": "StructuredFormatter", "message": "Failed to serialize log record and fallback."}'


# --- Decorator ---

def log_latency(logger_name: Optional[str] = None, operation: Optional[str] = None,
                success_level: int = logging.DEBUG, # Default to DEBUG for successful ops
                error_level: int = logging.WARNING,
                include_args: bool = False, include_result: bool = False,
                args_kwargs_max_len: int = 100, result_max_len: int = 100,
                threshold_ms: Optional[float] = None,
                tags: Optional[Dict[str, str]] = None):
    """
    Decorator to log function execution time and optionally arguments/results.

    Args:
        logger_name: Name of the logger. Defaults to the decorated function's module.
        operation: Name of the operation. Defaults to the decorated function's name.
        success_level: Log level for successful executions.
        error_level: Log level for failed executions (exceptions).
        include_args: Log function arguments (potentially truncated).
        include_result: Log function result (potentially truncated).
        args_kwargs_max_len: Max length for string representation of args/kwargs.
        result_max_len: Max length for string representation of the result.
        threshold_ms: Only log if duration exceeds this threshold (milliseconds).
        tags: Optional dictionary of tags to add to the latency log record.

    Returns:
        Decorated function.
    """
    def decorator(func):
        # Determine logger and operation name only once when decorating
        actual_logger_name = logger_name
        if not actual_logger_name:
            module = inspect.getmodule(func)
            actual_logger_name = module.__name__ if module else "unknown_module"

        actual_operation = operation or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(actual_logger_name) # Get logger instance each call
            start_time = time.perf_counter()
            result = None
            success = False
            exc_info = None

            try:
                result = func(*args, **kwargs)
                success = True
                return result # Return result before logging success
            except Exception as e:
                exc_info = sys.exc_info() # Capture exception info
                raise # Re-raise exception after logging
            finally:
                # This block executes whether an exception occurred or not
                duration_ms = (time.perf_counter() - start_time) * 1000

                # Check threshold
                if threshold_ms is None or duration_ms >= threshold_ms:
                    log_level = success_level if success else error_level
                    log_msg_parts = [f"Operation '{actual_operation}'"]

                    if include_args:
                         try:
                             args_repr = repr(args)
                             kwargs_repr = repr(kwargs)
                             args_str = args_repr[:args_kwargs_max_len] + ('...' if len(args_repr) > args_kwargs_max_len else '')
                             kwargs_str = kwargs_repr[:args_kwargs_max_len] + ('...' if len(kwargs_repr) > args_kwargs_max_len else '')
                             log_msg_parts.append(f"args={args_str} kwargs={kwargs_str}")
                         except Exception:
                             log_msg_parts.append("args/kwargs=(error representing)")

                    status_str = "completed" if success else "failed"
                    log_msg_parts.append(f"{status_str} in {duration_ms:.2f} ms")

                    if success and include_result:
                         try:
                             result_repr = repr(result)
                             result_str = result_repr[:result_max_len] + ('...' if len(result_repr) > result_max_len else '')
                             log_msg_parts.append(f"result={result_str}")
                         except Exception:
                             log_msg_parts.append("result=(error representing)")

                    if not success and exc_info:
                        log_msg_parts.append(f"error={exc_info[0].__name__}: {exc_info[1]}")


                    # Log the combined message
                    final_log_msg = " | ".join(log_msg_parts)
                    # Pass exc_info only if an exception actually occurred
                    log_exc_info = exc_info if not success else None
                    logger.log(log_level, final_log_msg, exc_info=log_exc_info)

                    # Log structured latency data
                    logger.log_latency(actual_operation, duration_ms, success, tags=tags)

        return wrapper
    return decorator


# --- Convenience Functions ---

def create_component_logger(component_id: str, component_type: Optional[str] = None,
                            base_logger_name: Optional[str] = None,
                            context: Optional[Dict[str, Any]] = None) -> TradingLogger:
    """
    Factory function to create a logger pre-configured for a specific system component.

    Args:
        component_id: Unique identifier for the component instance (e.g., 'strategy_MA_cross_1').
        component_type: Type of the component (e.g., 'strategy', 'execution_handler').
        base_logger_name: Base name for the logger (e.g., 'components'). Defaults to component_type or 'component'.
        context: Additional context key-value pairs to add to the logger.

    Returns:
        A TradingLogger instance with component context.
    """
    # Determine the full logger name
    base_name = base_logger_name or component_type or "component"
    logger_name = f"{base_name}.{component_id}"

    logger = get_logger(logger_name)

    # Prepare component context
    component_context = {'component_id': component_id}
    if component_type:
        component_context['component_type'] = component_type
    if context:
        component_context.update(context)

    # Set the initial context for this logger instance
    logger.set_context(**component_context)

    # Register component with health monitor if HM is available and set
    health_monitor = LoggerManager().get_health_monitor()
    if health_monitor and HEALTH_MONITOR_AVAILABLE and hasattr(health_monitor, 'register_component'):
        try:
            # Ensure register_component exists and call it
            health_monitor.register_component(component_id, component_type)
            logger.debug(f"Component '{component_id}' registered with health monitor.")
        except Exception as e:
            logger.warning(f"Failed to register component '{component_id}' with health monitor: {e}", exc_info=True)

    return logger


def get_logger(name: Optional[str] = None) -> TradingLogger:
    """
    Get a logger instance, inferring name from caller if not provided.

    Args:
        name: Logger name. If None, infers from the caller's module.

    Returns:
        TradingLogger instance.
    """
    if name is None:
        try:
            # Go back 1 frame to find the caller
            frame = inspect.currentframe().f_back
            name = frame.f_globals['__name__']
        except Exception:
            name = "root" # Fallback if inspection fails

    return LoggerManager().get_logger(name)


def track_performance(context_id: Optional[str] = None, component_id: Optional[str] = None) -> PerformanceTracker:
    """
    Create a performance tracker, inferring context ID from caller if not provided.

    Args:
        context_id: Identifier for the performance context. Defaults to caller's module.function.
        component_id: Component identifier for health monitoring integration.

    Returns:
        PerformanceTracker instance.
    """
    if context_id is None:
        try:
             # Go back 1 frame to find the caller
            frame = inspect.currentframe().f_back
            module_name = frame.f_globals['__name__']
            function_name = frame.f_code.co_name
            context_id = f"{module_name}.{function_name}"
        except Exception:
            context_id = "unknown_context" # Fallback

    return LoggerManager().start_performance_tracker(context_id, component_id)


# Global configure function remains, using LoggerManager's update method
def configure_logging(config_dict: Optional[Dict[str, Any]] = None):
    """
    Configure or reconfigure the logging system using a dictionary.

    Args:
        config_dict: A dictionary conforming to logging configuration schema,
                     or None to reload configuration from the default source.

    Returns:
        The effective LoggingConfig object used (or None if config failed).
    """
    try:
        manager = LoggerManager() # Ensures manager is initialized
        manager.update_configuration(config_dict)
        return manager.get_logging_config()
    except Exception as e:
         # Log critical failure to stderr as logging might be broken
         sys.stderr.write(f"CRITICAL FAILURE during configure_logging: {e}\n{traceback.format_exc()}")
         return None


# --- Initialize logging system at module import time ---
# This ensures LoggerManager() is called, which triggers __init__ and the
# initial configuration based on loaded config or defaults.
try:
    _manager_instance = LoggerManager()
    _initial_logger.info("LoggerManager initialized globally.")
except Exception as e:
    sys.stderr.write(f"Failed to initialize LoggerManager globally: {e}\n{traceback.format_exc()}")
    # Ensure basic logging is available even if manager init fails
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    logging.getLogger(__name__).critical("Global LoggerManager initialization failed. Using basic logging.")