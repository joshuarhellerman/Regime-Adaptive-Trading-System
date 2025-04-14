"""
system.py - Bootstrap Coordinator for Trading System

This module serves as the central coordinator for the trading system,
handling component initialization, lifecycle management, and system-wide operations.
It acts as a bootstrap mechanism for the entire platform.
"""

import logging
import threading
import time
import os
import psutil
from typing import Dict, Any, Optional, List, Tuple, Callable

from core.event_bus import EventTopics, create_event, get_event_bus, Event
from core.state_manager import get_state_manager, StateScope
from core.component_registry import get_component_registry
from data.market_data_service import get_market_data_service

logger = logging.getLogger(__name__)


class System:
    """
    Main system class that manages all components and trading operations.
    This class serves as the central controller for the trading system.
    """

    def __init__(self, config: Dict[str, Any], mode: str = "paper_trading"):
        """
        Initialize the system.

        Args:
            config: System configuration
            mode: Operation mode ("paper_trading", "live_trading", "backtest")
        """
        self.config = config
        self.mode = mode
        self.is_running = False
        self.start_time = None
        self.shutdown_requested = False
        self._startup_lock = threading.RLock()
        self._shutdown_lock = threading.RLock()

        # Initialize core services
        logger.info("Initializing core services")
        self.event_bus = get_event_bus()
        self.state_manager = get_state_manager()
        self.component_registry = get_component_registry()

        # Register system-level event handlers
        self.event_bus.subscribe(EventTopics.SYSTEM_SHUTDOWN, self._handle_shutdown_request)
        self.event_bus.subscribe(EventTopics.COMPONENT_ERROR, self._handle_component_error)

        # Set system state
        self.state_manager.set("system.mode", mode, StateScope.PERSISTENT)
        self.state_manager.set("system.config", config, StateScope.PERSISTENT)
        self.state_manager.set("system.status", "initialized", StateScope.PERSISTENT)

        logger.info(f"System initialized in {mode} mode")

    def start(self) -> bool:
        """
        Start the system and all registered components.

        Returns:
            Whether the system was successfully started
        """
        with self._startup_lock:
            if self.is_running:
                logger.warning("System is already running")
                return True

            try:
                logger.info(f"Starting system in {self.mode} mode")
                self.state_manager.set("system.status", "starting", StateScope.PERSISTENT)

                # Start core services
                logger.info("Starting core services")
                self.event_bus.start()
                self.state_manager.start()

                # Initialize market data service
                try:
                    self.market_data_service = get_market_data_service()
                    # Register with component registry
                    self.component_registry.register(
                        self.market_data_service,
                        "market_data_service",
                        "data_service"
                    )
                except Exception as e:
                    logger.error(f"Failed to initialize market data service: {str(e)}")
                    # Continue startup process even if market data service fails

                # Start all components through component registry
                logger.info("Starting registered components")
                if not self.component_registry.start_all():
                    logger.warning("Some components failed to start")

                # Set start time and running flag
                self.start_time = time.time()
                self.is_running = True
                self.shutdown_requested = False

                # Update system state
                self.state_manager.set("system.status", "running", StateScope.PERSISTENT)
                self.state_manager.set("system.start_time", self.start_time, StateScope.PERSISTENT)

                # Publish system start event
                event = create_event(
                    EventTopics.SYSTEM_START,
                    {
                        "mode": self.mode,
                        "time": self.start_time,
                        "config": self.config
                    }
                )
                self.event_bus.publish(event)

                logger.info(f"System started in {self.mode} mode")
                return True

            except Exception as e:
                logger.error(f"Error starting system: {str(e)}")

                # Attempt to stop any components that might have started
                try:
                    self.stop()
                except Exception as stop_error:
                    logger.error(f"Error during emergency shutdown: {str(stop_error)}")

                # Update system state
                self.state_manager.set("system.status", "error", StateScope.PERSISTENT)
                self.state_manager.set("system.error", str(e), StateScope.PERSISTENT)

                return False

    def stop(self) -> bool:
        """
        Stop the system and all registered components.

        Returns:
            Whether the system was successfully stopped
        """
        with self._shutdown_lock:
            if not self.is_running:
                logger.warning("System is not running")
                return True

            try:
                logger.info("Stopping system")
                self.state_manager.set("system.status", "stopping", StateScope.PERSISTENT)

                # Calculate uptime
                uptime = time.time() - self.start_time if self.start_time else 0

                # Publish system stop event
                event = create_event(
                    EventTopics.SYSTEM_SHUTDOWN,
                    {
                        "mode": self.mode,
                        "uptime": uptime,
                        "clean_shutdown": not self.shutdown_requested
                    }
                )
                self.event_bus.publish_sync(event)

                # Stop all components through component registry
                logger.info("Stopping registered components")
                self.component_registry.stop_all()

                # Stop core services (in reverse order)
                logger.info("Stopping core services")
                self.state_manager.stop()
                self.event_bus.stop()

                # Set running flag
                self.is_running = False

                # Update system state
                self.state_manager.set("system.status", "stopped", StateScope.PERSISTENT)
                self.state_manager.set("system.end_time", time.time(), StateScope.PERSISTENT)
                self.state_manager.set("system.uptime", uptime, StateScope.PERSISTENT)

                logger.info("System stopped")
                return True

            except Exception as e:
                logger.error(f"Error stopping system: {str(e)}")

                # Update system state
                self.state_manager.set("system.status", "error", StateScope.PERSISTENT)
                self.state_manager.set("system.error", str(e), StateScope.PERSISTENT)

                # Force stop in case of error
                self.is_running = False

                return False

    def restart(self) -> bool:
        """
        Restart the system.

        Returns:
            Whether the system was successfully restarted
        """
        logger.info("Restarting system")

        # Stop the system
        if not self.stop():
            logger.error("Failed to stop system during restart")
            return False

        # Short delay to ensure everything is properly stopped
        time.sleep(2)

        # Start the system
        return self.start()

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current system status.

        Returns:
            Dictionary with system status information
        """
        status = {
            "mode": self.mode,
            "is_running": self.is_running,
            "uptime": time.time() - self.start_time if self.start_time and self.is_running else 0,
            "components": {},
            "system_resources": self._get_system_resources()
        }

        # Get component status from registry
        for name in self.component_registry.get_component_names():
            component_info = self.component_registry.get_component_info(name)
            if component_info:
                status["components"][name] = component_info

        return status

    def _get_system_resources(self) -> Dict[str, Any]:
        """
        Get system resource usage information.

        Returns:
            Dictionary with resource usage information
        """
        resources = {}

        try:
            # CPU usage
            resources["cpu_percent"] = psutil.cpu_percent(interval=0.1)

            # Memory usage
            memory = psutil.virtual_memory()
            resources["memory"] = {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used
            }

            # Disk usage
            disk = psutil.disk_usage('/')
            resources["disk"] = {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent
            }

            # Network information (just connections count for now)
            resources["network"] = {
                "connections": len(psutil.net_connections())
            }

            # Process information
            process = psutil.Process()
            resources["process"] = {
                "cpu_percent": process.cpu_percent(interval=0.1),
                "memory_percent": process.memory_percent(),
                "threads": process.num_threads()
            }

        except Exception as e:
            resources["error"] = str(e)

        return resources

    def execute_command(self, command: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a system command.

        Args:
            command: Command to execute
            params: Command parameters

        Returns:
            Command execution result
        """
        params = params or {}
        result = {"success": False, "command": command}

        try:
            if command == "start":
                result["success"] = self.start()

            elif command == "stop":
                result["success"] = self.stop()

            elif command == "restart":
                result["success"] = self.restart()

            elif command == "status":
                result["success"] = True
                result["status"] = self.get_status()

            elif command == "component_command":
                # Execute a command on a specific component
                component_name = params.get("component")
                component_command = params.get("component_command")
                component_params = params.get("component_params", {})

                if not component_name or not component_command:
                    result["error"] = "Component name and command are required"
                    return result

                # Get the component
                component = self.component_registry.get(component_name)
                if not component:
                    result["error"] = f"Component not found: {component_name}"
                    return result

                # Execute the command on the component
                if hasattr(component, 'execute_command') and callable(getattr(component, 'execute_command')):
                    component_result = component.execute_command(component_command, component_params)
                    result["success"] = component_result.get("success", False)
                    result["component_result"] = component_result
                else:
                    result["error"] = f"Component does not support command execution: {component_name}"

            else:
                result["error"] = f"Unknown command: {command}"

        except Exception as e:
            result["error"] = str(e)

        return result

    def register_component(self, name: str, component: Any, component_type: str = None,
                          dependencies: List[str] = None) -> bool:
        """
        Register a new component with the system.

        Args:
            name: Component name
            component: Component object
            component_type: Component type
            dependencies: List of dependencies

        Returns:
            Whether the component was successfully registered
        """
        try:
            # Register with component registry
            self.component_registry.register(component, name, component_type, dependencies)

            # Start component if system is already running
            if self.is_running:
                self.component_registry.start(name)

            logger.info(f"Registered component: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to register component {name}: {str(e)}")
            return False

    def unregister_component(self, name: str) -> bool:
        """
        Unregister a component from the system.

        Args:
            name: Component name

        Returns:
            Whether the component was successfully unregistered
        """
        try:
            # Stop component if running
            if self.is_running:
                self.component_registry.stop(name)

            # Unregister from component registry
            result = self.component_registry.unregister(name)

            if result:
                logger.info(f"Unregistered component: {name}")
            else:
                logger.warning(f"Failed to unregister component: {name}")

            return result

        except Exception as e:
            logger.error(f"Error unregistering component {name}: {str(e)}")
            return False

    def get_components(self) -> Dict[str, Any]:
        """
        Get all registered components.

        Returns:
            Dictionary of registered components
        """
        components = {}

        for name in self.component_registry.get_component_names():
            components[name] = self.component_registry.get(name)

        return components

    def get_component(self, name: str) -> Optional[Any]:
        """
        Get a specific component by name.

        Args:
            name: Component name

        Returns:
            Component object or None if not found
        """
        return self.component_registry.get(name)

    def _handle_shutdown_request(self, event: Event) -> None:
        """
        Handle system shutdown request event.

        Args:
            event: Shutdown event
        """
        if self.shutdown_requested:
            logger.info("Shutdown already in progress, ignoring additional request")
            return

        logger.info("Received shutdown request")
        self.shutdown_requested = True

        # Schedule shutdown in a separate thread to not block event processing
        threading.Thread(target=self.stop, daemon=True).start()

    def _handle_component_error(self, event: Event) -> None:
        """
        Handle component error event.

        Args:
            event: Component error event
        """
        error_data = event.data
        component_name = error_data.get('name')
        error_message = error_data.get('error')

        logger.error(f"Component error: {component_name} - {error_message}")

        # Record error in state manager
        self.state_manager.set(f"system.component_errors.{component_name}", {
            "message": error_message,
            "timestamp": time.time()
        }, StateScope.PERSISTENT)

        # Check for critical components and decide if system should shut down
        critical_components = self.config.get("critical_components", [])
        if component_name in critical_components:
            logger.critical(f"Critical component {component_name} failed, initiating system shutdown")

            # Schedule shutdown in a separate thread to not block event processing
            self.shutdown_requested = True
            threading.Thread(target=self.stop, daemon=True).start()