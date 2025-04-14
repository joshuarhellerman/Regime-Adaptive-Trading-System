"""
component_registry.py - Dynamic Component Management System

This module implements a centralized registry for components with dependency tracking,
lifecycle management, and dynamic discovery. It ensures components are started and
stopped in the correct order based on their dependencies.
"""

import logging
import threading
import time
import inspect
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Tuple, Callable, Union
import networkx as nx

from core.event_bus import EventTopics, create_event, get_event_bus, Event, EventPriority

# Configure logger
logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """Status of a registered component"""
    REGISTERED = "registered"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class ComponentRegistry:
    """
    Central registry for system components with dependency management.

    The component registry maintains references to all registered components,
    tracks their dependencies, and manages their lifecycle (starting/stopping
    in the correct order).
    """

    def __init__(self):
        """Initialize the component registry"""
        self._components: Dict[str, Any] = {}
        self._component_info: Dict[str, Dict[str, Any]] = {}
        self._dependencies: Dict[str, List[str]] = {}  # Component -> List of dependencies
        self._dependents: Dict[str, List[str]] = {}    # Component -> List of components that depend on it
        self._status: Dict[str, ComponentStatus] = {}
        self._lock = threading.RLock()
        self._event_bus = get_event_bus()
        self._max_start_wait = 30.0  # Maximum time to wait for dependencies (seconds)
        self._max_stop_wait = 15.0   # Maximum time to wait for dependents (seconds)
        self._startup_complete = False
        self._shutdown_in_progress = False

    def register(self, component: Any, name: str, component_type: Optional[str] = None,
                dependencies: Optional[List[str]] = None) -> bool:
        """
        Register a component with the registry.

        Args:
            component: The component object
            name: A unique name for the component
            component_type: Type/category of the component (optional)
            dependencies: List of component names this component depends on

        Returns:
            Whether registration was successful

        Raises:
            ValueError: If the component name is already registered
        """
        with self._lock:
            # Check if component name already exists
            if name in self._components:
                logger.error(f"Component with name '{name}' already registered")
                raise ValueError(f"Component with name '{name}' already registered")

            # Register component
            self._components[name] = component

            # Initialize empty lists for this component in dependency tracking
            dependencies = dependencies or []
            self._dependencies[name] = dependencies.copy()
            self._dependents[name] = []

            # Update dependents of all dependencies
            for dep in dependencies:
                if dep in self._dependents:
                    self._dependents[dep].append(name)
                else:
                    # Create entry for dependency that hasn't been registered yet
                    self._dependents[dep] = [name]

            # Create component info
            self._component_info[name] = {
                'name': name,
                'type': component_type or 'unknown',
                'class': component.__class__.__name__,
                'module': component.__class__.__module__,
                'registration_time': time.time(),
                'dependencies': dependencies.copy(),
                'version': getattr(component, 'version', '1.0.0'),
                'description': getattr(component, 'description', ''),
                'methods': [method for method in dir(component)
                            if callable(getattr(component, method)) and not method.startswith('_')],
                'has_start': hasattr(component, 'start') and callable(getattr(component, 'start')),
                'has_stop': hasattr(component, 'stop') and callable(getattr(component, 'stop')),
                'has_status': hasattr(component, 'get_status') and callable(getattr(component, 'get_status')),
                'has_execute_command': hasattr(component, 'execute_command') and callable(getattr(component, 'execute_command'))
            }

            # Set initial status
            self._status[name] = ComponentStatus.REGISTERED

            # Publish registration event
            event = create_event(
                EventTopics.COMPONENT_REGISTERED,
                {
                    'name': name,
                    'type': component_type,
                    'timestamp': time.time(),
                    'dependencies': dependencies
                }
            )
            self._event_bus.publish(event)

            logger.info(f"Component '{name}' registered (type: {component_type})")
            return True

    def unregister(self, name: str) -> bool:
        """
        Unregister a component.

        Args:
            name: Name of the component to unregister

        Returns:
            Whether unregistration was successful
        """
        with self._lock:
            if name not in self._components:
                logger.warning(f"Cannot unregister component '{name}': not found")
                return False

            # Check if component is running
            if self._status[name] == ComponentStatus.RUNNING:
                logger.warning(f"Cannot unregister running component '{name}'. Stop it first.")
                return False

            # Check for dependents
            if self._dependents[name]:
                logger.warning(f"Cannot unregister component '{name}' with dependents: {self._dependents[name]}")
                return False

            # Remove component from dependencies of other components
            for component, deps in self._dependencies.items():
                if name in deps:
                    self._dependencies[component].remove(name)

            # Clean up dependency tracking
            for dep in self._dependencies[name]:
                if dep in self._dependents and name in self._dependents[dep]:
                    self._dependents[dep].remove(name)

            # Remove component
            del self._components[name]
            del self._component_info[name]
            del self._dependencies[name]
            del self._dependents[name]
            del self._status[name]

            logger.info(f"Component '{name}' unregistered")
            return True

    def get(self, name: str) -> Optional[Any]:
        """
        Get a component by name.

        Args:
            name: The name of the component

        Returns:
            The component object or None if not found
        """
        return self._components.get(name)

    def get_component_names(self) -> List[str]:
        """
        Get list of all registered component names.

        Returns:
            List of component names
        """
        with self._lock:
            return list(self._components.keys())

    def get_component_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a component.

        Args:
            name: The name of the component

        Returns:
            Dictionary of component information or None if not found
        """
        with self._lock:
            if name not in self._component_info:
                return None

            info = self._component_info[name].copy()

            # Add current status
            info['status'] = self._status[name].value

            # Add runtime information if available
            component = self._components.get(name)
            if component and self._status[name] == ComponentStatus.RUNNING:
                # Get component status if available
                if info['has_status']:
                    try:
                        status = component.get_status()
                        if isinstance(status, dict):
                            info['runtime'] = status
                    except Exception as e:
                        logger.error(f"Error getting status for component '{name}': {e}")

                # Add runtime info
                info['running_since'] = info.get('start_time', 0)
                info['uptime'] = time.time() - info.get('start_time', time.time())

            return info

    def get_all_component_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all components.

        Returns:
            Dictionary mapping component names to their information
        """
        with self._lock:
            result = {}
            for name in self._components:
                result[name] = self.get_component_info(name)
            return result

    def get_status(self, name: str) -> Optional[ComponentStatus]:
        """
        Get the status of a component.

        Args:
            name: The name of the component

        Returns:
            The component status or None if not found
        """
        with self._lock:
            return self._status.get(name)

    def get_dependency_graph(self) -> Dict[str, Any]:
        """
        Get a representation of the dependency graph.

        Returns:
            Dictionary with nodes and edges representing the dependency graph
        """
        with self._lock:
            nodes = []
            edges = []

            # Create nodes with status
            for name, status in self._status.items():
                component_type = self._component_info[name].get('type', 'unknown')
                nodes.append({
                    'id': name,
                    'label': name,
                    'status': status.value,
                    'type': component_type
                })

            # Create edges from dependencies
            for component, deps in self._dependencies.items():
                for dep in deps:
                    edges.append({
                        'from': dep,
                        'to': component
                    })

            return {
                'nodes': nodes,
                'edges': edges
            }

    def _get_dependency_ordering(self) -> List[List[str]]:
        """
        Get components grouped by dependency level for ordered startup/shutdown.

        Returns:
            List of lists of component names ordered by dependency level
        """
        with self._lock:
            # Build a directed graph of dependencies
            G = nx.DiGraph()

            # Add all nodes
            for name in self._components:
                G.add_node(name)

            # Add all edges (dependency -> component)
            for component, deps in self._dependencies.items():
                for dep in deps:
                    if dep in self._components:
                        G.add_edge(dep, component)

            # Check for cycles
            try:
                cycles = list(nx.simple_cycles(G))
                if cycles:
                    logger.error(f"Circular dependencies detected: {cycles}")
                    raise ValueError(f"Circular dependencies detected: {cycles}")
            except nx.NetworkXNoCycle:
                # No cycles found, this is good
                pass

            # Get topological generations (components grouped by dependency level)
            try:
                generations = list(nx.topological_generations(G))
                return generations
            except nx.NetworkXUnfeasible as e:
                logger.error(f"Cannot determine dependency ordering: {e}")
                # Fallback: just return all components in a single group
                return [list(self._components.keys())]
            except Exception as e:
                logger.error(f"Error determining dependency ordering: {e}")
                return [list(self._components.keys())]

    def start(self, name: str, wait_for_dependencies: bool = True, timeout: float = None) -> bool:
        """
        Start a specific component.

        Args:
            name: The name of the component to start
            wait_for_dependencies: Whether to wait for dependencies to start
            timeout: Maximum time to wait for dependencies (seconds)

        Returns:
            Whether the component was successfully started
        """
        if timeout is None:
            timeout = self._max_start_wait

        with self._lock:
            # Check if component exists
            if name not in self._components:
                logger.error(f"Cannot start component '{name}': not found")
                return False

            # Check if component is already running
            if self._status[name] == ComponentStatus.RUNNING:
                logger.debug(f"Component '{name}' is already running")
                return True

            # Check if component is in the process of starting
            if self._status[name] == ComponentStatus.STARTING:
                logger.debug(f"Component '{name}' is already starting")
                return True

            # Get component
            component = self._components[name]
            start_method = getattr(component, 'start', None)

            # Check if component has a start method
            if not start_method or not callable(start_method):
                logger.warning(f"Component '{name}' does not have a start method")
                # Update status to RUNNING since we can't actually start it
                self._status[name] = ComponentStatus.RUNNING
                self._component_info[name]['start_time'] = time.time()
                return True

            # Check dependencies
            dependencies = self._dependencies[name]
            unmet_dependencies = [dep for dep in dependencies
                                 if dep in self._components and
                                 self._status.get(dep) != ComponentStatus.RUNNING]

            if unmet_dependencies:
                logger.info(f"Component '{name}' has unmet dependencies: {unmet_dependencies}")

                if wait_for_dependencies:
                    # Start dependencies first (no need to check for cycles as we checked during ordering)
                    for dep in unmet_dependencies:
                        if not self.start(dep, wait_for_dependencies=True, timeout=timeout):
                            logger.error(f"Failed to start dependency '{dep}' for component '{name}'")
                            return False
                else:
                    logger.warning(f"Cannot start component '{name}' due to unmet dependencies")
                    return False

            # Update status to STARTING
            self._status[name] = ComponentStatus.STARTING
            logger.info(f"Starting component '{name}'")

            try:
                # Call start method
                start_result = start_method()

                # Update status based on result
                if start_result is False:  # Explicit False return
                    self._status[name] = ComponentStatus.ERROR
                    logger.error(f"Component '{name}' start method returned False")

                    # Publish event
                    event = create_event(
                        EventTopics.COMPONENT_ERROR,
                        {
                            'name': name,
                            'error': "Start method returned False",
                            'timestamp': time.time()
                        },
                        priority=EventPriority.HIGH
                    )
                    self._event_bus.publish(event)
                    return False
                else:
                    self._status[name] = ComponentStatus.RUNNING
                    self._component_info[name]['start_time'] = time.time()

                    # Publish event
                    event = create_event(
                        EventTopics.COMPONENT_STARTED,
                        {
                            'name': name,
                            'timestamp': time.time()
                        }
                    )
                    self._event_bus.publish(event)
                    logger.info(f"Component '{name}' started successfully")
                    return True

            except Exception as e:
                self._status[name] = ComponentStatus.ERROR
                logger.error(f"Error starting component '{name}': {e}")

                # Publish event
                event = create_event(
                    EventTopics.COMPONENT_ERROR,
                    {
                        'name': name,
                        'error': str(e),
                        'timestamp': time.time()
                    },
                    priority=EventPriority.HIGH
                )
                self._event_bus.publish(event)
                return False

    def stop(self, name: str, wait_for_dependents: bool = True, timeout: float = None) -> bool:
        """
        Stop a specific component.

        Args:
            name: The name of the component to stop
            wait_for_dependents: Whether to wait for dependents to stop
            timeout: Maximum time to wait for dependents (seconds)

        Returns:
            Whether the component was successfully stopped
        """
        if timeout is None:
            timeout = self._max_stop_wait

        with self._lock:
            # Check if component exists
            if name not in self._components:
                logger.error(f"Cannot stop component '{name}': not found")
                return False

            # Check if component is already stopped
            if self._status[name] in [ComponentStatus.STOPPED, ComponentStatus.REGISTERED]:
                logger.debug(f"Component '{name}' is already stopped")
                return True

            # Check if component is in the process of stopping
            if self._status[name] == ComponentStatus.STOPPING:
                logger.debug(f"Component '{name}' is already stopping")
                return True

            # Get component
            component = self._components[name]
            stop_method = getattr(component, 'stop', None)

            # Check for dependents
            dependents = self._dependents[name]
            active_dependents = [dep for dep in dependents
                               if dep in self._components and
                               self._status.get(dep) == ComponentStatus.RUNNING]

            if active_dependents:
                logger.info(f"Component '{name}' has active dependents: {active_dependents}")

                if wait_for_dependents:
                    # Stop dependents first (no need to check for cycles as we checked during ordering)
                    for dep in active_dependents:
                        if not self.stop(dep, wait_for_dependents=True, timeout=timeout):
                            logger.error(f"Failed to stop dependent '{dep}' for component '{name}'")
                            return False
                else:
                    logger.warning(f"Cannot stop component '{name}' due to active dependents")
                    return False

            # If no stop method, just update status
            if not stop_method or not callable(stop_method):
                logger.debug(f"Component '{name}' does not have a stop method")
                self._status[name] = ComponentStatus.STOPPED
                return True

            # Update status to STOPPING
            self._status[name] = ComponentStatus.STOPPING
            logger.info(f"Stopping component '{name}'")

            try:
                # Call stop method
                stop_result = stop_method()

                # Update status based on result
                if stop_result is False:  # Explicit False return
                    self._status[name] = ComponentStatus.ERROR
                    logger.error(f"Component '{name}' stop method returned False")

                    # Publish event
                    event = create_event(
                        EventTopics.COMPONENT_ERROR,
                        {
                            'name': name,
                            'error': "Stop method returned False",
                            'timestamp': time.time()
                        },
                        priority=EventPriority.HIGH
                    )
                    self._event_bus.publish(event)
                    return False
                else:
                    self._status[name] = ComponentStatus.STOPPED

                    # Calculate uptime
                    start_time = self._component_info[name].get('start_time', 0)
                    if start_time > 0:
                        uptime = time.time() - start_time
                        self._component_info[name]['last_uptime'] = uptime

                    # Publish event
                    event = create_event(
                        EventTopics.COMPONENT_STOPPED,
                        {
                            'name': name,
                            'timestamp': time.time()
                        }
                    )
                    self._event_bus.publish(event)
                    logger.info(f"Component '{name}' stopped successfully")
                    return True

            except Exception as e:
                self._status[name] = ComponentStatus.ERROR
                logger.error(f"Error stopping component '{name}': {e}")

                # Publish event
                event = create_event(
                    EventTopics.COMPONENT_ERROR,
                    {
                        'name': name,
                        'error': str(e),
                        'timestamp': time.time()
                    },
                    priority=EventPriority.HIGH
                )
                self._event_bus.publish(event)
                return False

    def start_all(self) -> bool:
        """
        Start all components in dependency order.

        Returns:
            Whether all components were successfully started
        """
        with self._lock:
            logger.info("Starting all components in dependency order")

            # Get components grouped by dependency level
            dependency_levels = self._get_dependency_ordering()

            self._startup_complete = False
            success = True

            # Start components level by level
            for level, component_group in enumerate(dependency_levels):
                logger.info(f"Starting dependency level {level+1} components: {component_group}")

                # Start all components at this level
                level_success = True
                for name in component_group:
                    if self._status.get(name) != ComponentStatus.RUNNING:
                        component_success = self.start(name, wait_for_dependencies=False)
                        level_success = level_success and component_success
                        if not component_success:
                            logger.error(f"Failed to start component '{name}' at level {level+1}")

                # Check if this level was successful
                if not level_success:
                    success = False
                    logger.error(f"Failed to start all components at level {level+1}")
                    # Continue to next level anyway

            self._startup_complete = True

            if success:
                logger.info("All components started successfully")
            else:
                logger.warning("Not all components started successfully")

            return success

    def stop_all(self) -> bool:
        """
        Stop all components in reverse dependency order.

        Returns:
            Whether all components were successfully stopped
        """
        with self._lock:
            logger.info("Stopping all components in reverse dependency order")

            # Get components grouped by dependency level
            dependency_levels = self._get_dependency_ordering()

            # Reverse the order for shutdown
            dependency_levels.reverse()

            self._shutdown_in_progress = True
            success = True

            # Stop components level by level (in reverse)
            for level, component_group in enumerate(dependency_levels):
                logger.info(f"Stopping dependency level {len(dependency_levels)-level} components: {component_group}")

                # Stop all components at this level
                level_success = True
                for name in component_group:
                    if self._status.get(name) == ComponentStatus.RUNNING:
                        component_success = self.stop(name, wait_for_dependents=False)
                        level_success = level_success and component_success
                        if not component_success:
                            logger.error(f"Failed to stop component '{name}' at level {len(dependency_levels)-level}")

                # Check if this level was successful
                if not level_success:
                    success = False
                    logger.error(f"Failed to stop all components at level {len(dependency_levels)-level}")
                    # Continue to next level anyway

            self._shutdown_in_progress = False

            if success:
                logger.info("All components stopped successfully")
            else:
                logger.warning("Not all components stopped successfully")

            return success

    def restart_all(self) -> bool:
        """
        Restart all components.

        Returns:
            Whether all components were successfully restarted
        """
        logger.info("Restarting all components")

        # Stop all components
        stop_success = self.stop_all()
        if not stop_success:
            logger.warning("Not all components stopped successfully during restart")

        # Short delay to ensure everything is properly stopped
        time.sleep(2)

        # Start all components
        return self.start_all()

    def restart_component(self, name: str) -> bool:
        """
        Restart a specific component.

        Args:
            name: The name of the component to restart

        Returns:
            Whether the component was successfully restarted
        """
        logger.info(f"Restarting component '{name}'")

        # Stop the component
        stop_success = self.stop(name)
        if not stop_success:
            logger.error(f"Failed to stop component '{name}' during restart")
            return False

        # Short delay to ensure it's properly stopped
        time.sleep(1)

        # Start the component
        return self.start(name)

    def execute_command(self, name: str, command: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a command on a specific component.

        Args:
            name: The name of the component
            command: The command to execute
            params: Command parameters

        Returns:
            Command execution result
        """
        params = params or {}
        result = {
            "success": False,
            "command": command,
            "component": name
        }

        # Get component
        component = self.get(name)
        if not component:
            result["error"] = f"Component '{name}' not found"
            return result

        # Check if component supports commands
        execute_command = getattr(component, 'execute_command', None)
        if not execute_command or not callable(execute_command):
            result["error"] = f"Component '{name}' does not support commands"
            return result

        try:
            # Execute command
            component_result = execute_command(command, params)

            # Merge results
            if isinstance(component_result, dict):
                result.update(component_result)
                if "success" not in result:
                    result["success"] = True
            else:
                result["success"] = True
                result["result"] = component_result

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Error executing command '{command}' on component '{name}': {e}")

        return result

    def is_dependency_satisfied(self, component_name: str, dependency_name: str) -> bool:
        """
        Check if a dependency is satisfied for a component.

        Args:
            component_name: The name of the component
            dependency_name: The name of the dependency

        Returns:
            Whether the dependency is satisfied
        """
        with self._lock:
            # Check if dependency exists and is running
            if dependency_name not in self._components:
                return False

            return self._status.get(dependency_name) == ComponentStatus.RUNNING

