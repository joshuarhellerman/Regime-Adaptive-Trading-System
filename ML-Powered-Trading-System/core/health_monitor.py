# File: core/health_monitor.py
import os
import time
import logging
import threading
import psutil
import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
import json
import sqlite3
from pathlib import Path

# Import event bus for alert publishing
from core.event_bus import Event, EventPriority

# Define alert levels and categories if not already defined elsewhere
class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

class AlertCategory(Enum):
    """Alert categories"""
    SYSTEM_HEALTH = "system_health"
    PERFORMANCE = "performance"
    MARKET_DATA = "market_data"
    EXECUTION = "execution"
    MODEL = "model"
    SECURITY = "security"

class ResourceType(Enum):
    """Types of resources to monitor"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    COMPONENT = "component"
    LATENCY = "latency"
    MARKET_DATA = "market_data"
    MODEL = "model"
    EXECUTION = "execution"


class HealthStatus(Enum):
    """Health statuses for components and resources"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthMetric:
    """Health metric data structure"""
    name: str
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    status: HealthStatus = HealthStatus.HEALTHY
    resource_type: ResourceType = ResourceType.COMPONENT
    component_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentHealth:
    """Component health data structure"""
    component_id: str
    status: HealthStatus
    last_update: float
    error_rate: float
    metrics: Dict[str, HealthMetric] = field(default_factory=dict)
    latency: Dict[str, float] = field(default_factory=dict)


class HealthMonitor:
    """
    System health monitor with comprehensive resource tracking,
    component performance monitoring, and historical analysis
    for ML-powered trading systems.
    """
    
    def __init__(self, event_bus=None, db_path: str = "data/metrics.db", 
                monitoring_interval: int = 60, history_retention_days: int = 7):
        """
        Initialize the health monitor.
        
        Args:
            event_bus: Event bus for publishing alerts
            db_path: Path to SQLite database for metric history
            monitoring_interval: Interval in seconds for resource monitoring
            history_retention_days: Number of days to retain metric history
        """
        self._logger = logging.getLogger(__name__)
        self._event_bus = event_bus
        self._db_path = db_path
        self._monitoring_interval = monitoring_interval
        self._history_retention_days = history_retention_days
        
        self._lock = threading.RLock()
        self._running = False
        self._monitor_thread = None
        
        # Component health data
        self._components: Dict[str, ComponentHealth] = {}
        
        # Resource thresholds with trading-specific defaults
        self._thresholds: Dict[ResourceType, Dict[HealthStatus, float]] = {
            ResourceType.CPU: {
                HealthStatus.WARNING: 70.0,  # Lower threshold for trading
                HealthStatus.CRITICAL: 90.0
            },
            ResourceType.MEMORY: {
                HealthStatus.WARNING: 70.0,  # Lower threshold for trading
                HealthStatus.CRITICAL: 85.0
            },
            ResourceType.DISK: {
                HealthStatus.WARNING: 80.0,
                HealthStatus.CRITICAL: 95.0
            },
            ResourceType.NETWORK: {
                HealthStatus.WARNING: 80.0,
                HealthStatus.CRITICAL: 95.0
            },
            ResourceType.COMPONENT: {
                HealthStatus.WARNING: 3.0,  # Stricter error rate for trading (%)
                HealthStatus.CRITICAL: 10.0
            },
            ResourceType.LATENCY: {
                HealthStatus.WARNING: 500.0,  # Stricter latency for trading (ms)
                HealthStatus.CRITICAL: 2000.0
            },
            ResourceType.MARKET_DATA: {
                HealthStatus.WARNING: 2.0,  # Data delay/lag threshold (sec)
                HealthStatus.CRITICAL: 5.0
            },
            ResourceType.MODEL: {
                HealthStatus.WARNING: 0.05,  # Model drift threshold
                HealthStatus.CRITICAL: 0.15
            },
            ResourceType.EXECUTION: {
                HealthStatus.WARNING: 1.0,  # Order failure rate (%)
                HealthStatus.CRITICAL: 5.0
            }
        }
        
        # Last alert times to avoid alert flooding
        self._last_alerts: Dict[str, float] = {}
        
        # Initialize database
        self._init_database()
    
    def start(self):
        """
        Start the health monitor.
        """
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._monitor_thread = threading.Thread(
                target=self._resource_monitor_loop,
                daemon=True,
                name="HealthMonitor"
            )
            self._monitor_thread.start()
            
            self._logger.info("Health monitor started")
    
    def stop(self):
        """
        Stop the health monitor.
        """
        with self._lock:
            if not self._running:
                return
            
            self._running = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5.0)
                self._monitor_thread = None
            
            self._logger.info("Health monitor stopped")
    
    def register_component(self, component_id: str, component_type: str = None):
        """
        Register a component for health monitoring.
        
        Args:
            component_id: Unique identifier for the component
            component_type: Type of component (optional)
        """
        with self._lock:
            if component_id not in self._components:
                self._components[component_id] = ComponentHealth(
                    component_id=component_id,
                    status=HealthStatus.UNKNOWN,
                    last_update=time.time(),
                    error_rate=0.0
                )
                
                # Add component type to metrics if provided
                if component_type:
                    metric = HealthMetric(
                        name="component_type",
                        value=0,
                        unit="type",
                        component_id=component_id,
                        details={"type": component_type}
                    )
                    self._components[component_id].metrics["component_type"] = metric
                
                self._logger.debug(f"Registered component for health monitoring: {component_id}")
    
    def unregister_component(self, component_id: str):
        """
        Unregister a component from health monitoring.
        
        Args:
            component_id: Unique identifier for the component
        """
        with self._lock:
            if component_id in self._components:
                del self._components[component_id]
                self._logger.debug(f"Unregistered component from health monitoring: {component_id}")
    
    def update_component_health(self, component_id: str, status: HealthStatus = None,
                              error_rate: float = None, metrics: Dict[str, Any] = None,
                              latency: Dict[str, float] = None):
        """
        Update the health status of a component.
        
        Args:
            component_id: Unique identifier for the component
            status: New health status
            error_rate: Error rate as a percentage
            metrics: Dictionary of metrics to record
            latency: Dictionary of operation latencies in milliseconds
        """
        with self._lock:
            # Register component if not already registered
            if component_id not in self._components:
                self.register_component(component_id)
            
            component = self._components[component_id]
            component.last_update = time.time()
            
            # Update status if provided
            if status is not None:
                old_status = component.status
                component.status = status
                
                # Log status change
                if old_status != status:
                    self._logger.info(f"Component {component_id} status changed from {old_status} to {status}")
                    
                    # Send alert if status degraded
                    if status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                        self._send_component_alert(component_id, old_status, status)
            
            # Update error rate if provided
            if error_rate is not None:
                component.error_rate = error_rate
                
                # Determine status from error rate
                error_status = self._get_status_from_threshold(
                    ResourceType.COMPONENT, error_rate
                )
                
                # Update status if error rate indicates a worse status
                if error_status.value > component.status.value:
                    old_status = component.status
                    component.status = error_status
                    self._logger.info(f"Component {component_id} status changed to {error_status} due to error rate: {error_rate}%")
                    
                    # Send alert
                    self._send_component_alert(component_id, old_status, error_status, 
                                            details={"error_rate": error_rate})
            
            # Update metrics if provided
            if metrics:
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        # Create or update metric
                        metric = component.metrics.get(metric_name)
                        
                        if metric is None:
                            # Create new metric
                            unit = "count"
                            if "percentage" in metric_name.lower() or metric_name.endswith("_pct"):
                                unit = "%"
                            elif "time" in metric_name.lower() or metric_name.endswith("_ms"):
                                unit = "ms"
                            
                            metric = HealthMetric(
                                name=metric_name,
                                value=metric_value,
                                unit=unit,
                                component_id=component_id
                            )
                        else:
                            # Update existing metric
                            metric.value = metric_value
                            metric.timestamp = time.time()
                        
                        component.metrics[metric_name] = metric
                        
                        # Record metric in database
                        self._record_metric(metric)
            
            # Update latency if provided
            if latency:
                for operation, latency_ms in latency.items():
                    component.latency[operation] = latency_ms
                    
                    # Determine status from latency
                    latency_status = self._get_status_from_threshold(
                        ResourceType.LATENCY, latency_ms
                    )
                    
                    # Create or update latency metric
                    metric_name = f"latency_{operation}"
                    metric = HealthMetric(
                        name=metric_name,
                        value=latency_ms,
                        unit="ms",
                        timestamp=time.time(),
                        status=latency_status,
                        resource_type=ResourceType.LATENCY,
                        component_id=component_id,
                        details={"operation": operation}
                    )
                    
                    component.metrics[metric_name] = metric
                    
                    # Record metric in database
                    self._record_metric(metric)
                    
                    # Send alert if latency is too high
                    if latency_status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                        self._send_latency_alert(component_id, operation, latency_ms, latency_status)
    
    def log_error(self, component_id: str, error_type: str, error_message: str):
        """
        Log an error for a component and update error rate.
        
        Args:
            component_id: Unique identifier for the component
            error_type: Type of error
            error_message: Error message
        """
        with self._lock:
            # Register component if not already registered
            if component_id not in self._components:
                self.register_component(component_id)
            
            # Get error count and total operation metrics
            error_count = self._get_metric_value(component_id, "error_count", 0)
            total_ops = self._get_metric_value(component_id, "operation_count", 0)
            
            # Increment error count
            error_count += 1
            
            # Update metrics
            self.update_component_health(
                component_id=component_id,
                metrics={
                    "error_count": error_count,
                    "last_error_time": time.time()
                }
            )
            
            # Calculate error rate if we have operations
            if total_ops > 0:
                error_rate = (error_count / total_ops) * 100.0
                self.update_component_health(
                    component_id=component_id,
                    error_rate=error_rate
                )
            
            # Log error
            self._logger.error(f"Component {component_id} error: {error_type} - {error_message}")
    
    def log_operation(self, component_id: str, operation_type: str, duration_ms: float,
                    success: bool = True):
        """
        Log an operation for a component with latency tracking.
        
        Args:
            component_id: Unique identifier for the component
            operation_type: Type of operation
            duration_ms: Duration of operation in milliseconds
            success: Whether the operation was successful
        """
        with self._lock:
            # Register component if not already registered
            if component_id not in self._components:
                self.register_component(component_id)
            
            # Get operation counts
            total_ops = self._get_metric_value(component_id, "operation_count", 0)
            op_count = self._get_metric_value(component_id, f"{operation_type}_count", 0)
            
            # Increment counts
            total_ops += 1
            op_count += 1
            
            # Update metrics
            metrics = {
                "operation_count": total_ops,
                f"{operation_type}_count": op_count,
                "last_operation_time": time.time()
            }
            
            # Update error count if operation failed
            if not success:
                error_count = self._get_metric_value(component_id, "error_count", 0)
                error_count += 1
                metrics["error_count"] = error_count
                
                # Calculate error rate
                error_rate = (error_count / total_ops) * 100.0
                
                self.update_component_health(
                    component_id=component_id,
                    error_rate=error_rate,
                    metrics=metrics,
                    latency={operation_type: duration_ms}
                )
            else:
                self.update_component_health(
                    component_id=component_id,
                    metrics=metrics,
                    latency={operation_type: duration_ms}
                )
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get the overall system health status.
        
        Returns:
            Dictionary with system health information
        """
        with self._lock:
            # Get current CPU, memory, and disk usage
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            # Get component statuses
            component_statuses = {}
            critical_components = []
            warning_components = []
            
            for component_id, component in self._components.items():
                component_statuses[component_id] = component.status.value
                
                if component.status == HealthStatus.CRITICAL:
                    critical_components.append(component_id)
                elif component.status == HealthStatus.WARNING:
                    warning_components.append(component_id)
            
            # Determine overall status
            overall_status = HealthStatus.HEALTHY
            
            # Check CPU
            cpu_status = self._get_status_from_threshold(ResourceType.CPU, cpu_percent)
            if cpu_status.value > overall_status.value:
                overall_status = cpu_status
            
            # Check memory
            memory_status = self._get_status_from_threshold(ResourceType.MEMORY, memory_percent)
            if memory_status.value > overall_status.value:
                overall_status = memory_status
            
            # Check disk
            disk_status = self._get_status_from_threshold(ResourceType.DISK, disk_percent)
            if disk_status.value > overall_status.value:
                overall_status = disk_status
            
            # Check components
            if critical_components:
                overall_status = HealthStatus.CRITICAL
            elif warning_components and overall_status != HealthStatus.CRITICAL:
                overall_status = HealthStatus.WARNING
            
            return {
                "status": overall_status.value,
                "timestamp": time.time(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent,
                "component_count": len(self._components),
                "component_statuses": component_statuses,
                "critical_components": critical_components,
                "warning_components": warning_components
            }
    
    def get_component_health(self, component_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the health status of a specific component.
        
        Args:
            component_id: Unique identifier for the component
            
        Returns:
            Dictionary with component health information, or None if component not found
        """
        with self._lock:
            if component_id not in self._components:
                return None
            
            component = self._components[component_id]
            
            # Convert metrics to dictionary
            metrics = {}
            for metric_name, metric in component.metrics.items():
                metrics[metric_name] = {
                    "value": metric.value,
                    "unit": metric.unit,
                    "timestamp": metric.timestamp,
                    "status": metric.status.value
                }
            
            return {
                "component_id": component_id,
                "status": component.status.value,
                "last_update": component.last_update,
                "error_rate": component.error_rate,
                "metrics": metrics,
                "latency": component.latency
            }
    
    def get_metrics_history(self, component_id: str, metric_name: str,
                          start_time: Optional[float] = None,
                          end_time: Optional[float] = None,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get historical metrics for a component.
        
        Args:
            component_id: Unique identifier for the component
            metric_name: Name of the metric
            start_time: Start time as Unix timestamp
            end_time: End time as Unix timestamp
            limit: Maximum number of records to return
            
        Returns:
            List of metric dictionaries
        """
        if not start_time:
            start_time = time.time() - (86400 * 7)  # Default to 7 days ago
        
        if not end_time:
            end_time = time.time()
        
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        try:
            query = """
                SELECT timestamp, value, status, resource_type, details
                FROM metrics
                WHERE component_id = ? AND name = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            
            cursor.execute(query, (component_id, metric_name, start_time, end_time, limit))
            
            metrics = []
            for row in cursor.fetchall():
                timestamp, value, status, resource_type, details_json = row
                
                try:
                    details = json.loads(details_json) if details_json else {}
                except json.JSONDecodeError:
                    details = {}
                
                metrics.append({
                    "timestamp": timestamp,
                    "value": value,
                    "status": status,
                    "resource_type": resource_type,
                    "details": details
                })
            
            return metrics
            
        finally:
            cursor.close()
            conn.close()
    
    def get_latency_history(self, component_id: str, operation: str,
                          start_time: Optional[float] = None,
                          end_time: Optional[float] = None,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get historical latency for a component operation.
        
        Args:
            component_id: Unique identifier for the component
            operation: Operation name
            start_time: Start time as Unix timestamp
            end_time: End time as Unix timestamp
            limit: Maximum number of records to return
            
        Returns:
            List of latency dictionaries
        """
        metric_name = f"latency_{operation}"
        return self.get_metrics_history(component_id, metric_name, start_time, end_time, limit)
    
    def get_all_components_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the health status of all components.
        
        Returns:
            Dictionary mapping component IDs to health information
        """
        with self._lock:
            result = {}
            
            for component_id in self._components:
                result[component_id] = self.get_component_health(component_id)
            
            return result
    
    def set_threshold(self, resource_type: ResourceType, warning_threshold: float,
                    critical_threshold: float):
        """
        Set the warning and critical thresholds for a resource type.
        
        Args:
            resource_type: Type of resource
            warning_threshold: Warning threshold value
            critical_threshold: Critical threshold value
        """
        with self._lock:
            if warning_threshold >= critical_threshold:
                self._logger.warning(f"Warning threshold should be less than critical threshold: {resource_type}")
            
            self._thresholds[resource_type] = {
                HealthStatus.WARNING: warning_threshold,
                HealthStatus.CRITICAL: critical_threshold
            }
            
            self._logger.debug(f"Set thresholds for {resource_type}: warning={warning_threshold}, critical={critical_threshold}")
    
    def get_threshold(self, resource_type: ResourceType) -> Dict[HealthStatus, float]:
        """
        Get the thresholds for a resource type.
        
        Args:
            resource_type: Type of resource
            
        Returns:
            Dictionary mapping status to threshold value
        """
        with self._lock:
            return self._thresholds.get(resource_type, {})
    
    def export_state(self) -> Dict[str, Any]:
        """
        Export the monitor state for disaster recovery.
        
        Returns:
            Dictionary with monitor state
        """
        with self._lock:
            # Convert components to serializable format
            components = {}
            for component_id, component in self._components.items():
                components[component_id] = {
                    "status": component.status.value,
                    "last_update": component.last_update,
                    "error_rate": component.error_rate,
                    "latency": component.latency
                }
            
            # Convert thresholds to serializable format
            thresholds = {}
            for resource_type, status_thresholds in self._thresholds.items():
                thresholds[resource_type.value] = {
                    status.value: threshold for status, threshold in status_thresholds.items()
                }
            
            return {
                "components": components,
                "thresholds": thresholds,
                "last_alerts": self._last_alerts
            }
    
    def import_state(self, state: Dict[str, Any]) -> bool:
        """
        Import the monitor state for disaster recovery.
        
        Args:
            state: State dictionary
            
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                # Import components
                components = state.get("components", {})
                for component_id, component_data in components.items():
                    self._components[component_id] = ComponentHealth(
                        component_id=component_id,
                        status=HealthStatus(component_data.get("status", HealthStatus.UNKNOWN.value)),
                        last_update=component_data.get("last_update", time.time()),
                        error_rate=component_data.get("error_rate", 0.0),
                        latency=component_data.get("latency", {})
                    )
                
                # Import thresholds
                thresholds = state.get("thresholds", {})
                for resource_type_value, status_thresholds in thresholds.items():
                    resource_type = ResourceType(resource_type_value)
                    self._thresholds[resource_type] = {
                        HealthStatus(status_value): threshold 
                        for status_value, threshold in status_thresholds.items()
                    }
                
                # Import last alerts
                self._last_alerts = state.get("last_alerts", {})
                
                self._logger.info(f"Imported monitor state with {len(components)} components")
                return True
                
            except Exception as e:
                self._logger.error(f"Error importing monitor state: {e}")
                return False
    
    # Trading-specific metrics and monitoring
    
    def monitor_market_data_latency(self, source_id: str, symbol: str, 
                                   reception_time: float, exchange_time: float = None):
        """
        Monitor and record market data latency.
        
        Args:
            source_id: Data source identifier
            symbol: Market symbol
            reception_time: Local reception timestamp
            exchange_time: Exchange timestamp (if available)
        """
        component_id = f"market_data_{source_id}"
        
        # Register component if not already registered
        if component_id not in self._components:
            self.register_component(component_id, "market_data")
        
        # Calculate data freshness/delay
        current_time = time.time()
        reception_delay = (current_time - reception_time) * 1000  # in ms
        
        # Calculate exchange delay if exchange time is available
        if exchange_time:
            exchange_delay = (current_time - exchange_time) * 1000  # in ms
            
            # Determine status from delay
            delay_status = self._get_status_from_threshold(
                ResourceType.MARKET_DATA, exchange_delay / 1000  # convert to seconds for threshold
            )
            
            # Record metrics
            self.update_component_health(
                component_id=component_id,
                metrics={
                    f"exchange_delay_{symbol}": exchange_delay,
                    f"reception_delay_{symbol}": reception_delay,
                    f"last_update_{symbol}": current_time
                },
                status=delay_status
            )
            
            # Send alert if delay is too high
            if delay_status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                self._send_market_data_alert(source_id, symbol, exchange_delay, delay_status)
        else:
            # Only reception delay is available
            self.update_component_health(
                component_id=component_id,
                metrics={
                    f"reception_delay_{symbol}": reception_delay,
                    f"last_update_{symbol}": current_time
                }
            )
    
    def monitor_order_execution(self, exchange_id: str, order_id: str, 
                               execution_time: float, status: str):
        """
        Monitor order execution performance.
        
        Args:
            exchange_id: Exchange identifier
            order_id: Order identifier
            execution_time: Order execution time in ms
            status: Order status (filled, partial, rejected, etc.)
        """
        component_id = f"execution_{exchange_id}"
        
        # Register component if needed
        if component_id not in self._components:
            self.register_component(component_id, "execution")
        
        # Update execution metrics
        metrics = {
            f"execution_time_{order_id}": execution_time,
            "last_order_time": time.time()
        }
        
        # Track order status
        if status.lower() in ["filled", "complete", "success"]:
            success_count = self._get_metric_value(component_id, "order_success_count", 0)
            metrics["order_success_count"] = success_count + 1
            success = True
        elif status.lower() in ["rejected", "failed", "error"]:
            failure_count = self._get_metric_value(component_id, "order_failure_count", 0)
            metrics["order_failure_count"] = failure_count + 1
            success = False
        else:
            # Partial fills or other statuses
            partial_count = self._get_metric_value(component_id, "order_partial_count", 0)
            metrics["order_partial_count"] = partial_count + 1
            success = True  # Consider partial as success for now
        
        # Calculate execution latency
        self.log_operation(
            component_id=component_id,
            operation_type="order_execution",
            duration_ms=execution_time,
            success=success
        )
        
        # Update component metrics
        self.update_component_health(
            component_id=component_id,
            metrics=metrics
        )
        
        # Calculate and check failure rate if we have orders
        total_orders = self._get_metric_value(component_id, "operation_count", 0)
        if total_orders > 0:
            failure_count = self._get_metric_value(component_id, "order_failure_count", 0)
            failure_rate = (failure_count / total_orders) * 100.0
            
            # Get status from failure rate
            execution_status = self._get_status_from_threshold(
                ResourceType.EXECUTION, failure_rate
            )
            
            # Update component status if needed
            self.update_component_health(
                component_id=component_id,
                status=execution_status,
                metrics={"order_failure_rate": failure_rate}
            )
            
            # Send alert if failure rate is too high
            if execution_status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                self._send_execution_alert(exchange_id, failure_rate, execution_status)
    
    def monitor_model_performance(self, model_id: str, prediction_error: float, 
                                inference_time: float, model_version: str = None):
        """
        Monitor ML model performance.
        
        Args:
            model_id: Model identifier
            prediction_error: Prediction error or drift metric
            inference_time: Inference time in milliseconds
            model_version: Model version identifier
        """
        component_id = f"model_{model_id}"
        
        # Register component if needed
        if component_id not in self._components:
            self.register_component(component_id, "model")
        
        # Determine status from prediction error
        model_status = self._get_status_from_threshold(
            ResourceType.MODEL, prediction_error
        )
        
        # Update metrics
        metrics = {
            "prediction_error": prediction_error,
            "inference_time": inference_time,
            "last_inference_time": time.time()
        }
        
        if model_version:
            metrics["model_version"] = float(model_version.replace("v", "").replace(".", ""))
        
        # Update component health
        self.update_component_health(
            component_id=component_id,
            status=model_status,
            metrics=metrics,
            latency={"inference": inference_time}
        )
        
        # Send alert if model performance is degrading
        if model_status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
            self._send_model_alert(model_id, prediction_error, model_status, model_version)
    
    # Database methods
    
    def _init_database(self):
        """
        Initialize the metrics database.
        """
        # Skip if using in-memory database
        if self._db_path == ":memory:":
            conn = sqlite3.connect(self._db_path)
        else:
            # Create directory if needed
            db_dir = os.path.dirname(self._db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
            
            conn = sqlite3.connect(self._db_path)
        
        cursor = conn.cursor()
        
        try:
            # Create metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY,
                    component_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    status TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    details TEXT
                )
            """)
            
            # Create index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_component_name_time
                ON metrics (component_id, name, timestamp)
            """)
            
            conn.commit()
            
        finally:
            cursor.close()
            conn.close()
    
    def _record_metric(self, metric: HealthMetric):
        """
        Record a metric in the database.
        
        Args:
            metric: Metric to record
        """
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        try:
            details_json = json.dumps(metric.details) if metric.details else None
            
            cursor.execute("""
                INSERT INTO metrics
                (component_id, name, value, unit, timestamp, status, resource_type, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.component_id,
                metric.name,
                metric.value,
                metric.unit,
                metric.timestamp,
                metric.status.value,
                metric.resource_type.value,
                details_json
            ))
            
            conn.commit()
            
            # Clean up old metrics
            self._cleanup_old_metrics(cursor)
            conn.commit()
            
        finally:
            cursor.close()
            conn.close()
    
    def _cleanup_old_metrics(self, cursor):
        """
        Clean up old metrics from the database.
        
        Args:
            cursor: Database cursor
        """
        # Skip if we're using an in-memory database
        if self._db_path == ":memory:":
            return
        
        # Calculate cutoff timestamp
        cutoff_time = time.time() - (86400 * self._history_retention_days)
        
        # Delete old metrics
        cursor.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff_time,))
    
    def _get_db_connection(self):
        """
        Get a connection to the metrics database.
        
        Returns:
            SQLite connection
        """
        return sqlite3.connect(self._db_path)
    
    def _get_metric_value(self, component_id: str, metric_name: str, default_value: float = 0.0) -> float:
        """
        Get the value of a metric for a component.
        
        Args:
            component_id: Unique identifier for the component
            metric_name: Name of the metric
            default_value: Default value if metric not found
            
        Returns:
            Metric value
        """
        component = self._components.get(component_id)
        if not component:
            return default_value
        
        metric = component.metrics.get(metric_name)
        if not metric:
            return default_value
        
        return metric.value
    
    def _get_status_from_threshold(self, resource_type: ResourceType, value: float) -> HealthStatus:
        """
        Get the health status based on thresholds.
        
        Args:
            resource_type: Type of resource
            value: Resource value
            
        Returns:
            Health status
        """
        thresholds = self._thresholds.get(resource_type, {})
        
        if value >= thresholds.get(HealthStatus.CRITICAL, float('inf')):
            return HealthStatus.CRITICAL
        elif value >= thresholds.get(HealthStatus.WARNING, float('inf')):
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    # Alert methods
    
    def _send_component_alert(self, component_id: str, old_status: HealthStatus,
                            new_status: HealthStatus, details: Dict[str, Any] = None):
        """
        Send an alert for a component status change.
        
        Args:
            component_id: Unique identifier for the component
            old_status: Previous status
            new_status: New status
            details: Additional details
        """
        if not self._event_bus:
            return
        
        # Avoid alert flooding
        alert_key = f"component_{component_id}_{new_status.value}"
        current_time = time.time()
        last_alert_time = self._last_alerts.get(alert_key, 0)
        
        # Only send alert if more than 5 minutes since last alert
        if current_time - last_alert_time < 300:
            return
        
        # Update last alert time
        self._last_alerts[alert_key] = current_time
        
        # Determine alert level
        if new_status == HealthStatus.CRITICAL:
            alert_level = AlertLevel.CRITICAL
        elif new_status == HealthStatus.WARNING:
            alert_level = AlertLevel.WARNING
        else:
            alert_level = AlertLevel.INFO
        
        # Create alert data
        alert_data = {
            "component_id": component_id,
            "old_status": old_status.value,
            "new_status": new_status.value,
            "timestamp": current_time
        }
        
        if details:
            alert_data.update(details)
        
        # Create and publish alert event
        event = Event(
            topic="alert.component",
            data={
                "level": alert_level.value,
                "category": AlertCategory.SYSTEM_HEALTH.value,
                "title": f"Component {component_id} Status Change",
                "message": f"Component {component_id} status changed from {old_status.value} to {new_status.value}",
                "data": alert_data
            },
            priority=EventPriority.HIGH
        )
        
        self._event_bus.publish(event)
    
    def _send_latency_alert(self, component_id: str, operation: str, latency: float,
                          status: HealthStatus):
        """
        Send an alert for high latency.
        
        Args:
            component_id: Unique identifier for the component
            operation: Operation name
            latency: Latency value in milliseconds
            status: Health status
        """
        if not self._event_bus:
            return
        
        # Avoid alert flooding
        alert_key = f"latency_{component_id}_{operation}_{status.value}"
        current_time = time.time()
        last_alert_time = self._last_alerts.get(alert_key, 0)
        
        # Only send alert if more than 5 minutes since last alert
        if current_time - last_alert_time < 300:
            return
        
        # Update last alert time
        self._last_alerts[alert_key] = current_time
        
        # Determine alert level
        if status == HealthStatus.CRITICAL:
            alert_level = AlertLevel.CRITICAL
        elif status == HealthStatus.WARNING:
            alert_level = AlertLevel.WARNING
        else:
            return  # Don't send alert for healthy status
        
        # Create alert data
        alert_data = {
            "component_id": component_id,
            "operation": operation,
            "latency": latency,
            "status": status.value,
            "timestamp": current_time
        }
        
        # Create and publish alert event
        event = Event(
            topic="alert.latency",
            data={
                "level": alert_level.value,
                "category": AlertCategory.PERFORMANCE.value,
                "title": f"High Latency in {component_id}",
                "message": f"Operation {operation} has high latency ({latency:.2f} ms)",
                "data": alert_data
            },
            priority=EventPriority.HIGH
        )
        
        self._event_bus.publish(event)
    
    def _send_market_data_alert(self, source_id: str, symbol: str, delay: float,
                              status: HealthStatus):
        """
        Send an alert for market data issues.
        
        Args:
            source_id: Data source identifier
            symbol: Market symbol
            delay: Data delay in milliseconds
            status: Health status
        """
        if not self._event_bus:
            return
        
        # Avoid alert flooding
        alert_key = f"market_data_{source_id}_{symbol}_{status.value}"
        current_time = time.time()
        last_alert_time = self._last_alerts.get(alert_key, 0)
        
        # Only send alert if more than 2 minutes since last alert (shorter for market data)
        if current_time - last_alert_time < 120:
            return
        
        # Update last alert time
        self._last_alerts[alert_key] = current_time
        
        # Determine alert level
        if status == HealthStatus.CRITICAL:
            alert_level = AlertLevel.CRITICAL
        elif status == HealthStatus.WARNING:
            alert_level = AlertLevel.WARNING
        else:
            return  # Don't send alert for healthy status
        
        # Create alert data
        alert_data = {
            "source_id": source_id,
            "symbol": symbol,
            "delay_ms": delay,
            "status": status.value,
            "timestamp": current_time
        }
        
        # Create and publish alert event
        event = Event(
            topic="alert.market_data",
            data={
                "level": alert_level.value,
                "category": AlertCategory.MARKET_DATA.value,
                "title": f"Market Data Delay for {symbol}",
                "message": f"Market data from {source_id} for {symbol} delayed by {delay:.2f} ms",
                "data": alert_data
            },
            priority=EventPriority.HIGH
        )
        
        self._event_bus.publish(event)
    
    def _send_execution_alert(self, exchange_id: str, failure_rate: float,
                            status: HealthStatus):
        """
        Send an alert for execution issues.
        
        Args:
            exchange_id: Exchange identifier
            failure_rate: Order failure rate
            status: Health status
        """
        if not self._event_bus:
            return
        
        # Avoid alert flooding
        alert_key = f"execution_{exchange_id}_{status.value}"
        current_time = time.time()
        last_alert_time = self._last_alerts.get(alert_key, 0)
        
        # Only send alert if more than 2 minutes since last alert
        if current_time - last_alert_time < 120:
            return
        
        # Update last alert time
        self._last_alerts[alert_key] = current_time
        
        # Determine alert level
        if status == HealthStatus.CRITICAL:
            alert_level = AlertLevel.CRITICAL
        elif status == HealthStatus.WARNING:
            alert_level = AlertLevel.WARNING
        else:
            return  # Don't send alert for healthy status
        
        # Create alert data
        alert_data = {
            "exchange_id": exchange_id,
            "failure_rate": failure_rate,
            "status": status.value,
            "timestamp": current_time
        }
        
        # Create and publish alert event
        event = Event(
            topic="alert.execution",
            data={
                "level": alert_level.value,
                "category": AlertCategory.EXECUTION.value,
                "title": f"High Order Failure Rate on {exchange_id}",
                "message": f"Order failure rate on {exchange_id} is {failure_rate:.2f}%",
                "data": alert_data
            },
            priority=EventPriority.HIGH
        )
        
        self._event_bus.publish(event)
    
    def _send_model_alert(self, model_id: str, error: float, status: HealthStatus,
                        model_version: str = None):
        """
        Send an alert for model performance issues.
        
        Args:
            model_id: Model identifier
            error: Prediction error
            status: Health status
            model_version: Model version
        """
        if not self._event_bus:
            return
        
        # Avoid alert flooding
        alert_key = f"model_{model_id}_{status.value}"
        current_time = time.time()
        last_alert_time = self._last_alerts.get(alert_key, 0)
        
        # Only send alert if more than 5 minutes since last alert
        if current_time - last_alert_time < 300:
            return
        
        # Update last alert time
        self._last_alerts[alert_key] = current_time
        
        # Determine alert level
        if status == HealthStatus.CRITICAL:
            alert_level = AlertLevel.CRITICAL
        elif status == HealthStatus.WARNING:
            alert_level = AlertLevel.WARNING
        else:
            return  # Don't send alert for healthy status
        
        # Create alert data
        alert_data = {
            "model_id": model_id,
            "error": error,
            "status": status.value,
            "timestamp": current_time
        }
        
        if model_version:
            alert_data["model_version"] = model_version
        
        # Create message
        message = f"Model {model_id} performance degraded with error {error:.4f}"
        if model_version:
            message += f" (version {model_version})"
        
        # Create and publish alert event
        event = Event(
            topic="alert.model",
            data={
                "level": alert_level.value,
                "category": AlertCategory.MODEL.value,
                "title": f"Model Performance Degradation",
                "message": message,
                "data": alert_data
            },
            priority=EventPriority.HIGH
        )
        
        self._event_bus.publish(event)
    
    def _send_resource_alert(self, resource_name: str, value: float, status: HealthStatus):
        """
        Send an alert for a system resource.
        
        Args:
            resource_name: Name of the resource
            value: Resource value
            status: Health status
        """
        if not self._event_bus:
            return
        
        # Determine alert level
        if status == HealthStatus.CRITICAL:
            alert_level = AlertLevel.CRITICAL
        elif status == HealthStatus.WARNING:
            alert_level = AlertLevel.WARNING
        else:
            return  # Don't send alert for healthy status
        
        # Create alert data
        alert_data = {
            "resource": resource_name,
            "value": value,
            "status": status.value,
            "timestamp": time.time()
        }
        
        # Create and publish alert event
        event = Event(
            topic="alert.resource",
            data={
                "level": alert_level.value,
                "category": AlertCategory.SYSTEM_HEALTH.value,
                "title": f"High {resource_name} Usage",
                "message": f"{resource_name} usage is {value:.2f}%",
                "data": alert_data
            },
            priority=EventPriority.HIGH
        )
        
        self._event_bus.publish(event)
    
    def _resource_monitor_loop(self):
        """
        Monitor system resources periodically.
        """
        while self._running:
            try:
                self._monitor_system_resources()
                
                # Sleep until next monitoring interval
                # Use small sleep chunks to allow faster shutdown
                remaining = self._monitoring_interval
                while remaining > 0 and self._running:
                    time.sleep(min(1, remaining))
                    remaining -= 1
                
            except Exception as e:
                self._logger.error(f"Error in resource monitor: {e}", exc_info=True)
                time.sleep(10)  # Sleep on error to avoid tight loops
    
    def _monitor_system_resources(self):
        """
        Monitor system resources and record metrics.
        """
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_status = self._get_status_from_threshold(ResourceType.CPU, cpu_percent)
            
            cpu_metric = HealthMetric(
                name="cpu_usage",
                value=cpu_percent,
                unit="%",
                status=cpu_status,
                resource_type=ResourceType.CPU,
                component_id="system"
            )
            
            self._record_metric(cpu_metric)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_status = self._get_status_from_threshold(ResourceType.MEMORY, memory_percent)
            
            memory_metric = HealthMetric(
                name="memory_usage",
                value=memory_percent,
                unit="%",
                status=memory_status,
                resource_type=ResourceType.MEMORY,
                component_id="system",
                details={
                    "total": memory.total,
                    "available": memory.available
                }
            )
            
            self._record_metric(memory_metric)
            
            # Get disk usage
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_percent = usage.percent
                    disk_status = self._get_status_from_threshold(ResourceType.DISK, disk_percent)
                    
                    disk_metric = HealthMetric(
                        name=f"disk_usage_{partition.device.replace(':', '_')}",
                        value=disk_percent,
                        unit="%",
                        status=disk_status,
                        resource_type=ResourceType.DISK,
                        component_id="system",
                        details={
                            "mountpoint": partition.mountpoint,
                            "total": usage.total,
                            "used": usage.used,
                            "free": usage.free
                        }
                    )
                    
                    self._record_metric(disk_metric)
                except (PermissionError, FileNotFoundError):
                    # Skip partitions we can't access
                    pass
            
            # Get network stats
            net_io = psutil.net_io_counters()
            
            net_metric = HealthMetric(
                name="network_io",
                value=0,  # No threshold for this metric
                unit="bytes",
                status=HealthStatus.HEALTHY,
                resource_type=ResourceType.NETWORK,
                component_id="system",
                details={
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv,
                    "errin": net_io.errin,
                    "errout": net_io.errout,
                    "dropin": net_io.dropin,
                    "dropout": net_io.dropout
                }
            )
            
            self._record_metric(net_metric)
            
            # Send alerts for critical resources
            current_time = time.time()
            
            # CPU alert
            if cpu_status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                alert_key = f"system_cpu_{cpu_status.value}"
                last_alert_time = self._last_alerts.get(alert_key, 0)
                
                if current_time - last_alert_time >= 300:  # 5 minutes
                    self._last_alerts[alert_key] = current_time
                    self._send_resource_alert("CPU", cpu_percent, cpu_status)
            
            # Memory alert
            if memory_status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                alert_key = f"system_memory_{memory_status.value}"
                last_alert_time = self._last_alerts.get(alert_key, 0)
                
                if current_time - last_alert_time >= 300:  # 5 minutes
                    self._last_alerts[alert_key] = current_time
                    self._send_resource_alert("Memory", memory_percent, memory_status)
            
            # Disk alerts
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_percent = usage.percent
                    disk_status = self._get_status_from_threshold(ResourceType.DISK, disk_percent)
                    
                    if disk_status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                        alert_key = f"system_disk_{partition.mountpoint}_{disk_status.value}"
                        last_alert_time = self._last_alerts.get(alert_key, 0)
                        
                        if current_time - last_alert_time >= 300:  # 5 minutes
                            self._last_alerts[alert_key] = current_time
                            self._send_resource_alert(
                                f"Disk {partition.mountpoint}", disk_percent, disk_status
                            )
                except (PermissionError, FileNotFoundError):
                    pass
                
        except Exception as e:
            self._logger.error(f"Error monitoring system resources: {e}", exc_info=True)