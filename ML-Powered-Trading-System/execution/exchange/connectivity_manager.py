"""
connectivity_manager.py - Exchange Connectivity Management System

This module provides robust connection management for exchanges and data sources,
implementing resilience patterns, connection pooling, authentication management,
and seamless failover between endpoints.
"""

import logging
import time
import threading
import queue
import asyncio
import aiohttp
import ssl
import json
import socket
import random
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set
import urllib3
from urllib3.util.retry import Retry
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException, ConnectionError, Timeout, SSLError

from core.event_bus import EventTopics, create_event, get_event_bus, Event, EventPriority

# Configure logger
logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """Status of a connection to an exchange or data source"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"


class ConnectionPriority(Enum):
    """Priority level of a connection endpoint"""
    PRIMARY = 0
    SECONDARY = 1
    FAILOVER = 2
    BACKUP = 3


class EndpointHealth(Enum):
    """Health status of an endpoint"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CircuitBreakerState(Enum):
    """State of a circuit breaker"""
    CLOSED = "closed"  # Normal operation, requests flow through
    OPEN = "open"  # Failing, not allowing requests
    HALF_OPEN = "half_open"  # Testing if system has recovered


class ConnectionEndpoint:
    """Represents a connection endpoint to an exchange or data source"""

    def __init__(self,
                 url: str,
                 priority: ConnectionPriority = ConnectionPriority.PRIMARY,
                 credentials: Optional[Dict[str, str]] = None,
                 options: Optional[Dict[str, Any]] = None):
        """
        Initialize a connection endpoint.

        Args:
            url: Endpoint URL
            priority: Priority level for this endpoint
            credentials: Authentication credentials
            options: Additional connection options
        """
        self.url = url
        self.priority = priority
        self.credentials = credentials or {}
        self.options = options or {}
        self.health = EndpointHealth.UNKNOWN
        self.status = ConnectionStatus.DISCONNECTED
        self.last_connected = None
        self.last_error = None
        self.error_count = 0
        self.success_count = 0
        self.response_times = []
        self.max_response_times = 100  # Keep track of this many response times
        self.session = None  # For persistent HTTP sessions

        # Circuit breaker parameters
        self.circuit_state = CircuitBreakerState.CLOSED
        self.circuit_failure_threshold = options.get('circuit_failure_threshold', 5)
        self.circuit_recovery_timeout = options.get('circuit_recovery_timeout', 30)
        self.circuit_last_state_change = time.time()

    def get_credentials(self) -> Dict[str, str]:
        """Get credentials for this endpoint"""
        return self.credentials.copy()

    def get_average_response_time(self) -> Optional[float]:
        """Get average response time in milliseconds"""
        if not self.response_times:
            return None
        return sum(self.response_times) / len(self.response_times)

    def record_success(self, response_time_ms: float) -> None:
        """Record a successful request"""
        self.success_count += 1
        self.response_times.append(response_time_ms)
        # Keep response times list bounded
        if len(self.response_times) > self.max_response_times:
            self.response_times.pop(0)

        # Update health status
        if self.health == EndpointHealth.UNKNOWN or self.health == EndpointHealth.UNHEALTHY:
            self.health = EndpointHealth.HEALTHY

        # Reset error count on success
        self.error_count = 0

        # Circuit breaker logic
        if self.circuit_state == CircuitBreakerState.HALF_OPEN:
            # Success in half-open state means the system has recovered
            self.circuit_state = CircuitBreakerState.CLOSED
            self.circuit_last_state_change = time.time()
            logger.info(f"Circuit breaker for {self.url} closed after successful test")

    def record_error(self, error: Exception) -> None:
        """Record a request error"""
        self.error_count += 1
        self.last_error = error

        # Update health status based on error count and type
        if self.error_count > 5 and self.success_count == 0:
            self.health = EndpointHealth.UNHEALTHY
        elif self.error_count > 10 and self.success_count / max(self.error_count, 1) < 0.5:
            self.health = EndpointHealth.DEGRADED

        # Circuit breaker logic
        if self.circuit_state == CircuitBreakerState.CLOSED:
            if self.error_count >= self.circuit_failure_threshold:
                self.circuit_state = CircuitBreakerState.OPEN
                self.circuit_last_state_change = time.time()
                logger.warning(f"Circuit breaker for {self.url} opened after {self.error_count} consecutive failures")
        elif self.circuit_state == CircuitBreakerState.HALF_OPEN:
            # Failure in half-open state means the system hasn't recovered
            self.circuit_state = CircuitBreakerState.OPEN
            self.circuit_last_state_change = time.time()
            logger.warning(f"Circuit breaker for {self.url} reopened after test failure")

    def check_circuit_breaker(self) -> bool:
        """
        Check if the circuit breaker allows requests.

        Returns:
            True if requests are allowed, False otherwise
        """
        current_time = time.time()

        if self.circuit_state == CircuitBreakerState.CLOSED:
            # Normal operation
            return True
        elif self.circuit_state == CircuitBreakerState.OPEN:
            # Check if we should try to recover
            if current_time - self.circuit_last_state_change > self.circuit_recovery_timeout:
                self.circuit_state = CircuitBreakerState.HALF_OPEN
                self.circuit_last_state_change = current_time
                logger.info(f"Circuit breaker for {self.url} entering half-open state to test recovery")
                return True
            return False
        elif self.circuit_state == CircuitBreakerState.HALF_OPEN:
            # In testing state, allow limited requests
            return True

        return True  # Default to allow if unknown state

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "url": self.url,
            "priority": self.priority.name,
            "status": self.status.name,
            "health": self.health.name,
            "circuit_state": self.circuit_state.name,
            "last_connected": self.last_connected,
            "error_count": self.error_count,
            "success_count": self.success_count,
            "avg_response_time": self.get_average_response_time(),
            "has_credentials": bool(self.credentials)
        }


class ConnectionConfig:
    """Configuration for a connection to an exchange or data source"""

    def __init__(self,
                 exchange_id: str,
                 endpoints: List[ConnectionEndpoint],
                 name: Optional[str] = None,
                 retry_policy: Optional[Dict[str, Any]] = None,
                 timeout: Optional[Dict[str, float]] = None,
                 rate_limits: Optional[Dict[str, Any]] = None,
                 ssl_verify: bool = True,
                 proxy: Optional[Dict[str, str]] = None,
                 websocket_config: Optional[Dict[str, Any]] = None):
        """
        Initialize a connection configuration.

        Args:
            exchange_id: ID of the exchange or data source
            endpoints: List of connection endpoints
            name: Human-readable name
            retry_policy: Retry policy settings
            timeout: Timeout settings
            rate_limits: Rate limit settings
            ssl_verify: Whether to verify SSL certificates
            proxy: Proxy settings
            websocket_config: WebSocket connection configuration
        """
        self.exchange_id = exchange_id
        self.name = name or exchange_id
        self.endpoints = sorted(endpoints, key=lambda e: e.priority.value)

        # Retry policy settings - defaults are conservative
        self.retry_policy = {
            "retries": 3,  # Number of retries
            "backoff_factor": 0.3,  # Exponential backoff factor
            "status_forcelist": [429, 500, 502, 503, 504],  # Status codes to retry
            "allowed_methods": ["HEAD", "GET", "OPTIONS", "POST", "PUT"]  # Methods to retry
        }
        if retry_policy:
            self.retry_policy.update(retry_policy)

        # Timeout settings - defaults are conservative
        self.timeout = {
            "connect": 10.0,  # Connection timeout
            "read": 30.0,  # Read timeout
            "total": 60.0  # Total timeout
        }
        if timeout:
            self.timeout.update(timeout)

        # Rate limit settings
        self.rate_limits = {
            "requests_per_second": 5.0,  # Default requests per second
            "requests_per_minute": 300,  # Default requests per minute
            "enforce": True,  # Whether to enforce rate limits
            "buffer_factor": 0.8  # Stay under limit by this factor
        }
        if rate_limits:
            self.rate_limits.update(rate_limits)

        # WebSocket configuration
        self.websocket_config = {
            "enabled": False,
            "url": None,
            "ping_interval": 30,
            "reconnect_interval": 5,
            "max_reconnect_attempts": 10
        }
        if websocket_config:
            self.websocket_config.update(websocket_config)

        # Other settings
        self.ssl_verify = ssl_verify
        self.proxy = proxy or {}

    def get_primary_endpoint(self) -> Optional[ConnectionEndpoint]:
        """Get the primary endpoint"""
        for endpoint in self.endpoints:
            if endpoint.priority == ConnectionPriority.PRIMARY:
                return endpoint
        # If no primary, return the first endpoint
        return self.endpoints[0] if self.endpoints else None

    def get_active_endpoints(self) -> List[ConnectionEndpoint]:
        """Get list of active endpoints sorted by priority"""
        # Only return endpoints that are healthy or unknown and with circuit breaker closed or half-open
        return [e for e in self.endpoints if
                (e.health != EndpointHealth.UNHEALTHY) and
                (e.circuit_state != CircuitBreakerState.OPEN)]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "exchange_id": self.exchange_id,
            "name": self.name,
            "endpoints": [endpoint.to_dict() for endpoint in self.endpoints],
            "retry_policy": self.retry_policy,
            "timeout": self.timeout,
            "rate_limits": self.rate_limits,
            "ssl_verify": self.ssl_verify,
            "proxy": self.proxy,
            "websocket_config": self.websocket_config
        }


class RateLimiter:
    """Rate limiter for API requests"""

    def __init__(self,
                 requests_per_second: float = 5.0,
                 requests_per_minute: int = 300,
                 buffer_factor: float = 0.8):
        """
        Initialize the rate limiter.

        Args:
            requests_per_second: Maximum requests per second
            requests_per_minute: Maximum requests per minute
            buffer_factor: Buffer factor to stay under limits
        """
        self.requests_per_second = requests_per_second * buffer_factor
        self.requests_per_minute = requests_per_minute * buffer_factor
        self.last_request_time = 0
        self.request_times = []  # Timestamps of recent requests
        self.lock = threading.RLock()

    def wait_if_needed(self) -> float:
        """
        Wait if necessary to comply with rate limits.

        Returns:
            Time spent waiting in seconds
        """
        with self.lock:
            current_time = time.time()

            # Clean up old requests outside the minute window
            self.request_times = [t for t in self.request_times if current_time - t < 60]

            # Check per-minute rate limit
            if len(self.request_times) >= self.requests_per_minute:
                # Calculate wait time to satisfy per-minute limit
                wait_time_minute = 60 - (current_time - self.request_times[0])
                wait_time_minute = max(0, wait_time_minute)
            else:
                wait_time_minute = 0

            # Check per-second rate limit
            time_since_last = current_time - self.last_request_time
            wait_time_second = 1.0 / self.requests_per_second - time_since_last
            wait_time_second = max(0, wait_time_second)

            # Use the longer wait time
            wait_time = max(wait_time_minute, wait_time_second)

            if wait_time > 0:
                time.sleep(wait_time)
                # Update current time after sleeping
                current_time = time.time()

            # Record this request
            self.last_request_time = current_time
            self.request_times.append(current_time)

            return wait_time

    async def wait_if_needed_async(self) -> float:
        """
        Async version of wait_if_needed.

        Returns:
            Time spent waiting in seconds
        """
        with self.lock:
            current_time = time.time()

            # Clean up old requests outside the minute window
            self.request_times = [t for t in self.request_times if current_time - t < 60]

            # Check per-minute rate limit
            if len(self.request_times) >= self.requests_per_minute:
                # Calculate wait time to satisfy per-minute limit
                wait_time_minute = 60 - (current_time - self.request_times[0])
                wait_time_minute = max(0, wait_time_minute)
            else:
                wait_time_minute = 0

            # Check per-second rate limit
            time_since_last = current_time - self.last_request_time
            wait_time_second = 1.0 / self.requests_per_second - time_since_last
            wait_time_second = max(0, wait_time_second)

            # Use the longer wait time
            wait_time = max(wait_time_minute, wait_time_second)

        # Sleep outside the lock to avoid blocking other requests
        if wait_time > 0:
            await asyncio.sleep(wait_time)
            current_time = time.time()

        with self.lock:
            # Record this request
            self.last_request_time = current_time
            self.request_times.append(current_time)

            return wait_time


class WebSocketManager:
    """
    Manager for WebSocket connections to exchanges.

    Provides persistent WebSocket connections with automatic reconnection
    and message handling.
    """

    def __init__(self, connection_config: ConnectionConfig, message_handler: Callable):
        """
        Initialize the WebSocket manager.

        Args:
            connection_config: Configuration for the connection
            message_handler: Function to handle incoming messages
        """
        self.connection_config = connection_config
        self.message_handler = message_handler
        self.ws = None
        self.running = False
        self.reconnect_attempts = 0
        self.last_message_time = 0
        self.task = None
        self.ping_task = None
        self.lock = threading.Lock()

    async def start(self) -> None:
        """Start the WebSocket connection"""
        with self.lock:
            if self.running:
                logger.warning(f"WebSocket for {self.connection_config.exchange_id} already running")
                return

            self.running = True
            self.reconnect_attempts = 0

            # Start connection in background task
            self.task = asyncio.create_task(self._connection_loop())
            logger.info(f"WebSocket manager for {self.connection_config.exchange_id} started")

    async def stop(self) -> None:
        """Stop the WebSocket connection"""
        with self.lock:
            if not self.running:
                return

            self.running = False

            # Cancel tasks
            if self.task:
                self.task.cancel()

            if self.ping_task:
                self.ping_task.cancel()

            # Close WebSocket
            if self.ws:
                await self.ws.close()
                self.ws = None

            logger.info(f"WebSocket manager for {self.connection_config.exchange_id} stopped")

    async def send(self, message: Union[str, Dict[str, Any]]) -> bool:
        """
        Send a message through the WebSocket.

        Args:
            message: Message to send (string or dict that will be JSON-encoded)

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.running or not self.ws:
            return False

        try:
            if isinstance(message, dict):
                await self.ws.send_json(message)
            else:
                await self.ws.send_str(str(message))
            return True
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {str(e)}")
            return False

    async def _connection_loop(self) -> None:
        """Main connection loop for WebSocket"""
        config = self.connection_config.websocket_config

        while self.running:
            try:
                # Get WebSocket URL
                ws_url = config["url"]
                if not ws_url:
                    # Use HTTP endpoint but replace protocol
                    endpoint = self.connection_config.get_primary_endpoint()
                    if endpoint:
                        http_url = endpoint.url
                        ws_url = http_url.replace('http://', 'ws://').replace('https://', 'wss://')
                    else:
                        raise ValueError(
                            f"No WebSocket URL or endpoint available for {self.connection_config.exchange_id}")

                # Connect to WebSocket
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(
                            ws_url,
                            ssl=None if not self.connection_config.ssl_verify else True,
                            proxy=self.connection_config.proxy.get('http') if self.connection_config.proxy else None,
                            timeout=aiohttp.ClientTimeout(total=self.connection_config.timeout["total"])
                    ) as ws:
                        self.ws = ws
                        self.reconnect_attempts = 0
                        logger.info(f"WebSocket connected to {ws_url}")

                        # Start ping task if needed
                        if config["ping_interval"] > 0:
                            self.ping_task = asyncio.create_task(self._ping_loop(config["ping_interval"]))

                        # Process incoming messages
                        async for msg in ws:
                            if not self.running:
                                break

                            self.last_message_time = time.time()

                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    # Parse and handle message
                                    data = json.loads(msg.data)
                                    await self.message_handler(data)
                                except Exception as e:
                                    logger.error(f"Error handling WebSocket message: {str(e)}")
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                logger.error(f"WebSocket error: {ws.exception()}")
                                break
                            elif msg.type == aiohttp.WSMsgType.CLOSED:
                                logger.info("WebSocket connection closed")
                                break

                        # WebSocket closed, clean up
                        if self.ping_task:
                            self.ping_task.cancel()
                            self.ping_task = None
                        self.ws = None

            except Exception as e:
                logger.error(f"WebSocket connection error: {str(e)}")
                self.ws = None

                # Increment reconnect attempts
                self.reconnect_attempts += 1

                # Check if we exceeded max reconnect attempts
                if config["max_reconnect_attempts"] > 0 and self.reconnect_attempts > config["max_reconnect_attempts"]:
                    logger.error(f"Exceeded maximum reconnect attempts ({config['max_reconnect_attempts']})")
                    self.running = False
                    break

                # Wait before reconnecting
                wait_time = config["reconnect_interval"] * min(self.reconnect_attempts,
                                                               5)  # Exponential backoff with cap
                logger.info(f"Reconnecting in {wait_time}s (attempt {self.reconnect_attempts})")
                await asyncio.sleep(wait_time)

        # Ensure WebSocket is closed
        if self.ws:
            await self.ws.close()
            self.ws = None

    async def _ping_loop(self, interval: int) -> None:
        """
        Loop to send periodic pings to keep connection alive.

        Args:
            interval: Ping interval in seconds
        """
        while self.running and self.ws:
            try:
                await asyncio.sleep(interval)
                if self.ws and self.ws.closed:
                    break
                await self.ws.ping()
            except Exception as e:
                logger.error(f"Error sending ping: {str(e)}")
                break


class ConnectivityManager:
    """
    Manager for connectivity to exchanges and data sources.

    Provides:
    - Connection pooling and management
    - Automatic reconnection and failover
    - Rate limiting and backpressure
    - Health monitoring and circuit breaking
    - WebSocket support for streaming data
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the connectivity manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._connections: Dict[str, ConnectionConfig] = {}
        self._sessions: Dict[str, requests.Session] = {}
        self._rate_limiters: Dict[str, RateLimiter] = {}
        self._websocket_managers: Dict[str, WebSocketManager] = {}
        self._lock = threading.RLock()
        self._health_check_interval = self.config.get("health_check_interval", 60)  # seconds
        self._health_checker = None
        self._running = False
        self._event_bus = get_event_bus()
        self._async_session = None  # For async requests

        # Configure global connection defaults
        self._configure_global_http()

        # Initialize connections from config
        self._init_connections()

        logger.info(f"ConnectivityManager initialized with {len(self._connections)} connections")

    def _configure_global_http(self) -> None:
        """Configure global HTTP settings for better performance"""
        # Increase max pool size for urllib3
        urllib3.PoolManager(maxsize=100, block=False)

        # Set up global SSL context
        ssl_context = ssl.create_default_context()

        # Configure urllib3 to use modern ciphers
        if hasattr(ssl, 'PROTOCOL_TLS'):
            ssl_context.options |= getattr(ssl, 'OP_NO_SSLv2', 0)
            ssl_context.options |= getattr(ssl, 'OP_NO_SSLv3', 0)
            ssl_context.options |= getattr(ssl, 'OP_NO_TLSv1', 0)
            ssl_context.options |= getattr(ssl, 'OP_NO_TLSv1_1', 0)

    def _init_connections(self) -> None:
        """Initialize connections from configuration"""
        connections_config = self.config.get("connections", {})

        for exchange_id, conn_config in connections_config.items():
            try:
                # Skip if disabled
                if not conn_config.get("enabled", True):
                    continue

                # Extract endpoints
                endpoints = []
                for endpoint_config in conn_config.get("endpoints", []):
                    endpoint = ConnectionEndpoint(
                        url=endpoint_config["url"],
                        priority=ConnectionPriority[endpoint_config.get("priority", "PRIMARY")],
                        credentials=endpoint_config.get("credentials"),
                        options=endpoint_config.get("options")
                    )
                    endpoints.append(endpoint)

                # Create connection config
                connection = ConnectionConfig(
                    exchange_id=exchange_id,
                    name=conn_config.get("name", exchange_id),
                    endpoints=endpoints,
                    retry_policy=conn_config.get("retry_policy"),
                    timeout=conn_config.get("timeout"),
                    rate_limits=conn_config.get("rate_limits"),
                    ssl_verify=conn_config.get("ssl_verify", True),
                    proxy=conn_config.get("proxy"),
                    websocket_config=conn_config.get("websocket_config")
                )

                self._connections[exchange_id] = connection

                # Create rate limiter
                rate_limits = connection.rate_limits
                self._rate_limiters[exchange_id] = RateLimiter(
                    requests_per_second=rate_limits["requests_per_second"],
                    requests_per_minute=rate_limits["requests_per_minute"],
                    buffer_factor=rate_limits["buffer_factor"]
                )

                # Initialize WebSocket if enabled
                if connection.websocket_config.get("enabled", False):
                    self._init_websocket(exchange_id, connection)

                logger.info(f"Initialized connection config for {exchange_id} with {len(endpoints)} endpoints")

            except Exception as e:
                logger.error(f"Error initializing connection for {exchange_id}: {str(e)}")

    def _init_websocket(self, exchange_id: str, connection: ConnectionConfig) -> None:
        """
        Initialize WebSocket for an exchange.

        Args:
            exchange_id: Exchange ID
            connection: Connection configuration
        """
        try:
            # Create message handler
            async def message_handler(data: Dict[str, Any]) -> None:
                try:
                    # Publish message to event bus
                    event = create_event(
                        f"websocket.{exchange_id}.message",
                        {
                            "exchange_id": exchange_id,
                            "data": data,
                            "timestamp": time.time()
                        }
                    )
                    self._event_bus.publish(event)
                except Exception as e:
                    logger.error(f"Error handling WebSocket message for {exchange_id}: {str(e)}")

            # Create WebSocket manager
            self._websocket_managers[exchange_id] = WebSocketManager(connection, message_handler)

            logger.info(f"Initialized WebSocket for {exchange_id}")
        except Exception as e:
            logger.error(f"Error initializing WebSocket for {exchange_id}: {str(e)}")

    def start(self) -> None:
        """Start the connectivity manager and health checks"""
        with self._lock:
            if self._running:
                logger.warning("ConnectivityManager is already running")
                return

            self._running = True

            # Start health checks
            self._health_checker = threading.Thread(
                target=self._health_check_loop,
                daemon=True,
                name="ConnectivityHealthCheck"
            )
            self._health_checker.start()

            # Start all WebSocket connections
            for exchange_id, ws_manager in self._websocket_managers.items():
                # Use asyncio.run to run coroutines from sync code
                if self._connections[exchange_id].websocket_config.get("enabled", False):
                    # Use a separate thread to start each WebSocket
                    threading.Thread(
                        target=lambda: asyncio.run(ws_manager.start()),
                        daemon=True,
                        name=f"WebSocket-{exchange_id}"
                    ).start()

            # Publish event
            event = create_event(
                EventTopics.COMPONENT_STARTED,
                {
                    "component": "connectivity_manager",
                    "exchanges": list(self._connections.keys()),
                    "timestamp": time.time()
                }
            )
            self._event_bus.publish(event)

            logger.info("ConnectivityManager started")

    def stop(self) -> None:
        """Stop the connectivity manager and close all connections"""
        with self._lock:
            if not self._running:
                logger.warning("ConnectivityManager is not running")
                return

            self._running = False

            # Close all sessions
            for exchange_id, session in self._sessions.items():
                try:
                    session.close()
                except Exception as e:
                    logger.error(f"Error closing session for {exchange_id}: {str(e)}")

            self._sessions.clear()

            # Stop all WebSocket connections
            for exchange_id, ws_manager in self._websocket_managers.items():
                # Use asyncio.run to run coroutines from sync code
                try:
                    asyncio.run(ws_manager.stop())
                except Exception as e:
                    logger.error(f"Error stopping WebSocket for {exchange_id}: {str(e)}")

            # Publish event
            event = create_event(
                EventTopics.COMPONENT_STOPPED,
                {
                    "component": "connectivity_manager",
                    "timestamp": time.time()
                }
            )
            self._event_bus.publish(event)

            logger.info("ConnectivityManager stopped")

    def get_session(self, exchange_id: str) -> requests.Session:
        """
        Get a session for an exchange.

        Args:
            exchange_id: Exchange ID

        Returns:
            Session for the exchange

        Raises:
            ValueError: If the exchange is not configured
        """
        with self._lock:
            # Check if connection exists
            if exchange_id not in self._connections:
                raise ValueError(f"Exchange {exchange_id} not configured")

            # Return existing session if available
            if exchange_id in self._sessions:
                return self._sessions[exchange_id]

            # Create new session
            connection = self._connections[exchange_id]
            session = requests.Session()

            # Configure session
            if not connection.ssl_verify:
                session.verify = False

            # Configure proxies
            if connection.proxy:
                session.proxies.update(connection.proxy)

            # Configure retry policy
            retry_policy = connection.retry_policy
            retries = Retry(
                total=retry_policy["retries"],
                backoff_factor=retry_policy["backoff_factor"],
                status_forcelist=retry_policy["status_forcelist"],
                allowed_methods=retry_policy["allowed_methods"]
            )
            adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=100)
            session.mount('http://', adapter)
            session.mount('https://', adapter)

            # Configure timeout
            session.timeout = (
                connection.timeout["connect"],
                connection.timeout["read"]
            )

            # Add default headers
            session.headers.update({
                'User-Agent': self.config.get('user_agent', 'ML-Trading-System/1.0'),
                'Accept': 'application/json'
            })

            # Store session
            self._sessions[exchange_id] = session

            logger.debug(f"Created session for {exchange_id}")
            return session

    def request(self,
                exchange_id: str,
                method: str,
                endpoint: str,
                params: Optional[Dict[str, Any]] = None,
                data: Optional[Union[Dict[str, Any], str]] = None,
                headers: Optional[Dict[str, str]] = None,
                auth: Optional[Tuple[str, str]] = None,
                timeout: Optional[float] = None,
                enforce_rate_limit: bool = True) -> requests.Response:
        """
        Make an HTTP request to an exchange.

        Args:
            exchange_id: Exchange ID
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            data: Request data
            headers: Request headers
            auth: Authentication tuple
            timeout: Request timeout
            enforce_rate_limit: Whether to enforce rate limiting

        Returns:
            Response object

        Raises:
            ValueError: If the exchange is not configured
            RequestException: If the request fails
        """
        if exchange_id not in self._connections:
            raise ValueError(f"Exchange {exchange_id} not configured")

        # Get connection config
        connection = self._connections[exchange_id]

        # Apply rate limiting if enabled
        if enforce_rate_limit and connection.rate_limits["enforce"]:
            wait_time = self._rate_limiters[exchange_id].wait_if_needed()
            if wait_time > 0:
                logger.debug(f"Rate limit enforced for {exchange_id}, waited {wait_time:.2f}s")

        # Get endpoint URL
        endpoints = connection.get_active_endpoints()
        if not endpoints:
            raise RequestException(f"No active endpoints available for {exchange_id}")

        # Start with the primary endpoint
        endpoint_obj = endpoints[0]
        base_url = endpoint_obj.url

        # Check circuit breaker
        if not endpoint_obj.check_circuit_breaker():
            # If primary is open, try to find another endpoint
            alternative_endpoints = [e for e in endpoints[1:] if e.check_circuit_breaker()]
            if not alternative_endpoints:
                raise RequestException(f"Circuit breaker open for all endpoints of {exchange_id}")
            endpoint_obj = alternative_endpoints[0]
            base_url = endpoint_obj.url
            logger.info(f"Using alternative endpoint {endpoint_obj.url} due to circuit breaker")

        # Construct full URL
        url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        # Get session
        session = self.get_session(exchange_id)

        # Set timeout
        request_timeout = timeout or connection.timeout["total"]

        # Make the request with automatic failover
        last_exception = None
        start_time = time.perf_counter()

        for i, endpoint_obj in enumerate(endpoints):
            if i > 0:
                # Using failover endpoint
                logger.warning(f"Failover to endpoint {i} for {exchange_id}")
                base_url = endpoint_obj.url
                url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"

                # Check circuit breaker for failover endpoint
                if not endpoint_obj.check_circuit_breaker():
                    logger.warning(f"Circuit breaker open for failover endpoint {endpoint_obj.url}, skipping")
                    continue

            try:
                # Make the request
                response = session.request(
                    method=method,
                    url=url,
                    params=params,
                    data=data,
                    headers=headers,
                    auth=auth,
                    timeout=request_timeout
                )

                # Calculate response time
                response_time_ms = (time.perf_counter() - start_time) * 1000

                # Record success
                endpoint_obj.record_success(response_time_ms)
                endpoint_obj.status = ConnectionStatus.CONNECTED
                endpoint_obj.last_connected = time.time()

                # Check if response indicates rate limiting
                if response.status_code == 429:
                    logger.warning(f"Rate limit hit for {exchange_id}")
                    endpoint_obj.status = ConnectionStatus.RATE_LIMITED

                    # Extract rate limit info if available
                    rate_limit_reset = response.headers.get('Retry-After')
                    if rate_limit_reset:
                        try:
                            wait_time = float(rate_limit_reset)
                            logger.info(f"Rate limit reset in {wait_time}s for {exchange_id}")
                            time.sleep(wait_time)
                        except ValueError:
                            # Header might be a date/time string
                            pass

                    # Try next endpoint if available
                    if i < len(endpoints) - 1:
                        continue

                    # Still return the response if no more endpoints
                    return response

                # Check if response indicates server error
                if response.status_code >= 500:
                    logger.warning(f"Server error {response.status_code} for {exchange_id}")
                    endpoint_obj.health = EndpointHealth.DEGRADED
                    endpoint_obj.record_error(Exception(f"Server error: {response.status_code}"))

                    # Try next endpoint if available
                    if i < len(endpoints) - 1:
                        continue

                # Emit metrics event
                self._emit_request_metrics(exchange_id, endpoint_obj.url, method, endpoint,
                                           response.status_code, response_time_ms)

                # Response was successful or we've exhausted all endpoints
                return response

            except (ConnectionError, Timeout, SSLError) as e:
                # Network-level error
                endpoint_obj.record_error(e)
                endpoint_obj.status = ConnectionStatus.ERROR
                last_exception = e

                # Emit failure event
                self._emit_request_failure(exchange_id, endpoint_obj.url, method, endpoint, str(e))

                # If this is the last endpoint, re-raise
                if i == len(endpoints) - 1:
                    logger.error(f"All endpoints failed for {exchange_id}: {str(e)}")
                    raise

                logger.warning(f"Connection error for {exchange_id}, trying next endpoint: {str(e)}")

        # If we get here, all endpoints failed
        if last_exception:
            raise last_exception
        else:
            raise RequestException(f"All endpoints failed for {exchange_id}")

    def _emit_request_metrics(self, exchange_id: str, url: str, method: str,
                              endpoint: str, status_code: int, response_time_ms: float) -> None:
        """
        Emit metrics for a request.

        Args:
            exchange_id: Exchange ID
            url: Endpoint URL
            method: HTTP method
            endpoint: API endpoint
            status_code: Response status code
            response_time_ms: Response time in milliseconds
        """
        try:
            # Create event
            event = create_event(
                EventTopics.REQUEST_METRICS,
                {
                    "exchange_id": exchange_id,
                    "url": url,
                    "method": method,
                    "endpoint": endpoint,
                    "status_code": status_code,
                    "response_time_ms": response_time_ms,
                    "timestamp": time.time()
                }
            )
            self._event_bus.publish(event)
        except Exception as e:
            logger.error(f"Error emitting request metrics: {str(e)}")

    def _emit_request_failure(self, exchange_id: str, url: str, method: str,
                              endpoint: str, error: str) -> None:
        """
        Emit event for request failure.

        Args:
            exchange_id: Exchange ID
            url: Endpoint URL
            method: HTTP method
            endpoint: API endpoint
            error: Error message
        """
        try:
            # Create event
            event = create_event(
                EventTopics.REQUEST_FAILURE,
                {
                    "exchange_id": exchange_id,
                    "url": url,
                    "method": method,
                    "endpoint": endpoint,
                    "error": error,
                    "timestamp": time.time()
                },
                priority=EventPriority.HIGH
            )
            self._event_bus.publish(event)
        except Exception as e:
            logger.error(f"Error emitting request failure event: {str(e)}")

    def get(self,
            exchange_id: str,
            endpoint: str,
            params: Optional[Dict[str, Any]] = None,
            **kwargs) -> requests.Response:
        """
        Make a GET request to an exchange.

        Args:
            exchange_id: Exchange ID
            endpoint: API endpoint
            params: Query parameters
            **kwargs: Additional arguments for request()

        Returns:
            Response object
        """
        return self.request(exchange_id, "GET", endpoint, params=params, **kwargs)

    def post(self,
             exchange_id: str,
             endpoint: str,
             data: Optional[Union[Dict[str, Any], str]] = None,
             **kwargs) -> requests.Response:
        """
        Make a POST request to an exchange.

        Args:
            exchange_id: Exchange ID
            endpoint: API endpoint
            data: Request data
            **kwargs: Additional arguments for request()

        Returns:
            Response object
        """
        return self.request(exchange_id, "POST", endpoint, data=data, **kwargs)

    def put(self,
            exchange_id: str,
            endpoint: str,
            data: Optional[Union[Dict[str, Any], str]] = None,
            **kwargs) -> requests.Response:
        """
        Make a PUT request to an exchange.

        Args:
            exchange_id: Exchange ID
            endpoint: API endpoint
            data: Request data
            **kwargs: Additional arguments for request()

        Returns:
            Response object
        """
        return self.request(exchange_id, "PUT", endpoint, data=data, **kwargs)

    def delete(self,
               exchange_id: str,
               endpoint: str,
               **kwargs) -> requests.Response:
        """
        Make a DELETE request to an exchange.

        Args:
            exchange_id: Exchange ID
            endpoint: API endpoint
            **kwargs: Additional arguments for request()

        Returns:
            Response object
        """
        return self.request(exchange_id, "DELETE", endpoint, **kwargs)

    def get_json(self,
                 exchange_id: str,
                 endpoint: str,
                 params: Optional[Dict[str, Any]] = None,
                 **kwargs) -> Dict[str, Any]:
        """
        Make a GET request and parse JSON response.

        Args:
            exchange_id: Exchange ID
            endpoint: API endpoint
            params: Query parameters
            **kwargs: Additional arguments for request()

        Returns:
            Parsed JSON response
        """
        response = self.get(exchange_id, endpoint, params=params, **kwargs)
        response.raise_for_status()
        return response.json()

    def post_json(self,
                  exchange_id: str,
                  endpoint: str,
                  data: Optional[Dict[str, Any]] = None,
                  **kwargs) -> Dict[str, Any]:
        """
        Make a POST request and parse JSON response.

        Args:
            exchange_id: Exchange ID
            endpoint: API endpoint
            data: Request data
            **kwargs: Additional arguments for request()

        Returns:
            Parsed JSON response
        """
        kwargs.setdefault('headers', {})
        kwargs['headers']['Content-Type'] = 'application/json'
        json_data = json.dumps(data) if data else None
        response = self.post(exchange_id, endpoint, data=json_data, **kwargs)
        response.raise_for_status()
        return response.json()

    async def get_async_session(self) -> aiohttp.ClientSession:
        """
        Get async HTTP session.

        Returns:
            Async HTTP session
        """
        if self._async_session is None or self._async_session.closed:
            # Create timeout
            timeout = aiohttp.ClientTimeout(
                total=self.config.get("timeout", {}).get("total", 60),
                connect=self.config.get("timeout", {}).get("connect", 10),
                sock_connect=self.config.get("timeout", {}).get("connect", 10),
                sock_read=self.config.get("timeout", {}).get("read", 30)
            )

            # Create session
            self._async_session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'User-Agent': self.config.get('user_agent', 'ML-Trading-System/1.0'),
                    'Accept': 'application/json'
                }
            )

        return self._async_session

    async def request_async(self,
                            exchange_id: str,
                            method: str,
                            endpoint: str,
                            params: Optional[Dict[str, Any]] = None,
                            data: Optional[Union[Dict[str, Any], str]] = None,
                            headers: Optional[Dict[str, str]] = None,
                            auth: Optional[Tuple[str, str]] = None,
                            timeout: Optional[float] = None,
                            enforce_rate_limit: bool = True) -> Dict[str, Any]:
        """
        Make an async HTTP request to an exchange.

        Args:
            exchange_id: Exchange ID
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            data: Request data
            headers: Request headers
            auth: Authentication tuple
            timeout: Request timeout
            enforce_rate_limit: Whether to enforce rate limiting

        Returns:
            Parsed JSON response

        Raises:
            ValueError: If the exchange is not configured
            aiohttp.ClientError: If the request fails
        """
        if exchange_id not in self._connections:
            raise ValueError(f"Exchange {exchange_id} not configured")

        # Get connection config
        connection = self._connections[exchange_id]

        # Apply rate limiting if enabled
        if enforce_rate_limit and connection.rate_limits["enforce"]:
            # Use async rate limiter
            await self._rate_limiters[exchange_id].wait_if_needed_async()

        # Get endpoint URL
        endpoints = connection.get_active_endpoints()
        if not endpoints:
            raise aiohttp.ClientError(f"No active endpoints available for {exchange_id}")

        # Start with the primary endpoint
        endpoint_obj = endpoints[0]
        base_url = endpoint_obj.url

        # Check circuit breaker
        if not endpoint_obj.check_circuit_breaker():
            # If primary is open, try to find another endpoint
            alternative_endpoints = [e for e in endpoints[1:] if e.check_circuit_breaker()]
            if not alternative_endpoints:
                raise aiohttp.ClientError(f"Circuit breaker open for all endpoints of {exchange_id}")
            endpoint_obj = alternative_endpoints[0]
            base_url = endpoint_obj.url
            logger.info(f"Using alternative endpoint {endpoint_obj.url} due to circuit breaker")

        # Construct full URL
        url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        # Set timeout
        request_timeout = timeout or connection.timeout["total"]

        # Prepare JSON data if needed
        json_data = None
        if data and isinstance(data, dict):
            json_data = data
            data = None

        # Make the request with automatic failover
        last_exception = None
        start_time = time.perf_counter()

        # Get session
        session = await self.get_async_session()

        for i, endpoint_obj in enumerate(endpoints):
            if i > 0:
                # Using failover endpoint
                logger.warning(f"Failover to endpoint {i} for {exchange_id}")
                base_url = endpoint_obj.url
                url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"

                # Check circuit breaker for failover endpoint
                if not endpoint_obj.check_circuit_breaker():
                    logger.warning(f"Circuit breaker open for failover endpoint {endpoint_obj.url}, skipping")
                    continue

            try:
                # Create auth if provided
                aiohttp_auth = aiohttp.BasicAuth(*auth) if auth else None

                # Make the request
                async with session.request(
                        method=method,
                        url=url,
                        params=params,
                        data=data,
                        json=json_data,
                        headers=headers,
                        auth=aiohttp_auth,
                        ssl=None if not connection.ssl_verify else True,
                        timeout=aiohttp.ClientTimeout(total=request_timeout)
                ) as response:
                    # Calculate response time
                    response_time_ms = (time.perf_counter() - start_time) * 1000

                    # Record success
                    endpoint_obj.record_success(response_time_ms)
                    endpoint_obj.status = ConnectionStatus.CONNECTED
                    endpoint_obj.last_connected = time.time()

                    # Check if response indicates rate limiting
                    if response.status == 429:
                        logger.warning(f"Rate limit hit for {exchange_id}")
                        endpoint_obj.status = ConnectionStatus.RATE_LIMITED

                        # Extract rate limit info if available
                        rate_limit_reset = response.headers.get('Retry-After')
                        if rate_limit_reset:
                            try:
                                wait_time = float(rate_limit_reset)
                                logger.info(f"Rate limit reset in {wait_time}s for {exchange_id}")
                                await asyncio.sleep(wait_time)
                            except ValueError:
                                # Header might be a date/time string
                                pass

                        # Try next endpoint if available
                        if i < len(endpoints) - 1:
                            continue

                    # Check if response indicates server error
                    if response.status >= 500:
                        logger.warning(f"Server error {response.status} for {exchange_id}")
                        endpoint_obj.health = EndpointHealth.DEGRADED
                        endpoint_obj.record_error(Exception(f"Server error: {response.status}"))

                        # Try next endpoint if available
                        if i < len(endpoints) - 1:
                            continue

                    # Emit metrics event
                    self._emit_request_metrics(exchange_id, endpoint_obj.url, method, endpoint,
                                               response.status, response_time_ms)

                    # Raise exception for client errors
                    if response.status >= 400:
                        response.raise_for_status()

                    # Parse JSON response
                    return await response.json()

            except aiohttp.ClientError as e:
                # Network-level error
                endpoint_obj.record_error(e)
                endpoint_obj.status = ConnectionStatus.ERROR
                last_exception = e

                # Emit failure event
                self._emit_request_failure(exchange_id, endpoint_obj.url, method, endpoint, str(e))

                # If this is the last endpoint, re-raise
                if i == len(endpoints) - 1:
                    logger.error(f"All endpoints failed for {exchange_id}: {str(e)}")
                    raise

                logger.warning(f"Connection error for {exchange_id}, trying next endpoint: {str(e)}")

        # If we get here, all endpoints failed
        if last_exception:
            raise last_exception
        else:
            raise aiohttp.ClientError(f"All endpoints failed for {exchange_id}")

    async def get_async(self,
                        exchange_id: str,
                        endpoint: str,
                        params: Optional[Dict[str, Any]] = None,
                        **kwargs) -> Dict[str, Any]:
        """
        Make an async GET request to an exchange.

        Args:
            exchange_id: Exchange ID
            endpoint: API endpoint
            params: Query parameters
            **kwargs: Additional arguments for request_async()

        Returns:
            Parsed JSON response
        """
        return await self.request_async(exchange_id, "GET", endpoint, params=params, **kwargs)

    async def post_async(self,
                         exchange_id: str,
                         endpoint: str,
                         data: Optional[Union[Dict[str, Any], str]] = None,
                         **kwargs) -> Dict[str, Any]:
        """
        Make an async POST request to an exchange.

        Args:
            exchange_id: Exchange ID
            endpoint: API endpoint
            data: Request data
            **kwargs: Additional arguments for request_async()

        Returns:
            Parsed JSON response
        """
        return await self.request_async(exchange_id, "POST", endpoint, data=data, **kwargs)

    async def websocket_send(self,
                             exchange_id: str,
                             message: Union[str, Dict[str, Any]]) -> bool:
        """
        Send a message through a WebSocket connection.

        Args:
            exchange_id: Exchange ID
            message: Message to send

        Returns:
            True if sent successfully, False otherwise

        Raises:
            ValueError: If the exchange is not configured or WebSocket is not enabled
        """
        if exchange_id not in self._connections:
            raise ValueError(f"Exchange {exchange_id} not configured")

        if exchange_id not in self._websocket_managers:
            raise ValueError(f"WebSocket not enabled for {exchange_id}")

        # Send message
        return await self._websocket_managers[exchange_id].send(message)

    def get_connection_status(self, exchange_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a connection.

        Args:
            exchange_id: Exchange ID

        Returns:
            Dictionary with connection status or None if not found
        """
        with self._lock:
            if exchange_id not in self._connections:
                return None

            connection = self._connections[exchange_id]

            # Check if WebSocket is enabled
            websocket_enabled = connection.websocket_config.get("enabled", False)
            websocket_running = False
            if websocket_enabled and exchange_id in self._websocket_managers:
                websocket_running = self._websocket_managers[exchange_id].running

            return {
                "exchange_id": exchange_id,
                "name": connection.name,
                "endpoints": [endpoint.to_dict() for endpoint in connection.endpoints],
                "active_endpoints": len(connection.get_active_endpoints()),
                "total_endpoints": len(connection.endpoints),
                "websocket_enabled": websocket_enabled,
                "websocket_running": websocket_running
            }

    def get_all_connection_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all connections.

        Returns:
            Dictionary mapping exchange IDs to connection status
        """
        with self._lock:
            return {exchange_id: self.get_connection_status(exchange_id)
                    for exchange_id in self._connections}

    def health_check(self, exchange_id: str) -> bool:
        """
        Perform a health check for an exchange.

        Args:
            exchange_id: Exchange ID

        Returns:
            True if healthy, False otherwise
        """
        if exchange_id not in self._connections:
            return False

        connection = self._connections[exchange_id]

        # Get health endpoint from config
        health_endpoint = self.config.get("health_check_path", "/api/v1/ping")

        # Check if custom health check endpoint is specified
        if "health_check_path" in connection.websocket_config:
            health_endpoint = connection.websocket_config["health_check_path"]

        # Check each endpoint
        any_healthy = False
        for endpoint in connection.endpoints:
            try:
                # Skip if circuit breaker is open
                if endpoint.circuit_state == CircuitBreakerState.OPEN:
                    logger.debug(f"Skipping health check for {endpoint.url} due to open circuit breaker")
                    continue

                # Construct health check URL
                base_url = endpoint.url
                url = f"{base_url.rstrip('/')}/{health_endpoint.lstrip('/')}"

                # Make a simple request to check health
                session = self.get_session(exchange_id)
                start_time = time.perf_counter()

                response = session.get(
                    url=url,
                    timeout=connection.timeout["connect"]
                )

                # Calculate response time
                response_time_ms = (time.perf_counter() - start_time) * 1000

                # Update endpoint health based on response
                if response.status_code < 400:
                    endpoint.health = EndpointHealth.HEALTHY
                    endpoint.status = ConnectionStatus.CONNECTED
                    endpoint.record_success(response_time_ms)
                    endpoint.last_connected = time.time()
                    any_healthy = True
                elif response.status_code == 429:
                    endpoint.health = EndpointHealth.DEGRADED
                    endpoint.status = ConnectionStatus.RATE_LIMITED
                    endpoint.record_error(Exception(f"Rate limited: {response.status_code}"))
                else:
                    endpoint.health = EndpointHealth.DEGRADED
                    endpoint.status = ConnectionStatus.ERROR
                    endpoint.record_error(Exception(f"Unhealthy response: {response.status_code}"))

            except Exception as e:
                endpoint.health = EndpointHealth.UNHEALTHY
                endpoint.status = ConnectionStatus.ERROR
                endpoint.record_error(e)
                logger.warning(f"Health check failed for {exchange_id} endpoint {endpoint.url}: {str(e)}")

        # Publish health check event
        self._emit_health_check_event(exchange_id, any_healthy)

        # Overall exchange health is based on whether any endpoints are healthy
        return any_healthy

    def _emit_health_check_event(self, exchange_id: str, healthy: bool) -> None:
        """
        Emit health check event.

        Args:
            exchange_id: Exchange ID
            healthy: Whether the exchange is healthy
        """
        try:
            # Create event
            event = create_event(
                EventTopics.HEALTH_CHECK,
                {
                    "component": "connectivity_manager",
                    "exchange_id": exchange_id,
                    "healthy": healthy,
                    "timestamp": time.time()
                }
            )
            self._event_bus.publish(event)
        except Exception as e:
            logger.error(f"Error emitting health check event: {str(e)}")

    def _health_check_loop(self) -> None:
        """Background thread to periodically check connection health"""
        logger.debug("Health check loop started")

        while self._running:
            try:
                # Check health of all connections
                for exchange_id in list(self._connections.keys()):
                    if not self._running:
                        break

                    try:
                        self.health_check(exchange_id)
                    except Exception as e:
                        logger.error(f"Error in health check for {exchange_id}: {str(e)}")

                    # Sleep briefly between checks to avoid hammering
                    time.sleep(0.5)

                # Sleep until next health check cycle
                for _ in range(int(self._health_check_interval * 2)):
                    if not self._running:
                        break
                    time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error in health check loop: {str(e)}")
                time.sleep(5)  # Sleep longer if there's an error

    def add_connection(self,
                       exchange_id: str,
                       name: str,
                       endpoints: List[Dict[str, Any]],
                       **kwargs) -> bool:
        """
        Add a new connection configuration.

        Args:
            exchange_id: Exchange ID
            name: Human-readable name
            endpoints: List of endpoint configurations
            **kwargs: Additional connection settings

        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            # Skip if already exists
            if exchange_id in self._connections:
                logger.warning(f"Connection {exchange_id} already exists")
                return False

            try:
                # Create endpoints
                endpoint_objs = []
                for endpoint_config in endpoints:
                    endpoint = ConnectionEndpoint(
                        url=endpoint_config["url"],
                        priority=ConnectionPriority[endpoint_config.get("priority", "PRIMARY")],
                        credentials=endpoint_config.get("credentials"),
                        options=endpoint_config.get("options")
                    )
                    endpoint_objs.append(endpoint)

                # Create connection config
                connection = ConnectionConfig(
                    exchange_id=exchange_id,
                    name=name,
                    endpoints=endpoint_objs,
                    **kwargs
                )

                self._connections[exchange_id] = connection

                # Create rate limiter
                rate_limits = connection.rate_limits
                self._rate_limiters[exchange_id] = RateLimiter(
                    requests_per_second=rate_limits["requests_per_second"],
                    requests_per_minute=rate_limits["requests_per_minute"],
                    buffer_factor=rate_limits["buffer_factor"]
                )

                # Initialize WebSocket if enabled
                if connection.websocket_config.get("enabled", False):
                    self._init_websocket(exchange_id, connection)

                    # Start WebSocket if system is running
                    if self._running:
                        threading.Thread(
                            target=lambda: asyncio.run(self._websocket_managers[exchange_id].start()),
                            daemon=True,
                            name=f"WebSocket-{exchange_id}"
                        ).start()

                logger.info(f"Added connection config for {exchange_id} with {len(endpoint_objs)} endpoints")

                # Publish event
                event = create_event(
                    EventTopics.CONNECTIVITY_ADDED,
                    {
                        "exchange_id": exchange_id,
                        "name": name,
                        "endpoint_count": len(endpoint_objs),
                        "websocket_enabled": connection.websocket_config.get("enabled", False),
                        "timestamp": time.time()
                    }
                )
                self._event_bus.publish(event)

                return True

            except Exception as e:
                logger.error(f"Error adding connection for {exchange_id}: {str(e)}")
                return False

    def remove_connection(self, exchange_id: str) -> bool:
        """
        Remove a connection configuration.

        Args:
            exchange_id: Exchange ID

        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            if exchange_id not in self._connections:
                logger.warning(f"Connection {exchange_id} not found")
                return False

            try:
                # Close session if exists
                if exchange_id in self._sessions:
                    self._sessions[exchange_id].close()
                    del self._sessions[exchange_id]

                # Stop WebSocket if exists
                if exchange_id in self._websocket_managers:
                    try:
                        asyncio.run(self._websocket_managers[exchange_id].stop())
                    except Exception as e:
                        logger.error(f"Error stopping WebSocket for {exchange_id}: {str(e)}")
                    del self._websocket_managers[exchange_id]

                # Remove connection
                del self._connections[exchange_id]

                # Remove rate limiter
                if exchange_id in self._rate_limiters:
                    del self._rate_limiters[exchange_id]

                # Publish event
                event = create_event(
                    EventTopics.CONNECTIVITY_REMOVED,
                    {
                        "exchange_id": exchange_id,
                        "timestamp": time.time()
                    }
                )
                self._event_bus.publish(event)

                logger.info(f"Removed connection config for {exchange_id}")
                return True

            except Exception as e:
                logger.error(f"Error removing connection for {exchange_id}: {str(e)}")
                return False

    def update_credentials(self, exchange_id: str, credentials: Dict[str, str]) -> bool:
        """
        Update credentials for an exchange.

        Args:
            exchange_id: Exchange ID
            credentials: New credentials

        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            if exchange_id not in self._connections:
                logger.warning(f"Connection {exchange_id} not found")
                return False

            try:
                # Update credentials for each endpoint
                for endpoint in self._connections[exchange_id].endpoints:
                    endpoint.credentials = credentials.copy()

                # Force re-creation of session to apply new credentials
                if exchange_id in self._sessions:
                    self._sessions[exchange_id].close()
                    del self._sessions[exchange_id]

                # Force reconnect of WebSocket to apply new credentials
                if exchange_id in self._websocket_managers:
                    try:
                        # Stop and restart WebSocket
                        asyncio.run(self._websocket_managers[exchange_id].stop())
                        asyncio.run(self._websocket_managers[exchange_id].start())
                    except Exception as e:
                        logger.error(f"Error restarting WebSocket for {exchange_id}: {str(e)}")

                logger.info(f"Updated credentials for {exchange_id}")
                return True

            except Exception as e:
                logger.error(f"Error updating credentials for {exchange_id}: {str(e)}")
                return False

    def execute_command(self, command: str, exchange_id: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a command for an exchange.

        Args:
            command: Command to execute
            exchange_id: Exchange ID
            params: Command parameters

        Returns:
            Command result
        """
        if params is None:
            params = {}

        result = {
            "success": False,
            "command": command,
            "exchange_id": exchange_id
        }

        try:
            if command == "status":
                # Get connection status
                status = self.get_connection_status(exchange_id)
                if status:
                    result["success"] = True
                    result["status"] = status
                else:
                    result["error"] = f"Connection {exchange_id} not found"

            elif command == "health_check":
                # Perform health check
                if exchange_id not in self._connections:
                    result["error"] = f"Connection {exchange_id} not found"
                else:
                    healthy = self.health_check(exchange_id)
                    result["success"] = True
                    result["healthy"] = healthy

            elif command == "reset":
                # Reset connection (close and reopen session)
                if exchange_id not in self._connections:
                    result["error"] = f"Connection {exchange_id} not found"
                else:
                    # Close session if exists
                    if exchange_id in self._sessions:
                        self._sessions[exchange_id].close()
                        del self._sessions[exchange_id]

                    # Reset circuit breakers
                    for endpoint in self._connections[exchange_id].endpoints:
                        endpoint.circuit_state = CircuitBreakerState.CLOSED
                        endpoint.error_count = 0
                        endpoint.circuit_last_state_change = time.time()

                    result["success"] = True

            elif command == "websocket_restart":
                # Restart WebSocket
                if exchange_id not in self._connections:
                    result["error"] = f"Connection {exchange_id} not found"
                elif exchange_id not in self._websocket_managers:
                    result["error"] = f"WebSocket not enabled for {exchange_id}"
                else:
                    try:
                        # Stop and restart WebSocket
                        asyncio.run(self._websocket_managers[exchange_id].stop())
                        asyncio.run(self._websocket_managers[exchange_id].start())
                        result["success"] = True
                    except Exception as e:
                        result["error"] = f"Error restarting WebSocket: {str(e)}"

            else:
                result["error"] = f"Unknown command: {command}"

        except Exception as e:
            result["error"] = str(e)

        return result

