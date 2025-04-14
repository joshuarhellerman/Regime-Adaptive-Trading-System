import unittest
from unittest.mock import Mock, patch, MagicMock, call
import pytest
import time
import threading
import json
import asyncio
import aiohttp
import requests
from requests.exceptions import RequestException, ConnectionError, Timeout, SSLError

# Import the module to be tested
from execution.exchange.connectivity_manager import (
    ConnectivityManager,
    ConnectionStatus,
    ConnectionPriority,
    EndpointHealth,
    CircuitBreakerState,
    ConnectionEndpoint,
    ConnectionConfig,
    RateLimiter,
    WebSocketManager
)


# Mock the event bus
@pytest.fixture
def mock_event_bus():
    mock_bus = Mock()
    mock_bus.publish = Mock()
    return mock_bus


@pytest.fixture
def mock_create_event():
    with patch('execution.exchange.connectivity_manager.create_event') as mock:
        mock.return_value = {"topic": "test", "data": {}}
        yield mock


@pytest.fixture
def mock_get_event_bus():
    with patch('execution.exchange.connectivity_manager.get_event_bus') as mock:
        mock.return_value = Mock()
        yield mock


class TestConnectionEndpoint(unittest.TestCase):
    """Tests for ConnectionEndpoint class"""

    def test_init(self):
        """Test initialization of ConnectionEndpoint"""
        endpoint = ConnectionEndpoint(
            url="https://api.example.com",
            priority=ConnectionPriority.PRIMARY,
            credentials={"api_key": "test_key"},
            options={"circuit_failure_threshold": 10}
        )

        self.assertEqual(endpoint.url, "https://api.example.com")
        self.assertEqual(endpoint.priority, ConnectionPriority.PRIMARY)
        self.assertEqual(endpoint.credentials, {"api_key": "test_key"})
        self.assertEqual(endpoint.circuit_failure_threshold, 10)
        self.assertEqual(endpoint.status, ConnectionStatus.DISCONNECTED)
        self.assertEqual(endpoint.health, EndpointHealth.UNKNOWN)
        self.assertEqual(endpoint.circuit_state, CircuitBreakerState.CLOSED)

    def test_record_success(self):
        """Test recording a successful request"""
        endpoint = ConnectionEndpoint(url="https://api.example.com")

        # Record a success
        endpoint.record_success(150.5)  # 150.5ms response time

        self.assertEqual(endpoint.success_count, 1)
        self.assertEqual(endpoint.response_times, [150.5])
        self.assertEqual(endpoint.error_count, 0)
        self.assertEqual(endpoint.health, EndpointHealth.HEALTHY)

        # Record another success
        endpoint.record_success(200.3)

        self.assertEqual(endpoint.success_count, 2)
        self.assertEqual(endpoint.response_times, [150.5, 200.3])
        self.assertEqual(endpoint.get_average_response_time(), 175.4)

    def test_record_error(self):
        """Test recording a request error"""
        endpoint = ConnectionEndpoint(url="https://api.example.com")

        # Record an error
        error = ConnectionError("Connection refused")
        endpoint.record_error(error)

        self.assertEqual(endpoint.error_count, 1)
        self.assertEqual(endpoint.last_error, error)

        # Record multiple errors to trigger health status change
        for _ in range(10):
            endpoint.record_error(error)

        self.assertEqual(endpoint.health, EndpointHealth.UNHEALTHY)

    def test_circuit_breaker(self):
        """Test circuit breaker functionality"""
        endpoint = ConnectionEndpoint(
            url="https://api.example.com",
            options={"circuit_failure_threshold": 3, "circuit_recovery_timeout": 0.1}
        )

        # Circuit should start closed
        self.assertEqual(endpoint.circuit_state, CircuitBreakerState.CLOSED)
        self.assertTrue(endpoint.check_circuit_breaker())

        # Record enough errors to open the circuit
        for _ in range(3):
            endpoint.record_error(ConnectionError("Connection refused"))

        # Circuit should be open now
        self.assertEqual(endpoint.circuit_state, CircuitBreakerState.OPEN)
        self.assertFalse(endpoint.check_circuit_breaker())

        # Wait for recovery timeout
        time.sleep(0.2)

        # Circuit should be half-open now
        self.assertTrue(endpoint.check_circuit_breaker())
        self.assertEqual(endpoint.circuit_state, CircuitBreakerState.HALF_OPEN)

        # Record a success to close the circuit
        endpoint.record_success(100.0)
        self.assertEqual(endpoint.circuit_state, CircuitBreakerState.CLOSED)

        # Record a failure in half-open state
        endpoint.circuit_state = CircuitBreakerState.HALF_OPEN
        endpoint.record_error(ConnectionError("Connection refused"))
        self.assertEqual(endpoint.circuit_state, CircuitBreakerState.OPEN)


class TestConnectionConfig(unittest.TestCase):
    """Tests for ConnectionConfig class"""

    def test_init(self):
        """Test initialization of ConnectionConfig"""
        endpoints = [
            ConnectionEndpoint(url="https://api1.example.com", priority=ConnectionPriority.PRIMARY),
            ConnectionEndpoint(url="https://api2.example.com", priority=ConnectionPriority.SECONDARY)
        ]

        config = ConnectionConfig(
            exchange_id="example",
            name="Example Exchange",
            endpoints=endpoints,
            retry_policy={"retries": 5},
            timeout={"connect": 5.0, "read": 10.0},
            rate_limits={"requests_per_second": 10.0}
        )

        self.assertEqual(config.exchange_id, "example")
        self.assertEqual(config.name, "Example Exchange")
        self.assertEqual(len(config.endpoints), 2)
        self.assertEqual(config.retry_policy["retries"], 5)
        self.assertEqual(config.timeout["connect"], 5.0)
        self.assertEqual(config.rate_limits["requests_per_second"], 10.0)

    def test_get_primary_endpoint(self):
        """Test getting the primary endpoint"""
        primary = ConnectionEndpoint(url="https://api1.example.com", priority=ConnectionPriority.PRIMARY)
        secondary = ConnectionEndpoint(url="https://api2.example.com", priority=ConnectionPriority.SECONDARY)

        config = ConnectionConfig(
            exchange_id="example",
            endpoints=[secondary, primary]  # Order shouldn't matter
        )

        # Should return the primary endpoint
        self.assertEqual(config.get_primary_endpoint(), primary)

    def test_get_active_endpoints(self):
        """Test getting active endpoints"""
        primary = ConnectionEndpoint(url="https://api1.example.com", priority=ConnectionPriority.PRIMARY)
        secondary = ConnectionEndpoint(url="https://api2.example.com", priority=ConnectionPriority.SECONDARY)
        failover = ConnectionEndpoint(url="https://api3.example.com", priority=ConnectionPriority.FAILOVER)

        # Mark secondary as unhealthy
        secondary.health = EndpointHealth.UNHEALTHY

        # Mark failover with open circuit
        failover.circuit_state = CircuitBreakerState.OPEN

        config = ConnectionConfig(
            exchange_id="example",
            endpoints=[primary, secondary, failover]
        )

        # Should only return healthy endpoints with closed circuit
        active_endpoints = config.get_active_endpoints()
        self.assertEqual(len(active_endpoints), 1)
        self.assertEqual(active_endpoints[0], primary)


class TestRateLimiter(unittest.TestCase):
    """Tests for RateLimiter class"""

    def test_wait_if_needed(self):
        """Test rate limiting functionality"""
        # Create a rate limiter with 10 requests per second
        limiter = RateLimiter(requests_per_second=10.0, requests_per_minute=100)

        # First request should not wait
        start_time = time.time()
        wait_time = limiter.wait_if_needed()
        elapsed = time.time() - start_time

        self.assertLess(wait_time, 0.01)  # Almost no wait
        self.assertLess(elapsed, 0.01)  # Almost no elapsed time

        # Simulate rapid requests
        for _ in range(10):
            limiter.wait_if_needed()

        # Next request should wait to respect the rate limit
        start_time = time.time()
        wait_time = limiter.wait_if_needed()
        elapsed = time.time() - start_time

        # Should wait approximately 0.1s for the next slot
        self.assertGreaterEqual(elapsed, 0.08)  # Allow some timing variance

    @pytest.mark.asyncio
    async def test_wait_if_needed_async(self):
        """Test async rate limiting functionality"""
        # Create a rate limiter with 10 requests per second
        limiter = RateLimiter(requests_per_second=10.0, requests_per_minute=100)

        # First request should not wait
        start_time = time.time()
        wait_time = await limiter.wait_if_needed_async()
        elapsed = time.time() - start_time

        self.assertLess(wait_time, 0.01)  # Almost no wait
        self.assertLess(elapsed, 0.01)  # Almost no elapsed time

        # Simulate rapid requests
        for _ in range(10):
            await limiter.wait_if_needed_async()

        # Next request should wait to respect the rate limit
        start_time = time.time()
        wait_time = await limiter.wait_if_needed_async()
        elapsed = time.time() - start_time

        # Should wait approximately 0.1s for the next slot
        self.assertGreaterEqual(elapsed, 0.08)  # Allow some timing variance


@patch('execution.exchange.connectivity_manager.get_event_bus')
class TestConnectivityManager(unittest.TestCase):
    """Tests for ConnectivityManager class"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            "health_check_interval": 0.1,
            "connections": {
                "example": {
                    "enabled": True,
                    "name": "Example Exchange",
                    "endpoints": [
                        {
                            "url": "https://api1.example.com",
                            "priority": "PRIMARY",
                            "credentials": {"api_key": "test_key"}
                        },
                        {
                            "url": "https://api2.example.com",
                            "priority": "SECONDARY"
                        }
                    ],
                    "retry_policy": {"retries": 2},
                    "timeout": {"connect": 2.0, "read": 5.0, "total": 10.0},
                    "rate_limits": {"requests_per_second": 5.0, "requests_per_minute": 200},
                    "websocket_config": {"enabled": False}
                }
            }
        }

    def test_init(self, mock_get_event_bus):
        """Test initialization of ConnectivityManager"""
        manager = ConnectivityManager(self.config)

        self.assertEqual(len(manager._connections), 1)
        self.assertEqual(len(manager._rate_limiters), 1)
        self.assertIn("example", manager._connections)
        self.assertIn("example", manager._rate_limiters)

    @patch('execution.exchange.connectivity_manager.requests.Session')
    def test_get_session(self, mock_session, mock_get_event_bus):
        """Test getting a session for an exchange"""
        manager = ConnectivityManager(self.config)

        # Get session for configured exchange
        session = manager.get_session("example")

        # Should create and return a session
        self.assertIsNotNone(session)
        self.assertIn("example", manager._sessions)

        # Getting it again should return the same session
        session2 = manager.get_session("example")
        self.assertEqual(session, session2)

        # Test with unknown exchange
        with self.assertRaises(ValueError):
            manager.get_session("unknown")

    @patch('execution.exchange.connectivity_manager.requests.Session')
    def test_request(self, mock_session, mock_get_event_bus):
        """Test making a request to an exchange"""
        # Set up mocks
        mock_response = Mock()
        mock_response.status_code = 200

        mock_session_instance = Mock()
        mock_session_instance.request.return_value = mock_response
        mock_session.return_value = mock_session_instance

        manager = ConnectivityManager(self.config)

        # Make a request
        response = manager.request(
            exchange_id="example",
            method="GET",
            endpoint="test/endpoint",
            params={"param": "value"}
        )

        # Should return the response
        self.assertEqual(response, mock_response)

        # Should have called request on the session
        mock_session_instance.request.assert_called_once()
        args, kwargs = mock_session_instance.request.call_args

        self.assertEqual(kwargs["method"], "GET")
        self.assertEqual(kwargs["url"], "https://api1.example.com/test/endpoint")
        self.assertEqual(kwargs["params"], {"param": "value"})

    @patch('execution.exchange.connectivity_manager.requests.Session')
    def test_request_failover(self, mock_session, mock_get_event_bus):
        """Test failover to secondary endpoint on error"""
        # Set up mocks - first request fails, second succeeds
        mock_response = Mock()
        mock_response.status_code = 200

        mock_session_instance = Mock()
        mock_session_instance.request.side_effect = [
            ConnectionError("Connection refused"),  # Primary endpoint fails
            mock_response  # Secondary endpoint succeeds
        ]
        mock_session.return_value = mock_session_instance

        manager = ConnectivityManager(self.config)

        # Make a request
        response = manager.request(
            exchange_id="example",
            method="GET",
            endpoint="test/endpoint"
        )

        # Should return the response from secondary endpoint
        self.assertEqual(response, mock_response)

        # Should have called request twice
        self.assertEqual(mock_session_instance.request.call_count, 2)

        # Check the second call used the secondary endpoint
        _, kwargs = mock_session_instance.request.call_args_list[1]
        self.assertEqual(kwargs["url"], "https://api2.example.com/test/endpoint")

    @patch('execution.exchange.connectivity_manager.requests.Session')
    def test_request_circuit_breaker(self, mock_session, mock_get_event_bus):
        """Test circuit breaker prevents requests to failing endpoint"""
        # Set up mocks
        mock_response = Mock()
        mock_response.status_code = 200

        mock_session_instance = Mock()
        mock_session_instance.request.side_effect = [
            ConnectionError("Connection refused"),  # First request fails
            ConnectionError("Connection refused"),  # Second request fails
            ConnectionError("Connection refused"),  # Third request fails
            ConnectionError("Connection refused"),  # Fourth request (would fail, but circuit breaks)
            mock_response  # Fifth request succeeds (on secondary endpoint)
        ]
        mock_session.return_value = mock_session_instance

        # Set a low threshold
        self.config["connections"]["example"]["endpoints"][0]["options"] = {"circuit_failure_threshold": 3}

        manager = ConnectivityManager(self.config)

        # Make 3 requests to trigger circuit breaker
        for _ in range(3):
            with self.assertRaises(RequestException):
                manager.request(
                    exchange_id="example",
                    method="GET",
                    endpoint="test/endpoint"
                )

        # Primary endpoint's circuit should be open now
        endpoint = manager._connections["example"].endpoints[0]
        self.assertEqual(endpoint.circuit_state, CircuitBreakerState.OPEN)

        # Next request should use secondary endpoint directly
        response = manager.request(
            exchange_id="example",
            method="GET",
            endpoint="test/endpoint"
        )

        # Should return the response from secondary endpoint
        self.assertEqual(response, mock_response)

        # Should have called request 5 times total
        self.assertEqual(mock_session_instance.request.call_count, 4)

    @patch('execution.exchange.connectivity_manager.requests.Session')
    def test_shortcut_methods(self, mock_session, mock_get_event_bus):
        """Test shortcut methods (get, post, etc.)"""
        # Set up mock
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}

        mock_session_instance = Mock()
        mock_session_instance.request.return_value = mock_response
        mock_session.return_value = mock_session_instance

        manager = ConnectivityManager(self.config)

        # Test GET method
        with patch.object(manager, 'request') as mock_request:
            mock_request.return_value = mock_response
            response = manager.get("example", "test/endpoint", params={"param": "value"})

            mock_request.assert_called_once_with(
                "example", "GET", "test/endpoint", params={"param": "value"}
            )
            self.assertEqual(response, mock_response)

        # Test POST method
        with patch.object(manager, 'request') as mock_request:
            mock_request.return_value = mock_response
            response = manager.post("example", "test/endpoint", data={"data": "value"})

            mock_request.assert_called_once_with(
                "example", "POST", "test/endpoint", data={"data": "value"}
            )
            self.assertEqual(response, mock_response)

        # Test JSON methods
        with patch.object(manager, 'get') as mock_get:
            mock_get.return_value = mock_response
            result = manager.get_json("example", "test/endpoint")

            mock_get.assert_called_once()
            self.assertEqual(result, {"result": "success"})

    @patch('execution.exchange.connectivity_manager.aiohttp.ClientSession')
    @pytest.mark.asyncio
    async def test_request_async(self, mock_session, mock_get_event_bus):
        """Test async request method"""
        # Set up mocks
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = MagicMock()
        mock_response.json.return_value = {"result": "success"}

        mock_context = MagicMock()
        mock_context.__aenter__.return_value = mock_response

        mock_session_instance = MagicMock()
        mock_session_instance.request.return_value = mock_context

        mock_session.return_value = mock_session_instance

        manager = ConnectivityManager(self.config)

        # Make an async request
        result = await manager.request_async(
            exchange_id="example",
            method="GET",
            endpoint="test/endpoint",
            params={"param": "value"}
        )

        # Should return the parsed JSON
        self.assertEqual(result, {"result": "success"})

        # Should have called request on the session
        mock_session_instance.request.assert_called_once()

    @patch('execution.exchange.connectivity_manager.requests.Session')
    def test_health_check(self, mock_session, mock_get_event_bus):
        """Test health check functionality"""
        # Set up mocks
        mock_response = Mock()
        mock_response.status_code = 200

        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session.return_value = mock_session_instance

        manager = ConnectivityManager(self.config)

        # Perform health check
        result = manager.health_check("example")

        # Should return True for healthy
        self.assertTrue(result)

        # Should have called get on the session
        mock_session_instance.get.assert_called_once()
        args, kwargs = mock_session_instance.get.call_args

        # Should use the health check endpoint
        self.assertEqual(kwargs["url"], "https://api1.example.com/api/v1/ping")

        # Endpoints should be marked as healthy
        self.assertEqual(manager._connections["example"].endpoints[0].health, EndpointHealth.HEALTHY)

    @patch('execution.exchange.connectivity_manager.requests.Session')
    def test_add_connection(self, mock_session, mock_get_event_bus):
        """Test adding a new connection"""
        manager = ConnectivityManager({})

        # Initially should have no connections
        self.assertEqual(len(manager._connections), 0)

        # Add a connection
        result = manager.add_connection(
            exchange_id="new_exchange",
            name="New Exchange",
            endpoints=[
                {
                    "url": "https://api.newexchange.com",
                    "priority": "PRIMARY"
                }
            ],
            timeout={"connect": 5.0}
        )

        # Should return True for success
        self.assertTrue(result)

        # Should have added the connection
        self.assertEqual(len(manager._connections), 1)
        self.assertIn("new_exchange", manager._connections)

        # Connection should have correct properties
        connection = manager._connections["new_exchange"]
        self.assertEqual(connection.name, "New Exchange")
        self.assertEqual(connection.timeout["connect"], 5.0)
        self.assertEqual(len(connection.endpoints), 1)
        self.assertEqual(connection.endpoints[0].url, "https://api.newexchange.com")

    @patch('execution.exchange.connectivity_manager.requests.Session')
    def test_remove_connection(self, mock_session, mock_get_event_bus):
        """Test removing a connection"""
        manager = ConnectivityManager(self.config)

        # Initially should have one connection
        self.assertEqual(len(manager._connections), 1)

        # Remove the connection
        result = manager.remove_connection("example")

        # Should return True for success
        self.assertTrue(result)

        # Should have removed the connection
        self.assertEqual(len(manager._connections), 0)
        self.assertNotIn("example", manager._connections)

        # Removing non-existent connection should return False
        result = manager.remove_connection("nonexistent")
        self.assertFalse(result)

    @patch('execution.exchange.connectivity_manager.requests.Session')
    def test_update_credentials(self, mock_session, mock_get_event_bus):
        """Test updating credentials"""
        manager = ConnectivityManager(self.config)

        # Initial credentials
        initial_creds = manager._connections["example"].endpoints[0].credentials
        self.assertEqual(initial_creds, {"api_key": "test_key"})

        # Update credentials
        result = manager.update_credentials(
            exchange_id="example",
            credentials={"api_key": "new_key", "secret": "new_secret"}
        )

        # Should return True for success
        self.assertTrue(result)

        # Credentials should be updated
        updated_creds = manager._connections["example"].endpoints[0].credentials
        self.assertEqual(updated_creds, {"api_key": "new_key", "secret": "new_secret"})

        # Session should be recreated
        self.assertNotIn("example", manager._sessions)


@pytest.mark.asyncio
class TestWebSocketManager:
    """Tests for WebSocketManager class"""

    @pytest.fixture
    def mock_connection_config(self):
        """Create a mock connection config"""
        endpoints = [
            ConnectionEndpoint(url="https://api.example.com", priority=ConnectionPriority.PRIMARY)
        ]

        return ConnectionConfig(
            exchange_id="example",
            name="Example Exchange",
            endpoints=endpoints,
            websocket_config={
                "enabled": True,
                "url": "wss://ws.example.com",
                "ping_interval": 0.1,
                "reconnect_interval": 0.1,
                "max_reconnect_attempts": 3
            }
        )

    @pytest.fixture
    def mock_message_handler(self):
        """Create a mock message handler"""

        async def handler(data):
            pass

        return Mock(side_effect=handler)

    @patch('execution.exchange.connectivity_manager.aiohttp.ClientSession')
    async def test_start_stop(self, mock_session, mock_connection_config, mock_message_handler):
        """Test starting and stopping WebSocket"""
        # Set up mocks
        mock_ws = Mock()
        mock_ws.closed = False
        mock_ws.send_str = AsyncMock()
        mock_ws.send_json = AsyncMock()
        mock_ws.ping = AsyncMock()
        mock_ws.close = AsyncMock()

        mock_context = Mock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_ws)
        mock_context.__aexit__ = AsyncMock()

        mock_session_instance = Mock()
        mock_session_instance.ws_connect = Mock(return_value=mock_context)

        mock_session.return_value = AsyncMock()
        mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session_instance)

        # Create WebSocket manager
        manager = WebSocketManager(mock_connection_config, mock_message_handler)

        # Start manager
        with patch.object(manager, '_connection_loop', AsyncMock()) as mock_loop:
            await manager.start()

            # Should set running to True
            assert manager.running == True

            # Should start connection loop
            mock_loop.assert_called_once()

        # Stop manager
        await manager.stop()

        # Should set running to False
        assert manager.running == False

        # Should close WebSocket if it exists
        if manager.ws:
            manager.ws.close.assert_called_once()

    @patch('execution.exchange.connectivity_manager.aiohttp.ClientSession')
    async def test_send(self, mock_session, mock_connection_config, mock_message_handler):
        """Test sending messages through WebSocket"""
        # Set up mocks
        mock_ws = Mock()
        mock_ws.send_str = AsyncMock()
        mock_ws.send_json = AsyncMock()

        # Create WebSocket manager
        manager = WebSocketManager(mock_connection_config, mock_message_handler)
        manager.running = True
        manager.ws = mock_ws

        # Send string message
        success = await manager.send("test message")
        assert success == True
        mock_ws.send_str.assert_called_with("test message")

        # Send JSON message
        success = await manager.send({"type": "test", "data": "value"})
        assert success == True
        mock_ws.send_json.assert_called_with({"type": "test", "data": "value"})

        # Test when not running
        manager.running = False
        success = await manager.send("test message")
        assert success == False


if __name__ == '__main__':
    unittest.main()