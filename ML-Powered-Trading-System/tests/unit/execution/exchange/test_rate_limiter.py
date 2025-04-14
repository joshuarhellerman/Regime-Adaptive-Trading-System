import unittest
import time
import threading
import pytest
from unittest.mock import patch, MagicMock
from execution.exchange.rate_limiter import (
    RateLimiter, 
    EndpointRateLimiter, 
    RateLimitAlgorithm, 
    RateLimitConfig,
    create_default_rate_limiter,
    create_forex_rate_limiter,
    create_crypto_rate_limiter
)


class TestRateLimiter(unittest.TestCase):
    """Test cases for the RateLimiter class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Patch logger to avoid cluttering test output
        self.logger_patcher = patch('execution.exchange.rate_limiter.logger')
        self.mock_logger = self.logger_patcher.start()

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        self.logger_patcher.stop()

    def test_simple_rate_limiter_init(self):
        """Test initialization of simple rate limiter."""
        limiter = RateLimiter(
            max_requests=10,
            time_window=1.0,
            algorithm=RateLimitAlgorithm.SIMPLE
        )
        
        self.assertEqual(limiter.config.max_requests, 10)
        self.assertEqual(limiter.config.time_window, 1.0)
        self.assertEqual(limiter.config.algorithm, RateLimitAlgorithm.SIMPLE)
        self.assertEqual(len(limiter.request_timestamps), 0)

    def test_token_bucket_rate_limiter_init(self):
        """Test initialization of token bucket rate limiter."""
        limiter = RateLimiter(
            max_requests=10,
            time_window=1.0,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            burst_capacity=15
        )
        
        self.assertEqual(limiter.config.max_requests, 10)
        self.assertEqual(limiter.config.burst_capacity, 15)
        self.assertEqual(limiter.config.algorithm, RateLimitAlgorithm.TOKEN_BUCKET)
        self.assertEqual(limiter.tokens, 15.0)

    def test_leaky_bucket_rate_limiter_init(self):
        """Test initialization of leaky bucket rate limiter."""
        limiter = RateLimiter(
            max_requests=10,
            time_window=1.0,
            algorithm=RateLimitAlgorithm.LEAKY_BUCKET
        )
        
        self.assertEqual(limiter.config.max_requests, 10)
        self.assertEqual(limiter.config.time_window, 1.0)
        self.assertEqual(limiter.config.algorithm, RateLimitAlgorithm.LEAKY_BUCKET)
        self.assertTrue(limiter.running)
        self.assertIsNotNone(limiter.worker_thread)
        
        # Clean up thread
        limiter.shutdown()

    def test_simple_rate_limiter_acquire(self):
        """Test simple rate limiter's acquire method."""
        max_requests = 5
        limiter = RateLimiter(
            max_requests=max_requests,
            time_window=1.0,
            algorithm=RateLimitAlgorithm.SIMPLE
        )
        
        # Acquire up to the limit
        start_time = time.time()
        for _ in range(max_requests):
            limiter.acquire()
        
        # This shouldn't have taken much time
        self.assertLess(time.time() - start_time, 0.1)
        
        # Verify the timestamps are stored
        self.assertEqual(len(limiter.request_timestamps), max_requests)

    def test_simple_rate_limiter_rate_limit(self):
        """Test simple rate limiter enforces the rate limit."""
        max_requests = 3
        time_window = 0.5
        limiter = RateLimiter(
            max_requests=max_requests,
            time_window=time_window,
            algorithm=RateLimitAlgorithm.SIMPLE
        )
        
        # Acquire up to the limit (shouldn't block)
        for _ in range(max_requests):
            limiter.acquire()
        
        # Try to acquire one more with a timeout thread
        def acquire_with_timeout():
            start = time.time()
            limiter.acquire()
            return time.time() - start
        
        # This acquisition should be delayed by approximately time_window
        thread = threading.Thread(target=acquire_with_timeout)
        start_time = time.time()
        thread.start()
        thread.join(timeout=time_window * 2)  # Give enough time for the thread to complete
        
        # Verify wait time was close to expected
        self.assertGreaterEqual(time.time() - start_time, time_window * 0.8)  # Allow for some timing variations

    def test_token_bucket_rate_limiter_acquire(self):
        """Test token bucket rate limiter's acquire method."""
        max_requests = 5
        burst_capacity = 10
        limiter = RateLimiter(
            max_requests=max_requests,
            time_window=1.0,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            burst_capacity=burst_capacity
        )
        
        # Should be able to burst up to burst_capacity
        start_time = time.time()
        for _ in range(burst_capacity):
            limiter.acquire()
        
        # This shouldn't have taken much time
        self.assertLess(time.time() - start_time, 0.1)
        
        # Token count should now be depleted
        self.assertLess(limiter.tokens, 1.0)

    def test_token_bucket_rate_limiter_refill(self):
        """Test token bucket refills over time."""
        max_requests = 10
        time_window = 0.5  # 10 tokens per 0.5 seconds = 20 tokens per second
        limiter = RateLimiter(
            max_requests=max_requests,
            time_window=time_window,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET
        )
        
        # Use some tokens
        for _ in range(5):
            limiter.acquire()
        
        # Check remaining tokens
        self.assertAlmostEqual(limiter.tokens, 5.0, delta=0.1)
        
        # Wait for tokens to refill
        time.sleep(time_window / 2)  # Should refill about 5 tokens
        
        # Manually refresh tokens (normally done during acquire)
        current_time = time.time()
        time_elapsed = current_time - limiter.last_token_refresh
        new_tokens = time_elapsed * limiter.config.fill_rate
        with limiter.lock:
            limiter.tokens = min(limiter.config.burst_capacity, limiter.tokens + new_tokens)
            limiter.last_token_refresh = current_time
        
        # Check if tokens were refilled
        self.assertGreaterEqual(limiter.tokens, 9.5)  # Should have at least 9.5 tokens now

    def test_leaky_bucket_rate_limiter_acquire(self):
        """Test leaky bucket rate limiter's acquire method."""
        max_requests = 10
        time_window = 0.4  # 10 requests per 0.4 seconds = 25 per second
        limiter = RateLimiter(
            max_requests=max_requests,
            time_window=time_window,
            algorithm=RateLimitAlgorithm.LEAKY_BUCKET
        )
        
        # Time how long it takes to process 5 requests
        start_time = time.time()
        for _ in range(5):
            limiter.acquire()
        elapsed = time.time() - start_time
        
        # Each request should take about time_window / max_requests
        expected_time = (time_window / max_requests) * 5
        
        # Allow for some timing variations in the test environment
        self.assertGreaterEqual(elapsed, expected_time * 0.8)
        
        # Clean up thread
        limiter.shutdown()

    def test_error_backoff(self):
        """Test error backoff functionality."""
        limiter = RateLimiter(
            max_requests=10,
            time_window=1.0,
            error_back_off=True,
            back_off_multiplier=2.0
        )
        
        # Initial state
        self.assertEqual(limiter.consecutive_errors, 0)
        self.assertEqual(limiter.current_back_off, 0.0)
        
        # Report first error
        limiter.report_error()
        self.assertEqual(limiter.consecutive_errors, 1)
        self.assertEqual(limiter.current_back_off, 1.0)
        
        # Report second error
        limiter.report_error()
        self.assertEqual(limiter.consecutive_errors, 2)
        self.assertEqual(limiter.current_back_off, 2.0)
        
        # Report third error
        limiter.report_error()
        self.assertEqual(limiter.consecutive_errors, 3)
        self.assertEqual(limiter.current_back_off, 4.0)
        
        # Report success, should reset
        limiter.report_success()
        self.assertEqual(limiter.consecutive_errors, 0)
        self.assertEqual(limiter.current_back_off, 0.0)

    def test_apply_error_backoff(self):
        """Test applying error backoff when making requests."""
        limiter = RateLimiter(
            max_requests=10,
            time_window=1.0,
            error_back_off=True
        )
        
        # Set up backoff state
        limiter.consecutive_errors = 1
        limiter.current_back_off = 0.2  # Small value for testing
        limiter.last_error_time = time.time()  # Now
        
        # Time the acquire with backoff
        start_time = time.time()
        limiter.acquire()
        elapsed = time.time() - start_time
        
        # Should have waited at least the backoff time
        self.assertGreaterEqual(elapsed, 0.18)  # Allow for small timing variations

    def test_endpoint_specific_rate_limiting(self):
        """Test endpoint-specific rate limiting."""
        limiter = RateLimiter(
            max_requests=10,
            time_window=1.0
        )
        
        # Create endpoint
        endpoint = "test_endpoint"
        limiter.acquire(endpoint=endpoint)
        
        # Endpoint limiter should be created
        self.assertIn(endpoint, limiter.endpoint_limiters)
        
        # Both limiters should be updated
        self.assertEqual(len(limiter.request_timestamps), 0)  # Main limiter not used
        self.assertEqual(len(limiter.endpoint_limiters[endpoint].request_timestamps), 1)  # Endpoint limiter used

    def test_reset(self):
        """Test resetting the rate limiter."""
        limiter = RateLimiter(
            max_requests=10,
            time_window=1.0,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET
        )
        
        # Use some capacity
        limiter.acquire()
        limiter.acquire()
        
        # Set up backoff
        limiter.consecutive_errors = 2
        limiter.current_back_off = 4.0
        
        # Reset
        limiter.reset()
        
        # State should be fresh
        self.assertEqual(len(limiter.request_timestamps), 0)
        self.assertEqual(limiter.tokens, 10.0)  # Default burst = max_requests
        self.assertEqual(limiter.consecutive_errors, 0)
        self.assertEqual(limiter.current_back_off, 0.0)

    def test_shutdown(self):
        """Test shutting down the rate limiter."""
        limiter = RateLimiter(
            max_requests=10,
            time_window=1.0,
            algorithm=RateLimitAlgorithm.LEAKY_BUCKET
        )
        
        # Shutdown
        limiter.shutdown()
        
        # Thread should be stopping
        self.assertFalse(limiter.running)
        
        # Wait for thread to complete
        if limiter.worker_thread:
            limiter.worker_thread.join(timeout=1.0)
            self.assertFalse(limiter.worker_thread.is_alive())


class TestEndpointRateLimiter(unittest.TestCase):
    """Test cases for the EndpointRateLimiter class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Patch logger to avoid cluttering test output
        self.logger_patcher = patch('execution.exchange.rate_limiter.logger')
        self.mock_logger = self.logger_patcher.start()
        
        # Create a global config
        self.global_config = RateLimitConfig(
            max_requests=100,
            time_window=1.0,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET
        )
        
        # Create endpoint configs
        self.endpoint_configs = {
            "pricing": RateLimitConfig(
                max_requests=50,
                time_window=1.0,
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET
            ),
            "orders": RateLimitConfig(
                max_requests=20,
                time_window=1.0,
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET
            )
        }

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        self.logger_patcher.stop()

    def test_endpoint_rate_limiter_init(self):
        """Test initialization of endpoint rate limiter."""
        limiter = EndpointRateLimiter(
            global_config=self.global_config,
            endpoint_configs=self.endpoint_configs
        )
        
        # Check global limiter
        self.assertEqual(limiter.global_limiter.config.max_requests, 100)
        
        # Check endpoint limiters
        self.assertEqual(len(limiter.endpoint_limiters), 2)
        self.assertEqual(limiter.endpoint_limiters["pricing"].config.max_requests, 50)
        self.assertEqual(limiter.endpoint_limiters["orders"].config.max_requests, 20)

    def test_endpoint_rate_limiter_acquire(self):
        """Test acquire with endpoint."""
        limiter = EndpointRateLimiter(
            global_config=self.global_config,
            endpoint_configs=self.endpoint_configs
        )
        
        # Acquire with endpoint
        limiter.acquire(endpoint="pricing")
        
        # Both should be updated
        self.assertLess(limiter.global_limiter.tokens, 100.0)
        self.assertLess(limiter.endpoint_limiters["pricing"].tokens, 50.0)
        
        # Other endpoint unchanged
        self.assertEqual(limiter.endpoint_limiters["orders"].tokens, 20.0)

    def test_endpoint_rate_limiter_acquire_unknown_endpoint(self):
        """Test acquire with unknown endpoint."""
        limiter = EndpointRateLimiter(
            global_config=self.global_config,
            endpoint_configs=self.endpoint_configs
        )
        
        # Acquire with unknown endpoint
        limiter.acquire(endpoint="unknown")
        
        # Only global should be updated
        self.assertLess(limiter.global_limiter.tokens, 100.0)
        self.assertEqual(limiter.endpoint_limiters["pricing"].tokens, 50.0)
        self.assertEqual(limiter.endpoint_limiters["orders"].tokens, 20.0)

    def test_endpoint_rate_limiter_report_success(self):
        """Test reporting success with endpoint."""
        limiter = EndpointRateLimiter(
            global_config=self.global_config,
            endpoint_configs=self.endpoint_configs
        )
        
        # Set up error state
        limiter.global_limiter.consecutive_errors = 2
        limiter.global_limiter.current_back_off = 4.0
        limiter.endpoint_limiters["pricing"].consecutive_errors = 1
        limiter.endpoint_limiters["pricing"].current_back_off = 1.0
        
        # Report success
        limiter.report_success(endpoint="pricing")
        
        # Both should be reset
        self.assertEqual(limiter.global_limiter.consecutive_errors, 0)
        self.assertEqual(limiter.global_limiter.current_back_off, 0.0)
        self.assertEqual(limiter.endpoint_limiters["pricing"].consecutive_errors, 0)
        self.assertEqual(limiter.endpoint_limiters["pricing"].current_back_off, 0.0)
        
        # Other endpoint unchanged
        self.assertEqual(limiter.endpoint_limiters["orders"].consecutive_errors, 0)

    def test_endpoint_rate_limiter_report_error(self):
        """Test reporting error with endpoint."""
        limiter = EndpointRateLimiter(
            global_config=self.global_config,
            endpoint_configs=self.endpoint_configs
        )
        
        # Report error
        limiter.report_error(endpoint="pricing")
        
        # Both should have errors
        self.assertEqual(limiter.global_limiter.consecutive_errors, 1)
        self.assertEqual(limiter.global_limiter.current_back_off, 1.0)
        self.assertEqual(limiter.endpoint_limiters["pricing"].consecutive_errors, 1)
        self.assertEqual(limiter.endpoint_limiters["pricing"].current_back_off, 1.0)
        
        # Other endpoint unchanged
        self.assertEqual(limiter.endpoint_limiters["orders"].consecutive_errors, 0)
        self.assertEqual(limiter.endpoint_limiters["orders"].current_back_off, 0.0)

    def test_endpoint_rate_limiter_reset(self):
        """Test resetting all limiters."""
        limiter = EndpointRateLimiter(
            global_config=self.global_config,
            endpoint_configs=self.endpoint_configs
        )
        
        # Set up error state
        limiter.global_limiter.consecutive_errors = 2
        limiter.global_limiter.current_back_off = 4.0
        limiter.endpoint_limiters["pricing"].consecutive_errors = 1
        limiter.endpoint_limiters["pricing"].current_back_off = 1.0
        
        # Acquire some capacity
        limiter.acquire(endpoint="pricing")
        
        # Reset
        limiter.reset()
        
        # All should be reset
        self.assertEqual(limiter.global_limiter.consecutive_errors, 0)
        self.assertEqual(limiter.global_limiter.current_back_off, 0.0)
        self.assertEqual(limiter.global_limiter.tokens, 100.0)
        self.assertEqual(limiter.endpoint_limiters["pricing"].consecutive_errors, 0)
        self.assertEqual(limiter.endpoint_limiters["pricing"].current_back_off, 0.0)
        self.assertEqual(limiter.endpoint_limiters["pricing"].tokens, 50.0)

    def test_endpoint_rate_limiter_shutdown(self):
        """Test shutting down all limiters."""
        # Create with leaky bucket for thread testing
        global_config = RateLimitConfig(
            max_requests=100,
            time_window=1.0,
            algorithm=RateLimitAlgorithm.LEAKY_BUCKET
        )
        
        endpoint_configs = {
            "test": RateLimitConfig(
                max_requests=50,
                time_window=1.0,
                algorithm=RateLimitAlgorithm.LEAKY_BUCKET
            )
        }
        
        limiter = EndpointRateLimiter(
            global_config=global_config,
            endpoint_configs=endpoint_configs
        )
        
        # Shutdown
        limiter.shutdown()
        
        # All should be stopped
        self.assertFalse(limiter.global_limiter.running)
        self.assertFalse(limiter.endpoint_limiters["test"].running)

    def test_add_endpoint_config(self):
        """Test adding a new endpoint configuration."""
        limiter = EndpointRateLimiter(
            global_config=self.global_config,
            endpoint_configs=self.endpoint_configs
        )
        
        # Add new endpoint
        new_config = RateLimitConfig(
            max_requests=30,
            time_window=1.0,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET
        )
        
        limiter.add_endpoint_config("new_endpoint", new_config)
        
        # Should have new endpoint
        self.assertIn("new_endpoint", limiter.endpoint_limiters)
        self.assertEqual(limiter.endpoint_limiters["new_endpoint"].config.max_requests, 30)


class TestFactoryFunctions(unittest.TestCase):
    """Test cases for the factory functions."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Patch logger to avoid cluttering test output
        self.logger_patcher = patch('execution.exchange.rate_limiter.logger')
        self.mock_logger = self.logger_patcher.start()

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        self.logger_patcher.stop()

    def test_create_default_rate_limiter(self):
        """Test creating a default rate limiter."""
        limiter = create_default_rate_limiter(max_requests=15, time_window=2.0)
        
        self.assertEqual(limiter.config.max_requests, 15)
        self.assertEqual(limiter.config.time_window, 2.0)
        self.assertEqual(limiter.config.algorithm, RateLimitAlgorithm.TOKEN_BUCKET)
        self.assertTrue(limiter.config.error_back_off)

    def test_create_forex_rate_limiter(self):
        """Test creating a forex rate limiter."""
        limiter = create_forex_rate_limiter()
        
        # Check global config
        self.assertEqual(limiter.global_limiter.config.max_requests, 120)
        self.assertEqual(limiter.global_limiter.config.time_window, 60.0)
        self.assertEqual(limiter.global_limiter.config.algorithm, RateLimitAlgorithm.TOKEN_BUCKET)
        self.assertEqual(limiter.global_limiter.config.priority_levels, 3)
        
        # Check endpoints
        self.assertEqual(len(limiter.endpoint_limiters), 3)
        self.assertIn("pricing", limiter.endpoint_limiters)
        self.assertIn("orders", limiter.endpoint_limiters)
        self.assertIn("accounts", limiter.endpoint_limiters)
        
        # Check endpoint-specific config
        self.assertEqual(limiter.endpoint_limiters["pricing"].config.max_requests, 180)
        self.assertEqual(limiter.endpoint_limiters["orders"].config.max_requests, 60)
        self.assertEqual(limiter.endpoint_limiters["accounts"].config.max_requests, 30)

    def test_create_crypto_rate_limiter(self):
        """Test creating a crypto rate limiter."""
        limiter = create_crypto_rate_limiter()
        
        # Check global config
        self.assertEqual(limiter.global_limiter.config.max_requests, 1200)
        self.assertEqual(limiter.global_limiter.config.time_window, 60.0)
        self.assertEqual(limiter.global_limiter.config.back_off_multiplier, 4.0)
        
        # Check endpoints
        self.assertEqual(len(limiter.endpoint_limiters), 3)
        self.assertIn("market_data", limiter.endpoint_limiters)
        self.assertIn("orders", limiter.endpoint_limiters)
        self.assertIn("account", limiter.endpoint_limiters)
        
        # Check endpoint-specific config
        self.assertEqual(limiter.endpoint_limiters["market_data"].config.max_requests, 1800)
        self.assertEqual(limiter.endpoint_limiters["orders"].config.max_requests, 600)
        self.assertEqual(limiter.endpoint_limiters["account"].config.max_requests, 120)


@pytest.mark.parametrize("algorithm", [
    RateLimitAlgorithm.SIMPLE,
    RateLimitAlgorithm.TOKEN_BUCKET,
    RateLimitAlgorithm.LEAKY_BUCKET
])
def test_rate_limiter_algorithms_performance(algorithm):
    """Test performance characteristics of different algorithms."""
    max_requests = 10
    time_window = 0.5
    
    limiter = RateLimiter(
        max_requests=max_requests,
        time_window=time_window,
        algorithm=algorithm
    )
    
    # Make half the maximum requests
    start_time = time.time()
    for _ in range(max_requests // 2):
        limiter.acquire()
    first_batch_time = time.time() - start_time
    
    # Make another batch (may be rate limited depending on algorithm)
    start_time = time.time()
    for _ in range(max_requests):
        limiter.acquire()
    second_batch_time = time.time() - start_time
    
    # For all algorithms, first batch should be quick
    assert first_batch_time < time_window, "First batch should complete quickly"
    
    # For SIMPLE, second batch should be partially rate limited
    # For TOKEN_BUCKET, second batch may be partially rate limited
    # For LEAKY_BUCKET, second batch should be fully rate limited
    if algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
        assert second_batch_time >= time_window, "Leaky bucket should enforce constant rate"
    
    # Clean up
    limiter.shutdown()


if __name__ == '__main__':
    unittest.main()