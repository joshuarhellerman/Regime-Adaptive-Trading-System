"""
rate_limiter.py - Rate limiting implementation for exchange API requests

This module provides a configurable rate limiter to prevent exceeding API rate limits
when interacting with exchanges. It supports both simple rate limiting and token bucket
algorithms for more sophisticated rate control.
"""

import time
import logging
import threading
from typing import Dict, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass
import queue

logger = logging.getLogger(__name__)

class RateLimitAlgorithm(Enum):
    """Enum for supported rate limiting algorithms."""
    SIMPLE = 1      # Simple requests-per-time-window limiting
    TOKEN_BUCKET = 2  # Token bucket algorithm with configurable burst
    LEAKY_BUCKET = 3  # Leaky bucket algorithm


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    max_requests: int  # Maximum number of requests
    time_window: float  # Time window in seconds
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SIMPLE
    burst_capacity: Optional[int] = None  # For token bucket, max burst size
    fill_rate: Optional[float] = None  # For token bucket, tokens per second
    error_back_off: bool = True  # Whether to back off on rate limit errors
    back_off_multiplier: float = 2.0  # Multiplier for back off time
    max_back_off: float = 300.0  # Maximum back off time in seconds
    priority_levels: int = 1  # Number of priority levels (1 = no priorities)


class RateLimiter:
    """
    Rate limiter for API requests to prevent exceeding rate limits.

    This class implements different rate limiting algorithms:
    1. Simple: Tracks requests within a sliding window
    2. Token Bucket: Allows for bursts of requests up to a limit
    3. Leaky Bucket: Enforces a constant outflow rate

    It also supports:
    - Prioritization of requests
    - Error-based backoff
    - Per-endpoint limiting
    """

    def __init__(
        self,
        max_requests: int,
        time_window: float,
        algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SIMPLE,
        burst_capacity: Optional[int] = None,
        fill_rate: Optional[float] = None,
        error_back_off: bool = True,
        back_off_multiplier: float = 2.0,
        max_back_off: float = 300.0,
        priority_levels: int = 1
    ):
        """
        Initialize the rate limiter.

        Args:
            max_requests: Maximum number of requests allowed in the time window
            time_window: Time window in seconds
            algorithm: Rate limiting algorithm to use
            burst_capacity: For token bucket, maximum burst size
            fill_rate: For token bucket, tokens per second (defaults to max_requests/time_window)
            error_back_off: Whether to back off on rate limit errors
            back_off_multiplier: Multiplier for back off time
            max_back_off: Maximum back off time in seconds
            priority_levels: Number of priority levels (1 = no priorities)
        """
        self.config = RateLimitConfig(
            max_requests=max_requests,
            time_window=time_window,
            algorithm=algorithm,
            burst_capacity=burst_capacity or max_requests,
            fill_rate=fill_rate or (max_requests / time_window),
            error_back_off=error_back_off,
            back_off_multiplier=back_off_multiplier,
            max_back_off=max_back_off,
            priority_levels=max(1, priority_levels)
        )

        # For simple rate limiting
        self.request_timestamps: List[float] = []

        # For token bucket
        self.tokens = float(self.config.burst_capacity or max_requests)
        self.last_token_refresh = time.time()

        # For leaky bucket
        self.request_queue = queue.PriorityQueue()
        self.worker_thread = None
        self.running = False

        # For endpoint-specific rate limiting
        self.endpoint_limiters: Dict[str, RateLimiter] = {}

        # For error backoff
        self.current_back_off = 0.0
        self.consecutive_errors = 0
        self.last_error_time = 0.0

        # Thread safety
        self.lock = threading.RLock()

        # Start worker thread if using leaky bucket
        if algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
            self.running = True
            self.worker_thread = threading.Thread(
                target=self._process_queue,
                daemon=True,
                name="rate-limiter-worker"
            )
            self.worker_thread.start()

        logger.debug(
            f"Rate limiter initialized: {max_requests} requests per {time_window}s "
            f"using {algorithm.name} algorithm"
        )

    def acquire(self, priority: int = 0, endpoint: Optional[str] = None) -> None:
        """
        Acquire permission to make a request, blocking if necessary.

        Args:
            priority: Request priority (lower = higher priority, 0 = highest)
            endpoint: Optional endpoint-specific rate limiting
        """
        # If endpoint-specific limiting is requested
        if endpoint and endpoint not in self.endpoint_limiters:
            # Create endpoint-specific limiter with same config but separate counters
            self.endpoint_limiters[endpoint] = RateLimiter(
                max_requests=self.config.max_requests,
                time_window=self.config.time_window,
                algorithm=self.config.algorithm,
                burst_capacity=self.config.burst_capacity,
                fill_rate=self.config.fill_rate,
                error_back_off=self.config.error_back_off,
                back_off_multiplier=self.config.back_off_multiplier,
                max_back_off=self.config.max_back_off,
                priority_levels=self.config.priority_levels
            )

        if endpoint:
            # Use endpoint-specific limiter
            self.endpoint_limiters[endpoint].acquire(priority)
            return

        # Apply error back-off if active
        self._apply_error_backoff()

        # Normalize priority
        priority = max(0, min(priority, self.config.priority_levels - 1))

        with self.lock:
            if self.config.algorithm == RateLimitAlgorithm.SIMPLE:
                self._acquire_simple(priority)
            elif self.config.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
                self._acquire_token_bucket(priority)
            elif self.config.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
                self._acquire_leaky_bucket(priority)

    def _acquire_simple(self, priority: int) -> None:
        """
        Simple rate limiting implementation.

        Args:
            priority: Request priority
        """
        current_time = time.time()

        # Remove timestamps outside the window
        window_start = current_time - self.config.time_window
        self.request_timestamps = [ts for ts in self.request_timestamps if ts >= window_start]

        # Check if we've hit the limit
        while len(self.request_timestamps) >= self.config.max_requests:
            # Calculate time to wait until oldest request expires from window
            wait_time = self.request_timestamps[0] - window_start

            # Release lock while waiting
            self.lock.release()
            time.sleep(max(0.01, wait_time))
            self.lock.acquire()

            # Recalculate window after waiting
            current_time = time.time()
            window_start = current_time - self.config.time_window
            self.request_timestamps = [ts for ts in self.request_timestamps if ts >= window_start]

        # Add new timestamp
        self.request_timestamps.append(current_time)

        logger.debug(f"Rate limiter: request permitted ({len(self.request_timestamps)}/{self.config.max_requests})")

    def _acquire_token_bucket(self, priority: int) -> None:
        """
        Token bucket rate limiting implementation.

        Args:
            priority: Request priority
        """
        current_time = time.time()

        # Refill tokens based on time elapsed
        time_elapsed = current_time - self.last_token_refresh
        new_tokens = time_elapsed * self.config.fill_rate
        self.tokens = min(self.config.burst_capacity, self.tokens + new_tokens)
        self.last_token_refresh = current_time

        # Wait if not enough tokens
        while self.tokens < 1.0:
            # Calculate time to wait for enough tokens
            wait_time = (1.0 - self.tokens) / self.config.fill_rate

            # Release lock while waiting
            self.lock.release()
            time.sleep(max(0.01, wait_time))
            self.lock.acquire()

            # Recalculate tokens after waiting
            current_time = time.time()
            time_elapsed = current_time - self.last_token_refresh
            new_tokens = time_elapsed * self.config.fill_rate
            self.tokens = min(self.config.burst_capacity, self.tokens + new_tokens)
            self.last_token_refresh = current_time

        # Consume token
        self.tokens -= 1.0

        logger.debug(f"Rate limiter: token consumed ({self.tokens:.2f}/{self.config.burst_capacity} remaining)")

    def _acquire_leaky_bucket(self, priority: int) -> None:
        """
        Leaky bucket rate limiting implementation.

        Args:
            priority: Request priority
        """
        # Create request with priority and timestamp
        request = (priority, time.time())

        # Add to priority queue and wait for processing
        completion_event = threading.Event()
        self.request_queue.put((request, completion_event))

        # Release lock while waiting
        self.lock.release()
        completion_event.wait()
        self.lock.acquire()

        logger.debug(f"Rate limiter: leaky bucket request processed")

    def _process_queue(self) -> None:
        """Process the leaky bucket queue at a constant rate."""
        interval = self.config.time_window / self.config.max_requests

        while self.running:
            try:
                # Get next request from queue (blocking)
                (priority, timestamp), completion_event = self.request_queue.get(block=True)

                # Process at constant rate
                time.sleep(interval)

                # Mark request as processed
                completion_event.set()
                self.request_queue.task_done()

            except Exception as e:
                logger.error(f"Error in rate limiter queue processing: {e}", exc_info=True)
                time.sleep(0.1)  # Prevent tight loop on error

    def report_success(self) -> None:
        """Report a successful request, resetting error backoff."""
        with self.lock:
            if self.consecutive_errors > 0:
                logger.info(f"Rate limiter: Resetting backoff after {self.consecutive_errors} consecutive errors")

            self.consecutive_errors = 0
            self.current_back_off = 0.0

    def report_error(self, is_rate_limit_error: bool = True) -> None:
        """
        Report a rate limit error, increasing backoff time.

        Args:
            is_rate_limit_error: Whether this was a rate limit error
        """
        if not self.config.error_back_off or not is_rate_limit_error:
            return

        with self.lock:
            self.consecutive_errors += 1
            self.last_error_time = time.time()

            # Exponential backoff with max limit
            if self.current_back_off == 0:
                self.current_back_off = 1.0
            else:
                self.current_back_off = min(
                    self.config.max_back_off,
                    self.current_back_off * self.config.back_off_multiplier
                )

            logger.warning(
                f"Rate limiter: Backoff increased to {self.current_back_off:.2f}s "
                f"after {self.consecutive_errors} consecutive errors"
            )

    def _apply_error_backoff(self) -> None:
        """Apply any active error backoff by sleeping."""
        if self.current_back_off <= 0:
            return

        # Calculate time since last error
        time_since_error = time.time() - self.last_error_time

        # If we're still in backoff period
        if time_since_error < self.current_back_off:
            wait_time = self.current_back_off - time_since_error

            logger.info(f"Rate limiter: Applying backoff, waiting {wait_time:.2f}s")

            # Release lock while waiting
            self.lock.release()
            time.sleep(wait_time)
            self.lock.acquire()

            # Update last error time to prevent waiting again
            self.last_error_time = time.time() - self.current_back_off

    def reset(self) -> None:
        """Reset the rate limiter state."""
        with self.lock:
            self.request_timestamps = []
            self.tokens = float(self.config.burst_capacity or self.config.max_requests)
            self.last_token_refresh = time.time()
            self.consecutive_errors = 0
            self.current_back_off = 0.0

            # Reset endpoint-specific limiters
            for limiter in self.endpoint_limiters.values():
                limiter.reset()

            logger.debug("Rate limiter reset")

    def shutdown(self) -> None:
        """Shutdown the rate limiter and its worker threads."""
        self.running = False

        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)

        # Shutdown endpoint-specific limiters
        for limiter in self.endpoint_limiters.values():
            limiter.shutdown()


class EndpointRateLimiter:
    """
    Enhanced rate limiter with per-endpoint tracking.

    This class manages multiple rate limiters for different endpoints,
    allowing for more precise control over API request rates.
    """

    def __init__(
        self,
        global_config: RateLimitConfig,
        endpoint_configs: Optional[Dict[str, RateLimitConfig]] = None
    ):
        """
        Initialize the endpoint rate limiter.

        Args:
            global_config: Global rate limit configuration
            endpoint_configs: Endpoint-specific configurations
        """
        # Create global limiter
        self.global_limiter = RateLimiter(
            max_requests=global_config.max_requests,
            time_window=global_config.time_window,
            algorithm=global_config.algorithm,
            burst_capacity=global_config.burst_capacity,
            fill_rate=global_config.fill_rate,
            error_back_off=global_config.error_back_off,
            back_off_multiplier=global_config.back_off_multiplier,
            max_back_off=global_config.max_back_off,
            priority_levels=global_config.priority_levels
        )

        # Create endpoint-specific limiters
        self.endpoint_limiters: Dict[str, RateLimiter] = {}

        if endpoint_configs:
            for endpoint, config in endpoint_configs.items():
                self.endpoint_limiters[endpoint] = RateLimiter(
                    max_requests=config.max_requests,
                    time_window=config.time_window,
                    algorithm=config.algorithm,
                    burst_capacity=config.burst_capacity,
                    fill_rate=config.fill_rate,
                    error_back_off=config.error_back_off,
                    back_off_multiplier=config.back_off_multiplier,
                    max_back_off=config.max_back_off,
                    priority_levels=config.priority_levels
                )

        logger.info(f"Endpoint rate limiter initialized with {len(self.endpoint_limiters)} endpoint configurations")

    def acquire(self, endpoint: Optional[str] = None, priority: int = 0) -> None:
        """
        Acquire permission to make a request, respecting both global and endpoint limits.

        Args:
            endpoint: The API endpoint being accessed
            priority: Request priority
        """
        # First, acquire permission from the global limiter
        self.global_limiter.acquire(priority=priority)

        # Then, if endpoint is specified and has a specific limiter, acquire from that too
        if endpoint and endpoint in self.endpoint_limiters:
            self.endpoint_limiters[endpoint].acquire(priority=priority)

    def report_success(self, endpoint: Optional[str] = None) -> None:
        """
        Report a successful request.

        Args:
            endpoint: The API endpoint that was accessed
        """
        self.global_limiter.report_success()

        if endpoint and endpoint in self.endpoint_limiters:
            self.endpoint_limiters[endpoint].report_success()

    def report_error(self, endpoint: Optional[str] = None, is_rate_limit_error: bool = True) -> None:
        """
        Report a rate limit error.

        Args:
            endpoint: The API endpoint that had an error
            is_rate_limit_error: Whether this was a rate limit error
        """
        self.global_limiter.report_error(is_rate_limit_error)

        if endpoint and endpoint in self.endpoint_limiters:
            self.endpoint_limiters[endpoint].report_error(is_rate_limit_error)

    def reset(self) -> None:
        """Reset all limiters."""
        self.global_limiter.reset()

        for limiter in self.endpoint_limiters.values():
            limiter.reset()

    def shutdown(self) -> None:
        """Shutdown all limiters."""
        self.global_limiter.shutdown()

        for limiter in self.endpoint_limiters.values():
            limiter.shutdown()

    def add_endpoint_config(self, endpoint: str, config: RateLimitConfig) -> None:
        """
        Add a new endpoint-specific rate limit configuration.

        Args:
            endpoint: The API endpoint to configure
            config: Rate limit configuration for this endpoint
        """
        if endpoint in self.endpoint_limiters:
            self.endpoint_limiters[endpoint].shutdown()

        self.endpoint_limiters[endpoint] = RateLimiter(
            max_requests=config.max_requests,
            time_window=config.time_window,
            algorithm=config.algorithm,
            burst_capacity=config.burst_capacity,
            fill_rate=config.fill_rate,
            error_back_off=config.error_back_off,
            back_off_multiplier=config.back_off_multiplier,
            max_back_off=config.max_back_off,
            priority_levels=config.priority_levels
        )

        logger.info(f"Added rate limit configuration for endpoint: {endpoint}")


# Convenience factory functions

def create_default_rate_limiter(max_requests: int = 10, time_window: float = 1.0) -> RateLimiter:
    """
    Create a rate limiter with sensible defaults.

    Args:
        max_requests: Maximum requests per time window
        time_window: Time window in seconds

    Returns:
        Configured rate limiter
    """
    return RateLimiter(
        max_requests=max_requests,
        time_window=time_window,
        algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
        error_back_off=True
    )


def create_forex_rate_limiter() -> EndpointRateLimiter:
    """
    Create a rate limiter configured for typical forex API limitations.

    Returns:
        Configured endpoint rate limiter
    """
    # Global config - moderate overall rate
    global_config = RateLimitConfig(
        max_requests=120,
        time_window=60.0,  # 2 requests per second on average
        algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
        burst_capacity=30,  # Allow short bursts
        error_back_off=True,
        priority_levels=3  # Support different priorities
    )

    # Endpoint-specific configs
    endpoint_configs = {
        # Price data - higher limits
        "pricing": RateLimitConfig(
            max_requests=180,
            time_window=60.0,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            burst_capacity=40
        ),

        # Order creation - lower limits
        "orders": RateLimitConfig(
            max_requests=60,
            time_window=60.0,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            burst_capacity=20
        ),

        # Account data - lowest limits
        "accounts": RateLimitConfig(
            max_requests=30,
            time_window=60.0,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            burst_capacity=10
        )
    }

    return EndpointRateLimiter(global_config, endpoint_configs)


def create_crypto_rate_limiter() -> EndpointRateLimiter:
    """
    Create a rate limiter configured for typical crypto exchange API limitations.

    Returns:
        Configured endpoint rate limiter
    """
    # Global config - higher rate but with more aggressive backoff
    global_config = RateLimitConfig(
        max_requests=1200,
        time_window=60.0,  # 20 requests per second on average
        algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
        burst_capacity=100,  # Allow larger bursts
        error_back_off=True,
        back_off_multiplier=4.0,  # More aggressive backoff
        priority_levels=3
    )

    # Endpoint-specific configs
    endpoint_configs = {
        # Market data - highest limits
        "market_data": RateLimitConfig(
            max_requests=1800,
            time_window=60.0,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            burst_capacity=150
        ),

        # Order management - moderate limits
        "orders": RateLimitConfig(
            max_requests=600,
            time_window=60.0,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            burst_capacity=50
        ),

        # Account operations - lowest limits
        "account": RateLimitConfig(
            max_requests=120,
            time_window=60.0,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            burst_capacity=20
        )
    }

    return EndpointRateLimiter(global_config, endpoint_configs)