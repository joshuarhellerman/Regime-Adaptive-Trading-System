"""
prediction_service.py - Model Serving with Latency Guarantees

This module provides a high-performance model serving framework with latency
guarantees, caching, batching, and performance monitoring.
"""

import logging
import threading
import time
import uuid
import queue
import numpy as np
import functools
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, TypeVar
from enum import Enum
from dataclasses import dataclass, field
import concurrent.futures
from datetime import datetime

from core.event_bus import EventTopics, Event, get_event_bus, create_event, EventPriority
from core.health_monitor import HealthMonitor, HealthStatus
from models.production.model_deployer import ModelDeployer, DeploymentType

logger = logging.getLogger(__name__)

# Type variable for generic prediction input/output
T = TypeVar('T')
U = TypeVar('U')


class PredictionPriority(Enum):
    """Priority levels for prediction requests"""
    CRITICAL = 0    # Highest priority, must be processed immediately
    HIGH = 1        # High priority, processed before normal requests
    NORMAL = 2      # Normal priority
    LOW = 3         # Low priority, can be delayed if system is busy
    BACKGROUND = 4  # Lowest priority, processed when system is idle


@dataclass
class PredictionRequest:
    """Represents a prediction request"""
    id: str
    model_type: str
    scope: str
    features: Any
    priority: PredictionPriority
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    batch_id: Optional[str] = None
    max_latency_ms: Optional[float] = None
    callback: Optional[Callable[[Any], None]] = None


@dataclass
class PredictionResult:
    """Represents a prediction result"""
    request_id: str
    model_id: str
    prediction: Any
    confidence: Optional[float] = None
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class PredictionService:
    """
    Provides high-performance model serving with latency guarantees,
    batch processing, caching, and performance monitoring.
    """

    def __init__(self,
                 model_deployer: Optional[ModelDeployer] = None,
                 health_monitor: Optional[HealthMonitor] = None,
                 event_bus=None,
                 max_batch_size: int = 32,
                 max_batch_latency_ms: float = 50.0,
                 prediction_timeout_ms: float = 500.0,
                 max_concurrent_batches: int = 4,
                 enable_caching: bool = True,
                 cache_ttl_ms: float = 1000.0,
                 max_cache_size: int = 1000,
                 preload_models: bool = True):
        """
        Initialize the prediction service.

        Args:
            model_deployer: Model deployer instance
            health_monitor: Health monitor instance
            event_bus: Event bus instance
            max_batch_size: Maximum batch size for prediction requests
            max_batch_latency_ms: Maximum latency for batch formation (ms)
            prediction_timeout_ms: Timeout for prediction requests (ms)
            max_concurrent_batches: Maximum number of concurrent batch predictions
            enable_caching: Whether to enable prediction caching
            cache_ttl_ms: Time-to-live for cached predictions (ms)
            max_cache_size: Maximum number of cached predictions
            preload_models: Whether to preload models on startup
        """
        self.model_deployer = model_deployer
        self.health_monitor = health_monitor
        self.event_bus = event_bus or get_event_bus()

        # Configuration
        self.max_batch_size = max_batch_size
        self.max_batch_latency_ms = max_batch_latency_ms
        self.prediction_timeout_ms = prediction_timeout_ms
        self.max_concurrent_batches = max_concurrent_batches
        self.enable_caching = enable_caching
        self.cache_ttl_ms = cache_ttl_ms
        self.max_cache_size = max_cache_size

        # Priority queues for requests
        self.request_queues = {
            priority: queue.PriorityQueue() for priority in PredictionPriority
        }

        # Prediction cache
        self.prediction_cache: Dict[str, Tuple[Any, float]] = {}  # (prediction, timestamp)

        # Batch processing queues
        self.batch_queues: Dict[str, List[PredictionRequest]] = {}
        self.batch_timers: Dict[str, float] = {}

        # Thread pool for predictions
        self.prediction_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent_batches,
            thread_name_prefix="PredictionWorker"
        )

        # Thread for batch formation
        self.batch_thread = None

        # Loaded models cache
        self.model_cache: Dict[str, Any] = {}

        # Metrics
        self.metrics = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_error": 0,
            "batch_count": 0,
            "avg_batch_size": 0,
            "avg_latency_ms": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "latency_histogram": {
                "0-10ms": 0,
                "10-50ms": 0,
                "50-100ms": 0,
                "100-500ms": 0,
                "500ms+": 0
            }
        }

        # Running state
        self._running = False
        self._lock = threading.RLock()

        # Register for model deployment events
        self.event_bus.subscribe(
            EventTopics.MODEL_DEPLOYED,
            self._handle_model_deployed
        )

        # Register with health monitor
        if self.health_monitor:
            self.health_monitor.register_component("prediction_service", "service")

        logger.info("Prediction service initialized")

        # Preload models if configured
        if preload_models and self.model_deployer:
            self._preload_models()

    def start(self):
        """Start the prediction service"""
        with self._lock:
            if self._running:
                logger.warning("Prediction service is already running")
                return

            self._running = True

            # Start batch formation thread
            self.batch_thread = threading.Thread(
                target=self._batch_formation_loop,
                name="BatchFormationThread",
                daemon=True
            )
            self.batch_thread.start()

            logger.info("Prediction service started")

            # Report initial health status
            if self.health_monitor:
                self.health_monitor.update_component_health(
                    "prediction_service",
                    status=HealthStatus.HEALTHY,
                    metrics={
                        "start_time": time.time(),
                        "max_batch_size": self.max_batch_size,
                        "max_batch_latency_ms": self.max_batch_latency_ms,
                        "max_concurrent_batches": self.max_concurrent_batches,
                        "enable_caching": self.enable_caching
                    }
                )

    def stop(self):
        """Stop the prediction service"""
        with self._lock:
            if not self._running:
                logger.warning("Prediction service is not running")
                return

            self._running = False

            # Wait for batch thread to finish
            if self.batch_thread and self.batch_thread.is_alive():
                self.batch_thread.join(timeout=2.0)

            # Shutdown thread pool
            self.prediction_executor.shutdown(wait=True)

            logger.info("Prediction service stopped")

    def predict(self,
               model_type: Union[DeploymentType, str],
               features: Any,
               scope: str = "default",
               priority: PredictionPriority = PredictionPriority.NORMAL,
               metadata: Optional[Dict[str, Any]] = None,
               max_latency_ms: Optional[float] = None,
               batch: bool = True,
               request_id: Optional[str] = None,
               use_cache: bool = True) -> PredictionResult:
        """
        Request a prediction synchronously.

        Args:
            model_type: Type of model to use
            features: Input features for prediction
            scope: Model scope
            priority: Priority of the request
            metadata: Additional metadata
            max_latency_ms: Maximum latency for this request (ms)
            batch: Whether to allow batching this request
            request_id: Optional request ID
            use_cache: Whether to check the cache for this request

        Returns:
            Prediction result
        """
        # Convert enum to string if necessary
        if isinstance(model_type, DeploymentType):
            model_type = model_type.value

        # Generate request ID if not provided
        if request_id is None:
            request_id = str(uuid.uuid4())

        # Create prediction request
        request = PredictionRequest(
            id=request_id,
            model_type=model_type,
            scope=scope,
            features=features,
            priority=priority,
            metadata=metadata or {},
            max_latency_ms=max_latency_ms
        )

        start_time = time.time()

        # Check cache if enabled
        if use_cache and self.enable_caching:
            cache_result = self._check_cache(request)
            if cache_result is not None:
                return cache_result

        if not self._running:
            return PredictionResult(
                request_id=request_id,
                model_id="",
                prediction=None,
                latency_ms=(time.time() - start_time) * 1000,
                error="Prediction service is not running"
            )

        # Handle immediate processing vs. batching
        if batch and not max_latency_ms:
            # Process in batch
            result_queue = queue.Queue()

            # Add callback to request
            request.callback = lambda res: result_queue.put(res)

            # Enqueue request
            self._enqueue_request(request)

            # Wait for result with timeout
            try:
                result = result_queue.get(timeout=self.prediction_timeout_ms / 1000)
                return result
            except queue.Empty:
                return PredictionResult(
                    request_id=request_id,
                    model_id="",
                    prediction=None,
                    latency_ms=self.prediction_timeout_ms,
                    error="Prediction timed out"
                )
        else:
            # Process immediately
            return self._process_single_request(request)

    async def predict_async(self,
                         model_type: Union[DeploymentType, str],
                         features: Any,
                         scope: str = "default",
                         priority: PredictionPriority = PredictionPriority.NORMAL,
                         metadata: Optional[Dict[str, Any]] = None,
                         max_latency_ms: Optional[float] = None,
                         batch: bool = True,
                         request_id: Optional[str] = None,
                         use_cache: bool = True) -> PredictionResult:
        """
        Request a prediction asynchronously.

        Args:
            model_type: Type of model to use
            features: Input features for prediction
            scope: Model scope
            priority: Priority of the request
            metadata: Additional metadata
            max_latency_ms: Maximum latency for this request (ms)
            batch: Whether to allow batching this request
            request_id: Optional request ID
            use_cache: Whether to check the cache for this request

        Returns:
            Prediction result
        """
        import asyncio

        # Convert enum to string if necessary
        if isinstance(model_type, DeploymentType):
            model_type = model_type.value

        # Generate request ID if not provided
        if request_id is None:
            request_id = str(uuid.uuid4())

        # Create prediction request
        request = PredictionRequest(
            id=request_id,
            model_type=model_type,
            scope=scope,
            features=features,
            priority=priority,
            metadata=metadata or {},
            max_latency_ms=max_latency_ms
        )

        start_time = time.time()

        # Check cache if enabled
        if use_cache and self.enable_caching:
            cache_result = self._check_cache(request)
            if cache_result is not None:
                return cache_result

        if not self._running:
            return PredictionResult(
                request_id=request_id,
                model_id="",
                prediction=None,
                latency_ms=(time.time() - start_time) * 1000,
                error="Prediction service is not running"
            )

        # Set up async result handling
        future = asyncio.get_event_loop().create_future()

        def callback(result):
            asyncio.run_coroutine_threadsafe(
                self._set_future(future, result),
                asyncio.get_event_loop()
            )

        # Add callback to request
        request.callback = callback

        if batch:
            # Process in batch
            self._enqueue_request(request)

            # Wait for result with timeout
            try:
                return await asyncio.wait_for(
                    future,
                    timeout=self.prediction_timeout_ms / 1000
                )
            except asyncio.TimeoutError:
                return PredictionResult(
                    request_id=request_id,
                    model_id="",
                    prediction=None,
                    latency_ms=self.prediction_timeout_ms,
                    error="Prediction timed out"
                )
        else:
            # Process immediately using thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._process_single_request(request)
            )
            return result

    async def _set_future(self, future, result):
        """Helper to set future result for async predictions"""
        if not future.done():
            future.set_result(result)

    def predict_batch(self,
                     model_type: Union[DeploymentType, str],
                     features_batch: List[Any],
                     scope: str = "default",
                     priority: PredictionPriority = PredictionPriority.NORMAL,
                     metadata: Optional[Dict[str, Any]] = None,
                     max_latency_ms: Optional[float] = None,
                     batch_id: Optional[str] = None) -> List[PredictionResult]:
        """
        Process a batch of predictions synchronously.

        Args:
            model_type: Type of model to use
            features_batch: List of input features
            scope: Model scope
            priority: Priority of the request
            metadata: Additional metadata
            max_latency_ms: Maximum latency for this batch (ms)
            batch_id: Optional batch ID

        Returns:
            List of prediction results
        """
        # Convert enum to string if necessary
        if isinstance(model_type, DeploymentType):
            model_type = model_type.value

        # Generate batch ID if not provided
        if batch_id is None:
            batch_id = str(uuid.uuid4())

        if not features_batch:
            return []

        if not self._running:
            return [
                PredictionResult(
                    request_id=f"{batch_id}_{i}",
                    model_id="",
                    prediction=None,
                    error="Prediction service is not running"
                )
                for i in range(len(features_batch))
            ]

        # Create batch key
        batch_key = f"{model_type}_{scope}"

        # Process batch directly
        start_time = time.time()

        # Get model
        model, model_id = self._get_model(model_type, scope)
        if model is None:
            return [
                PredictionResult(
                    request_id=f"{batch_id}_{i}",
                    model_id="",
                    prediction=None,
                    error=f"Model not found for type {model_type} and scope {scope}"
                )
                for i in range(len(features_batch))
            ]

        try:
            # Check if model supports batch prediction
            if hasattr(model, 'predict_batch'):
                predictions = model.predict_batch(features_batch)
            else:
                # Fall back to individual predictions
                predictions = [model.predict(features) for features in features_batch]

            # Calculate latency
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            # Create results
            results = []
            for i, prediction in enumerate(predictions):
                result = PredictionResult(
                    request_id=f"{batch_id}_{i}",
                    model_id=model_id,
                    prediction=prediction,
                    latency_ms=latency_ms,
                    metadata={
                        "batch_id": batch_id,
                        "batch_size": len(features_batch),
                        "index": i
                    }
                )
                results.append(result)

                # Add to cache if enabled
                if self.enable_caching:
                    cache_key = self._make_cache_key(model_type, scope, features_batch[i])
                    self._add_to_cache(cache_key, prediction)

            # Update metrics
            with self._lock:
                self.metrics["requests_total"] += len(features_batch)
                self.metrics["requests_success"] += len(features_batch)
                self.metrics["batch_count"] += 1

                # Update latency histogram
                self._update_latency_histogram(latency_ms)

                # Update average batch size
                n = self.metrics["batch_count"]
                self.metrics["avg_batch_size"] = (
                    (n - 1) * self.metrics["avg_batch_size"] + len(features_batch)
                ) / n

            # Report to health monitor
            if self.health_monitor:
                self.health_monitor.update_component_health(
                    component_id="prediction_service",
                    metrics={
                        "batch_count": self.metrics["batch_count"],
                        "avg_batch_size": self.metrics["avg_batch_size"],
                        "last_batch_latency_ms": latency_ms,
                        "last_batch_size": len(features_batch)
                    }
                )

                # Log model performance
                self.health_monitor.monitor_model_performance(
                    model_id=model_id,
                    prediction_error=0.0,  # Can't calculate without ground truth
                    inference_time=latency_ms / len(features_batch),
                    model_version=metadata.get("version") if metadata else None
                )

            return results

        except Exception as e:
            logger.error(f"Error processing batch prediction: {str(e)}", exc_info=True)

            # Update metrics
            with self._lock:
                self.metrics["requests_total"] += len(features_batch)
                self.metrics["requests_error"] += len(features_batch)

            # Report error to health monitor
            if self.health_monitor:
                self.health_monitor.log_error(
                    component_id="prediction_service",
                    error_type="batch_prediction",
                    error_message=str(e)
                )

            return [
                PredictionResult(
                    request_id=f"{batch_id}_{i}",
                    model_id=model_id,
                    prediction=None,
                    error=str(e)
                )
                for i in range(len(features_batch))
            ]

    def _enqueue_request(self, request: PredictionRequest):
        """
        Enqueue a prediction request.

        Args:
            request: Prediction request
        """
        # Determine batch key
        batch_key = f"{request.model_type}_{request.scope}"

        # Add to appropriate queue based on priority
        queue_item = (request.timestamp, request)
        self.request_queues[request.priority].put(queue_item)

        # Update metrics
        with self._lock:
            self.metrics["requests_total"] += 1

    def _batch_formation_loop(self):
        """Main loop for batch formation"""
        logger.debug("Batch formation loop started")

        while self._running:
            try:
                # Process existing batches that have reached timeout
                self._process_timed_out_batches()

                # Process requests from queues in priority order
                self._process_queues()

                # Sleep a short time to avoid tight loop
                time.sleep(0.001)  # 1ms sleep

            except Exception as e:
                logger.error(f"Error in batch formation loop: {str(e)}", exc_info=True)
                time.sleep(0.1)  # Sleep longer on error

    def _process_timed_out_batches(self):
        """Process batches that have reached their timeout"""
        current_time = time.time()

        # Find batches that have timed out
        timed_out_keys = []
        with self._lock:
            for batch_key, start_time in self.batch_timers.items():
                elapsed_ms = (current_time - start_time) * 1000
                if elapsed_ms >= self.max_batch_latency_ms:
                    timed_out_keys.append(batch_key)

        # Process each timed out batch
        for batch_key in timed_out_keys:
            with self._lock:
                if batch_key in self.batch_queues:
                    batch = self.batch_queues[batch_key]
                    del self.batch_queues[batch_key]
                    del self.batch_timers[batch_key]
                else:
                    batch = []

            if batch:
                self._process_batch(batch_key, batch)

    def _process_queues(self):
        """Process requests from queues in priority order"""
        # Check each priority queue in order
        for priority in PredictionPriority:
            queue_obj = self.request_queues[priority]

            # Process until queue is empty or batch limits reached
            while not queue_obj.empty():
                # Check if we can process any more batches
                with self._lock:
                    active_batches = len(self.batch_queues)
                    if active_batches >= self.max_concurrent_batches:
                        break

                try:
                    # Get request from queue
                    _, request = queue_obj.get_nowait()
                    queue_obj.task_done()
                except queue.Empty:
                    break

                # Determine batch key
                batch_key = f"{request.model_type}_{request.scope}"

                # Check if request has a maximum latency requirement
                if request.max_latency_ms is not None and request.max_latency_ms < self.max_batch_latency_ms:
                    # Process immediately if latency requirement is strict
                    result = self._process_single_request(request)

                    # Call callback if provided
                    if request.callback:
                        request.callback(result)

                    continue

                # Add to batch queue
                with self._lock:
                    if batch_key not in self.batch_queues:
                        self.batch_queues[batch_key] = []
                        self.batch_timers[batch_key] = time.time()

                    self.batch_queues[batch_key].append(request)

                    # If batch is full, process it
                    if len(self.batch_queues[batch_key]) >= self.max_batch_size:
                        batch = self.batch_queues[batch_key]
                        del self.batch_queues[batch_key]
                        del self.batch_timers[batch_key]

                        # Process batch in a separate thread
                        self.prediction_executor.submit(self._process_batch, batch_key, batch)

    def _process_batch(self, batch_key: str, batch: List[PredictionRequest]):
        """
        Process a batch of prediction requests.

        Args:
            batch_key: Key identifying the model type and scope
            batch: List of prediction requests
        """
        if not batch:
            return

        # Extract model type and scope from batch key
        model_type, scope = batch_key.split('_', 1)

        # Get model
        model, model_id = self._get_model(model_type, scope)
        if model is None:
            # Handle case where model is not available
            error = f"Model not found for type {model_type} and scope {scope}"
            logger.error(error)

            # Return error for each request
            for request in batch:
                result = PredictionResult(
                    request_id=request.id,
                    model_id="",
                    prediction=None,
                    error=error
                )

                if request.callback:
                    request.callback(result)

            return

        # Prepare batch inputs
        features_batch = [request.features for request in batch]

        try:
            start_time = time.time()

            # Check if model supports batch prediction
            if hasattr(model, 'predict_batch'):
                predictions = model.predict_batch(features_batch)
            else:
                # Fall back to individual predictions
                predictions = [model.predict(features) for features in features_batch]

            # Calculate latency
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            # Send results to callbacks
            for i, request in enumerate(batch):
                prediction = predictions[i]

                # Create result
                result = PredictionResult(
                    request_id=request.id,
                    model_id=model_id,
                    prediction=prediction,
                    latency_ms=latency_ms,
                    timestamp=end_time,
                    metadata={
                        "batch_size": len(batch),
                        "batch_index": i,
                        **request.metadata
                    }
                )

                # Add to cache if enabled
                if self.enable_caching:
                    cache_key = self._make_cache_key(model_type, scope, request.features)
                    self._add_to_cache(cache_key, prediction)

                # Invoke callback if provided
                if request.callback:
                    request.callback(result)

            # Update metrics
            with self._lock:
                self.metrics["requests_success"] += len(batch)
                self.metrics["batch_count"] += 1

                # Update latency histogram
                self._update_latency_histogram(latency_ms)

                # Update average batch size
                n = self.metrics["batch_count"]
                self.metrics["avg_batch_size"] = (
                    (n - 1) * self.metrics["avg_batch_size"] + len(batch)
                ) / n

                # Update average latency
                n = self.metrics["requests_success"]
                self.metrics["avg_latency_ms"] = (
                    (n - len(batch)) * self.metrics["avg_latency_ms"] + latency_ms * len(batch)
                ) / n

            # Report to health monitor
            if self.health_monitor:
                self.health_monitor.update_component_health(
                    component_id="prediction_service",
                    metrics={
                        "batch_count": self.metrics["batch_count"],
                        "avg_batch_size": self.metrics["avg_batch_size"],
                        "last_batch_latency_ms": latency_ms,
                        "last_batch_size": len(batch)
                    }
                )

                # Log model performance
                self.health_monitor.monitor_model_performance(
                    model_id=model_id,
                    prediction_error=0.0,  # Can't calculate without ground truth
                    inference_time=latency_ms / len(batch)
                )

            logger.debug(f"Processed batch of {len(batch)} predictions for {model_type}/{scope} in {latency_ms:.2f}ms")

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}", exc_info=True)

            # Update metrics
            with self._lock:
                self.metrics["requests_error"] += len(batch)

            # Report error to health monitor
            if self.health_monitor:
                self.health_monitor.log_error(
                    component_id="prediction_service",
                    error_type="batch_prediction",
                    error_message=str(e)
                )

            # Send error to each request
            for request in batch:
                result = PredictionResult(
                    request_id=request.id,
                    model_id=model_id,
                    prediction=None,
                    error=str(e)
                )

                if request.callback:
                    request.callback(result)

    def _process_single_request(self, request: PredictionRequest) -> PredictionResult:
        """
        Process a single prediction request immediately.

        Args:
            request: Prediction request

        Returns:
            Prediction result
        """
        start_time = time.time()

        # Get model
        model, model_id = self._get_model(request.model_type, request.scope)
        if model is None:
            error = f"Model not found for type {request.model_type} and scope {request.scope}"
            logger.error(error)

            # Update metrics
            with self._lock:
                self.metrics["requests_total"] += 1
                self.metrics["requests_error"] += 1

            return PredictionResult(
                request_id=request.id,
                model_id="",
                prediction=None,
                latency_ms=(time.time() - start_time) * 1000,
                error=error
            )

        try:
            # Make prediction
            prediction = model.predict(request.features)

            # Calculate latency
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000

            # Add to cache if enabled
            if self.enable_caching:
                cache_key = self._make_cache_key(request.model_type, request.scope, request.features)
                self._add_to_cache(cache_key, prediction)

            # Create result
            result = PredictionResult(
                request_id=request.id,
                model_id=model_id,
                prediction=prediction,
                latency_ms=latency_ms,
                timestamp=end_time,
                metadata=request.metadata
            )

            # Update metrics
            with self._lock:
                self.metrics["requests_total"] += 1
                self.metrics["requests_success"] += 1

                # Update latency histogram
                self._update_latency_histogram(latency_ms)

                # Update average latency
                n = self.metrics["requests_success"]
                self.metrics["avg_latency_ms"] = (
                    (n - 1) * self.metrics["avg_latency_ms"] + latency_ms
                ) / n

            # Report to health monitor
            if self.health_monitor:
                self.health_monitor.log_operation(
                    component_id="prediction_service",
                    operation_type="single_prediction",
                    duration_ms=latency_ms,
                    success=True
                )

                # Log model performance
                self.health_monitor.monitor_model_performance(
                    model_id=model_id,
                    prediction_error=0.0,  # Can't calculate without ground truth
                    inference_time=latency_ms
                )

            return result

        except Exception as e:
            logger.error(f"Error processing prediction: {str(e)}", exc_info=True)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Update metrics
            with self._lock:
                self.metrics["requests_total"] += 1
                self.metrics["requests_error"] += 1

            # Report error to health monitor
            if self.health_monitor:
                self.health_monitor.log_error(
                    component_id="prediction_service",
                    error_type="single_prediction",
                    error_message=str(e)
                )

                self.health_monitor.log_operation(
                    component_id="prediction_service",
                    operation_type="single_prediction",
                    duration_ms=latency_ms,
                    success=False
                )

            return PredictionResult(
                request_id=request.id,
                model_id=model_id,
                prediction=None,
                latency_ms=latency_ms,
                error=str(e)
            )

    def _get_model(self, model_type: str, scope: str) -> Tuple[Optional[Any], str]:
        """
        Get a model for prediction.

        Args:
            model_type: Type of model
            scope: Scope of model

        Returns:
            Tuple of (model object, model_id)
        """
        if not self.model_deployer:
            logger.error("No model deployer configured")
            return None, ""

        # Check model cache first
        cache_key = f"{model_type}_{scope}"
        if cache_key in self.model_cache:
            model, model_id = self.model_cache[cache_key]
            return model, model_id

        # Get model info
        model_info = self.model_deployer.get_active_model(model_type, scope)
        if model_info is None:
            logger.error(f"No active model found for type {model_type} and scope {scope}")
            return None, ""

        # Load model
        model = self.model_deployer.load_model(model_info["model_id"])
        if model is None:
            logger.error(f"Failed to load model {model_info['model_id']}")
            return None, ""

        # Cache model
        self.model_cache[cache_key] = (model, model_info["model_id"])

        return model, model_info["model_id"]

    def _preload_models(self):
        """Preload active models into memory"""
        if not self.model_deployer:
            return

        # Get list of active models
        active_models = []
        for model_type in DeploymentType:
            model_info = self.model_deployer.get_active_model(model_type.value)
            if model_info:
                active_models.append((model_type.value, "default", model_info["model_id"]))

        logger.info(f"Preloading {len(active_models)} active models")

        # Load each model
        for model_type, scope, model_id in active_models:
            try:
                model = self.model_deployer.load_model(model_id)
                if model is not None:
                    cache_key = f"{model_type}_{scope}"
                    self.model_cache[cache_key] = (model, model_id)
                    logger.info(f"Preloaded model {model_id} for {model_type}/{scope}")
            except Exception as e:
                logger.error(f"Error preloading model {model_id}: {str(e)}")

    def _make_cache_key(self, model_type: str, scope: str, features: Any) -> str:
        """
        Create a cache key for a prediction.

        Args:
            model_type: Type of model
            scope: Scope of model
            features: Input features

        Returns:
            Cache key string
        """
        # Basic model/scope identifier
        base_key = f"{model_type}_{scope}"

        # Handle different feature types
        if isinstance(features, (list, tuple)) and all(isinstance(x, (int, float)) for x in features):
            # For simple numeric lists/arrays, use a rounded string representation
            features_str = "_".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in features)
        elif isinstance(features, dict):
            # For dictionaries, use sorted keys
            features_str = "_".join(f"{k}:{v}" for k, v in sorted(features.items()))
        elif hasattr(features, 'tolist') and callable(getattr(features, 'tolist')):
            # For numpy arrays or similar
            features_str = "_".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in features.tolist())
        else:
            # For other types, use hash
            features_str = str(hash(str(features)))

        return f"{base_key}_{features_str}"

    def _check_cache(self, request: PredictionRequest) -> Optional[PredictionResult]:
        """
        Check if a prediction is available in cache.

        Args:
            request: Prediction request

        Returns:
            Cached prediction result or None if not found
        """
        if not self.enable_caching:
            return None

        cache_key = self._make_cache_key(request.model_type, request.scope, request.features)

        with self._lock:
            if cache_key in self.prediction_cache:
                prediction, timestamp = self.prediction_cache[cache_key]

                # Check if cache entry is expired
                current_time = time.time()
                if (current_time - timestamp) * 1000 > self.cache_ttl_ms:
                    # Remove expired entry
                    del self.prediction_cache[cache_key]
                    self.metrics["cache_misses"] += 1
                    return None

                # Return cached result
                self.metrics["cache_hits"] += 1

                return PredictionResult(
                    request_id=request.id,
                    model_id="cached",
                    prediction=prediction,
                    latency_ms=0.0,
                    timestamp=current_time,
                    metadata={
                        **request.metadata,
                        "cached": True,
                        "cache_age_ms": (current_time - timestamp) * 1000
                    }
                )

        # Cache miss
        with self._lock:
            self.metrics["cache_misses"] += 1

        return None

    def _add_to_cache(self, cache_key: str, prediction: Any):
        """
        Add a prediction to the cache.

        Args:
            cache_key: Cache key
            prediction: Prediction to cache
        """
        if not self.enable_caching:
            return

        with self._lock:
            # Check if cache is full
            if len(self.prediction_cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = min(self.prediction_cache.items(), key=lambda x: x[1][1])[0]
                del self.prediction_cache[oldest_key]

            # Add to cache
            self.prediction_cache[cache_key] = (prediction, time.time())

    def _update_latency_histogram(self, latency_ms: float):
        """
        Update latency histogram metrics.

        Args:
            latency_ms: Latency in milliseconds
        """
        if latency_ms < 10:
            self.metrics["latency_histogram"]["0-10ms"] += 1
        elif latency_ms < 50:
            self.metrics["latency_histogram"]["10-50ms"] += 1
        elif latency_ms < 100:
            self.metrics["latency_histogram"]["50-100ms"] += 1
        elif latency_ms < 500:
            self.metrics["latency_histogram"]["100-500ms"] += 1
        else:
            self.metrics["latency_histogram"]["500ms+"] += 1

    def _handle_model_deployed(self, event: Event):
        """
        Handle model deployment event.

        Args:
            event: Model deployed event
        """
        if not event.data:
            return

        model_id = event.data.get("model_id")
        model_type = event.data.get("model_type")
        scope = event.data.get("scope", "default")

        if not model_id or not model_type:
            return

        # Invalidate model cache
        cache_key = f"{model_type}_{scope}"
        if cache_key in self.model_cache:
            del self.model_cache[cache_key]

        # Preload new model
        if self.model_deployer:
            try:
                model = self.model_deployer.load_model(model_id)
                if model is not None:
                    self.model_cache[cache_key] = (model, model_id)
                    logger.info(f"Loaded new model {model_id} for {model_type}/{scope}")
            except Exception as e:
                logger.error(f"Error loading new model {model_id}: {str(e)}")

        # Invalidate prediction cache for this model type/scope
        with self._lock:
            # Remove cache entries for this model type/scope
            cache_prefix = f"{model_type}_{scope}_"
            for key in list(self.prediction_cache.keys()):
                if key.startswith(cache_prefix):
                    del self.prediction_cache[key]

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get service metrics.

        Returns:
            Dictionary of service metrics
        """
        with self._lock:
            # Create a copy of metrics
            metrics_copy = {**self.metrics}

            # Add queue sizes
            metrics_copy["queue_sizes"] = {
                priority.name: self.request_queues[priority].qsize()
                for priority in PredictionPriority
            }

            # Add batch information
            metrics_copy["active_batches"] = len(self.batch_queues)
            metrics_copy["batch_sizes"] = {
                key: len(batch) for key, batch in self.batch_queues.items()
            }

            # Add cache information
            metrics_copy["cache_size"] = len(self.prediction_cache)
            metrics_copy["cache_hit_rate"] = (
                self.metrics["cache_hits"] /
                (self.metrics["cache_hits"] + self.metrics["cache_misses"])
                if (self.metrics["cache_hits"] + self.metrics["cache_misses"]) > 0
                else 0.0
            )

            return metrics_copy

    def clear_cache(self):
        """Clear prediction cache"""
        with self._lock:
            self.prediction_cache.clear()
            logger.info("Prediction cache cleared")

    def set_configuration(self, config: Dict[str, Any]) -> bool:
        """
        Update service configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Success boolean
        """
        with self._lock:
            try:
                # Update configuration parameters
                if "max_batch_size" in config:
                    self.max_batch_size = int(config["max_batch_size"])

                if "max_batch_latency_ms" in config:
                    self.max_batch_latency_ms = float(config["max_batch_latency_ms"])

                if "prediction_timeout_ms" in config:
                    self.prediction_timeout_ms = float(config["prediction_timeout_ms"])

                if "max_concurrent_batches" in config:
                    self.max_concurrent_batches = int(config["max_concurrent_batches"])

                if "enable_caching" in config:
                    self.enable_caching = bool(config["enable_caching"])

                if "cache_ttl_ms" in config:
                    self.cache_ttl_ms = float(config["cache_ttl_ms"])

                if "max_cache_size" in config:
                    self.max_cache_size = int(config["max_cache_size"])

                # Update health monitor
                if self.health_monitor:
                    self.health_monitor.update_component_health(
                        component_id="prediction_service",
                        metrics={
                            "max_batch_size": self.max_batch_size,
                            "max_batch_latency_ms": self.max_batch_latency_ms,
                            "max_concurrent_batches": self.max_concurrent_batches,
                            "enable_caching": self.enable_caching,
                            "config_update_time": time.time()
                        }
                    )

                logger.info("Prediction service configuration updated")
                return True

            except Exception as e:
                logger.error(f"Error updating configuration: {str(e)}")
                return False

    def get_configuration(self) -> Dict[str, Any]:
        """
        Get current service configuration.

        Returns:
            Configuration dictionary
        """
        with self._lock:
            return {
                "max_batch_size": self.max_batch_size,
                "max_batch_latency_ms": self.max_batch_latency_ms,
                "prediction_timeout_ms": self.prediction_timeout_ms,
                "max_concurrent_batches": self.max_concurrent_batches,
                "enable_caching": self.enable_caching,
                "cache_ttl_ms": self.cache_ttl_ms,
                "max_cache_size": self.max_cache_size
            }