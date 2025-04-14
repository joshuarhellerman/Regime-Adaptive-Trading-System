import pytest
import time
import threading
import uuid
from unittest.mock import Mock, MagicMock, patch, call

import numpy as np
from models.production.prediction_service import (
    PredictionService,
    PredictionRequest,
    PredictionResult,
    PredictionPriority,
)
from core.health_monitor import HealthMonitor, HealthStatus
from models.production.model_deployer import ModelDeployer, DeploymentType


class TestPredictionService:
    @pytest.fixture
    def model_deployer(self):
        """Create a mock model deployer."""
        mock_deployer = Mock(spec=ModelDeployer)
        
        # Set up model responses
        mock_model = Mock()
        mock_model.predict.return_value = {"score": 0.95}
        mock_model.predict_batch = Mock(return_value=[{"score": 0.95}, {"score": 0.85}])
        
        # Configure get_active_model method
        mock_deployer.get_active_model.return_value = {"model_id": "test-model-123"}
        mock_deployer.load_model.return_value = mock_model
        
        return mock_deployer
    
    @pytest.fixture
    def health_monitor(self):
        """Create a mock health monitor."""
        return Mock(spec=HealthMonitor)
    
    @pytest.fixture
    def event_bus(self):
        """Create a mock event bus."""
        mock_bus = Mock()
        mock_bus.subscribe = Mock()
        return mock_bus
    
    @pytest.fixture
    def prediction_service(self, model_deployer, health_monitor, event_bus):
        """Create a prediction service instance with mocks."""
        service = PredictionService(
            model_deployer=model_deployer,
            health_monitor=health_monitor,
            event_bus=event_bus,
            max_batch_size=10,
            max_batch_latency_ms=50.0,
            prediction_timeout_ms=500.0,
            max_concurrent_batches=4,
            enable_caching=True,
            cache_ttl_ms=1000.0,
            max_cache_size=100,
            preload_models=False
        )
        return service
    
    def test_init(self, prediction_service, event_bus, health_monitor):
        """Test initialization of prediction service."""
        assert prediction_service.model_deployer is not None
        assert prediction_service.health_monitor is not None
        assert prediction_service.event_bus is not None
        
        # Check subscriptions
        event_bus.subscribe.assert_called_once()
        
        # Check health monitor registration
        health_monitor.register_component.assert_called_once_with(
            "prediction_service", "service"
        )
    
    def test_start_stop(self, prediction_service, health_monitor):
        """Test starting and stopping the prediction service."""
        # Test start
        prediction_service.start()
        assert prediction_service._running is True
        assert prediction_service.batch_thread is not None
        assert prediction_service.batch_thread.is_alive()
        
        # Check health status update
        health_monitor.update_component_health.assert_called_once()
        args, kwargs = health_monitor.update_component_health.call_args
        assert kwargs["component_id"] == "prediction_service"
        assert kwargs["status"] == HealthStatus.HEALTHY
        
        # Test stop
        prediction_service.stop()
        assert prediction_service._running is False
        time.sleep(0.1)  # Allow thread to stop
        assert not prediction_service.batch_thread.is_alive()
    
    def test_predict_sync(self, prediction_service, model_deployer):
        """Test synchronous prediction."""
        # Start the service
        prediction_service.start()
        try:
            # Make a prediction
            result = prediction_service.predict(
                model_type="classifier",
                features=[1, 2, 3, 4],
                scope="default",
                priority=PredictionPriority.NORMAL,
                batch=False  # Process immediately
            )
            
            # Verify result
            assert isinstance(result, PredictionResult)
            assert result.model_id == "test-model-123"
            assert result.prediction == {"score": 0.95}
            assert result.error is None
            assert result.latency_ms > 0
            
            # Check model retrieval
            model_deployer.get_active_model.assert_called_once_with("classifier", "default")
            model_deployer.load_model.assert_called_once_with("test-model-123")
        finally:
            prediction_service.stop()
    
    def test_predict_with_error(self, prediction_service, model_deployer):
        """Test prediction with model error."""
        # Configure model to raise an exception
        model_deployer.load_model.return_value.predict.side_effect = ValueError("Model error")
        
        # Start the service
        prediction_service.start()
        try:
            # Make a prediction
            result = prediction_service.predict(
                model_type="classifier",
                features=[1, 2, 3, 4],
                scope="default",
                priority=PredictionPriority.NORMAL,
                batch=False
            )
            
            # Verify result
            assert isinstance(result, PredictionResult)
            assert result.prediction is None
            assert "Model error" in result.error
            assert result.latency_ms > 0
        finally:
            prediction_service.stop()
    
    def test_predict_model_not_found(self, prediction_service, model_deployer):
        """Test prediction when model is not found."""
        # Configure model deployer to return None for active model
        model_deployer.get_active_model.return_value = None
        
        # Start the service
        prediction_service.start()
        try:
            # Make a prediction
            result = prediction_service.predict(
                model_type="unknown_model",
                features=[1, 2, 3, 4],
                scope="default",
                priority=PredictionPriority.NORMAL,
                batch=False
            )
            
            # Verify result
            assert isinstance(result, PredictionResult)
            assert result.prediction is None
            assert "not found" in result.error.lower()
            assert result.model_id == ""
        finally:
            prediction_service.stop()
    
    def test_predict_batch(self, prediction_service, model_deployer):
        """Test batch prediction."""
        # Start the service
        prediction_service.start()
        try:
            # Make a batch prediction
            results = prediction_service.predict_batch(
                model_type="classifier",
                features_batch=[[1, 2, 3], [4, 5, 6]],
                scope="default",
                priority=PredictionPriority.NORMAL,
            )
            
            # Verify results
            assert len(results) == 2
            for result in results:
                assert isinstance(result, PredictionResult)
                assert result.model_id == "test-model-123"
                assert result.prediction in [{"score": 0.95}, {"score": 0.85}]
                assert result.error is None
                assert result.latency_ms > 0
                
            # Verify the batch method was called
            model = model_deployer.load_model.return_value
            model.predict_batch.assert_called_once_with([[1, 2, 3], [4, 5, 6]])
        finally:
            prediction_service.stop()
    
    def test_predict_fallback_to_individual(self, prediction_service, model_deployer):
        """Test batch prediction fallback to individual predictions."""
        # Remove the predict_batch method to force individual predictions
        model = model_deployer.load_model.return_value
        del model.predict_batch
        
        # Start the service
        prediction_service.start()
        try:
            # Make a batch prediction
            results = prediction_service.predict_batch(
                model_type="classifier",
                features_batch=[[1, 2, 3], [4, 5, 6]],
                scope="default",
                priority=PredictionPriority.NORMAL,
            )
            
            # Verify results
            assert len(results) == 2
            
            # Verify individual predict calls
            assert model.predict.call_count == 2
            model.predict.assert_has_calls([
                call([1, 2, 3]),
                call([4, 5, 6])
            ])
        finally:
            prediction_service.stop()
    
    @pytest.mark.asyncio
    async def test_predict_async(self, prediction_service):
        """Test asynchronous prediction."""
        # Start the service
        prediction_service.start()
        try:
            # Make an async prediction
            result = await prediction_service.predict_async(
                model_type="classifier",
                features=[1, 2, 3, 4],
                scope="default",
                priority=PredictionPriority.NORMAL,
                batch=False
            )
            
            # Verify result
            assert isinstance(result, PredictionResult)
            assert result.model_id == "test-model-123"
            assert result.prediction == {"score": 0.95}
            assert result.error is None
            assert result.latency_ms > 0
        finally:
            prediction_service.stop()
    
    def test_cache_hit(self, prediction_service):
        """Test prediction cache hit."""
        # Start the service
        prediction_service.start()
        try:
            # Make first prediction to populate cache
            prediction_service.predict(
                model_type="classifier",
                features=[1, 2, 3, 4],
                scope="default",
                batch=False
            )
            
            # Clear mock to verify it's not called again
            model_deployer = prediction_service.model_deployer
            model_deployer.load_model.return_value.predict.reset_mock()
            
            # Make second prediction with same features
            result = prediction_service.predict(
                model_type="classifier",
                features=[1, 2, 3, 4],
                scope="default",
                batch=False
            )
            
            # Verify cache was used
            assert result.model_id == "cached"
            assert "cached" in result.metadata
            assert result.metadata["cached"] is True
            
            # Verify model wasn't called again
            model = model_deployer.load_model.return_value
            model.predict.assert_not_called()
            
            # Check metrics
            metrics = prediction_service.get_metrics()
            assert metrics["cache_hits"] > 0
        finally:
            prediction_service.stop()
    
    def test_cache_expiration(self, prediction_service):
        """Test cache expiration."""
        # Set short cache TTL
        prediction_service.cache_ttl_ms = 50.0
        
        # Start the service
        prediction_service.start()
        try:
            # Make first prediction to populate cache
            prediction_service.predict(
                model_type="classifier",
                features=[1, 2, 3, 4],
                scope="default",
                batch=False
            )
            
            # Clear mock to verify it's called again after expiration
            model = prediction_service.model_deployer.load_model.return_value
            model.predict.reset_mock()
            
            # Wait for cache to expire
            time.sleep(0.1)  # 100ms > 50ms TTL
            
            # Make second prediction with same features
            result = prediction_service.predict(
                model_type="classifier",
                features=[1, 2, 3, 4],
                scope="default",
                batch=False
            )
            
            # Verify model was called again
            model.predict.assert_called_once()
            assert result.model_id == "test-model-123"  # Not from cache
        finally:
            prediction_service.stop()
    
    def test_clear_cache(self, prediction_service):
        """Test clearing the prediction cache."""
        # Start the service
        prediction_service.start()
        try:
            # Make first prediction to populate cache
            prediction_service.predict(
                model_type="classifier",
                features=[1, 2, 3, 4],
                scope="default",
                batch=False
            )
            
            # Verify cache has entries
            assert len(prediction_service.prediction_cache) > 0
            
            # Clear cache
            prediction_service.clear_cache()
            
            # Verify cache is empty
            assert len(prediction_service.prediction_cache) == 0
        finally:
            prediction_service.stop()
    
    def test_batch_formation(self, prediction_service):
        """Test batch formation logic."""
        # Patch the _process_batch method to capture batch
        with patch.object(prediction_service, '_process_batch') as mock_process_batch:
            # Start the service with a longer batch latency to ensure controlled timing
            prediction_service.max_batch_latency_ms = 100.0
            prediction_service.max_batch_size = 2
            prediction_service.start()
            
            try:
                # Queue two requests
                prediction_service._enqueue_request(PredictionRequest(
                    id="req1",
                    model_type="classifier",
                    scope="default",
                    features=[1, 2, 3],
                    priority=PredictionPriority.NORMAL
                ))
                
                prediction_service._enqueue_request(PredictionRequest(
                    id="req2",
                    model_type="classifier",
                    scope="default",
                    features=[4, 5, 6],
                    priority=PredictionPriority.NORMAL
                ))
                
                # Give time for batch formation
                time.sleep(0.2)  # 200ms > 100ms batch latency
                
                # Verify batch was formed and processed
                mock_process_batch.assert_called()
                batch_key, batch = mock_process_batch.call_args[0]
                
                assert batch_key == "classifier_default"
                assert len(batch) == 2
                assert batch[0].id == "req1"
                assert batch[1].id == "req2"
            finally:
                prediction_service.stop()
    
    def test_priority_order(self, prediction_service):
        """Test requests are processed in priority order."""
        # Configure batch formation to process one request at a time
        prediction_service.max_batch_size = 1
        
        # Track processed requests
        processed_requests = []
        
        # Override _process_single_request to track order
        original_process = prediction_service._process_single_request
        
        def track_process(request):
            processed_requests.append(request.id)
            return original_process(request)
        
        prediction_service._process_single_request = track_process
        
        # Start the service
        prediction_service.start()
        try:
            # Queue requests with different priorities
            request_low = PredictionRequest(
                id="low",
                model_type="classifier",
                scope="default",
                features=[1, 2, 3],
                priority=PredictionPriority.LOW
            )
            
            request_high = PredictionRequest(
                id="high",
                model_type="classifier",
                scope="default",
                features=[4, 5, 6],
                priority=PredictionPriority.HIGH
            )
            
            request_normal = PredictionRequest(
                id="normal",
                model_type="classifier",
                scope="default",
                features=[7, 8, 9],
                priority=PredictionPriority.NORMAL
            )
            
            # Queue in reverse priority order
            prediction_service._enqueue_request(request_low)
            prediction_service._enqueue_request(request_normal)
            prediction_service._enqueue_request(request_high)
            
            # Ensure requests are processed
            time.sleep(0.3)
            
            # Verify order: high priority first, then normal, then low
            assert len(processed_requests) == 3
            assert processed_requests[0] == "high"
            assert processed_requests[1] == "normal"
            assert processed_requests[2] == "low"
        finally:
            prediction_service.stop()
    
    def test_get_metrics(self, prediction_service):
        """Test metrics collection and retrieval."""
        # Start the service
        prediction_service.start()
        try:
            # Make a prediction to generate metrics
            prediction_service.predict(
                model_type="classifier",
                features=[1, 2, 3, 4],
                scope="default",
                batch=False
            )
            
            # Get metrics
            metrics = prediction_service.get_metrics()
            
            # Verify metrics structure
            assert "requests_total" in metrics
            assert "requests_success" in metrics
            assert "requests_error" in metrics
            assert "avg_latency_ms" in metrics
            assert "cache_hits" in metrics
            assert "cache_misses" in metrics
            assert "queue_sizes" in metrics
            assert "active_batches" in metrics
            assert "cache_size" in metrics
            assert "latency_histogram" in metrics
            
            # Verify values
            assert metrics["requests_total"] == 1
            assert metrics["requests_success"] == 1
            assert metrics["requests_error"] == 0
            assert metrics["avg_latency_ms"] > 0
        finally:
            prediction_service.stop()
    
    def test_set_configuration(self, prediction_service, health_monitor):
        """Test updating service configuration."""
        # Start the service
        prediction_service.start()
        try:
            # Update configuration
            success = prediction_service.set_configuration({
                "max_batch_size": 20,
                "max_batch_latency_ms": 100.0,
                "enable_caching": False
            })
            
            # Verify update was successful
            assert success is True
            assert prediction_service.max_batch_size == 20
            assert prediction_service.max_batch_latency_ms == 100.0
            assert prediction_service.enable_caching is False
            
            # Verify health monitor was updated
            health_monitor.update_component_health.assert_called_with(
                component_id="prediction_service",
                metrics={
                    "max_batch_size": 20,
                    "max_batch_latency_ms": 100.0,
                    "max_concurrent_batches": 4,
                    "enable_caching": False,
                    "config_update_time": pytest.approx(time.time(), abs=1)
                }
            )
            
            # Get configuration
            config = prediction_service.get_configuration()
            
            # Verify configuration
            assert config["max_batch_size"] == 20
            assert config["max_batch_latency_ms"] == 100.0
            assert config["enable_caching"] is False
        finally:
            prediction_service.stop()
    
    def test_model_deployed_event(self, prediction_service, event_bus):
        """Test handling model deployment events."""
        # Get event handler
        event_handler = None
        for args, kwargs in event_bus.subscribe.call_args_list:
            topic, handler = args
            if topic == "MODEL_DEPLOYED":
                event_handler = handler
                break
        
        assert event_handler is not None
        
        # Start the service and populate cache
        prediction_service.start()
        try:
            # Add a model to the cache
            cache_key = "classifier_default"
            mock_model = Mock()
            prediction_service.model_cache[cache_key] = (mock_model, "old-model-123")
            
            # Add a prediction to the cache
            cache_pred_key = "classifier_default_1_2_3_4"
            prediction_service.prediction_cache[cache_pred_key] = ({"score": 0.9}, time.time())
            
            # Create deployment event
            event = Mock()
            event.data = {
                "model_id": "new-model-456",
                "model_type": "classifier",
                "scope": "default"
            }
            
            # Handle event
            event_handler(event)
            
            # Verify model cache was invalidated
            assert cache_key not in prediction_service.model_cache
            
            # Verify prediction cache was invalidated
            assert cache_pred_key not in prediction_service.prediction_cache
            
            # Verify new model was loaded
            model_deployer = prediction_service.model_deployer
            model_deployer.load_model.assert_called_with("new-model-456")
        finally:
            prediction_service.stop()
    
    def test_preload_models(self, model_deployer):
        """Test preloading models on initialization."""
        # Configure model deployer
        model_types = [DeploymentType.CLASSIFIER, DeploymentType.REGRESSOR]
        model_ids = {
            DeploymentType.CLASSIFIER.value: "classifier-model-123",
            DeploymentType.REGRESSOR.value: "regressor-model-456"
        }
        
        def get_active_model_side_effect(model_type, scope="default"):
            if model_type in [t.value for t in model_types]:
                return {"model_id": model_ids[model_type]}
            return None
        
        model_deployer.get_active_model.side_effect = get_active_model_side_effect
        
        # Create service with preloading
        service = PredictionService(
            model_deployer=model_deployer,
            preload_models=True
        )
        
        # Verify models were loaded
        assert model_deployer.load_model.call_count == 2
        model_deployer.load_model.assert_any_call("classifier-model-123")
        model_deployer.load_model.assert_any_call("regressor-model-456")
        
        # Verify models are in cache
        assert "CLASSIFIER_default" in service.model_cache
        assert "REGRESSOR_default" in service.model_cache
    
    def test_concurrent_predictions(self, prediction_service):
        """Test concurrent predictions with threading."""
        # Start the service
        prediction_service.start()
        try:
            # Number of concurrent predictions
            n_threads = 10
            results = [None] * n_threads
            
            # Thread function
            def predict_thread(idx):
                result = prediction_service.predict(
                    model_type="classifier",
                    features=[idx, idx+1, idx+2],
                    scope="default",
                    batch=False
                )
                results[idx] = result
            
            # Start threads
            threads = []
            for i in range(n_threads):
                thread = threading.Thread(target=predict_thread, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify all predictions completed successfully
            for i, result in enumerate(results):
                assert result is not None
                assert result.prediction == {"score": 0.95}
                assert result.error is None
        finally:
            prediction_service.stop()

    def test_numpy_array_features(self, prediction_service):
        """Test prediction with numpy array features."""
        # Start the service
        prediction_service.start()
        try:
            # Make a prediction with numpy array
            features = np.array([1.0, 2.0, 3.0, 4.0])
            result = prediction_service.predict(
                model_type="classifier",
                features=features,
                scope="default",
                batch=False
            )
            
            # Verify result
            assert result.prediction == {"score": 0.95}
            assert result.error is None
            
            # Check cache key generation worked with numpy array
            cache_key = prediction_service._make_cache_key("classifier", "default", features)
            assert cache_key is not None
            assert len(cache_key) > 0
        finally:
            prediction_service.stop()
    
    def test_max_latency_requirement(self, prediction_service):
        """Test handling requests with max latency requirements."""
        # Patch _process_single_request to verify it's called directly
        with patch.object(prediction_service, '_process_single_request') as mock_process:
            mock_process.return_value = MagicMock(spec=PredictionResult)
            
            # Start the service
            prediction_service.start()
            try:
                # Make a prediction with strict latency requirement
                prediction_service.predict(
                    model_type="classifier",
                    features=[1, 2, 3, 4],
                    max_latency_ms=10.0,  # Less than batch latency
                    batch=True  # Would normally batch, but max_latency should override
                )
                
                # Verify single request processing was used
                mock_process.assert_called_once()
            finally:
                prediction_service.stop()