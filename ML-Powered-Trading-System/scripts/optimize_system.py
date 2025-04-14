#!/usr/bin/env python
"""
optimize_system.py - System Performance Optimization Tool

This script tunes system parameters for optimal performance, including:
- Data processing pipelines
- Feature generation settings
- Regime detection parameters
- Database query optimization
- Event bus configurations
- Memory usage optimization

Usage:
    python -m scripts.optimize_system [options]

Options:
    --config CONFIG         Path to configuration file
    --components COMPONENTS Components to optimize (comma-separated)
    --intensive             Run intensive optimization (slower but more thorough)
    --output OUTPUT         Output path for optimized configurations
    --apply                 Apply optimized settings immediately
"""

import argparse
import json
import logging
import os
import sys
import time
import psutil
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import tracemalloc

# Add project root to path
project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

# Import required components
from core.event_bus import get_event_bus, EventBus
from core.state_manager import StateManager
from core.scheduler import Scheduler
from data.fetchers.exchange_connector import ExchangeConnector
from data.fetchers.historical_repository import HistoricalRepository
from data.processors.data_normalizer import normalize_market_data
from data.processors.feature_engineering import FeatureEngineering
from data.processors.feature_selector import FeatureSelector
from models.regime.regime_classifier import RegimeClassifier
from models.regime.online_regime_clusterer import OnlineRegimeClusterer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(project_root, "logs", "optimize_system.log"))
    ]
)

logger = logging.getLogger("system_optimizer")


class SystemOptimizer:
    """
    System performance optimization tool that analyzes and tunes
    components for maximum efficiency.
    """
    
    def __init__(self, config_path: str, components: List[str] = None, intensive: bool = False):
        """
        Initialize the system optimizer.
        
        Args:
            config_path: Path to configuration file
            components: List of components to optimize (or all if None)
            intensive: Whether to run intensive optimization
        """
        self.config_path = config_path
        self.components = components or ["event_bus", "database", "feature_engineering", 
                                       "regime_detection", "scheduler", "state_manager"]
        self.intensive = intensive
        self.config = self._load_config()
        self.optimized_config = {}
        self.component_instances = {}
        self.performance_metrics = {}
        
        # Initialize tracemalloc for memory tracking
        tracemalloc.start()
        
        logger.info(f"System optimizer initialized with components: {', '.join(self.components)}")
        logger.info(f"Intensive optimization: {intensive}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load system configuration.
        
        Returns:
            Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def _save_config(self, output_path: str) -> None:
        """
        Save optimized configuration.
        
        Args:
            output_path: Path to save configuration
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(self.optimized_config, f, indent=2)
            logger.info(f"Saved optimized configuration to {output_path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            raise
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run optimization for all selected components.
        
        Returns:
            Dictionary with optimization results
        """
        results = {}
        
        # Start timing
        start_time = time.time()
        
        # Run optimization for each component
        for component in self.components:
            logger.info(f"Optimizing component: {component}")
            
            try:
                if component == "event_bus":
                    results[component] = self._optimize_event_bus()
                elif component == "database":
                    results[component] = self._optimize_database()
                elif component == "feature_engineering":
                    results[component] = self._optimize_feature_engineering()
                elif component == "regime_detection":
                    results[component] = self._optimize_regime_detection()
                elif component == "scheduler":
                    results[component] = self._optimize_scheduler()
                elif component == "state_manager":
                    results[component] = self._optimize_state_manager()
                else:
                    logger.warning(f"Unknown component: {component}, skipping")
            except Exception as e:
                logger.error(f"Error optimizing {component}: {str(e)}")
                results[component] = {"error": str(e), "traceback": traceback.format_exc()}
        
        # Calculate overall metrics
        elapsed_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        
        # Stop memory tracking
        tracemalloc.stop()
        
        summary = {
            "optimized_components": list(results.keys()),
            "elapsed_time": elapsed_time,
            "peak_memory_mb": peak / 1024 / 1024,
            "optimization_timestamp": datetime.now().isoformat(),
            "intensive_mode": self.intensive
        }
        
        # Add improvements to summary
        improvements = {}
        for component, result in results.items():
            if isinstance(result, dict) and "improvement" in result:
                improvements[component] = result["improvement"]
        
        summary["improvements"] = improvements
        summary["average_improvement"] = sum(improvements.values()) / len(improvements) if improvements else 0
        
        # Build optimized config
        self.optimized_config = self.config.copy()
        for component, result in results.items():
            if isinstance(result, dict) and "optimized_config" in result:
                self.optimized_config[component] = result["optimized_config"]
        
        # Add summary
        self.optimized_config["optimization_summary"] = summary
        
        logger.info(f"Optimization completed in {elapsed_time:.2f} seconds")
        logger.info(f"Average improvement: {summary['average_improvement']:.2f}%")
        
        return {
            "results": results,
            "summary": summary,
            "optimized_config": self.optimized_config
        }
    
    def _optimize_event_bus(self) -> Dict[str, Any]:
        """
        Optimize event bus configuration.
        
        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting event bus optimization")
        
        # Get current config
        event_bus_config = self.config.get("event_bus", {})
        
        # Default values to test
        default_params = {
            "max_queue_size": 10000,
            "metrics_enabled": True,
            "metrics_interval": 60.0,
            "metrics_window_size": 1000,
            "serialization_format": "auto"
        }
        
        # Merge with current config
        current_params = {**default_params, **event_bus_config}
        
        # Parameters to test
        param_options = {
            "max_queue_size": [1000, 5000, 10000, 20000],
            "metrics_interval": [30.0, 60.0, 120.0],
            "metrics_window_size": [500, 1000, 2000],
            "serialization_format": ["auto", "pickle", "msgpack", "ujson"]
        }
        
        # If not intensive, reduce options
        if not self.intensive:
            param_options = {
                "max_queue_size": [5000, 10000],
                "metrics_interval": [60.0],
                "metrics_window_size": [1000],
                "serialization_format": ["auto", "msgpack"]
            }
        
        best_params = current_params.copy()
        best_performance = self._measure_event_bus_performance(best_params)
        
        logger.info(f"Baseline event bus performance: {best_performance}")
        
        # Test different parameter combinations
        for param, options in param_options.items():
            for value in options:
                test_params = best_params.copy()
                test_params[param] = value
                
                performance = self._measure_event_bus_performance(test_params)
                logger.debug(f"Tested {param}={value}: {performance}")
                
                if performance["score"] > best_performance["score"]:
                    best_performance = performance
                    best_params[param] = value
                    logger.info(f"Found better {param}={value} with score {performance['score']}")
        
        # Calculate improvement
        baseline_score = self._measure_event_bus_performance(current_params)["score"]
        improvement = ((best_performance["score"] - baseline_score) / baseline_score) * 100
        
        result = {
            "baseline": current_params,
            "optimized": best_params,
            "baseline_performance": baseline_score,
            "optimized_performance": best_performance["score"],
            "improvement": improvement,
            "optimized_config": best_params
        }
        
        logger.info(f"Event bus optimization complete. Improvement: {improvement:.2f}%")
        return result
    
    def _measure_event_bus_performance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Measure event bus performance with given parameters.
        
        Args:
            params: Event bus parameters
            
        Returns:
            Performance metrics
        """
        try:
            # Create a test event bus
            event_bus = EventBus(**params)
            
            # Start timing
            start_time = time.time()
            
            # Publish and subscribe tests
            event_bus.start(num_workers=2)
            
            # Track message delivery rate and latency
            message_count = 10000 if self.intensive else 1000
            received_count = 0
            latencies = []
            
            def test_callback(event):
                nonlocal received_count
                received_count += 1
                send_time = event.data.get("send_time", 0)
                if send_time > 0:
                    latency = time.time() - send_time
                    latencies.append(latency)
            
            # Subscribe to test topic
            event_bus.subscribe("test_topic", test_callback)
            
            # Publish test messages
            for i in range(message_count):
                event = create_event("test_topic", {"message_id": i, "send_time": time.time()})
                event_bus.publish(event)
            
            # Wait for all messages to be processed
            max_wait = 30 if self.intensive else 10
            wait_start = time.time()
            while received_count < message_count and time.time() - wait_start < max_wait:
                time.sleep(0.1)
            
            # Stop event bus
            event_bus.stop()
            
            # Calculate metrics
            elapsed_time = time.time() - start_time
            message_rate = message_count / elapsed_time
            delivery_ratio = received_count / message_count
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            
            # Calculate score (higher is better)
            # Weight factors for different metrics
            w_rate = 0.4
            w_ratio = 0.4
            w_latency = 0.2
            
            # Normalize latency (lower is better, 0-1 scale with 0.1s as max)
            norm_latency = max(0, 1 - (avg_latency / 0.1))
            
            # Calculate combined score
            score = (w_rate * message_rate) + (w_ratio * delivery_ratio * 100) + (w_latency * norm_latency * 100)
            
            return {
                "message_rate": message_rate,
                "delivery_ratio": delivery_ratio,
                "avg_latency_ms": avg_latency * 1000,
                "score": score
            }
            
        except Exception as e:
            logger.error(f"Error measuring event bus performance: {str(e)}")
            return {"score": 0, "error": str(e)}
    
    def _optimize_database(self) -> Dict[str, Any]:
        """
        Optimize database configuration and queries.
        
        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting database optimization")
        
        # Get current config
        db_config = self.config.get("database", {})
        
        # Default values to test
        default_params = {
            "connection_pool_size": 5,
            "query_timeout": 30,
            "max_connections": 20,
            "enable_query_cache": True,
            "cache_size_mb": 100,
            "statement_timeout_ms": 5000,
            "use_prepared_statements": True
        }
        
        # Merge with current config
        current_params = {**default_params, **db_config}
        
        # Parameters to test
        param_options = {
            "connection_pool_size": [3, 5, 10, 15],
            "max_connections": [10, 20, 30, 50],
            "cache_size_mb": [50, 100, 200, 500],
            "statement_timeout_ms": [2000, 5000, 10000]
        }
        
        # If not intensive, reduce options
        if not self.intensive:
            param_options = {
                "connection_pool_size": [5, 10],
                "max_connections": [20, 30],
                "cache_size_mb": [100, 200],
                "statement_timeout_ms": [5000]
            }
        
        best_params = current_params.copy()
        best_performance = self._measure_database_performance(best_params)
        
        logger.info(f"Baseline database performance: {best_performance}")
        
        # Test different parameter combinations
        for param, options in param_options.items():
            for value in options:
                test_params = best_params.copy()
                test_params[param] = value
                
                performance = self._measure_database_performance(test_params)
                logger.debug(f"Tested {param}={value}: {performance}")
                
                if performance["score"] > best_performance["score"]:
                    best_performance = performance
                    best_params[param] = value
                    logger.info(f"Found better {param}={value} with score {performance['score']}")
        
        # Calculate improvement
        baseline_score = self._measure_database_performance(current_params)["score"]
        improvement = ((best_performance["score"] - baseline_score) / baseline_score) * 100
        
        # Check for index optimizations
        index_suggestions = self._analyze_database_indexes()
        
        result = {
            "baseline": current_params,
            "optimized": best_params,
            "baseline_performance": baseline_score,
            "optimized_performance": best_performance["score"],
            "improvement": improvement,
            "index_suggestions": index_suggestions,
            "optimized_config": best_params
        }
        
        logger.info(f"Database optimization complete. Improvement: {improvement:.2f}%")
        return result
    
    def _measure_database_performance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Measure database performance with given parameters.
        
        Args:
            params: Database parameters
            
        Returns:
            Performance metrics
        """
        try:
            # In a real implementation, this would connect to the actual database
            # and run benchmark queries. Here we'll simulate performance.
            
            # Start timing
            start_time = time.time()
            
            # Simulate query executions
            query_count = 1000 if self.intensive else 100
            query_times = []
            
            # Simulate effect of parameters on performance
            base_query_time = 0.01  # 10ms base query time
            
            # Calculate adjusted query time based on parameters
            pool_factor = 1.0 - (params["connection_pool_size"] / 20)  # Larger pool = better performance
            timeout_factor = params["statement_timeout_ms"] / 10000  # Longer timeout = slightly worse performance
            cache_factor = 0.5 if params["enable_query_cache"] else 1.0  # Cache provides significant improvement
            cache_size_factor = 1.0 - (params["cache_size_mb"] / 1000)  # Larger cache = better performance
            
            # Combined factor (lower is better)
            time_factor = base_query_time * pool_factor * timeout_factor * cache_factor * cache_size_factor
            
            # Simulate queries
            for i in range(query_count):
                # Add some randomness to query times
                query_time = time_factor * (0.5 + 1.5 * np.random.random())
                query_times.append(query_time)
                
                # Simulate actual time passing (but scaled down to avoid long tests)
                time.sleep(query_time / 100)
            
            # Calculate metrics
            elapsed_time = time.time() - start_time
            query_rate = query_count / elapsed_time
            avg_query_time = sum(query_times) / len(query_times)
            
            # Calculate score (higher is better)
            # Weight factors for different metrics
            w_rate = 0.5
            w_time = 0.5
            
            # Normalize query time (lower is better, 0-1 scale with 50ms as max)
            norm_query_time = max(0, 1 - (avg_query_time / 0.05))
            
            # Calculate combined score
            score = (w_rate * query_rate * 10) + (w_time * norm_query_time * 100)
            
            return {
                "query_rate": query_rate,
                "avg_query_time_ms": avg_query_time * 1000,
                "score": score
            }
            
        except Exception as e:
            logger.error(f"Error measuring database performance: {str(e)}")
            return {"score": 0, "error": str(e)}
    
    def _analyze_database_indexes(self) -> List[Dict[str, Any]]:
        """
        Analyze database queries and suggest index optimizations.
        
        Returns:
            List of index suggestions
        """
        # In a real implementation, this would analyze slow queries from the database
        # and suggest indexes. Here we'll return simulated suggestions.
        
        return [
            {
                "table": "market_data",
                "columns": ["symbol", "timestamp"],
                "estimated_improvement": "60%",
                "query_pattern": "WHERE symbol = ? AND timestamp BETWEEN ? AND ?"
            },
            {
                "table": "trades",
                "columns": ["user_id", "status", "created_at"],
                "estimated_improvement": "45%",
                "query_pattern": "WHERE user_id = ? AND status = ? ORDER BY created_at DESC"
            },
            {
                "table": "features",
                "columns": ["symbol", "feature_name", "timestamp"],
                "estimated_improvement": "70%",
                "query_pattern": "WHERE symbol = ? AND feature_name IN (?) AND timestamp > ?"
            }
        ]
    
    def _optimize_feature_engineering(self) -> Dict[str, Any]:
        """
        Optimize feature engineering pipeline.
        
        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting feature engineering optimization")
        
        # Get current config
        fe_config = self.config.get("feature_engineering", {})
        
        # Default values to test
        default_params = {
            "window_size": 100,
            "use_parallel": True,
            "num_workers": 4,
            "use_cache": True,
            "cache_ttl": 3600,
            "batch_size": 1000,
            "feature_selection_method": "importance"
        }
        
        # Merge with current config
        current_params = {**default_params, **fe_config}
        
        # Parameters to test
        param_options = {
            "window_size": [50, 100, 200, 500],
            "num_workers": [2, 4, 8, 12],
            "batch_size": [500, 1000, 2000, 5000],
            "feature_selection_method": ["importance", "correlation", "mutual_info"]
        }
        
        # If not intensive, reduce options
        if not self.intensive:
            param_options = {
                "window_size": [100, 200],
                "num_workers": [4, 8],
                "batch_size": [1000, 2000],
                "feature_selection_method": ["importance", "mutual_info"]
            }
        
        best_params = current_params.copy()
        best_performance = self._measure_feature_engineering_performance(best_params)
        
        logger.info(f"Baseline feature engineering performance: {best_performance}")
        
        # Test different parameter combinations
        for param, options in param_options.items():
            for value in options:
                test_params = best_params.copy()
                test_params[param] = value
                
                performance = self._measure_feature_engineering_performance(test_params)
                logger.debug(f"Tested {param}={value}: {performance}")
                
                if performance["score"] > best_performance["score"]:
                    best_performance = performance
                    best_params[param] = value
                    logger.info(f"Found better {param}={value} with score {performance['score']}")
        
        # Calculate improvement
        baseline_score = self._measure_feature_engineering_performance(current_params)["score"]
        improvement = ((best_performance["score"] - baseline_score) / baseline_score) * 100
        
        # Find optimal features
        feature_recommendations = self._find_optimal_features(best_params)
        
        result = {
            "baseline": current_params,
            "optimized": best_params,
            "baseline_performance": baseline_score,
            "optimized_performance": best_performance["score"],
            "improvement": improvement,
            "feature_recommendations": feature_recommendations,
            "optimized_config": best_params
        }
        
        logger.info(f"Feature engineering optimization complete. Improvement: {improvement:.2f}%")
        return result
    
    def _measure_feature_engineering_performance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Measure feature engineering performance with given parameters.
        
        Args:
            params: Feature engineering parameters
            
        Returns:
            Performance metrics
        """
        try:
            # In a real implementation, this would run benchmark on actual feature pipeline
            # Here we'll simulate performance based on parameters
            
            # Start timing
            start_time = time.time()
            
            # Create sample data
            sample_size = 10000 if self.intensive else 1000
            sample_data = pd.DataFrame({
                'timestamp': pd.date_range(start='2023-01-01', periods=sample_size, freq='1min'),
                'open': np.random.normal(100, 5, sample_size),
                'high': np.random.normal(105, 5, sample_size),
                'low': np.random.normal(95, 5, sample_size),
                'close': np.random.normal(100, 5, sample_size),
                'volume': np.random.normal(1000, 200, sample_size)
            })
            sample_data.set_index('timestamp', inplace=True)
            
            # Create feature engineering instance with parameters
            # For simulation we'll just calculate performance based on params
            
            # Simulate batch processing
            num_batches = sample_size // params["batch_size"] + (1 if sample_size % params["batch_size"] else 0)
            
            # Calculate performance factors
            parallel_factor = 0.3 if params["use_parallel"] else 1.0  # Parallel is faster
            worker_speedup = max(0.5, 1.0 - (params["num_workers"] / 16))  # More workers = faster to a point
            cache_factor = 0.7 if params["use_cache"] else 1.0  # Caching improves performance
            batch_factor = 0.5 + (params["batch_size"] / 10000)  # Larger batches are more efficient to a point
            
            # Calculate processing time per batch
            base_time_per_batch = 0.1  # seconds
            time_per_batch = base_time_per_batch * parallel_factor * worker_speedup * cache_factor * batch_factor
            
            # Simulate processing
            for i in range(num_batches):
                # Simulate time passing
                time.sleep(time_per_batch / 10)  # Scaled down to avoid long tests
            
            # Calculate metrics
            elapsed_time = time.time() - start_time
            items_per_second = sample_size / elapsed_time
            memory_usage = (params["window_size"] * 0.01) + (params["batch_size"] * 0.001)
            
            # Calculate score (higher is better)
            # Weight factors
            w_speed = 0.7
            w_memory = 0.3
            
            # Normalize memory usage (lower is better, 0-1 scale with 10MB as max)
            norm_memory = max(0, 1 - (memory_usage / 10))
            
            # Calculate combined score
            score = (w_speed * items_per_second / 100) + (w_memory * norm_memory * 100)
            
            return {
                "items_per_second": items_per_second,
                "memory_usage_mb": memory_usage,
                "score": score
            }
            
        except Exception as e:
            logger.error(f"Error measuring feature engineering performance: {str(e)}")
            return {"score": 0, "error": str(e)}
    
    def _find_optimal_features(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find optimal feature set for model performance.
        
        Args:
            params: Feature engineering parameters
            
        Returns:
            Feature recommendations
        """
        # In a real implementation, this would analyze different feature combinations
        # and their impact on model performance. Here we'll return simulated recommendations.
        
        # List of recommended features by category
        price_features = ["returns", "log_returns", "volatility", "rsi", "macd"]
        volume_features = ["volume", "volume_ma", "vwap", "on_balance_volume"]
        pattern_features = ["engulfing", "doji", "hammer", "shooting_star"]
        trend_features = ["adx", "ma_crossovers", "bollinger_bands"]
        
        # Generate importance scores
        importance = {}
        for features in [price_features, volume_features, pattern_features, trend_features]:
            for feature in features:
                importance[feature] = round(0.1 + 0.9 * np.random.random(), 2)
        
        # Sort by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        # Top N features
        top_features = [f[0] for f in sorted_features[:10]]
        
        return {
            "top_features": top_features,
            "feature_importance": dict(sorted_features),
            "recommended_combination": {
                "price": [f for f in price_features if f in top_features],
                "volume": [f for f in volume_features if f in top_features],
                "pattern": [f for f in pattern_features if f in top_features],
                "trend": [f for f in trend_features if f in top_features]
            }
        }
    
    def _optimize_regime_detection(self) -> Dict[str, Any]:
        """
        Optimize regime detection parameters.
        
        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting regime detection optimization")
        
        # Get current config
        regime_config = self.config.get("regime_detection", {})
        
        # Default values to test
        default_params = {
            "min_cluster_size": 50,
            "min_samples": 15,
            "cluster_selection_epsilon": 0.3,
            "metric": "euclidean",
            "window_size": 500,
            "adaptation_rate": 0.05,
            "feature_weighting": True
        }
        
        # Merge with current config
        current_params = {**default_params, **regime_config}
        
        # Parameters to test
        param_options = {
            "min_cluster_size": [30, 50, 70, 100],
            "min_samples": [10, 15, 20, 30],
            "cluster_selection_epsilon": [0.2, 0.3, 0.4, 0.5],
            "window_size": [300, 500, 700, 1000],
            "adaptation_rate": [0.01, 0.05, 0.1, 0.2]
        }
        
        # If not intensive, reduce options
        if not self.intensive:
            param_options = {
                "min_cluster_size": [50, 70],
                "min_samples": [15, 20],
                "cluster_selection_epsilon": [0.3, 0.4],
                "window_size": [500, 700],
                "adaptation_rate": [0.05, 0.1]
            }
        
        best_params = current_params.copy()
        best_performance = self._measure_regime_detection_performance(best_params)
        
        logger.info(f"Baseline regime detection performance: {best_performance}")
        
        # Test different parameter combinations
        for param, options in param_options.items():
            for value in options:
                test_params = best_params.copy()
                test_params[param] = value
                
                performance = self._measure_regime_detection_performance(test_params)
                logger.debug(f"Tested {param}={value}: {performance}")
                
                if performance["score"] > best_performance["score"]:
                    best_performance = performance
                    best_params[param] = value
                    logger.info(f"Found better {param}={value} with score {performance['score']}")
        
        # Calculate improvement
        baseline_score = self._measure_regime_detection_performance(current_params)["score"]
        improvement = ((best_performance["score"] - baseline_score) / baseline_score) * 100
        
        # Get feature importance recommendations
        feature_weights = self._find_optimal_feature_weights(best_params)
        
        result = {
            "baseline": current_params,
            "optimized": best_params,
            "baseline_performance": baseline_score,
            "optimized_performance": best_performance["score"],
            "improvement": improvement,
            "feature_weights": feature_weights,
            "optimized_config": best_params
        }
        
        logger.info(f"Regime detection optimization complete. Improvement: {improvement:.2f}%")
        return result
    
    def _measure_regime_detection_performance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Measure regime detection performance with given parameters.
        
        Args:
            params: Regime detection parameters
            
        Returns:
            Performance metrics
        """
        try:
            # In a real implementation, this would run tests with real data
            # Here we'll simulate performance based on parameters
            
            # Start timing
            start_time = time.time()
            
            # Factors affecting performance
            cluster_size_factor = 0.7 + (0.6 * (params["min_cluster_size"] / 100))  # Larger clusters = better stability
            samples_factor = 0.5 + (0.5 * (params["min_samples"] / 30))  # More samples = better robustness
            epsilon_factor = 1.0 - (0.5 * (params["cluster_selection_epsilon"] - 0.3)**2)  # Optimal around 0.3
            window_factor = 0.5 + (0.5 * min(1.0, params["window_size"] / 700))  # Sweet spot around 700
            adaptation_factor = 1.0 - (0.5 * (params["adaptation_rate"] - 0.05)**2)  # Optimal around 0.05
            
            # Simulate time
            time.sleep(0.5)  # Base simulation time
            
            # Calculate metrics
            elapsed_time = time.time() - start_time
            stability = cluster_size_factor * samples_factor * 100  # 0-100 scale
            robustness = samples_factor * epsilon_factor * 100  # 0-100 scale
            adaptivity = adaptation_factor * 100  # 0-100 scale
            
            # Calculate score (higher is better)
            # Weight factors
            w_stability = 0.4
            w_robustness = 0.3
            w_adaptivity = 0.3
            
            # Calculate combined score
            score = (w_stability * stability) + (w_robustness * robustness) + (w_adaptivity * adaptivity)
            
            return {
                "stability": stability,
                "robustness": robustness,
                "adaptivity": adaptivity,
                "execution_time": elapsed_time,
                "score": score
            }
            
        except Exception as e:
            logger.error(f"Error measuring regime detection performance: {str(e)}")
            return {"score": 0, "error": str(e)}
    
    def _find_optimal_feature_weights(self, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Find optimal feature weights for regime detection.
        
        Args:
            params: Regime detection parameters
            
        Returns:
            Dictionary mapping features to weights
        """
        # In a real implementation, this would analyze feature importance
        # Here we'll return simulated feature weights
        
        features = [
            "returns", "volatility", "rsi", "macd", "volume", 
            "on_balance_volume", "adx", "bollinger_width", "money_flow"
        ]
        
        weights = {}
        for feature in features:
            # Generate random weights between 0.1 and 1.0
            weights[feature] = round(0.1 + 0.9 * np.random.random(), 2)
        
        # Ensure weights sum to something reasonable
        total = sum(weights.values())
        normalized_weights = {k: v / total * len(weights) for k, v in weights.items()}
        
        return normalized_weights
    
    def _optimize_scheduler(self) -> Dict[str, Any]:
        """
        Optimize scheduler parameters.
        
        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting scheduler optimization")
        
        # Get current config
        scheduler_config = self.config.get("scheduler", {})
        
        # Default values to test
        default_params = {
            "max_workers": 10,
            "task_batch_size": 100,
            "retry_delay_ms": 1000,
            "max_retries": 3,
            "priority_queue_size": 1000
        }
        
        # Merge with current config
        current_params = {**default_params, **scheduler_config}
        
        # Parameters to test
        param_options = {
            "max_workers": [5, 10, 15, 20],
            "task_batch_size": [50, 100, 200, 500],
            "retry_delay_ms": [500, 1000, 2000, 5000],
            "max_retries": [1, 3, 5, 10]
        }
        
        # If not intensive, reduce options
        if not self.intensive:
            param_options = {
                "max_workers": [10, 15],
                "task_batch_size": [100, 200],
                "retry_delay_ms": [1000, 2000],
                "max_retries": [3, 5]
            }
        
        best_params = current_params.copy()
        best_performance = self._measure_scheduler_performance(best_params)
        
        logger.info(f"Baseline scheduler performance: {best_performance}")
        
        # Test different parameter combinations
        for param, options in param_options.items():
            for value in options:
                test_params = best_params.copy()
                test_params[param] = value
                
                performance = self._measure_scheduler_performance(test_params)
                logger.debug(f"Tested {param}={value}: {performance}")
                
                if performance["score"] > best_performance["score"]:
                    best_performance = performance
                    best_params[param] = value
                    logger.info(f"Found better {param}={value} with score {performance['score']}")
        
        # Calculate improvement
        baseline_score = self._measure_scheduler_performance(current_params)["score"]
        improvement = ((best_performance["score"] - baseline_score) / baseline_score) * 100
        
        result = {
            "baseline": current_params,
            "optimized": best_params,
            "baseline_performance": baseline_score,
            "optimized_performance": best_performance["score"],
            "improvement": improvement,
            "optimized_config": best_params
        }
        
        logger.info(f"Scheduler optimization complete. Improvement: {improvement:.2f}%")
        return result
    
    def _measure_scheduler_performance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Measure scheduler performance with given parameters.
        
        Args:
            params: Scheduler parameters
            
        Returns:
            Performance metrics
        """
        try:
            # In a real implementation, this would run tests with the scheduler
            # Here we'll simulate performance based on parameters
            
            # Start timing
            start_time = time.time()
            
            # Simulate task workload
            task_count = 1000 if self.intensive else 100
            tasks_completed = 0
            tasks_failed = 0
            
            # Calculate performance factors
            worker_factor = min(1.0, params["max_workers"] / 15)  # More workers = better throughput up to a point
            batch_factor = min(1.0, params["task_batch_size"] / 200)  # Larger batches = better throughput up to a point
            retry_factor = params["max_retries"] / 5  # More retries = higher success rate but slower
            
            # Simulate task execution
            for i in range(task_count):
                # Simulate success probability
                if np.random.random() < 0.95:  # 95% base success rate
                    tasks_completed += 1
                else:
                    # Try retries
                    retry_success = False
                    for retry in range(params["max_retries"]):
                        # Each retry has a higher success probability
                        if np.random.random() < 0.5 + (0.1 * retry):
                            retry_success = True
                            tasks_completed += 1
                            break
                        
                        # Simulate retry delay
                        time.sleep(params["retry_delay_ms"] / 1000000)  # Convert to seconds, then scale down
                    
                    if not retry_success:
                        tasks_failed += 1
            
            # Calculate metrics
            elapsed_time = time.time() - start_time
            task_rate = task_count / elapsed_time
            success_rate = tasks_completed / task_count
            
            # Calculate score (higher is better)
            # Weight factors
            w_rate = 0.6
            w_success = 0.4
            
            # Calculate combined score
            score = (w_rate * task_rate * 10) + (w_success * success_rate * 100)
            
            return {
                "task_rate": task_rate,
                "success_rate": success_rate,
                "elapsed_time": elapsed_time,
                "score": score
            }
            
        except Exception as e:
            logger.error(f"Error measuring scheduler performance: {str(e)}")
            return {"score": 0, "error": str(e)}
    
    def _optimize_state_manager(self) -> Dict[str, Any]:
        """
        Optimize state manager parameters.
        
        Returns:
            Dictionary with optimization results
        """
        logger.info("Starting state manager optimization")
        
        # Get current config
        state_config = self.config.get("state_manager", {})
        
        # Default values to test
        default_params = {
            "snapshot_interval": 100,
            "max_snapshot_count": 10,
            "max_transaction_retry": 3,
            "journal_batch_size": 50,
            "time_series_resolution": "minute"
        }
        
        # Merge with current config
        current_params = {**default_params, **state_config}
        
        # Parameters to test
        param_options = {
            "snapshot_interval": [50, 100, 200, 500],
            "max_snapshot_count": [5, 10, 15, 20],
            "max_transaction_retry": [2, 3, 5, 10],
            "journal_batch_size": [20, 50, 100, 200],
            "time_series_resolution": ["second", "minute", "hour"]
        }
        
        # If not intensive, reduce options
        if not self.intensive:
            param_options = {
                "snapshot_interval": [100, 200],
                "max_snapshot_count": [10, 15],
                "max_transaction_retry": [3, 5],
                "journal_batch_size": [50, 100],
                "time_series_resolution": ["minute", "hour"]
            }
        
        best_params = current_params.copy()
        best_performance = self._measure_state_manager_performance(best_params)
        
        logger.info(f"Baseline state manager performance: {best_performance}")
        
        # Test different parameter combinations
        for param, options in param_options.items():
            for value in options:
                test_params = best_params.copy()
                test_params[param] = value
                
                performance = self._measure_state_manager_performance(test_params)
                logger.debug(f"Tested {param}={value}: {performance}")
                
                if performance["score"] > best_performance["score"]:
                    best_performance = performance
                    best_params[param] = value
                    logger.info(f"Found better {param}={value} with score {performance['score']}")
        
        # Calculate improvement
        baseline_score = self._measure_state_manager_performance(current_params)["score"]
        improvement = ((best_performance["score"] - baseline_score) / baseline_score) * 100
        
        result = {
            "baseline": current_params,
            "optimized": best_params,
            "baseline_performance": baseline_score,
            "optimized_performance": best_performance["score"],
            "improvement": improvement,
            "optimized_config": best_params
        }
        
        logger.info(f"State manager optimization complete. Improvement: {improvement:.2f}%")
        return result
    
    def _measure_state_manager_performance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Measure state manager performance with given parameters.
        
        Args:
            params: State manager parameters
            
        Returns:
            Performance metrics
        """
        try:
            # In a real implementation, this would run tests with the state manager
            # Here we'll simulate performance based on parameters
            
            # Start timing
            start_time = time.time()
            
            # Simulate state operations
            operation_count = 1000 if self.intensive else 100
            write_count = int(operation_count * 0.7)  # 70% writes, 30% reads
            read_count = operation_count - write_count
            
            # Calculate performance factors
            snapshot_factor = 0.5 + (0.5 * min(1.0, 200 / params["snapshot_interval"]))  # Fewer snapshots = faster writes
            journal_factor = 0.5 + (0.5 * min(1.0, params["journal_batch_size"] / 100))  # Larger batch = more efficient
            resolution_factor = {"second": 0.5, "minute": 1.0, "hour": 1.5}[params["time_series_resolution"]]
            
            # Combined write factor (higher = faster)
            write_factor = snapshot_factor * journal_factor
            
            # Simulate write operations
            for i in range(write_count):
                # Add some randomness
                op_time = 0.001 / write_factor * (0.5 + np.random.random())
                # Scale down time to avoid long tests
                time.sleep(op_time / 100)
            
            # Simulate read operations
            for i in range(read_count):
                # Read speed affected by resolution
                op_time = 0.0005 / resolution_factor * (0.5 + np.random.random())
                # Scale down time to avoid long tests
                time.sleep(op_time / 100)
            
            # Calculate metrics
            elapsed_time = time.time() - start_time
            operations_per_second = operation_count / elapsed_time
            
            # Estimate memory usage based on parameters
            memory_factor = params["snapshot_interval"] / 100  # More snapshots = more memory
            memory_usage = 100 * memory_factor  # Base 100MB
            
            # Calculate score (higher is better)
            # Weight factors
            w_speed = 0.7
            w_memory = 0.3
            
            # Normalize memory usage (lower is better, 0-1 scale with 200MB as max)
            norm_memory = max(0, 1 - (memory_usage / 200))
            
            # Calculate combined score
            score = (w_speed * operations_per_second) + (w_memory * norm_memory * 100)
            
            return {
                "operations_per_second": operations_per_second,
                "memory_usage_mb": memory_usage,
                "elapsed_time": elapsed_time,
                "score": score
            }
            
        except Exception as e:
            logger.error(f"Error measuring state manager performance: {str(e)}")
            return {"score": 0, "error": str(e)}


def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Optimize system parameters')
    parser.add_argument('--config', type=str, default='config/system_config.json',
                      help='Path to configuration file')
    parser.add_argument('--components', type=str, default=None,
                      help='Comma-separated list of components to optimize')
    parser.add_argument('--intensive', action='store_true',
                      help='Run intensive optimization (slower but more thorough)')
    parser.add_argument('--output', type=str, default=None,
                      help='Output path for optimized configurations')
    parser.add_argument('--apply', action='store_true',
                      help='Apply optimized settings immediately')
    
    args = parser.parse_args()
    
    # Parse components if specified
    components = None
    if args.components:
        components = [c.strip() for c in args.components.split(',')]
    
    # Default output path if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"config/optimized_config_{timestamp}.json"
    
    try:
        # Print system info
        logger.info(f"System information:")
        logger.info(f"  Platform: {platform.platform()}")
        logger.info(f"  Python: {platform.python_version()}")
        logger.info(f"  CPU: {psutil.cpu_count(logical=False)} cores ({psutil.cpu_count()} threads)")
        logger.info(f"  Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
        # Initialize optimizer
        optimizer = SystemOptimizer(
            config_path=args.config,
            components=components,
            intensive=args.intensive
        )
        
        # Run optimization
        logger.info("Starting system optimization process...")
        result = optimizer.optimize()
        
        # Save optimized config
        optimizer._save_config(args.output)
        
        # Apply if requested
        if args.apply:
            logger.info("Applying optimized configuration...")
            # This would restart components with new configuration
            # For now, just print that it would be applied
            logger.info("Configuration would be applied (not implemented)")
        
        # Print summary
        summary = result["summary"]
        print("\n" + "="*50)
        print("System Optimization Results")
        print("="*50)
        print(f"Optimized components: {', '.join(summary['optimized_components'])}")
        print(f"Average improvement: {summary['average_improvement']:.2f}%")
        print(f"Optimization time: {summary['elapsed_time']:.2f} seconds")
        print(f"Peak memory usage: {summary['peak_memory_mb']:.2f} MB")
        print(f"Configuration saved to: {args.output}")
        
        # Print component-specific improvements
        print("\nComponent Improvements:")
        for component, improvement in summary["improvements"].items():
            print(f"  {component}: {improvement:.2f}%")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()