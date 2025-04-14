"""
Feature Store for ML-Powered Trading System.

This module provides a centralized feature repository for storing, retrieving,
and managing features with lineage tracking, versioning, and metadata management.
It implements the Single Source of Truth architecture principle.
"""

import pandas as pd
import numpy as np
import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Set, Tuple, Any
from pathlib import Path
import pickle
import hashlib
from functools import lru_cache

# Import system dependencies
from utils.logger import get_logger
from data.storage.time_series_store import TimeSeriesStore
from data.processors.data_integrity import DataValidator


class FeatureStore:
    """
    Centralized feature repository implementing single source of truth for features.

    This class provides:
    1. Storage and retrieval of feature data with efficient time series indexing
    2. Feature metadata management and lineage tracking
    3. Feature versioning and rollback capabilities
    4. Integration with the ML-Powered Trading System architecture
    """

    def __init__(self,
                 base_path: Optional[str] = None,
                 time_series_store: Optional[TimeSeriesStore] = None,
                 data_validator: Optional[DataValidator] = None,
                 enable_caching: bool = True,
                 cache_ttl: int = 3600,
                 enable_versioning: bool = True,
                 compression: bool = True,
                 logging_level: int = logging.INFO):
        """
        Initialize the feature store.

        Args:
            base_path: Base directory for storing features
            time_series_store: Optional TimeSeriesStore for storage backend
            data_validator: Optional DataValidator for data validation
            enable_caching: Whether to cache feature data in memory
            cache_ttl: Time-to-live for cached data in seconds
            enable_versioning: Whether to maintain feature versions
            compression: Whether to compress stored data
            logging_level: Logging level
        """
        # Set up logging
        self.logger = get_logger("feature_store")
        self.logger.setLevel(logging_level)

        # Set up paths
        self.base_path = base_path or os.path.join(os.getcwd(), 'data', 'features')
        self.data_path = os.path.join(self.base_path, 'data')
        self.metadata_path = os.path.join(self.base_path, 'metadata')
        self.version_path = os.path.join(self.base_path, 'versions')

        # Create directories if they don't exist
        for path in [self.data_path, self.metadata_path, self.version_path]:
            os.makedirs(path, exist_ok=True)

        # Initialize dependencies
        self.time_series_store = time_series_store or TimeSeriesStore()
        self.data_validator = data_validator or DataValidator()

        # Configuration
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self.enable_versioning = enable_versioning
        self.compression = compression

        # Initialize caches
        self.feature_cache = {}
        self.metadata_cache = {}
        self.cache_timestamps = {}

        # Feature registry
        self.feature_registry = self._load_feature_registry()

        # Performance tracking
        self.performance_stats = {
            'reads': 0,
            'writes': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_read_time': 0,
            'avg_write_time': 0
        }

        self.logger.info(f"Feature store initialized at {self.base_path}")

    def store_features(self,
                       data: pd.DataFrame,
                       symbol: str,
                       timeframe: str,
                       feature_set: str,
                       metadata: Optional[Dict[str, Any]] = None,
                       version: Optional[str] = None) -> bool:
        """
        Store feature data in the feature store.

        Args:
            data: DataFrame containing feature data
            symbol: Trading symbol
            timeframe: Data timeframe
            feature_set: Feature set name
            metadata: Optional metadata about the features
            version: Optional version identifier

        Returns:
            Boolean indicating success
        """
        start_time = time.time()

        if data.empty:
            self.logger.warning(f"Attempted to store empty data for {symbol} {timeframe} {feature_set}")
            return False

        # Validate data
        if self.data_validator and not self.data_validator.validate_dataframe(data):
            self.logger.error(f"Data validation failed for {symbol} {timeframe} {feature_set}")
            return False

        # Generate key
        key = self._generate_key(symbol, timeframe, feature_set)

        # Generate version if not provided and versioning enabled
        if self.enable_versioning:
            if version is None:
                version = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(data)}"

        try:
            # Use time series store if available, otherwise use pickle
            if self.time_series_store:
                self.time_series_store.store(key, data)
            else:
                # Ensure directory exists
                file_path = os.path.join(self.data_path, f"{key}.pkl")
                directory = os.path.dirname(file_path)
                os.makedirs(directory, exist_ok=True)

                # Save with pickle
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Store metadata
            self._store_metadata(key, {
                'symbol': symbol,
                'timeframe': timeframe,
                'feature_set': feature_set,
                'columns': list(data.columns),
                'features': list(set(data.columns) - {'open', 'high', 'low', 'close', 'volume'}),
                'start_time': data.index.min().isoformat(),
                'end_time': data.index.max().isoformat(),
                'row_count': len(data),
                'last_updated': datetime.now().isoformat(),
                'version': version,
                **({} if metadata is None else metadata)
            })

            # Store version if versioning enabled
            if self.enable_versioning and version:
                self._store_version(key, version, data, metadata)

            # Update cache if enabled
            if self.enable_caching:
                self.feature_cache[key] = data
                self.cache_timestamps[key] = time.time()

            # Update feature registry
            self._register_feature(symbol, timeframe, feature_set, list(data.columns))

            # Update performance stats
            self.performance_stats['writes'] += 1
            write_time = time.time() - start_time
            self.performance_stats['avg_write_time'] = (
                (self.performance_stats['avg_write_time'] * (self.performance_stats['writes'] - 1) + write_time) /
                self.performance_stats['writes']
            )

            self.logger.info(f"Stored {len(data)} rows for {symbol} {timeframe} {feature_set}")
            return True

        except Exception as e:
            self.logger.error(f"Error storing features: {str(e)}")
            return False

    def get_features(self,
                     symbol: str,
                     timeframe: str,
                     feature_set: str,
                     start_time: Optional[Union[str, datetime]] = None,
                     end_time: Optional[Union[str, datetime]] = None,
                     columns: Optional[List[str]] = None,
                     limit: Optional[int] = None,
                     version: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Retrieve feature data from the feature store.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            feature_set: Feature set name
            start_time: Optional start time filter
            end_time: Optional end time filter
            columns: Optional list of columns to retrieve
            limit: Optional maximum number of rows to retrieve
            version: Optional version to retrieve

        Returns:
            DataFrame with requested features or None if not found
        """
        start_time_perf = time.time()

        # Generate key
        key = self._generate_key(symbol, timeframe, feature_set)

        # Check cache first if enabled
        if self.enable_caching and key in self.feature_cache:
            cache_age = time.time() - self.cache_timestamps.get(key, 0)
            if cache_age < self.cache_ttl:
                self.performance_stats['cache_hits'] += 1
                data = self.feature_cache[key]

                # Apply filters
                data = self._apply_filters(data, start_time, end_time, columns, limit)

                self.logger.debug(f"Cache hit for {symbol} {timeframe} {feature_set}")

                # Update performance stats
                read_time = time.time() - start_time_perf
                self.performance_stats['avg_read_time'] = (
                    (self.performance_stats['avg_read_time'] * self.performance_stats['reads'] + read_time) /
                    (self.performance_stats['reads'] + 1)
                )
                self.performance_stats['reads'] += 1

                return data

        self.performance_stats['cache_misses'] += 1

        # Check if version is specified and versioning is enabled
        if version and self.enable_versioning:
            data = self._get_version(key, version)
            if data is not None:
                data = self._apply_filters(data, start_time, end_time, columns, limit)
                self.logger.info(f"Retrieved version {version} for {symbol} {timeframe} {feature_set}")
                return data

        # Try to load from storage
        try:
            if self.time_series_store:
                data = self.time_series_store.retrieve(key)
            else:
                file_path = os.path.join(self.data_path, f"{key}.pkl")
                if not os.path.exists(file_path):
                    self.logger.warning(f"No data found for {symbol} {timeframe} {feature_set}")
                    return None

                with open(file_path, 'rb') as f:
                    data = pickle.load(f)

            if data is not None:
                # Apply filters
                data = self._apply_filters(data, start_time, end_time, columns, limit)

                # Update cache if enabled
                if self.enable_caching:
                    self.feature_cache[key] = data
                    self.cache_timestamps[key] = time.time()

                self.logger.info(f"Retrieved {len(data)} rows for {symbol} {timeframe} {feature_set}")

                # Update performance stats
                read_time = time.time() - start_time_perf
                self.performance_stats['avg_read_time'] = (
                    (self.performance_stats['avg_read_time'] * self.performance_stats['reads'] + read_time) /
                    (self.performance_stats['reads'] + 1)
                )
                self.performance_stats['reads'] += 1

                return data

            return None

        except Exception as e:
            self.logger.error(f"Error retrieving features: {str(e)}")
            return None

    def update_features(self,
                       data: pd.DataFrame,
                       symbol: str,
                       timeframe: str,
                       feature_set: str,
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update existing features with new data.

        Args:
            data: DataFrame with new feature data
            symbol: Trading symbol
            timeframe: Data timeframe
            feature_set: Feature set name
            metadata: Optional metadata to update

        Returns:
            Boolean indicating success
        """
        # Generate key
        key = self._generate_key(symbol, timeframe, feature_set)

        # Get existing data
        existing_data = self.get_features(symbol, timeframe, feature_set)

        if existing_data is None:
            # No existing data, just store the new data
            return self.store_features(data, symbol, timeframe, feature_set, metadata)

        # Combine data, overwriting existing data with new data where timestamps match
        combined_data = pd.concat([existing_data, data])

        # Remove duplicates by index (timestamp), keeping the latest entry
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]

        # Sort by timestamp
        combined_data.sort_index(inplace=True)

        # Store the updated data
        return self.store_features(combined_data, symbol, timeframe, feature_set, metadata)

    def get_available_features(self,
                              symbol: Optional[str] = None,
                              timeframe: Optional[str] = None,
                              feature_set: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Get available features, optionally filtered.

        Args:
            symbol: Optional symbol filter
            timeframe: Optional timeframe filter
            feature_set: Optional feature set filter

        Returns:
            Dictionary mapping keys to lists of feature names
        """
        result = {}

        for key, info in self.feature_registry.items():
            # Apply filters
            if symbol and info['symbol'] != symbol:
                continue

            if timeframe and info['timeframe'] != timeframe:
                continue

            if feature_set and info['feature_set'] != feature_set:
                continue

            # Include in result
            result[key] = info['features']

        return result

    def get_metadata(self,
                    symbol: str,
                    timeframe: str,
                    feature_set: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific feature set.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            feature_set: Feature set name

        Returns:
            Metadata dictionary or None if not found
        """
        key = self._generate_key(symbol, timeframe, feature_set)

        # Check cache first
        if key in self.metadata_cache:
            return self.metadata_cache[key]

        # Try to load from storage
        try:
            metadata_path = os.path.join(self.metadata_path, f"{key}.json")
            if not os.path.exists(metadata_path):
                return None

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Update cache
            self.metadata_cache[key] = metadata

            return metadata

        except Exception as e:
            self.logger.error(f"Error retrieving metadata: {str(e)}")
            return None

    def update_feature_metadata(self,
                               features: List[str],
                               metadata: Dict[str, Any],
                               symbol: Optional[str] = None,
                               timeframe: Optional[str] = None,
                               feature_set: Optional[str] = None) -> bool:
        """
        Update metadata for specific features.

        Args:
            features: List of feature names to update
            metadata: Metadata to apply
            symbol: Optional symbol filter
            timeframe: Optional timeframe filter
            feature_set: Optional feature set filter

        Returns:
            Boolean indicating success
        """
        # Find all feature sets that match the criteria
        feature_sets = self.get_available_features(symbol, timeframe, feature_set)

        success = True
        for key, available_features in feature_sets.items():
            # Check if any of our features are in this feature set
            matching_features = [f for f in features if f in available_features]

            if not matching_features:
                continue

            # Get existing metadata
            existing_metadata = self.get_metadata(*self._parse_key(key))

            if existing_metadata is None:
                existing_metadata = {}

            # Update feature-specific metadata
            for feature in matching_features:
                if 'feature_metadata' not in existing_metadata:
                    existing_metadata['feature_metadata'] = {}

                if feature not in existing_metadata['feature_metadata']:
                    existing_metadata['feature_metadata'][feature] = {}

                # Update with new metadata
                for meta_key, meta_value in metadata.items():
                    if isinstance(meta_value, dict) and feature in meta_value:
                        # If metadata has feature-specific values
                        existing_metadata['feature_metadata'][feature][meta_key] = meta_value[feature]
                    else:
                        # Otherwise apply the same value to all features
                        existing_metadata['feature_metadata'][feature][meta_key] = meta_value

            # Store updated metadata
            try:
                symbol, timeframe, feature_set = self._parse_key(key)
                self._store_metadata(key, existing_metadata)

                # Update cache
                self.metadata_cache[key] = existing_metadata

                self.logger.info(f"Updated metadata for {len(matching_features)} features in {key}")

            except Exception as e:
                self.logger.error(f"Error updating metadata for {key}: {str(e)}")
                success = False

        return success

    def delete_features(self,
                       symbol: str,
                       timeframe: str,
                       feature_set: str,
                       version: Optional[str] = None) -> bool:
        """
        Delete features from the store.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            feature_set: Feature set name
            version: Optional specific version to delete

        Returns:
            Boolean indicating success
        """
        key = self._generate_key(symbol, timeframe, feature_set)

        try:
            if version and self.enable_versioning:
                # Delete specific version
                version_path = os.path.join(self.version_path, key, f"{version}.pkl")
                if os.path.exists(version_path):
                    os.remove(version_path)
                    self.logger.info(f"Deleted version {version} for {key}")
                    return True
                else:
                    self.logger.warning(f"Version {version} not found for {key}")
                    return False

            # Delete current data
            if self.time_series_store:
                self.time_series_store.delete(key)
            else:
                file_path = os.path.join(self.data_path, f"{key}.pkl")
                if os.path.exists(file_path):
                    os.remove(file_path)

            # Delete metadata
            metadata_path = os.path.join(self.metadata_path, f"{key}.json")
            if os.path.exists(metadata_path):
                os.remove(metadata_path)

            # Delete versions if versioning enabled
            if self.enable_versioning:
                version_dir = os.path.join(self.version_path, key)
                if os.path.exists(version_dir):
                    import shutil
                    shutil.rmtree(version_dir)

            # Update caches
            if key in self.feature_cache:
                del self.feature_cache[key]

            if key in self.cache_timestamps:
                del self.cache_timestamps[key]

            if key in self.metadata_cache:
                del self.metadata_cache[key]

            # Update feature registry
            if key in self.feature_registry:
                del self.feature_registry[key]
                self._save_feature_registry()

            self.logger.info(f"Deleted features for {key}")
            return True

        except Exception as e:
            self.logger.error(f"Error deleting features: {str(e)}")
            return False

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the feature store.

        Returns:
            Dictionary with performance statistics
        """
        stats = {**self.performance_stats}

        # Add cache stats
        if self.enable_caching:
            stats['cache_size'] = len(self.feature_cache)
            stats['cache_memory_usage'] = sum(df.memory_usage(deep=True).sum()
                                            for df in self.feature_cache.values())

        return stats

    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self.feature_cache.clear()
        self.metadata_cache.clear()
        self.cache_timestamps.clear()
        self.logger.info("Cache cleared")

    def get_feature_versions(self,
                            symbol: str,
                            timeframe: str,
                            feature_set: str) -> List[str]:
        """
        Get available versions for a feature set.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            feature_set: Feature set name

        Returns:
            List of version identifiers
        """
        if not self.enable_versioning:
            return []

        key = self._generate_key(symbol, timeframe, feature_set)
        version_dir = os.path.join(self.version_path, key)

        if not os.path.exists(version_dir):
            return []

        versions = [os.path.splitext(file)[0] for file in os.listdir(version_dir)
                   if file.endswith('.pkl')]

        return sorted(versions, reverse=True)

    def _generate_key(self, symbol: str, timeframe: str, feature_set: str) -> str:
        """Generate a unique key for a feature set."""
        return f"{symbol}_{timeframe}_{feature_set}"

    def _parse_key(self, key: str) -> Tuple[str, str, str]:
        """Parse a key into symbol, timeframe, and feature_set."""
        parts = key.split('_')

        # Handle case where symbol might contain underscores
        if len(parts) < 3:
            raise ValueError(f"Invalid key format: {key}")

        timeframe = parts[-2]
        feature_set = parts[-1]
        symbol = '_'.join(parts[:-2])

        return symbol, timeframe, feature_set

    def _store_metadata(self, key: str, metadata: Dict[str, Any]) -> None:
        """Store metadata for a feature set."""
        metadata_path = os.path.join(self.metadata_path, f"{key}.json")
        directory = os.path.dirname(metadata_path)
        os.makedirs(directory, exist_ok=True)

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Update cache
        self.metadata_cache[key] = metadata

    def _store_version(self, key: str, version: str, data: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store a specific version of feature data."""
        if not self.enable_versioning:
            return

        # Create version directory
        version_dir = os.path.join(self.version_path, key)
        os.makedirs(version_dir, exist_ok=True)

        # Store version data
        version_path = os.path.join(version_dir, f"{version}.pkl")
        with open(version_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Store version metadata
        if metadata is not None:
            version_meta_path = os.path.join(version_dir, f"{version}.json")
            with open(version_meta_path, 'w') as f:
                json.dump({
                    **metadata,
                    'version': version,
                    'timestamp': datetime.now().isoformat(),
                    'row_count': len(data),
                    'columns': list(data.columns)
                }, f, indent=2)

        self.logger.debug(f"Stored version {version} for {key}")

    def _get_version(self, key: str, version: str) -> Optional[pd.DataFrame]:
        """Retrieve a specific version of feature data."""
        if not self.enable_versioning:
            return None

        version_path = os.path.join(self.version_path, key, f"{version}.pkl")

        if not os.path.exists(version_path):
            self.logger.warning(f"Version {version} not found for {key}")
            return None

        try:
            with open(version_path, 'rb') as f:
                data = pickle.load(f)

            return data

        except Exception as e:
            self.logger.error(f"Error retrieving version {version} for {key}: {str(e)}")
            return None

    def _apply_filters(self,
                      data: pd.DataFrame,
                      start_time: Optional[Union[str, datetime]] = None,
                      end_time: Optional[Union[str, datetime]] = None,
                      columns: Optional[List[str]] = None,
                      limit: Optional[int] = None) -> pd.DataFrame:
        """Apply filters to retrieved data."""
        # Make a copy to avoid modifying cached data
        result = data.copy()

        # Apply time filters
        if start_time is not None:
            if isinstance(start_time, str):
                start_time = pd.Timestamp(start_time)
            result = result[result.index >= start_time]

        if end_time is not None:
            if isinstance(end_time, str):
                end_time = pd.Timestamp(end_time)
            result = result[result.index <= end_time]

        # Apply column filter
        if columns is not None:
            # Make sure all requested columns exist
            valid_columns = [col for col in columns if col in result.columns]
            if len(valid_columns) < len(columns):
                missing = set(columns) - set(valid_columns)
                self.logger.warning(f"Some requested columns not found: {missing}")

            result = result[valid_columns]

        # Apply limit
        if limit is not None:
            result = result.tail(limit)

        return result

    def _load_feature_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load the feature registry from disk."""
        registry_path = os.path.join(self.base_path, 'feature_registry.json')

        if not os.path.exists(registry_path):
            return {}

        try:
            with open(registry_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading feature registry: {str(e)}")
            return {}

    def _save_feature_registry(self) -> None:
        """Save the feature registry to disk."""
        registry_path = os.path.join(self.base_path, 'feature_registry.json')

        try:
            with open(registry_path, 'w') as f:
                json.dump(self.feature_registry, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving feature registry: {str(e)}")

    def _register_feature(self, symbol: str, timeframe: str, feature_set: str, columns: List[str]) -> None:
        """Register a feature set in the registry."""
        key = self._generate_key(symbol, timeframe, feature_set)

        features = [col for col in columns if col not in {'open', 'high', 'low', 'close', 'volume'}]

        self.feature_registry[key] = {
            'symbol': symbol,
            'timeframe': timeframe,
            'feature_set': feature_set,
            'features': features,
            'last_updated': datetime.now().isoformat()
        }

        self._save_feature_registry()