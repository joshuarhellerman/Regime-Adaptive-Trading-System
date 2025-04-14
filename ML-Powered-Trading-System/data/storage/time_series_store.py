"""
time_series_store.py - Optimized Time Series Storage System

This module provides efficient storage and retrieval of time-series data,
with support for downsampling, aggregation, and efficient range queries.
It's used for historical data that needs to be analyzed over time.
"""

import json
import logging
import os
import pickle
import threading
import time
import bisect
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, TypeVar, Generic, Callable

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Aggregation(Enum):
    """Supported aggregation methods for time series data"""
    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    FIRST = "first"
    LAST = "last"
    COUNT = "count"


class Resolution(Enum):
    """Time series resolution options"""
    RAW = "raw"  # No downsampling
    MILLISECOND = "millisecond"
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class TimeSeriesStore:
    """
    Efficient time series data store with support for multiple series,
    downsampling, and aggregation.
    """

    def __init__(self, directory: str):
        """
        Initialize the time series store.

        Args:
            directory: Directory where time series files will be stored
        """
        self._directory = Path(directory)
        self._lock = threading.RLock()
        self._series_cache: Dict[str, Dict[Resolution, List[Tuple[float, Any]]]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._max_memory_points = 10000  # Maximum points per series in memory

        # Ensure directory exists
        os.makedirs(self._directory, exist_ok=True)

        # Create resolution subdirectories
        for resolution in Resolution:
            os.makedirs(self._directory / resolution.value, exist_ok=True)

        # Load metadata
        self._load_metadata()

        logger.info(f"Time series store initialized at {directory}")

    def store(self, series: str, value: Any, timestamp: Optional[float] = None) -> bool:
        """
        Store a value in a time series.

        Args:
            series: Name of the time series
            value: Value to store
            timestamp: Optional timestamp (defaults to current time)

        Returns:
            True if successful, False otherwise
        """
        if timestamp is None:
            timestamp = time.time()

        with self._lock:
            # Initialize series if not exists
            if series not in self._series_cache:
                self._series_cache[series] = {Resolution.RAW: []}
                self._ensure_metadata(series)

            # Add to raw data
            raw_data = self._series_cache[series][Resolution.RAW]

            # Ensure timestamps are in order (binary search for insertion point)
            insertion_point = bisect.bisect_left([t for t, _ in raw_data], timestamp)
            raw_data.insert(insertion_point, (timestamp, value))

            # Update metadata
            self._update_series_metadata(series, timestamp, value)

            # Flush to disk if needed
            if len(raw_data) >= self._max_memory_points:
                self._flush_series(series, Resolution.RAW)

            return True

    def get(self, series: str, start_time: float, end_time: float,
            resolution: Resolution = Resolution.RAW,
            aggregation: Aggregation = Aggregation.LAST) -> List[Tuple[float, Any]]:
        """
        Get values from a time series for a specific time range.

        Args:
            series: Name of the time series
            start_time: Start timestamp
            end_time: End timestamp
            resolution: Time resolution for downsampling
            aggregation: Aggregation method if downsampling

        Returns:
            List of (timestamp, value) tuples
        """
        with self._lock:
            # Check if series exists
            if series not in self._metadata:
                return []

            # Load data if not in cache
            if series not in self._series_cache or resolution not in self._series_cache[series]:
                self._load_series(series, resolution)

            if resolution not in self._series_cache[series]:
                # If requested resolution doesn't exist, use raw data and downsample
                if Resolution.RAW not in self._series_cache[series]:
                    self._load_series(series, Resolution.RAW)

                if Resolution.RAW not in self._series_cache[series]:
                    return []

                raw_data = self._series_cache[series][Resolution.RAW]

                # Downsample the data
                downsampled = self._downsample(raw_data, resolution, aggregation)

                # Store in cache
                self._series_cache[series][resolution] = downsampled

            # Get data for the requested time range
            data = self._series_cache[series][resolution]

            # Find bounds using binary search
            start_idx = bisect.bisect_left([t for t, _ in data], start_time)
            end_idx = bisect.bisect_right([t for t, _ in data], end_time)

            return data[start_idx:end_idx]

    def get_last(self, series: str) -> Optional[Tuple[float, Any]]:
        """
        Get the last value from a time series.

        Args:
            series: Name of the time series

        Returns:
            Tuple of (timestamp, value) or None if series is empty
        """
        with self._lock:
            # Check if series exists
            if series not in self._metadata:
                return None

            # Get bounds from metadata
            metadata = self._metadata[series]

            if 'last_timestamp' not in metadata:
                return None

            last_timestamp = metadata['last_timestamp']

            # Try to get from cache first
            if series in self._series_cache and Resolution.RAW in self._series_cache[series]:
                raw_data = self._series_cache[series][Resolution.RAW]
                if raw_data:
                    # Search for the point with this timestamp
                    for i in range(len(raw_data) - 1, -1, -1):
                        if raw_data[i][0] == last_timestamp:
                            return raw_data[i]

            # If not in cache, load specific point from disk
            return self._load_point(series, last_timestamp)

    def get_first(self, series: str) -> Optional[Tuple[float, Any]]:
        """
        Get the first value from a time series.

        Args:
            series: Name of the time series

        Returns:
            Tuple of (timestamp, value) or None if series is empty
        """
        with self._lock:
            # Check if series exists
            if series not in self._metadata:
                return None

            # Get bounds from metadata
            metadata = self._metadata[series]

            if 'first_timestamp' not in metadata:
                return None

            first_timestamp = metadata['first_timestamp']

            # Try to get from cache first
            if series in self._series_cache and Resolution.RAW in self._series_cache[series]:
                raw_data = self._series_cache[series][Resolution.RAW]
                if raw_data:
                    # Search for the point with this timestamp
                    for point in raw_data:
                        if point[0] == first_timestamp:
                            return point

            # If not in cache, load specific point from disk
            return self._load_point(series, first_timestamp)

    def get_range(self, series: str) -> Optional[Tuple[float, float]]:
        """
        Get the time range of a time series.

        Args:
            series: Name of the time series

        Returns:
            Tuple of (first_timestamp, last_timestamp) or None if series is empty
        """
        with self._lock:
            # Check if series exists
            if series not in self._metadata:
                return None

            # Get bounds from metadata
            metadata = self._metadata[series]

            if 'first_timestamp' not in metadata or 'last_timestamp' not in metadata:
                return None

            return (metadata['first_timestamp'], metadata['last_timestamp'])

    def get_series_names(self) -> List[str]:
        """
        Get names of all available time series.

        Returns:
            List of series names
        """
        with self._lock:
            return list(self._metadata.keys())

    def get_metadata(self, series: str) -> Dict[str, Any]:
        """
        Get metadata for a specific time series.

        Args:
            series: Name of the time series

        Returns:
            Dictionary of metadata
        """
        with self._lock:
            return self._metadata.get(series, {}).copy()

    def count(self, series: str) -> int:
        """
        Get the number of points in a time series.

        Args:
            series: Name of the time series

        Returns:
            Number of points
        """
        with self._lock:
            if series not in self._metadata:
                return 0

            return self._metadata[series].get('count', 0)

    def delete(self, series: str) -> bool:
        """
        Delete a time series and all its data.

        Args:
            series: Name of the time series

        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            if series not in self._metadata:
                return False

            # Remove from cache
            if series in self._series_cache:
                del self._series_cache[series]

            # Remove from metadata
            del self._metadata[series]

            # Remove files
            series_dir = self._get_series_dir(series)

            try:
                # Remove from all resolution directories
                for resolution in Resolution:
                    res_path = self._directory / resolution.value / series_dir
                    if res_path.exists():
                        for file in res_path.glob("*.data"):
                            try:
                                os.remove(file)
                            except OSError as e:
                                logger.warning(f"Error removing time series file {file}: {e}")
                        try:
                            os.rmdir(res_path)
                        except OSError:
                            pass

                # Save metadata
                self._save_metadata()

                return True
            except Exception as e:
                logger.error(f"Error deleting time series {series}: {e}")
                return False

    def clear(self) -> int:
        """
        Clear all time series data.

        Returns:
            Number of series cleared
        """
        with self._lock:
            count = len(self._metadata)

            # Clear cache
            self._series_cache.clear()

            # Clear metadata
            self._metadata.clear()

            # Remove files (keep directories)
            try:
                for resolution in Resolution:
                    res_dir = self._directory / resolution.value
                    if res_dir.exists():
                        for file in res_dir.glob("**/*.data"):
                            try:
                                os.remove(file)
                            except OSError as e:
                                logger.warning(f"Error removing time series file {file}: {e}")

                # Save empty metadata
                self._save_metadata()

                return count
            except Exception as e:
                logger.error(f"Error clearing time series store: {e}")
                return 0

    def compact(self, series: Optional[str] = None) -> bool:
        """
        Compact time series data to reclaim disk space.

        Args:
            series: Specific series to compact, or all if None

        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                if series is not None:
                    if series not in self._metadata:
                        return False

                    # Flush all cached data
                    if series in self._series_cache:
                        for resolution in list(self._series_cache[series].keys()):
                            self._flush_series(series, resolution)

                    # Reload and rewrite all files
                    self._compact_series(series)
                else:
                    # Compact all series
                    for s in list(self._metadata.keys()):
                        # Flush all cached data
                        if s in self._series_cache:
                            for resolution in list(self._series_cache[s].keys()):
                                self._flush_series(s, resolution)

                        # Reload and rewrite all files
                        self._compact_series(s)

                return True
            except Exception as e:
                logger.error(f"Error compacting time series data: {e}")
                return False

    def flush(self) -> bool:
        """
        Flush all cached data to disk.

        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            try:
                for series in list(self._series_cache.keys()):
                    for resolution in list(self._series_cache[series].keys()):
                        self._flush_series(series, resolution)

                # Save metadata
                self._save_metadata()

                return True
            except Exception as e:
                logger.error(f"Error flushing time series data: {e}")
                return False

    def _load_metadata(self) -> None:
        """Load metadata from disk"""
        metadata_file = self._directory / "metadata.json"

        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self._metadata = json.load(f)
            except Exception as e:
                logger.error(f"Error loading time series metadata: {e}")
                self._metadata = {}
        else:
            self._metadata = {}

    def _save_metadata(self) -> None:
        """Save metadata to disk"""
        metadata_file = self._directory / "metadata.json"

        try:
            with open(metadata_file, 'w') as f:
                json.dump(self._metadata, f)
        except Exception as e:
            logger.error(f"Error saving time series metadata: {e}")

    def _ensure_metadata(self, series: str) -> None:
        """Ensure metadata exists for a series"""
        if series not in self._metadata:
            self._metadata[series] = {
                'count': 0,
                'first_timestamp': None,
                'last_timestamp': None,
                'min_value': None,
                'max_value': None
            }

    def _update_series_metadata(self, series: str, timestamp: float, value: Any) -> None:
        """Update metadata for a series"""
        metadata = self._metadata[series]

        # Update count
        metadata['count'] = metadata.get('count', 0) + 1

        # Update timestamps
        if metadata.get('first_timestamp') is None or timestamp < metadata['first_timestamp']:
            metadata['first_timestamp'] = timestamp

        if metadata.get('last_timestamp') is None or timestamp > metadata['last_timestamp']:
            metadata['last_timestamp'] = timestamp

        # Update min/max if numeric
        if isinstance(value, (int, float)):
            if metadata.get('min_value') is None or value < metadata['min_value']:
                metadata['min_value'] = value

            if metadata.get('max_value') is None or value > metadata['max_value']:
                metadata['max_value'] = value

    def _get_series_dir(self, series: str) -> str:
        """Get directory name for a series"""
        # Use a simple hash function to distribute series into subdirectories
        hash_val = hash(series) % 256
        return f"{hash_val:02x}/{series}"

    def _get_series_file(self, series: str, resolution: Resolution, chunk_time: float) -> Path:
        """Get file path for a series chunk"""
        series_dir = self._get_series_dir(series)
        chunk_dir = self._directory / resolution.value / series_dir

        # Create directory if not exists
        os.makedirs(chunk_dir, exist_ok=True)

        # Use chunk time as filename
        return chunk_dir / f"{int(chunk_time)}.data"

    def _flush_series(self, series: str, resolution: Resolution) -> None:
        """Flush series data to disk"""
        if series not in self._series_cache or resolution not in self._series_cache[series]:
            return

        data = self._series_cache[series][resolution]
        if not data:
            return

        # Group data by chunk
        chunks = defaultdict(list)

        # Determine chunk size based on resolution
        if resolution == Resolution.RAW:
            chunk_size = 3600  # 1 hour chunks for raw data
        else:
            chunk_size = 86400  # 1 day chunks for downsampled data

        for timestamp, value in data:
            chunk_time = int(timestamp / chunk_size) * chunk_size
            chunks[chunk_time].append((timestamp, value))

        # Save each chunk
        for chunk_time, chunk_data in chunks.items():
            file_path = self._get_series_file(series, resolution, chunk_time)

            # Load existing data if file exists
            existing_data = []
            if file_path.exists():
                try:
                    with open(file_path, 'rb') as f:
                        existing_data = pickle.load(f)
                except Exception as e:
                    logger.error(f"Error loading time series chunk {file_path}: {e}")
                    existing_data = []

            # Merge with existing data
            merged_data = existing_data + chunk_data

            # Sort by timestamp
            merged_data.sort(key=lambda x: x[0])

            # Remove duplicates (keep last value for each timestamp)
            if merged_data:
                unique_data = []
                prev_timestamp = None
                for timestamp, value in merged_data:
                    if timestamp != prev_timestamp:
                        unique_data.append((timestamp, value))
                    else:
                        unique_data[-1] = (timestamp, value)
                    prev_timestamp = timestamp
                merged_data = unique_data

            # Save merged data
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(merged_data, f)
            except Exception as e:
                logger.error(f"Error saving time series chunk {file_path}: {e}")

        # Clear cache for this resolution
        self._series_cache[series][resolution] = []

        # Save metadata
        self._save_metadata()

    def _load_series(self, series: str, resolution: Resolution) -> None:
        """Load series data from disk"""
        if series not in self._metadata:
            return

        # Initialize cache for this resolution
        if series not in self._series_cache:
            self._series_cache[series] = {}

        self._series_cache[series][resolution] = []

        # Get series directory
        series_dir = self._get_series_dir(series)
        res_dir = self._directory / resolution.value / series_dir

        if not res_dir.exists():
            return

        # Load all chunks
        data = []
        for file in sorted(res_dir.glob("*.data")):
            try:
                with open(file, 'rb') as f:
                    chunk_data = pickle.load(f)
                data.extend(chunk_data)
            except Exception as e:
                logger.error(f"Error loading time series chunk {file}: {e}")

        # Sort by timestamp
        data.sort(key=lambda x: x[0])

        # Limit memory usage
        if len(data) > self._max_memory_points:
            # Keep most recent data in memory
            data = data[-self._max_memory_points:]

        self._series_cache[series][resolution] = data

    def _load_point(self, series: str, timestamp: float) -> Optional[Tuple[float, Any]]:
        """Load a specific point from disk"""
        # Get series directory
        series_dir = self._get_series_dir(series)

        # Determine resolution (always use RAW)
        resolution = Resolution.RAW

        # Determine chunk
        chunk_size = 3600  # 1 hour chunks for raw data
        chunk_time = int(timestamp / chunk_size) * chunk_size

        # Get file path
        file_path = self._get_series_file(series, resolution, chunk_time)

        if not file_path.exists():
            return None

        # Load chunk
        try:
            with open(file_path, 'rb') as f:
                chunk_data = pickle.load(f)

            # Find the point with this timestamp
            for point in chunk_data:
                if point[0] == timestamp:
                    return point

            return None
        except Exception as e:
            logger.error(f"Error loading time series point {file_path}: {e}")
            return None

    def _downsample(self, data: List[Tuple[float, Any]], resolution: Resolution,
                    aggregation: Aggregation) -> List[Tuple[float, Any]]:
        """Downsample data to the specified resolution"""
        if not data:
            return []

        # Determine bucket size based on resolution
        if resolution == Resolution.MILLISECOND:
            bucket_size = 0.001
        elif resolution == Resolution.SECOND:
            bucket_size = 1
        elif resolution == Resolution.MINUTE:
            bucket_size = 60
        elif resolution == Resolution.HOUR:
            bucket_size = 3600
        elif resolution == Resolution.DAY:
            bucket_size = 86400
        elif resolution == Resolution.WEEK:
            bucket_size = 604800
        elif resolution == Resolution.MONTH:
            bucket_size = 2592000  # 30 days
        else:
            return data  # RAW resolution

        # Group data into buckets
        buckets = defaultdict(list)
        for timestamp, value in data:
            bucket_time = int(timestamp / bucket_size) * bucket_size
            buckets[bucket_time].append((timestamp, value))

        # Aggregate data in each bucket
        result = []
        for bucket_time, points in sorted(buckets.items()):
            if aggregation == Aggregation.MEAN:
                # Only aggregate numeric values
                numeric_values = [v for _, v in points if isinstance(v, (int, float))]
                if numeric_values:
                    result.append((bucket_time, sum(numeric_values) / len(numeric_values)))
                else:
                    result.append((bucket_time, points[-1][1]))  # Use last value if not numeric
            elif aggregation == Aggregation.SUM:
                numeric_values = [v for _, v in points if isinstance(v, (int, float))]
                if numeric_values:
                    result.append((bucket_time, sum(numeric_values)))
                else:
                    result.append((bucket_time, points[-1][1]))
            elif aggregation == Aggregation.MIN:
                numeric_values = [v for _, v in points if isinstance(v, (int, float))]
                if numeric_values:
                    result.append((bucket_time, min(numeric_values)))
                else:
                    result.append((bucket_time, points[-1][1]))
            elif aggregation == Aggregation.MAX:
                numeric_values = [v for _, v in points if isinstance(v, (int, float))]
                if numeric_values:
                    result.append((bucket_time, max(numeric_values)))
                else:
                    result.append((bucket_time, points[-1][1]))
            elif aggregation == Aggregation.FIRST:
                result.append((bucket_time, points[0][1]))
            elif aggregation == Aggregation.LAST:
                result.append((bucket_time, points[-1][1]))
            elif aggregation == Aggregation.COUNT:
                result.append((bucket_time, len(points)))

        return result

    def _compact_series(self, series: str) -> None:
        """Compact a series by rewriting all its files"""
        # Get series directory
        series_dir = self._get_series_dir(series)

        # Process each resolution
        for resolution in Resolution:
            res_dir = self._directory / resolution.value / series_dir

            if not res_dir.exists():
                continue

            # Get all data files
            data_files = sorted(res_dir.glob("*.data"))

            if not data_files:
                continue

            # Load all data
            all_data = []
            for file in data_files:
                try:
                    with open(file, 'rb') as f:
                        chunk_data = pickle.load(f)
                    all_data.extend(chunk_data)
                except Exception as e:
                    logger.error(f"Error loading time series chunk during compaction {file}: {e}")

            if not all_data:
                continue

            # Sort by timestamp
            all_data.sort(key=lambda x: x[0])

            # Remove duplicates
            unique_data = []
            prev_timestamp = None
            for timestamp, value in all_data:
                if timestamp != prev_timestamp:
                    unique_data.append((timestamp, value))
                else:
                    unique_data[-1] = (timestamp, value)
                prev_timestamp = timestamp

            # Delete all existing files
            for file in data_files:
                try:
                    os.remove(file)
                except OSError as e:
                    logger.warning(f"Error removing time series file during compaction {file}: {e}")

            # Determine chunk size based on resolution
            if resolution == Resolution.RAW:
                chunk_size = 3600  # 1 hour chunks for raw data
            else:
                chunk_size = 86400  # 1 day chunks for downsampled data

            # Group data by chunk
            chunks = defaultdict(list)
            for timestamp, value in unique_data:
                chunk_time = int(timestamp / chunk_size) * chunk_size
                chunks[chunk_time].append((timestamp, value))

            # Save each chunk
            for chunk_time, chunk_data in chunks.items():
                file_path = self._get_series_file(series, resolution, chunk_time)
                try:
                    with open(file_path, 'wb') as f:
                        pickle.dump(chunk_data, f)
                except Exception as e:
                    logger.error(f"Error saving time series chunk during compaction {file_path}: {e}")

            # Update metadata
            if resolution == Resolution.RAW and unique_data:
                # Update count
                self._metadata[series]['count'] = len(unique_data)

                # Update timestamps
                self._metadata[series]['first_timestamp'] = unique_data[0][0]
                self._metadata[series]['last_timestamp'] = unique_data[-1][0]

                # Update min/max
                numeric_values = [v for _, v in unique_data if isinstance(v, (int, float))]
                if numeric_values:
                    self._metadata[series]['min_value'] = min(numeric_values)
                    self._metadata[series]['max_value'] = max(numeric_values)