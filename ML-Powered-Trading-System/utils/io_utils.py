"""
utils/io_utils.py - Input/Output Utilities

This module provides standardized I/O operations for the trading system,
including file handling, serialization/deserialization, and data import/export
with proper error handling and performance considerations.

The module follows the system architecture principles:
- Single Source of Truth: Consistent interfaces for all I/O operations
- Strict Performance Budget: Efficient I/O operations with minimal overhead
- Stateless Where Possible: Functions don't maintain internal state
- Fault Isolation: Robust error handling prevents cascading failures
- Observable System: Comprehensive logging of I/O operations
"""

import os
import json
import pickle
import csv
import logging
import gzip
import hashlib
import shutil
import tempfile
import time
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, BinaryIO, TextIO, Callable, Type
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import contextlib
import io

# Configure logger
logger = logging.getLogger(__name__)

# Type aliases for clarity
PathLike = Union[str, Path]
DataFrameOrSeries = Union[pd.DataFrame, pd.Series]


# File Operations with Error Handling
def ensure_directory(directory: PathLike) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory: Directory path

    Returns:
        Path object of the directory

    Raises:
        IOError: If directory cannot be created
    """
    path = Path(directory)
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception as e:
        logger.error(f"Failed to create directory {directory}: {str(e)}")
        raise IOError(f"Failed to create directory {directory}: {str(e)}")


def safe_write(path: PathLike, data: Union[str, bytes], binary: bool = False) -> bool:
    """
    Safely write data to a file using a temporary file and atomic rename.

    Args:
        path: Path to the file
        data: Data to write (string or bytes)
        binary: Whether to write in binary mode

    Returns:
        True if successful, False otherwise
    """
    path = Path(path)

    # Create directory if it doesn't exist
    ensure_directory(path.parent)

    # Create a temporary file in the same directory
    temp_path = path.with_suffix(f"{path.suffix}.tmp{int(time.time() * 1000)}")

    try:
        mode = 'wb' if binary else 'w'
        with open(temp_path, mode) as f:
            f.write(data)

        # Atomic rename
        temp_path.replace(path)
        return True

    except Exception as e:
        logger.error(f"Failed to write to {path}: {str(e)}")
        # Clean up temp file if it exists
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
        return False


def safe_read(path: PathLike, binary: bool = False) -> Optional[Union[str, bytes]]:
    """
    Safely read data from a file with proper error handling.

    Args:
        path: Path to the file
        binary: Whether to read in binary mode

    Returns:
        File contents or None if file cannot be read
    """
    path = Path(path)

    if not path.exists():
        logger.warning(f"File not found: {path}")
        return None


# Utility Functions for Working with Time Series Store
def export_time_series_to_dataframe(ts_store, series_names: List[str],
                                  start_time: Optional[float] = None,
                                  end_time: Optional[float] = None) -> pd.DataFrame:
    """
    Export data from TimeSeriesStore to a pandas DataFrame.

    Args:
        ts_store: TimeSeriesStore instance
        series_names: List of series names to export
        start_time: Start timestamp (if None, uses earliest available)
        end_time: End timestamp (if None, uses latest available)

    Returns:
        DataFrame with time series data
    """
    # Validate input
    if not series_names:
        logger.warning("No series names provided for export")
        return pd.DataFrame()

    # Get data range if not specified
    if start_time is None or end_time is None:
        ranges = {}
        for series in series_names:
            series_range = ts_store.get_range(series)
            if series_range:
                ranges[series] = series_range

        if not ranges:
            logger.warning("No valid ranges found for any series")
            return pd.DataFrame()

        # Use the widest range across all series
        if start_time is None:
            start_time = min(r[0] for r in ranges.values())

        if end_time is None:
            end_time = max(r[1] for r in ranges.values())

    # Get data for each series
    data = {}
    timestamps = set()

    for series in series_names:
        series_data = ts_store.get(series, start_time, end_time)
        if series_data:
            data[series] = dict(series_data)  # Convert to dict for faster lookups
            timestamps.update(data[series].keys())

    if not timestamps:
        logger.warning("No data found for the specified time range")
        return pd.DataFrame()

    # Create DataFrame
    sorted_timestamps = sorted(timestamps)
    result = pd.DataFrame(index=sorted_timestamps)

    for series in series_names:
        if series in data:
            result[series] = [data[series].get(ts) for ts in sorted_timestamps]

    # Convert timestamp index to datetime
    result.index = pd.to_datetime(result.index, unit='s')
    result.index.name = 'timestamp'

    return result


def import_dataframe_to_time_series(ts_store, df: pd.DataFrame,
                                  timestamp_column: Optional[str] = None) -> Dict[str, int]:
    """
    Import data from a pandas DataFrame to a TimeSeriesStore.

    Args:
        ts_store: TimeSeriesStore instance
        df: DataFrame with time series data
        timestamp_column: Name of timestamp column (if None, uses index)

    Returns:
        Dictionary mapping series names to number of points imported
    """
    # Use index as timestamp if no column specified
    if timestamp_column is None:
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("DataFrame index must be DatetimeIndex when timestamp_column is None")
            return {}

        # Convert datetime index to timestamp
        timestamps = df.index.astype('int64') // 10**9  # nanoseconds to seconds
        data_columns = df.columns
    else:
        # Ensure timestamp column exists
        if timestamp_column not in df.columns:
            logger.error(f"Timestamp column '{timestamp_column}' not found in DataFrame")
            return {}

        # Convert timestamp column to timestamps
        if pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
            timestamps = df[timestamp_column].astype('int64') // 10**9
        else:
            # Assume already numeric
            timestamps = df[timestamp_column]

        data_columns = [col for col in df.columns if col != timestamp_column]

    # Import each column as a separate series
    result = {}

    for column in data_columns:
        count = 0
        for i, (timestamp, value) in enumerate(zip(timestamps, df[column])):
            if pd.notna(value):  # Skip NaN values
                ts_store.store(column, value, timestamp)
                count += 1

        result[column] = count

    return result


# Functionality Specific to Research Environment
def save_research_config(config: Any, directory: PathLike, name: str) -> bool:
    """
    Save a research configuration to a file.

    Args:
        config: Research configuration object
        directory: Directory to save to
        name: Name for the configuration file

    Returns:
        True if successful, False otherwise
    """
    directory = Path(directory)
    ensure_directory(directory)

    path = directory / f"{name}.yaml"

    try:
        # Convert to dict if possible
        if hasattr(config, '__dict__'):
            config_dict = config.__dict__
        elif hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        else:
            config_dict = config

        # Write to YAML file
        return write_yaml(path, config_dict)

    except Exception as e:
        logger.error(f"Failed to save research config: {str(e)}")
        return False


def load_research_config(path: PathLike) -> Optional[Dict[str, Any]]:
    """
    Load a research configuration from a file.

    Args:
        path: Path to the configuration file

    Returns:
        Configuration dictionary or None if loading fails
    """
    return read_yaml(path)


# Exchange Connector Utilities
def export_market_data(market_data: Dict[str, Any], path: PathLike, format: str = 'json') -> bool:
    """
    Export market data from exchange connector to a file.

    Args:
        market_data: Market data dictionary from exchange connector
        path: Path to the output file
        format: Export format ('json', 'csv', 'parquet')

    Returns:
        True if successful, False otherwise
    """
    try:
        if format.lower() == 'json':
            return write_json(path, market_data)

        elif format.lower() == 'csv':
            # Convert to DataFrame first
            if 'data' in market_data and isinstance(market_data['data'], list):
                df = pd.DataFrame(market_data['data'])
                return write_csv(path, df)
            else:
                # Flatten dictionary structure if possible
                flattened = {}
                for key, value in market_data.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            flattened[f"{key}_{subkey}"] = subvalue
                    else:
                        flattened[key] = value

                df = pd.DataFrame([flattened])
                return write_csv(path, df)

        elif format.lower() == 'parquet':
            # Similar to CSV, convert to DataFrame first
            if 'data' in market_data and isinstance(market_data['data'], list):
                df = pd.DataFrame(market_data['data'])
                return write_parquet(path, df)
            else:
                # Flatten dictionary
                flattened = {}
                for key, value in market_data.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            flattened[f"{key}_{subkey}"] = subvalue
                    else:
                        flattened[key] = value

                df = pd.DataFrame([flattened])
                return write_parquet(path, df)
        else:
            raise ValueError(f"Unsupported market data export format: {format}")

    except Exception as e:
        logger.error(f"Failed to export market data: {str(e)}")
        return False


# Performance metrics utilities
def save_performance_metrics(metrics: Dict[str, Any], path: PathLike,
                           format: str = 'json', append: bool = False) -> bool:
    """
    Save performance metrics to a file.

    Args:
        metrics: Performance metrics dictionary
        path: Path to the output file
        format: Export format ('json', 'csv')
        append: Whether to append to existing file

    Returns:
        True if successful, False otherwise
    """
    try:
        path = Path(path)
        ensure_directory(path.parent)

        # Add timestamp if not present
        if 'timestamp' not in metrics:
            metrics = metrics.copy()
            metrics['timestamp'] = time.time()

        if format.lower() == 'json':
            if append and path.exists():
                # Load existing data
                existing_data = read_json(path)

                if existing_data is None:
                    existing_data = []
                elif not isinstance(existing_data, list):
                    existing_data = [existing_data]

                # Append new metrics
                existing_data.append(metrics)

                # Write back
                return write_json(path, existing_data)
            else:
                # Create new file
                return write_json(path, [metrics] if append else metrics)

        elif format.lower() == 'csv':
            # Convert to DataFrame for CSV
            df = pd.DataFrame([metrics])

            if append and path.exists():
                return append_csv(path, df, index=False)
            else:
                return write_csv(path, df, index=False)

        else:
            raise ValueError(f"Unsupported metrics export format: {format}")

    except Exception as e:
        logger.error(f"Failed to save performance metrics: {str(e)}")
        return False


def atomic_file_lock(path: PathLike, timeout: int = 30) -> Optional[contextlib.AbstractContextManager]:
    """
    Create an atomic file lock for coordinating access to a shared resource.

    Args:
        path: Path to the lock file
        timeout: Maximum time to wait for the lock (in seconds)

    Returns:
        Context manager for the lock or None if locking fails
    """
    path = Path(path)
    lock_path = path.with_suffix(f"{path.suffix}.lock")

    class FileLockContext:
        def __init__(self, lock_path, timeout):
            self.lock_path = lock_path
            self.timeout = timeout
            self.acquired = False

        def __enter__(self):
            # Try to create the lock file
            start_time = time.time()

            while time.time() - start_time < self.timeout:
                try:
                    # Try to create the lock file
                    fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(fd)

                    # Write process ID to the lock file
                    with open(self.lock_path, 'w') as f:
                        f.write(str(os.getpid()))

                    self.acquired = True
                    return self
                except FileExistsError:
                    # Lock exists, check if it's stale
                    try:
                        lock_time = get_file_modification_time(self.lock_path)
                        if lock_time and time.time() - lock_time > self.timeout:
                            # Lock is stale, try to remove it
                            os.remove(self.lock_path)
                            continue
                    except:
                        pass

                    # Wait and retry
                    time.sleep(0.1)

            # Timeout expired
            logger.warning(f"Timed out waiting for lock: {self.lock_path}")
            return None

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.acquired:
                try:
                    os.remove(self.lock_path)
                except:
                    logger.warning(f"Failed to remove lock file: {self.lock_path}")

    return FileLockContext(lock_path, timeout)


# Module initialization
def init_io_directories(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Initialize I/O directories from configuration.

    Args:
        config: Configuration dictionary with directory paths

    Returns:
        Dictionary mapping directory names to Path objects
    """
    directories = {}

    # Extract directory configurations
    dir_config = config.get('directories', {})

    # Common directories to initialize
    common_dirs = [
        'data', 'models', 'logs', 'config', 'output', 'cache', 'temp', 'research'
    ]

    # Create each directory
    for dir_name in common_dirs:
        if dir_name in dir_config:
            path = dir_config[dir_name]
        else:
            # Use default path
            path = Path(dir_name)

        # Create directory if it doesn't exist
        directories[dir_name] = ensure_directory(path)

    # Create subdirectories if specified
    for dir_name, subdir_config in dir_config.items():
        if isinstance(subdir_config, dict) and 'subdirs' in subdir_config:
            base_dir = directories.get(dir_name, ensure_directory(subdir_config.get('path', dir_name)))

            for subdir in subdir_config['subdirs']:
                subdir_path = base_dir / subdir
                directories[f"{dir_name}_{subdir}"] = ensure_directory(subdir_path)

    logger.info(f"Initialized {len(directories)} I/O directories")
    return directories


# File cleanup utilities
def cleanup_old_files(directory: PathLike, pattern: str = '*',
                     max_age_days: int = 30, dry_run: bool = False) -> int:
    """
    Clean up old files in a directory.

    Args:
        directory: Directory to clean up
        pattern: File pattern to match
        max_age_days: Maximum age of files to keep (in days)
        dry_run: If True, only report files that would be deleted

    Returns:
        Number of files deleted
    """
    directory = Path(directory)

    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return 0

    count = 0
    current_time = time.time()
    max_age_seconds = max_age_days * 86400

    try:
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime

                if file_age > max_age_seconds:
                    if dry_run:
                        logger.info(f"Would delete: {file_path} (age: {file_age/86400:.1f} days)")
                    else:
                        try:
                            file_path.unlink()
                            logger.debug(f"Deleted: {file_path} (age: {file_age/86400:.1f} days)")
                            count += 1
                        except Exception as e:
                            logger.warning(f"Failed to delete {file_path}: {str(e)}")

        return count

    except Exception as e:
        logger.error(f"Error during file cleanup: {str(e)}")
        return count


def archive_files(source_dir: PathLike, archive_name: PathLike,
                 pattern: str = '*', compression: str = 'zip',
                 delete_originals: bool = False) -> bool:
    """
    Archive files in a directory.

    Args:
        source_dir: Directory containing files to archive
        archive_name: Path to the output archive file
        pattern: File pattern to match
        compression: Compression format ('zip', 'tar', 'gztar', or 'bztar')
        delete_originals: Whether to delete original files after archiving

    Returns:
        True if successful, False otherwise
    """
    source_dir = Path(source_dir)
    archive_name = Path(archive_name)

    if not source_dir.exists():
        logger.warning(f"Source directory not found: {source_dir}")
        return False

    # Create directory for the archive if needed
    ensure_directory(archive_name.parent)

    try:
        import shutil

        # Get matching files
        files = list(source_dir.glob(pattern))

        if not files:
            logger.warning(f"No files found matching pattern '{pattern}' in {source_dir}")
            return False

        # Convert source_dir to absolute path for proper relative paths in archive
        source_dir = source_dir.absolute()

        # Create archive
        archive_path = shutil.make_archive(
            str(archive_name.with_suffix('')),
            format=compression,
            root_dir=source_dir,
            base_dir=None
        )

        # Delete original files if requested
        if delete_originals:
            for file_path in files:
                try:
                    file_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete {file_path}: {str(e)}")

        logger.info(f"Created archive {archive_path} with {len(files)} files")
        return True

    except Exception as e:
        logger.error(f"Failed to create archive: {str(e)}")
        return False


# File synchronization utilities
def sync_directories(source: PathLike, target: PathLike, pattern: str = '*',
                    delete_extras: bool = False, preserve_times: bool = True) -> Tuple[int, int, int]:
    """
    Synchronize files between two directories.

    Args:
        source: Source directory
        target: Target directory
        pattern: File pattern to match
        delete_extras: Whether to delete files in target that aren't in source
        preserve_times: Whether to preserve modification times

    Returns:
        Tuple of (files_copied, files_updated, files_deleted)
    """
    source = Path(source)
    target = Path(target)

    if not source.exists():
        logger.warning(f"Source directory not found: {source}")
        return 0, 0, 0

    # Create target directory if it doesn't exist
    ensure_directory(target)

    copied = 0
    updated = 0
    deleted = 0

    try:
        # Get source files
        source_files = list(source.glob(pattern))
        source_paths = {p.relative_to(source) for p in source_files}

        # Copy/update files
        for rel_path in source_paths:
            source_file = source / rel_path
            target_file = target / rel_path

            # Create parent directories if needed
            ensure_directory(target_file.parent)

            # Check if file exists and needs updating
            if not target_file.exists():
                # Copy new file
                shutil.copy2(source_file, target_file) if preserve_times else shutil.copy(source_file, target_file)
                copied += 1
            else:
                # Compare modification times and sizes
                source_mtime = source_file.stat().st_mtime
                target_mtime = target_file.stat().st_mtime
                source_size = source_file.stat().st_size
                target_size = target_file.stat().st_size

                if source_mtime > target_mtime or source_size != target_size:
                    # Update existing file
                    shutil.copy2(source_file, target_file) if preserve_times else shutil.copy(source_file, target_file)
                    updated += 1

        # Delete extra files if requested
        if delete_extras:
            target_files = list(target.glob(pattern))
            target_paths = {p.relative_to(target) for p in target_files}

            extra_paths = target_paths - source_paths

            for rel_path in extra_paths:
                target_file = target / rel_path
                try:
                    target_file.unlink()
                    deleted += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {target_file}: {str(e)}")

        logger.info(f"Synchronized directories: {copied} copied, {updated} updated, {deleted} deleted")
        return copied, updated, deleted

    except Exception as e:
        logger.error(f"Failed to synchronize directories: {str(e)}")
        return copied, updated, deleted


# Binary data utilities
def compress_data(data: bytes, method: str = 'gzip', level: int = 9) -> bytes:
    """
    Compress binary data.

    Args:
        data: Binary data to compress
        method: Compression method ('gzip', 'bz2', or 'lzma')
        level: Compression level (1-9, where 9 is highest)

    Returns:
        Compressed binary data
    """
    try:
        if method == 'gzip':
            import gzip
            return gzip.compress(data, level)
        elif method == 'bz2':
            import bz2
            return bz2.compress(data, level)
        elif method == 'lzma':
            import lzma
            return lzma.compress(data, preset=level)
        else:
            raise ValueError(f"Unsupported compression method: {method}")
    except Exception as e:
        logger.error(f"Failed to compress data: {str(e)}")
        return data  # Return original data on error


def decompress_data(data: bytes, method: str = 'gzip') -> bytes:
    """
    Decompress binary data.

    Args:
        data: Compressed binary data
        method: Compression method ('gzip', 'bz2', or 'lzma')

    Returns:
        Decompressed binary data
    """
    try:
        if method == 'gzip':
            import gzip
            return gzip.decompress(data)
        elif method == 'bz2':
            import bz2
            return bz2.decompress(data)
        elif method == 'lzma':
            import lzma
            return lzma.decompress(data)
        else:
            raise ValueError(f"Unsupported compression method: {method}")
    except Exception as e:
        logger.error(f"Failed to decompress data: {str(e)}")
        return data  # Return original data on error


# Data validation utilities
def validate_csv_file(file_path: PathLike, required_columns: Optional[List[str]] = None,
                     min_rows: int = 1, encoding: str = 'utf-8') -> Tuple[bool, str]:
    """
    Validate a CSV file for basic formatting and required columns.

    Args:
        file_path: Path to the CSV file
        required_columns: List of required column names
        min_rows: Minimum number of data rows required
        encoding: File encoding

    Returns:
        Tuple of (is_valid, error_message)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return False, f"File not found: {file_path}"

    try:
        # Try to read the CSV file
        with open(file_path, 'r', encoding=encoding) as f:
            # Read header
            header_line = f.readline().strip()
            if not header_line:
                return False, "Empty file or no header row"

            header = [col.strip() for col in header_line.split(',')]

            # Check required columns
            if required_columns:
                missing_columns = [col for col in required_columns if col not in header]
                if missing_columns:
                    return False, f"Missing required columns: {', '.join(missing_columns)}"

            # Check number of rows
            row_count = 0
            for _ in f:
                row_count += 1

            if row_count < min_rows:
                return False, f"Not enough data rows. Found {row_count}, required {min_rows}"

        return True, "File is valid"

    except Exception as e:
        return False, f"Error validating CSV file: {str(e)}"


# This module provides additional I/O utilities tailored to the ML-powered trading system
# following the architectural principles of modularity, fault isolation, and performance.


    try:
        mode = 'rb' if binary else 'r'
        with open(path, mode) as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read from {path}: {str(e)}")
        return None


def file_exists(path: PathLike) -> bool:
    """
    Check if a file exists.

    Args:
        path: Path to the file

    Returns:
        True if file exists, False otherwise
    """
    return Path(path).exists()


def get_file_size(path: PathLike) -> Optional[int]:
    """
    Get file size in bytes.

    Args:
        path: Path to the file

    Returns:
        File size in bytes or None if file doesn't exist
    """
    path = Path(path)

    if not path.exists():
        return None

    try:
        return path.stat().st_size
    except Exception as e:
        logger.error(f"Failed to get size of {path}: {str(e)}")
        return None


def get_file_modification_time(path: PathLike) -> Optional[float]:
    """
    Get file modification time as a timestamp.

    Args:
        path: Path to the file

    Returns:
        Modification time as a timestamp or None if file doesn't exist
    """
    path = Path(path)

    if not path.exists():
        return None

    try:
        return path.stat().st_mtime
    except Exception as e:
        logger.error(f"Failed to get modification time of {path}: {str(e)}")
        return None


def calculate_file_hash(path: PathLike, algorithm: str = 'sha256') -> Optional[str]:
    """
    Calculate the hash of a file.

    Args:
        path: Path to the file
        algorithm: Hash algorithm to use ('md5', 'sha1', 'sha256', etc.)

    Returns:
        Hash string or None if file cannot be read
    """
    path = Path(path)

    if not path.exists():
        logger.warning(f"File not found: {path}")
        return None

    try:
        hash_func = getattr(hashlib, algorithm)()

        with open(path, 'rb') as f:
            # Read in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)

        return hash_func.hexdigest()
    except Exception as e:
        logger.error(f"Failed to calculate hash of {path}: {str(e)}")
        return None


def list_files(directory: PathLike, pattern: str = '*', recursive: bool = False) -> List[Path]:
    """
    List files in a directory matching a pattern.

    Args:
        directory: Directory path
        pattern: Glob pattern for matching files
        recursive: Whether to search recursively

    Returns:
        List of Path objects for matching files
    """
    directory = Path(directory)

    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return []

    try:
        if recursive:
            return list(directory.glob(f"**/{pattern}"))
        else:
            return list(directory.glob(pattern))
    except Exception as e:
        logger.error(f"Failed to list files in {directory}: {str(e)}")
        return []


@contextlib.contextmanager
def safe_open(path: PathLike, mode: str = 'r') -> Union[TextIO, BinaryIO]:
    """
    Safely open a file with proper error handling and cleanup.

    Args:
        path: Path to the file
        mode: File open mode

    Yields:
        Open file object

    Raises:
        IOError: If file cannot be opened
    """
    file = None
    try:
        # Create directory if it doesn't exist and we're writing
        if 'w' in mode or 'a' in mode:
            ensure_directory(Path(path).parent)

        file = open(path, mode)
        yield file
    except Exception as e:
        logger.error(f"Error accessing file {path}: {str(e)}")
        raise IOError(f"Error accessing file {path}: {str(e)}")
    finally:
        if file is not None:
            file.close()


# JSON Operations
def write_json(path: PathLike, data: Any, indent: int = 4, ensure_ascii: bool = False) -> bool:
    """
    Write data to a JSON file.

    Args:
        path: Path to the JSON file
        data: Data to write
        indent: JSON indentation level
        ensure_ascii: Whether to escape non-ASCII characters

    Returns:
        True if successful, False otherwise
    """
    try:
        json_str = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii, default=_json_serializer)
        return safe_write(path, json_str)
    except Exception as e:
        logger.error(f"Failed to write JSON to {path}: {str(e)}")
        return False


def read_json(path: PathLike) -> Optional[Any]:
    """
    Read data from a JSON file.

    Args:
        path: Path to the JSON file

    Returns:
        Parsed JSON data or None if file cannot be read
    """
    content = safe_read(path)

    if content is None:
        return None

    try:
        return json.loads(content)
    except Exception as e:
        logger.error(f"Failed to parse JSON from {path}: {str(e)}")
        return None


def _json_serializer(obj: Any) -> Any:
    """
    Custom JSON serializer for handling non-serializable types.

    Args:
        obj: Object to serialize

    Returns:
        Serializable representation of the object

    Raises:
        TypeError: If object cannot be serialized
    """
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        raise TypeError(f"Type {type(obj)} not serializable")


# YAML Operations
def write_yaml(path: PathLike, data: Any) -> bool:
    """
    Write data to a YAML file.

    Args:
        path: Path to the YAML file
        data: Data to write

    Returns:
        True if successful, False otherwise
    """
    try:
        yaml_str = yaml.dump(data, default_flow_style=False)
        return safe_write(path, yaml_str)
    except Exception as e:
        logger.error(f"Failed to write YAML to {path}: {str(e)}")
        return False


def read_yaml(path: PathLike) -> Optional[Any]:
    """
    Read data from a YAML file.

    Args:
        path: Path to the YAML file

    Returns:
        Parsed YAML data or None if file cannot be read
    """
    content = safe_read(path)

    if content is None:
        return None

    try:
        return yaml.safe_load(content)
    except Exception as e:
        logger.error(f"Failed to parse YAML from {path}: {str(e)}")
        return None


# Pickle Operations
def write_pickle(path: PathLike, data: Any, compress: bool = False) -> bool:
    """
    Write data to a pickle file.

    Args:
        path: Path to the pickle file
        data: Data to write
        compress: Whether to compress the pickle file

    Returns:
        True if successful, False otherwise
    """
    try:
        if compress:
            with gzip.open(path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    except Exception as e:
        logger.error(f"Failed to write pickle to {path}: {str(e)}")
        return False


def read_pickle(path: PathLike, decompress: bool = False) -> Optional[Any]:
    """
    Read data from a pickle file.

    Args:
        path: Path to the pickle file
        decompress: Whether to decompress the pickle file

    Returns:
        Unpickled data or None if file cannot be read
    """
    path = Path(path)

    if not path.exists():
        logger.warning(f"File not found: {path}")
        return None

    try:
        if decompress:
            with gzip.open(path, 'rb') as f:
                return pickle.load(f)
        else:
            with open(path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to read pickle from {path}: {str(e)}")
        return None


# CSV Operations
def write_csv(path: PathLike, data: Union[pd.DataFrame, List[List[Any]], List[Dict[str, Any]]],
              columns: Optional[List[str]] = None, index: bool = False) -> bool:
    """
    Write data to a CSV file.

    Args:
        path: Path to the CSV file
        data: Data to write (DataFrame, list of lists, or list of dicts)
        columns: Column names (required for list of lists)
        index: Whether to include DataFrame index

    Returns:
        True if successful, False otherwise
    """
    try:
        if isinstance(data, pd.DataFrame):
            data.to_csv(path, index=index)
        elif isinstance(data, list):
            if not data:
                return safe_write(path, "")

            if isinstance(data[0], dict):
                # List of dictionaries
                if columns is None:
                    columns = list(data[0].keys())

                with open(path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(data)
            else:
                # List of lists
                if columns is None:
                    raise ValueError("Column names must be provided for list of lists")

                with open(path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(columns)
                    writer.writerows(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        return True
    except Exception as e:
        logger.error(f"Failed to write CSV to {path}: {str(e)}")
        return False


def read_csv(path: PathLike, **kwargs) -> Optional[pd.DataFrame]:
    """
    Read data from a CSV file into a DataFrame.

    Args:
        path: Path to the CSV file
        **kwargs: Additional arguments for pd.read_csv

    Returns:
        DataFrame or None if file cannot be read
    """
    path = Path(path)

    if not path.exists():
        logger.warning(f"File not found: {path}")
        return None

    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        logger.error(f"Failed to read CSV from {path}: {str(e)}")
        return None


def append_csv(path: PathLike, data: Union[pd.DataFrame, List[List[Any]], List[Dict[str, Any]]],
              columns: Optional[List[str]] = None, index: bool = False) -> bool:
    """
    Append data to a CSV file.

    Args:
        path: Path to the CSV file
        data: Data to append (DataFrame, list of lists, or list of dicts)
        columns: Column names (for new file or list of lists)
        index: Whether to include DataFrame index

    Returns:
        True if successful, False otherwise
    """
    path = Path(path)
    file_exists = path.exists()

    try:
        if isinstance(data, pd.DataFrame):
            data.to_csv(path, mode='a', header=not file_exists, index=index)
        elif isinstance(data, list):
            if not data:
                return True  # Nothing to append

            mode = 'a' if file_exists else 'w'
            write_header = not file_exists

            if isinstance(data[0], dict):
                # List of dictionaries
                if columns is None and not file_exists:
                    columns = list(data[0].keys())

                with open(path, mode, newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=columns)
                    if write_header:
                        writer.writeheader()
                    writer.writerows(data)
            else:
                # List of lists
                if columns is None and not file_exists:
                    raise ValueError("Column names must be provided for new file with list of lists")

                with open(path, mode, newline='') as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow(columns)
                    writer.writerows(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        return True
    except Exception as e:
        logger.error(f"Failed to append CSV to {path}: {str(e)}")
        return False


# Parquet Operations
def write_parquet(path: PathLike, data: pd.DataFrame, compression: str = 'snappy') -> bool:
    """
    Write DataFrame to a Parquet file.

    Args:
        path: Path to the Parquet file
        data: DataFrame to write
        compression: Compression codec to use

    Returns:
        True if successful, False otherwise
    """
    try:
        data.to_parquet(path, compression=compression)
        return True
    except Exception as e:
        logger.error(f"Failed to write Parquet to {path}: {str(e)}")
        return False


def read_parquet(path: PathLike, columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """
    Read DataFrame from a Parquet file.

    Args:
        path: Path to the Parquet file
        columns: List of columns to read

    Returns:
        DataFrame or None if file cannot be read
    """
    path = Path(path)

    if not path.exists():
        logger.warning(f"File not found: {path}")
        return None

    try:
        return pd.read_parquet(path, columns=columns)
    except Exception as e:
        logger.error(f"Failed to read Parquet from {path}: {str(e)}")
        return None


# HDF5 Operations
def write_hdf(path: PathLike, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
             key: str = 'data', mode: str = 'w', complevel: int = 9) -> bool:
    """
    Write DataFrame to an HDF5 file.

    Args:
        path: Path to the HDF5 file
        data: DataFrame or dict of DataFrames to write
        key: Key for the DataFrame (ignored if data is a dict)
        mode: File open mode ('w' or 'a')
        complevel: Compression level (0-9)

    Returns:
        True if successful, False otherwise
    """
    try:
        if isinstance(data, pd.DataFrame):
            data.to_hdf(path, key=key, mode=mode, complevel=complevel)
        elif isinstance(data, dict):
            for k, df in data.items():
                df.to_hdf(path, key=k, mode='a' if k != list(data.keys())[0] or mode == 'a' else 'w', complevel=complevel)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        return True
    except Exception as e:
        logger.error(f"Failed to write HDF5 to {path}: {str(e)}")
        return False


def read_hdf(path: PathLike, key: Optional[str] = None) -> Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]:
    """
    Read DataFrame from an HDF5 file.

    Args:
        path: Path to the HDF5 file
        key: Key for the DataFrame (if None, returns all keys)

    Returns:
        DataFrame, dict of DataFrames, or None if file cannot be read
    """
    path = Path(path)

    if not path.exists():
        logger.warning(f"File not found: {path}")
        return None

    try:
        if key is not None:
            return pd.read_hdf(path, key=key)
        else:
            import tables
            with tables.open_file(path, mode='r') as f:
                # Get all group names under root
                group_paths = ['/' + node._v_name for node in f.root._f_iter_nodes() if isinstance(node, tables.Group)]

            # Read each group
            result = {}
            for group_path in group_paths:
                key = group_path[1:]  # Remove leading '/'
                result[key] = pd.read_hdf(path, key=key)

            return result
    except Exception as e:
        logger.error(f"Failed to read HDF5 from {path}: {str(e)}")
        return None


# Numpy Operations
def write_numpy(path: PathLike, data: Union[np.ndarray, Dict[str, np.ndarray]], compressed: bool = True) -> bool:
    """
    Write NumPy array to a file.

    Args:
        path: Path to the NumPy file
        data: NumPy array or dict of arrays
        compressed: Whether to use compressed format

    Returns:
        True if successful, False otherwise
    """
    try:
        if compressed:
            if isinstance(data, dict):
                np.savez_compressed(path, **data)
            else:
                np.savez_compressed(path, arr=data)
        else:
            if isinstance(data, dict):
                np.savez(path, **data)
            else:
                np.save(path, data)

        return True
    except Exception as e:
        logger.error(f"Failed to write NumPy array to {path}: {str(e)}")
        return False


def read_numpy(path: PathLike) -> Optional[Union[np.ndarray, Dict[str, np.ndarray]]]:
    """
    Read NumPy array from a file.

    Args:
        path: Path to the NumPy file

    Returns:
        NumPy array, dict of arrays, or None if file cannot be read
    """
    path = Path(path)

    if not path.exists():
        logger.warning(f"File not found: {path}")
        return None

    try:
        # Check file extension to determine format
        if path.suffix == '.npz':
            with np.load(path, allow_pickle=True) as data:
                # For .npz files, return a dict of arrays
                return {key: data[key] for key in data.files}
        else:
            # For .npy files, return a single array
            return np.load(path, allow_pickle=True)
    except Exception as e:
        logger.error(f"Failed to read NumPy array from {path}: {str(e)}")
        return None


# DataFrame Operations
def split_dataframe(df: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
    """
    Split a DataFrame into chunks for processing large datasets.

    Args:
        df: DataFrame to split
        chunk_size: Number of rows per chunk

    Returns:
        List of DataFrame chunks
    """
    return [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]


def parallel_apply(df: pd.DataFrame, func: Callable, axis: int = 1,
                  n_jobs: int = -1, **kwargs) -> pd.Series:
    """
    Apply a function to a DataFrame in parallel.

    Args:
        df: DataFrame to process
        func: Function to apply
        axis: Axis along which to apply the function
        n_jobs: Number of jobs to run in parallel (-1 for all cores)
        **kwargs: Additional arguments for the function

    Returns:
        Series with the results
    """
    if n_jobs == -1:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()

    # Split DataFrame into chunks
    chunks = np.array_split(df, n_jobs)

    # Process each chunk in parallel
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(
            lambda chunk: chunk.apply(func, axis=axis, **kwargs),
            chunks
        ))

    # Combine results
    return pd.concat(results)


# Time Series Data Operations
def write_time_series(path: PathLike, data: DataFrameOrSeries,
                     format: str = 'csv', compress: bool = False) -> bool:
    """
    Write time series data to a file in the specified format.

    Args:
        path: Path to the file
        data: DataFrame or Series with time series data
        format: File format ('csv', 'parquet', 'hdf', 'pickle')
        compress: Whether to compress the file

    Returns:
        True if successful, False otherwise
    """
    # Ensure data has a name if it's a Series
    if isinstance(data, pd.Series) and data.name is None:
        data = data.copy()
        data.name = 'value'

    try:
        path = Path(path)

        if compress and not format == 'parquet':  # Parquet already has compression
            path = path.with_suffix(f"{path.suffix}.gz")

        if format == 'csv':
            if compress:
                with gzip.open(path, 'wt') as f:
                    data.to_csv(f)
            else:
                data.to_csv(path)
        elif format == 'parquet':
            if isinstance(data, pd.Series):
                data = data.to_frame()
            data.to_parquet(path)
        elif format == 'hdf':
            data.to_hdf(path, key='data')
        elif format == 'pickle':
            if compress:
                with gzip.open(path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return True
    except Exception as e:
        logger.error(f"Failed to write time series to {path}: {str(e)}")
        return False


def read_time_series(path: PathLike,
                    format: Optional[str] = None,
                    decompress: bool = False) -> Optional[DataFrameOrSeries]:
    """
    Read time series data from a file.

    Args:
        path: Path to the file
        format: File format (if None, inferred from extension)
        decompress: Whether to decompress the file

    Returns:
        DataFrame, Series, or None if file cannot be read
    """
    path = Path(path)

    if not path.exists():
        logger.warning(f"File not found: {path}")
        return None

    try:
        # Infer format from extension if not provided
        if format is None:
            suffix = path.suffix.lower()
            if suffix == '.csv' or suffix == '.gz' and path.stem.endswith('.csv'):
                format = 'csv'
            elif suffix == '.parquet' or suffix == '.pq':
                format = 'parquet'
            elif suffix == '.h5' or suffix == '.hdf' or suffix == '.hdf5':
                format = 'hdf'
            elif suffix == '.pkl' or suffix == '.pickle':
                format = 'pickle'
            else:
                raise ValueError(f"Cannot infer format from file extension: {suffix}")

        # Check if file is compressed
        is_gzip = path.suffix.lower() == '.gz'
        decompress = decompress or is_gzip

        if format == 'csv':
            if decompress:
                with gzip.open(path, 'rt') as f:
                    return pd.read_csv(f, index_col=0, parse_dates=True)
            else:
                return pd.read_csv(path, index_col=0, parse_dates=True)
        elif format == 'parquet':
            return pd.read_parquet(path)
        elif format == 'hdf':
            return pd.read_hdf(path, key='data')
        elif format == 'pickle':
            if decompress:
                with gzip.open(path, 'rb') as f:
                    return pickle.load(f)
            else:
                with open(path, 'rb') as f:
                    return pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
    except Exception as e:
        logger.error(f"Failed to read time series from {path}: {str(e)}")
        return None


# Model and Research Data Operations
def save_model(model: Any, path: PathLike, save_format: str = 'pickle',
              additional_info: Optional[Dict[str, Any]] = None) -> bool:
    """
    Save a model to a file.

    Args:
        model: Model to save
        path: Path to the model file
        save_format: Format to use ('pickle', 'joblib', or 'custom')
        additional_info: Additional information to save with the model

    Returns:
        True if successful, False otherwise
    """
    path = Path(path)

    # Create directory if it doesn't exist
    ensure_directory(path.parent)

    try:
        if save_format == 'pickle':
            # Include additional info with the model
            if additional_info:
                data = {'model': model, 'info': additional_info}
            else:
                data = model

            with open(path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        elif save_format == 'joblib':
            import joblib

            # Include additional info with the model
            if additional_info:
                data = {'model': model, 'info': additional_info}
            else:
                data = model

            joblib.dump(data, path)

        elif save_format == 'custom':
            # Custom format for specialized models (e.g., TensorFlow, PyTorch)
            if hasattr(model, 'save'):
                # Save model using its own save method
                model.save(path)

                # Save additional info separately if provided
                if additional_info:
                    info_path = path.with_suffix('.info.json')
                    write_json(info_path, additional_info)
            else:
                raise ValueError(f"Model does not have a 'save' method for custom format")
        else:
            raise ValueError(f"Unsupported save format: {save_format}")

        logger.info(f"Model saved to {path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save model to {path}: {str(e)}")
        return False


def load_model(path: PathLike, load_format: str = 'pickle', custom_loader: Optional[Callable] = None) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """
    Load a model from a file.

    Args:
        path: Path to the model file
        load_format: Format to use ('pickle', 'joblib', or 'custom')
        custom_loader: Custom function to load the model (for 'custom' format)

    Returns:
        Tuple of (model, additional_info) or (None, None) if loading fails
    """
    path = Path(path)

    if not path.exists():
        logger.warning(f"Model file not found: {path}")
        return None, None

    try:
        if load_format == 'pickle':
            with open(path, 'rb') as f:
                data = pickle.load(f)

            # Extract model and info if saved as a dict
            if isinstance(data, dict) and 'model' in data and 'info' in data:
                return data['model'], data['info']
            else:
                return data, None

        elif load_format == 'joblib':
            import joblib
            data = joblib.load(path)

            # Extract model and info if saved as a dict
            if isinstance(data, dict) and 'model' in data and 'info' in data:
                return data['model'], data['info']
            else:
                return data, None

        elif load_format == 'custom':
            if custom_loader is not None:
                # Use custom loader function
                model = custom_loader(path)

                # Check for additional info
                info_path = path.with_suffix('.info.json')
                info = read_json(info_path) if info_path.exists() else None

                return model, info
            else:
                raise ValueError("Custom loader function must be provided for 'custom' format")
        else:
            raise ValueError(f"Unsupported load format: {load_format}")

    except Exception as e:
        logger.error(f"Failed to load model from {path}: {str(e)}")
        return None, None


def save_research_result(result: Any, directory: PathLike, name: str,
                        timestamp: Optional[float] = None) -> bool:
    """
    Save a research result to a directory with metadata.

    Args:
        result: Research result to save
        directory: Directory to save to
        name: Name for the result
        timestamp: Optional timestamp (defaults to current time)

    Returns:
        True if successful, False otherwise
    """
    directory = Path(directory)

    # Create directory if it doesn't exist
    ensure_directory(directory)

    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = time.time()

    # Format timestamp for filename
    time_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")

    try:
        # Create result directory
        result_dir = directory / f"{name}_{time_str}"
        ensure_directory(result_dir)

        # Save metadata
        metadata = {
            'name': name,
            'timestamp': timestamp,
            'created_at': datetime.now().isoformat(),
            'type': type(result).__name__
        }

        write_json(result_dir / 'metadata.json', metadata)

        # Save result based on type
        if hasattr(result, 'to_dict'):
            # For objects with to_dict method (like pandas DataFrame)
            result_dict = result.to_dict()
            write_json(result_dir / 'result.json', result_dict)

        elif isinstance(result, pd.DataFrame):
            # For DataFrame
            write_parquet(result_dir / 'result.parquet', result)

        elif isinstance(result, dict):
            # For dictionary
            # Check if values are DataFrames
            has_dataframes = any(isinstance(v, pd.DataFrame) for v in result.values())

            if has_dataframes:
                # Save each DataFrame separately
                for key, value in result.items():
                    if isinstance(value, pd.DataFrame):
                        write_parquet(result_dir / f"{key}.parquet", value)
                    else:
                        write_json(result_dir / f"{key}.json", value)
            else:
                # Save as JSON
                write_json(result_dir / 'result.json', result)
        else:
            # For other types, use pickle
            write_pickle(result_dir / 'result.pkl', result)

        logger.info(f"Research result saved to {result_dir}")
        return True

    except Exception as e:
        logger.error(f"Failed to save research result: {str(e)}")
        return False


def load_research_result(path: PathLike) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """
    Load a research result from a directory.

    Args:
        path: Path to the research result directory

    Returns:
        Tuple of (result, metadata) or (None, None) if loading fails
    """
    path = Path(path)

    if not path.exists():
        logger.warning(f"Research result directory not found: {path}")
        return None, None

    try:
        # Load metadata
        metadata = read_json(path / 'metadata.json')

        if metadata is None:
            logger.warning(f"Metadata not found in {path}")
            return None, None

        # Determine result type and load
        result = None

        # Try loading from various formats
        if (path / 'result.parquet').exists():
            result = read_parquet(path / 'result.parquet')
        elif (path / 'result.json').exists():
            result = read_json(path / 'result.json')
        elif (path / 'result.pkl').exists():
            result = read_pickle(path / 'result.pkl')
        else:
            # Check for multiple files (dict of DataFrames case)
            parquet_files = list(path.glob('*.parquet'))
            json_files = list(path.glob('*.json'))

            if parquet_files and json_files:
                # Load as dictionary
                result = {}

                # Load parquet files
                for file in parquet_files:
                    if file.stem != 'result':  # Skip result.parquet if it exists
                        result[file.stem] = read_parquet(file)

                # Load JSON files
                for file in json_files:
                    if file.stem != 'metadata':  # Skip metadata.json
                        result[file.stem] = read_json(file)

        return result, metadata

    except Exception as e:
        logger.error(f"Failed to load research result from {path}: {str(e)}")
        return None, None


# Data Exchange with External Systems
def export_data(data: Any, format: str, path: PathLike) -> bool:
    """
    Export data to a file in the specified format for external systems.

    Args:
        data: Data to export
        format: Export format ('csv', 'json', 'xml', 'excel', etc.)
        path: Path to the output file

    Returns:
        True if successful, False otherwise
    """
    path = Path(path)

    try:
        if format.lower() == 'csv':
            if isinstance(data, pd.DataFrame):
                data.to_csv(path, index=False)
            elif isinstance(data, (list, dict)):
                if isinstance(data, dict):
                    data = [data]  # Convert single dict to list

                if isinstance(data[0], dict):
                    # List of dictionaries
                    pd.DataFrame(data).to_csv(path, index=False)
                else:
                    # List of lists
                    with open(path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(data)
            else:
                raise ValueError(f"Unsupported data type for CSV export: {type(data)}")

        elif format.lower() == 'json':
            if isinstance(data, pd.DataFrame):
                data.to_json(path, orient='records')
            else:
                write_json(path, data)

        elif format.lower() == 'xml':
            import xml.etree.ElementTree as ET
            from xml.dom import minidom

            if isinstance(data, pd.DataFrame):
                # Convert DataFrame to list of dictionaries
                records = data.to_dict(orient='records')

                # Create XML structure
                root = ET.Element('data')
                for record in records:
                    item = ET.SubElement(root, 'item')
                    for key, value in record.items():
                        if pd.notna(value):  # Skip NaN values
                            field = ET.SubElement(item, str(key))
                            field.text = str(value)

                # Pretty print XML
                xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")

                # Write to file
                with open(path, 'w') as f:
                    f.write(xml_str)
            else:
                raise ValueError(f"XML export only supports DataFrame, got {type(data)}")

        elif format.lower() == 'excel':
            if isinstance(data, pd.DataFrame):
                data.to_excel(path, index=False)
            elif isinstance(data, dict) and all(isinstance(v, pd.DataFrame) for v in data.values()):
                # Dictionary of DataFrames (multiple sheets)
                with pd.ExcelWriter(path) as writer:
                    for sheet_name, df in data.items():
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                raise ValueError(f"Excel export requires DataFrame or dict of DataFrames, got {type(data)}")

        elif format.lower() == 'html':
            if isinstance(data, pd.DataFrame):
                html = data.to_html()
                with open(path, 'w') as f:
                    f.write(html)
            else:
                raise ValueError(f"HTML export only supports DataFrame, got {type(data)}")

        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Data exported to {path} in {format} format")
        return True

    except Exception as e:
        logger.error(f"Failed to export data to {path}: {str(e)}")
        return False


def import_data(path: PathLike, format: Optional[str] = None) -> Optional[Any]:
    """
    Import data from a file.

    Args:
        path: Path to the input file
        format: Import format (if None, inferred from file extension)

    Returns:
        Imported data or None if import fails
    """
    path = Path(path)

    if not path.exists():
        logger.warning(f"File not found: {path}")
        return None

    try:
        # Infer format from extension if not provided
        if format is None:
            format = path.suffix.lower().lstrip('.')

            # Handle special cases
            if format == 'xls' or format == 'xlsx':
                format = 'excel'
            elif format == 'htm':
                format = 'html'

        if format.lower() == 'csv':
            return pd.read_csv(path)

        elif format.lower() == 'json':
            try:
                # Try first as DataFrame
                return pd.read_json(path)
            except:
                # Fall back to raw JSON
                return read_json(path)

        elif format.lower() == 'xml':
            import xml.etree.ElementTree as ET

            tree = ET.parse(path)
            root = tree.getroot()

            # Try to convert to a list of dictionaries
            items = []
            for item in root.findall('./item'):
                data = {}
                for child in item:
                    data[child.tag] = child.text
                items.append(data)

            # Convert to DataFrame if possible
            if items:
                return pd.DataFrame(items)
            else:
                # Return raw XML
                return ET.tostring(root).decode()

        elif format.lower() == 'excel':
            # Check if multiple sheets
            import pandas as pd
            xls = pd.ExcelFile(path)

            if len(xls.sheet_names) == 1:
                # Single sheet
                return pd.read_excel(path)
            else:
                # Multiple sheets - return dictionary of DataFrames
                return {sheet: pd.read_excel(path, sheet_name=sheet) for sheet in xls.sheet_names}

        elif format.lower() == 'html':
            return pd.read_html(path)[0]  # Return first table found

        elif format.lower() == 'parquet':
            return read_parquet(path)

        elif format.lower() in ['pickle', 'pkl']:
            return read_pickle(path)

        else:
            raise ValueError(f"Unsupported import format: {format}")

    except Exception as e:
        logger.error(f"Failed to import data from {path}: {str(e)}")
        return None


# Streaming I/O Operations
def stream_write(path: PathLike, data_generator: Iterator[Any],
                format: str = 'csv', chunk_size: int = 1000) -> bool:
    """
    Write data to a file in streaming mode to handle large datasets.

    Args:
        path: Path to the output file
        data_generator: Generator yielding data chunks
        format: File format ('csv', 'json', etc.)
        chunk_size: Number of records per chunk

    Returns:
        True if successful, False otherwise
    """
    path = Path(path)

    # Create directory if it doesn't exist
    ensure_directory(path.parent)

    try:
        if format.lower() == 'csv':
            # Write CSV in chunks
            header_written = False

            for chunk in data_generator:
                # Skip empty chunks
                if len(chunk) == 0:
                    continue

                # Convert to DataFrame if needed
                if not isinstance(chunk, pd.DataFrame):
                    chunk = pd.DataFrame(chunk)

                # Write header only once
                mode = 'a' if header_written else 'w'
                header = not header_written

                # Write chunk
                chunk.to_csv(path, mode=mode, header=header, index=False)
                header_written = True

        elif format.lower() == 'json':
            # For JSON, we'll collect all data and write at once
            # This is not memory-efficient but maintains proper JSON structure
            all_data = []

            for chunk in data_generator:
                if isinstance(chunk, list):
                    all_data.extend(chunk)
                else:
                    all_data.append(chunk)

            write_json(path, all_data)

        elif format.lower() == 'parquet':
            # For Parquet, convert all chunks to one DataFrame
            chunks = []

            for chunk in data_generator:
                if not isinstance(chunk, pd.DataFrame):
                    chunk = pd.DataFrame(chunk)
                chunks.append(chunk)

            # Combine chunks
            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                write_parquet(path, df)
            else:
                # Create empty DataFrame
                write_parquet(path, pd.DataFrame())

        else:
            raise ValueError(f"Unsupported streaming format: {format}")

        logger.info(f"Data streamed to {path} in {format} format")
        return True

    except Exception as e:
        logger.error(f"Failed to stream data to {path}: {str(e)}")
        return False


def stream_read(path: PathLike, format: Optional[str] = None,
               chunk_size: int = 1000) -> Optional[Iterator[Any]]:
    """
    Read data from a file in streaming mode to handle large datasets.

    Args:
        path: Path to the input file
        format: File format (if None, inferred from file extension)
        chunk_size: Number of records per chunk

    Returns:
        Generator yielding data chunks or None if reading fails
    """
    path = Path(path)

    if not path.exists():
        logger.warning(f"File not found: {path}")
        return None

    try:
        # Infer format from extension if not provided
        if format is None:
            format = path.suffix.lower().lstrip('.')

        if format.lower() == 'csv':
            # Use pandas chunk reading
            return pd.read_csv(path, chunksize=chunk_size)

        elif format.lower() == 'json':
            # For JSON, we need to read the entire file
            # but can yield chunks of records
            data = read_json(path)

            if data is None:
                return None

            if isinstance(data, list):
                # Yield chunks of the list
                for i in range(0, len(data), chunk_size):
                    yield data[i:i+chunk_size]
            else:
                # Single object, yield as one chunk
                yield data

        elif format.lower() == 'parquet':
            # Read using pyarrow for chunking
            import pyarrow.parquet as pq

            table = pq.read_table(path)
            total_rows = table.num_rows

            # Yield chunks
            for i in range(0, total_rows, chunk_size):
                end = min(i + chunk_size, total_rows)
                chunk = table.slice(i, end - i).to_pandas()
                yield chunk

        else:
            raise ValueError(f"Unsupported streaming format: {format}")

    except Exception as e:
        logger.error(f"Failed to stream data from {path}: {str(e)}")
        return None