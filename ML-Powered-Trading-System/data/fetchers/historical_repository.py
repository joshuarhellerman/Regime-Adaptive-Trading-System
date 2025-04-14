"""
data/fetchers/historical_repository.py - Historical Data Repository

This module provides access to archived historical market data from various sources,
with versioning and consistent data formats regardless of the original source.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
import os
import glob
import json
import h5py
from pathlib import Path

from core.event_bus import EventBus, create_event
from data.processors.data_normalizer import DataNormalizer
from data.storage.time_series_store import TimeSeriesStore
from utils.logger import get_logger
from data.fetchers.exchange_connector import MarketType, DataType, DataFrequency

class HistoricalRepository:
    """
    Repository for accessing historical market data.

    This class provides access to archived historical data with:
    - Point-in-time accuracy to prevent look-ahead bias
    - Consistent data formats regardless of the original source
    - Version control for research reproducibility
    - Efficient storage and retrieval of large datasets
    """

    def __init__(
        self,
        time_series_store: TimeSeriesStore,
        data_normalizer: DataNormalizer,
        event_bus: EventBus,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the historical repository.

        Args:
            time_series_store: Storage for time series data
            data_normalizer: Normalizes data from different sources
            event_bus: System-wide event bus for events
            config: Configuration for the repository
        """
        self.logger = get_logger(__name__)
        self.time_series_store = time_series_store
        self.normalizer = data_normalizer
        self.event_bus = event_bus
        self.config = config or {}

        # Base path for historical data storage
        self.base_path = Path(self.config.get('base_path', 'data/historical'))

        # Create directory if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Data sources configuration
        self.sources = self.config.get('sources', {})

        # Versioning configuration
        self.versioning_enabled = self.config.get('versioning_enabled', True)
        self.current_version = self.config.get('current_version', 'latest')

        # Mapping for data types to file formats
        self.format_mapping = {
            DataType.OHLCV: 'parquet',
            DataType.QUOTE: 'parquet',
            DataType.TRADE: 'parquet',
            DataType.DEPTH: 'hdf5',
            DataType.FUNDAMENTAL: 'json'
        }

        self.logger.info(f"Historical repository initialized with {len(self.sources)} sources")

    async def get_historical_data(
        self,
        instruments: Union[str, List[str]],
        frequency: Union[str, DataFrequency],
        start_time: Union[datetime, int],
        end_time: Union[datetime, int] = None,
        data_type: DataType = DataType.OHLCV,
        market_type: Optional[MarketType] = None,
        source: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Fetch historical data for instrument(s) from the repository.

        Args:
            instruments: Single instrument or list of instruments
            frequency: Data frequency
            start_time: Start time as datetime or timestamp
            end_time: End time
            data_type: Type of data to fetch
            market_type: Type of financial market
            source: Original data source
            version: Data version (default: latest)
            **kwargs: Additional parameters

        Returns:
            DataFrame with historical data, or dict mapping instruments to DataFrames
        """
        # Convert single instrument to list
        single_instrument = isinstance(instruments, str)
        instruments_list = [instruments] if single_instrument else instruments

        # Convert frequency enum to string if needed
        if isinstance(frequency, DataFrequency):
            frequency = frequency.value

        # Convert market type enum to string if needed
        market_str = market_type.value if market_type else None

        # Set default version if not specified
        version = version or self.current_version

        # Convert timestamps to datetime if needed
        if isinstance(start_time, int):
            start_time = datetime.fromtimestamp(start_time / 1000.0)

        if isinstance(end_time, int):
            end_time = datetime.fromtimestamp(end_time / 1000.0)
        elif end_time is None:
            end_time = datetime.now()

        try:
            # Fetch data for all instruments
            results = {}

            for instrument in instruments_list:
                # Get data path
                data_path = self._get_data_path(
                    instrument,
                    frequency,
                    data_type,
                    market_str,
                    source,
                    version
                )

                if not data_path.exists():
                    self.logger.warning(f"No historical data found for {instrument}")
                    results[instrument] = pd.DataFrame()
                    continue

                # Read data based on format
                data_format = self._get_data_format(data_type)

                if data_format == 'parquet':
                    df = self._read_parquet(data_path)
                elif data_format == 'hdf5':
                    df = self._read_hdf5(data_path, instrument)
                elif data_format == 'csv':
                    df = self._read_csv(data_path)
                elif data_format == 'json':
                    df = self._read_json(data_path)
                else:
                    self.logger.error(f"Unsupported data format: {data_format}")
                    results[instrument] = pd.DataFrame()
                    continue

                # Apply time filter
                if not df.empty:
                    # Ensure datetime index
                    if not isinstance(df.index, pd.DatetimeIndex):
                        if 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            df = df.set_index('timestamp')
                        else:
                            self.logger.warning(f"No timestamp column found for {instrument}")

                    # Filter by date range
                    df = df[(df.index >= start_time) & (df.index <= end_time)]

                results[instrument] = df

            # Return single DataFrame for single instrument
            if single_instrument:
                return results[instruments]

            return results

        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}")
            await self._publish_error_event(
                "error.data.historical",
                str(market_type),
                str(instruments),
                str(e)
            )
            raise

    async def store_historical_data(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        instruments: Union[str, List[str]],
        frequency: Union[str, DataFrequency],
        data_type: DataType = DataType.OHLCV,
        market_type: Optional[MarketType] = None,
        source: Optional[str] = None,
        version: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Store historical data in the repository.

        Args:
            data: DataFrame or dict of DataFrames with historical data
            instruments: Single instrument or list of instruments
            frequency: Data frequency
            data_type: Type of data
            market_type: Type of financial market
            source: Original data source
            version: Data version (default: current version)
            **kwargs: Additional parameters

        Returns:
            True if successful, False otherwise
        """
        # Convert single instrument to list
        single_instrument = isinstance(instruments, str)
        instruments_list = [instruments] if single_instrument else instruments

        # Handle single DataFrame case
        if isinstance(data, pd.DataFrame):
            data_dict = {instruments: data} if single_instrument else {instr: data for instr in instruments_list}
        else:
            data_dict = data

        # Convert frequency enum to string if needed
        if isinstance(frequency, DataFrequency):
            frequency = frequency.value

        # Convert market type enum to string if needed
        market_str = market_type.value if market_type else None

        # Set default version if not specified
        version = version or self.current_version

        try:
            # Store data for all instruments
            for instrument, df in data_dict.items():
                # Skip empty data
                if df is None or df.empty:
                    self.logger.warning(f"Empty data for {instrument}, skipping")
                    continue

                # Get data path
                data_path = self._get_data_path(
                    instrument,
                    frequency,
                    data_type,
                    market_str,
                    source,
                    version,
                    create_dirs=True
                )

                # Write data based on format
                data_format = self._get_data_format(data_type)

                if data_format == 'parquet':
                    self._write_parquet(df, data_path)
                elif data_format == 'hdf5':
                    self._write_hdf5(df, data_path, instrument)
                elif data_format == 'csv':
                    self._write_csv(df, data_path)
                elif data_format == 'json':
                    self._write_json(df, data_path)
                else:
                    self.logger.error(f"Unsupported data format: {data_format}")
                    return False

                self.logger.info(f"Stored historical data for {instrument} at {data_path}")

            return True

        except Exception as e:
            self.logger.error(f"Error storing historical data: {str(e)}")
            return False

    async def list_available_data(
        self,
        market_type: Optional[MarketType] = None,
        data_type: Optional[DataType] = None,
        frequency: Optional[Union[str, DataFrequency]] = None,
        source: Optional[str] = None,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List available historical data in the repository.

        Args:
            market_type: Type of financial market
            data_type: Type of data
            frequency: Data frequency
            source: Original data source
            version: Data version

        Returns:
            Dict with available data information
        """
        # Convert enums to strings if needed
        market_str = market_type.value if market_type else None
        data_type_str = data_type.value if data_type else None
        freq_str = frequency.value if isinstance(frequency, DataFrequency) else frequency

        # Set default version if not specified
        version = version or self.current_version

        # Build path pattern
        path_pattern = self._build_path_pattern(
            market_str,
            data_type_str,
            freq_str,
            source,
            version
        )

        # Find matching files
        matching_files = list(self.base_path.glob(path_pattern))

        # Process results
        results = {
            'version': version,
            'count': len(matching_files),
            'market_types': set(),
            'data_types': set(),
            'frequencies': set(),
            'sources': set(),
            'instruments': set()
        }

        for file_path in matching_files:
            # Extract information from file path
            rel_path = file_path.relative_to(self.base_path)
            parts = list(rel_path.parts)

            if len(parts) >= 5:  # version/market/type/frequency/source/instrument.*
                results['market_types'].add(parts[1])
                results['data_types'].add(parts[2])
                results['frequencies'].add(parts[3])
                results['sources'].add(parts[4])

                # Extract instrument from filename
                instrument = file_path.stem.split('.')[0]
                results['instruments'].add(instrument)

        # Convert sets to lists for JSON serialization
        results['market_types'] = list(results['market_types'])
        results['data_types'] = list(results['data_types'])
        results['frequencies'] = list(results['frequencies'])
        results['sources'] = list(results['sources'])
        results['instruments'] = list(results['instruments'])

        return results

    async def list_versions(self) -> List[str]:
        """
        List available data versions in the repository.

        Returns:
            List of version strings
        """
        # Find all version directories
        version_dirs = [d for d in self.base_path.iterdir() if d.is_dir()]

        # Extract version names
        versions = [d.name for d in version_dirs]

        return versions

    async def create_version(
        self,
        version: str,
        description: Optional[str] = None,
        base_version: Optional[str] = None
    ) -> bool:
        """
        Create a new data version.

        Args:
            version: Version name
            description: Version description
            base_version: Base version to copy from

        Returns:
            True if successful, False otherwise
        """
        if not self.versioning_enabled:
            self.logger.warning("Versioning is disabled, cannot create new version")
            return False

        # Check if version already exists
        version_path = self.base_path / version
        if version_path.exists():
            self.logger.warning(f"Version {version} already exists")
            return False

        try:
            # Create version directory
            version_path.mkdir(parents=True, exist_ok=True)

            # Create metadata file
            metadata = {
                'version': version,
                'description': description,
                'created_at': datetime.now().isoformat(),
                'based_on': base_version
            }

            with open(version_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            # If base version is specified, create symlinks to base version data
            if base_version:
                base_version_path = self.base_path / base_version
                if not base_version_path.exists():
                    self.logger.warning(f"Base version {base_version} not found")
                    return False

                # Create symlinks for all files
                for file_path in base_version_path.glob('**/*'):
                    if file_path.is_file() and file_path.name != 'metadata.json':
                        # Get relative path from base version
                        rel_path = file_path.relative_to(base_version_path)

                        # Create target directory
                        target_dir = version_path / rel_path.parent
                        target_dir.mkdir(parents=True, exist_ok=True)

                        # Create symlink
                        target_path = version_path / rel_path
                        os.symlink(file_path, target_path)

            self.logger.info(f"Created version {version}")
            return True

        except Exception as e:
            self.logger.error(f"Error creating version {version}: {str(e)}")
            return False

    async def set_current_version(self, version: str) -> bool:
        """
        Set the current data version.

        Args:
            version: Version name

        Returns:
            True if successful, False otherwise
        """
        # Check if version exists
        version_path = self.base_path / version
        if not version_path.exists():
            self.logger.warning(f"Version {version} not found")
            return False

        # Update current version
        self.current_version = version

        # Update config
        if 'current_version' in self.config:
            self.config['current_version'] = version

        self.logger.info(f"Set current version to {version}")
        return True

    def _get_data_path(
        self,
        instrument: str,
        frequency: str,
        data_type: DataType,
        market_type: Optional[str],
        source: Optional[str],
        version: str,
        create_dirs: bool = False
    ) -> Path:
        """
        Get the path to a data file.

        Args:
            instrument: Instrument identifier
            frequency: Data frequency
            data_type: Type of data
            market_type: Type of financial market
            source: Original data source
            version: Data version
            create_dirs: Whether to create directories if they don't exist

        Returns:
            Path to data file
        """
        # Use defaults for missing values
        market_type = market_type or 'default'
        source = source or 'default'
        data_type_str = data_type.value

        # Build path
        path = self.base_path / version / market_type / data_type_str / frequency / source

        # Create directories if needed
        if create_dirs:
            path.mkdir(parents=True, exist_ok=True)

        # Get file extension based on data type
        extension = self._get_file_extension(data_type)

        # Return full path
        return path / f"{instrument}{extension}"

    def _build_path_pattern(
        self,
        market_type: Optional[str],
        data_type: Optional[str],
        frequency: Optional[str],
        source: Optional[str],
        version: str
    ) -> str:
        """
        Build a path pattern for glob.

        Args:
            market_type: Type of financial market
            data_type: Type of data
            frequency: Data frequency
            source: Original data source
            version: Data version

        Returns:
            Path pattern for glob
        """
        # Use wildcards for missing values
        market_part = market_type or '*'
        data_type_part = data_type or '*'
        frequency_part = frequency or '*'
        source_part = source or '*'

        # Build pattern
        pattern = f"{version}/{market_part}/{data_type_part}/{frequency_part}/{source_part}/*"

        return pattern

    def _get_data_format(self, data_type: DataType) -> str:
        """
        Get the storage format for a data type.

        Args:
            data_type: Type of data

        Returns:
            Storage format
        """
        return self.format_mapping.get(data_type, 'parquet')

    def _get_file_extension(self, data_type: DataType) -> str:
        """
        Get the file extension for a data type.

        Args:
            data_type: Type of data

        Returns:
            File extension
        """
        format_mapping = {
            'parquet': '.parquet',
            'hdf5': '.h5',
            'csv': '.csv',
            'json': '.json'
        }

        data_format = self._get_data_format(data_type)
        return format_mapping.get(data_format, '.parquet')

    def _read_parquet(self, path: Path) -> pd.DataFrame:
        """Read data from parquet file"""
        try:
            return pd.read_parquet(path)
        except Exception as e:
            self.logger.error(f"Error reading parquet file {path}: {str(e)}")
            return pd.DataFrame()

    def _write_parquet(self, df: pd.DataFrame, path: Path) -> None:
        """Write data to parquet file"""
        df.to_parquet(path)

    def _read_hdf5(self, path: Path, key: str) -> pd.DataFrame:
        """Read data from HDF5 file"""
        try:
            return pd.read_hdf(path, key=key)
        except Exception as e:
            self.logger.error(f"Error reading HDF5 file {path}: {str(e)}")
            return pd.DataFrame()

    def _write_hdf5(self, df: pd.DataFrame, path: Path, key: str) -> None:
        """Write data to HDF5 file"""
        df.to_hdf(path, key=key, mode='w')

    def _read_csv(self, path: Path) -> pd.DataFrame:
        """Read data from CSV file"""
        try:
            return pd.read_csv(path, parse_dates=['timestamp'], index_col='timestamp')
        except Exception as e:
            self.logger.error(f"Error reading CSV file {path}: {str(e)}")
            return pd.DataFrame()

    def _write_csv(self, df: pd.DataFrame, path: Path) -> None:
        """Write data to CSV file"""
        df.to_csv(path)

    def _read_json(self, path: Path) -> pd.DataFrame:
        """Read data from JSON file"""
        try:
            return pd.read_json(path)
        except Exception as e:
            self.logger.error(f"Error reading JSON file {path}: {str(e)}")
            return pd.DataFrame()

    def _write_json(self, df: pd.DataFrame, path: Path) -> None:
        """Write data to JSON file"""
        df.to_json(path)

    async def _publish_error_event(
        self,
        event_type: str,
        market: str,
        instruments: str,
        error: str
    ) -> None:
        """
        Publish an error event to the event bus.

        Args:
            event_type: Type of error event
            market: Market type
            instruments: Instruments
            error: Error message
        """
        try:
            event = create_event(
                event_type,
                {
                    "market": market,
                    "instruments": instruments,
                    "error": error,
                    "component": "historical_repository",
                    "timestamp": datetime.now().timestamp()
                }
            )
            await self.event_bus.publish(event)
        except Exception as e:
            self.logger.error(f"Error publishing error event: {str(e)}")