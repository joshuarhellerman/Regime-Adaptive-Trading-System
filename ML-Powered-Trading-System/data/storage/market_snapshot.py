"""
market_snapshot.py - Point-in-time market state storage

This module provides a consistent snapshot of the market state at specific
points in time. It allows for storing and retrieving comprehensive market
state information, including order books, recent trades, and derived metrics.
It's used for creating consistent views of the market for strategy execution
and analysis.
"""

import copy
import json
import logging
import threading
import time
from collections import defaultdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar('T')


class MarketSnapshotError(Exception):
    """Base exception for market snapshot errors"""
    pass


class SnapshotType(Enum):
    """Types of market snapshots that can be stored"""
    ORDERBOOK = "orderbook"          # Order book state
    TRADES = "trades"                # Recent trades
    TICKERS = "tickers"              # Current ticker data
    DERIVED_METRICS = "metrics"      # Calculated market metrics
    QUOTES = "quotes"                # Best bid/ask quotes
    FULL = "full"                    # Complete market state


class MarketSnapshotManager:
    """
    Manages point-in-time snapshots of market state.
    Provides thread-safe storage and retrieval of market state information.
    """

    def __init__(self, directory: str, max_memory_snapshots: int = 100):
        """
        Initialize the market snapshot manager.

        Args:
            directory: Directory where snapshot files will be stored
            max_memory_snapshots: Maximum number of snapshots to keep in memory
        """
        self._directory = Path(directory)
        self._max_memory_snapshots = max_memory_snapshots
        self._snapshots: Dict[str, Dict[str, Any]] = {}  # id -> snapshot
        self._lookup_indices: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        self._lock = threading.RLock()
        self._latest_snapshot_id: Optional[str] = None

        # Ensure directories exist
        self._directory.mkdir(parents=True, exist_ok=True)
        for snapshot_type in SnapshotType:
            (self._directory / snapshot_type.value).mkdir(exist_ok=True)

        logger.info(f"Market snapshot manager initialized at {directory}")

    def create_snapshot(self,
                       snapshot_type: SnapshotType,
                       data: Dict[str, Any],
                       timestamp: Optional[float] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new market snapshot.

        Args:
            snapshot_type: Type of snapshot being created
            data: Snapshot data
            timestamp: Optional timestamp (defaults to current time)
            metadata: Optional metadata for the snapshot

        Returns:
            Snapshot ID
        """
        if timestamp is None:
            timestamp = time.time()

        # Generate a unique snapshot ID
        snapshot_id = f"{snapshot_type.value}_{int(timestamp * 1000)}"

        # Create snapshot with metadata
        metadata = metadata or {}
        metadata.update({
            "timestamp": timestamp,
            "created_at": time.time(),
            "type": snapshot_type.value
        })

        snapshot = {
            "id": snapshot_id,
            "metadata": metadata,
            "data": data
        }

        with self._lock:
            # Store in memory if we have space
            if len(self._snapshots) < self._max_memory_snapshots:
                self._snapshots[snapshot_id] = snapshot

            # Update indices for quick lookup
            self._update_indices(snapshot_id, snapshot)

            # Update latest snapshot reference
            self._latest_snapshot_id = snapshot_id

            # Save to disk
            self._persist_snapshot(snapshot_id, snapshot)

        return snapshot_id

    def get_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific snapshot by ID.

        Args:
            snapshot_id: ID of the snapshot to retrieve

        Returns:
            Snapshot data or None if not found
        """
        with self._lock:
            # Try to get from memory first
            if snapshot_id in self._snapshots:
                return copy.deepcopy(self._snapshots[snapshot_id])

            # Otherwise load from disk
            return self._load_snapshot(snapshot_id)

    def get_latest_snapshot(self,
                           snapshot_type: Optional[SnapshotType] = None) -> Optional[Dict[str, Any]]:
        """
        Get the most recent snapshot, optionally of a specific type.

        Args:
            snapshot_type: Optional type filter

        Returns:
            Most recent snapshot or None if not found
        """
        with self._lock:
            if snapshot_type is None:
                # Return the latest snapshot regardless of type
                if self._latest_snapshot_id:
                    return self.get_snapshot(self._latest_snapshot_id)
                return None

            # Find the latest snapshot of the specified type
            type_dir = self._directory / snapshot_type.value
            if not type_dir.exists():
                return None

            # Get all snapshot files of this type, sorted by timestamp (newest first)
            snapshot_files = sorted(type_dir.glob("*.snapshot"), reverse=True)
            if not snapshot_files:
                return None

            # Load the most recent one
            snapshot_id = snapshot_files[0].stem
            return self.get_snapshot(snapshot_id)

    def find_snapshots(self,
                      snapshot_type: Optional[SnapshotType] = None,
                      start_time: Optional[float] = None,
                      end_time: Optional[float] = None,
                      filters: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Find snapshots matching the specified criteria.

        Args:
            snapshot_type: Optional type filter
            start_time: Optional start timestamp
            end_time: Optional end timestamp
            filters: Optional metadata filters

        Returns:
            List of snapshot IDs matching the criteria
        """
        with self._lock:
            # Start with all snapshots if no type specified
            if snapshot_type is None:
                # We need to scan all type directories
                result_ids = set()
                for s_type in SnapshotType:
                    result_ids.update(self._find_snapshots_by_type(s_type, start_time, end_time, filters))
                return sorted(list(result_ids))
            else:
                # Find snapshots of the specified type
                return sorted(list(self._find_snapshots_by_type(snapshot_type, start_time, end_time, filters)))

    def _find_snapshots_by_type(self,
                              snapshot_type: SnapshotType,
                              start_time: Optional[float] = None,
                              end_time: Optional[float] = None,
                              filters: Optional[Dict[str, Any]] = None) -> Set[str]:
        """
        Helper method to find snapshots of a specific type.

        Args:
            snapshot_type: Type of snapshots to find
            start_time: Optional start timestamp
            end_time: Optional end timestamp
            filters: Optional metadata filters

        Returns:
            Set of matching snapshot IDs
        """
        # Start with all snapshots of this type
        type_dir = self._directory / snapshot_type.value
        if not type_dir.exists():
            return set()

        # Get all snapshot files of this type
        result_ids = set()

        # First check our index for time range
        if start_time is not None or end_time is not None:
            for snapshot_id in self._lookup_indices.get(snapshot_type.value, {}).get("timestamp", set()):
                # Check if we have metadata for this snapshot
                snapshot = self._snapshots.get(snapshot_id)
                if snapshot:
                    timestamp = snapshot["metadata"].get("timestamp")
                    if timestamp:
                        # Check time range
                        if start_time is not None and timestamp < start_time:
                            continue
                        if end_time is not None and timestamp > end_time:
                            continue
                        result_ids.add(snapshot_id)
                else:
                    # Need to load from disk
                    snapshot = self._load_snapshot(snapshot_id)
                    if snapshot:
                        timestamp = snapshot["metadata"].get("timestamp")
                        if timestamp:
                            # Check time range
                            if start_time is not None and timestamp < start_time:
                                continue
                            if end_time is not None and timestamp > end_time:
                                continue
                            result_ids.add(snapshot_id)
        else:
            # No time filtering, get all snapshots of this type
            result_ids = self._lookup_indices.get(snapshot_type.value, {}).get("timestamp", set()).copy()

            # If index is empty, scan disk
            if not result_ids:
                for file in type_dir.glob("*.snapshot"):
                    result_ids.add(file.stem)

        # Apply additional filters if specified
        if filters:
            filtered_ids = set()
            for snapshot_id in result_ids:
                # Check if we have metadata for this snapshot
                snapshot = self._snapshots.get(snapshot_id)
                if snapshot is None:
                    # Need to load from disk
                    snapshot = self._load_snapshot(snapshot_id)

                if snapshot:
                    # Check if all filters match
                    match = True
                    for key, value in filters.items():
                        if key not in snapshot["metadata"] or snapshot["metadata"][key] != value:
                            match = False
                            break

                    if match:
                        filtered_ids.add(snapshot_id)

            return filtered_ids

        return result_ids

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """
        Delete a specific snapshot.

        Args:
            snapshot_id: ID of the snapshot to delete

        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            # Remove from memory if present
            snapshot = self._snapshots.pop(snapshot_id, None)

            # Remove from indices
            if snapshot:
                self._remove_from_indices(snapshot_id, snapshot)

            # Update latest snapshot reference if needed
            if self._latest_snapshot_id == snapshot_id:
                self._latest_snapshot_id = None

                # Find the new latest snapshot
                latest_time = 0
                for s_id, s in self._snapshots.items():
                    timestamp = s["metadata"].get("timestamp", 0)
                    if timestamp > latest_time:
                        latest_time = timestamp
                        self._latest_snapshot_id = s_id

            # Try to delete from disk
            try:
                # First figure out the snapshot type from the ID
                snapshot_type = None
                for s_type in SnapshotType:
                    if snapshot_id.startswith(f"{s_type.value}_"):
                        snapshot_type = s_type
                        break

                if snapshot_type:
                    file_path = self._directory / snapshot_type.value / f"{snapshot_id}.snapshot"
                    if file_path.exists():
                        file_path.unlink()
                        return True

                # If we couldn't determine the type, try each directory
                for s_type in SnapshotType:
                    file_path = self._directory / s_type.value / f"{snapshot_id}.snapshot"
                    if file_path.exists():
                        file_path.unlink()
                        return True

                return False
            except Exception as e:
                logger.error(f"Error deleting snapshot {snapshot_id}: {e}")
                return False

    def clean_old_snapshots(self,
                           max_age: float,
                           snapshot_type: Optional[SnapshotType] = None) -> int:
        """
        Remove snapshots older than the specified age.

        Args:
            max_age: Maximum age in seconds
            snapshot_type: Optional type filter

        Returns:
            Number of snapshots removed
        """
        current_time = time.time()
        min_timestamp = current_time - max_age

        with self._lock:
            # Find snapshots older than the specified age
            old_snapshot_ids = self.find_snapshots(
                snapshot_type=snapshot_type,
                end_time=min_timestamp
            )

            # Delete each one
            count = 0
            for snapshot_id in old_snapshot_ids:
                if self.delete_snapshot(snapshot_id):
                    count += 1

            return count

    def merge_snapshots(self,
                       snapshot_ids: List[str],
                       new_type: SnapshotType = SnapshotType.FULL) -> Optional[str]:
        """
        Merge multiple snapshots into a new combined snapshot.

        Args:
            snapshot_ids: List of snapshot IDs to merge
            new_type: Type of the resulting snapshot

        Returns:
            New snapshot ID or None if unsuccessful
        """
        if not snapshot_ids:
            return None

        with self._lock:
            snapshots = []
            for snapshot_id in snapshot_ids:
                snapshot = self.get_snapshot(snapshot_id)
                if snapshot:
                    snapshots.append(snapshot)

            if not snapshots:
                return None

            # Use the most recent timestamp
            latest_timestamp = max(s["metadata"].get("timestamp", 0) for s in snapshots)

            # Merge data dictionaries
            merged_data = {}
            for snapshot in snapshots:
                merged_data.update(snapshot["data"])

            # Merge metadata
            merged_metadata = {
                "source_snapshots": snapshot_ids,
                "merged_at": time.time()
            }

            # Create new snapshot
            return self.create_snapshot(
                snapshot_type=new_type,
                data=merged_data,
                timestamp=latest_timestamp,
                metadata=merged_metadata
            )

    def snapshot_diff(self,
                     snapshot_id1: str,
                     snapshot_id2: str) -> Dict[str, Any]:
        """
        Calculate the difference between two snapshots.

        Args:
            snapshot_id1: First snapshot ID
            snapshot_id2: Second snapshot ID

        Returns:
            Dictionary containing the differences
        """
        with self._lock:
            # Load both snapshots
            snapshot1 = self.get_snapshot(snapshot_id1)
            snapshot2 = self.get_snapshot(snapshot_id2)

            if not snapshot1 or not snapshot2:
                raise MarketSnapshotError("One or both snapshots not found")

            # Calculate differences
            diff = {
                "metadata": {
                    "snapshot1": snapshot_id1,
                    "snapshot2": snapshot_id2,
                    "timestamp1": snapshot1["metadata"].get("timestamp"),
                    "timestamp2": snapshot2["metadata"].get("timestamp"),
                    "time_diff": snapshot2["metadata"].get("timestamp", 0) -
                                 snapshot1["metadata"].get("timestamp", 0)
                },
                "data_diff": self._calculate_data_diff(snapshot1["data"], snapshot2["data"])
            }

            return diff

    def _calculate_data_diff(self,
                           data1: Dict[str, Any],
                           data2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the difference between two data dictionaries.

        Args:
            data1: First data dictionary
            data2: Second data dictionary

        Returns:
            Dictionary of differences
        """
        diff = {}

        # Get all keys
        all_keys = set(data1.keys()) | set(data2.keys())

        for key in all_keys:
            # Key only in data1
            if key in data1 and key not in data2:
                diff[key] = {"action": "removed", "value": data1[key]}
            # Key only in data2
            elif key in data2 and key not in data1:
                diff[key] = {"action": "added", "value": data2[key]}
            # Key in both
            else:
                if data1[key] != data2[key]:
                    # Special handling for order books
                    if isinstance(data1[key], dict) and isinstance(data2[key], dict) and key.endswith("_book"):
                        # Order book diff
                        book_diff = self._calculate_orderbook_diff(data1[key], data2[key])
                        if book_diff:
                            diff[key] = {"action": "changed", "diff": book_diff}
                    else:
                        # Regular diff
                        diff[key] = {
                            "action": "changed",
                            "from": data1[key],
                            "to": data2[key]
                        }

        return diff

    def _calculate_orderbook_diff(self,
                                book1: Dict[str, Any],
                                book2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the difference between two order books.

        Args:
            book1: First order book
            book2: Second order book

        Returns:
            Order book difference
        """
        diff = {}

        # Check bids
        if "bids" in book1 and "bids" in book2:
            bids1 = {float(price): qty for price, qty in book1["bids"]}
            bids2 = {float(price): qty for price, qty in book2["bids"]}

            bids_diff = []
            all_prices = sorted(set(bids1.keys()) | set(bids2.keys()), reverse=True)

            for price in all_prices:
                if price in bids1 and price in bids2:
                    if bids1[price] != bids2[price]:
                        bids_diff.append({
                            "price": price,
                            "from": bids1[price],
                            "to": bids2[price],
                            "diff": bids2[price] - bids1[price]
                        })
                elif price in bids1:
                    bids_diff.append({
                        "price": price,
                        "from": bids1[price],
                        "to": 0,
                        "diff": -bids1[price]
                    })
                else:
                    bids_diff.append({
                        "price": price,
                        "from": 0,
                        "to": bids2[price],
                        "diff": bids2[price]
                    })

            if bids_diff:
                diff["bids"] = bids_diff

        # Check asks
        if "asks" in book1 and "asks" in book2:
            asks1 = {float(price): qty for price, qty in book1["asks"]}
            asks2 = {float(price): qty for price, qty in book2["asks"]}

            asks_diff = []
            all_prices = sorted(set(asks1.keys()) | set(asks2.keys()))

            for price in all_prices:
                if price in asks1 and price in asks2:
                    if asks1[price] != asks2[price]:
                        asks_diff.append({
                            "price": price,
                            "from": asks1[price],
                            "to": asks2[price],
                            "diff": asks2[price] - asks1[price]
                        })
                elif price in asks1:
                    asks_diff.append({
                        "price": price,
                        "from": asks1[price],
                        "to": 0,
                        "diff": -asks1[price]
                    })
                else:
                    asks_diff.append({
                        "price": price,
                        "from": 0,
                        "to": asks2[price],
                        "diff": asks2[price]
                    })

            if asks_diff:
                diff["asks"] = asks_diff

        return diff

    def _persist_snapshot(self,
                        snapshot_id: str,
                        snapshot: Dict[str, Any]) -> bool:
        """
        Persist a snapshot to disk.

        Args:
            snapshot_id: Snapshot ID
            snapshot: Snapshot data

        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine the snapshot type
            snapshot_type = snapshot["metadata"].get("type")
            if not snapshot_type:
                # Try to extract from ID
                for s_type in SnapshotType:
                    if snapshot_id.startswith(f"{s_type.value}_"):
                        snapshot_type = s_type.value
                        break

                if not snapshot_type:
                    # Default to FULL
                    snapshot_type = SnapshotType.FULL.value

            # Create file path
            file_path = self._directory / snapshot_type / f"{snapshot_id}.snapshot"

            # Create parent directory if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to file
            with open(file_path, 'w') as f:
                json.dump(snapshot, f)

            return True
        except Exception as e:
            logger.error(f"Error persisting snapshot {snapshot_id}: {e}")
            return False

    def _load_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a snapshot from disk.

        Args:
            snapshot_id: Snapshot ID

        Returns:
            Snapshot data or None if not found
        """
        try:
            # Try each snapshot type directory
            for snapshot_type in SnapshotType:
                file_path = self._directory / snapshot_type.value / f"{snapshot_id}.snapshot"
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        snapshot = json.load(f)

                    # Update indices for this snapshot
                    self._update_indices(snapshot_id, snapshot)

                    # Store in memory if we have space
                    if len(self._snapshots) < self._max_memory_snapshots:
                        self._snapshots[snapshot_id] = snapshot

                    return snapshot

            return None
        except Exception as e:
            logger.error(f"Error loading snapshot {snapshot_id}: {e}")
            return None

    def _update_indices(self, snapshot_id: str, snapshot: Dict[str, Any]) -> None:
        """
        Update lookup indices for a snapshot.

        Args:
            snapshot_id: Snapshot ID
            snapshot: Snapshot data
        """
        # Get snapshot type
        snapshot_type = snapshot["metadata"].get("type")
        if not snapshot_type:
            # Try to extract from ID
            for s_type in SnapshotType:
                if snapshot_id.startswith(f"{s_type.value}_"):
                    snapshot_type = s_type.value
                    break

            if not snapshot_type:
                # Default to FULL
                snapshot_type = SnapshotType.FULL.value

        # Add to type index
        self._lookup_indices[snapshot_type]["timestamp"].add(snapshot_id)

        # Index metadata fields
        for key, value in snapshot["metadata"].items():
            # Only index scalar values
            if isinstance(value, (str, int, float, bool)):
                self._lookup_indices[snapshot_type][f"metadata.{key}"].add(snapshot_id)

    def _remove_from_indices(self, snapshot_id: str, snapshot: Dict[str, Any]) -> None:
        """
        Remove a snapshot from lookup indices.

        Args:
            snapshot_id: Snapshot ID
            snapshot: Snapshot data
        """
        # Get snapshot type
        snapshot_type = snapshot["metadata"].get("type")
        if not snapshot_type:
            # Try to extract from ID
            for s_type in SnapshotType:
                if snapshot_id.startswith(f"{s_type.value}_"):
                    snapshot_type = s_type.value
                    break

            if not snapshot_type:
                # Default to FULL
                snapshot_type = SnapshotType.FULL.value

        # Remove from all indices for this type
        for index_key, index_values in self._lookup_indices.get(snapshot_type, {}).items():
            index_values.discard(snapshot_id)


class MarketSnapshotService:
    """
    High-level service for working with market snapshots.
    Provides methods for common snapshot operations.
    """

    def __init__(self, snapshot_manager: MarketSnapshotManager):
        """
        Initialize the market snapshot service.

        Args:
            snapshot_manager: Market snapshot manager instance
        """
        self._manager = snapshot_manager
        self._lock = threading.RLock()

    def create_orderbook_snapshot(self,
                                symbol: str,
                                bids: List[Tuple[float, float]],
                                asks: List[Tuple[float, float]],
                                timestamp: Optional[float] = None,
                                exchange: Optional[str] = None) -> str:
        """
        Create an order book snapshot.

        Args:
            symbol: Trading symbol
            bids: List of (price, quantity) tuples for bids
            asks: List of (price, quantity) tuples for asks
            timestamp: Optional timestamp
            exchange: Optional exchange name

        Returns:
            Snapshot ID
        """
        data = {
            f"{symbol}_book": {
                "bids": bids,
                "asks": asks,
                "symbol": symbol,
                "exchange": exchange
            }
        }

        metadata = {
            "symbol": symbol
        }

        if exchange:
            metadata["exchange"] = exchange

        return self._manager.create_snapshot(
            snapshot_type=SnapshotType.ORDERBOOK,
            data=data,
            timestamp=timestamp,
            metadata=metadata
        )

    def create_ticker_snapshot(self,
                             tickers: Dict[str, Dict[str, Any]],
                             timestamp: Optional[float] = None,
                             exchange: Optional[str] = None) -> str:
        """
        Create a ticker snapshot with multiple symbols.

        Args:
            tickers: Dictionary of symbol -> ticker data
            timestamp: Optional timestamp
            exchange: Optional exchange name

        Returns:
            Snapshot ID
        """
        data = {}
        symbols = []

        for symbol, ticker_data in tickers.items():
            data[f"{symbol}_ticker"] = ticker_data
            symbols.append(symbol)

        metadata = {
            "symbols": ",".join(symbols)
        }

        if exchange:
            metadata["exchange"] = exchange

        return self._manager.create_snapshot(
            snapshot_type=SnapshotType.TICKERS,
            data=data,
            timestamp=timestamp,
            metadata=metadata
        )

    def create_trades_snapshot(self,
                             symbol: str,
                             trades: List[Dict[str, Any]],
                             timestamp: Optional[float] = None,
                             exchange: Optional[str] = None) -> str:
        """
        Create a recent trades snapshot.

        Args:
            symbol: Trading symbol
            trades: List of recent trades
            timestamp: Optional timestamp
            exchange: Optional exchange name

        Returns:
            Snapshot ID
        """
        data = {
            f"{symbol}_trades": trades
        }

        metadata = {
            "symbol": symbol,
            "trade_count": len(trades)
        }

        if exchange:
            metadata["exchange"] = exchange

        return self._manager.create_snapshot(
            snapshot_type=SnapshotType.TRADES,
            data=data,
            timestamp=timestamp,
            metadata=metadata
        )

    def create_derived_metrics_snapshot(self,
                                      metrics: Dict[str, Dict[str, Any]],
                                      timestamp: Optional[float] = None) -> str:
        """
        Create a derived metrics snapshot.

        Args:
            metrics: Dictionary of metric name -> metric data
            timestamp: Optional timestamp

        Returns:
            Snapshot ID
        """
        metadata = {
            "metric_count": len(metrics)
        }

        return self._manager.create_snapshot(
            snapshot_type=SnapshotType.DERIVED_METRICS,
            data=metrics,
            timestamp=timestamp,
            metadata=metadata
        )

    def create_full_market_snapshot(self,
                                  data: Dict[str, Any],
                                  symbols: List[str],
                                  timestamp: Optional[float] = None,
                                  exchange: Optional[str] = None) -> str:
        """
        Create a full market snapshot.

        Args:
            data: Complete market data
            symbols: List of symbols included
            timestamp: Optional timestamp
            exchange: Optional exchange name

        Returns:
            Snapshot ID
        """
        metadata = {
            "symbols": ",".join(symbols),
            "full_snapshot": True
        }

        if exchange:
            metadata["exchange"] = exchange

        return self._manager.create_snapshot(
            snapshot_type=SnapshotType.FULL,
            data=data,
            timestamp=timestamp,
            metadata=metadata
        )

    def get_latest_orderbook(self,
                           symbol: str,
                           exchange: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the latest order book for a symbol.

        Args:
            symbol: Trading symbol
            exchange: Optional exchange name

        Returns:
            Order book data or None if not found
        """
        # Create filter
        filters = {"symbol": symbol}
        if exchange:
            filters["exchange"] = exchange

        # Find snapshots
        snapshot_ids = self._manager.find_snapshots(
            snapshot_type=SnapshotType.ORDERBOOK,
            filters=filters
        )

        if not snapshot_ids:
            return None

        # Get the most recent one
        snapshot = self._manager.get_snapshot(snapshot_ids[-1])
        if not snapshot:
            return None

        return snapshot["data"].get(f"{symbol}_book")

    def get_latest_ticker(self,
                        symbol: str,
                        exchange: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the latest ticker for a symbol.

        Args:
            symbol: Trading symbol
            exchange: Optional exchange name

        Returns:
            Ticker data or None if not found
        """
        # Find snapshots containing this symbol
        filters = {}
        if exchange:
            filters["exchange"] = exchange

        # Find snapshots
        snapshot_ids = self._manager.find_snapshots(
            snapshot_type=SnapshotType.TICKERS,
            filters=filters
        )

        # Filter to snapshots containing this symbol
        matching_snapshots = []
        for snapshot_id in snapshot_ids:
            snapshot = self._manager.get_snapshot(snapshot_id)
            if snapshot and f"{symbol}_ticker" in snapshot["data"]:
                matching_snapshots.append(snapshot)

        if not matching_snapshots:
            return None

        # Sort by timestamp and get the most recent
        matching_snapshots.sort(key=lambda s: s["metadata"].get("timestamp", 0))
        latest_snapshot = matching_snapshots[-1]

        return latest_snapshot["data"].get(f"{symbol}_ticker")

    def get_recent_trades(self,
                         symbol: str,
                         limit: int = 100,
                         exchange: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """
        Get recent trades for a symbol.

        Args:
            symbol: Trading symbol
            limit: Maximum number of trades to return
            exchange: Optional exchange name

        Returns:
            List of recent trades or None if not found
        """
        # Create filter
        filters = {"symbol": symbol}
        if exchange:
            filters["exchange"] = exchange

        # Find snapshots
        snapshot_ids = self._manager.find_snapshots(
            snapshot_type=SnapshotType.TRADES,
            filters=filters
        )

        if not snapshot_ids:
            return None

        # Get the most recent one
        snapshot = self._manager.get_snapshot(snapshot_ids[-1])
        if not snapshot:
            return None

        trades = snapshot["data"].get(f"{symbol}_trades", [])
        return trades[:limit] if len(trades) > limit else trades

    def get_derived_metric(self,
                          metric_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest value of a derived metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Metric data or None if not found
        """
        # Find the latest metrics snapshot
        snapshot = self._manager.get_latest_snapshot(SnapshotType.DERIVED_METRICS)
        if not snapshot:
            return None

        return snapshot["data"].get(metric_name)

    def create_market_state_snapshot(self,
                                   symbols: List[str],
                                   exchange: Optional[str] = None) -> Optional[str]:
        """
        Create a full market state snapshot by combining the latest data
        for the given symbols.

        Args:
            symbols: List of symbols to include
            exchange: Optional exchange name

        Returns:
            Snapshot ID or None if unsuccessful
        """
        with self._lock:
            # Collect all the latest data for these symbols
            timestamp = time.time()
            full_data = {}

            for symbol in symbols:
                # Get latest order book
                orderbook = self.get_latest_orderbook(symbol, exchange)
                if orderbook:
                    full_data[f"{symbol}_book"] = orderbook

                # Get latest ticker
                ticker = self.get_latest_ticker(symbol, exchange)
                if ticker:
                    full_data[f"{symbol}_ticker"] = ticker

                # Get recent trades
                trades = self.get_recent_trades(symbol, exchange=exchange)
                if trades:
                    full_data[f"{symbol}_trades"] = trades

            # Get derived metrics
            metrics_snapshot = self._manager.get_latest_snapshot(SnapshotType.DERIVED_METRICS)
            if metrics_snapshot:
                for key, value in metrics_snapshot["data"].items():
                    full_data[key] = value

            # Create the full snapshot
            if full_data:
                return self.create_full_market_snapshot(
                    data=full_data,
                    symbols=symbols,
                    timestamp=timestamp,
                    exchange=exchange
                )

            return None

    def compare_orderbooks(self,
                         symbol: str,
                         start_time: float,
                         end_time: float,
                         exchange: Optional[str] = None) -> Dict[str, Any]:
        """
        Compare order books for a symbol between two time points.

        Args:
            symbol: Trading symbol
            start_time: Start timestamp
            end_time: End timestamp
            exchange: Optional exchange name

        Returns:
            Comparison results
        """
        # Create filter
        filters = {"symbol": symbol}
        if exchange:
            filters["exchange"] = exchange

        # Find snapshots in the time range
        snapshot_ids = self._manager.find_snapshots(
            snapshot_type=SnapshotType.ORDERBOOK,
            start_time=start_time,
            end_time=end_time,
            filters=filters
        )

        if len(snapshot_ids) < 2:
            raise MarketSnapshotError("Not enough snapshots for comparison")

        # Get the first and last snapshots
        first_snapshot = self._manager.get_snapshot(snapshot_ids[0])
        last_snapshot = self._manager.get_snapshot(snapshot_ids[-1])

        if not first_snapshot or not last_snapshot:
            raise MarketSnapshotError("Failed to load snapshots for comparison")

        # Calculate the diff
        return self._manager.snapshot_diff(snapshot_ids[0], snapshot_ids[-1])

    def get_market_state_history(self,
                              symbol: str,
                              start_time: float,
                              end_time: float,
                              interval: float = 60.0,  # 1 minute
                              exchange: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get historical market state at regular intervals.

        Args:
            symbol: Trading symbol
            start_time: Start timestamp
            end_time: End timestamp
            interval: Time interval in seconds
            exchange: Optional exchange name

        Returns:
            List of market states
        """
        # Create filter
        filters = {"symbol": symbol}
        if exchange:
            filters["exchange"] = exchange

        # Find all relevant snapshots
        all_snapshots = []

        for snapshot_type in [SnapshotType.ORDERBOOK, SnapshotType.TICKERS, SnapshotType.TRADES]:
            snapshot_ids = self._manager.find_snapshots(
                snapshot_type=snapshot_type,
                start_time=start_time,
                end_time=end_time,
                filters=filters
            )

            for snapshot_id in snapshot_ids:
                snapshot = self._manager.get_snapshot(snapshot_id)
                if snapshot:
                    all_snapshots.append(snapshot)

        # Sort by timestamp
        all_snapshots.sort(key=lambda s: s["metadata"].get("timestamp", 0))

        # Group snapshots by interval
        interval_snapshots = []
        current_time = start_time

        while current_time <= end_time:
            next_time = current_time + interval

            # Find snapshots in this interval
            interval_group = [s for s in all_snapshots
                              if current_time <= s["metadata"].get("timestamp", 0) < next_time]

            if interval_group:
                # Merge snapshots for this interval
                merged_data = {}
                for snapshot in interval_group:
                    merged_data.update(snapshot["data"])

                interval_snapshots.append({
                    "timestamp": current_time,
                    "data": merged_data
                })

            current_time = next_time

        return interval_snapshots

    def clean_old_snapshots(self, retention_period: float) -> int:
        """
        Clean up old snapshots across all types.

        Args:
            retention_period: Retention period in seconds

        Returns:
            Number of snapshots removed
        """
        total_removed = 0

        for snapshot_type in SnapshotType:
            removed = self._manager.clean_old_snapshots(
                max_age=retention_period,
                snapshot_type=snapshot_type
            )
            total_removed += removed

        return total_removed

    def get_snapshot_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored snapshots.

        Returns:
            Statistics dictionary
        """
        stats = {
            "total_snapshots": 0,
            "by_type": {}
        }

        for snapshot_type in SnapshotType:
            snapshot_ids = self._manager.find_snapshots(snapshot_type=snapshot_type)
            stats["by_type"][snapshot_type.value] = len(snapshot_ids)
            stats["total_snapshots"] += len(snapshot_ids)

        return stats