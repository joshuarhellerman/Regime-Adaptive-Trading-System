"""
VWAP Execution Algorithm Module

This module implements a Volume-Weighted Average Price (VWAP) execution algorithm.
VWAP executes orders following an expected volume profile over a specified time period,
aiming to achieve an average execution price close to the market's volume-weighted average price.

This algorithm is suitable for traders who are more concerned with execution quality
relative to VWAP benchmark than with execution speed. By following historical volume
patterns, it aims to reduce market impact by trading more heavily during periods of
high market volume.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple, Set
import random
import math
import json

from execution.order.order import Order, OrderStatus, OrderSide, OrderType, TimeInForce
from execution.order.order_factory import OrderFactory
from execution.algorithm.execution_algorithm import ExecutionAlgorithm, ExecutionProgress, AlgorithmGoal

# Configure logger
logger = logging.getLogger(__name__)


class VolumeProfile:
    """Class to represent a trading volume profile over time periods"""

    def __init__(self, buckets: int = 12):
        """
        Initialize a volume profile.

        Args:
            buckets: Number of time buckets in the profile
        """
        self.buckets = buckets
        self.percentages = self._default_profile(buckets)

    def _default_profile(self, buckets: int) -> List[float]:
        """
        Create a default volume profile based on typical market patterns.

        Args:
            buckets: Number of time buckets

        Returns:
            List of volume percentages for each bucket
        """
        if buckets == 12:  # Hourly buckets for a trading day
            # Typical U-shaped volume profile with higher volume at open and close
            return [
                0.12,  # Opening hour
                0.08,
                0.07,
                0.06,
                0.06,
                0.07,
                0.07,
                0.08,
                0.08,
                0.09,
                0.10,
                0.12   # Closing hour
            ]
        else:
            # For other bucket counts, create a simple U-shape
            middle = buckets // 2
            profile = []
            for i in range(buckets):
                # Distance from middle, normalized to 0-1 range
                dist = abs(i - middle) / middle
                # Higher values at edges, lower in middle
                value = 0.05 + 0.1 * dist
                profile.append(value)

            # Normalize to sum to 1.0
            total = sum(profile)
            return [v / total for v in profile]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VolumeProfile':
        """
        Create a volume profile from dictionary data.

        Args:
            data: Dictionary with profile data

        Returns:
            VolumeProfile instance
        """
        profile = cls(buckets=len(data.get("percentages", [])))
        profile.percentages = data.get("percentages", profile.percentages)
        return profile

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert profile to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "buckets": self.buckets,
            "percentages": self.percentages
        }

    def get_bucket_for_time(self, start_time: datetime, end_time: datetime, current_time: datetime) -> int:
        """
        Determine which bucket a time falls into.

        Args:
            start_time: Start of the time window
            end_time: End of the time window
            current_time: Time to check

        Returns:
            Bucket index (0-based)
        """
        # Total duration in seconds
        total_seconds = (end_time - start_time).total_seconds()
        if total_seconds <= 0:
            return 0

        # Current position in seconds
        current_seconds = (current_time - start_time).total_seconds()
        if current_seconds < 0:
            return 0
        if current_seconds >= total_seconds:
            return self.buckets - 1

        # Calculate bucket
        bucket = int((current_seconds / total_seconds) * self.buckets)
        return min(bucket, self.buckets - 1)

    def calculate_slice_sizes(self, total_quantity: float) -> List[float]:
        """
        Calculate slice sizes based on volume profile.

        Args:
            total_quantity: Total quantity to execute

        Returns:
            List of slice sizes for each bucket
        """
        return [total_quantity * pct for pct in self.percentages]
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple, Set
import random
import math
import json

from execution.order.order import Order, OrderStatus, OrderSide, OrderType
from execution.order.order_factory import OrderFactory
from execution.algorithm.execution_algorithm import ExecutionAlgorithm, ExecutionProgress, AlgorithmGoal

# Configure logger
logger = logging.getLogger(__name__)


class VwapAlgorithm(ExecutionAlgorithm):
    """
    Volume-Weighted Average Price (VWAP) execution algorithm.

    VWAP divides an order into smaller orders and executes them following a volume
    profile over a specified time period. This aims to achieve an execution price
    close to the volume-weighted average market price and minimize market impact
    by trading more when liquidity is higher.

    def __init__(self,
                order_factory: OrderFactory,
                config: Dict[str, Any] = None):
        """
        Initialize the VWAP algorithm.

        Args:
            order_factory: Factory for creating child orders
            config: Configuration parameters
        """
        super().__init__("VWAP", config)

        self.order_factory = order_factory

        # Default configuration
        self.default_duration_minutes = self.config.get("default_duration_minutes", 60)
        self.default_num_buckets = self.config.get("default_num_buckets", 12)
        self.randomize_within_bucket = self.config.get("randomize_within_bucket", True)
        self.randomize_sizes = self.config.get("randomize_sizes", False)
        self.size_variance_percent = self.config.get("size_variance_percent", 5)

        # Load volume profiles
        self.default_profile = VolumeProfile(buckets=self.default_num_buckets)
        self.symbol_profiles: Dict[str, VolumeProfile] = {}
        self._load_symbol_profiles()

        # Execution state
        self._execution_threads: Dict[str, threading.Thread] = {}
        self._execution_stop_signals: Set[str] = set()
        self._execution_paused: Set[str] = set()
        self._lock = threading.RLock()

        logger.info(f"Initialized VWAP algorithm with {self.default_num_buckets} time buckets "
                   f"over {self.default_duration_minutes} minutes")

    def _load_symbol_profiles(self) -> None:
        """Load volume profiles for specific symbols"""
        profile_data = self.config.get("symbol_profiles", {})

        for symbol, data in profile_data.items():
            try:
                profile = VolumeProfile.from_dict(data)
                self.symbol_profiles[symbol] = profile
                logger.debug(f"Loaded custom volume profile for {symbol}")
            except Exception as e:
                logger.error(f"Error loading volume profile for {symbol}: {str(e)}")

    def start_execution(self, order: Order) -> bool:
        """
        Start executing an order using VWAP.

        Args:
            order: Order to execute

        Returns:
            True if execution started successfully
        """
        order_id = order.order_id

        # Check if already executing
        with self._lock:
            if order_id in self._execution_threads and self._execution_threads[order_id].is_alive():
                logger.warning(f"VWAP execution already active for order {order_id}")
                return False

        # Check if this algorithm can execute the order
        can_execute, reason = self.can_execute_order(order)
        if not can_execute:
            logger.warning(f"Cannot execute order {order_id} with VWAP: {reason}")
            return False

        # Get VWAP parameters
        duration_minutes = order.params.get("vwap_duration_minutes", self.default_duration_minutes)
        num_buckets = order.params.get("vwap_num_buckets", self.default_num_buckets)

        # Get volume profile
        profile = self._get_profile_for_symbol(order.symbol, num_buckets)

        # Define time window
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=duration_minutes)

        # Calculate slice sizes
        slice_sizes = profile.calculate_slice_sizes(order.quantity)

        # Apply randomization if configured
        if self.randomize_sizes and self.size_variance_percent > 0:
            slice_sizes = self._randomize_slice_sizes(slice_sizes)

        # Create progress tracker
        progress = ExecutionProgress(order.quantity)
        progress.slices_total = len(slice_sizes)
        progress.extra_info = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_minutes": duration_minutes,
            "num_buckets": num_buckets,
            "profile": profile.to_dict()
        }
        self.progress[order_id] = progress

        # Start execution thread
        thread = threading.Thread(
            target=self._execute_vwap,
            args=(order, profile, start_time, end_time, slice_sizes),
            daemon=True,
            name=f"VWAP-{order_id}"
        )

        with self._lock:
            self._execution_threads[order_id] = thread
            if order_id in self._execution_stop_signals:
                self._execution_stop_signals.remove(order_id)
            if order_id in self._execution_paused:
                self._execution_paused.remove(order_id)

        thread.start()
        logger.info(f"Started VWAP execution for order {order_id} with {num_buckets} buckets "
                   f"over {duration_minutes} minutes")

        return True

    def pause_execution(self, order_id: str) -> bool:
        """
        Pause the execution of an order.

        Args:
            order_id: Order ID

        Returns:
            True if execution was paused
        """
        with self._lock:
            if order_id not in self._execution_threads or not self._execution_threads[order_id].is_alive():
                logger.warning(f"No active VWAP execution for order {order_id}")
                return False

            self._execution_paused.add(order_id)
            logger.info(f"Paused VWAP execution for order {order_id}")
            return True

    def resume_execution(self, order_id: str) -> bool:
        """
        Resume the execution of a paused order.

        Args:
            order_id: Order ID

        Returns:
            True if execution was resumed
        """
        with self._lock:
            if order_id not in self._execution_threads or not self._execution_threads[order_id].is_alive():
                logger.warning(f"No active VWAP execution for order {order_id}")
                return False

            if order_id in self._execution_paused:
                self._execution_paused.remove(order_id)
                logger.info(f"Resumed VWAP execution for order {order_id}")
                return True
            else:
                logger.warning(f"VWAP execution for order {order_id} is not paused")
                return False

    def cancel_execution(self, order_id: str) -> bool:
        """
        Cancel the execution of an order.

        Args:
            order_id: Order ID

        Returns:
            True if execution was canceled
        """
        with self._lock:
            if order_id not in self._execution_threads or not self._execution_threads[order_id].is_alive():
                logger.warning(f"No active VWAP execution for order {order_id}")
                return False

            self._execution_stop_signals.add(order_id)
            logger.info(f"Requested cancellation of VWAP execution for order {order_id}")
            return True

    def update_parameters(self, order_id: str, params: Dict[str, Any]) -> bool:
        """
        Update execution parameters for an order.

        Args:
            order_id: Order ID
            params: New parameters

        Returns:
            True if parameters were updated
        """
        # Similar to TWAP, updating VWAP parameters requires restarting the execution
        # Get progress and original order
        progress = self.get_progress(order_id)
        if not progress:
            logger.warning(f"No VWAP execution found for order {order_id}")
            return False

        # Cancel current execution
        self.cancel_execution(order_id)

        # Wait for thread to stop (with timeout)
        with self._lock:
            if order_id in self._execution_threads:
                thread = self._execution_threads[order_id]
                if thread.is_alive():
                    thread.join(timeout=5.0)

        # Get updated parameters
        remaining_quantity = progress.remaining_quantity
        if remaining_quantity <= 0:
            logger.info(f"VWAP execution already complete for order {order_id}")
            return False

        # Create new order for remaining quantity
        original_order_params = params.get("original_order", {})
        if not original_order_params:
            logger.error(f"Missing original order data for VWAP update: {order_id}")
            return False

        # Apply new parameters
        for key, value in params.items():
            if key != "original_order":
                original_order_params[key] = value

        # Adjust quantity to remaining
        original_order_params["quantity"] = remaining_quantity

        # Create new order
        try:
            new_order = Order.from_dict(original_order_params)

            # Start new execution
            return self.start_execution(new_order)

        except Exception as e:
            logger.error(f"Error creating new order for VWAP update: {str(e)}")
            return False

    def can_execute_order(self, order: Order) -> Tuple[bool, str]:
        """
        Check if this algorithm can execute a given order.

        Args:
            order: Order to check

        Returns:
            (can_execute, reason) tuple
        """
        # Check order type
        if order.order_type not in [OrderType.MARKET, OrderType.LIMIT]:
            return False, f"Unsupported order type: {order.order_type.value}"

        # Check quantity
        if order.quantity <= 0:
            return False, "Invalid order quantity"

        # Check venue
        if order.exchange_id and not self.is_supported_venue(order.exchange_id):
            return False, f"Unsupported venue: {order.exchange_id}"

        # Check symbol
        if not self.is_supported_asset(order.symbol):
            return False, f"Unsupported asset: {order.symbol}"

        return True, ""

    def _get_profile_for_symbol(self, symbol: str, num_buckets: int) -> VolumeProfile:
        """
        Get the volume profile for a symbol.

        Args:
            symbol: Trading symbol
            num_buckets: Number of time buckets

        Returns:
            Volume profile
        """
        # Check if we have a custom profile for this symbol
        if symbol in self.symbol_profiles:
            profile = self.symbol_profiles[symbol]

            # If bucket count matches, use as-is
            if profile.buckets == num_buckets:
                return profile

        # Use default profile with requested bucket count
        return VolumeProfile(buckets=num_buckets)

    def _randomize_slice_sizes(self, slice_sizes: List[float]) -> List[float]:
        """
        Randomize slice sizes while preserving the total.

        Args:
            slice_sizes: Original slice sizes

        Returns:
            Randomized slice sizes
        """
        num_slices = len(slice_sizes)
        if num_slices <= 1:
            return slice_sizes

        total_quantity = sum(slice_sizes)

        # Calculate maximum variance for each slice
        max_variances = [size * (self.size_variance_percent / 100.0) for size in slice_sizes]

        # Apply random adjustments but ensure total remains the same
        adjustments = []
        for i, max_var in enumerate(max_variances):
            # Random value between -max_var and +max_var
            adj = random.uniform(-max_var, max_var)
            adjustments.append(adj)

        # Make sure adjustments sum to zero to preserve total quantity
        adjustment_sum = sum(adjustments)
        for i in range(len(adjustments)):
            adjustments[i] -= adjustment_sum / num_slices

        # Apply adjustments
        result = []
        for i, size in enumerate(slice_sizes):
            new_size = size + adjustments[i]
            # Ensure no negative sizes
            new_size = max(new_size, 0.0)
            result.append(new_size)

        # Ensure slices add up to total (fix any rounding errors)
        current_sum = sum(result)
        if abs(current_sum - total_quantity) > 0.000001:
            # Distribute the difference proportionally
            diff = total_quantity - current_sum
            for i in range(len(result)):
                portion = slice_sizes[i] / total_quantity
                result[i] += diff * portion

        return result

    def _execute_vwap(self,
                     order: Order,
                     profile: VolumeProfile,
                     start_time: datetime,
                     end_time: datetime,
                     slice_sizes: List[float]) -> None:
        """
        Execute a VWAP order.

        Args:
            order: Parent order
            profile: Volume profile
            start_time: Execution start time
            end_time: Execution end time
            slice_sizes: List of slice sizes
        """
        order_id = order.order_id
        num_buckets = len(slice_sizes)

        try:
            # Initialize tracking
            executed_buckets = set()
            executed_quantity = 0.0
            total_cost = 0.0

            # Main execution loop - run until end time or all buckets executed
            while datetime.utcnow() < end_time and len(executed_buckets) < num_buckets:
                # Check for stop signal
                with self._lock:
                    if order_id in self._execution_stop_signals:
                        logger.info(f"VWAP execution stopped for order {order_id}")
                        return

                # Check for pause signal
                with self._lock:
                    is_paused = order_id in self._execution_paused

                if is_paused:
                    # While paused, keep checking for resume/stop
                    time.sleep(1.0)
                    continue

                # Get current bucket
                current_time = datetime.utcnow()
                current_bucket = profile.get_bucket_for_time(start_time, end_time, current_time)

                # Check if this bucket has already been executed
                if current_bucket in executed_buckets:
                    # Sleep briefly before checking again
                    time.sleep(0.5)
                    continue

                # Get slice size for this bucket
                slice_size = slice_sizes[current_bucket]
                if slice_size <= 0:
                    # Mark bucket as executed and continue
                    executed_buckets.add(current_bucket)
                    continue

                # Execute the slice
                try:
                    # Create and execute child order
                    child_order = self._create_child_order(order, slice_size, current_bucket)
                    execution_price = self._execute_child_order(child_order)

                    if execution_price > 0:
                        # Update tracking
                        executed_quantity += slice_size
                        total_cost += slice_size * execution_price
                        executed_buckets.add(current_bucket)

                        # Update progress
                        progress = self.progress[order_id]
                        progress.update(slice_size, execution_price)
                        progress.slices_completed = len(executed_buckets)

                        logger.info(f"VWAP bucket {current_bucket+1}/{num_buckets} executed for order {order_id}, "
                                   f"price: {execution_price}, quantity: {slice_size}")
                    else:
                        logger.warning(f"VWAP bucket {current_bucket+1}/{num_buckets} failed for order {order_id}")
                        # Don't mark as executed, will retry

                except Exception as e:
                    logger.error(f"Error executing VWAP bucket for order {order_id}: {str(e)}")

                # Sleep before checking next bucket
                time.sleep(1.0)

            # Check if all buckets were executed
            if len(executed_buckets) < num_buckets:
                logger.warning(f"VWAP execution incomplete for order {order_id}, "
                              f"executed {len(executed_buckets)}/{num_buckets} buckets")

            # Mark execution as complete
            progress = self.progress[order_id]
            progress.is_complete = True
            logger.info(f"VWAP execution completed for order {order_id}, "
                       f"average price: {progress.average_price}")

        except Exception as e:
            logger.error(f"Error in VWAP execution for order {order_id}: {str(e)}", exc_info=True)

        finally:
            # Clean up
            with self._lock:
                if order_id in self._execution_threads:
                    del self._execution_threads[order_id]
                if order_id in self._execution_stop_signals:
                    self._execution_stop_signals.remove(order_id)
                if order_id in self._execution_paused:
                    self._execution_paused.remove(order_id)

    def _create_child_order(self, parent_order: Order, quantity: float, bucket_index: int) -> Order:
        """
        Create a child order for a VWAP bucket.

        Args:
            parent_order: Parent order
            quantity: Bucket quantity
            bucket_index: Index of the bucket

        Returns:
            Child order
        """
        # Determine order type
        if parent_order.order_type == OrderType.LIMIT:
            # Use the same limit price
            child_order = self.order_factory.create_limit_order(
                symbol=parent_order.symbol,
                side=parent_order.side,
                quantity=quantity,
                price=parent_order.price,
                exchange_id=parent_order.exchange_id,
                exchange_account=parent_order.exchange_account,
                time_in_force=TimeInForce.IOC,  # Immediate or Cancel for slices
                params={
                    "parent_order_id": parent_order.order_id,
                    "vwap_bucket_index": bucket_index
                }
            )
        else:
            # Market order
            child_order = self.order_factory.create_market_order(
                symbol=parent_order.symbol,
                side=parent_order.side,
                quantity=quantity,
                exchange_id=parent_order.exchange_id,
                exchange_account=parent_order.exchange_account,
                params={
                    "parent_order_id": parent_order.order_id,
                    "vwap_bucket_index": bucket_index
                }
            )

        return child_order

    def _execute_child_order(self, order: Order) -> float:
        """
        Execute a child order and return the execution price.

        Args:
            order: Child order to execute

        Returns:
            Execution price (0 if failed)
        """
        # This method would normally submit the order to the execution service
        # and wait for a fill. For this implementation, we'll simulate execution.

        # Simulate market price
        base_price = 100.0  # Replace with actual market data in real implementation
        random_offset = random.uniform(-0.5, 0.5)
        execution_price = base_price + random_offset

        # In a real implementation, this would be replaced with:
        # - Submit order to execution service
        # - Wait for fill confirmation
        # - Return the actual fill price

        # Simulate execution delay
        time.sleep(0.5)

        return execution_price

    def _get_primary_goal(self) -> AlgorithmGoal:
        """Get the primary goal of VWAP"""
        return AlgorithmGoal.PRICE