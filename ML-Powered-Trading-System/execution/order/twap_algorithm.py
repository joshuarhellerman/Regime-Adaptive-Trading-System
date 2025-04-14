"""
TWAP Execution Algorithm Module

This module implements a Time-Weighted Average Price (TWAP) execution algorithm.
TWAP aims to execute orders evenly over a specified time period to achieve
an average execution price close to the time-weighted average price.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple, Set
import random
import math

from execution.order.order import Order, OrderStatus, OrderSide, OrderType
from execution.order.order_factory import OrderFactory
from execution.algorithm.execution_algorithm import ExecutionAlgorithm, ExecutionProgress, AlgorithmGoal

# Configure logger
logger = logging.getLogger(__name__)


class TwapAlgorithm(ExecutionAlgorithm):
    """
    Time-Weighted Average Price (TWAP) execution algorithm.

    TWAP divides an order into equal-sized smaller orders and executes them at
    regular time intervals over a specified time period. This approach aims to
    achieve an execution price close to the time-weighted average market price.
    """

    def __init__(self,
                order_factory: OrderFactory,
                config: Dict[str, Any] = None):
        """
        Initialize the TWAP algorithm.

        Args:
            order_factory: Factory for creating child orders
            config: Configuration parameters
        """
        super().__init__("TWAP", config)

        self.order_factory = order_factory

        # Default configuration
        self.default_duration_minutes = self.config.get("default_duration_minutes", 30)
        self.default_num_slices = self.config.get("default_num_slices", 10)
        self.randomize_sizes = self.config.get("randomize_sizes", False)
        self.randomize_times = self.config.get("randomize_times", False)
        self.size_variance_percent = self.config.get("size_variance_percent", 10)
        self.time_variance_percent = self.config.get("time_variance_percent", 10)

        # Execution state
        self._execution_threads: Dict[str, threading.Thread] = {}
        self._execution_stop_signals: Set[str] = set()
        self._execution_paused: Set[str] = set()
        self._lock = threading.RLock()

        logger.info(f"Initialized TWAP algorithm with {self.default_num_slices} default slices "
                   f"over {self.default_duration_minutes} minutes")

    def start_execution(self, order: Order) -> bool:
        """
        Start executing an order using TWAP.

        Args:
            order: Order to execute

        Returns:
            True if execution started successfully
        """
        order_id = order.order_id

        # Check if already executing
        with self._lock:
            if order_id in self._execution_threads and self._execution_threads[order_id].is_alive():
                logger.warning(f"TWAP execution already active for order {order_id}")
                return False

        # Check if this algorithm can execute the order
        can_execute, reason = self.can_execute_order(order)
        if not can_execute:
            logger.warning(f"Cannot execute order {order_id} with TWAP: {reason}")
            return False

        # Get TWAP parameters
        duration_minutes = order.params.get("twap_duration_minutes", self.default_duration_minutes)
        num_slices = order.params.get("twap_num_slices", self.default_num_slices)
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=duration_minutes)

        # Create progress tracker
        progress = ExecutionProgress(order.quantity)
        progress.slices_total = num_slices
        progress.extra_info = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_minutes": duration_minutes,
            "num_slices": num_slices
        }
        self.progress[order_id] = progress

        # Generate slice sizes
        slice_sizes = self._generate_slice_sizes(order.quantity, num_slices)

        # Start execution thread
        thread = threading.Thread(
            target=self._execute_twap,
            args=(order, start_time, end_time, slice_sizes),
            daemon=True,
            name=f"TWAP-{order_id}"
        )

        with self._lock:
            self._execution_threads[order_id] = thread
            if order_id in self._execution_stop_signals:
                self._execution_stop_signals.remove(order_id)
            if order_id in self._execution_paused:
                self._execution_paused.remove(order_id)

        thread.start()
        logger.info(f"Started TWAP execution for order {order_id} with {num_slices} slices "
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
                logger.warning(f"No active TWAP execution for order {order_id}")
                return False

            self._execution_paused.add(order_id)
            logger.info(f"Paused TWAP execution for order {order_id}")
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
                logger.warning(f"No active TWAP execution for order {order_id}")
                return False

            if order_id in self._execution_paused:
                self._execution_paused.remove(order_id)
                logger.info(f"Resumed TWAP execution for order {order_id}")
                return True
            else:
                logger.warning(f"TWAP execution for order {order_id} is not paused")
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
                logger.warning(f"No active TWAP execution for order {order_id}")
                return False

            self._execution_stop_signals.add(order_id)
            logger.info(f"Requested cancellation of TWAP execution for order {order_id}")
            return True

    def update_parameters(self, order_id: str, params: Dict[str, Any]) -> bool:
        """
        Update execution parameters for an order.

        Note: For TWAP, updating parameters requires restarting the execution,
        so this method cancels the current execution and starts a new one.

        Args:
            order_id: Order ID
            params: New parameters

        Returns:
            True if parameters were updated
        """
        # Get progress and original order
        progress = self.get_progress(order_id)
        if not progress:
            logger.warning(f"No TWAP execution found for order {order_id}")
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
            logger.info(f"TWAP execution already complete for order {order_id}")
            return False

        # Create new order for remaining quantity
        original_order_params = params.get("original_order", {})
        if not original_order_params:
            logger.error(f"Missing original order data for TWAP update: {order_id}")
            return False

        # Apply new parameters
        for key, value in params.items():
            if key != "original_order":
                original_order_params[key] = value

        # Adjust quantity to remaining
        original_order_params["quantity"] = remaining_quantity

        # Create new order
        try:
            from execution.order.order import Order
            new_order = Order.from_dict(original_order_params)

            # Start new execution
            return self.start_execution(new_order)

        except Exception as e:
            logger.error(f"Error creating new order for TWAP update: {str(e)}")
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

    def _generate_slice_sizes(self, total_quantity: float, num_slices: int) -> List[float]:
        """
        Generate slice sizes for TWAP execution.

        Args:
            total_quantity: Total quantity to execute
            num_slices: Number of slices

        Returns:
            List of slice sizes
        """
        if num_slices <= 0:
            return [total_quantity]

        # Base case: equal-sized slices
        base_slice_size = total_quantity / num_slices
        slice_sizes = [base_slice_size] * num_slices

        # Apply randomization if configured
        if self.randomize_sizes and self.size_variance_percent > 0:
            # Calculate maximum variance
            max_variance = base_slice_size * (self.size_variance_percent / 100.0)

            # Apply random adjustments but ensure total remains the same
            adjustments = []
            for _ in range(num_slices):
                # Random value between -max_variance and +max_variance
                adj = random.uniform(-max_variance, max_variance)
                adjustments.append(adj)

            # Make sure adjustments sum to zero to preserve total quantity
            adjustment_sum = sum(adjustments)
            for i in range(len(adjustments)):
                adjustments[i] -= adjustment_sum / num_slices

            # Apply adjustments
            for i in range(num_slices):
                slice_sizes[i] += adjustments[i]

                # Ensure no negative sizes
                slice_sizes[i] = max(slice_sizes[i], 0.0)

        # Ensure slices add up to total (fix any rounding errors)
        current_sum = sum(slice_sizes)
        if current_sum != total_quantity:
            # Adjust last slice
            slice_sizes[-1] += (total_quantity - current_sum)

        # Check for minimum lot size constraints
        return slice_sizes

    def _calculate_slice_times(self,
                              start_time: datetime,
                              end_time: datetime,
                              num_slices: int) -> List[datetime]:
        """
        Calculate execution times for TWAP slices.

        Args:
            start_time: Start time
            end_time: End time
            num_slices: Number of slices

        Returns:
            List of execution times
        """
        if num_slices <= 0:
            return []

        if num_slices == 1:
            return [start_time]

        # Calculate time window
        total_seconds = (end_time - start_time).total_seconds()

        # Base case: evenly spaced times
        interval_seconds = total_seconds / (num_slices - 1) if num_slices > 1 else 0
        times = []

        for i in range(num_slices):
            seconds_offset = i * interval_seconds
            slice_time = start_time + timedelta(seconds=seconds_offset)
            times.append(slice_time)

        # Apply randomization if configured
        if self.randomize_times and self.time_variance_percent > 0 and num_slices > 2:
            # Maximum variance is a percentage of the interval
            max_variance_seconds = interval_seconds * (self.time_variance_percent / 100.0)

            # Skip first and last times (keep them fixed)
            for i in range(1, num_slices - 1):
                # Random adjustment between -max_variance and +max_variance
                adj_seconds = random.uniform(-max_variance_seconds, max_variance_seconds)
                times[i] += timedelta(seconds=adj_seconds)

                # Ensure times remain in order
                if i > 0 and times[i] <= times[i-1]:
                    times[i] = times[i-1] + timedelta(seconds=1)
                if i < num_slices - 1 and times[i] >= times[i+1]:
                    times[i] = times[i+1] - timedelta(seconds=1)

        return times

    def _execute_twap(self,
                     order: Order,
                     start_time: datetime,
                     end_time: datetime,
                     slice_sizes: List[float]) -> None:
        """
        Execute a TWAP order.

        Args:
            order: Parent order
            start_time: Execution start time
            end_time: Execution end time
            slice_sizes: List of slice sizes
        """
        order_id = order.order_id
        num_slices = len(slice_sizes)

        try:
            # Calculate slice execution times
            slice_times = self._calculate_slice_times(start_time, end_time, num_slices)

            # Execute each slice
            for i, (slice_size, slice_time) in enumerate(zip(slice_sizes, slice_times)):
                # Check for stop signal
                with self._lock:
                    if order_id in self._execution_stop_signals:
                        logger.info(f"TWAP execution stopped for order {order_id}")
                        return

                # Wait until slice time
                wait_seconds = (slice_time - datetime.utcnow()).total_seconds()
                if wait_seconds > 0:
                    # Break wait into smaller chunks to check for stop signals
                    chunk_size = 1.0  # seconds
                    chunks = math.ceil(wait_seconds / chunk_size)

                    for _ in range(chunks):
                        # Check for stop signal
                        with self._lock:
                            if order_id in self._execution_stop_signals:
                                logger.info(f"TWAP execution stopped during wait for order {order_id}")
                                return

                        # Check for pause signal
                        with self._lock:
                            is_paused = order_id in self._execution_paused

                        if is_paused:
                            # While paused, keep checking for resume/stop
                            while True:
                                with self._lock:
                                    if order_id in self._execution_stop_signals:
                                        logger.info(f"TWAP execution stopped while paused for order {order_id}")
                                        return

                                    if order_id not in self._execution_paused:
                                        # Resumed
                                        break

                                time.sleep(0.5)

                        # Wait a chunk
                        remaining = (slice_time - datetime.utcnow()).total_seconds()
                        if remaining <= 0:
                            break
                        time.sleep(min(chunk_size, remaining))

                # Create and execute child order for this slice
                try:
                    # Create child order
                    child_order = self._create_child_order(order, slice_size, i)

                    # Execute the child order
                    execution_price = self._execute_child_order(child_order)

                    if execution_price > 0:
                        # Update progress
                        progress = self.progress[order_id]
                        progress.update(slice_size, execution_price)
                        progress.slices_completed = i + 1

                        logger.info(f"TWAP slice {i+1}/{num_slices} executed for order {order_id}, "
                                  f"price: {execution_price}, quantity: {slice_size}")
                    else:
                        logger.warning(f"TWAP slice {i+1}/{num_slices} failed for order {order_id}")

                except Exception as e:
                    logger.error(f"Error executing TWAP slice for order {order_id}: {str(e)}")

            # Mark execution as complete
            progress = self.progress[order_id]
            progress.is_complete = True
            logger.info(f"TWAP execution completed for order {order_id}, "
                       f"average price: {progress.average_price}")

        except Exception as e:
            logger.error(f"Error in TWAP execution for order {order_id}: {str(e)}", exc_info=True)

            # Clean up
            with self._lock:
                if order_id in self._execution_threads:
                    del self._execution_threads[order_id]
                if order_id in self._execution_stop_signals:
                    self._execution_stop_signals.remove(order_id)
                if order_id in self._execution_paused:
                    self._execution_paused.remove(order_id)

    def _create_child_order(self, parent_order: Order, quantity: float, slice_index: int) -> Order:
        """
        Create a child order for a TWAP slice.

        Args:
            parent_order: Parent order
            quantity: Slice quantity
            slice_index: Index of the slice

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
                    "twap_slice_index": slice_index
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
                    "twap_slice_index": slice_index
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
        """Get the primary goal of TWAP"""
        return AlgorithmGoal.CONSISTENCY