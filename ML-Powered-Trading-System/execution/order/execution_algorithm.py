"""
Execution Algorithm Module

This module defines the base class and interfaces for order execution algorithms.
These algorithms implement different strategies for executing orders to optimize
for various factors such as price, impact, visibility, and urgency.

Execution algorithms handle the details of how orders are executed in the market,
breaking large orders into smaller pieces, timing the execution of these pieces,
and adapting to market conditions to achieve specific execution goals.
"""

import logging
import abc
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple

from execution.order.order import Order, OrderStatus, OrderSide, OrderType

# Configure logger
logger = logging.getLogger(__name__)


class AlgorithmGoal(Enum):
    """Goals that an execution algorithm might optimize for"""
    PRICE = "price"  # Best execution price
    IMPACT = "impact"  # Minimize market impact
    SPEED = "speed"  # Fast execution
    ANONYMITY = "anonymity"  # Hide size/intent
    PARTICIPATION = "participation"  # Control market participation rate
    CONSISTENCY = "consistency"  # Consistent execution over time
    OPPORTUNISTIC = "opportunistic"  # Look for favorable price points


class ExecutionProgress:
    """Class to track the progress of an execution algorithm"""

    def __init__(self, total_quantity: float):
        """
        Initialize execution progress tracker.

        Args:
            total_quantity: Total quantity to be executed
        """
        self.start_time = datetime.utcnow()
        self.last_update_time = self.start_time
        self.total_quantity = total_quantity
        self.executed_quantity = 0.0
        self.total_cost = 0.0
        self.num_trades = 0
        self.slices_completed = 0
        self.slices_total = 0
        self.is_complete = False
        self.extra_info: Dict[str, Any] = {}

    @property
    def average_price(self) -> float:
        """Calculate volume-weighted average price of all executions"""
        if self.executed_quantity > 0:
            return self.total_cost / self.executed_quantity
        return 0.0

    @property
    def remaining_quantity(self) -> float:
        """Calculate remaining quantity to execute"""
        return self.total_quantity - self.executed_quantity

    @property
    def completion_percentage(self) -> float:
        """Calculate percentage of total quantity executed"""
        if self.total_quantity > 0:
            return (self.executed_quantity / self.total_quantity) * 100.0
        return 0.0

    @property
    def elapsed_time(self) -> timedelta:
        """Calculate time elapsed since start"""
        return datetime.utcnow() - self.start_time

    def update(self, executed_quantity: float, price: float) -> None:
        """
        Update progress with a new execution.

        Args:
            executed_quantity: Quantity executed in this update
            price: Execution price
        """
        self.executed_quantity += executed_quantity
        self.total_cost += executed_quantity * price
        self.num_trades += 1
        self.last_update_time = datetime.utcnow()

        # Check if complete
        if self.executed_quantity >= self.total_quantity:
            self.is_complete = True

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert progress to dictionary for serialization.

        Returns:
            Dictionary with progress information
        """
        return {
            "start_time": self.start_time.isoformat(),
            "last_update_time": self.last_update_time.isoformat(),
            "total_quantity": self.total_quantity,
            "executed_quantity": self.executed_quantity,
            "remaining_quantity": self.remaining_quantity,
            "average_price": self.average_price,
            "total_cost": self.total_cost,
            "num_trades": self.num_trades,
            "slices_completed": self.slices_completed,
            "slices_total": self.slices_total,
            "completion_percentage": self.completion_percentage,
            "elapsed_seconds": self.elapsed_time.total_seconds(),
            "is_complete": self.is_complete,
            "extra_info": self.extra_info
        }


class ExecutionAlgorithm(abc.ABC):
    """
    Base class for all execution algorithms.

    This abstract class defines the interface that all execution algorithms
    must implement. Subclasses will implement specific strategies for order
    execution such as TWAP, VWAP, etc.
    """

    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Initialize the execution algorithm.

        Args:
            name: Algorithm name
            config: Configuration parameters
        """
        self.name = name
        self.config = config or {}
        self.goal = self._get_primary_goal()
        self.progress: Dict[str, ExecutionProgress] = {}

        logger.info(f"Initialized {name} execution algorithm")

    def _get_primary_goal(self) -> AlgorithmGoal:
        """Get the primary goal of this algorithm"""
        goal_str = self.config.get("primary_goal", AlgorithmGoal.PRICE.value)
        try:
            return AlgorithmGoal(goal_str)
        except ValueError:
            logger.warning(f"Invalid goal: {goal_str}, using {AlgorithmGoal.PRICE.value}")
            return AlgorithmGoal.PRICE

    @abc.abstractmethod
    def start_execution(self, order: Order) -> bool:
        """
        Start executing an order.

        Args:
            order: Order to execute

        Returns:
            True if execution started successfully
        """
        pass

    @abc.abstractmethod
    def pause_execution(self, order_id: str) -> bool:
        """
        Pause the execution of an order.

        Args:
            order_id: Order ID

        Returns:
            True if execution was paused
        """
        pass

    @abc.abstractmethod
    def resume_execution(self, order_id: str) -> bool:
        """
        Resume the execution of a paused order.

        Args:
            order_id: Order ID

        Returns:
            True if execution was resumed
        """
        pass

    @abc.abstractmethod
    def cancel_execution(self, order_id: str) -> bool:
        """
        Cancel the execution of an order.

        Args:
            order_id: Order ID

        Returns:
            True if execution was canceled
        """
        pass

    @abc.abstractmethod
    def update_parameters(self, order_id: str, params: Dict[str, Any]) -> bool:
        """
        Update execution parameters for an order.

        Args:
            order_id: Order ID
            params: New parameters

        Returns:
            True if parameters were updated
        """
        pass

    def get_progress(self, order_id: str) -> Optional[ExecutionProgress]:
        """
        Get the current progress of an order execution.

        Args:
            order_id: Order ID

        Returns:
            ExecutionProgress or None if order not found
        """
        return self.progress.get(order_id)

    def get_estimated_completion_time(self, order_id: str) -> Optional[datetime]:
        """
        Estimate when an order execution will complete.

        Args:
            order_id: Order ID

        Returns:
            Estimated completion time or None if unknown
        """
        # Base implementation provides a linear estimate
        progress = self.get_progress(order_id)
        if not progress or progress.is_complete:
            return None

        # If no execution yet, can't estimate
        if progress.executed_quantity <= 0:
            return None

        # Calculate rate of execution
        elapsed_seconds = progress.elapsed_time.total_seconds()
        if elapsed_seconds <= 0:
            return None

        execution_rate = progress.executed_quantity / elapsed_seconds  # quantity per second
        if execution_rate <= 0:
            return None

        # Estimate remaining time
        remaining_seconds = progress.remaining_quantity / execution_rate
        return datetime.utcnow() + timedelta(seconds=remaining_seconds)

    def get_all_active_executions(self) -> List[str]:
        """
        Get IDs of all actively executing orders.

        Returns:
            List of order IDs
        """
        active_ids = []
        for order_id, progress in self.progress.items():
            if not progress.is_complete:
                active_ids.append(order_id)
        return active_ids

    @abc.abstractmethod
    def can_execute_order(self, order: Order) -> Tuple[bool, str]:
        """
        Check if this algorithm can execute a given order.

        Args:
            order: Order to check

        Returns:
            (can_execute, reason) tuple
        """
        pass

    def is_supported_venue(self, venue_id: str) -> bool:
        """
        Check if a venue is supported by this algorithm.

        Args:
            venue_id: Venue/exchange ID

        Returns:
            True if supported
        """
        supported_venues = self.config.get("supported_venues", [])
        return len(supported_venues) == 0 or venue_id in supported_venues

    def is_supported_asset(self, asset_id: str) -> bool:
        """
        Check if an asset is supported by this algorithm.

        Args:
            asset_id: Asset/symbol ID

        Returns:
            True if supported
        """
        supported_assets = self.config.get("supported_assets", [])
        excluded_assets = self.config.get("excluded_assets", [])

        if asset_id in excluded_assets:
            return False

        return len(supported_assets) == 0 or asset_id in supported_assets

    def get_algorithm_info(self) -> Dict[str, Any]:
        """
        Get information about this algorithm.

        Returns:
            Dictionary with algorithm information
        """
        return {
            "name": self.name,
            "goal": self.goal.value,
            "active_executions": len(self.get_all_active_executions()),
            "config": self.config.copy()
        }