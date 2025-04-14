"""
execution_service.py - Central Coordination Service for Order Execution

This module serves as the primary coordination layer for all order execution functionality
within the trading system. It orchestrates the order lifecycle from creation through execution
to completion, while enforcing risk management constraints and providing execution analytics.

The ExecutionService manages:
1. Order routing and submission to appropriate exchanges
2. Order state tracking and lifecycle management
3. Risk checks and pre-trade validation
4. Execution quality monitoring
5. Order execution strategies (e.g., TWAP, VWAP, etc.)
6. Synchronization with portfolio management
"""

import logging
import threading
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple, Callable, Set
import queue
import asyncio
from dataclasses import dataclass

from core.event_bus import EventTopics, create_event, get_event_bus, Event, EventPriority
from execution.exchange.connectivity_manager import ConnectivityManager
from execution.order.order import Order, OrderStatus, OrderType, TimeInForce
from execution.order.order_book import OrderBook
from execution.order.order_factory import OrderFactory

# Configure logger
logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Execution strategies for orders"""
    MARKET = "market"  # Immediate execution at market price
    LIMIT = "limit"    # Execution at specified price or better
    TWAP = "twap"      # Time-Weighted Average Price
    VWAP = "vwap"      # Volume-Weighted Average Price
    PERCENTAGE_OF_VOLUME = "pov"  # Percentage of Volume
    IMPLEMENTATION_SHORTFALL = "is"  # Implementation Shortfall
    ICEBERG = "iceberg"  # Iceberg/Reserve order
    SNIPER = "sniper"  # Opportunistic executions


class RiskCheckLevel(Enum):
    """Risk check severity levels"""
    INFO = "info"  # Informational only
    WARNING = "warning"  # Warning but allows execution
    ERROR = "error"  # Blocks execution


@dataclass
class RiskCheckResult:
    """Result of a risk check"""
    level: RiskCheckLevel
    message: str
    check_name: str
    details: Optional[Dict[str, Any]] = None


class ExecutionMode(Enum):
    """Mode of execution for the service"""
    LIVE = "live"  # Live trading with real execution
    PAPER = "paper"  # Paper trading with simulated execution
    BACKTEST = "backtest"  # Backtesting mode
    SIMULATION = "simulation"  # Real-time simulation


class ExecutionService:
    """
    Central service for coordinating all order execution activities.
    
    Responsibilities:
    - Submit orders to appropriate exchanges
    - Track order status and lifecycle
    - Apply risk management rules
    - Execute complex order strategies
    - Provide execution analytics
    - Interface with portfolio management
    """

    def _execute_vwap_simulation(self, order_id: str, handler, start_time, end_time, slice_sizes) -> None:
        """
        Execute a VWAP order simulation.
        
        Args:
            order_id: Order ID
            handler: Exchange handler
            start_time: Start time
            end_time: End time
            slice_sizes: List of slice sizes
        """
        try:
            # Get order
            order = self.order_book.get_order(order_id)
            if not order or order.status not in [OrderStatus.WORKING]:
                logger.warning(f"Cannot execute VWAP for order {order_id}, invalid status")
                return
            
            # Calculate time per slice
            now = datetime.utcnow()
            if start_time < now:
                start_time = now
                
            if end_time <= start_time:
                logger.error(f"Invalid VWAP time window for order {order_id}")
                order.status = OrderStatus.ERROR
                order.status_message = "Invalid VWAP time window"
                self.order_book.update_order(order)
                self._publish_order_event(EventTopics.ORDER_ERROR, order)
                return
                
            time_window = (end_time - start_time).total_seconds()
            num_slices = len(slice_sizes)
            time_per_slice = time_window / num_slices
            
            # Execute each slice
            remaining_quantity = order.quantity
            total_cost = 0
            executed_slices = 0
            
            for i, slice_size in enumerate(slice_sizes):
                # Check if order was canceled
                order = self.order_book.get_order(order_id)
                if not order or order.status in [OrderStatus.CANCELED, OrderStatus.ERROR]:
                    logger.info(f"VWAP execution stopped for order {order_id}, status: {order.status.value if order else 'None'}")
                    return
                
                # Wait until slice time
                slice_time = start_time + datetime.timedelta(seconds=i * time_per_slice)
                wait_seconds = (slice_time - datetime.utcnow()).total_seconds()
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
                
                # Simulate execution
                try:
                    # Get simulated price
                    price = handler.get_simulated_price(order)
                    
                    # Update order
                    executed_slices += 1
                    executed_size = min(slice_size, remaining_quantity)
                    remaining_quantity -= executed_size
                    filled_quantity = order.quantity - remaining_quantity
                    total_cost += executed_size * price
                    average_price = total_cost / filled_quantity if filled_quantity > 0 else 0
                    
                    order.filled_quantity = filled_quantity
                    order.average_price = average_price
                    order.status = OrderStatus.PARTIALLY_FILLED if remaining_quantity > 0 else OrderStatus.FILLED
                    order.status_message = f"VWAP execution: {executed_slices}/{num_slices} slices"
                    order.params['vwap_slices_executed'] = executed_slices
                    order.params['vwap_slice_times'].append(slice_time.isoformat())
                    
                    self.order_book.update_order(order)
                    
                    # Publish event
                    if order.status == OrderStatus.FILLED:
                        self._publish_order_event(EventTopics.ORDER_FILLED, order)
                        # Run post-trade risk checks
                        self._run_post_trade_risk_checks(order)
                    else:
                        self._publish_order_event(EventTopics.ORDER_PARTIALLY_FILLED, order)
                        
                    logger.info(f"VWAP slice {executed_slices}/{num_slices} executed for order {order_id}, "
                               f"price: {price}, size: {executed_size}")
                    
                except Exception as e:
                    logger.error(f"Error executing VWAP slice for order {order_id}: {str(e)}")
                
                # Break if all quantity executed
                if remaining_quantity <= 0:
                    break
            
            # Final update if not complete
            if remaining_quantity > 0:
                logger.warning(f"VWAP execution incomplete for order {order_id}, remaining: {remaining_quantity}")
                order.status = OrderStatus.PARTIALLY_FILLED
                order.status_message = f"VWAP execution incomplete, filled: {order.filled_quantity}/{order.quantity}"
                self.order_book.update_order(order)
                self._publish_order_event(EventTopics.ORDER_PARTIALLY_FILLED, order)
            
        except Exception as e:
            logger.error(f"Error in VWAP simulation for order {order_id}: {str(e)}", exc_info=True)
            
            # Update order status
            order = self.order_book.get_order(order_id)
            if order:
                order.status = OrderStatus.ERROR
                order.status_message = f"VWAP execution error: {str(e)}"
                self.order_book.update_order(order)
                self._publish_order_event(EventTopics.ORDER_ERROR, order) __init__(self, 
                 connectivity_manager: ConnectivityManager,
                 config: Dict[str, Any] = None,
                 execution_mode: ExecutionMode = ExecutionMode.PAPER):
        """
        Initialize the execution service.
        
        Args:
            connectivity_manager: ConnectivityManager instance for exchange communication
            config: Configuration dictionary
            execution_mode: Mode of execution (live, paper, backtest, simulation)
        """
        self.connectivity_manager = connectivity_manager
        self.config = config or {}
        self.execution_mode = execution_mode
        
        # Event bus for publishing execution events
        self._event_bus = get_event_bus()
        
        # Order tracking
        self.order_book = OrderBook()
        self.order_factory = OrderFactory()
        
        # Order processing queues
        self._order_queue = queue.Queue()
        self._cancel_queue = queue.Queue()
        self._update_queue = queue.Queue()
        
        # Strategy handlers
        self._strategy_handlers = {
            ExecutionStrategy.MARKET: self._execute_market_order,
            ExecutionStrategy.LIMIT: self._execute_limit_order,
            ExecutionStrategy.TWAP: self._execute_twap_order,
            ExecutionStrategy.VWAP: self._execute_vwap_order,
            ExecutionStrategy.PERCENTAGE_OF_VOLUME: self._execute_pov_order,
            ExecutionStrategy.IMPLEMENTATION_SHORTFALL: self._execute_is_order,
            ExecutionStrategy.ICEBERG: self._execute_iceberg_order,
            ExecutionStrategy.SNIPER: self._execute_sniper_order,
        }
        
        # Risk check handlers
        self._pre_trade_risk_checks = []
        self._post_trade_risk_checks = []
        
        # Execution analytics
        self._execution_metrics = {}
        
        # Service state
        self._running = False
        self._worker_threads = []
        self._lock = threading.RLock()
        
        # Exchange-specific handlers
        self._exchange_handlers = {}
        
        # Initialize risk checks
        self._init_risk_checks()
        
        # Initialize exchange handlers
        self._init_exchange_handlers()
        
        logger.info(f"ExecutionService initialized in {execution_mode.value} mode")
    
    def _init_risk_checks(self) -> None:
        """Initialize risk check handlers"""
        # Pre-trade risk checks
        self._pre_trade_risk_checks = [
            self._check_order_size_limits,
            self._check_position_limits,
            self._check_order_frequency,
            self._check_price_deviation,
            self._check_market_hours,
            self._check_exchange_connectivity,
            self._check_available_capital
        ]
        
        # Post-trade risk checks
        self._post_trade_risk_checks = [
            self._check_execution_quality,
            self._check_portfolio_impact,
            self._check_risk_metrics
        ]
        
        logger.debug(f"Initialized {len(self._pre_trade_risk_checks)} pre-trade and "
                     f"{len(self._post_trade_risk_checks)} post-trade risk checks")
    
    def _init_exchange_handlers(self) -> None:
        """Initialize exchange-specific handlers"""
        exchange_configs = self.config.get("exchanges", {})
        
        for exchange_id, config in exchange_configs.items():
            # Skip if disabled
            if not config.get("enabled", True):
                continue
                
            handler_class_name = config.get("handler_class")
            if handler_class_name:
                try:
                    # Dynamically import handler class
                    module_path, class_name = handler_class_name.rsplit(".", 1)
                    module = __import__(module_path, fromlist=[class_name])
                    handler_class = getattr(module, class_name)
                    
                    # Create handler instance
                    handler = handler_class(
                        exchange_id=exchange_id,
                        connectivity_manager=self.connectivity_manager,
                        config=config
                    )
                    
                    self._exchange_handlers[exchange_id] = handler
                    logger.info(f"Initialized exchange handler for {exchange_id}: {handler_class_name}")
                    
                except Exception as e:
                    logger.error(f"Error initializing exchange handler for {exchange_id}: {str(e)}")
            else:
                # Use default handler if specific one is not specified
                from execution.exchange.default_exchange_handler import DefaultExchangeHandler
                handler = DefaultExchangeHandler(
                    exchange_id=exchange_id,
                    connectivity_manager=self.connectivity_manager,
                    config=config
                )
                self._exchange_handlers[exchange_id] = handler
                logger.info(f"Initialized default exchange handler for {exchange_id}")
    
    def start(self) -> None:
        """Start the execution service and worker threads"""
        with self._lock:
            if self._running:
                logger.warning("ExecutionService already running")
                return
            
            self._running = True
            
            # Start worker threads
            num_workers = self.config.get("num_order_workers", 2)
            for i in range(num_workers):
                worker = threading.Thread(
                    target=self._order_worker,
                    args=(i,),
                    daemon=True,
                    name=f"OrderWorker-{i}"
                )
                worker.start()
                self._worker_threads.append(worker)
            
            # Start cancel worker
            cancel_worker = threading.Thread(
                target=self._cancel_worker,
                daemon=True,
                name="CancelWorker"
            )
            cancel_worker.start()
            self._worker_threads.append(cancel_worker)
            
            # Start update worker
            update_worker = threading.Thread(
                target=self._update_worker,
                daemon=True,
                name="UpdateWorker"
            )
            update_worker.start()
            self._worker_threads.append(update_worker)
            
            # Start status polling if configured
            polling_interval = self.config.get("status_polling_interval_seconds")
            if polling_interval:
                polling_worker = threading.Thread(
                    target=self._status_polling_worker,
                    args=(polling_interval,),
                    daemon=True,
                    name="StatusPollingWorker"
                )
                polling_worker.start()
                self._worker_threads.append(polling_worker)
            
            # Publish event
            event = create_event(
                EventTopics.COMPONENT_STARTED,
                {
                    "component": "execution_service",
                    "mode": self.execution_mode.value,
                    "timestamp": time.time()
                }
            )
            self._event_bus.publish(event)
            
            logger.info(f"ExecutionService started with {num_workers} order workers")
    
    def stop(self) -> None:
        """Stop the execution service and worker threads"""
        with self._lock:
            if not self._running:
                logger.warning("ExecutionService not running")
                return
            
            self._running = False
            
            # Signal threads to stop by adding None to queues
            for _ in range(len(self._worker_threads)):
                self._order_queue.put(None)
                self._cancel_queue.put(None)
                self._update_queue.put(None)
            
            # Wait for threads to stop
            for i, thread in enumerate(self._worker_threads):
                thread.join(timeout=5.0)
                if thread.is_alive():
                    logger.warning(f"Worker thread {thread.name} did not terminate")
            
            self._worker_threads = []
            
            # Publish event
            event = create_event(
                EventTopics.COMPONENT_STOPPED,
                {
                    "component": "execution_service",
                    "timestamp": time.time()
                }
            )
            self._event_bus.publish(event)
            
            logger.info("ExecutionService stopped")
    
    def submit_order(self, order: Order) -> str:
        """
        Submit an order for execution.
        
        Args:
            order: Order to execute
            
        Returns:
            Order ID
        """
        if not self._running:
            raise RuntimeError("ExecutionService is not running")
        
        # Ensure order has a unique ID
        if not order.order_id:
            order.order_id = str(uuid.uuid4())
        
        # Set initial status
        order.status = OrderStatus.PENDING
        order.creation_time = datetime.utcnow()
        
        # Add to order book
        self.order_book.add_order(order)
        
        # Publish order created event
        self._publish_order_event(EventTopics.ORDER_CREATED, order)
        
        # Queue for execution
        self._order_queue.put(order)
        
        logger.info(f"Order submitted: ID={order.order_id}, "
                   f"Symbol={order.symbol}, Type={order.order_type.value}, "
                   f"Side={order.side.value}, Quantity={order.quantity}")
        
        return order.order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            True if the cancel request was accepted, False otherwise
        """
        if not self._running:
            raise RuntimeError("ExecutionService is not running")
        
        # Find the order
        order = self.order_book.get_order(order_id)
        if not order:
            logger.warning(f"Order not found for cancellation: {order_id}")
            return False
        
        # Check if order can be cancelled
        if order.status not in [OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]:
            logger.warning(f"Cannot cancel order in status {order.status.value}: {order_id}")
            return False
        
        # Update order status
        order.status = OrderStatus.PENDING_CANCEL
        self.order_book.update_order(order)
        
        # Publish event
        self._publish_order_event(EventTopics.ORDER_CANCEL_REQUESTED, order)
        
        # Queue for cancellation
        self._cancel_queue.put(order)
        
        logger.info(f"Order cancellation requested: {order_id}")
        return True
    
    def update_order(self, order_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an order's parameters.
        
        Args:
            order_id: ID of the order to update
            updates: Dictionary of parameters to update
            
        Returns:
            True if the update request was accepted, False otherwise
        """
        if not self._running:
            raise RuntimeError("ExecutionService is not running")
        
        # Find the order
        order = self.order_book.get_order(order_id)
        if not order:
            logger.warning(f"Order not found for update: {order_id}")
            return False
        
        # Check if order can be updated
        if order.status not in [OrderStatus.PENDING, OrderStatus.OPEN]:
            logger.warning(f"Cannot update order in status {order.status.value}: {order_id}")
            return False
        
        # Apply updates to a copy of the order
        updated_order = order.copy()
        for key, value in updates.items():
            if hasattr(updated_order, key):
                setattr(updated_order, key, value)
        
        # Update status
        updated_order.status = OrderStatus.PENDING_UPDATE
        
        # Update in order book
        self.order_book.update_order(updated_order)
        
        # Publish event
        self._publish_order_event(EventTopics.ORDER_UPDATE_REQUESTED, updated_order)
        
        # Queue for update
        self._update_queue.put((order, updated_order))
        
        logger.info(f"Order update requested: {order_id}, updates: {updates}")
        return True
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """
        Get the current status of an order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order status or None if order not found
        """
        order = self.order_book.get_order(order_id)
        return order.status if order else None
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order details.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order object or None if not found
        """
        return self.order_book.get_order(order_id)
    
    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """
        Get all orders with the specified status.
        
        Args:
            status: Order status
            
        Returns:
            List of orders with the specified status
        """
        return self.order_book.get_orders_by_status(status)
    
    def get_active_orders(self) -> List[Order]:
        """
        Get all active orders (pending, open, or partially filled).
        
        Returns:
            List of active orders
        """
        return self.order_book.get_active_orders()
    
    def get_execution_metrics(self, order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get execution metrics for an order or all orders.
        
        Args:
            order_id: Order ID or None for all
            
        Returns:
            Dictionary of execution metrics
        """
        if order_id:
            return self._execution_metrics.get(order_id, {})
        else:
            return self._execution_metrics.copy()
    
    def _order_worker(self, worker_id: int) -> None:
        """
        Worker thread for processing orders.
        
        Args:
            worker_id: Worker thread ID
        """
        logger.info(f"Order worker {worker_id} started")
        
        while self._running:
            try:
                # Get order from queue
                order = self._order_queue.get(timeout=1.0)
                
                # Check for stop signal
                if order is None:
                    logger.debug(f"Order worker {worker_id} received stop signal")
                    break
                
                # Process order
                try:
                    self._process_order(order)
                except Exception as e:
                    logger.error(f"Error processing order {order.order_id}: {str(e)}", exc_info=True)
                    order.status = OrderStatus.ERROR
                    order.status_message = str(e)
                    self.order_book.update_order(order)
                    self._publish_order_event(EventTopics.ORDER_ERROR, order)
                
                # Mark as done
                self._order_queue.task_done()
                
            except queue.Empty:
                # Timeout, check if still running
                continue
            except Exception as e:
                logger.error(f"Error in order worker {worker_id}: {str(e)}", exc_info=True)
        
        logger.info(f"Order worker {worker_id} stopped")
    
    def _cancel_worker(self) -> None:
        """Worker thread for processing cancellations"""
        logger.info("Cancel worker started")
        
        while self._running:
            try:
                # Get order from queue
                order = self._cancel_queue.get(timeout=1.0)
                
                # Check for stop signal
                if order is None:
                    logger.debug("Cancel worker received stop signal")
                    break
                
                # Process cancellation
                try:
                    self._process_cancel(order)
                except Exception as e:
                    logger.error(f"Error cancelling order {order.order_id}: {str(e)}", exc_info=True)
                    order.status = OrderStatus.ERROR
                    order.status_message = f"Cancel error: {str(e)}"
                    self.order_book.update_order(order)
                    self._publish_order_event(EventTopics.ORDER_ERROR, order)
                
                # Mark as done
                self._cancel_queue.task_done()
                
            except queue.Empty:
                # Timeout, check if still running
                continue
            except Exception as e:
                logger.error(f"Error in cancel worker: {str(e)}", exc_info=True)
        
        logger.info("Cancel worker stopped")
    
    def _update_worker(self) -> None:
        """Worker thread for processing order updates"""
        logger.info("Update worker started")
        
        while self._running:
            try:
                # Get order update from queue
                update_item = self._update_queue.get(timeout=1.0)
                
                # Check for stop signal
                if update_item is None:
                    logger.debug("Update worker received stop signal")
                    break
                
                # Unpack the update
                original_order, updated_order = update_item
                
                # Process update
                try:
                    self._process_update(original_order, updated_order)
                except Exception as e:
                    logger.error(f"Error updating order {original_order.order_id}: {str(e)}", exc_info=True)
                    updated_order.status = OrderStatus.ERROR
                    updated_order.status_message = f"Update error: {str(e)}"
                    self.order_book.update_order(updated_order)
                    self._publish_order_event(EventTopics.ORDER_ERROR, updated_order)
                
                # Mark as done
                self._update_queue.task_done()
                
            except queue.Empty:
                # Timeout, check if still running
                continue
            except Exception as e:
                logger.error(f"Error in update worker: {str(e)}", exc_info=True)
        
        logger.info("Update worker stopped")
    
    def _status_polling_worker(self, interval_seconds: float) -> None:
        """
        Worker thread for polling order status.
        
        Args:
            interval_seconds: Polling interval in seconds
        """
        logger.info(f"Status polling worker started with interval {interval_seconds}s")
        
        while self._running:
            try:
                # Get active orders to poll
                active_orders = self.order_book.get_active_orders()
                
                for order in active_orders:
                    # Skip orders in certain states
                    if order.status in [OrderStatus.PENDING, OrderStatus.PENDING_CANCEL, 
                                       OrderStatus.PENDING_UPDATE, OrderStatus.ERROR]:
                        continue
                    
                    try:
                        # Get exchange handler
                        handler = self._get_exchange_handler(order.exchange_id)
                        if handler:
                            # Poll order status
                            updated_status = handler.get_order_status(order)
                            
                            # Update order if status changed
                            if updated_status and updated_status != order.status:
                                order.status = updated_status
                                self.order_book.update_order(order)
                                self._publish_order_event(EventTopics.ORDER_STATUS_UPDATED, order)
                                logger.info(f"Order status updated from polling: {order.order_id} -> {updated_status.value}")
                    
                    except Exception as e:
                        logger.error(f"Error polling status for order {order.order_id}: {str(e)}")
                
                # Sleep until next poll
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in status polling worker: {str(e)}", exc_info=True)
                # Sleep a bit to avoid spinning if there's a persistent error
                time.sleep(5)
        
        logger.info("Status polling worker stopped")
    
    def _process_order(self, order: Order) -> None:
        """
        Process an order for execution.
        
        Args:
            order: Order to process
        """
        # Skip if already processed
        if order.status != OrderStatus.PENDING:
            logger.warning(f"Order {order.order_id} already processed, status: {order.status.value}")
            return
        
        # Run pre-trade risk checks
        risk_results = self._run_pre_trade_risk_checks(order)
        
        # Check if any errors
        if any(result.level == RiskCheckLevel.ERROR for result in risk_results):
            error_checks = [result for result in risk_results if result.level == RiskCheckLevel.ERROR]
            error_messages = "; ".join(f"{result.check_name}: {result.message}" for result in error_checks)
            
            order.status = OrderStatus.REJECTED
            order.status_message = f"Failed risk checks: {error_messages}"
            self.order_book.update_order(order)
            
            self._publish_order_event(EventTopics.ORDER_REJECTED, order)
            logger.warning(f"Order {order.order_id} rejected: {error_messages}")
            return
        
        # Log warnings
        warning_checks = [result for result in risk_results if result.level == RiskCheckLevel.WARNING]
        if warning_checks:
            warning_messages = "; ".join(f"{result.check_name}: {result.message}" for result in warning_checks)
            logger.warning(f"Order {order.order_id} has risk warnings: {warning_messages}")
        
        # Determine execution strategy
        strategy = ExecutionStrategy.MARKET
        if "execution_strategy" in order.params:
            try:
                strategy = ExecutionStrategy(order.params["execution_strategy"])
            except ValueError:
                logger.warning(f"Invalid execution strategy: {order.params.get('execution_strategy')}, "
                              f"falling back to {strategy.value}")
        
        # Execute order according to strategy
        if strategy in self._strategy_handlers:
            self._strategy_handlers[strategy](order)
        else:
            logger.error(f"No handler for execution strategy: {strategy.value}")
            order.status = OrderStatus.ERROR
            order.status_message = f"Unsupported execution strategy: {strategy.value}"
            self.order_book.update_order(order)
            self._publish_order_event(EventTopics.ORDER_ERROR, order)
    
    def _process_cancel(self, order: Order) -> None:
        """
        Process an order cancellation.
        
        Args:
            order: Order to cancel
        """
        # Get exchange handler
        handler = self._get_exchange_handler(order.exchange_id)
        if not handler:
            order.status = OrderStatus.ERROR
            order.status_message = f"No handler for exchange: {order.exchange_id}"
            self.order_book.update_order(order)
            self._publish_order_event(EventTopics.ORDER_ERROR, order)
            return
        
        # In paper or simulation mode, immediately cancel
        if self.execution_mode in [ExecutionMode.PAPER, ExecutionMode.SIMULATION]:
            order.status = OrderStatus.CANCELED
            order.status_message = "Canceled (simulated)"
            self.order_book.update_order(order)
            self._publish_order_event(EventTopics.ORDER_CANCELED, order)
            return
        
        # Send cancel request to exchange
        try:
            success = handler.cancel_order(order)
            
            if success:
                # Update immediately (real status will be updated by status polling)
                order.status = OrderStatus.CANCELING
                self.order_book.update_order(order)
                self._publish_order_event(EventTopics.ORDER_CANCELING, order)
            else:
                order.status = OrderStatus.ERROR
                order.status_message = "Cancel request failed"
                self.order_book.update_order(order)
                self._publish_order_event(EventTopics.ORDER_ERROR, order)
                
        except Exception as e:
            order.status = OrderStatus.ERROR
            order.status_message = f"Cancel error: {str(e)}"
            self.order_book.update_order(order)
            self._publish_order_event(EventTopics.ORDER_ERROR, order)
    
    def _process_update(self, original_order: Order, updated_order: Order) -> None:
        """
        Process an order update.
        
        Args:
            original_order: Original order
            updated_order: Updated order
        """
        # Get exchange handler
        handler = self._get_exchange_handler(original_order.exchange_id)
        if not handler:
            updated_order.status = OrderStatus.ERROR
            updated_order.status_message = f"No handler for exchange: {original_order.exchange_id}"
            self.order_book.update_order(updated_order)
            self._publish_order_event(EventTopics.ORDER_ERROR, updated_order)
            return
        
        # In paper or simulation mode, immediately update
        if self.execution_mode in [ExecutionMode.PAPER, ExecutionMode.SIMULATION]:
            updated_order.status = OrderStatus.OPEN
            updated_order.status_message = "Updated (simulated)"
            self.order_book.update_order(updated_order)
            self._publish_order_event(EventTopics.ORDER_UPDATED, updated_order)
            return
        
        # Send update request to exchange
        try:
            success = handler.update_order(original_order, updated_order)
            
            if success:
                # Status will be updated by status polling
                updated_order.status = OrderStatus.UPDATING
                self.order_book.update_order(updated_order)
                self._publish_order_event(EventTopics.ORDER_UPDATING, updated_order)
            else:
                updated_order.status = OrderStatus.ERROR
                updated_order.status_message = "Update request failed"
                self.order_book.update_order(updated_order)
                self._publish_order_event(EventTopics.ORDER_ERROR, updated_order)
                
        except Exception as e:
            updated_order.status = OrderStatus.ERROR
            updated_order.status_message = f"Update error: {str(e)}"
            self.order_book.update_order(updated_order)
            self._publish_order_event(EventTopics.ORDER_ERROR, updated_order)
    
    def _execute_market_order(self, order: Order) -> None:
        """
        Execute a market order.
        
        Args:
            order: Order to execute
        """
        # Get exchange handler
        handler = self._get_exchange_handler(order.exchange_id)
        if not handler:
            order.status = OrderStatus.ERROR
            order.status_message = f"No handler for exchange: {order.exchange_id}"
            self.order_book.update_order(order)
            self._publish_order_event(EventTopics.ORDER_ERROR, order)
            return
        
        # In paper or simulation mode, immediately fill
        if self.execution_mode in [ExecutionMode.PAPER, ExecutionMode.SIMULATION]:
            # Get current price for simulation
            try:
                price = handler.get_simulated_price(order)
            except Exception as e:
                logger.error(f"Error getting simulated price: {str(e)}")
                price = order.limit_price or 0
            
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.average_price = price
            order.status_message = "Filled (simulated)"
            order.execution_time = datetime.utcnow()
            
            self.order_book.update_order(order)
            self._publish_order_event(EventTopics.ORDER_FILLED, order)
            
            # Run post-trade risk checks
            self._run_post_trade_risk_checks(order)
            
            return
        
        # Send order to exchange
        try:
            success = handler.submit_order(order)
            
            if success:
                # Order sent to exchange
                order.status = OrderStatus.OPEN
                self.order_book.update_order(order)
                self._publish_order_event(EventTopics.ORDER_OPEN, order)
            else:
                # Order rejected by handler
                order.status = OrderStatus.REJECTED
                order.status_message = "Rejected by exchange handler"
                self.order_book.update_order(order)
                self._publish_order_event(EventTopics.ORDER_REJECTED, order)
                
        except Exception as e:
            # Error submitting order
            order.status = OrderStatus.ERROR
            order.status_message = f"Submission error: {str(e)}"
            self.order_book.update_order(order)
            self._publish_order_event(EventTopics.ORDER_ERROR, order)

        def _execute_sniper_live(self, order_id: str, handler, target_price, max_wait_time, price_tolerance) -> None:
            """
            Execute a Sniper order in live mode.

            Args:
                order_id: Order ID
                handler: Exchange handler
                target_price: Target price to execute at
                max_wait_time: Maximum wait time in seconds
                price_tolerance: Price tolerance as a percentage
            """
            try:
                # Get order
                order = self.order_book.get_order(order_id)
                if not order or order.status not in [OrderStatus.WORKING]:
                    logger.warning(f"Cannot execute Sniper for order {order_id}, invalid status")
                    return

                # Calculate price band
                price_band = target_price * price_tolerance
                min_acceptable = target_price - price_band if order.side.is_buy() else target_price
                max_acceptable = target_price if order.side.is_buy() else target_price + price_band

                # Calculate timeout
                end_time = datetime.utcnow() + datetime.timedelta(seconds=max_wait_time)

                # Initialize variables
                poll_interval = 1.0  # seconds

                # Main sniper loop - wait for price to hit target
                while datetime.utcnow() < end_time:
                    # Check if order was canceled
                    order = self.order_book.get_order(order_id)
                    if not order or order.status in [OrderStatus.CANCELED, OrderStatus.ERROR]:
                        logger.info(
                            f"Sniper execution stopped for order {order_id}, status: {order.status.value if order else 'None'}")
                        return

                    # Get current market price
                    current_price = handler.get_current_price(order.symbol)

                    # Check if price is within acceptable range
                    if (order.side.is_buy() and current_price <= max_acceptable) or \
                            (order.side.is_sell() and current_price >= min_acceptable):
                        # Price is acceptable, execute order
                        logger.info(
                            f"Sniper price target hit for order {order_id}: target {target_price}, current {current_price}")

                        # Create and submit a market order to execute immediately
                        child_order = self.order_factory.create_market_order(
                            symbol=order.symbol,
                            side=order.side,
                            quantity=order.quantity,
                            exchange_id=order.exchange_id,
                            params={"parent_order_id": order_id, "is_sniper_execution": True}
                        )

                        # Submit to exchange
                        success = handler.submit_order(child_order)

                        if success:
                            # Wait for fill
                            filled = self._wait_for_fill(handler, child_order, 10)  # Short timeout for market order

                            if filled:
                                # Update parent order
                                order.status = OrderStatus.FILLED
                                order.filled_quantity = child_order.filled_quantity
                                order.average_price = child_order.average_price
                                order.execution_time = datetime.utcnow()
                                order.status_message = f"Sniper order executed at {child_order.average_price}"

                                self.order_book.update_order(order)
                                self._publish_order_event(EventTopics.ORDER_FILLED, order)

                                # Run post-trade risk checks
                                self._run_post_trade_risk_checks(order)
                            else:
                                # Child order not filled or only partially filled
                                if child_order.status == OrderStatus.PARTIALLY_FILLED:
                                    # Update with partial fill
                                    order.status = OrderStatus.PARTIALLY_FILLED
                                    order.filled_quantity = child_order.filled_quantity
                                    order.average_price = child_order.average_price
                                    order.status_message = f"Sniper order partially executed at {child_order.average_price}"

                                    self.order_book.update_order(order)
                                    self._publish_order_event(EventTopics.ORDER_PARTIALLY_FILLED, order)
                                else:
                                    # Failed to execute
                                    logger.warning(f"Sniper child order failed to execute for {order_id}")

                                    # Continue monitoring for next opportunity
                                    continue
                        else:
                            logger.error(f"Failed to submit Sniper execution order for {order_id}")

                            # Continue monitoring for next opportunity
                            continue

                        # If we got here with any fill, exit loop
                        if order.filled_quantity > 0:
                            return

                    # Price not acceptable yet, wait for next check
                    time.sleep(poll_interval)

                # Timeout reached without execution
                logger.warning(f"Sniper order {order_id} timed out without hitting price target")

                # Update order status
                order.status = OrderStatus.EXPIRED
                order.status_message = f"Sniper order expired without hitting price target {target_price}"
                self.order_book.update_order(order)
                self._publish_order_event(EventTopics.ORDER_EXPIRED, order)

            except Exception as e:
                logger.error(f"Error in Sniper live execution for order {order_id}: {str(e)}", exc_info=True)

                # Update order status
                order = self.order_book.get_order(order_id)
                if order:
                    order.status = OrderStatus.ERROR
                    order.status_message = f"Sniper execution error: {str(e)}"
                    self.order_book.update_order(order)
                    self._publish_order_event(EventTopics.ORDER_ERROR, order)

                    def _execute_iceberg_live(self, order_id: str, handler, display_size, price) -> None:
            """
            Execute an Iceberg order in live mode.

            Args:
                order_id: Order ID
                handler: Exchange handler
                display_size: Visible order size
                price: Limit price
            """
            try:
                # Get order
                order = self.order_book.get_order(order_id)
                if not order or order.status not in [OrderStatus.WORKING]:
                    logger.warning(f"Cannot execute Iceberg for order {order_id}, invalid status")
                    return

                # Initialize iceberg state
                remaining_quantity = order.quantity
                total_cost = 0
                executed_slices = 0

                # Main iceberg execution loop
                while remaining_quantity > 0:
                    # Check if order was canceled
                    order = self.order_book.get_order(order_id)
                    if not order or order.status in [OrderStatus.CANCELED, OrderStatus.ERROR]:
                        logger.info(
                            f"Iceberg execution stopped for order {order_id}, status: {order.status.value if order else 'None'}")
                        return

                    # Calculate slice size for this iteration
                    current_slice_size = min(display_size, remaining_quantity)

                    # Create child limit order for this slice
                    child_order = self.order_factory.create_limit_order(
                        symbol=order.symbol,
                        side=order.side,
                        quantity=current_slice_size,
                        price=price,
                        exchange_id=order.exchange_id,
                        params={"parent_order_id": order_id}
                    )

                    # Submit child order
                    try:
                        success = handler.submit_order(child_order)

                        if success:
                            # Wait for full or partial fill
                            while child_order.status in [OrderStatus.OPEN,
                                                         OrderStatus.PARTIALLY_FILLED] and child_order.filled_quantity < current_slice_size:
                                # Check if parent order was canceled
                                parent_order = self.order_book.get_order(order_id)
                                if not parent_order or parent_order.status in [OrderStatus.CANCELED, OrderStatus.ERROR]:
                                    # Cancel the child order
                                    handler.cancel_order(child_order)
                                    return

                                # Poll order status
                                updated_order = handler.get_order_status(child_order)
                                if updated_order:
                                    child_order.status = updated_order.status
                                    child_order.filled_quantity = updated_order.filled_quantity
                                    child_order.average_price = updated_order.average_price

                                # If filled, update parent and continue to next slice
                                if child_order.status == OrderStatus.FILLED:
                                    break

                                # If partially filled, we could cancel and move on, or wait for full fill
                                # For now, wait for full fill or timeout

                                # Check for timeout or market conditions
                                # In a real implementation, we might consider factors like:
                                # - Time in market
                                # - Price movement
                                # - Market condition changes

                                # For now, just sleep a bit before checking again
                                time.sleep(1)

                            # Update parent order with fill information
                            if child_order.filled_quantity > 0:
                                executed_slices += 1
                                remaining_quantity -= child_order.filled_quantity
                                filled_quantity = order.quantity - remaining_quantity

                                # Calculate new average price
                                slice_cost = child_order.filled_quantity * child_order.average_price
                                total_cost += slice_cost
                                average_price = total_cost / filled_quantity if filled_quantity > 0 else 0

                                order.filled_quantity = filled_quantity
                                order.average_price = average_price
                                order.status = OrderStatus.PARTIALLY_FILLED if remaining_quantity > 0 else OrderStatus.FILLED
                                order.status_message = f"Iceberg execution: {executed_slices} slices, {filled_quantity}/{order.quantity} filled"
                                order.params['iceberg_slices_executed'] = executed_slices

                                self.order_book.update_order(order)

                                # Publish event
                                if order.status == OrderStatus.FILLED:
                                    self._publish_order_event(EventTopics.ORDER_FILLED, order)
                                    # Run post-trade risk checks
                                    self._run_post_trade_risk_checks(order)
                                else:
                                    self._publish_order_event(EventTopics.ORDER_PARTIALLY_FILLED, order)

                                logger.info(f"Iceberg slice {executed_slices} executed for order {order_id}, "
                                            f"price: {child_order.average_price}, size: {child_order.filled_quantity}")
                        else:
                            logger.error(f"Failed to submit Iceberg slice order for {order_id}")
                            time.sleep(5)  # Wait before retrying

                    except Exception as e:
                        logger.error(f"Error executing Iceberg slice for order {order_id}: {str(e)}")
                        time.sleep(5)  # Wait before retrying

            except Exception as e:
                logger.error(f"Error in Iceberg live execution for order {order_id}: {str(e)}", exc_info=True)

                # Update order status
                order = self.order_book.get_order(order_id)
                if order:
                    order.status = OrderStatus.ERROR
                    order.status_message = f"Iceberg execution error: {str(e)}"
                    self.order_book.update_order(order)
                    self._publish_order_event(EventTopics.ORDER_ERROR, order)