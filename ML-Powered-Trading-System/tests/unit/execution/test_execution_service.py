import unittest
from unittest.mock import Mock, patch, MagicMock, call
import uuid
import datetime
from queue import Queue
from typing import Dict, List, Any

from core.event_bus import EventTopics, get_event_bus, Event
from execution.exchange.connectivity_manager import ConnectivityManager
from execution.order.order import Order, OrderStatus, OrderType, Side, TimeInForce
from execution.order.order_book import OrderBook
from execution.order.order_factory import OrderFactory
from execution.execution_service import (
    ExecutionService, ExecutionMode, ExecutionStrategy,
    RiskCheckLevel, RiskCheckResult
)


class TestExecutionService(unittest.TestCase):
    """Test the ExecutionService class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mocks
        self.mock_connectivity_manager = Mock(spec=ConnectivityManager)
        self.mock_event_bus = Mock()
        self.mock_order_book = Mock(spec=OrderBook)
        self.mock_order_factory = Mock(spec=OrderFactory)
        
        # Patch dependencies
        self.patchers = [
            patch('execution.execution_service.get_event_bus'),
            patch('execution.execution_service.OrderBook'),
            patch('execution.execution_service.OrderFactory'),
            patch('execution.execution_service.threading'),
            patch('execution.execution_service.Queue', return_value=Mock(spec=Queue)),
            patch('execution.execution_service.uuid.uuid4', return_value='test-uuid')
        ]
        
        # Start patches
        self.mocks = [patcher.start() for patcher in self.patchers]
        
        # Set up mock returns
        self.mocks[0].return_value = self.mock_event_bus
        self.mocks[1].return_value = self.mock_order_book
        self.mocks[2].return_value = self.mock_order_factory
        
        # Create test config
        self.test_config = {
            "num_order_workers": 2,
            "status_polling_interval_seconds": 5,
            "exchanges": {
                "test_exchange": {
                    "enabled": True,
                    "handler_class": "execution.exchange.default_exchange_handler.DefaultExchangeHandler"
                }
            }
        }
        
        # Create service
        self.service = ExecutionService(
            connectivity_manager=self.mock_connectivity_manager,
            config=self.test_config,
            execution_mode=ExecutionMode.PAPER
        )
        
        # Create a sample order for tests
        self.sample_order = Mock(spec=Order)
        self.sample_order.order_id = "test-order-id"
        self.sample_order.symbol = "BTC-USD"
        self.sample_order.order_type = OrderType.MARKET
        self.sample_order.side = Side.BUY
        self.sample_order.quantity = 1.0
        self.sample_order.filled_quantity = 0.0
        self.sample_order.exchange_id = "test_exchange"
        self.sample_order.status = OrderStatus.PENDING
        self.sample_order.params = {}

    def tearDown(self):
        """Tear down test fixtures."""
        for patcher in self.patchers:
            patcher.stop()

    def test_init(self):
        """Test service initialization."""
        self.assertEqual(self.service.connectivity_manager, self.mock_connectivity_manager)
        self.assertEqual(self.service.config, self.test_config)
        self.assertEqual(self.service.execution_mode, ExecutionMode.PAPER)
        
        # Check if internal components are initialized
        self.assertEqual(self.service._event_bus, self.mock_event_bus)
        self.assertEqual(self.service.order_book, self.mock_order_book)
        self.assertEqual(self.service.order_factory, self.mock_order_factory)
        
        # Check if strategy handlers are initialized
        self.assertEqual(len(self.service._strategy_handlers), 8)
        self.assertTrue(ExecutionStrategy.MARKET in self.service._strategy_handlers)
        self.assertTrue(ExecutionStrategy.LIMIT in self.service._strategy_handlers)
        self.assertTrue(ExecutionStrategy.TWAP in self.service._strategy_handlers)
        self.assertTrue(ExecutionStrategy.VWAP in self.service._strategy_handlers)
        
        # Check if risk checks are initialized
        self.assertTrue(len(self.service._pre_trade_risk_checks) > 0)
        self.assertTrue(len(self.service._post_trade_risk_checks) > 0)

    def test_start_stop(self):
        """Test starting and stopping the service."""
        # Test start
        self.service.start()
        
        # Check if threads are started
        self.assertTrue(self.service._running)
        self.assertGreater(len(self.service._worker_threads), 0)
        
        # Check if event was published
        self.mock_event_bus.publish.assert_called()
        
        # Test stop
        self.service.stop()
        
        # Check if threads are stopped
        self.assertFalse(self.service._running)
        self.assertEqual(len(self.service._worker_threads), 0)
        
        # Check if event was published
        self.assertEqual(self.mock_event_bus.publish.call_count, 2)

    def test_submit_order(self):
        """Test submitting an order."""
        # Setup
        self.service._running = True
        
        # Call service
        order_id = self.service.submit_order(self.sample_order)
        
        # Check results
        self.assertEqual(order_id, self.sample_order.order_id)
        self.assertEqual(self.sample_order.status, OrderStatus.PENDING)
        self.assertTrue(hasattr(self.sample_order, 'creation_time'))
        
        # Check if added to order book and queued
        self.mock_order_book.add_order.assert_called_once_with(self.sample_order)
        
        # Check if event was published
        self.mock_event_bus.publish.assert_called()

    def test_submit_order_not_running(self):
        """Test submitting an order when service is not running."""
        # Setup
        self.service._running = False
        
        # Verify exception is raised
        with self.assertRaises(RuntimeError):
            self.service.submit_order(self.sample_order)

    def test_cancel_order(self):
        """Test canceling an order."""
        # Setup
        self.service._running = True
        self.mock_order_book.get_order.return_value = self.sample_order
        
        # Call service
        result = self.service.cancel_order("test-order-id")
        
        # Check results
        self.assertTrue(result)
        self.assertEqual(self.sample_order.status, OrderStatus.PENDING_CANCEL)
        
        # Check if updated in order book and queued
        self.mock_order_book.update_order.assert_called_once_with(self.sample_order)
        
        # Check if event was published
        self.mock_event_bus.publish.assert_called()

    def test_cancel_order_not_found(self):
        """Test canceling an order that doesn't exist."""
        # Setup
        self.service._running = True
        self.mock_order_book.get_order.return_value = None
        
        # Call service
        result = self.service.cancel_order("non-existent-order")
        
        # Check results
        self.assertFalse(result)
        
        # Check if no updates happened
        self.mock_order_book.update_order.assert_not_called()

    def test_update_order(self):
        """Test updating an order."""
        # Setup
        self.service._running = True
        self.mock_order_book.get_order.return_value = self.sample_order
        self.sample_order.copy = Mock(return_value=self.sample_order)
        
        updates = {"limit_price": 50000.0}
        
        # Call service
        result = self.service.update_order("test-order-id", updates)
        
        # Check results
        self.assertTrue(result)
        self.assertEqual(self.sample_order.status, OrderStatus.PENDING_UPDATE)
        
        # Check if updated in order book and queued
        self.mock_order_book.update_order.assert_called_once_with(self.sample_order)
        
        # Check if event was published
        self.mock_event_bus.publish.assert_called()

    def test_get_order_status(self):
        """Test getting an order's status."""
        # Setup
        self.mock_order_book.get_order.return_value = self.sample_order
        
        # Call service
        status = self.service.get_order_status("test-order-id")
        
        # Check results
        self.assertEqual(status, self.sample_order.status)
        self.mock_order_book.get_order.assert_called_once_with("test-order-id")

    def test_get_order(self):
        """Test getting an order."""
        # Setup
        self.mock_order_book.get_order.return_value = self.sample_order
        
        # Call service
        order = self.service.get_order("test-order-id")
        
        # Check results
        self.assertEqual(order, self.sample_order)
        self.mock_order_book.get_order.assert_called_once_with("test-order-id")

    def test_get_orders_by_status(self):
        """Test getting orders by status."""
        # Setup
        expected_orders = [self.sample_order]
        self.mock_order_book.get_orders_by_status.return_value = expected_orders
        
        # Call service
        orders = self.service.get_orders_by_status(OrderStatus.PENDING)
        
        # Check results
        self.assertEqual(orders, expected_orders)
        self.mock_order_book.get_orders_by_status.assert_called_once_with(OrderStatus.PENDING)

    def test_get_active_orders(self):
        """Test getting active orders."""
        # Setup
        expected_orders = [self.sample_order]
        self.mock_order_book.get_active_orders.return_value = expected_orders
        
        # Call service
        orders = self.service.get_active_orders()
        
        # Check results
        self.assertEqual(orders, expected_orders)
        self.mock_order_book.get_active_orders.assert_called_once()

    def test_get_execution_metrics(self):
        """Test getting execution metrics."""
        # Setup
        self.service._execution_metrics = {
            "test-order-id": {"slippage": 0.01}
        }
        
        # Call service - for a specific order
        metrics = self.service.get_execution_metrics("test-order-id")
        
        # Check results
        self.assertEqual(metrics, {"slippage": 0.01})
        
        # Call service - for all orders
        metrics = self.service.get_execution_metrics()
        
        # Check results
        self.assertEqual(metrics, {"test-order-id": {"slippage": 0.01}})

    @patch('execution.execution_service.ExecutionService._process_order')
    def test_order_worker(self, mock_process_order):
        """Test the order worker thread."""
        # Setup mocks
        mock_queue = Mock()
        mock_queue.get.side_effect = [self.sample_order, None]  # First return sample order, then signal to stop
        self.service._order_queue = mock_queue
        self.service._running = True
        
        # Run worker
        self.service._order_worker(0)
        
        # Check if order was processed
        mock_process_order.assert_called_once_with(self.sample_order)
        self.assertEqual(mock_queue.task_done.call_count, 1)

    @patch('execution.execution_service.ExecutionService._run_pre_trade_risk_checks')
    @patch('execution.execution_service.ExecutionService._execute_market_order')
    def test_process_order(self, mock_execute_market, mock_run_risk_checks):
        """Test processing an order."""
        # Setup
        mock_run_risk_checks.return_value = []  # No risk issues
        
        # Call service
        self.service._process_order(self.sample_order)
        
        # Check if risk checks were run
        mock_run_risk_checks.assert_called_once_with(self.sample_order)
        
        # Check if market order was executed (default strategy)
        mock_execute_market.assert_called_once_with(self.sample_order)

    @patch('execution.execution_service.ExecutionService._run_pre_trade_risk_checks')
    @patch('execution.execution_service.ExecutionService._execute_vwap_order')
    def test_process_order_vwap_strategy(self, mock_execute_vwap, mock_run_risk_checks):
        """Test processing an order with VWAP strategy."""
        # Setup
        mock_run_risk_checks.return_value = []  # No risk issues
        self.sample_order.params = {"execution_strategy": "vwap"}
        
        # Call service
        self.service._process_order(self.sample_order)
        
        # Check if risk checks were run
        mock_run_risk_checks.assert_called_once_with(self.sample_order)
        
        # Check if VWAP order was executed
        mock_execute_vwap.assert_called_once_with(self.sample_order)

    @patch('execution.execution_service.ExecutionService._run_pre_trade_risk_checks')
    @patch('execution.execution_service.ExecutionService._publish_order_event')
    def test_process_order_risk_error(self, mock_publish_event, mock_run_risk_checks):
        """Test processing an order that fails risk checks."""
        # Setup - Create a risk check error
        error_result = RiskCheckResult(
            level=RiskCheckLevel.ERROR,
            message="Order size exceeds limit",
            check_name="size_limit_check"
        )
        mock_run_risk_checks.return_value = [error_result]
        
        # Call service
        self.service._process_order(self.sample_order)
        
        # Check if order was rejected
        self.assertEqual(self.sample_order.status, OrderStatus.REJECTED)
        self.mock_order_book.update_order.assert_called_once_with(self.sample_order)
        
        # Check if event was published
        mock_publish_event.assert_called_once_with(EventTopics.ORDER_REJECTED, self.sample_order)

    @patch('execution.execution_service.ExecutionService._get_exchange_handler')
    @patch('execution.execution_service.ExecutionService._publish_order_event')
    def test_execute_market_order_paper_mode(self, mock_publish_event, mock_get_handler):
        """Test executing a market order in paper trading mode."""
        # Setup
        mock_handler = Mock()
        mock_handler.get_simulated_price.return_value = 50000.0
        mock_get_handler.return_value = mock_handler
        
        # Call service
        self.service._execute_market_order(self.sample_order)
        
        # Check if handler was used
        mock_get_handler.assert_called_once_with(self.sample_order.exchange_id)
        mock_handler.get_simulated_price.assert_called_once_with(self.sample_order)
        
        # Check if order was filled
        self.assertEqual(self.sample_order.status, OrderStatus.FILLED)
        self.assertEqual(self.sample_order.filled_quantity, self.sample_order.quantity)
        self.assertEqual(self.sample_order.average_price, 50000.0)
        self.mock_order_book.update_order.assert_called_once_with(self.sample_order)
        
        # Check if event was published
        mock_publish_event.assert_called_once_with(EventTopics.ORDER_FILLED, self.sample_order)

    @patch('execution.execution_service.ExecutionService._get_exchange_handler')
    @patch('execution.execution_service.ExecutionService._publish_order_event')
    @patch('execution.execution_service.ExecutionMode')
    def test_execute_market_order_live_mode(self, mock_mode, mock_publish_event, mock_get_handler):
        """Test executing a market order in live trading mode."""
        # Setup
        self.service.execution_mode = ExecutionMode.LIVE
        
        mock_handler = Mock()
        mock_handler.submit_order.return_value = True
        mock_get_handler.return_value = mock_handler
        
        # Call service
        self.service._execute_market_order(self.sample_order)
        
        # Check if handler was used
        mock_get_handler.assert_called_once_with(self.sample_order.exchange_id)
        mock_handler.submit_order.assert_called_once_with(self.sample_order)
        
        # Check if order status was updated
        self.assertEqual(self.sample_order.status, OrderStatus.OPEN)
        self.mock_order_book.update_order.assert_called_once_with(self.sample_order)
        
        # Check if event was published
        mock_publish_event.assert_called_once_with(EventTopics.ORDER_OPEN, self.sample_order)

    def test_exchange_handler_management(self):
        """Test the exchange handler management."""
        # Call service
        handler = self.service._get_exchange_handler("test_exchange")
        
        # Check if handler exists
        self.assertIsNotNone(handler)
        self.assertEqual(handler.exchange_id, "test_exchange")
        
        # Test non-existent handler
        handler = self.service._get_exchange_handler("non_existent_exchange")
        self.assertIsNone(handler)

    @patch('execution.execution_service.datetime')
    @patch('execution.execution_service.time')
    @patch('execution.execution_service.ExecutionService._get_exchange_handler')
    @patch('execution.execution_service.ExecutionService._publish_order_event')
    @patch('execution.execution_service.ExecutionService._run_post_trade_risk_checks')
    def test_execute_vwap_simulation(self, mock_post_risk, mock_publish_event, mock_get_handler, 
                                    mock_time, mock_datetime):
        """Test VWAP order simulation."""
        # Setup mocks
        mock_handler = Mock()
        mock_handler.get_simulated_price.return_value = 50000.0
        
        # Setup datetime mock for consistent behavior
        start_time = datetime.datetime(2023, 1, 1, 10, 0, 0)
        end_time = datetime.datetime(2023, 1, 1, 11, 0, 0)
        now = datetime.datetime(2023, 1, 1, 9, 55, 0)
        mock_datetime.utcnow.return_value = now
        
        # Setup order
        self.sample_order.status = OrderStatus.WORKING
        self.sample_order.params = {
            'vwap_slices_executed': 0,
            'vwap_slice_times': []
        }
        self.mock_order_book.get_order.return_value = self.sample_order
        
        # Call service with slice sizes
        slice_sizes = [0.3, 0.4, 0.3]  # 30%, 40%, 30% of total quantity
        self.service._execute_vwap_simulation(
            order_id=self.sample_order.order_id,
            handler=mock_handler,
            start_time=start_time,
            end_time=end_time,
            slice_sizes=slice_sizes
        )
        
        # Check if all slices were executed
        self.assertEqual(self.sample_order.status, OrderStatus.FILLED)
        self.assertEqual(self.sample_order.filled_quantity, self.sample_order.quantity)
        self.assertEqual(self.sample_order.average_price, 50000.0)
        
        # Check post-trade risk check was called
        mock_post_risk.assert_called_once_with(self.sample_order)
        
        # Check if event was published for final fill
        mock_publish_event.assert_called_with(EventTopics.ORDER_FILLED, self.sample_order)

    def test_run_pre_trade_risk_checks(self):
        """Test running pre-trade risk checks."""
        # Setup - Create mock risk check functions
        def mock_check1(order):
            return RiskCheckResult(
                level=RiskCheckLevel.INFO,
                message="Info check",
                check_name="check1"
            )
            
        def mock_check2(order):
            return RiskCheckResult(
                level=RiskCheckLevel.WARNING,
                message="Warning check",
                check_name="check2"
            )
        
        self.service._pre_trade_risk_checks = [mock_check1, mock_check2]
        
        # Call service
        results = self.service._run_pre_trade_risk_checks(self.sample_order)
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].level, RiskCheckLevel.INFO)
        self.assertEqual(results[1].level, RiskCheckLevel.WARNING)
        
    @patch('execution.execution_service.create_event')
    def test_publish_order_event(self, mock_create_event):
        """Test publishing order events."""
        # Setup
        mock_event = Mock(spec=Event)
        mock_create_event.return_value = mock_event
        
        # Call service
        self.service._publish_order_event(EventTopics.ORDER_CREATED, self.sample_order)
        
        # Check if event was created and published
        mock_create_event.assert_called_once()
        self.mock_event_bus.publish.assert_called_once_with(mock_event)


if __name__ == '__main__':
    unittest.main()