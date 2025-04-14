import unittest
from unittest.mock import MagicMock, patch
import threading
import time
from datetime import datetime, timedelta

from execution.order.order import Order, OrderStatus, OrderSide, OrderType
from execution.order.order_factory import OrderFactory
from execution.algorithm.execution_algorithm import ExecutionProgress, AlgorithmGoal
from execution.algorithm.twap_algorithm import TwapAlgorithm


class TestTwapAlgorithm(unittest.TestCase):
    """Test suite for the TWAP execution algorithm."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.order_factory = MagicMock(spec=OrderFactory)
        
        # Create a standard test configuration
        self.config = {
            "default_duration_minutes": 10,
            "default_num_slices": 5,
            "randomize_sizes": False,
            "randomize_times": False,
            "size_variance_percent": 5,
            "time_variance_percent": 5
        }
        
        # Initialize the TWAP algorithm
        self.twap = TwapAlgorithm(self.order_factory, self.config)
        
        # Create a sample test order
        self.test_order = MagicMock(spec=Order)
        self.test_order.order_id = "test-order-001"
        self.test_order.symbol = "BTC-USD"
        self.test_order.side = OrderSide.BUY
        self.test_order.order_type = OrderType.MARKET
        self.test_order.quantity = 10.0
        self.test_order.price = 50000.0
        self.test_order.exchange_id = "BINANCE"
        self.test_order.exchange_account = "default"
        self.test_order.params = {}

    def tearDown(self):
        """Clean up after each test method."""
        # Cancel any running executions
        for order_id in list(self.twap._execution_threads.keys()):
            self.twap.cancel_execution(order_id)
        
        # Wait for all threads to complete
        for thread in self.twap._execution_threads.values():
            if thread.is_alive():
                thread.join(timeout=0.5)

    def test_init(self):
        """Test initialization of the TWAP algorithm."""
        self.assertEqual(self.twap.algorithm_name, "TWAP")
        self.assertEqual(self.twap.default_duration_minutes, 10)
        self.assertEqual(self.twap.default_num_slices, 5)
        self.assertFalse(self.twap.randomize_sizes)
        self.assertFalse(self.twap.randomize_times)
        self.assertEqual(self.twap.size_variance_percent, 5)
        self.assertEqual(self.twap.time_variance_percent, 5)

    def test_can_execute_order(self):
        """Test the can_execute_order method."""
        # Test valid order
        can_execute, reason = self.twap.can_execute_order(self.test_order)
        self.assertTrue(can_execute)
        
        # Test invalid order type
        invalid_order = MagicMock(spec=Order)
        invalid_order.order_type = OrderType.STOP_LIMIT
        invalid_order.quantity = 10.0
        invalid_order.exchange_id = "BINANCE"
        invalid_order.symbol = "BTC-USD"
        
        can_execute, reason = self.twap.can_execute_order(invalid_order)
        self.assertFalse(can_execute)
        self.assertIn("Unsupported order type", reason)
        
        # Test invalid quantity
        invalid_order.order_type = OrderType.MARKET
        invalid_order.quantity = 0.0
        
        can_execute, reason = self.twap.can_execute_order(invalid_order)
        self.assertFalse(can_execute)
        self.assertEqual("Invalid order quantity", reason)

    def test_generate_slice_sizes_equal(self):
        """Test generating equal slice sizes."""
        total_quantity = 10.0
        num_slices = 5
        
        slice_sizes = self.twap._generate_slice_sizes(total_quantity, num_slices)
        
        # Check correct number of slices
        self.assertEqual(len(slice_sizes), num_slices)
        
        # Check each slice size is equal
        expected_size = total_quantity / num_slices
        for size in slice_sizes:
            self.assertAlmostEqual(size, expected_size)
        
        # Check total quantity is preserved
        self.assertAlmostEqual(sum(slice_sizes), total_quantity)

    def test_generate_slice_sizes_randomized(self):
        """Test generating randomized slice sizes."""
        # Enable randomization
        self.twap.randomize_sizes = True
        self.twap.size_variance_percent = 20
        
        total_quantity = 100.0
        num_slices = 10
        
        slice_sizes = self.twap._generate_slice_sizes(total_quantity, num_slices)
        
        # Check correct number of slices
        self.assertEqual(len(slice_sizes), num_slices)
        
        # Check no negative sizes
        for size in slice_sizes:
            self.assertGreaterEqual(size, 0.0)
        
        # Check total quantity is preserved
        self.assertAlmostEqual(sum(slice_sizes), total_quantity)
        
        # Check that sizes are not all equal (with high probability)
        expected_size = total_quantity / num_slices
        has_different_sizes = any(abs(size - expected_size) > 0.01 for size in slice_sizes)
        self.assertTrue(has_different_sizes)

    def test_calculate_slice_times_equal(self):
        """Test calculating equally spaced slice times."""
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=10)
        num_slices = 5
        
        slice_times = self.twap._calculate_slice_times(start_time, end_time, num_slices)
        
        # Check correct number of times
        self.assertEqual(len(slice_times), num_slices)
        
        # Check start and end times
        self.assertEqual(slice_times[0], start_time)
        self.assertEqual(slice_times[-1], end_time)
        
        # Check times are in order
        for i in range(1, len(slice_times)):
            self.assertLess(slice_times[i-1], slice_times[i])
        
        # Check intervals are equal
        expected_interval = (end_time - start_time) / (num_slices - 1)
        for i in range(1, len(slice_times)):
            interval = slice_times[i] - slice_times[i-1]
            self.assertAlmostEqual(interval.total_seconds(), expected_interval.total_seconds(), delta=0.001)

    def test_calculate_slice_times_randomized(self):
        """Test calculating randomized slice times."""
        # Enable randomization
        self.twap.randomize_times = True
        self.twap.time_variance_percent = 20
        
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=10)
        num_slices = 5
        
        slice_times = self.twap._calculate_slice_times(start_time, end_time, num_slices)
        
        # Check correct number of times
        self.assertEqual(len(slice_times), num_slices)
        
        # Check start and end times (these should not be randomized)
        self.assertEqual(slice_times[0], start_time)
        self.assertEqual(slice_times[-1], end_time)
        
        # Check times are in order
        for i in range(1, len(slice_times)):
            self.assertLess(slice_times[i-1], slice_times[i])
        
        # With randomization, middle times should not have equal intervals
        expected_interval = (end_time - start_time) / (num_slices - 1)
        has_different_intervals = False
        
        for i in range(1, len(slice_times) - 1):
            interval1 = slice_times[i] - slice_times[i-1]
            interval2 = slice_times[i+1] - slice_times[i]
            
            if abs(interval1.total_seconds() - interval2.total_seconds()) > 0.1:
                has_different_intervals = True
                break
                
        self.assertTrue(has_different_intervals)

    def test_create_child_order_market(self):
        """Test creating child market orders for TWAP slices."""
        # Mock the order factory's create_market_order method
        mock_child_order = MagicMock(spec=Order)
        self.order_factory.create_market_order.return_value = mock_child_order
        
        # Set parent order as market order
        self.test_order.order_type = OrderType.MARKET
        
        # Create a child order
        child_order = self.twap._create_child_order(self.test_order, 2.0, 0)
        
        # Verify the order factory was called correctly
        self.order_factory.create_market_order.assert_called_once()
        args, kwargs = self.order_factory.create_market_order.call_args
        
        self.assertEqual(kwargs["symbol"], self.test_order.symbol)
        self.assertEqual(kwargs["side"], self.test_order.side)
        self.assertEqual(kwargs["quantity"], 2.0)
        self.assertEqual(kwargs["exchange_id"], self.test_order.exchange_id)
        self.assertEqual(kwargs["exchange_account"], self.test_order.exchange_account)
        self.assertEqual(kwargs["params"]["parent_order_id"], self.test_order.order_id)
        self.assertEqual(kwargs["params"]["twap_slice_index"], 0)

    @patch('execution.algorithm.twap_algorithm.time')
    def test_execute_child_order(self, mock_time):
        """Test executing a child order."""
        # Mock the child order
        child_order = MagicMock(spec=Order)
        
        # Execute the child order and get the execution price
        execution_price = self.twap._execute_child_order(child_order)
        
        # Verify execution price is within expected range (99.5 to 100.5)
        self.assertGreaterEqual(execution_price, 99.5)
        self.assertLessEqual(execution_price, 100.5)
        
        # Verify time.sleep was called
        mock_time.sleep.assert_called_once()

    @patch('execution.algorithm.twap_algorithm.threading.Thread')
    def test_start_execution(self, mock_thread):
        """Test starting TWAP execution."""
        # Configure the thread mock
        mock_thread_instance = MagicMock()
        mock_thread.return_value = mock_thread_instance
        
        # Start execution
        result = self.twap.start_execution(self.test_order)
        
        # Verify execution was started
        self.assertTrue(result)
        
        # Verify thread was created with correct arguments
        mock_thread.assert_called_once()
        args, kwargs = mock_thread.call_args
        
        self.assertEqual(kwargs["target"], self.twap._execute_twap)
        self.assertEqual(kwargs["args"][0], self.test_order)
        self.assertTrue(isinstance(kwargs["args"][1], datetime))  # start_time
        self.assertTrue(isinstance(kwargs["args"][2], datetime))  # end_time
        self.assertTrue(isinstance(kwargs["args"][3], list))      # slice_sizes
        
        # Verify thread was started
        mock_thread_instance.start.assert_called_once()
        
        # Verify progress was initialized
        self.assertIn(self.test_order.order_id, self.twap.progress)
        progress = self.twap.progress[self.test_order.order_id]
        self.assertEqual(progress.total_quantity, self.test_order.quantity)
        self.assertEqual(progress.slices_total, self.config["default_num_slices"])

    def test_pause_resume_execution(self):
        """Test pausing and resuming TWAP execution."""
        # Mock a running thread
        thread_mock = MagicMock()
        thread_mock.is_alive.return_value = True
        self.twap._execution_threads[self.test_order.order_id] = thread_mock
        
        # Pause execution
        result = self.twap.pause_execution(self.test_order.order_id)
        self.assertTrue(result)
        self.assertIn(self.test_order.order_id, self.twap._execution_paused)
        
        # Try to pause again (should still return True as execution exists)
        result = self.twap.pause_execution(self.test_order.order_id)
        self.assertTrue(result)
        
        # Resume execution
        result = self.twap.resume_execution(self.test_order.order_id)
        self.assertTrue(result)
        self.assertNotIn(self.test_order.order_id, self.twap._execution_paused)
        
        # Try to resume again (should return False as not paused)
        result = self.twap.resume_execution(self.test_order.order_id)
        self.assertFalse(result)

    def test_cancel_execution(self):
        """Test canceling TWAP execution."""
        # Mock a running thread
        thread_mock = MagicMock()
        thread_mock.is_alive.return_value = True
        self.twap._execution_threads[self.test_order.order_id] = thread_mock
        
        # Cancel execution
        result = self.twap.cancel_execution(self.test_order.order_id)
        self.assertTrue(result)
        self.assertIn(self.test_order.order_id, self.twap._execution_stop_signals)
        
        # Try to cancel non-existent execution
        result = self.twap.cancel_execution("non-existent-order")
        self.assertFalse(result)

    @patch('execution.algorithm.twap_algorithm.Order')
    def test_update_parameters(self, mock_order_class):
        """Test updating TWAP execution parameters."""
        # Mock the progress
        progress = ExecutionProgress(total_quantity=10.0)
        progress.executed_quantity = 4.0
        progress.remaining_quantity = 6.0
        self.twap.progress[self.test_order.order_id] = progress
        
        # Mock a running thread
        thread_mock = MagicMock()
        thread_mock.is_alive.return_value = True
        self.twap._execution_threads[self.test_order.order_id] = thread_mock
        
        # Mock the Order.from_dict method
        new_order = MagicMock(spec=Order)
        mock_order_class.from_dict.return_value = new_order
        
        # Create new parameters
        new_params = {
            "original_order": {
                "order_id": "test-order-001",
                "symbol": "BTC-USD",
                "side": "BUY",
                "order_type": "MARKET",
                "quantity": 10.0,
                "exchange_id": "BINANCE"
            },
            "twap_duration_minutes": 15,
            "twap_num_slices": 3
        }
        
        # Mock start_execution
        self.twap.start_execution = MagicMock(return_value=True)
        
        # Update parameters
        result = self.twap.update_parameters(self.test_order.order_id, new_params)
        self.assertTrue(result)
        
        # Verify Order.from_dict was called with correct params
        mock_order_class.from_dict.assert_called_once()
        args, kwargs = mock_order_class.from_dict.call_args
        params_dict = args[0]
        
        # Check quantity was updated to remaining quantity
        self.assertEqual(params_dict["quantity"], 6.0)
        
        # Check new parameters were added
        self.assertEqual(params_dict["twap_duration_minutes"], 15)
        self.assertEqual(params_dict["twap_num_slices"], 3)
        
        # Verify start_execution was called with new order
        self.twap.start_execution.assert_called_once_with(new_order)

    def test_get_primary_goal(self):
        """Test getting the primary goal of TWAP."""
        goal = self.twap._get_primary_goal()
        self.assertEqual(goal, AlgorithmGoal.CONSISTENCY)

    @patch('execution.algorithm.twap_algorithm.time')
    @patch('execution.algorithm.twap_algorithm.datetime')
    def test_execute_twap(self, mock_datetime, mock_time):
        """Test the _execute_twap method."""
        # Set up mocks
        now = datetime.utcnow()
        future = now + timedelta(minutes=1)
        
        # Mock datetime.utcnow to return controlled values
        mock_datetime.utcnow.side_effect = [
            now,  # Initial call
            now + timedelta(seconds=10),  # Check for wait time
            now + timedelta(seconds=20),  # After sleep
            now + timedelta(seconds=30),  # Check for next wait time
            now + timedelta(seconds=40),  # After sleep
            now + timedelta(seconds=50),  # Final check
        ]
        
        # Set up test parameters
        start_time = now
        end_time = now + timedelta(minutes=2)
        slice_sizes = [2.0, 3.0, 5.0]
        
        # Mock _calculate_slice_times
        self.twap._calculate_slice_times = MagicMock(return_value=[
            now,  # First slice immediately
            now + timedelta(seconds=30),  # Second slice after 30 seconds
            now + timedelta(seconds=60),  # Third slice after 60 seconds
        ])
        
        # Mock _create_child_order and _execute_child_order
        child_order = MagicMock(spec=Order)
        self.twap._create_child_order = MagicMock(return_value=child_order)
        self.twap._execute_child_order = MagicMock(side_effect=[100.0, 101.0, 102.0])
        
        # Call _execute_twap
        self.twap._execute_twap(self.test_order, start_time, end_time, slice_sizes)
        
        # Verify _calculate_slice_times was called correctly
        self.twap._calculate_slice_times.assert_called_once_with(start_time, end_time, 3)
        
        # Verify _create_child_order was called for each slice
        self.assertEqual(self.twap._create_child_order.call_count, 3)
        
        # Verify _execute_child_order was called for each slice
        self.assertEqual(self.twap._execute_child_order.call_count, 3)
        
        # Check progress updates
        progress = self.twap.progress[self.test_order.order_id]
        self.assertTrue(progress.is_complete)
        self.assertEqual(progress.slices_completed, 3)
        self.assertEqual(progress.executed_quantity, sum(slice_sizes))
        
        # Verify weighted average price calculation
        expected_avg_price = (2.0 * 100.0 + 3.0 * 101.0 + 5.0 * 102.0) / 10.0
        self.assertAlmostEqual(progress.average_price, expected_avg_price)


if __name__ == "__main__":
    unittest.main()