import unittest
from unittest.mock import MagicMock, patch
import datetime
from datetime import timedelta
import threading

from execution.order.order import Order, OrderStatus, OrderSide, OrderType, TimeInForce
from execution.order.order_factory import OrderFactory
from execution.algorithm.execution_algorithm import ExecutionProgress, AlgorithmGoal
from execution.exchange.order.vwap_algorithm import VwapAlgorithm, VolumeProfile


class TestVolumeProfile(unittest.TestCase):
    """Test cases for the VolumeProfile class"""

    def test_default_profile_initialization(self):
        """Test that default profile is initialized correctly"""
        profile = VolumeProfile(buckets=12)
        self.assertEqual(len(profile.percentages), 12)
        self.assertAlmostEqual(sum(profile.percentages), 1.0, places=6)

    def test_custom_bucket_count(self):
        """Test profile with custom bucket count"""
        profile = VolumeProfile(buckets=6)
        self.assertEqual(len(profile.percentages), 6)
        self.assertAlmostEqual(sum(profile.percentages), 1.0, places=6)

    def test_from_dict(self):
        """Test creating profile from dictionary"""
        data = {
            "buckets": 4,
            "percentages": [0.1, 0.4, 0.3, 0.2]
        }
        profile = VolumeProfile.from_dict(data)
        self.assertEqual(profile.buckets, 4)
        self.assertEqual(profile.percentages, [0.1, 0.4, 0.3, 0.2])

    def test_to_dict(self):
        """Test converting profile to dictionary"""
        profile = VolumeProfile(buckets=4)
        profile.percentages = [0.1, 0.4, 0.3, 0.2]
        data = profile.to_dict()
        self.assertEqual(data["buckets"], 4)
        self.assertEqual(data["percentages"], [0.1, 0.4, 0.3, 0.2])

    def test_get_bucket_for_time(self):
        """Test bucket calculation based on time"""
        profile = VolumeProfile(buckets=10)
        
        # Define time window
        start_time = datetime.datetime(2023, 1, 1, 9, 0, 0)  # 9:00 AM
        end_time = datetime.datetime(2023, 1, 1, 10, 0, 0)   # 10:00 AM
        
        # Test different times
        current_time = start_time
        self.assertEqual(profile.get_bucket_for_time(start_time, end_time, current_time), 0)
        
        current_time = start_time + timedelta(minutes=6)  # 9:06 AM (10% of the way)
        self.assertEqual(profile.get_bucket_for_time(start_time, end_time, current_time), 1)
        
        current_time = start_time + timedelta(minutes=30)  # 9:30 AM (50% of the way)
        self.assertEqual(profile.get_bucket_for_time(start_time, end_time, current_time), 5)
        
        current_time = end_time  # 10:00 AM (100% of the way)
        self.assertEqual(profile.get_bucket_for_time(start_time, end_time, current_time), 9)
        
        # Test edge cases
        current_time = start_time - timedelta(minutes=5)  # Before start time
        self.assertEqual(profile.get_bucket_for_time(start_time, end_time, current_time), 0)
        
        current_time = end_time + timedelta(minutes=5)  # After end time
        self.assertEqual(profile.get_bucket_for_time(start_time, end_time, current_time), 9)

    def test_calculate_slice_sizes(self):
        """Test slice size calculation"""
        profile = VolumeProfile(buckets=4)
        profile.percentages = [0.1, 0.4, 0.3, 0.2]
        
        slices = profile.calculate_slice_sizes(1000)
        
        self.assertEqual(len(slices), 4)
        self.assertEqual(slices, [100, 400, 300, 200])
        self.assertEqual(sum(slices), 1000)


class TestVwapAlgorithm(unittest.TestCase):
    """Test cases for the VwapAlgorithm class"""

    def setUp(self):
        """Set up test fixtures"""
        self.order_factory = MagicMock(spec=OrderFactory)
        self.config = {
            "default_duration_minutes": 30,
            "default_num_buckets": 6,
            "randomize_within_bucket": True,
            "randomize_sizes": True,
            "size_variance_percent": 10,
            "symbol_profiles": {
                "AAPL": {
                    "buckets": 4,
                    "percentages": [0.25, 0.25, 0.25, 0.25]
                }
            }
        }
        
        self.vwap = VwapAlgorithm(self.order_factory, self.config)
        
    def test_initialization(self):
        """Test VWAP algorithm initialization"""
        self.assertEqual(self.vwap.name, "VWAP")
        self.assertEqual(self.vwap.default_duration_minutes, 30)
        self.assertEqual(self.vwap.default_num_buckets, 6)
        self.assertEqual(self.vwap.randomize_within_bucket, True)
        self.assertEqual(self.vwap.randomize_sizes, True)
        self.assertEqual(self.vwap.size_variance_percent, 10)
        
        # Check symbol profiles
        self.assertIn("AAPL", self.vwap.symbol_profiles)
        aapl_profile = self.vwap.symbol_profiles["AAPL"]
        self.assertEqual(aapl_profile.buckets, 4)
        self.assertEqual(aapl_profile.percentages, [0.25, 0.25, 0.25, 0.25])

    def test_get_profile_for_symbol(self):
        """Test retrieval of volume profile for symbols"""
        # Symbol with custom profile
        profile = self.vwap._get_profile_for_symbol("AAPL", 4)
        self.assertEqual(profile.buckets, 4)
        self.assertEqual(profile.percentages, [0.25, 0.25, 0.25, 0.25])
        
        # Symbol with custom profile but different bucket count
        profile = self.vwap._get_profile_for_symbol("AAPL", 8)
        self.assertEqual(profile.buckets, 8)
        self.assertNotEqual(profile.percentages, [0.25, 0.25, 0.25, 0.25])
        
        # Symbol without custom profile
        profile = self.vwap._get_profile_for_symbol("MSFT", 6)
        self.assertEqual(profile.buckets, 6)

    def test_randomize_slice_sizes(self):
        """Test randomization of slice sizes"""
        original_slices = [100, 200, 300, 400]
        total = sum(original_slices)
        
        # Test with randomization enabled
        self.vwap.randomize_sizes = True
        self.vwap.size_variance_percent = 10
        
        randomized = self.vwap._randomize_slice_sizes(original_slices)
        
        # Check that count is preserved
        self.assertEqual(len(randomized), len(original_slices))
        
        # Check that total is preserved
        self.assertAlmostEqual(sum(randomized), total, places=4)
        
        # Check that values are different
        different = False
        for i in range(len(original_slices)):
            if abs(original_slices[i] - randomized[i]) > 0.001:
                different = True
                break
        
        # This could occasionally fail by chance, but it's unlikely
        self.assertTrue(different, "Randomization did not change any values")

    def test_can_execute_order(self):
        """Test order validation"""
        # Valid market order
        market_order = MagicMock(spec=Order)
        market_order.order_type = OrderType.MARKET
        market_order.quantity = 100
        market_order.exchange_id = "NASDAQ"
        market_order.symbol = "AAPL"
        
        # The algorithm supports all venues and assets by default
        self.vwap.is_supported_venue = MagicMock(return_value=True)
        self.vwap.is_supported_asset = MagicMock(return_value=True)
        
        can_execute, _ = self.vwap.can_execute_order(market_order)
        self.assertTrue(can_execute)
        
        # Valid limit order
        limit_order = MagicMock(spec=Order)
        limit_order.order_type = OrderType.LIMIT
        limit_order.quantity = 100
        limit_order.exchange_id = "NASDAQ"
        limit_order.symbol = "AAPL"
        
        can_execute, _ = self.vwap.can_execute_order(limit_order)
        self.assertTrue(can_execute)
        
        # Invalid order type
        invalid_order = MagicMock(spec=Order)
        invalid_order.order_type = OrderType.STOP
        invalid_order.quantity = 100
        
        can_execute, reason = self.vwap.can_execute_order(invalid_order)
        self.assertFalse(can_execute)
        self.assertIn("Unsupported order type", reason)
        
        # Invalid quantity
        invalid_order = MagicMock(spec=Order)
        invalid_order.order_type = OrderType.MARKET
        invalid_order.quantity = 0
        
        can_execute, reason = self.vwap.can_execute_order(invalid_order)
        self.assertFalse(can_execute)
        self.assertIn("Invalid order quantity", reason)
        
        # Unsupported venue
        self.vwap.is_supported_venue = MagicMock(return_value=False)
        
        can_execute, reason = self.vwap.can_execute_order(market_order)
        self.assertFalse(can_execute)
        self.assertIn("Unsupported venue", reason)
        
        # Unsupported asset
        self.vwap.is_supported_venue = MagicMock(return_value=True)
        self.vwap.is_supported_asset = MagicMock(return_value=False)
        
        can_execute, reason = self.vwap.can_execute_order(market_order)
        self.assertFalse(can_execute)
        self.assertIn("Unsupported asset", reason)

    @patch('execution.exchange.order.vwap_algorithm.datetime')
    def test_start_execution(self, mock_datetime):
        """Test starting VWAP execution"""
        # Set up mock for datetime.utcnow
        mock_now = datetime.datetime(2023, 1, 1, 9, 0, 0)
        mock_datetime.utcnow.return_value = mock_now
        
        # Create test order
        order = MagicMock(spec=Order)
        order.order_id = "test_order_1"
        order.order_type = OrderType.MARKET
        order.quantity = 1000
        order.exchange_id = "NASDAQ"
        order.symbol = "AAPL"
        order.params = {}
        
        # Patches for thread handling
        with patch.object(threading, 'Thread') as mock_thread:
            # Configure mocks
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance
            
            # Order can be executed
            self.vwap.can_execute_order = MagicMock(return_value=(True, ""))
            
            # Start execution
            result = self.vwap.start_execution(order)
            
            # Verify thread was created and started
            self.assertTrue(result)
            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()
            
            # Check progress was created
            self.assertIn(order.order_id, self.vwap.progress)
            progress = self.vwap.progress[order.order_id]
            self.assertEqual(progress.total_quantity, 1000)
            
            # Verify execution thread was saved
            self.assertIn(order.order_id, self.vwap._execution_threads)
            self.assertEqual(self.vwap._execution_threads[order.order_id], mock_thread_instance)

    def test_start_execution_already_running(self):
        """Test starting execution when already running"""
        order_id = "test_order_2"
        
        # Setup a mock thread that is alive
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        
        with self.vwap._lock:
            self.vwap._execution_threads[order_id] = mock_thread
        
        # Create test order
        order = MagicMock(spec=Order)
        order.order_id = order_id
        
        # Try to start execution
        result = self.vwap.start_execution(order)
        
        # Should fail because already running
        self.assertFalse(result)

    def test_start_execution_invalid_order(self):
        """Test starting execution with invalid order"""
        # Create test order
        order = MagicMock(spec=Order)
        order.order_id = "test_order_3"
        
        # Order cannot be executed
        self.vwap.can_execute_order = MagicMock(return_value=(False, "Invalid order"))
        
        # Try to start execution
        result = self.vwap.start_execution(order)
        
        # Should fail because order is invalid
        self.assertFalse(result)

    def test_pause_resume_execution(self):
        """Test pausing and resuming execution"""
        order_id = "test_order_4"
        
        # Setup a mock thread that is alive
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        
        with self.vwap._lock:
            self.vwap._execution_threads[order_id] = mock_thread
        
        # Pause execution
        result = self.vwap.pause_execution(order_id)
        self.assertTrue(result)
        self.assertIn(order_id, self.vwap._execution_paused)
        
        # Resume execution
        result = self.vwap.resume_execution(order_id)
        self.assertTrue(result)
        self.assertNotIn(order_id, self.vwap._execution_paused)
        
        # Try to resume when not paused
        result = self.vwap.resume_execution(order_id)
        self.assertFalse(result)
        
        # Try to pause non-existent execution
        result = self.vwap.pause_execution("non_existent")
        self.assertFalse(result)

    def test_cancel_execution(self):
        """Test canceling execution"""
        order_id = "test_order_5"
        
        # Setup a mock thread that is alive
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        
        with self.vwap._lock:
            self.vwap._execution_threads[order_id] = mock_thread
        
        # Cancel execution
        result = self.vwap.cancel_execution(order_id)
        self.assertTrue(result)
        self.assertIn(order_id, self.vwap._execution_stop_signals)
        
        # Try to cancel non-existent execution
        result = self.vwap.cancel_execution("non_existent")
        self.assertFalse(result)

    def test_create_child_order_market(self):
        """Test creating child market order"""
        # Parent market order
        parent_order = MagicMock(spec=Order)
        parent_order.order_id = "parent_1"
        parent_order.order_type = OrderType.MARKET
        parent_order.symbol = "AAPL"
        parent_order.side = OrderSide.BUY
        parent_order.exchange_id = "NASDAQ"
        parent_order.exchange_account = "test_account"
        
        # Child order mock
        child_order = MagicMock(spec=Order)
        self.order_factory.create_market_order.return_value = child_order
        
        # Create child order
        result = self.vwap._create_child_order(parent_order, 100, 2)
        
        # Check result
        self.assertEqual(result, child_order)
        
        # Verify factory call
        self.order_factory.create_market_order.assert_called_once_with(
            symbol=parent_order.symbol,
            side=parent_order.side,
            quantity=100,
            exchange_id=parent_order.exchange_id,
            exchange_account=parent_order.exchange_account,
            params={
                "parent_order_id": parent_order.order_id,
                "vwap_bucket_index": 2
            }
        )

    def test_create_child_order_limit(self):
        """Test creating child limit order"""
        # Parent limit order
        parent_order = MagicMock(spec=Order)
        parent_order.order_id = "parent_2"
        parent_order.order_type = OrderType.LIMIT
        parent_order.symbol = "AAPL"
        parent_order.side = OrderSide.SELL
        parent_order.price = 150.0
        parent_order.exchange_id = "NASDAQ"
        parent_order.exchange_account = "test_account"
        
        # Child order mock
        child_order = MagicMock(spec=Order)
        self.order_factory.create_limit_order.return_value = child_order
        
        # Create child order
        result = self.vwap._create_child_order(parent_order, 50, 3)
        
        # Check result
        self.assertEqual(result, child_order)
        
        # Verify factory call
        self.order_factory.create_limit_order.assert_called_once_with(
            symbol=parent_order.symbol,
            side=parent_order.side,
            quantity=50,
            price=parent_order.price,
            exchange_id=parent_order.exchange_id,
            exchange_account=parent_order.exchange_account,
            time_in_force=TimeInForce.IOC,
            params={
                "parent_order_id": parent_order.order_id,
                "vwap_bucket_index": 3
            }
        )

    def test_execute_child_order(self):
        """Test executing child order"""
        # This is testing a simulated execution, result should be non-zero
        order = MagicMock(spec=Order)
        
        result = self.vwap._execute_child_order(order)
        
        self.assertGreater(result, 0)

    def test_primary_goal(self):
        """Test primary goal of VWAP"""
        goal = self.vwap._get_primary_goal()
        self.assertEqual(goal, AlgorithmGoal.PRICE)

    @patch('execution.exchange.order.vwap_algorithm.datetime')
    @patch('execution.exchange.order.vwap_algorithm.time')
    def test_execute_vwap(self, mock_time, mock_datetime):
        """Test the VWAP execution process"""
        # Set up mocks
        mock_now = datetime.datetime(2023, 1, 1, 9, 0, 0)
        mock_end = datetime.datetime(2023, 1, 1, 9, 30, 0)  # 30 min later
        
        # Return mock_now first, then mock_end to simulate time passing
        mock_datetime.utcnow.side_effect = [mock_now, mock_now, mock_end]
        
        # Setup test data
        order = MagicMock(spec=Order)
        order.order_id = "test_order_6"
        order.symbol = "AAPL"
        
        profile = VolumeProfile(buckets=2)
        profile.percentages = [0.6, 0.4]
        
        start_time = mock_now
        end_time = mock_end
        slice_sizes = [600, 400]
        
        # Setup progress tracking
        self.vwap.progress[order.order_id] = ExecutionProgress(1000)
        
        # Mock child order creation and execution
        self.vwap._create_child_order = MagicMock()
        child_order = MagicMock(spec=Order)
        self.vwap._create_child_order.return_value = child_order
        
        self.vwap._execute_child_order = MagicMock(return_value=100.0)  # Execution price
        
        # Execute VWAP
        self.vwap._execute_vwap(order, profile, start_time, end_time, slice_sizes)
        
        # Verify execution
        # Should create and execute orders for both buckets
        self.assertEqual(self.vwap._create_child_order.call_count, 2)
        self.assertEqual(self.vwap._execute_child_order.call_count, 2)
        
        # Progress should be complete
        progress = self.vwap.progress[order.order_id]
        self.assertTrue(progress.is_complete)
        
        # Clean up check: the execution thread should be removed
        self.assertNotIn(order.order_id, self.vwap._execution_threads)
        self.assertNotIn(order.order_id, self.vwap._execution_stop_signals)
        self.assertNotIn(order.order_id, self.vwap._execution_paused)

    def test_update_parameters(self):
        """Test updating execution parameters"""
        order_id = "test_order_7"
        
        # Setup a mock thread
        mock_thread = MagicMock()
        
        # Setup initial progress
        progress = ExecutionProgress(1000)
        progress.executed_quantity = 400
        self.vwap.progress[order_id] = progress
        
        # Mock methods to avoid threading issues
        self.vwap.cancel_execution = MagicMock(return_value=True)
        self.vwap.start_execution = MagicMock(return_value=True)
        
        with self.vwap._lock:
            self.vwap._execution_threads[order_id] = mock_thread
        
        # Test parameters update
        params = {
            "original_order": {
                "order_id": order_id,
                "symbol": "AAPL",
                "side": "BUY",
                "order_type": "MARKET",
                "quantity": 1000
            },
            "vwap_duration_minutes": 45,  # Changed parameter
            "vwap_num_buckets": 8         # Changed parameter
        }
        
        result = self.vwap.update_parameters(order_id, params)
        
        # Should succeed
        self.assertTrue(result)
        
        # Should have called cancel and start
        self.vwap.cancel_execution.assert_called_once_with(order_id)
        self.vwap.start_execution.assert_called_once()


if __name__ == '__main__':
    unittest.main()