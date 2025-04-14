"""
Tests for the execution algorithm module.

This module contains unit tests for the ExecutionAlgorithm base class
and related components such as ExecutionProgress and AlgorithmGoal.
"""

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import time
from typing import Dict, Any, Tuple

from execution.order.order import Order, OrderStatus, OrderSide, OrderType
from execution.order.execution_algorithm import (
    ExecutionAlgorithm,
    ExecutionProgress,
    AlgorithmGoal
)

class TestExecutionProgress(unittest.TestCase):
    """Tests for the ExecutionProgress class"""

    def setUp(self):
        """Set up test fixtures, if any"""
        self.progress = ExecutionProgress(total_quantity=1000.0)

    def test_initialization(self):
        """Test proper initialization of execution progress"""
        self.assertEqual(self.progress.total_quantity, 1000.0)
        self.assertEqual(self.progress.executed_quantity, 0.0)
        self.assertEqual(self.progress.total_cost, 0.0)
        self.assertEqual(self.progress.num_trades, 0)
        self.assertEqual(self.progress.slices_completed, 0)
        self.assertEqual(self.progress.slices_total, 0)
        self.assertFalse(self.progress.is_complete)
        self.assertIsInstance(self.progress.extra_info, dict)
        self.assertEqual(len(self.progress.extra_info), 0)

    def test_properties(self):
        """Test computed properties"""
        # Initial state
        self.assertEqual(self.progress.average_price, 0.0)
        self.assertEqual(self.progress.remaining_quantity, 1000.0)
        self.assertEqual(self.progress.completion_percentage, 0.0)

        # After some execution
        self.progress.update(400.0, 10.5)
        self.assertEqual(self.progress.average_price, 10.5)  # 400 * 10.5 / 400
        self.assertEqual(self.progress.remaining_quantity, 600.0)
        self.assertEqual(self.progress.completion_percentage, 40.0)

        # After more execution at different price
        self.progress.update(300.0, 11.0)
        expected_avg_price = (400.0*10.5 + 300.0*11.0) / 700.0
        self.assertAlmostEqual(self.progress.average_price, expected_avg_price)
        self.assertEqual(self.progress.remaining_quantity, 300.0)
        self.assertEqual(self.progress.completion_percentage, 70.0)

    def test_update(self):
        """Test update method"""
        self.progress.update(400.0, 10.0)
        self.assertEqual(self.progress.executed_quantity, 400.0)
        self.assertEqual(self.progress.total_cost, 4000.0)
        self.assertEqual(self.progress.num_trades, 1)
        self.assertFalse(self.progress.is_complete)

        self.progress.update(600.0, 10.5)
        self.assertEqual(self.progress.executed_quantity, 1000.0)
        self.assertEqual(self.progress.total_cost, 4000.0 + 6300.0)
        self.assertEqual(self.progress.num_trades, 2)
        self.assertTrue(self.progress.is_complete)

        # Execute more than total (edge case)
        self.progress.update(50.0, 11.0)
        self.assertEqual(self.progress.executed_quantity, 1050.0)
        self.assertTrue(self.progress.is_complete)

    def test_elapsed_time(self):
        """Test elapsed_time property"""
        start_time = datetime.utcnow()

        with patch.object(self.progress, 'start_time', start_time - timedelta(minutes=5)):
            elapsed = self.progress.elapsed_time
            self.assertGreaterEqual(elapsed.total_seconds(), 300)  # 5 minutes in seconds
            self.assertLess(elapsed.total_seconds(), 301)  # allowing small processing time

    def test_to_dict(self):
        """Test serialization to dictionary"""
        self.progress.update(500.0, 10.0)
        self.progress.extra_info = {"test_key": "test_value"}

        result = self.progress.to_dict()

        self.assertIsInstance(result, dict)
        self.assertEqual(result["total_quantity"], 1000.0)
        self.assertEqual(result["executed_quantity"], 500.0)
        self.assertEqual(result["remaining_quantity"], 500.0)
        self.assertEqual(result["average_price"], 10.0)
        self.assertEqual(result["total_cost"], 5000.0)
        self.assertEqual(result["num_trades"], 1)
        self.assertEqual(result["completion_percentage"], 50.0)
        self.assertFalse(result["is_complete"])
        self.assertEqual(result["extra_info"], {"test_key": "test_value"})
        self.assertIn("start_time", result)
        self.assertIn("elapsed_seconds", result)


class MockExecutionAlgorithm(ExecutionAlgorithm):
    """Mock implementation of ExecutionAlgorithm for testing purposes"""

    def start_execution(self, order: Order) -> bool:
        """Mock implementation of start_execution"""
        self.progress[order.order_id] = ExecutionProgress(order.quantity)
        return True

    def pause_execution(self, order_id: str) -> bool:
        """Mock implementation of pause_execution"""
        return order_id in self.progress

    def resume_execution(self, order_id: str) -> bool:
        """Mock implementation of resume_execution"""
        return order_id in self.progress

    def cancel_execution(self, order_id: str) -> bool:
        """Mock implementation of cancel_execution"""
        if order_id in self.progress:
            self.progress.pop(order_id)
            return True
        return False

    def update_parameters(self, order_id: str, params: Dict[str, Any]) -> bool:
        """Mock implementation of update_parameters"""
        return order_id in self.progress

    def can_execute_order(self, order: Order) -> Tuple[bool, str]:
        """Mock implementation of can_execute_order"""
        venue_supported = self.is_supported_venue(order.venue_id)
        asset_supported = self.is_supported_asset(order.asset_id)

        if not venue_supported:
            return False, "Venue not supported"
        if not asset_supported:
            return False, "Asset not supported"

        return True, "Order can be executed"


class TestExecutionAlgorithm(unittest.TestCase):
    """Tests for the ExecutionAlgorithm base class"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            "primary_goal": "impact",
            "supported_venues": ["venue1", "venue2"],
            "supported_assets": ["BTC", "ETH"],
            "excluded_assets": ["XRP"]
        }
        self.algorithm = MockExecutionAlgorithm("TestAlgo", self.config)

        # Create mock orders
        self.order1 = MagicMock(spec=Order)
        self.order1.order_id = "order1"
        self.order1.quantity = 100.0
        self.order1.venue_id = "venue1"
        self.order1.asset_id = "BTC"

        self.order2 = MagicMock(spec=Order)
        self.order2.order_id = "order2"
        self.order2.quantity = 200.0
        self.order2.venue_id = "venue3"  # Unsupported venue
        self.order2.asset_id = "ETH"

        self.order3 = MagicMock(spec=Order)
        self.order3.order_id = "order3"
        self.order3.quantity = 300.0
        self.order3.venue_id = "venue1"
        self.order3.asset_id = "XRP"  # Excluded asset

    def test_initialization(self):
        """Test proper initialization of algorithm"""
        self.assertEqual(self.algorithm.name, "TestAlgo")
        self.assertEqual(self.algorithm.config, self.config)
        self.assertEqual(self.algorithm.goal, AlgorithmGoal.IMPACT)
        self.assertIsInstance(self.algorithm.progress, dict)

    def test_get_primary_goal(self):
        """Test goal extraction from config"""
        # Test valid goal
        self.assertEqual(self.algorithm._get_primary_goal(), AlgorithmGoal.IMPACT)

        # Test default goal when not specified
        algo = MockExecutionAlgorithm("DefaultGoal", {})
        self.assertEqual(algo._get_primary_goal(), AlgorithmGoal.PRICE)

        # Test invalid goal
        algo = MockExecutionAlgorithm("InvalidGoal", {"primary_goal": "invalid_goal"})
        self.assertEqual(algo._get_primary_goal(), AlgorithmGoal.PRICE)

    def test_venue_asset_support(self):
        """Test venue and asset support checking"""
        # Supported venue
        self.assertTrue(self.algorithm.is_supported_venue("venue1"))
        self.assertTrue(self.algorithm.is_supported_venue("venue2"))

        # Unsupported venue
        self.assertFalse(self.algorithm.is_supported_venue("venue3"))

        # Supported asset
        self.assertTrue(self.algorithm.is_supported_asset("BTC"))
        self.assertTrue(self.algorithm.is_supported_asset("ETH"))

        # Excluded asset
        self.assertFalse(self.algorithm.is_supported_asset("XRP"))

        # Unknown asset (not in supported or excluded)
        self.assertFalse(self.algorithm.is_supported_asset("LTC"))

        # Test with empty supported list (all supported except excluded)
        algo = MockExecutionAlgorithm("AllAssets", {
            "supported_assets": [],
            "excluded_assets": ["XRP"]
        })
        self.assertTrue(algo.is_supported_asset("BTC"))
        self.assertTrue(algo.is_supported_asset("ETH"))
        self.assertFalse(algo.is_supported_asset("XRP"))

    def test_can_execute_order(self):
        """Test order validation"""
        # Valid order
        can_execute, reason = self.algorithm.can_execute_order(self.order1)
        self.assertTrue(can_execute)
        self.assertEqual(reason, "Order can be executed")

        # Invalid venue
        can_execute, reason = self.algorithm.can_execute_order(self.order2)
        self.assertFalse(can_execute)
        self.assertEqual(reason, "Venue not supported")

        # Excluded asset
        can_execute, reason = self.algorithm.can_execute_order(self.order3)
        self.assertFalse(can_execute)
        self.assertEqual(reason, "Asset not supported")

    def test_start_execution(self):
        """Test execution start"""
        result = self.algorithm.start_execution(self.order1)
        self.assertTrue(result)
        self.assertIn(self.order1.order_id, self.algorithm.progress)
        self.assertEqual(
            self.algorithm.progress[self.order1.order_id].total_quantity,
            self.order1.quantity
        )

    def test_get_progress(self):
        """Test progress retrieval"""
        # Non-existent order
        self.assertIsNone(self.algorithm.get_progress("non_existent"))

        # Existing order
        self.algorithm.start_execution(self.order1)
        progress = self.algorithm.get_progress(self.order1.order_id)
        self.assertIsInstance(progress, ExecutionProgress)
        self.assertEqual(progress.total_quantity, self.order1.quantity)

    def test_cancel_execution(self):
        """Test execution cancellation"""
        # Setup
        self.algorithm.start_execution(self.order1)
        self.assertIn(self.order1.order_id, self.algorithm.progress)

        # Cancel existing order
        result = self.algorithm.cancel_execution(self.order1.order_id)
        self.assertTrue(result)
        self.assertNotIn(self.order1.order_id, self.algorithm.progress)

        # Cancel non-existent order
        result = self.algorithm.cancel_execution("non_existent")
        self.assertFalse(result)

    def test_estimated_completion_time(self):
        """Test completion time estimation"""
        # No order
        self.assertIsNone(self.algorithm.get_estimated_completion_time("non_existent"))

        # Order with no execution yet
        self.algorithm.start_execution(self.order1)
        self.assertIsNone(self.algorithm.get_estimated_completion_time(self.order1.order_id))

        # Order with some execution
        progress = self.algorithm.get_progress(self.order1.order_id)

        # Mock elapsed time to 10 seconds
        with patch.object(progress, 'elapsed_time', timedelta(seconds=10)):
            # Execute 20% in 10 seconds
            progress.update(20.0, 100.0)

            # Should estimate completion in 40 more seconds (50 total)
            estimated = self.algorithm.get_estimated_completion_time(self.order1.order_id)
            self.assertIsNotNone(estimated)

            # Check that estimated time is roughly 40 seconds from now
            expected_time = datetime.utcnow() + timedelta(seconds=40)
            self.assertLess(abs((estimated - expected_time).total_seconds()), 1.0)

        # Order that's complete
        progress.update(80.0, 100.0)  # Complete the order
        self.assertIsNone(self.algorithm.get_estimated_completion_time(self.order1.order_id))

    def test_get_all_active_executions(self):
        """Test retrieving active executions"""
        # No active executions
        self.assertEqual(len(self.algorithm.get_all_active_executions()), 0)

        # One active execution
        self.algorithm.start_execution(self.order1)
        active = self.algorithm.get_all_active_executions()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0], self.order1.order_id)

        # Multiple active executions
        order4 = MagicMock(spec=Order)
        order4.order_id = "order4"
        order4.quantity = 400.0
        order4.venue_id = "venue1"
        order4.asset_id = "ETH"
        self.algorithm.start_execution(order4)

        active = self.algorithm.get_all_active_executions()
        self.assertEqual(len(active), 2)
        self.assertIn(self.order1.order_id, active)
        self.assertIn(order4.order_id, active)

        # Complete one execution
        progress = self.algorithm.get_progress(self.order1.order_id)
        progress.update(100.0, 10.0)  # Complete the order

        active = self.algorithm.get_all_active_executions()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0], order4.order_id)

    def test_get_algorithm_info(self):
        """Test algorithm info retrieval"""
        # Setup active executions
        self.algorithm.start_execution(self.order1)

        info = self.algorithm.get_algorithm_info()
        self.assertEqual(info["name"], "TestAlgo")
        self.assertEqual(info["goal"], "impact")
        self.assertEqual(info["active_executions"], 1)
        self.assertEqual(info["config"], self.config)


if __name__ == "__main__":
    unittest.main()