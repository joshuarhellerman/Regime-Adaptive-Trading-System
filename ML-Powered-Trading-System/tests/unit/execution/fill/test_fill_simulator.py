"""
Tests for the fill_simulator.py module.

This tests the realistic fill simulation for paper trading, including:
- Latency models
- Partial fill models
- Price impact models
- FillSimulator functionality
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, ANY
import asyncio
import threading
import json
import random
import time
from datetime import datetime
from decimal import Decimal

# Import the module components to test
from execution.fill_simulator import (
    SimulationModel,
    LatencyModel,
    PartialFillModel,
    PriceImpactModel,
    FillSimulator
)
from core.event_bus import EventBus, Event, EventTopics, EventPriority
from execution.order.order import Order, OrderStatus, OrderSide, OrderType
from execution.fill.fill_model import Fill


class TestSimulationModel(unittest.TestCase):
    """Tests for the base SimulationModel class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        model = SimulationModel()
        self.assertEqual(model.config, {})

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = {"test_param": 123}
        model = SimulationModel(config)
        self.assertEqual(model.config, config)


class TestLatencyModel(unittest.TestCase):
    """Tests for the LatencyModel class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        model = LatencyModel()
        self.assertEqual(model.base_latency, 50)
        self.assertEqual(model.latency_std_dev, 25)
        self.assertEqual(model.market_load_factor, 1.0)
        self.assertEqual(model.size_factor, 0.1)
        self.assertEqual(model.max_latency, 5000)

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = {
            "base_latency_ms": 100,
            "latency_std_dev_ms": 50,
            "market_load_factor": 2.0,
            "size_factor": 0.2,
            "max_latency_ms": 10000
        }
        model = LatencyModel(config)
        self.assertEqual(model.base_latency, 100)
        self.assertEqual(model.latency_std_dev, 50)
        self.assertEqual(model.market_load_factor, 2.0)
        self.assertEqual(model.size_factor, 0.2)
        self.assertEqual(model.max_latency, 10000)

    def test_get_latency_no_market_volume(self):
        """Test latency calculation without market volume."""
        random.seed(42)  # For reproducible tests
        model = LatencyModel()
        latency = model.get_latency(100)
        # With random values, we can only check range and type
        self.assertIsInstance(latency, float)
        self.assertGreaterEqual(latency, 10)  # Minimum latency is 10ms
        self.assertLessEqual(latency, model.max_latency)

    def test_get_latency_with_market_volume(self):
        """Test latency calculation with market volume."""
        random.seed(42)  # For reproducible tests
        model = LatencyModel()
        # Order is 10% of market volume
        latency = model.get_latency(100, 1000)
        # Size penalty should be 1.0 + (100/1000 * 0.1) = 1.01
        self.assertIsInstance(latency, float)
        self.assertGreaterEqual(latency, 10)  # Minimum latency is 10ms
        self.assertLessEqual(latency, model.max_latency)

    def test_max_latency_cap(self):
        """Test that latency is capped at max_latency."""
        model = LatencyModel({"base_latency_ms": 10000, "latency_std_dev_ms": 0})
        latency = model.get_latency(100)
        self.assertEqual(latency, model.max_latency)


class TestPartialFillModel(unittest.TestCase):
    """Tests for the PartialFillModel class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        model = PartialFillModel()
        self.assertEqual(model.partial_fill_probability, 0.3)
        self.assertEqual(model.min_fills, 1)
        self.assertEqual(model.max_fills, 5)
        self.assertEqual(model.liquidity_threshold, 0.1)

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = {
            "partial_fill_probability": 0.5,
            "min_fills": 2,
            "max_fills": 10,
            "liquidity_threshold": 0.2,
        }
        model = PartialFillModel(config)
        self.assertEqual(model.partial_fill_probability, 0.5)
        self.assertEqual(model.min_fills, 2)
        self.assertEqual(model.max_fills, 10)
        self.assertEqual(model.liquidity_threshold, 0.2)

    def test_generate_fills_single_fill(self):
        """Test generation of a single fill."""
        random.seed(1)  # Force no partial fills
        model = PartialFillModel({"partial_fill_probability": 0})
        fills = model.generate_fills(100)
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0], 100)

    def test_generate_fills_partial(self):
        """Test generation of partial fills."""
        random.seed(42)  # For reproducible tests
        model = PartialFillModel({"partial_fill_probability": 1.0})
        fills = model.generate_fills(100)
        # Check that we have multiple fills
        self.assertGreater(len(fills), 1)
        # Check that they sum to the order size
        self.assertAlmostEqual(sum(fills), 100, delta=0.001)

    def test_generate_fills_with_market_volume(self):
        """Test partial fills with market volume."""
        random.seed(42)  # For reproducible tests
        model = PartialFillModel({"partial_fill_probability": 1.0})
        # Large order relative to market
        fills = model.generate_fills(100, 200)
        # Check that we have multiple fills
        self.assertGreater(len(fills), 1)
        # Check that they sum to the order size
        self.assertAlmostEqual(sum(fills), 100, delta=0.001)


class TestPriceImpactModel(unittest.TestCase):
    """Tests for the PriceImpactModel class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        model = PriceImpactModel()
        self.assertEqual(model.market_impact_factor, 0.1)
        self.assertEqual(model.volatility_factor, 0.2)
        self.assertEqual(model.bid_ask_spread_bps, 5)
        self.assertTrue(model.use_spread)
        self.assertTrue(model.randomize_within_spread)

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = {
            "market_impact_factor": 0.2,
            "volatility_factor": 0.3,
            "bid_ask_spread_bps": 10,
            "use_spread": False,
            "randomize_within_spread": False
        }
        model = PriceImpactModel(config)
        self.assertEqual(model.market_impact_factor, 0.2)
        self.assertEqual(model.volatility_factor, 0.3)
        self.assertEqual(model.bid_ask_spread_bps, 10)
        self.assertFalse(model.use_spread)
        self.assertFalse(model.randomize_within_spread)

    def test_calculate_price_buy(self):
        """Test price calculation for buy orders."""
        random.seed(42)  # For reproducible tests
        model = PriceImpactModel()
        market_data = {
            "mid_price": 100,
            "volume": 1000,
            "volatility": 0.01
        }
        price = model.calculate_price(OrderSide.BUY, 100, market_data)
        # Price should be higher than mid price due to spread and impact
        self.assertGreater(price, market_data["mid_price"])

    def test_calculate_price_sell(self):
        """Test price calculation for sell orders."""
        random.seed(42)  # For reproducible tests
        model = PriceImpactModel()
        market_data = {
            "mid_price": 100,
            "volume": 1000,
            "volatility": 0.01
        }
        price = model.calculate_price(OrderSide.SELL, 100, market_data)
        # Price should be lower than mid price due to spread and impact
        self.assertLess(price, market_data["mid_price"])

    def test_calculate_price_no_spread(self):
        """Test price calculation without spread."""
        random.seed(42)  # For reproducible tests
        model = PriceImpactModel({"use_spread": False})
        market_data = {
            "mid_price": 100,
            "volume": 1000,
            "volatility": 0.01
        }
        price_buy = model.calculate_price(OrderSide.BUY, 100, market_data)
        price_sell = model.calculate_price(OrderSide.SELL, 100, market_data)
        # Buy should be higher than sell due to impact
        self.assertGreater(price_buy, price_sell)


@patch('execution.fill_simulator.logging')
class TestFillSimulator(unittest.TestCase):
    """Tests for the main FillSimulator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.market_data_service = Mock()
        self.event_bus = Mock(spec=EventBus)
        self.fill_simulator = FillSimulator(
            self.market_data_service,
            self.event_bus
        )

        # Mock market data
        self.market_data_service.get_ticker.return_value = {
            "price": 100,
            "bid": 99.8,
            "ask": 100.2,
            "volume": 10000,
            "high": 105,
            "low": 95
        }
        self.market_data_service.get_market_depth.return_value = {
            "bids": [[99.8, 100], [99.7, 200]],
            "asks": [[100.2, 100], [100.3, 200]]
        }

    def test_init(self, mock_logging):
        """Test simulator initialization."""
        # Check that models are initialized
        self.assertIsInstance(self.fill_simulator.fill_latency_model, LatencyModel)
        self.assertIsInstance(self.fill_simulator.partial_fill_model, PartialFillModel)
        self.assertIsInstance(self.fill_simulator.price_impact_model, PriceImpactModel)

        # Check that it subscribed to order events
        self.event_bus.subscribe.assert_called_once_with(
            "paper.order.new",
            self.fill_simulator.simulate_fill
        )

    def test_start_stop(self, mock_logging):
        """Test starting and stopping the simulator."""
        # Start
        self.fill_simulator.start()
        self.assertTrue(self.fill_simulator._running)
        self.assertIsNotNone(self.fill_simulator._thread)

        # Stop
        self.fill_simulator.stop()
        self.assertFalse(self.fill_simulator._running)

    def test_get_market_data(self, mock_logging):
        """Test getting market data."""
        data = self.fill_simulator._get_market_data("BTC/USD")

        # Check that it called the market data service
        self.market_data_service.get_ticker.assert_called_once_with("BTC/USD")
        self.market_data_service.get_market_depth.assert_called_once_with("BTC/USD")

        # Check data
        self.assertEqual(data["price"], 100)
        self.assertEqual(data["bid"], 99.8)
        self.assertEqual(data["ask"], 100.2)
        self.assertEqual(data["volume"], 10000)
        self.assertEqual(data["mid_price"], 100.0)
        self.assertAlmostEqual(data["volatility"], 0.095238, places=5)

    def test_get_market_data_error(self, mock_logging):
        """Test handling errors in market data."""
        self.market_data_service.get_ticker.side_effect = Exception("Test error")

        data = self.fill_simulator._get_market_data("BTC/USD")

        # Should return default data
        self.assertEqual(data["price"], 0)
        self.assertEqual(data["volume"], 0)
        mock_logging.error.assert_called_once()

    @patch('execution.fill_simulator.uuid')
    def test_simulate_fill(self, mock_uuid, mock_logging):
        """Test simulating a fill."""
        # Setup mocks
        mock_uuid.uuid4.return_value = "test-uuid"
        mock_order = Mock(spec=Order)
        mock_order.order_id = "test-order"
        mock_order.symbol = "BTC/USD"
        mock_order.quantity = 100
        mock_order.side = OrderSide.BUY

        mock_event = Mock(spec=Event)
        mock_event.data = mock_order

        # Mock asyncio methods
        self.fill_simulator._loop = Mock()
        self.fill_simulator._loop.call_soon_threadsafe = Mock()

        # Call the method
        self.fill_simulator.simulate_fill(mock_event)

        # Check that it scheduled the async task
        self.fill_simulator._loop.call_soon_threadsafe.assert_called_once()

    @patch('execution.fill_simulator.asyncio.sleep')
    @patch('execution.fill_simulator.create_event')
    @patch('execution.fill_simulator.datetime')
    def test_simulate_fill_async(self, mock_datetime, mock_create_event, mock_sleep, mock_logging):
        """Test the async fill simulation."""
        # Setup mocks
        mock_now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = mock_now
        mock_datetime.utcnow.return_value = mock_now

        mock_order = Mock(spec=Order)
        mock_order.order_id = "test-order"
        mock_order.symbol = "BTC/USD"
        mock_order.quantity = 100
        mock_order.side = OrderSide.BUY

        # Mock internal methods
        self.fill_simulator._is_simulation_active = Mock(return_value=True)
        self.fill_simulator._get_market_data = Mock(return_value={
            "price": 100,
            "volume": 10000,
            "volatility": 0.01
        })
        self.fill_simulator._publish_fill = Mock()

        # Mock simulation models
        self.fill_simulator.fill_latency_model.get_latency = Mock(return_value=50)
        self.fill_simulator.partial_fill_model.generate_fills = Mock(return_value=[100])
        self.fill_simulator.price_impact_model.calculate_price = Mock(return_value=100.5)

        # Create an event loop for testing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the async method
            task = loop.create_task(self.fill_simulator._simulate_fill_async(mock_order))
            loop.run_until_complete(task)

            # Check that it waited for latency
            mock_sleep.assert_called_with(0.05)  # 50ms

            # Check that it created a fill
            self.fill_simulator._publish_fill.assert_called_once()
            fill = self.fill_simulator._publish_fill.call_args[0][0]
            self.assertEqual(fill.order_id, "test-order")
            self.assertEqual(fill.quantity, Decimal("100"))
            self.assertEqual(fill.price, Decimal("100.5"))
        finally:
            loop.close()

    @patch('execution.fill_simulator.asyncio.sleep')
    def test_execute_fill_partial(self, mock_sleep, mock_logging):
        """Test executing partial fills."""
        # Setup mocks
        mock_now = datetime(2023, 1, 1, 12, 0, 0)
        with patch('execution.fill_simulator.datetime') as mock_datetime:
            mock_datetime.now.return_value = mock_now
            mock_datetime.utcnow.return_value = mock_now

            mock_order = Mock(spec=Order)
            mock_order.order_id = "test-order"
            mock_order.symbol = "BTC/USD"
            mock_order.quantity = 100
            mock_order.side = OrderSide.BUY

            # Mock multiple partial fills
            self.fill_simulator._is_simulation_active = Mock(return_value=True)
            self.fill_simulator._get_market_data = Mock(return_value={
                "price": 100,
                "volume": 10000,
                "volatility": 0.01
            })
            self.fill_simulator._publish_fill = Mock()

            self.fill_simulator.partial_fill_model.generate_fills = Mock(return_value=[40, 60])
            self.fill_simulator.price_impact_model.calculate_price = Mock(side_effect=[100.2, 100.5])

            # Create an event loop for testing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Run the async method
                task = loop.create_task(
                    self.fill_simulator._execute_fill(mock_order, "test-sim", {})
                )
                loop.run_until_complete(task)

                # Check that it created two fills
                self.assertEqual(self.fill_simulator._publish_fill.call_count, 2)

                # First fill
                fill1 = self.fill_simulator._publish_fill.call_args_list[0][0][0]
                self.assertEqual(fill1.order_id, "test-order")
                self.assertEqual(fill1.quantity, Decimal("40"))
                self.assertEqual(fill1.price, Decimal("100.2"))

                # Second fill
                fill2 = self.fill_simulator._publish_fill.call_args_list[1][0][0]
                self.assertEqual(fill2.order_id, "test-order")
                self.assertEqual(fill2.quantity, Decimal("60"))
                self.assertEqual(fill2.price, Decimal("100.5"))

                # Check that it slept between fills
                mock_sleep.assert_called_once()
            finally:
                loop.close()

    def test_publish_fill(self, mock_logging):
        """Test publishing fills."""
        # Setup mocks
        with patch('execution.fill_simulator.create_event') as mock_create_event:
            mock_create_event.return_value = "test_event"

            # Create a test fill
            fill = Fill(
                order_id="test-order",
                fill_id="test-fill",
                timestamp=datetime(2023, 1, 1, 12, 0, 0),
                instrument="BTC/USD",
                quantity=Decimal("100"),
                price=Decimal("100.5"),
                fees=Decimal("0.1"),
                exchange_id="paper_exchange",
                is_maker=True,
                metadata={}
            )

            # Call the method
            self.fill_simulator._publish_fill(fill)

            # Check that it created and published an event
            mock_create_event.assert_called_once_with(
                "paper.order.filled",
                fill,
                priority=EventPriority.HIGH,
                source="fill_simulator"
            )
            self.event_bus.publish.assert_called_once_with("test_event")

    def test_cancel_simulation(self, mock_logging):
        """Test canceling simulations."""
        # Just check that it logs the cancellation
        self.fill_simulator.cancel_simulation("test-order")
        mock_logging.info.assert_called_once()


if __name__ == "__main__":
    unittest.main()