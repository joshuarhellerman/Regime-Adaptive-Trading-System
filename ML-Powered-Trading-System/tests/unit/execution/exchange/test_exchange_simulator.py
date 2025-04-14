"""
Tests for ExchangeSimulator class.

This module contains tests for the ExchangeSimulator class, which provides
a high-fidelity simulation of exchange behavior for paper trading.
"""

import pytest
import unittest.mock as mock
from datetime import datetime
import time
import threading
from typing import Dict, Any
import uuid

# Import simulator module
from execution.exchange.exchange_simulator import ExchangeSimulator, SimulationConfig, SimulationMode
from execution.order.order import Order, OrderStatus, OrderType, Side, TimeInForce
from execution.fill.fill_model import Fill
from core.event_bus import EventBus, EventType
from core.state_manager import StateManager
from data.market_data_service import MarketDataService


class TestExchangeSimulator:
    """Test suite for ExchangeSimulator."""

    @pytest.fixture
    def mock_market_data_service(self):
        """Create a mock market data service."""
        market_data_service = mock.MagicMock(spec=MarketDataService)
        # Setup market data for different instruments
        market_data = {
            "BTC/USD": {"bid": 50000.0, "ask": 50100.0, "price": 50050.0},
            "ETH/USD": {"bid": 3000.0, "ask": 3010.0, "price": 3005.0},
            "EUR/USD": {"bid": 1.0950, "ask": 1.0952, "price": 1.0951},
        }

        market_data_service.get_last_price = lambda instrument: market_data.get(
            instrument, {"bid": 0, "ask": 0, "price": 0}
        )
        return market_data_service

    @pytest.fixture
    def mock_state_manager(self):
        """Create a mock state manager."""
        return mock.MagicMock(spec=StateManager)

    @pytest.fixture
    def mock_event_bus(self):
        """Create a mock event bus."""
        event_bus = mock.MagicMock(spec=EventBus)
        # Keep track of emitted events for assertions
        event_bus.emitted_events = []
        # Override the emit method to track events
        event_bus.emit = lambda event_type, data: event_bus.emitted_events.append((event_type, data))
        return event_bus

    @pytest.fixture
    def simulation_config(self):
        """Create a simulation configuration with faster processing for tests."""
        return SimulationConfig(
            mode=SimulationMode.REALISTIC,
            base_latency_ms=1,  # Low latency for faster tests
            latency_stdev_ms=0,  # No variance for deterministic tests
            exchange_processing_ms=1,
            partial_fill_probability=0.5,  # Higher probability to test partial fills
            rejection_probability=0.05,  # Some rejections for testing
            time_acceleration=10.0,  # Faster simulation for tests
        )

    @pytest.fixture
    def exchange_simulator(self, mock_market_data_service, mock_state_manager, mock_event_bus, simulation_config):
        """Create an instance of ExchangeSimulator for testing."""
        simulator = ExchangeSimulator(
            exchange_id="test_exchange",
            market_data_service=mock_market_data_service,
            state_manager=mock_state_manager,
            event_bus=mock_event_bus,
            config=simulation_config
        )

        # Initialize some balances for testing
        simulator.set_balance("USD", 100000.0)
        simulator.set_balance("BTC", 5.0)
        simulator.set_balance("ETH", 50.0)

        yield simulator

        # Clean up
        simulator.disconnect()

    @pytest.fixture
    def sample_order(self):
        """Create a sample order for testing."""
        return Order(
            id=str(uuid.uuid4()),
            instrument="BTC/USD",
            quantity=1.0,
            side=Side.BUY,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
            price=None,  # Market order doesn't need price
            client_order_id="test_client_order_1"
        )

    @pytest.fixture
    def sample_limit_order(self):
        """Create a sample limit order for testing."""
        return Order(
            id=str(uuid.uuid4()),
            instrument="BTC/USD",
            quantity=0.5,
            side=Side.BUY,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.GTC,
            price=49500.0,  # Limit price below current bid
            client_order_id="test_client_order_2"
        )

    def test_connect_disconnect(self, exchange_simulator):
        """Test connect and disconnect methods."""
        # Connect should always return True in simulation
        assert exchange_simulator.connect() is True

        # Test disconnect
        assert exchange_simulator.disconnect() is True
        assert exchange_simulator.running is False

    def test_place_order(self, exchange_simulator, sample_order, mock_event_bus):
        """Test placing a market order."""
        # Place the order
        order_id = exchange_simulator.place_order(sample_order)

        # Verify order ID was generated
        assert order_id is not None
        assert order_id.startswith("sim_test_exchange_")

        # Verify order was added to open orders
        assert order_id in exchange_simulator.open_orders

        # Verify order status is PENDING
        assert exchange_simulator.open_orders[order_id].status == OrderStatus.PENDING

        # Verify ORDER_ACCEPTED event was emitted
        acceptance_events = [(et, data) for et, data in mock_event_bus.emitted_events
                             if et == EventType.ORDER_ACCEPTED]
        assert len(acceptance_events) == 1
        assert acceptance_events[0][1]["order"].exchange_order_id == order_id

    def test_order_lifecycle(self, exchange_simulator, sample_order, mock_event_bus):
        """Test the full lifecycle of an order from placement to fill."""
        # Place the order
        order_id = exchange_simulator.place_order(sample_order)

        # Allow time for order processing
        time.sleep(0.1)

        # Verify order status progression
        # Get the most recent order state
        processed_order = exchange_simulator.get_order(order_id)

        # Order should eventually be filled or at least moved from PENDING
        assert processed_order.status != OrderStatus.PENDING

        # Check for expected events
        events = mock_event_bus.emitted_events

        # There should be at least: ORDER_ACCEPTED, ORDER_NEW
        event_types = [et for et, _ in events]
        assert EventType.ORDER_ACCEPTED in event_types
        assert EventType.ORDER_NEW in event_types

        # And either ORDER_FILLED or ORDER_PARTIALLY_FILLED
        assert (EventType.ORDER_FILLED in event_types or
                EventType.ORDER_PARTIALLY_FILLED in event_types)

    def test_cancel_order(self, exchange_simulator, sample_limit_order, mock_event_bus):
        """Test canceling an order."""
        # Place a limit order (less likely to fill immediately)
        order_id = exchange_simulator.place_order(sample_limit_order)

        # Small delay to ensure order is processed
        time.sleep(0.05)

        # Cancel the order
        result = exchange_simulator.cancel_order(order_id)
        assert result is True

        # Allow time for cancellation processing
        time.sleep(0.1)

        # Verify order status is CANCELED
        canceled_order = exchange_simulator.get_order(order_id)
        assert canceled_order.status == OrderStatus.CANCELED

        # Check for cancel event
        cancel_events = [(et, data) for et, data in mock_event_bus.emitted_events
                         if et == EventType.ORDER_CANCELED]
        assert len(cancel_events) > 0
        assert cancel_events[0][1]["order"].exchange_order_id == order_id

    def test_limit_order_execution(self, exchange_simulator, mock_event_bus):
        """Test limit order execution when price matches."""
        # Create a limit order that should execute immediately (price > ask)
        limit_order = Order(
            id=str(uuid.uuid4()),
            instrument="BTC/USD",
            quantity=0.2,
            side=Side.BUY,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.GTC,
            price=51000.0,  # Above current ask price
            client_order_id="test_limit_execute"
        )

        # Place the order
        order_id = exchange_simulator.place_order(limit_order)

        # Allow time for order processing
        time.sleep(0.1)

        # Verify order was filled
        filled_order = exchange_simulator.get_order(order_id)
        assert filled_order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]

        # Verify fill events
        fill_events = [(et, data) for et, data in mock_event_bus.emitted_events
                       if et in [EventType.ORDER_FILLED, EventType.ORDER_PARTIALLY_FILLED]]
        assert len(fill_events) > 0

        # Price should never exceed the limit price
        for _, data in fill_events:
            assert data["fill"].price <= limit_order.price

    def test_position_updates(self, exchange_simulator):
        """Test that positions are updated after fills."""
        initial_btc = exchange_simulator.get_balance("BTC")
        initial_usd = exchange_simulator.get_balance("USD")

        # Create a market sell order
        sell_order = Order(
            id=str(uuid.uuid4()),
            instrument="BTC/USD",
            quantity=1.0,
            side=Side.SELL,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
            client_order_id="test_position_update"
        )

        # Place the order
        order_id = exchange_simulator.place_order(sell_order)

        # Allow time for order processing
        time.sleep(0.1)

        # Verify position and balance updates
        current_btc = exchange_simulator.get_balance("BTC")
        current_usd = exchange_simulator.get_balance("USD")

        # BTC should decrease
        assert current_btc < initial_btc

        # USD should increase (minus fees)
        assert current_usd > initial_usd

    def test_get_open_orders(self, exchange_simulator):
        """Test retrieving open orders."""
        # Place multiple orders
        orders = []
        for i in range(3):
            order = Order(
                id=str(uuid.uuid4()),
                instrument="ETH/USD",
                quantity=0.1 * (i + 1),
                side=Side.BUY,
                type=OrderType.LIMIT,
                time_in_force=TimeInForce.GTC,
                price=2900.0 - (i * 100),  # Different prices
                client_order_id=f"test_open_orders_{i}"
            )
            orders.append(order)
            exchange_simulator.place_order(order)

        # Allow time for order processing
        time.sleep(0.05)

        # Get open orders
        open_orders = exchange_simulator.get_open_orders()

        # All limit orders should be open (not filled since price is below bid)
        assert len(open_orders) == 3

        # Cancel one order
        exchange_simulator.cancel_order(open_orders[0].exchange_order_id)
        time.sleep(0.05)

        # Should now have 2 open orders
        open_orders = exchange_simulator.get_open_orders()
        assert len(open_orders) == 2

    def test_market_data_updates(self, exchange_simulator, mock_event_bus):
        """Test handling of market data updates for limit orders."""
        # Place a limit buy order slightly below current bid
        limit_order = Order(
            id=str(uuid.uuid4()),
            instrument="ETH/USD",
            quantity=0.5,
            side=Side.BUY,
            type=OrderType.LIMIT,
            time_in_force=TimeInForce.GTC,
            price=2950.0,  # Below current bid
            client_order_id="test_market_update"
        )
        order_id = exchange_simulator.place_order(limit_order)
        time.sleep(0.05)

        # Simulate market data update with price dropping below limit price
        exchange_simulator._handle_market_data_update({
            "instrument": "ETH/USD",
            "data": {"bid": 2900.0, "ask": 2930.0, "price": 2915.0}
        })

        # Allow time for processing
        time.sleep(0.1)

        # Order should be filled or partially filled
        updated_order = exchange_simulator.get_order(order_id)
        assert updated_order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]

    def test_position_tracking(self, exchange_simulator):
        """Test position tracking for multiple trades."""
        # Current ETH position should be 0
        assert exchange_simulator.get_position("ETH/USD") == 0.0

        # Place buy order
        buy_order = Order(
            id=str(uuid.uuid4()),
            instrument="ETH/USD",
            quantity=1.0,
            side=Side.BUY,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
            client_order_id="test_position_buy"
        )
        exchange_simulator.place_order(buy_order)
        time.sleep(0.1)

        # Position should be positive
        assert exchange_simulator.get_position("ETH/USD") > 0

        # Place sell order for half
        sell_order = Order(
            id=str(uuid.uuid4()),
            instrument="ETH/USD",
            quantity=0.5,
            side=Side.SELL,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
            client_order_id="test_position_sell"
        )
        exchange_simulator.place_order(sell_order)
        time.sleep(0.1)

        # Position should decrease but remain positive
        assert 0 < exchange_simulator.get_position("ETH/USD") < 1.0

    def test_reset_simulator(self, exchange_simulator, sample_limit_order):
        """Test resetting the simulator state."""
        # Setup some initial state
        exchange_simulator.place_order(sample_limit_order)
        time.sleep(0.05)

        # Before reset
        assert len(exchange_simulator.open_orders) > 0
        exchange_simulator.positions["BTC/USD"] = 2.5

        # Reset simulator
        exchange_simulator.reset()

        # After reset
        assert len(exchange_simulator.open_orders) == 0
        assert len(exchange_simulator.positions) == 0
        assert len(exchange_simulator.balances) == 0

    def test_multiple_order_processing(self, exchange_simulator):
        """Test processing multiple orders concurrently."""
        # Place multiple orders rapidly
        order_ids = []
        for i in range(5):
            order = Order(
                id=str(uuid.uuid4()),
                instrument="BTC/USD",
                quantity=0.1,
                side=Side.BUY if i % 2 == 0 else Side.SELL,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.GTC,
                client_order_id=f"test_multi_{i}"
            )
            order_id = exchange_simulator.place_order(order)
            order_ids.append(order_id)

        # Allow time for all orders to process
        time.sleep(0.2)

        # Verify all orders have been processed
        statuses = [exchange_simulator.get_order(oid).status for oid in order_ids]
        for status in statuses:
            assert status != OrderStatus.PENDING
            assert status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED, OrderStatus.REJECTED]

    def test_order_book_simulation(self, exchange_simulator):
        """Test order book simulation functionality."""
        # Get simulated order book
        order_book = exchange_simulator.simulate_order_book(
            "BTC/USD",
            {"bid": 50000.0, "ask": 50100.0, "price": 50050.0}
        )

        # Verify order book structure
        assert len(order_book.bids) > 0
        assert len(order_book.asks) > 0

        # Highest bid should be less than lowest ask
        highest_bid = max(price for price, _ in order_book.bids.items())
        lowest_ask = min(price for price, _ in order_book.asks.items())
        assert highest_bid < lowest_ask

    def test_instrument_formatting(self, exchange_simulator):
        """Test instrument formatting for different exchanges."""
        # Test formatting for different exchanges
        exchange_simulator.exchange_id = "binance"
        assert exchange_simulator.format_instrument_for_exchange("BTC/USD") == "BTCUSD"

        exchange_simulator.exchange_id = "oanda"
        assert exchange_simulator.format_instrument_for_exchange("EUR/USD") == "EUR_USD"

        exchange_simulator.exchange_id = "coinbase"
        assert exchange_simulator.format_instrument_for_exchange("BTC/USD") == "BTC-USD"

        # Test parsing back
        exchange_simulator.exchange_id = "binance"
        assert exchange_simulator.parse_exchange_instrument("BTCUSDT") == "BTC/USDT"

        exchange_simulator.exchange_id = "oanda"
        assert exchange_simulator.parse_exchange_instrument("EUR_USD") == "EUR/USD"

    def test_execution_price_calculation(self, exchange_simulator):
        """Test execution price calculation with slippage."""
        # Small order should have minimal slippage
        small_price = exchange_simulator._calculate_execution_price(
            "BTC/USD",
            Side.BUY,
            0.01,
            {"bid": 50000.0, "ask": 50100.0}
        )

        # Large order should have more slippage
        large_price = exchange_simulator._calculate_execution_price(
            "BTC/USD",
            Side.BUY,
            10.0,
            {"bid": 50000.0, "ask": 50100.0}
        )

        # Larger order should have higher execution price for buys
        assert large_price > small_price

        # Sell orders should have lower price with more slippage
        small_sell_price = exchange_simulator._calculate_execution_price(
            "BTC/USD",
            Side.SELL,
            0.01,
            {"bid": 50000.0, "ask": 50100.0}
        )

        large_sell_price = exchange_simulator._calculate_execution_price(
            "BTC/USD",
            Side.SELL,
            10.0,
            {"bid": 50000.0, "ask": 50100.0}
        )

        # Larger sell should execute at lower price
        assert large_sell_price < small_sell_price

    def test_fee_calculation(self, exchange_simulator, sample_order):
        """Test fee calculation for different exchanges and order types."""
        # Set up to track fees
        order_id = exchange_simulator.place_order(sample_order)

        # Allow time for order processing
        time.sleep(0.1)

        # Check fees were applied and balances updated
        events = [(et, data) for et, data in exchange_simulator.event_bus.emitted_events
                 if et in [EventType.ORDER_FILLED, EventType.ORDER_PARTIALLY_FILLED]]

        # Should have at least one fill event
        assert len(events) > 0

        # Verify fee structure
        for _, data in events:
            fill = data["fill"]
            assert fill.fee > 0
            assert fill.fee_currency == "USD"  # For BTC/USD pair

    def test_partial_fills(self, exchange_simulator, mock_event_bus):
        """Test partial fill simulation."""
        # Set partial fill probability to 1.0 to force partial fills
        exchange_simulator.config.partial_fill_probability = 1.0
        exchange_simulator.config.min_fill_ratio = 0.7

        # Create a larger order to increase partial fill chance
        large_order = Order(
            id=str(uuid.uuid4()),
            instrument="BTC/USD",
            quantity=2.0,
            side=Side.BUY,
            type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC,
            client_order_id="test_partial_fill"
        )

        # Place the order
        order_id = exchange_simulator.place_order(large_order)

        # Allow time for order processing
        time.sleep(0.1)

        # Check for partial fill events
        partial_events = [(et, data) for et, data in mock_event_bus.emitted_events
                          if et == EventType.ORDER_PARTIALLY_FILLED]

        # Should have at least one partial fill
        assert len(partial_events) > 0

        # Partial fill should have quantity less than original
        fill = partial_events[0][1]["fill"]
        assert fill.quantity < large_order.quantity

    def test_latency_simulation(self, exchange_simulator):
        """Test latency simulation."""
        # Configure higher latency for this test
        exchange_simulator.config.base_latency_ms = 20
        exchange_simulator.config.latency_stdev_ms = 5

        # Call the latency function multiple times
        latencies = [exchange_simulator._simulate_latency() for _ in range(10)]

        # All latencies should be positive
        assert all(l > 0 for l in latencies)

        # Should have some variance
        assert len(set(latencies)) > 1