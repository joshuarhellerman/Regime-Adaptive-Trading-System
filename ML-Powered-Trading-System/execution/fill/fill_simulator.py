"""
fill_simulator.py - Realistic fill simulation for paper trading

This module provides a realistic simulation of order fills for paper trading,
including market impact, partial fills, and execution latency.
"""

import logging
import random
import time
import uuid
import threading
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from decimal import Decimal
import math
import numpy as np

from core.event_bus import EventBus, Event, EventTopics, EventPriority, create_event
from data.market_data_service import MarketDataService
from execution.order.order import Order, OrderStatus, OrderSide, OrderType
from execution.fill.fill_model import Fill

logger = logging.getLogger(__name__)

class SimulationModel:
    """Base class for simulation models."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the simulation model.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}


class LatencyModel(SimulationModel):
    """Model for simulating execution latency."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the latency model.

        Args:
            config: Configuration dictionary with latency settings
        """
        super().__init__(config)

        # Default latency parameters (ms)
        self.base_latency = self.config.get("base_latency_ms", 50)
        self.latency_std_dev = self.config.get("latency_std_dev_ms", 25)
        self.market_load_factor = self.config.get("market_load_factor", 1.0)
        self.size_factor = self.config.get("size_factor", 0.1)
        self.max_latency = self.config.get("max_latency_ms", 5000)

    def get_latency(self, order_size: float, market_volume: float = None) -> float:
        """
        Calculate simulated execution latency.

        Args:
            order_size: Size of the order
            market_volume: Current market volume (if available)

        Returns:
            Latency in milliseconds
        """
        # Base latency with normal distribution
        latency = max(10, random.normalvariate(self.base_latency, self.latency_std_dev))

        # Add penalty for order size
        size_penalty = 1.0
        if market_volume and market_volume > 0:
            # Relative to market volume
            size_penalty = 1.0 + (order_size / market_volume * self.size_factor)

        # Apply market load factor (time of day, volatility, etc.)
        latency *= self.market_load_factor * size_penalty

        # Cap at maximum latency
        return min(latency, self.max_latency)


class PartialFillModel(SimulationModel):
    """Model for simulating partial fills."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the partial fill model.

        Args:
            config: Configuration dictionary with partial fill settings
        """
        super().__init__(config)

        # Default partial fill parameters
        self.partial_fill_probability = self.config.get("partial_fill_probability", 0.3)
        self.min_fills = self.config.get("min_fills", 1)
        self.max_fills = self.config.get("max_fills", 5)
        self.liquidity_threshold = self.config.get("liquidity_threshold", 0.1)

    def generate_fills(self, order_size: float, market_volume: float = None) -> List[float]:
        """
        Generate a sequence of partial fills.

        Args:
            order_size: Size of the order
            market_volume: Current market volume (if available)

        Returns:
            List of fill sizes that add up to order_size
        """
        # Determine if we should do partial fills
        do_partial = random.random() < self.partial_fill_probability

        if not do_partial:
            return [order_size]

        # Adjust based on market volume
        size_ratio = 1.0
        if market_volume and market_volume > 0:
            size_ratio = min(1.0, order_size / market_volume)

        # More fills for larger orders relative to market
        num_fills = min(
            self.max_fills,
            max(self.min_fills,
                int(1 + self.max_fills * size_ratio / self.liquidity_threshold))
        )

        # Generate random fill sizes
        if num_fills == 1:
            return [order_size]

        # Generate random weights for each fill
        weights = [random.random() for _ in range(num_fills)]
        total_weight = sum(weights)

        # Normalize weights and calculate fill sizes
        fill_sizes = [order_size * weight / total_weight for weight in weights]

        # Ensure the sum is exactly the order size
        fill_sizes[-1] = order_size - sum(fill_sizes[:-1])

        # Return in time order (smaller fills tend to come first)
        return sorted(fill_sizes)


class PriceImpactModel(SimulationModel):
    """Model for simulating market impact on execution price."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the price impact model.

        Args:
            config: Configuration dictionary with price impact settings
        """
        super().__init__(config)

        # Default price impact parameters
        self.market_impact_factor = self.config.get("market_impact_factor", 0.1)
        self.volatility_factor = self.config.get("volatility_factor", 0.2)
        self.bid_ask_spread_bps = self.config.get("bid_ask_spread_bps", 5)
        self.use_spread = self.config.get("use_spread", True)
        self.randomize_within_spread = self.config.get("randomize_within_spread", True)

    def calculate_price(self, side: OrderSide, size: float, market_data: Dict[str, Any]) -> float:
        """
        Calculate execution price with simulated market impact.

        Args:
            side: Buy or sell side
            size: Order size
            market_data: Current market data

        Returns:
            Execution price
        """
        # Extract market data
        if 'mid_price' in market_data:
            price = market_data['mid_price']
        elif 'close' in market_data:
            price = market_data['close']
        else:
            price = market_data.get('price', 0)

        volume = market_data.get('volume', 1000)

        # Simulate bid-ask spread
        spread_pct = self.bid_ask_spread_bps / 10000.0
        half_spread = price * spread_pct / 2

        if self.use_spread:
            # Adjust price based on side (buy at ask, sell at bid)
            if side == OrderSide.BUY:
                base_price = price + half_spread
            else:
                base_price = price - half_spread
        else:
            base_price = price

        # Calculate market impact based on order size relative to volume
        if volume > 0:
            relative_size = size / volume
            impact_pct = self.market_impact_factor * math.sqrt(relative_size)

            # Impact direction depends on side
            if side == OrderSide.BUY:
                impact = base_price * impact_pct
            else:
                impact = -base_price * impact_pct
        else:
            impact = 0

        # Add random price noise based on volatility
        if 'volatility' in market_data:
            volatility = market_data['volatility']
        else:
            volatility = 0.01  # Default 1% volatility

        noise_scale = price * volatility * self.volatility_factor
        noise = random.normalvariate(0, noise_scale)

        # Calculate final price
        final_price = base_price + impact + noise

        # Ensure price is positive
        return max(0.00000001, final_price)


class FillSimulator:
    """
    Simulates realistic fills for paper trading and backtesting.

    Features:
    - Latency simulation
    - Partial fills
    - Price impact modeling
    - Realistic order book interaction
    """

    def __init__(self,
                market_data_service: MarketDataService,
                event_bus: EventBus,
                config: Dict[str, Any] = None):
        """
        Initialize the fill simulator.

        Args:
            market_data_service: Market data service
            event_bus: Event bus
            config: Configuration dictionary
        """
        self.market_data_service = market_data_service
        self.event_bus = event_bus
        self.config = config or {}

        # Initialize simulation models
        self.fill_latency_model = LatencyModel(self.config.get("latency", {}))
        self.partial_fill_model = PartialFillModel(self.config.get("partial_fills", {}))
        self.price_impact_model = PriceImpactModel(self.config.get("price_impact", {}))

        # Internal state
        self._active_simulations = set()
        self._lock = threading.RLock()
        self._loop = asyncio.new_event_loop()
        self._running = False
        self._thread = None

        # Subscribe to paper trading order events
        self.event_bus.subscribe("paper.order.new", self.simulate_fill)

    def start(self):
        """Start the fill simulator service."""
        with self._lock:
            if self._running:
                return

            self._running = True

            # Start event loop thread
            self._thread = threading.Thread(
                target=self._run_event_loop,
                name="FillSimulatorThread",
                daemon=True
            )
            self._thread.start()

            logger.info("FillSimulator started")

    def stop(self):
        """Stop the fill simulator service."""
        with self._lock:
            if not self._running:
                return

            self._running = False

            # Stop event loop
            if self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)

            # Wait for thread to finish
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5.0)

            logger.info("FillSimulator stopped")

    def _run_event_loop(self):
        """Run the asyncio event loop in a separate thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()
        logger.debug("FillSimulator event loop stopped")

    def simulate_fill(self, event: Event) -> None:
        """
        Handle an order event and simulate fills.

        Args:
            event: Order event from paper trading
        """
        try:
            # Get order from event
            order = event.data

            # Schedule simulation in event loop
            self._loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(self._simulate_fill_async(order))
            )
        except Exception as e:
            logger.error(f"Error in fill simulation for order: {e}", exc_info=True)

    async def _simulate_fill_async(self, order: Order) -> None:
        """
        Simulate fills for a paper trading order asynchronously.

        Args:
            order: Order to simulate fills for
        """
        try:
            # Generate a unique simulation ID
            simulation_id = str(uuid.uuid4())

            # Track active simulation
            with self._lock:
                self._active_simulations.add(simulation_id)

            # Get current market data
            market_data = self._get_market_data(order.symbol)

            # Calculate execution latency
            volume = market_data.get('volume', 0)
            latency_ms = self.fill_latency_model.get_latency(order.quantity, volume)

            # Wait for simulated latency
            await asyncio.sleep(latency_ms / 1000)

            # Check if simulation was cancelled
            if not self._is_simulation_active(simulation_id):
                return

            # Simulate fill
            await self._execute_fill(order, simulation_id, market_data)

        except Exception as e:
            logger.error(f"Error in fill simulation for {order.order_id}: {e}", exc_info=True)
        finally:
            # Remove from active simulations
            with self._lock:
                self._active_simulations.discard(simulation_id)

    async def _execute_fill(self, order: Order, simulation_id: str, market_data: Dict[str, Any]) -> None:
        """
        Execute the simulated fill.

        Args:
            order: Order to fill
            simulation_id: Unique simulation ID
            market_data: Current market data
        """
        # Check if simulation is still active
        if not self._is_simulation_active(simulation_id):
            return

        # Generate fill sizes
        volume = market_data.get('volume', 0)
        fill_sizes = self.partial_fill_model.generate_fills(order.quantity, volume)

        # Process each fill
        fill_time = datetime.utcnow()
        total_filled = 0
        total_value = 0

        for i, size in enumerate(fill_sizes):
            # Wait between partial fills
            if i > 0:
                delay = random.uniform(0.5, 2.0)
                await asyncio.sleep(delay)

                # Check if simulation is still active
                if not self._is_simulation_active(simulation_id):
                    return

                # Update market data
                market_data = self._get_market_data(order.symbol)
                fill_time = datetime.utcnow()

            # Simulate price impact
            price = self.price_impact_model.calculate_price(order.side, size, market_data)

            # Simulate fees (simple model - improve this for production)
            fees = price * size * 0.001  # 0.1% fee

            # Record the fill
            fill = Fill(
                order_id=order.order_id,
                fill_id=f"{order.order_id}_fill_{i + 1}",
                timestamp=fill_time,
                instrument=order.symbol,
                quantity=Decimal(str(size)),
                price=Decimal(str(price)),
                fees=Decimal(str(fees)),
                exchange_id="paper_exchange",
                is_maker=bool(random.random() < 0.7),  # 70% chance of maker
                metadata={
                    "simulated": True,
                    "fill_index": i,
                    "total_fills": len(fill_sizes),
                    "market_data": {k: v for k, v in market_data.items() if k != 'orderbook'}
                }
            )

            # Publish fill event
            self._publish_fill(fill)

            # Update totals
            total_filled += size
            total_value += size * price

            logger.debug(f"Simulated fill {i + 1}/{len(fill_sizes)} for order {order.order_id}: "
                         f"{size} @ {price}, time: {fill_time.isoformat()}")

        logger.info(f"Completed fill simulation for order {order.order_id}: "
                    f"{total_filled} filled in {len(fill_sizes)} parts, "
                    f"avg price: {total_value / total_filled if total_filled else 0}")

    def _get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get current market data for a symbol.

        Args:
            symbol: Instrument symbol

        Returns:
            Dictionary with market data
        """
        try:
            # Get ticker data
            ticker = self.market_data_service.get_ticker(symbol)

            # Get order book
            orderbook = self.market_data_service.get_market_depth(symbol)

            # Combine data
            data = {
                'price': ticker.get('price', ticker.get('last', 0)),
                'bid': ticker.get('bid', 0),
                'ask': ticker.get('ask', 0),
                'volume': ticker.get('volume', 0),
                'mid_price': (ticker.get('bid', 0) + ticker.get('ask', 0)) / 2,
                'timestamp': datetime.now(),
                'orderbook': orderbook
            }

            # Calculate volatility if available
            if 'high' in ticker and 'low' in ticker and ticker['high'] > 0:
                data['volatility'] = (ticker['high'] - ticker['low']) / ticker['high']

            return data

        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")

            # Return minimal default data
            return {
                'price': 0,
                'bid': 0,
                'ask': 0,
                'volume': 0,
                'mid_price': 0,
                'timestamp': datetime.now()
            }

    def _is_simulation_active(self, simulation_id: str) -> bool:
        """
        Check if a simulation is still active.

        Args:
            simulation_id: Simulation ID to check

        Returns:
            True if the simulation is active
        """
        with self._lock:
            return simulation_id in self._active_simulations and self._running

    def _publish_fill(self, fill: Fill) -> None:
        """
        Publish a fill event.

        Args:
            fill: Fill to publish
        """
        event = create_event(
            "paper.order.filled",
            fill,
            priority=EventPriority.HIGH,
            source="fill_simulator"
        )

        self.event_bus.publish(event)

    def cancel_simulation(self, order_id: str) -> None:
        """
        Cancel any ongoing fill simulations for an order.

        Args:
            order_id: Order ID to cancel simulations for
        """
        # In a more complex implementation, we would track simulations by order ID
        # For now, we'll just log the cancellation
        logger.info(f"Received cancellation for order {order_id}, but individual simulation tracking not implemented")