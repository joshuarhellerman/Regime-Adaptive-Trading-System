"""
Trade generation module.

This module contains the TradeGenerator class, which is responsible for
converting portfolio weight changes into concrete trade instructions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .portfolio import Portfolio

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class TradeInstruction:
    """A single trade instruction to be sent to execution."""
    symbol: str
    quantity: float
    side: str  # 'BUY' or 'SELL'
    order_type: str = 'MARKET'  # 'MARKET', 'LIMIT', etc.
    limit_price: Optional[float] = None
    time_in_force: str = 'DAY'  # 'DAY', 'GTC', etc.
    exchange: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and clean up the trade instruction."""
        self.side = self.side.upper()
        self.order_type = self.order_type.upper()
        self.time_in_force = self.time_in_force.upper()

        # Generate a unique ID if not provided
        if self.id is None:
            import uuid
            self.id = str(uuid.uuid4())

    @property
    def value(self) -> float:
        """Calculate the approximate value of the trade."""
        price = self.limit_price
        if price is None:
            # Try to get price from metadata
            price = self.metadata.get('current_price', 0.0)

        return abs(self.quantity * price)

    def __str__(self) -> str:
        """String representation of the trade instruction."""
        return (f"{self.side} {abs(self.quantity)} {self.symbol} "
                f"@ {self.order_type}{' ' + str(self.limit_price) if self.limit_price else ''}")


@dataclass
class TradeGenerationResult:
    """Container for the results of trade generation."""
    trades: List[TradeInstruction]
    total_value: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """String representation of the trade generation result."""
        return (f"Generated {len(self.trades)} trades "
                f"with total value {self.total_value:.2f}")


class TradeGenerator:
    """
    Generates trade instructions from portfolio weight changes.

    The trade generator takes current portfolio positions and target weights,
    and calculates the trades needed to achieve the target allocation.
    """

    def __init__(self,
                 min_trade_value: float = 100.0,
                 min_trade_quantity: float = 1.0,
                 max_trade_value: Optional[float] = None,
                 round_quantities: bool = True,
                 cash_key: str = 'CASH'):
        """
        Initialize a trade generator.

        Args:
            min_trade_value: Minimum value of a trade to execute
            min_trade_quantity: Minimum quantity for a trade
            max_trade_value: Maximum value for a single trade
            round_quantities: Whether to round quantities to integers
            cash_key: Symbol used to represent cash in the portfolio
        """
        self.min_trade_value = min_trade_value
        self.min_trade_quantity = min_trade_quantity
        self.max_trade_value = max_trade_value
        self.round_quantities = round_quantities
        self.cash_key = cash_key

    def generate_trades(self,
                       current_portfolio: Portfolio,
                       target_weights: Dict[str, float],
                       prices: Optional[Dict[str, float]] = None,
                       lot_sizes: Optional[Dict[str, float]] = None) -> TradeGenerationResult:
        """
        Generate trades to achieve target portfolio weights.

        Args:
            current_portfolio: Current portfolio object
            target_weights: Target portfolio weights
            prices: Current prices for assets (if None, uses portfolio prices)
            lot_sizes: Minimum lot sizes for each symbol

        Returns:
            TradeGenerationResult containing trade instructions
        """
        # Validate inputs
        if not target_weights:
            return TradeGenerationResult(trades=[], total_value=0.0)

        # If prices not provided, use prices from the portfolio
        if prices is None:
            prices = {}
            for symbol, position in current_portfolio.positions.items():
                prices[symbol] = position.current_price

        # Extract current weights from the portfolio
        current_weights = current_portfolio.weights
        total_value = current_portfolio.total_value

        # Identify differences between current and target weights
        trade_fractions = {}
        for symbol in set(list(current_weights.keys()) + list(target_weights.keys())):
            current_weight = current_weights.get(symbol, 0.0)
            target_weight = target_weights.get(symbol, 0.0)

            # Skip cash and negligible differences
            if symbol == self.cash_key:
                continue

            weight_diff = target_weight - current_weight
            if abs(weight_diff) < 1e-6:
                continue

            trade_fractions[symbol] = weight_diff

        # Convert weight differences to trade values
        trade_values = {symbol: fraction * total_value for symbol, fraction in trade_fractions.items()}

        # Convert trade values to quantities
        trades = []
        for symbol, value in trade_values.items():
            # Skip trades below minimum value
            if abs(value) < self.min_trade_value:
                continue

            # Get price for this symbol
            if symbol not in prices:
                logger.warning(f"No price available for {symbol}, skipping trade")
                continue

            price = prices[symbol]
            if price <= 0:
                logger.warning(f"Invalid price {price} for {symbol}, skipping trade")
                continue

            # Calculate quantity
            quantity = value / price

            # Apply lot sizes if provided
            if lot_sizes and symbol in lot_sizes and lot_sizes[symbol] > 0:
                # Round to nearest lot size
                lot_size = lot_sizes[symbol]
                quantity = round(quantity / lot_size) * lot_size

            # Round to integers if requested
            if self.round_quantities:
                quantity = round(quantity)

            # Skip trades below minimum quantity
            if abs(quantity) < self.min_trade_quantity:
                continue

            # Cap trade value if maximum is specified
            if self.max_trade_value is not None and abs(value) > self.max_trade_value:
                sign = 1 if quantity > 0 else -1
                quantity = sign * (self.max_trade_value / price)
                if self.round_quantities:
                    quantity = round(quantity)

            # Create trade instruction
            side = 'BUY' if quantity > 0 else 'SELL'
            trades.append(TradeInstruction(
                symbol=symbol,
                quantity=abs(quantity),
                side=side,
                limit_price=price,
                metadata={
                    'current_price': price,
                    'target_weight': target_weights.get(symbol, 0.0),
                    'current_weight': current_weights.get(symbol, 0.0),
                    'weight_change': trade_fractions.get(symbol, 0.0)
                }
            ))

        # Calculate total trade value
        total_trade_value = sum(trade.value for trade in trades)

        return TradeGenerationResult(
            trades=trades,
            total_value=total_trade_value,
            metadata={
                'portfolio_value': total_value,
                'current_weights': current_weights,
                'target_weights': target_weights
            }
        )

    def generate_trades_from_signals(self,
                                    current_portfolio: Portfolio,
                                    alpha_signals: Dict[str, float],
                                    prices: Optional[Dict[str, float]] = None,
                                    risk_model: Optional[pd.DataFrame] = None) -> TradeGenerationResult:
        """
        Generate trades directly from alpha signals, handling the optimization step.

        Args:
            current_portfolio: Current portfolio object
            alpha_signals: Dictionary mapping symbols to alpha signals
            prices: Current prices for assets (if None, uses portfolio prices)
            risk_model: Covariance matrix for risk management

        Returns:
            TradeGenerationResult containing trade instructions
        """
        # Import here to avoid circular imports
        from .optimizer import PortfolioOptimizer

        # Create an optimizer and convert signals to weights
        optimizer = PortfolioOptimizer()

        # Extract current weights from the portfolio
        current_weights = current_portfolio.weights

        # Generate target weights
        target_weights = optimizer.alpha_to_weights(
            alpha_results=alpha_signals,
            current_portfolio=current_weights,
            risk_model=risk_model
        )

        # Generate trades from the target weights
        return self.generate_trades(
            current_portfolio=current_portfolio,
            target_weights=target_weights,
            prices=prices
        )

    def generate_rebalance_trades(self, portfolio: Portfolio) -> TradeGenerationResult:
        """
        Generate trades to rebalance a portfolio to its target weights.

        Args:
            portfolio: Portfolio object with target_weights set

        Returns:
            TradeGenerationResult containing trade instructions
        """
        if not portfolio.target_weights:
            logger.warning("No target weights set for portfolio, cannot rebalance")
            return TradeGenerationResult(trades=[], total_value=0.0)

        return self.generate_trades(
            current_portfolio=portfolio,
            target_weights=portfolio.target_weights
        )

    def split_large_trades(self,
                          trades: List[TradeInstruction],
                          max_value: float,
                          max_quantity: Optional[float] = None) -> List[TradeInstruction]:
        """
        Split large trades into smaller chunks to minimize market impact.

        Args:
            trades: List of trade instructions
            max_value: Maximum value per trade
            max_quantity: Maximum quantity per trade

        Returns:
            List of trade instructions with large trades split
        """
        result = []

        for trade in trades:
            # Check if trade needs splitting by value
            if trade.value <= max_value and (max_quantity is None or trade.quantity <= max_quantity):
                # Trade is small enough, include as is
                result.append(trade)
                continue

            # Determine the limiting factor (value or quantity)
            splits_by_value = 1
            splits_by_quantity = 1

            if trade.value > max_value:
                splits_by_value = int(np.ceil(trade.value / max_value))

            if max_quantity is not None and trade.quantity > max_quantity:
                splits_by_quantity = int(np.ceil(trade.quantity / max_quantity))

            # Use the higher split count
            num_splits = max(splits_by_value, splits_by_quantity)

            # Split the trade
            for i in range(num_splits):
                # Calculate quantity for this chunk
                if i < num_splits - 1:
                    # Equal chunks except possibly the last one
                    chunk_quantity = trade.quantity / num_splits
                    if self.round_quantities:
                        chunk_quantity = round(chunk_quantity)
                else:
                    # Last chunk gets the remainder
                    prev_chunks = num_splits - 1
                    prev_quantity = prev_chunks * (trade.quantity // num_splits)
                    chunk_quantity = trade.quantity - prev_quantity

                # Create a new trade instruction for this chunk
                chunk_trade = TradeInstruction(
                    symbol=trade.symbol,
                    quantity=chunk_quantity,
                    side=trade.side,
                    order_type=trade.order_type,
                    limit_price=trade.limit_price,
                    time_in_force=trade.time_in_force,
                    exchange=trade.exchange,
                    metadata={
                        **trade.metadata,
                        'parent_id': trade.id,
                        'chunk': i + 1,
                        'total_chunks': num_splits
                    }
                )
                result.append(chunk_trade)

        return result