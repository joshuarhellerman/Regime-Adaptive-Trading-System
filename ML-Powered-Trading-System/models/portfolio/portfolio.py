"""
Portfolio representation module.

This module defines the Portfolio class, which represents a collection of positions
with associated metrics and functionality for analysis and manipulation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Position:
    """Represents a single position in a portfolio."""
    symbol: str
    quantity: float
    current_price: float
    cost_basis: float = 0.0
    sector: Optional[str] = None
    last_update: datetime = field(default_factory=datetime.now)

    @property
    def market_value(self) -> float:
        """Calculate the current market value of the position."""
        return self.quantity * self.current_price

    @property
    def pnl(self) -> float:
        """Calculate the unrealized profit/loss for this position."""
        return self.market_value - (self.cost_basis * self.quantity)

    @property
    def pnl_percentage(self) -> float:
        """Calculate the percentage return for this position."""
        if self.cost_basis <= 0 or self.quantity <= 0:
            return 0.0
        return (self.current_price / self.cost_basis - 1) * 100


class Portfolio:
    """
    Represents a portfolio of financial positions.

    A portfolio tracks the current holdings, their market values, weights,
    and provides methods for analysis and adjustment.
    """

    def __init__(self, cash: float = 0.0, name: str = "Default Portfolio"):
        """
        Initialize a portfolio.

        Args:
            cash: Initial cash balance
            name: Portfolio name for identification
        """
        self.positions: Dict[str, Position] = {}
        self.cash = cash
        self.name = name
        self.target_weights: Dict[str, float] = {}
        self.last_update = datetime.now()

    def add_position(self, position: Position) -> None:
        """Add or update a position in the portfolio."""
        if position.symbol in self.positions:
            # Average down/up the cost basis if position already exists
            existing = self.positions[position.symbol]
            total_quantity = existing.quantity + position.quantity
            if total_quantity > 0:
                # Weighted average of cost basis
                existing.cost_basis = ((existing.quantity * existing.cost_basis) +
                                      (position.quantity * position.cost_basis)) / total_quantity
            existing.quantity = total_quantity
            existing.current_price = position.current_price
            existing.last_update = datetime.now()
        else:
            self.positions[position.symbol] = position

        self.last_update = datetime.now()

    def update_prices(self, price_dict: Dict[str, float]) -> None:
        """
        Update the current prices of positions in the portfolio.

        Args:
            price_dict: Dictionary mapping symbols to current prices
        """
        for symbol, price in price_dict.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price
                self.positions[symbol].last_update = datetime.now()

        self.last_update = datetime.now()

    @property
    def total_value(self) -> float:
        """Calculate the total portfolio value including cash."""
        return sum(pos.market_value for pos in self.positions.values()) + self.cash

    @property
    def market_value(self) -> float:
        """Calculate the total market value of all positions (excluding cash)."""
        return sum(pos.market_value for pos in self.positions.values())

    @property
    def weights(self) -> Dict[str, float]:
        """Calculate the current portfolio weights as a fraction of total value."""
        if self.total_value <= 0:
            return {symbol: 0.0 for symbol in self.positions}

        weights = {symbol: pos.market_value / self.total_value
                  for symbol, pos in self.positions.items()}
        # Include cash weight
        weights['CASH'] = self.cash / self.total_value
        return weights

    def get_position_value(self, symbol: str) -> float:
        """Get the market value of a specific position."""
        if symbol in self.positions:
            return self.positions[symbol].market_value
        return 0.0

    def get_position_weight(self, symbol: str) -> float:
        """Get the weight of a specific position as a fraction of total portfolio value."""
        weights = self.weights
        return weights.get(symbol, 0.0)

    def set_target_weights(self, target_weights: Dict[str, float]) -> None:
        """
        Set target weights for the portfolio optimization.

        Args:
            target_weights: Dictionary mapping symbols to target portfolio weights
        """
        # Validate that weights sum to approximately 1.0
        weight_sum = sum(target_weights.values())
        if not (0.99 <= weight_sum <= 1.01):
            raise ValueError(f"Target weights must sum to 1.0, got {weight_sum}")

        self.target_weights = target_weights

    def get_rebalance_trades(self) -> Dict[str, float]:
        """
        Calculate the trades needed to reach target weights.

        Returns:
            Dictionary mapping symbols to quantity changes (positive for buys, negative for sells)
        """
        if not self.target_weights:
            return {}

        trades = {}
        target_value = self.total_value

        # Calculate target position values
        target_values = {symbol: target_value * weight
                        for symbol, weight in self.target_weights.items() if symbol != 'CASH'}

        # Calculate trade amounts
        for symbol, target_value in target_values.items():
            current_value = self.get_position_value(symbol)
            value_difference = target_value - current_value

            # Skip very small trades
            if abs(value_difference) < 0.01:
                continue

            if symbol in self.positions:
                # Convert value difference to quantity
                price = self.positions[symbol].current_price
                if price > 0:
                    quantity_change = value_difference / price
                    trades[symbol] = quantity_change
            else:
                # New position - need to get a price from elsewhere
                # This is a placeholder - in practice you would need to get the price
                # from a data service
                # For now, we'll just note that we need a price
                trades[symbol] = f"Need price for {symbol}"

        # Handle cash separately if it's in target weights
        if 'CASH' in self.target_weights:
            target_cash = target_value * self.target_weights['CASH']
            cash_difference = target_cash - self.cash
            trades['CASH'] = cash_difference

        return trades

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the portfolio to a DataFrame for analysis.

        Returns:
            DataFrame with position details
        """
        if not self.positions:
            return pd.DataFrame(columns=['symbol', 'quantity', 'price', 'market_value',
                                        'cost_basis', 'pnl', 'pnl_pct', 'weight'])

        data = []
        for symbol, pos in self.positions.items():
            data.append({
                'symbol': symbol,
                'quantity': pos.quantity,
                'price': pos.current_price,
                'market_value': pos.market_value,
                'cost_basis': pos.cost_basis,
                'pnl': pos.pnl,
                'pnl_pct': pos.pnl_percentage,
                'weight': self.get_position_weight(symbol),
                'sector': pos.sector
            })

        # Add cash position
        data.append({
            'symbol': 'CASH',
            'quantity': self.cash,
            'price': 1.0,
            'market_value': self.cash,
            'cost_basis': 1.0,
            'pnl': 0.0,
            'pnl_pct': 0.0,
            'weight': self.get_position_weight('CASH'),
            'sector': 'Cash'
        })

        return pd.DataFrame(data)

    def get_sector_exposures(self) -> Dict[str, float]:
        """
        Calculate sector exposures as a percentage of portfolio value.

        Returns:
            Dictionary mapping sectors to their weight in the portfolio
        """
        sector_values = {}
        for pos in self.positions.values():
            sector = pos.sector or 'Unknown'
            if sector not in sector_values:
                sector_values[sector] = 0
            sector_values[sector] += pos.market_value

        # Add cash as its own 'sector'
        sector_values['Cash'] = self.cash

        # Convert to percentages
        if self.total_value <= 0:
            return {sector: 0.0 for sector in sector_values}

        return {sector: value / self.total_value for sector, value in sector_values.items()}

    def __str__(self) -> str:
        """String representation of the portfolio."""
        return (f"Portfolio '{self.name}': {len(self.positions)} positions, "
                f"${self.market_value:.2f} invested, ${self.cash:.2f} cash, "
                f"${self.total_value:.2f} total")

    def summary(self) -> str:
        """Generate a text summary of the portfolio."""
        df = self.to_dataframe()

        # Sort by market value (descending)
        df = df.sort_values('market_value', ascending=False)

        # Format the output
        summary = f"Portfolio Summary: {self.name}\n"
        summary += f"Total Value: ${self.total_value:.2f}\n"
        summary += f"Invested: ${self.market_value:.2f} ({self.market_value/self.total_value*100:.2f}%)\n"
        summary += f"Cash: ${self.cash:.2f} ({self.cash/self.total_value*100:.2f}%)\n\n"

        # Top positions
        summary += "Top Positions:\n"
        for _, row in df.head(5).iterrows():
            summary += (f"  {row['symbol']}: ${row['market_value']:.2f} "
                       f"({row['weight']*100:.2f}% of portfolio)\n")

        # Sector allocation
        summary += "\nSector Allocation:\n"
        for sector, weight in self.get_sector_exposures().items():
            summary += f"  {sector}: {weight*100:.2f}%\n"

        return summary