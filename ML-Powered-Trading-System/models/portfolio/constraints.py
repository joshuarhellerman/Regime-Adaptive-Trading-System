"""
Portfolio constraints module.

This module defines constraints that can be applied during portfolio optimization,
such as leverage limits, position size limits, sector exposure limits, etc.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Set, Union
import pandas as pd


class PortfolioConstraint(ABC):
    """Base class for all portfolio constraints."""

    @abstractmethod
    def is_satisfied(self, weights: Dict[str, float], metadata: Optional[Dict] = None) -> bool:
        """
        Check if the constraint is satisfied by the given weights.

        Args:
            weights: Dictionary mapping symbols to portfolio weights
            metadata: Additional metadata needed for constraint evaluation

        Returns:
            True if the constraint is satisfied, False otherwise
        """
        pass

    @abstractmethod
    def apply(self, weights: Dict[str, float], metadata: Optional[Dict] = None) -> Dict[str, float]:
        """
        Apply the constraint to the given weights, modifying them if necessary.

        Args:
            weights: Dictionary mapping symbols to portfolio weights
            metadata: Additional metadata needed for constraint application

        Returns:
            Modified weights that satisfy the constraint
        """
        pass

    def __str__(self) -> str:
        """String representation of the constraint."""
        return self.__class__.__name__


class LeverageConstraint(PortfolioConstraint):
    """
    Constraint on the total leverage of the portfolio.

    This constraint ensures that the sum of absolute weights does not exceed
    the maximum allowed leverage.
    """

    def __init__(self, max_leverage: float = 1.0, exclude_cash: bool = True):
        """
        Initialize a leverage constraint.

        Args:
            max_leverage: Maximum allowed leverage (1.0 = no leverage)
            exclude_cash: Whether to exclude cash from leverage calculation
        """
        if max_leverage <= 0:
            raise ValueError("Maximum leverage must be positive")

        self.max_leverage = max_leverage
        self.exclude_cash = exclude_cash

    def is_satisfied(self, weights: Dict[str, float], metadata: Optional[Dict] = None) -> bool:
        """
        Check if the leverage constraint is satisfied.

        Args:
            weights: Dictionary mapping symbols to portfolio weights
            metadata: Not used for this constraint

        Returns:
            True if the leverage constraint is satisfied, False otherwise
        """
        # Calculate the sum of absolute weights (leverage)
        if self.exclude_cash and 'CASH' in weights:
            cash_weight = weights['CASH']
            leverage = sum(abs(w) for s, w in weights.items() if s != 'CASH')
        else:
            leverage = sum(abs(w) for w in weights.values())

        return leverage <= self.max_leverage

    def apply(self, weights: Dict[str, float], metadata: Optional[Dict] = None) -> Dict[str, float]:
        """
        Apply the leverage constraint by scaling weights if necessary.

        Args:
            weights: Dictionary mapping symbols to portfolio weights
            metadata: Not used for this constraint

        Returns:
            Modified weights that satisfy the leverage constraint
        """
        if self.is_satisfied(weights):
            return weights.copy()

        result = weights.copy()

        # Calculate the current leverage
        if self.exclude_cash and 'CASH' in weights:
            cash_weight = weights['CASH']
            non_cash_weights = {s: w for s, w in weights.items() if s != 'CASH'}
            leverage = sum(abs(w) for w in non_cash_weights.values())

            # Scale non-cash weights to meet leverage constraint
            scaling_factor = self.max_leverage / leverage
            for symbol in non_cash_weights:
                result[symbol] *= scaling_factor

            # Adjust cash to maintain sum of weights = 1
            result['CASH'] = 1.0 - sum(result[s] for s in non_cash_weights)
        else:
            leverage = sum(abs(w) for w in weights.values())
            scaling_factor = self.max_leverage / leverage

            # Scale all weights
            for symbol in weights:
                result[symbol] *= scaling_factor

        return result

    def __str__(self) -> str:
        """String representation of the leverage constraint."""
        return f"LeverageConstraint(max_leverage={self.max_leverage}, exclude_cash={self.exclude_cash})"


class PositionSizeConstraint(PortfolioConstraint):
    """
    Constraint on the size of individual positions.

    This constraint ensures that no position exceeds a maximum size,
    and optionally that all positions meet a minimum size.
    """

    def __init__(self, max_size: float = 0.2, min_size: float = 0.0, exclude_cash: bool = True):
        """
        Initialize a position size constraint.

        Args:
            max_size: Maximum allowed position size as a fraction of the portfolio
            min_size: Minimum allowed position size (positions below this will be eliminated)
            exclude_cash: Whether to exclude cash from position size constraints
        """
        if max_size <= 0 or max_size > 1:
            raise ValueError("Maximum position size must be between 0 and 1")
        if min_size < 0 or min_size >= max_size:
            raise ValueError("Minimum position size must be between 0 and max_size")

        self.max_size = max_size
        self.min_size = min_size
        self.exclude_cash = exclude_cash

    def is_satisfied(self, weights: Dict[str, float], metadata: Optional[Dict] = None) -> bool:
        """
        Check if position size constraints are satisfied.

        Args:
            weights: Dictionary mapping symbols to portfolio weights
            metadata: Not used for this constraint

        Returns:
            True if all position size constraints are satisfied, False otherwise
        """
        for symbol, weight in weights.items():
            if self.exclude_cash and symbol == 'CASH':
                continue

            # Check if any position exceeds the maximum size
            if abs(weight) > self.max_size:
                return False

            # Check if any position is below the minimum size but not zero
            if self.min_size > 0 and 0 < abs(weight) < self.min_size:
                return False

        return True

    def apply(self, weights: Dict[str, float], metadata: Optional[Dict] = None) -> Dict[str, float]:
        """
        Apply position size constraints by capping oversized positions and
        eliminating undersized positions.

        Args:
            weights: Dictionary mapping symbols to portfolio weights
            metadata: Not used for this constraint

        Returns:
            Modified weights that satisfy position size constraints
        """
        if self.is_satisfied(weights):
            return weights.copy()

        result = weights.copy()
        cash_key = 'CASH'
        excess = 0.0

        # First pass: cap oversized positions and eliminate small positions
        for symbol in list(result.keys()):  # Use list to avoid dictionary size change during iteration
            if self.exclude_cash and symbol == cash_key:
                continue

            weight = result[symbol]
            abs_weight = abs(weight)

            # Cap oversized positions
            if abs_weight > self.max_size:
                sign = 1 if weight > 0 else -1
                excess += abs_weight - self.max_size
                result[symbol] = sign * self.max_size

            # Eliminate small positions
            elif self.min_size > 0 and 0 < abs_weight < self.min_size:
                excess += abs_weight
                result[symbol] = 0.0

        # Second pass: redistribute excess to remaining positions proportionally
        if excess > 0:
            remaining_symbols = [s for s, w in result.items()
                                 if s != cash_key and abs(w) >= self.min_size]

            if remaining_symbols:
                total_remaining = sum(abs(result[s]) for s in remaining_symbols)

                if total_remaining > 0:
                    for symbol in remaining_symbols:
                        weight = result[symbol]
                        abs_weight = abs(weight)
                        sign = 1 if weight > 0 else -1

                        # Distribute excess proportionally, but respect max_size
                        proportion = abs_weight / total_remaining
                        additional = min(excess * proportion,
                                         self.max_size - abs_weight)

                        result[symbol] = sign * (abs_weight + additional)
                        excess -= additional

            # Put any remaining excess into cash
            if excess > 0 and cash_key in result:
                result[cash_key] += excess

        # Normalize weights to ensure they sum to 1.0
        weight_sum = sum(result.values())
        if abs(weight_sum - 1.0) > 1e-10:
            for symbol in result:
                result[symbol] /= weight_sum

        return result

    def __str__(self) -> str:
        """String representation of the position size constraint."""
        return (f"PositionSizeConstraint(max_size={self.max_size}, "
                f"min_size={self.min_size}, exclude_cash={self.exclude_cash})")


class SectorConstraint(PortfolioConstraint):
    """
    Constraint on sector exposures in the portfolio.

    This constraint ensures that sector allocations stay within specified limits.
    """

    def __init__(self,
                 max_sector_exposure: float = 0.3,
                 min_sector_exposure: float = 0.0,
                 sector_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize a sector constraint.

        Args:
            max_sector_exposure: Maximum allowed exposure to any single sector
            min_sector_exposure: Minimum required exposure to each sector
            sector_mapping: Dictionary mapping symbols to their respective sectors
        """
        if max_sector_exposure <= 0 or max_sector_exposure > 1:
            raise ValueError("Maximum sector exposure must be between 0 and 1")
        if min_sector_exposure < 0 or min_sector_exposure >= max_sector_exposure:
            raise ValueError("Minimum sector exposure must be between 0 and max_sector_exposure")

        self.max_sector_exposure = max_sector_exposure
        self.min_sector_exposure = min_sector_exposure
        self.sector_mapping = sector_mapping or {}

    def is_satisfied(self, weights: Dict[str, float], metadata: Optional[Dict] = None) -> bool:
        """
        Check if sector constraints are satisfied.

        Args:
            weights: Dictionary mapping symbols to portfolio weights
            metadata: Dictionary that can include a 'sector_mapping' if not provided at init

        Returns:
            True if all sector constraints are satisfied, False otherwise
        """
        sector_mapping = metadata.get('sector_mapping', self.sector_mapping) if metadata else self.sector_mapping
        if not sector_mapping:
            # If no sector mapping is available, we can't evaluate sector constraints
            return True

        # Calculate sector exposures
        sector_exposures = {}
        for symbol, weight in weights.items():
            if symbol == 'CASH':
                continue

            sector = sector_mapping.get(symbol, 'Unknown')
            if sector not in sector_exposures:
                sector_exposures[sector] = 0.0
            sector_exposures[sector] += abs(weight)

        # Check if any sector exceeds the maximum exposure
        if any(exposure > self.max_sector_exposure for exposure in sector_exposures.values()):
            return False

        # Check if any sector is below the minimum exposure but not zero
        if self.min_sector_exposure > 0:
            all_sectors = set(sector_mapping.values())
            for sector in all_sectors:
                exposure = sector_exposures.get(sector, 0.0)
                if 0 < exposure < self.min_sector_exposure:
                    return False

        return True

    def apply(self, weights: Dict[str, float], metadata: Optional[Dict] = None) -> Dict[str, float]:
        """
        Apply sector constraints by scaling sector exposures if necessary.

        Args:
            weights: Dictionary mapping symbols to portfolio weights
            metadata: Dictionary that can include a 'sector_mapping' if not provided at init

        Returns:
            Modified weights that satisfy sector constraints
        """
        sector_mapping = metadata.get('sector_mapping', self.sector_mapping) if metadata else self.sector_mapping
        if not sector_mapping or self.is_satisfied(weights, metadata):
            return weights.copy()

        result = weights.copy()
        cash_key = 'CASH'

        # Group symbols by sector
        sectors = {}
        for symbol, weight in weights.items():
            if symbol == cash_key:
                continue

            sector = sector_mapping.get(symbol, 'Unknown')
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(symbol)

        # Calculate current sector exposures
        sector_exposures = {}
        for sector, symbols in sectors.items():
            sector_exposures[sector] = sum(abs(result[s]) for s in symbols)

        excess = 0.0

        # First pass: cap oversized sectors and eliminate undersized sectors
        for sector, exposure in sector_exposures.items():
            # Cap oversized sectors
            if exposure > self.max_sector_exposure:
                # Scale down all positions in this sector proportionally
                scaling_factor = self.max_sector_exposure / exposure
                for symbol in sectors[sector]:
                    original = result[symbol]
                    result[symbol] *= scaling_factor
                    excess += abs(original) - abs(result[symbol])

            # Eliminate undersized sectors
            elif self.min_sector_exposure > 0 and 0 < exposure < self.min_sector_exposure:
                for symbol in sectors[sector]:
                    excess += abs(result[symbol])
                    result[symbol] = 0.0

        # Second pass: redistribute excess to remaining sectors proportionally
        if excess > 0:
            remaining_sectors = [s for s, e in sector_exposures.items()
                                 if e >= self.min_sector_exposure and e < self.max_sector_exposure]

            if remaining_sectors:
                # Recalculate sector exposures after first pass
                sector_exposures = {sector: sum(abs(result[s]) for s in symbols)
                                    for sector, symbols in sectors.items()}

                total_remaining = sum(sector_exposures[s] for s in remaining_sectors)

                if total_remaining > 0:
                    for sector in remaining_sectors:
                        exposure = sector_exposures[sector]

                        # How much more can this sector take before hitting max_exposure
                        sector_capacity = self.max_sector_exposure - exposure

                        # Distribute excess proportionally to sector capacity
                        proportion = exposure / total_remaining
                        sector_additional = min(excess * proportion, sector_capacity)

                        # Distribute additional within the sector proportionally
                        sector_symbols = sectors[sector]
                        sector_total = sum(abs(result[s]) for s in sector_symbols)

                        if sector_total > 0:
                            for symbol in sector_symbols:
                                weight = result[symbol]
                                abs_weight = abs(weight)
                                sign = 1 if weight > 0 else -1

                                symbol_proportion = abs_weight / sector_total
                                symbol_additional = sector_additional * symbol_proportion

                                result[symbol] = sign * (abs_weight + symbol_additional)
                                excess -= symbol_additional

            # Put any remaining excess into cash
            if excess > 0 and cash_key in result:
                result[cash_key] += excess

        # Normalize weights to ensure they sum to 1.0
        weight_sum = sum(result.values())
        if abs(weight_sum - 1.0) > 1e-10:
            for symbol in result:
                result[symbol] /= weight_sum

        return result

    def __str__(self) -> str:
        """String representation of the sector constraint."""
        return (f"SectorConstraint(max_exposure={self.max_sector_exposure}, "
                f"min_exposure={self.min_sector_exposure})")


class TurnoverConstraint(PortfolioConstraint):
    """
    Constraint on portfolio turnover.

    This constraint limits the amount of portfolio change between rebalancings.
    """

    def __init__(self, max_turnover: float = 0.2, current_weights: Optional[Dict[str, float]] = None):
        """
        Initialize a turnover constraint.

        Args:
            max_turnover: Maximum allowed turnover (sum of absolute weight changes)
            current_weights: Current portfolio weights (if available at init time)
        """
        if max_turnover < 0 or max_turnover > 2:
            raise ValueError("Maximum turnover must be between 0 and 2")

        self.max_turnover = max_turnover
        self.current_weights = current_weights or {}

    def is_satisfied(self, weights: Dict[str, float], metadata: Optional[Dict] = None) -> bool:
        """
        Check if the turnover constraint is satisfied.

        Args:
            weights: Target portfolio weights
            metadata: Dictionary that can include 'current_weights' if not provided at init

        Returns:
            True if the turnover constraint is satisfied, False otherwise
        """
        current_weights = metadata.get('current_weights', self.current_weights) if metadata else self.current_weights
        if not current_weights:
            # If no current weights are available, we can't evaluate turnover
            return True

        # Calculate turnover as sum of absolute weight differences
        turnover = 0.0
        all_symbols = set(list(weights.keys()) + list(current_weights.keys()))

        for symbol in all_symbols:
            current = current_weights.get(symbol, 0.0)
            target = weights.get(symbol, 0.0)
            turnover += abs(target - current)

        # Total turnover is the sum divided by 2 (since each trade affects two positions)
        turnover /= 2.0

        return turnover <= self.max_turnover

    def apply(self, weights: Dict[str, float], metadata: Optional[Dict] = None) -> Dict[str, float]:
        """
        Apply the turnover constraint by moving weights partially toward target.

        Args:
            weights: Target portfolio weights
            metadata: Dictionary that can include 'current_weights' if not provided at init

        Returns:
            Modified weights that satisfy the turnover constraint
        """
        current_weights = metadata.get('current_weights', self.current_weights) if metadata else self.current_weights
        if not current_weights or self.is_satisfied(weights, metadata):
            return weights.copy()

        # Calculate current turnover
        all_symbols = set(list(weights.keys()) + list(current_weights.keys()))
        turnover = 0.0
        for symbol in all_symbols:
            current = current_weights.get(symbol, 0.0)
            target = weights.get(symbol, 0.0)
            turnover += abs(target - current)
        turnover /= 2.0

        # If turnover exceeds limit, scale back the changes
        if turnover > self.max_turnover:
            scaling_factor = self.max_turnover / turnover
            result = {}

            for symbol in all_symbols:
                current = current_weights.get(symbol, 0.0)
                target = weights.get(symbol, 0.0)
                change = target - current

                # Only move partially toward target
                result[symbol] = current + (change * scaling_factor)
        else:
            result = weights.copy()

        # Normalize weights to ensure they sum to 1.0
        weight_sum = sum(result.values())
        if abs(weight_sum - 1.0) > 1e-10:
            for symbol in result:
                result[symbol] /= weight_sum

        return result

    def __str__(self) -> str:
        """String representation of the turnover constraint."""
        return f"TurnoverConstraint(max_turnover={self.max_turnover})"