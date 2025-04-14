"""
Allocation manager module.

This module provides capital allocation functionality for portfolio management,
handling the distribution of capital across strategies, sectors, and assets
based on performance, risk, and other constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .portfolio import Portfolio
from .constraints import PortfolioConstraint
from .objectives import PortfolioObjective

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class AllocationResult:
    """Container for the results of a capital allocation."""
    allocations: Dict[str, float]
    objective_value: float
    constraints_satisfied: bool
    strategy_allocations: Optional[Dict[str, float]] = None
    sector_allocations: Optional[Dict[str, float]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """String representation of the allocation result."""
        return (f"Capital allocation with objective value {self.objective_value:.6f}. "
                f"Constraints satisfied: {self.constraints_satisfied}")


class AllocationManager:
    """
    Capital allocation manager for portfolio construction.

    The AllocationManager is responsible for distributing capital across
    various strategies, sectors, and assets based on expected returns,
    risk considerations, and other constraints.
    """

    def __init__(self,
                 total_capital: float = 100000.0,
                 cash_buffer_pct: float = 0.05,
                 risk_budget: Optional[Dict[str, float]] = None,
                 strategy_limits: Optional[Dict[str, Tuple[float, float]]] = None,
                 sector_limits: Optional[Dict[str, Tuple[float, float]]] = None,
                 cash_key: str = 'CASH'):
        """
        Initialize an allocation manager.

        Args:
            total_capital: Total capital available for allocation
            cash_buffer_pct: Percentage of capital to keep as cash buffer
            risk_budget: Dictionary mapping strategies to risk budget proportions
            strategy_limits: Dictionary mapping strategies to (min, max) allocation percentages
            sector_limits: Dictionary mapping sectors to (min, max) allocation percentages
            cash_key: Symbol used to represent cash in the portfolio
        """
        self.total_capital = total_capital
        self.cash_buffer_pct = cash_buffer_pct
        self.risk_budget = risk_budget or {}
        self.strategy_limits = strategy_limits or {}
        self.sector_limits = sector_limits or {}
        self.cash_key = cash_key

    def allocate_capital(self,
                        strategies: Dict[str, Dict[str, float]],
                        expected_returns: Dict[str, float],
                        risk_model: Optional[Any] = None,
                        constraints: Optional[List[PortfolioConstraint]] = None,
                        objective: Optional[PortfolioObjective] = None,
                        sector_mapping: Optional[Dict[str, str]] = None) -> AllocationResult:
        """
        Allocate capital across strategies and assets.

        Args:
            strategies: Dict mapping strategy names to weight dictionaries
            expected_returns: Dict mapping symbols to expected returns
            risk_model: Optional risk model for risk-aware allocation
            constraints: List of portfolio constraints to apply
            objective: Objective function for allocation optimization
            sector_mapping: Dict mapping symbols to their sectors

        Returns:
            AllocationResult containing allocations and metadata
        """
        if not strategies:
            logger.warning("No strategies provided for allocation")
            return AllocationResult(
                allocations={self.cash_key: 1.0},
                objective_value=0.0,
                constraints_satisfied=True
            )

        # Determine allocatable capital (total minus cash buffer)
        allocatable_capital = self.total_capital * (1 - self.cash_buffer_pct)

        # First, allocate capital between strategies
        strategy_allocations = self._allocate_to_strategies(strategies, expected_returns, risk_model)

        # Then, combine strategy portfolios with their allocations
        combined_weights = {}
        for strategy, allocation in strategy_allocations.items():
            strategy_weights = strategies[strategy]
            allocation_fraction = allocation / allocatable_capital

            for symbol, weight in strategy_weights.items():
                if symbol in combined_weights:
                    combined_weights[symbol] += weight * allocation_fraction
                else:
                    combined_weights[symbol] = weight * allocation_fraction

        # Add cash allocation
        cash_allocation = self.total_capital - allocatable_capital
        if self.cash_key in combined_weights:
            combined_weights[self.cash_key] += cash_allocation / self.total_capital
        else:
            combined_weights[self.cash_key] = cash_allocation / self.total_capital

        # Ensure weights sum to 1.0
        weight_sum = sum(combined_weights.values())
        if abs(weight_sum - 1.0) > 1e-10:
            for symbol in combined_weights:
                combined_weights[symbol] /= weight_sum

        # Apply constraints if provided
        constraints_satisfied = True
        if constraints:
            metadata = {
                'expected_returns': expected_returns,
                'sector_mapping': sector_mapping
            }

            for constraint in constraints:
                if not constraint.is_satisfied(combined_weights, metadata):
                    combined_weights = constraint.apply(combined_weights, metadata)
                    # Check if constraints are still not satisfied
                    if not constraint.is_satisfied(combined_weights, metadata):
                        constraints_satisfied = False
                        logger.warning(f"Constraint {constraint} could not be satisfied")

        # Calculate objective value if objective is provided
        objective_value = 0.0
        if objective:
            metadata = {
                'expected_returns': expected_returns,
                'sector_mapping': sector_mapping
            }
            objective_value = objective.evaluate(combined_weights, metadata)

        # Calculate sector allocations if sector mapping is provided
        sector_allocations = None
        if sector_mapping:
            sector_allocations = {}
            for symbol, weight in combined_weights.items():
                if symbol == self.cash_key:
                    sector = 'Cash'
                else:
                    sector = sector_mapping.get(symbol, 'Unknown')

                if sector in sector_allocations:
                    sector_allocations[sector] += weight
                else:
                    sector_allocations[sector] = weight

        # Prepare strategy allocations as percentages of total capital
        strategy_allocation_pcts = {
            strategy: allocation / self.total_capital
            for strategy, allocation in strategy_allocations.items()
        }

        return AllocationResult(
            allocations=combined_weights,
            objective_value=objective_value,
            constraints_satisfied=constraints_satisfied,
            strategy_allocations=strategy_allocation_pcts,
            sector_allocations=sector_allocations,
            metadata={
                'total_capital': self.total_capital,
                'allocatable_capital': allocatable_capital,
                'cash_buffer': cash_allocation
            }
        )

    def _allocate_to_strategies(self,
                              strategies: Dict[str, Dict[str, float]],
                              expected_returns: Dict[str, float],
                              risk_model: Optional[Any] = None) -> Dict[str, float]:
        """
        Allocate capital between strategies.

        This method distributes capital across strategies based on either risk budgeting
        (if risk_model is provided) or expected returns. It also applies strategy-specific
        allocation limits.

        Args:
            strategies: Dict mapping strategy names to weight dictionaries
            expected_returns: Dict mapping symbols to expected returns
            risk_model: Optional risk model for risk-aware allocation

        Returns:
            Dict mapping strategy names to capital allocations
        """
        allocatable_capital = self.total_capital * (1 - self.cash_buffer_pct)

        # Calculate expected return for each strategy
        strategy_returns = {}
        for strategy_name, weights in strategies.items():
            # Calculate expected return of the strategy
            strategy_return = 0.0
            for symbol, weight in weights.items():
                if symbol in expected_returns:
                    strategy_return += weight * expected_returns[symbol]

            strategy_returns[strategy_name] = strategy_return

        # Calculate allocations based on available allocation methods
        if self.risk_budget and risk_model:
            # Use risk budgeting approach
            allocations = self._allocate_by_risk_budget(strategies, risk_model)
        elif len(strategies) == 1:
            # Only one strategy, allocate all capital to it
            strategy_name = list(strategies.keys())[0]
            allocations = {strategy_name: allocatable_capital}
        else:
            # Default to expected return weighting
            allocations = self._allocate_by_expected_return(strategies, strategy_returns)

        # Apply strategy-specific limits
        for strategy_name, (min_pct, max_pct) in self.strategy_limits.items():
            if strategy_name in allocations:
                # Convert percentages to amounts
                min_amount = min_pct * allocatable_capital
                max_amount = max_pct * allocatable_capital

                # Ensure allocation is within bounds
                allocations[strategy_name] = max(min_amount, min(max_amount, allocations[strategy_name]))

        # Normalize allocations to ensure they sum to allocatable_capital
        allocation_sum = sum(allocations.values())
        if allocation_sum > 0 and abs(allocation_sum - allocatable_capital) > 1e-10:
            for strategy in allocations:
                allocations[strategy] = (allocations[strategy] / allocation_sum) * allocatable_capital

        return allocations

    def _allocate_by_expected_return(self,
                                   strategies: Dict[str, Dict[str, float]],
                                   strategy_returns: Dict[str, float]) -> Dict[str, float]:
        """
        Allocate capital based on expected returns.

        This method allocates capital proportionally to the expected return of each strategy.
        Strategies with negative expected returns receive no allocation.

        Args:
            strategies: Dict mapping strategy names to weight dictionaries
            strategy_returns: Dict mapping strategy names to expected returns

        Returns:
            Dict mapping strategy names to capital allocations
        """
        allocatable_capital = self.total_capital * (1 - self.cash_buffer_pct)

        # Remove negative expected returns
        positive_returns = {k: v for k, v in strategy_returns.items() if v > 0}

        if not positive_returns:
            # No positive returns, allocate equally
            equal_allocation = allocatable_capital / len(strategies)
            return {strategy: equal_allocation for strategy in strategies}

        # Allocate proportionally to expected returns
        total_return = sum(positive_returns.values())

        if total_return > 0:
            return {
                strategy: (positive_returns[strategy] / total_return) * allocatable_capital
                for strategy in positive_returns
            }
        else:
            # Fallback to equal allocation
            equal_allocation = allocatable_capital / len(strategies)
            return {strategy: equal_allocation for strategy in strategies}