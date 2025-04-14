"""
Portfolio construction and optimization module.

This module takes alpha signals from prediction models and transforms them into
actionable portfolio allocations and trades. The key components are:

1. Portfolio representation - the current and target portfolio states
2. Portfolio optimization - converting alphas to optimal portfolio weights
3. Trade generation - determining what trades to execute to reach the target portfolio
"""

from .portfolio import Portfolio
from .optimizer import PortfolioOptimizer
from .constraints import PortfolioConstraint, LeverageConstraint, SectorConstraint
from .objectives import PortfolioObjective, MaximizeReturn, MinimizeRisk, MaximizeSharpe
from .trade_generator import TradeGenerator

__all__ = [
    'Portfolio',
    'PortfolioOptimizer',
    'PortfolioConstraint',
    'LeverageConstraint',
    'SectorConstraint',
    'PortfolioObjective',
    'MaximizeReturn',
    'MinimizeRisk',
    'MaximizeSharpe',
    'TradeGenerator',
]