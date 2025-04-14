"""
Portfolio optimization module.

This module contains the PortfolioOptimizer class, which is responsible for
converting alpha signals into optimal portfolio weights based on various
constraints and objectives.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import scipy.optimize as sco
from datetime import datetime
import logging

from .constraints import PortfolioConstraint, LeverageConstraint, PositionSizeConstraint
from .objectives import PortfolioObjective, MaximizeReturn, MinimizeRisk, MaximizeSharpe

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Container for the results of a portfolio optimization."""
    weights: Dict[str, float]
    objective_value: float
    success: bool
    message: str
    optimization_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """String representation of the optimization result."""
        return (f"Optimization{'succeeded' if self.success else 'failed'} "
                f"with objective value {self.objective_value:.6f}. "
                f"Message: {self.message}")


class PortfolioOptimizer:
    """
    Portfolio optimizer that converts alpha signals to optimal portfolio weights.

    The optimizer takes alpha signals (expected returns) and other market data,
    and generates optimal portfolio weights based on objectives and constraints.
    """

    def __init__(self,
                 objective: Optional[PortfolioObjective] = None,
                 constraints: Optional[List[PortfolioConstraint]] = None,
                 default_bounds: Tuple[float, float] = (-1.0, 1.0),
                 cash_key: str = 'CASH'):
        """
        Initialize a portfolio optimizer.

        Args:
            objective: The objective function to optimize
            constraints: List of constraints to apply during optimization
            default_bounds: Default (lower, upper) bounds for individual weights
            cash_key: Symbol used to represent cash in the portfolio
        """
        self.objective = objective
        self.constraints = constraints or []
        self.default_bounds = default_bounds
        self.cash_key = cash_key

        # Add default constraints if none provided
        if not self.constraints:
            self.constraints = [
                LeverageConstraint(max_leverage=1.0, exclude_cash=True),
                PositionSizeConstraint(max_size=0.3, min_size=0.01, exclude_cash=True)
            ]

        # Default to return maximization if no objective provided
        if self.objective is None:
            self.objective = MaximizeReturn()

    def optimize(self,
                alpha_signals: Dict[str, float],
                initial_weights: Optional[Dict[str, float]] = None,
                covariance_matrix: Optional[pd.DataFrame] = None,
                bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                metadata: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """
        Optimize portfolio weights based on alpha signals and constraints.

        Args:
            alpha_signals: Dictionary mapping symbols to alpha signals (expected returns)
            initial_weights: Initial portfolio weights to start optimization from
            covariance_matrix: Covariance matrix of asset returns
            bounds: Dictionary mapping symbols to (lower, upper) bounds
            metadata: Additional data needed for optimization

        Returns:
            OptimizationResult containing the optimal weights and metadata
        """
        start_time = datetime.now()

        # Set up metadata for objective and constraint evaluation
        meta = metadata or {}
        meta['expected_returns'] = alpha_signals
        if covariance_matrix is not None:
            meta['covariance_matrix'] = covariance_matrix

        # For simplicity, we'll use alpha signals as expected returns directly
        if 'expected_returns' not in meta:
            meta['expected_returns'] = alpha_signals

        # Set up the symbols list
        all_symbols = list(alpha_signals.keys())
        if self.cash_key not in all_symbols:
            all_symbols.append(self.cash_key)

        # Set up initial weights
        if initial_weights is None:
            initial_weights = {symbol: 1.0 / len(all_symbols) for symbol in all_symbols}
        else:
            # Ensure all symbols are in initial weights
            for symbol in all_symbols:
                if symbol not in initial_weights:
                    initial_weights[symbol] = 0.0

        # Set up bounds for each symbol
        weight_bounds = []
        for symbol in all_symbols:
            if bounds and symbol in bounds:
                weight_bounds.append(bounds[symbol])
            else:
                weight_bounds.append(self.default_bounds)

        # Convert dictionary weights to array for scipy optimizer
        initial_weights_array = np.array([initial_weights.get(symbol, 0.0) for symbol in all_symbols])

        # Define the objective function for scipy optimizer
        def objective_function(weights_array):
            weights_dict = {symbol: weight for symbol, weight in zip(all_symbols, weights_array)}
            # We negate because scipy minimizes by default, but we want to maximize
            return -self.objective.evaluate(weights_dict, meta)

        # Define the constraint that weights sum to 1
        def constraint_sum_to_one(weights_array):
            return np.sum(weights_array) - 1.0

        # Set up the optimization constraints for scipy
        constraints = [
            {'type': 'eq', 'fun': constraint_sum_to_one}
        ]

        # Attempt the optimization
        try:
            result = sco.minimize(
                objective_function,
                initial_weights_array,
                method='SLSQP',
                bounds=weight_bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )

            # Convert the result back to a dictionary
            optimized_weights = {symbol: weight for symbol, weight in zip(all_symbols, result.x)}

            # Apply portfolio constraints
            for constraint in self.constraints:
                optimized_weights = constraint.apply(optimized_weights, meta)

            # Calculate final objective value after applying constraints
            final_objective = self.objective.evaluate(optimized_weights, meta)

            optimization_time = (datetime.now() - start_time).total_seconds()

            return OptimizationResult(
                weights=optimized_weights,
                objective_value=final_objective,
                success=result.success,
                message=result.message,
                optimization_time=optimization_time,
                metadata={
                    'iterations': result.nit,
                    'function_calls': result.nfev,
                    'initial_weights': initial_weights,
                    'constraints_applied': [str(c) for c in self.constraints],
                    'objective': str(self.objective)
                }
            )

        except Exception as e:
            logger.exception("Portfolio optimization failed")

            # Return original weights if optimization fails
            return OptimizationResult(
                weights=initial_weights,
                objective_value=self.objective.evaluate(initial_weights, meta),
                success=False,
                message=str(e),
                optimization_time=(datetime.now() - start_time).total_seconds(),
                metadata={'error': str(e)}
            )

    def alpha_to_weights(self,
                        alpha_results: Dict[str, float],
                        current_portfolio: Optional[Dict[str, float]] = None,
                        risk_model: Optional[pd.DataFrame] = None,
                        target_leverage: float = 1.0) -> Dict[str, float]:
        """
        Convenient wrapper to convert alpha signals to portfolio weights.

        Args:
            alpha_results: Dictionary mapping symbols to alpha signals
            current_portfolio: Current portfolio weights
            risk_model: Covariance matrix for risk management
            target_leverage: Target portfolio leverage

        Returns:
            Dictionary mapping symbols to target weights
        """
        # Filter out non-positive alpha signals to avoid shorting
        positive_alphas = {symbol: signal for symbol, signal in alpha_results.items() if signal > 0}

        # Set basic metadata
        metadata = {
            'target_leverage': target_leverage,
            'current_weights': current_portfolio
        }

        # If we have a risk model, use Sharpe maximization
        if risk_model is not None:
            objective = MaximizeSharpe()
            metadata['covariance_matrix'] = risk_model
        else:
            # Otherwise just maximize return
            objective = MaximizeReturn()

        # Set up constraints based on target leverage
        constraints = [
            LeverageConstraint(max_leverage=target_leverage, exclude_cash=True),
            PositionSizeConstraint(max_size=0.3, min_size=0.01, exclude_cash=True)
        ]

        # If we have current weights, add turnover constraint
        if current_portfolio:
            from .constraints import TurnoverConstraint
            constraints.append(TurnoverConstraint(max_turnover=0.2, current_weights=current_portfolio))

        # Create a temporary optimizer with these settings
        optimizer = PortfolioOptimizer(
            objective=objective,
            constraints=constraints
        )

        # Run the optimization
        result = optimizer.optimize(
            alpha_signals=positive_alphas,
            initial_weights=current_portfolio,
            covariance_matrix=risk_model,
            metadata=metadata
        )

        if not result.success:
            logger.warning(f"Portfolio optimization failed: {result.message}")
            if current_portfolio:
                return current_portfolio
            else:
                # Equal weight fallback if no current portfolio
                equal_weight = 1.0 / (len(positive_alphas) + 1)  # +1 for cash
                weights = {symbol: equal_weight for symbol in positive_alphas}
                weights[self.cash_key] = equal_weight
                return weights

        return result.weights

    def optimize_with_different_objectives(self,
                                         alpha_signals: Dict[str, float],
                                         risk_model: pd.DataFrame,
                                         initial_weights: Optional[Dict[str, float]] = None) -> Dict[str, OptimizationResult]:
        """
        Run optimization with different objectives for comparison.

        Args:
            alpha_signals: Dictionary mapping symbols to alpha signals
            risk_model: Covariance matrix for risk calculations
            initial_weights: Initial portfolio weights

        Returns:
            Dictionary mapping objective names to optimization results
        """
        # Set up metadata
        metadata = {
            'expected_returns': alpha_signals,
            'covariance_matrix': risk_model,
            'risk_free_rate': 0.0
        }

        # Create different objectives
        objectives = {
            'max_return': MaximizeReturn(alpha_signals),
            'min_risk': MinimizeRisk(risk_model),
            'max_sharpe': MaximizeSharpe(alpha_signals, risk_model)
        }

        # Run optimization for each objective
        results = {}
        for name, objective in objectives.items():
            temp_optimizer = PortfolioOptimizer(
                objective=objective,
                constraints=self.constraints
            )

            result = temp_optimizer.optimize(
                alpha_signals=alpha_signals,
                initial_weights=initial_weights,
                covariance_matrix=risk_model,
                metadata=metadata
            )

            results[name] = result

        return results