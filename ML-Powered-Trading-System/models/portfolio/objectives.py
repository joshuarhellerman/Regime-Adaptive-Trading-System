"""
Portfolio objectives module.

This module defines objectives that can be used during portfolio optimization,
such as maximizing returns, minimizing risk, or maximizing Sharpe ratio.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd


class PortfolioObjective(ABC):
    """Base class for all portfolio objectives."""

    @abstractmethod
    def evaluate(self, weights: Dict[str, float], metadata: Dict) -> float:
        """
        Evaluate the objective function for the given weights.

        Args:
            weights: Dictionary mapping symbols to portfolio weights
            metadata: Additional data needed for objective evaluation

        Returns:
            A value representing the objective function (higher is better)
        """
        pass

    def __str__(self) -> str:
        """String representation of the objective."""
        return self.__class__.__name__


class MaximizeReturn(PortfolioObjective):
    """
    Objective to maximize expected portfolio return.

    This objective uses expected returns for each asset to calculate
    the portfolio's expected return, which it aims to maximize.
    """

    def __init__(self, expected_returns: Optional[Dict[str, float]] = None):
        """
        Initialize a return maximization objective.

        Args:
            expected_returns: Dictionary mapping symbols to expected returns
        """
        self.expected_returns = expected_returns or {}

    def evaluate(self, weights: Dict[str, float], metadata: Dict) -> float:
        """
        Calculate the expected portfolio return.

        Args:
            weights: Dictionary mapping symbols to portfolio weights
            metadata: Dictionary that should include 'expected_returns' if not provided at init

        Returns:
            Expected portfolio return (higher is better)
        """
        expected_returns = metadata.get('expected_returns', self.expected_returns)
        if not expected_returns:
            raise ValueError("Expected returns must be provided either at init or in metadata")

        # Calculate weighted sum of expected returns
        portfolio_return = 0.0
        for symbol, weight in weights.items():
            if symbol in expected_returns:
                portfolio_return += weight * expected_returns[symbol]
            elif symbol != 'CASH' and weight != 0:
                # For non-cash assets without expected returns, warn or use a default
                print(f"Warning: No expected return for {symbol} with weight {weight}")

        return portfolio_return

    def __str__(self) -> str:
        """String representation of the return objective."""
        return "MaximizeReturn"


class MinimizeRisk(PortfolioObjective):
    """
    Objective to minimize portfolio risk (volatility).

    This objective uses a covariance matrix to calculate the portfolio's
    expected volatility, which it aims to minimize.
    """

    def __init__(self, covariance_matrix: Optional[pd.DataFrame] = None):
        """
        Initialize a risk minimization objective.

        Args:
            covariance_matrix: DataFrame with covariance matrix of returns
        """
        self.covariance_matrix = covariance_matrix

    def evaluate(self, weights: Dict[str, float], metadata: Dict) -> float:
        """
        Calculate the negative of portfolio variance (since we maximize objectives).

        Args:
            weights: Dictionary mapping symbols to portfolio weights
            metadata: Dictionary that should include 'covariance_matrix' if not provided at init

        Returns:
            Negative portfolio variance (higher is better, meaning lower risk)
        """
        cov_matrix = metadata.get('covariance_matrix', self.covariance_matrix)
        if cov_matrix is None:
            raise ValueError("Covariance matrix must be provided either at init or in metadata")

        # Convert weights to a vector ordered by the covariance matrix
        symbols = [s for s in cov_matrix.index if s in weights and s != 'CASH']
        weight_vector = np.array([weights.get(s, 0.0) for s in symbols])

        # Extract the relevant part of the covariance matrix
        sub_cov_matrix = cov_matrix.loc[symbols, symbols].values

        # Calculate portfolio variance
        portfolio_variance = weight_vector.T @ sub_cov_matrix @ weight_vector

        # Return negative variance (higher is better for optimization)
        return -portfolio_variance

    def __str__(self) -> str:
        """String representation of the risk objective."""
        return "MinimizeRisk"


class MaximizeSharpe(PortfolioObjective):
    """
    Objective to maximize the Sharpe ratio.

    This objective aims to find the portfolio with the highest
    risk-adjusted return, as measured by the Sharpe ratio.
    """

    def __init__(self,
                 expected_returns: Optional[Dict[str, float]] = None,
                 covariance_matrix: Optional[pd.DataFrame] = None,
                 risk_free_rate: float = 0.0):
        """
        Initialize a Sharpe ratio maximization objective.

        Args:
            expected_returns: Dictionary mapping symbols to expected returns
            covariance_matrix: DataFrame with covariance matrix of returns
            risk_free_rate: Risk-free rate used in Sharpe ratio calculation
        """
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.risk_free_rate = risk_free_rate

    def evaluate(self, weights: Dict[str, float], metadata: Dict) -> float:
        """
        Calculate the Sharpe ratio of the portfolio.

        Args:
            weights: Dictionary mapping symbols to portfolio weights
            metadata: Dictionary with expected_returns and covariance_matrix

        Returns:
            Sharpe ratio (higher is better)
        """
        expected_returns = metadata.get('expected_returns', self.expected_returns)
        cov_matrix = metadata.get('covariance_matrix', self.covariance_matrix)
        risk_free_rate = metadata.get('risk_free_rate', self.risk_free_rate)

        if expected_returns is None or cov_matrix is None:
            raise ValueError("Expected returns and covariance matrix must be provided")

        # Create return maximization and risk minimization objectives
        return_obj = MaximizeReturn(expected_returns)
        risk_obj = MinimizeRisk(cov_matrix)

        # Calculate expected return and risk
        expected_return = return_obj.evaluate(weights, metadata)
        portfolio_variance = -risk_obj.evaluate(weights, metadata)  # Negate to get actual variance

        # Calculate Sharpe ratio
        if portfolio_variance <= 0:
            return float('-inf')  # or another suitable default

        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = (expected_return - risk_free_rate) / portfolio_volatility

        return sharpe_ratio

    def __str__(self) -> str:
        """String representation of the Sharpe ratio objective."""
        return f"MaximizeSharpe(risk_free_rate={self.risk_free_rate})"


class MinimizeMaximumDrawdown(PortfolioObjective):
    """
    Objective to minimize the maximum drawdown of the portfolio.

    This objective uses historical data to estimate the maximum drawdown
    of the portfolio, which it aims to minimize.
    """

    def __init__(self, historical_returns: Optional[pd.DataFrame] = None):
        """
        Initialize a drawdown minimization objective.

        Args:
            historical_returns: DataFrame with historical returns for each asset
        """
        self.historical_returns = historical_returns

    def evaluate(self, weights: Dict[str, float], metadata: Dict) -> float:
        """
        Calculate the negative of the estimated maximum drawdown.

        Args:
            weights: Dictionary mapping symbols to portfolio weights
            metadata: Dictionary that should include 'historical_returns'

        Returns:
            Negative maximum drawdown (higher is better, meaning smaller drawdown)
        """
        historical_returns = metadata.get('historical_returns', self.historical_returns)
        if historical_returns is None:
            raise ValueError("Historical returns must be provided")

        # Filter to only include assets in the portfolio
        symbols = [s for s in historical_returns.columns if s in weights and weights[s] != 0]

        if not symbols:
            return 0.0  # No assets with weight

        # Create a portfolio return series
        portfolio_returns = np.zeros(len(historical_returns))
        for symbol in symbols:
            if symbol in historical_returns.columns:
                portfolio_returns += historical_returns[symbol].values * weights[symbol]

        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()

        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)

        # Calculate drawdowns
        drawdowns = (cumulative_returns / running_max) - 1.0

        # Find maximum drawdown
        max_drawdown = np.min(drawdowns)

        # Return negative drawdown (higher is better for optimization)
        return -max_drawdown

    def __str__(self) -> str:
        """String representation of the drawdown objective."""
        return "MinimizeMaximumDrawdown"