"""
Strategy Protocol Module

This module defines the protocol interface that all trading strategies must implement.
It uses Python's typing and Protocol system to enforce a consistent interface
across all strategy implementations.
"""

from typing import Dict, List, Optional, Set, Any, Protocol, Type, Union, Callable, TypeVar
from enum import Enum
import pandas as pd


class DirectionalBias(Enum):
    """Enum representing market directional bias"""
    NEUTRAL = 1
    UPWARD = 2
    DOWNWARD = 3


class VolatilityRegime(Enum):
    """Enum representing market volatility regime"""
    LOW = 1
    NORMAL = 2
    HIGH = 3


class MarketRegime(Protocol):
    """Protocol defining the expected structure of market regime data"""
    id: str
    directional_bias: DirectionalBias
    volatility_regime: VolatilityRegime
    peak_hour: Optional[int]
    metrics: Dict[str, float]


class SignalMetadata(Protocol):
    """Protocol defining the expected structure of signal metadata"""
    confidence: float
    indicators: Dict[str, float]
    timestamp: float
    context: Optional[Dict[str, Any]]


class TradeResult(Protocol):
    """Protocol defining the expected structure of trade result data"""
    id: str
    pnl: float
    pnl_pct: float
    duration: float
    entry_price: float
    exit_price: float
    direction: str
    timestamp: float
    regime_id: Optional[str]
    reason: Optional[str]


# Type variable for strategy class types
T = TypeVar('T', bound='StrategyProtocol')


class StrategyProtocol(Protocol):
    """
    Protocol defining the required interface for all trading strategies.

    This protocol ensures that all strategies implement the necessary
    methods with compatible signatures.
    """

    id: str
    name: str
    parameters: Dict[str, Any]

    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate that the input data has the required columns.

        Args:
            data: DataFrame with market data

        Raises:
            ValueError: If data doesn't contain required columns
        """
        ...

    def generate_signal(self, data: pd.DataFrame) -> Optional[str]:
        """
        Generate a trading signal.

        Args:
            data: DataFrame with market data

        Returns:
            str: 'long', 'short', or None for no signal
        """
        ...

    def get_signal_metadata(self, data: pd.DataFrame, signal: str) -> Dict[str, Any]:
        """
        Get metadata about the generated signal.

        Args:
            data: DataFrame with market data
            signal: The signal ('long', 'short') that was generated

        Returns:
            Dict with signal metadata including confidence and context
        """
        ...

    def risk_parameters(self, data: pd.DataFrame, entry_price: float) -> Dict[str, float]:
        """
        Calculate risk parameters.

        Args:
            data: DataFrame with market data
            entry_price: Entry price for the trade

        Returns:
            Dict with stop loss, take profit, and position size
        """
        ...

    def exit_signal(self, data: pd.DataFrame, position: Dict[str, Any]) -> bool:
        """
        Check if an exit signal is generated for the position.

        Args:
            data: DataFrame with market data
            position: Current position information

        Returns:
            bool: True if should exit, False otherwise
        """
        ...

    def get_exit_reason(self, data: pd.DataFrame, position: Dict[str, Any]) -> Optional[str]:
        """
        Get the reason for an exit signal.

        Args:
            data: DataFrame with market data
            position: Current position information

        Returns:
            str or None: Exit reason if there is one, None otherwise
        """
        ...

    def adapt_to_regime(self, regime_data: Dict[str, Any]) -> None:
        """
        Adapt strategy parameters based on market regime.

        Args:
            regime_data: Market regime information
        """
        ...

    def cluster_fit(self, cluster_metrics: Dict[str, float]) -> float:
        """
        Determine how well the strategy fits the given cluster.

        Args:
            cluster_metrics: Dict of cluster characteristics

        Returns:
            float: Fitness score between 0.0 and 1.0
        """
        ...

    def on_trade_completed(self, trade_result: Dict[str, Any]) -> None:
        """
        Callback for when a trade is completed.

        Args:
            trade_result: Dict with trade result information
        """
        ...

    def update_parameters_online(self, performance_metrics: Dict[str, float],
                               market_conditions: Dict[str, Any]) -> None:
        """
        Update strategy parameters based on recent performance and market conditions.

        Args:
            performance_metrics: Performance metrics
            market_conditions: Current market conditions
        """
        ...

    def get_required_features(self) -> Set[str]:
        """
        Return the list of features required by the strategy.

        Returns:
            Set of feature names
        """
        ...

    def set_account_balance(self, balance: float) -> None:
        """
        Set the current account balance for position sizing calculations.

        Args:
            balance: Current account balance
        """
        ...

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get current performance metrics for the strategy.

        Returns:
            Dict of performance metrics
        """
        ...

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the strategy.

        Returns:
            Dict containing the current state for serialization
        """
        ...

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the state of the strategy from external source.

        Args:
            state: Dict containing state to set
        """
        ...

    @classmethod
    def register(cls: Type[T]) -> None:
        """
        Register strategy with the strategy registry.
        """
        ...


class AdvancedStrategyProtocol(StrategyProtocol, Protocol):
    """
    Extended protocol for advanced strategies with additional capabilities.
    """

    def visualize(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate visualization data for the strategy.

        Args:
            data: DataFrame with market data

        Returns:
            Dict with visualization data
        """
        ...

    def backtest(self, data: pd.DataFrame, initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Run a backtest of the strategy on the provided data.

        Args:
            data: DataFrame with market data
            initial_capital: Initial capital for the backtest

        Returns:
            Dict with backtest results
        """
        ...

    def optimize(self, data: pd.DataFrame, param_grid: Dict[str, List[Any]],
                 metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """
        Optimize strategy parameters on the provided data.

        Args:
            data: DataFrame with market data
            param_grid: Grid of parameters to test
            metric: Metric to optimize for

        Returns:
            Dict with optimization results
        """
        ...

    def monte_carlo(self, backtest_results: Dict[str, Any], simulations: int = 1000) -> Dict[str, Any]:
        """
        Run Monte Carlo simulations on backtest results.

        Args:
            backtest_results: Results from a backtest
            simulations: Number of simulations to run

        Returns:
            Dict with simulation results
        """
        ...

    def get_correlation(self, other_strategy: 'AdvancedStrategyProtocol',
                        data: pd.DataFrame) -> float:
        """
        Calculate correlation between this strategy and another strategy.

        Args:
            other_strategy: Another strategy to compare with
            data: DataFrame with market data

        Returns:
            float: Correlation coefficient
        """
        ...