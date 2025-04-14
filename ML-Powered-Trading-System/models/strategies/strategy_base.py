"""
Strategy Base Module

This module defines the base class for all trading strategies.
It provides common functionality for signal generation, risk management,
and regime adaptation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import logging
import time
import uuid
from enum import Enum, auto

from core.event_bus import EventBus

# Configure logger
logger = logging.getLogger(__name__)


class DirectionalBias(Enum):
    """Enum representing market directional bias"""
    NEUTRAL = auto()
    UPWARD = auto()
    DOWNWARD = auto()


class VolatilityRegime(Enum):
    """Enum representing market volatility regime"""
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()


class TradingStrategy:
    """
    Base class for all trading strategies.

    This class provides common functionality for all strategy implementations
    including risk management, regime adaptation, and performance tracking.
    """

    def __init__(self, name: str, parameters: Dict[str, Any], strategy_id: str = None):
        """
        Initialize the strategy.

        Args:
            name: Strategy name
            parameters: Strategy-specific parameters
            strategy_id: Unique identifier for the strategy instance
        """
        self.name = name
        self.parameters = parameters
        self.id = strategy_id or f"{name.lower()}-{uuid.uuid4().hex[:8]}"

        # Risk management settings
        self.stop_loss_modifier = 1.0
        self.profit_target_modifier = 1.0
        self.position_size_multiplier = 1.0
        self.max_position_pct = 0.05  # 5% of equity by default
        self.account_balance = 100000  # Default value, should be updated

        # Regime adaptation
        self.trend_filter_strength = 1.0
        self.signal_threshold_modifier = 1.0
        self.regime_characteristics = None
        self.currency_pair = None

        # Signal tracking
        self.last_signal_time = 0
        self.min_time_between_signals = 3600  # 1 hour in seconds

        # Validate parameters
        self._validate_parameters()

        logger.info(f"Initialized strategy {self.name} with ID {self.id}")

    def _validate_parameters(self) -> None:
        """
        Validate strategy parameters.

        To be overridden by subclasses with strategy-specific validation.

        Raises:
            ValueError: If parameters are invalid
        """
        pass

    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate that the input data has the required columns.

        Args:
            data: DataFrame with market data

        Raises:
            ValueError: If data doesn't contain required columns
        """
        required_columns = {'open', 'high', 'low', 'close'}
        missing_columns = required_columns - set(data.columns)

        if missing_columns:
            raise ValueError(f"Data missing required columns: {missing_columns}")

    def generate_signal(self, data: pd.DataFrame) -> Optional[str]:
        """
        Generate a trading signal.

        This method should be overridden by subclasses to implement
        strategy-specific signal generation logic.

        Args:
            data: DataFrame with market data

        Returns:
            str: 'long', 'short', or None for no signal
        """
        # Update timestamp for signal tracking
        self.last_signal_time = time.time()
        return None

    def risk_parameters(self, data: pd.DataFrame, entry_price: float) -> Dict[str, float]:
        """
        Calculate risk parameters.

        This method should be overridden by subclasses to implement
        strategy-specific risk parameter calculations.

        Args:
            data: DataFrame with market data
            entry_price: Entry price for the trade

        Returns:
            Dict with stop loss, take profit, and position size
        """
        # Default implementation
        return {
            'stop_loss_pct': 0.02 * self.stop_loss_modifier,
            'take_profit_pct': 0.04 * self.profit_target_modifier,
            'position_size': self.dynamic_position_size(self.account_balance)
        }

    def exit_signal(self, data: pd.DataFrame, position: Dict[str, Any]) -> bool:
        """
        Check if an exit signal is generated for the position.

        This method should be overridden by subclasses to implement
        strategy-specific exit logic.

        Args:
            data: DataFrame with market data
            position: Current position information

        Returns:
            bool: True if should exit, False otherwise
        """
        # Base implementation just tracks timestamp
        return False

    def adapt_to_regime(self, regime_data: Dict[str, Any]) -> None:
        """
        Adapt strategy parameters based on market regime.

        This method should be overridden by subclasses to implement
        strategy-specific regime adaptation.

        Args:
            regime_data: Market regime information
        """
        # Store regime characteristics
        self.regime_characteristics = regime_data

        # If currency pair is provided, store it
        if 'currency_pair' in regime_data:
            self.currency_pair = regime_data['currency_pair']

        # Call specific optimizations
        self._optimize_for_time_of_day()
        self._optimize_for_directional_bias()
        self._optimize_for_volatility_regime()
        self._optimize_for_currency_pair()

    def _optimize_for_time_of_day(self) -> None:
        """
        Optimize strategy parameters based on time of day.

        To be overridden by subclasses with strategy-specific optimizations.
        """
        pass

    def _optimize_for_directional_bias(self) -> None:
        """
        Optimize strategy parameters based on directional bias.

        To be overridden by subclasses with strategy-specific optimizations.
        """
        pass

    def _optimize_for_volatility_regime(self) -> None:
        """
        Optimize strategy parameters based on volatility regime.

        To be overridden by subclasses with strategy-specific optimizations.
        """
        pass

    def _optimize_for_currency_pair(self) -> None:
        """
        Optimize strategy parameters based on currency pair.

        To be overridden by subclasses with strategy-specific optimizations.
        """
        pass

    def cluster_fit(self, cluster_metrics: Dict[str, float]) -> float:
        """
        Determine how well the strategy fits the given cluster.

        This method should be overridden by subclasses to implement
        strategy-specific cluster fitting logic.

        Args:
            cluster_metrics: Dict of cluster characteristics

        Returns:
            float: Fitness score between 0.0 and 1.0
        """
        # Default implementation returns neutral score
        return 0.5

    def on_trade_completed(self, trade_result: Dict[str, Any]) -> None:
        """
        Callback for when a trade is completed.

        This method should be overridden by subclasses to implement
        strategy-specific parameter adaptation based on trade results.

        Args:
            trade_result: Dict with trade result information
        """
        pass

    def update_parameters_online(self, performance_metrics: Dict[str, float],
                                 market_conditions: Dict[str, Any]) -> None:
        """
        Update strategy parameters based on recent performance and market conditions.

        This method should be overridden by subclasses to implement
        strategy-specific online learning.

        Args:
            performance_metrics: Performance metrics
            market_conditions: Current market conditions
        """
        pass

    def get_required_features(self) -> Set[str]:
        """
        Return the list of features required by the strategy.

        This method should be overridden by subclasses to specify
        the data features needed by the strategy.

        Returns:
            Set of feature names
        """
        # Base required features
        return {'open', 'high', 'low', 'close'}

    def dynamic_position_size(self, equity: float) -> float:
        """
        Calculate dynamic position size based on strategy settings.

        Args:
            equity: Current equity value

        Returns:
            float: Position size in base currency
        """
        # Base implementation uses a fixed percentage of equity
        return equity * self.max_position_pct * self.position_size_multiplier

    def log_signal(self, signal_type: str, data: pd.DataFrame, reason: str) -> None:
        """
        Log a signal for debugging and analysis.

        Args:
            signal_type: Type of signal ('long', 'short', 'exit')
            data: DataFrame with market data
            reason: Reason for the signal
        """
        price = data['close'].iloc[-1]
        timestamp = data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else None

        logger.info(f"{self.name} {signal_type.upper()} signal at price {price:.5f} "
                    f"time: {timestamp or time.time()}: {reason}")

    @classmethod
    def register(cls) -> None:
        """
        Register strategy with the strategy registry.

        This method should be implemented by subclasses to register
        themselves with the strategy registry.
        """
        pass