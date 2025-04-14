"""
Strategy Context Module

This module implements a context manager for trading strategies, facilitating
their instantiation, configuration, and coordination.

It serves as the central component for strategy management, enabling dynamic
strategy selection, weighting, and adaptation based on market conditions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set, Any, Type, Union, Callable, Tuple
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import json
import os
from enum import Enum

from models.strategies.strategy_base import TradingStrategy, DirectionalBias, VolatilityRegime
from models.strategies.strategy_protocol import StrategyProtocol, AdvancedStrategyProtocol
from core.event_bus import EventBus
from core.performance_metrics import PerformanceMetrics

# Configure logger
logger = logging.getLogger(__name__)


class StrategyExecutionMode(Enum):
    """Enum representing the strategy execution mode"""
    SINGLE = 1       # Use a single strategy
    ENSEMBLE = 2     # Use a weighted combination of strategies
    ROTATIONAL = 3   # Switch between strategies based on conditions
    ADAPTIVE = 4     # Dynamically adapt strategy weights


class StrategyContext:
    """
    Context manager for trading strategies.

    This class manages multiple strategy instances, coordinating their execution,
    parameter management, and adaptation to market conditions.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the strategy context.

        Args:
            config: Configuration dictionary for the context
        """
        self.strategies: Dict[str, TradingStrategy] = {}
        self.active_strategies: Dict[str, float] = {}  # strategy_id: weight
        self.regime_manager = None
        self.performance_tracker = None
        self.execution_mode = StrategyExecutionMode.SINGLE
        self.metadata_service = None
        self.last_adaptation_time = 0
        self.adaptation_interval = 3600  # 1 hour in seconds
        self.current_regime = None
        self.current_instrument = None
        self.config = config or {}
        self.account_balance = self.config.get('initial_balance', 10000.0)
        self.risk_limits = {
            'max_drawdown_pct': self.config.get('max_drawdown_pct', 20),
            'max_position_pct': self.config.get('max_position_pct', 5),
            'max_open_positions': self.config.get('max_open_positions', 5),
            'max_correlation': self.config.get('max_correlation', 0.7),
        }

        # Initialize subcomponents if config provided
        if config:
            self._initialize_components()

        # Register event listeners
        self._register_event_listeners()

        logger.info(f"Initialized StrategyContext with mode {self.execution_mode}")

    def _initialize_components(self) -> None:
        """Initialize context components based on configuration."""
        # Set execution mode
        mode_str = self.config.get('execution_mode', 'SINGLE')
        self.execution_mode = StrategyExecutionMode[mode_str]

        # Initialize regime manager if specified
        regime_manager_config = self.config.get('regime_manager', None)
        if regime_manager_config:
            from models.market_regime.regime_manager import RegimeManager
            self.regime_manager = RegimeManager(regime_manager_config)

        # Initialize performance tracker if specified
        performance_config = self.config.get('performance_tracker', None)
        if performance_config:
            from models.performance.performance_tracker import PerformanceTracker
            self.performance_tracker = PerformanceTracker(performance_config)

        # Initialize metadata service if specified
        metadata_config = self.config.get('metadata_service', None)
        if metadata_config:
            from services.metadata_service import MetadataService
            self.metadata_service = MetadataService(metadata_config)

    def _register_event_listeners(self) -> None:
        """Register event listeners for the strategy context."""
        EventBus.subscribe("market.regime_change", self._handle_regime_change)
        EventBus.subscribe("strategy.signal", self._handle_strategy_signal)
        EventBus.subscribe("strategy.parameter_adaptation", self._handle_parameter_adaptation)
        EventBus.subscribe("account.balance_update", self._handle_balance_update)

    def _handle_regime_change(self, event_data: Dict[str, Any]) -> None:
        """
        Handle market regime change events.

        Args:
            event_data: Event data containing regime information
        """
        if 'regime' in event_data:
            logger.info(f"Market regime changed to {event_data.get('regime_id')}")
            self.current_regime = event_data['regime']

            # Adapt strategies to new regime
            self._adapt_strategies_to_regime(event_data['regime'])

            # Potentially adjust strategy weights
            if self.execution_mode in [StrategyExecutionMode.ENSEMBLE,
                                      StrategyExecutionMode.ADAPTIVE]:
                self._update_strategy_weights()

    def _handle_strategy_signal(self, event_data: Dict[str, Any]) -> None:
        """
        Handle strategy signal events.

        Args:
            event_data: Event data containing signal information
        """
        # Record signal in metadata service if available
        if self.metadata_service:
            self.metadata_service.record_signal(event_data)

        # Log signal for monitoring
        logger.info(
            f"Signal from {event_data.get('strategy_id')}: "
            f"{event_data.get('signal')} on {event_data.get('instrument')} "
            f"with confidence {event_data.get('confidence', 0):.2f}"
        )

    def _handle_parameter_adaptation(self, event_data: Dict[str, Any]) -> None:
        """
        Handle strategy parameter adaptation events.

        Args:
            event_data: Event data containing parameter changes
        """
        # Record adaptation in metadata service if available
        if self.metadata_service:
            self.metadata_service.record_adaptation(event_data)

        # Log adaptation for monitoring
        logger.info(
            f"Parameter adaptation for {event_data.get('strategy_id')}: "
            f"{event_data.get('parameter_changes')}"
        )

    def _handle_balance_update(self, event_data: Dict[str, Any]) -> None:
        """
        Handle account balance update events.

        Args:
            event_data: Event data containing balance information
        """
        if 'balance' in event_data:
            self.account_balance = event_data['balance']

            # Update balance in all strategies
            for strategy in self.strategies.values():
                strategy.set_account_balance(self.account_balance)

    def register_strategy(self, strategy: TradingStrategy, weight: float = 1.0) -> None:
        """
        Register a strategy with the context.

        Args:
            strategy: Strategy instance to register
            weight: Initial weight for the strategy
        """
        # Add to strategy dictionary
        self.strategies[strategy.id] = strategy

        # Set initial weight if in ensemble mode
        if self.execution_mode in [StrategyExecutionMode.ENSEMBLE,
                                  StrategyExecutionMode.ADAPTIVE]:
            self.active_strategies[strategy.id] = weight
        elif self.execution_mode == StrategyExecutionMode.SINGLE and not self.active_strategies:
            # If single mode and no active strategy yet, set this as active
            self.active_strategies[strategy.id] = 1.0

        # Update strategy with current account balance
        strategy.set_account_balance(self.account_balance)

        # If we have current regime data, send it to the strategy
        if self.current_regime:
            strategy.adapt_to_regime(self.current_regime)

        # If we have current instrument, set it
        if self.current_instrument:
            strategy.currency_pair = self.current_instrument

        logger.info(f"Registered strategy {strategy.name} with ID {strategy.id}")

    def unregister_strategy(self, strategy_id: str) -> None:
        """
        Unregister a strategy from the context.

        Args:
            strategy_id: ID of strategy to unregister
        """
        if strategy_id in self.strategies:
            # Remove from strategies dict
            del self.strategies[strategy_id]

            # Remove from active strategies if present
            if strategy_id in self.active_strategies:
                del self.active_strategies[strategy_id]

                # Rebalance weights if in ensemble mode
                if self.execution_mode in [StrategyExecutionMode.ENSEMBLE,
                                          StrategyExecutionMode.ADAPTIVE]:
                    self._normalize_weights()

            logger.info(f"Unregistered strategy with ID {strategy_id}")
        else:
            logger.warning(f"Attempted to unregister unknown strategy with ID {strategy_id}")

    def set_active_strategy(self, strategy_id: str) -> None:
        """
        Set a single strategy as active.

        Args:
            strategy_id: ID of strategy to set as active

        Raises:
            ValueError: If strategy_id is not registered
        """
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy with ID {strategy_id} not registered")

        # Clear active strategies and set the new one
        self.active_strategies = {strategy_id: 1.0}

        # Set execution mode to SINGLE
        self.execution_mode = StrategyExecutionMode.SINGLE

        logger.info(f"Set {strategy_id} as the active strategy")

    def set_strategy_weights(self, weights: Dict[str, float]) -> None:
        """
        Set weights for multiple strategies.

        Args:
            weights: Dictionary mapping strategy IDs to weights

        Raises:
            ValueError: If any strategy ID is not registered
        """
        # Validate all strategy IDs
        unknown_strategies = set(weights.keys()) - set(self.strategies.keys())
        if unknown_strategies:
            raise ValueError(f"Unknown strategy IDs: {unknown_strategies}")

        # Set weights
        self.active_strategies = weights.copy()

        # Normalize weights
        self._normalize_weights()

        # Set execution mode to ENSEMBLE if more than one strategy
        if len(weights) > 1:
            self.execution_mode = StrategyExecutionMode.ENSEMBLE
        else:
            self.execution_mode = StrategyExecutionMode.SINGLE

        logger.info(f"Set strategy weights: {self.active_strategies}")

    def _normalize_weights(self) -> None:
        """Normalize strategy weights to sum to 1.0."""
        if not self.active_strategies:
            return

        total_weight = sum(self.active_strategies.values())

        if total_weight > 0:
            for strategy_id in self.active_strategies:
                self.active_strategies[strategy_id] /= total_weight

    def set_execution_mode(self, mode: StrategyExecutionMode) -> None:
        """
        Set the execution mode for the context.

        Args:
            mode: Execution mode to set
        """
        self.execution_mode = mode

        # If switching to SINGLE mode and we have multiple active strategies,
        # select the one with highest weight
        if mode == StrategyExecutionMode.SINGLE and len(self.active_strategies) > 1:
            best_strategy_id = max(self.active_strategies.items(), key=lambda x: x[1])[0]
            self.active_strategies = {best_strategy_id: 1.0}

        logger.info(f"Set execution mode to {mode}")

    def set_current_instrument(self, instrument: str) -> None:
        """
        Set the current trading instrument.

        Args:
            instrument: Instrument symbol (e.g., 'EURUSD')
        """
        self.current_instrument = instrument

        # Update instrument in all strategies
        for strategy in self.strategies.values():
            strategy.currency_pair = instrument

        logger.info(f"Set current instrument to {instrument}")

    def generate_signal(self, data: pd.DataFrame) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Generate a trading signal using active strategies.

        Args:
            data: DataFrame with market data

        Returns:
            Tuple of (signal, metadata)
        """
        start_time = time.time()

        # Handle different execution modes
        if self.execution_mode == StrategyExecutionMode.SINGLE:
            return self._generate_single_signal(data)
        elif self.execution_mode == StrategyExecutionMode.ENSEMBLE:
            return self._generate_ensemble_signal(data)
        elif self.execution_mode == StrategyExecutionMode.ROTATIONAL:
            return self._generate_rotational_signal(data)
        elif self.execution_mode == StrategyExecutionMode.ADAPTIVE:
            # Check if we need to update weights
            if time.time() - self.last_adaptation_time > self.adaptation_interval:
                self._update_strategy_weights()
                self.last_adaptation_time = time.time()
            return self._generate_ensemble_signal(data)

        # Default to no signal if no strategies or unknown mode
        logger.warning(f"No signal generated (mode: {self.execution_mode}, "
                     f"active strategies: {len(self.active_strategies)})")
        return None, {}

    def _generate_single_signal(self, data: pd.DataFrame) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Generate a signal using the single active strategy.

        Args:
            data: DataFrame with market data

        Returns:
            Tuple of (signal, metadata)
        """
        if not self.active_strategies:
            return None, {}

        # Get the active strategy ID
        strategy_id = next(iter(self.active_strategies))
        strategy = self.strategies[strategy_id]

        # Generate signal
        try:
            signal = strategy.generate_signal(data)

            # If signal generated, get metadata
            if signal:
                metadata = strategy.get_signal_metadata(data, signal)
                metadata['strategy_id'] = strategy_id
                metadata['strategy_name'] = strategy.name

                # Log signal
                logger.info(f"Signal {signal} generated by {strategy.name}")

                return signal, metadata
        except Exception as e:
            logger.error(f"Error generating signal with strategy {strategy_id}: {str(e)}")

        return None, {}

    def _generate_ensemble_signal(self, data: pd.DataFrame) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Generate a signal using a weighted ensemble of active strategies.

        Args:
            data: DataFrame with market data

        Returns:
            Tuple of (signal, metadata)
        """
        if not self.active_strategies:
            return None, {}

        # Collect signals from all active strategies
        signals = {}
        signal_weights = {'long': 0.0, 'short': 0.0, None: 0.0}
        signal_metadata = {}

        for strategy_id, weight in self.active_strategies.items():
            strategy = self.strategies[strategy_id]

            try:
                # Generate signal
                signal = strategy.generate_signal(data)

                # Store signal and update weights
                signals[strategy_id] = signal
                signal_weights[signal] = signal_weights.get(signal, 0.0) + weight

                # If signal is not None, get metadata
                if signal:
                    metadata = strategy.get_signal_metadata(data, signal)
                    metadata['strategy_id'] = strategy_id
                    metadata['strategy_name'] = strategy.name
                    metadata['weight'] = weight
                    signal_metadata[strategy_id] = metadata
            except Exception as e:
                logger.error(f"Error generating signal with strategy {strategy_id}: {str(e)}")

        # Determine final signal (highest weight wins)
        long_weight = signal_weights.get('long', 0.0)
        short_weight = signal_weights.get('short', 0.0)
        no_signal_weight = signal_weights.get(None, 0.0)

        # If weights are tied or no signal has highest weight, return None
        if no_signal_weight >= max(long_weight, short_weight) or long_weight == short_weight:
            return None, {}

        # Determine final signal
        final_signal = 'long' if long_weight > short_weight else 'short'

        # Create merged metadata
        merged_metadata = {
            'ensemble_weights': self.active_strategies,
            'signal_weights': {s: w for s, w in signal_weights.items() if s},
            'contributing_strategies': [sid for sid, s in signals.items() if s == final_signal],
            'confidence': self._calculate_ensemble_confidence(final_signal, signals, signal_metadata)
        }

        logger.info(f"Ensemble signal {final_signal} generated with weight {signal_weights[final_signal]:.2f}")

        return final_signal, merged_metadata

    def _generate_rotational_signal(self, data: pd.DataFrame) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Generate a signal by selecting the best strategy for current conditions.

        Args:
            data: DataFrame with market data

        Returns:
            Tuple of (signal, metadata)
        """
        # If we have a regime manager and current regime, use it to find best strategy
        if self.regime_manager and self.current_regime:
            # Get strategy fitness scores for current regime
            fitness_scores = {}

            for strategy_id, strategy in self.strategies.items():
                fitness = strategy.cluster_fit(self.current_regime)
                fitness_scores[strategy_id] = fitness

            # Select strategy with highest fitness
            if fitness_scores:
                best_strategy_id = max(fitness_scores.items(), key=lambda x: x[1])[0]

                # Set as the only active strategy
                self.active_strategies = {best_strategy_id: 1.0}

                # Log rotation
                logger.info(f"Rotated to strategy {best_strategy_id} with fitness {fitness_scores[best_strategy_id]:.2f}")

                # Generate signal using this strategy
                return self._generate_single_signal(data)

        # If no regime manager or no fitness scores, fall back to ensemble
        logger.warning("No regime data for rotation, falling back to ensemble")
        return self._generate_ensemble_signal(data)

    def _calculate_ensemble_confidence(self, signal: str,
                                      signals: Dict[str, Optional[str]],
                                      metadata: Dict[str, Dict[str, Any]]) -> float:
        """
        Calculate confidence level for ensemble signal.

        Args:
            signal: Final signal ('long' or 'short')
            signals: Signals from each strategy
            metadata: Signal metadata from each strategy

        Returns:
            float: Confidence level between 0.0 and 1.0
        """
        # Get strategies that contributed to this signal
        contributing = [sid for sid, s in signals.items() if s == signal]

        if not contributing:
            return 0.0

        # Calculate weighted average of confidence levels
        total_weight = 0.0
        weighted_confidence = 0.0

        for strategy_id in contributing:
            if strategy_id in self.active_strategies and strategy_id in metadata:
                weight = self.active_strategies[strategy_id]
                confidence = metadata[strategy_id].get('confidence', 0.5)

                weighted_confidence += weight * confidence
                total_weight += weight

        if total_weight > 0:
            return weighted_confidence / total_weight
        else:
            return 0.5  # Default confidence

    def risk_parameters(self, data: pd.DataFrame, signal: str, entry_price: float) -> Dict[str, float]:
        """
        Calculate risk parameters for a trade.

        Args:
            data: DataFrame with market data
            signal: Trading signal ('long' or 'short')
            entry_price: Entry price for the trade

        Returns:
            Dict with risk parameters
        """
        # Calculate risk parameters from each active strategy
        risk_params_list = []

        for strategy_id, weight in self.active_strategies.items():
            strategy = self.strategies[strategy_id]

            try:
                risk_params = strategy.risk_parameters(data, entry_price)
                risk_params['weight'] = weight
                risk_params_list.append(risk_params)
            except Exception as e:
                logger.error(f"Error calculating risk parameters with strategy {strategy_id}: {str(e)}")

        if not risk_params_list:
            # Default conservative risk parameters
            return {
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04,
                'position_size': 0.01 * self.account_balance
            }

        # Calculate weighted average for each parameter
        merged_params = {}
        keys_to_average = ['stop_loss_pct', 'take_profit_pct', 'position_size']
        total_weight = sum(p.get('weight', 1.0) for p in risk_params_list)

        for key in keys_to_average:
            weighted_sum = sum(p.get(key, 0) * p.get('weight', 1.0)
                              for p in risk_params_list)
            merged_params[key] = weighted_sum / total_weight if total_weight > 0 else 0

        # Apply global risk limits
        merged_params['position_size'] = min(
            merged_params['position_size'],
            self.account_balance * self.risk_limits['max_position_pct'] / 100
        )

        # Calculate stop and target prices
        merged_params['stop_price'] = entry_price * (1 - merged_params['stop_loss_pct']) \
                                    if signal == 'long' else \
                                    entry_price * (1 + merged_params['stop_loss_pct'])

        merged_params['take_profit_price'] = entry_price * (1 + merged_params['take_profit_pct']) \
                                          if signal == 'long' else \
                                          entry_price * (1 - merged_params['take_profit_pct'])

        logger.info(f"Calculated risk parameters: SL={merged_params['stop_loss_pct']:.2%}, "
                   f"TP={merged_params['take_profit_pct']:.2%}, Size={merged_params['position_size']:.2f}")

        return merged_params

    def exit_signal(self, data: pd.DataFrame, position: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Check if an exit signal is generated for the position.

        Args:
            data: DataFrame with market data
            position: Current position information

        Returns:
            Tuple of (should_exit, reason)
        """
        # If using a single strategy, just use its exit signal
        if self.execution_mode == StrategyExecutionMode.SINGLE:
            strategy_id = next(iter(self.active_strategies))
            strategy = self.strategies[strategy_id]

            try:
                should_exit = strategy.exit_signal(data, position)
                reason = strategy.get_exit_reason(data, position) if should_exit else None

                if should_exit:
                    logger.info(f"Exit signal from {strategy.name}: {reason}")

                return should_exit, reason
            except Exception as e:
                logger.error(f"Error generating exit signal with strategy {strategy_id}: {str(e)}")
                return False, None

        # For ensemble modes, use weighted voting
        exit_votes = 0
        total_votes = 0
        exit_reasons = []

        for strategy_id, weight in self.active_strategies.items():
            strategy = self.strategies[strategy_id]

            try:
                should_exit = strategy.exit_signal(data, position)

                if should_exit:
                    exit_votes += weight
                    reason = strategy.get_exit_reason(data, position)
                    if reason:
                        exit_reasons.append((reason, weight))

                total_votes += weight
            except Exception as e:
                logger.error(f"Error generating exit signal with strategy {strategy_id}: {str(e)}")

        # Determine if we should exit based on weighted voting
        if total_votes > 0 and exit_votes / total_votes > 0.5:
            # Find the highest weighted reason
            if exit_reasons:
                exit_reasons.sort(key=lambda x: x[1], reverse=True)
                primary_reason = exit_reasons[0][0]
            else:
                primary_reason = "Majority vote exit"

            logger.info(f"Ensemble exit signal with {exit_votes/total_votes:.1%} vote: {primary_reason}")
            return True, primary_reason

        return False, None