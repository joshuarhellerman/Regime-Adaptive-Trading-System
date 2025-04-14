"""
Momentum Strategy Module

This module implements a momentum-based strategy that adapts to different market regimes.
The strategy identifies and trades with market momentum using various indicators
such as RSI, MACD, and price rate of change.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import logging
import time
from collections import deque

from models.strategies.strategy_base import TradingStrategy, DirectionalBias, VolatilityRegime
from core.event_bus import EventBus
from core.performance_metrics import PerformanceMetrics

# Configure logger
logger = logging.getLogger(__name__)


class MomentumStrategy(TradingStrategy):
    """
    Momentum strategy that adapts to market regimes.

    This strategy identifies and trades with market momentum using RSI, MACD,
    price rate of change, and other indicators with regime-specific adaptations.
    """

    def __init__(self, name: str = None, parameters: Dict[str, Any] = None, strategy_id: str = None):
        """
        Initialize the momentum strategy.

        Args:
            name: Strategy name
            parameters: Strategy-specific parameters
            strategy_id: Unique identifier for the strategy instance
        """
        # Default parameters
        default_params = {
            'rsi_period': 14,
            'rsi_threshold': 55,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'roc_period': 10,
            'roc_threshold': 0.0,
            'volume_confirmation': True,
            'use_macd': True,
            'use_rsi': True,
            'use_roc': True,
            'min_momentum_score': 0.6,  # Minimum score to generate signal (0-1)
            'min_bars': 50  # Minimum bars needed for calculation
        }

        # Merge default parameters with provided parameters
        if parameters:
            for key, value in parameters.items():
                default_params[key] = value

        # Initialize the base class
        super().__init__(name or "Momentum", default_params, strategy_id)

        # Strategy-specific attributes
        self._state = {
            'momentum_threshold': self.parameters['min_momentum_score'],
            'rsi_threshold_upper': self.parameters['rsi_threshold'],
            'rsi_threshold_lower': 100 - self.parameters['rsi_threshold'],
            'stop_loss_atr_multiplier': 2.0,
            'take_profit_atr_multiplier': 3.0,
            'current_momentum_score': 0.5,  # Neutral momentum
            'entry_optimization': "default",  # Can be "pullback", "breakout", or "default"
            'last_signal_time': 0.0,
            'latency_metrics': deque(maxlen=100),  # Use deque with max length for better performance
            'recent_signals': deque(maxlen=20)    # Use deque with max length for better performance
        }

        # Store the original generate_signal method for restoration after regime adaptation
        self._original_generate_signal = self.generate_signal

        logger.info(
            f"Initialized MomentumStrategy with rsi_period={self.parameters['rsi_period']}, "
            f"macd_fast={self.parameters['macd_fast']}, macd_slow={self.parameters['macd_slow']}, "
            f"roc_period={self.parameters['roc_period']}"
        )

        # Register to central event bus
        self._register_with_event_bus()

    def _register_with_event_bus(self) -> None:
        """Register strategy with the event bus."""
        EventBus.subscribe(f"strategy.{self.id}.parameter_update", self._handle_parameter_update)
        EventBus.subscribe("market.regime_change", self._handle_regime_change)

    def _handle_parameter_update(self, event_data: Dict[str, Any]) -> None:
        """Handle parameter update events."""
        if 'parameters' in event_data:
            for key, value in event_data['parameters'].items():
                if key in self.parameters:
                    self.parameters[key] = value
                    logger.info(f"Updated parameter {key} to {value} for strategy {self.id}")

            # Re-validate parameters after update
            self._validate_parameters()

    def _handle_regime_change(self, event_data: Dict[str, Any]) -> None:
        """Handle market regime change events."""
        if 'regime' in event_data:
            regime_id = event_data.get('regime_id')
            logger.info(f"Market regime changed to {regime_id}, adapting parameters")
            # Trigger regime-specific optimizations
            self.adapt_to_regime(event_data['regime'])

    def _validate_parameters(self) -> None:
        """
        Validate strategy parameters.

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate period values
        for param in ['rsi_period', 'macd_fast', 'macd_slow', 'macd_signal', 'roc_period', 'min_bars']:
            if self.parameters.get(param, 0) <= 0:
                raise ValueError(f"Parameter {param} must be greater than 0")

        # Validate threshold values
        if not 0 <= self.parameters['min_momentum_score'] <= 1.0:
            raise ValueError("Minimum momentum score must be between 0 and 1")

        if self.parameters['macd_fast'] >= self.parameters['macd_slow']:
            raise ValueError("MACD fast period must be less than slow period")

        if not 50 <= self.parameters['rsi_threshold'] <= 70:
            raise ValueError("RSI threshold must be between 50 and 70")

        if not 0 <= self.parameters['rsi_oversold'] < self.parameters['rsi_overbought'] <= 100:
            raise ValueError("RSI oversold must be less than overbought, and both between 0 and 100")

    def generate_signal(self, data: pd.DataFrame) -> Optional[str]:
        """
        Generate a momentum-based signal.

        Args:
            data: DataFrame with market data

        Returns:
            str: 'long', 'short', or None for no signal
        """
        # Begin performance tracking
        start_time = time.time()

        # Call base class method to update timestamps
        super().generate_signal(data)

        # Validate data
        try:
            self.validate_data(data)
        except ValueError as e:
            logger.error(f"Data validation failed: {str(e)}")
            return None

        # Ensure we have enough data
        if len(data) < self.parameters['min_bars']:
            logger.debug(f"Insufficient data for signal generation: {len(data)} bars")
            return None

        # Calculate momentum indicators if not present in data
        indicators = {}

        # RSI calculation
        if self.parameters['use_rsi']:
            if 'rsi_14' not in data.columns:
                indicators['rsi'] = self._calculate_rsi(data['close'], self.parameters['rsi_period'])
            else:
                indicators['rsi'] = data['rsi_14']

        # MACD calculation
        if self.parameters['use_macd']:
            if 'macd' not in data.columns:
                indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = self._calculate_macd(
                    data['close'],
                    self.parameters['macd_fast'],
                    self.parameters['macd_slow'],
                    self.parameters['macd_signal']
                )
            else:
                indicators['macd'] = data['macd']
                indicators['macd_signal'] = data['macd_signal']
                indicators['macd_hist'] = data['macd_hist']

        # Rate of Change calculation
        if self.parameters['use_roc']:
            if 'roc_10' not in data.columns:
                indicators['roc'] = self._calculate_roc(data['close'], self.parameters['roc_period'])
            else:
                indicators['roc'] = data['roc_10']

        # Calculate overall momentum score
        momentum_score, direction = self._calculate_momentum_score(data, indicators)
        self._state['current_momentum_score'] = momentum_score

        # Check if momentum score exceeds threshold
        threshold = self._state['momentum_threshold'] * self.signal_threshold_modifier

        if momentum_score >= threshold:
            # Apply entry timing optimization
            if not self._is_optimal_entry_timing(data, indicators, direction):
                logger.debug(f"Skipping {direction} signal: waiting for optimal entry timing")
                self._record_latency(start_time, "entry_timing", "not_optimal")
                return None

            # Apply volume confirmation if required
            if self.parameters['volume_confirmation'] and 'volume' in data.columns:
                if not self._confirm_volume(data, direction):
                    logger.debug(f"Volume confirmation failed for {direction} signal")
                    self._record_latency(start_time, "volume_filtered")
                    return None

            # Generate signal based on momentum direction
            if direction == 'up':
                signal = 'long'
                self.log_signal('long', data, f"Bullish momentum signal: score={momentum_score:.2f}")
                self._emit_signal_event(signal, data, {
                    'momentum_score': momentum_score,
                    'direction': direction,
                    'entry_optimization': self._state['entry_optimization'],
                    'indicators': {k: float(v.iloc[-1]) if isinstance(v, pd.Series) else v
                                for k, v in indicators.items() if isinstance(v, (pd.Series, float, int))}
                })
                self._record_latency(start_time, "signal_generation", signal)
                return signal
            elif direction == 'down':
                signal = 'short'
                self.log_signal('short', data, f"Bearish momentum signal: score={momentum_score:.2f}")
                self._emit_signal_event(signal, data, {
                    'momentum_score': momentum_score,
                    'direction': direction,
                    'entry_optimization': self._state['entry_optimization'],
                    'indicators': {k: float(v.iloc[-1]) if isinstance(v, pd.Series) else v
                                for k, v in indicators.items() if isinstance(v, (pd.Series, float, int))}
                })
                self._record_latency(start_time, "signal_generation", signal)
                return signal

        self._record_latency(start_time, "no_signal")
        return None

    def _record_latency(self, start_time: float, stage: str, signal: Optional[str] = None) -> None:
        """
        Record latency for performance monitoring.

        Args:
            start_time: Start time of operation
            stage: Processing stage name
            signal: Optional signal generated
        """
        latency = time.time() - start_time
        metric = {
            'timestamp': time.time(),
            'latency': latency,
            'stage': stage,
            'signal': signal
        }
        self._state['latency_metrics'].append(metric)

        # Report latency metrics
        PerformanceMetrics.record(
            f"strategy.{self.id}.latency",
            latency,
            tags={'stage': stage, 'signal': signal or 'none'}
        )

    def _emit_signal_event(self, signal: str, data: pd.DataFrame, metadata: Dict[str, Any]) -> None:
        """
        Emit signal event to the event bus.

        Args:
            signal: Signal type ('long' or 'short')
            data: DataFrame with market data
            metadata: Additional signal metadata
        """
        # Store signal in recent signals history
        self._state['recent_signals'].append({
            'timestamp': time.time(),
            'signal': signal,
            'price': data['close'].iloc[-1],
            'metadata': metadata
        })

        # Calculate confidence based on momentum score and entry optimization
        momentum_score = metadata.get('momentum_score', 0.5)

        # Base confidence on momentum score (scale 0.5-1.0 to 0.0-1.0)
        base_confidence = (momentum_score - 0.5) * 2 if momentum_score > 0.5 else 0

        # Adjust confidence based on entry optimization
        entry_optimization = metadata.get('entry_optimization', 'default')
        if entry_optimization == 'breakout':
            confidence_multiplier = 1.2  # Highest confidence for breakout entries
        elif entry_optimization == 'pullback':
            confidence_multiplier = 1.1  # Good confidence for pullback entries
        else:
            confidence_multiplier = 1.0  # Normal confidence for default entries

        # Calculate final confidence (capped at 1.0)
        confidence = min(1.0, base_confidence * confidence_multiplier)

        EventBus.emit("strategy.signal", {
            'strategy_id': self.id,
            'strategy_type': 'momentum',
            'timestamp': time.time(),
            'signal': signal,
            'instrument': data.get('symbol', 'unknown'),
            'price': data['close'].iloc[-1],
            'confidence': confidence,
            'metadata': metadata
        })

    def risk_parameters(self, data: pd.DataFrame, entry_price: float) -> Dict[str, float]:
        """
        Calculate risk parameters for the momentum strategy.

        Args:
            data: DataFrame with market data
            entry_price: Entry price for the trade

        Returns:
            Dict with stop loss, take profit, and position size
        """
        start_time = time.time()

        # Calculate or get ATR
        if 'atr_14' not in data.columns:
            atr = self._calculate_atr(data, 14).iloc[-1]
        else:
            atr = data['atr_14'].iloc[-1]

        # Adjust risk parameters based on momentum strength
        momentum_strength = self._state['current_momentum_score']

        # Stronger momentum = wider stops and targets
        stop_loss_multiplier = self._state['stop_loss_atr_multiplier'] * self.stop_loss_modifier
        take_profit_multiplier = self._state['take_profit_atr_multiplier'] * self.profit_target_modifier

        # Adjust based on momentum strength (stronger momentum = wider targets)
        if momentum_strength > 0.8:  # Very strong momentum
            take_profit_multiplier *= 1.2
        elif momentum_strength < 0.65:  # Weaker momentum
            take_profit_multiplier *= 0.8

        # Adjust based on entry optimization
        if self._state['entry_optimization'] == "pullback":
            # Tighter stop for pullback entries
            stop_loss_multiplier *= 0.8
        elif self._state['entry_optimization'] == "breakout":
            # Wider stop for breakout entries
            stop_loss_multiplier *= 1.2

        # Calculate stop loss and take profit distances
        stop_loss_pct = (atr * stop_loss_multiplier) / entry_price
        take_profit_pct = (atr * take_profit_multiplier) / entry_price

        # Calculate position size with adjusted momentum
        position_sizing_factor = 1.0
        if momentum_strength > 0.8:
            position_sizing_factor = 1.2  # Increase size for strong momentum
        elif momentum_strength < 0.65:
            position_sizing_factor = 0.8  # Decrease size for weaker momentum

        position_size = self.dynamic_position_size(self.account_balance) * position_sizing_factor

        # Emit risk parameters event
        self._emit_risk_event(entry_price, stop_loss_pct, take_profit_pct, position_size, momentum_strength)

        # Record performance
        self._record_latency(start_time, "risk_parameters")

        return {
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'position_size': position_size,
            'stop_price': entry_price * (1 - stop_loss_pct) if entry_price > 0 else 0,
            'take_profit_price': entry_price * (1 + take_profit_pct) if entry_price > 0 else 0,
            'atr': atr,
            'momentum_strength': momentum_strength
        }

    def _emit_risk_event(self, entry_price: float, stop_loss_pct: float,
                        take_profit_pct: float, position_size: float,
                        momentum_strength: float) -> None:
        """
        Emit risk parameters event to the event bus.

        Args:
            entry_price: Entry price
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            position_size: Position size
            momentum_strength: Current momentum strength
        """
        EventBus.emit("strategy.risk_parameters", {
            'strategy_id': self.id,
            'strategy_type': 'momentum',
            'timestamp': time.time(),
            'entry_price': entry_price,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'position_size': position_size,
            'risk_reward_ratio': take_profit_pct / stop_loss_pct if stop_loss_pct > 0 else 0,
            'momentum_strength': momentum_strength,
            'entry_optimization': self._state['entry_optimization']
        })

    def exit_signal(self, data: pd.DataFrame, position: Dict[str, Any]) -> bool:
        """
        Check if an exit signal is generated for the position.

        Args:
            data: DataFrame with market data
            position: Current position information

        Returns:
            bool: True if should exit, False otherwise
        """
        start_time = time.time()

        super().exit_signal(data, position)

        # Calculate momentum indicators if not present in data
        indicators = {}

        # RSI calculation
        if self.parameters['use_rsi']:
            if 'rsi_14' not in data.columns:
                indicators['rsi'] = self._calculate_rsi(data['close'], self.parameters['rsi_period'])
            else:
                indicators['rsi'] = data['rsi_14']

        # MACD calculation
        if self.parameters['use_macd']:
            if 'macd' not in data.columns:
                indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = self._calculate_macd(
                    data['close'],
                    self.parameters['macd_fast'],
                    self.parameters['macd_slow'],
                    self.parameters['macd_signal']
                )
            else:
                indicators['macd'] = data['macd']
                indicators['macd_signal'] = data['macd_signal']
                indicators['macd_hist'] = data['macd_hist']

        # Rate of Change calculation
        if self.parameters['use_roc']:
            if 'roc_10' not in data.columns:
                indicators['roc'] = self._calculate_roc(data['close'], self.parameters['roc_period'])
            else:
                indicators['roc'] = data['roc_10']

        # Check for momentum reversal or significant weakening
        if position['direction'] == 'long':
            # Exit long position if momentum weakens or reverses

            # Check RSI if used
            if self.parameters['use_rsi'] and 'rsi' in indicators:
                # RSI crossing below threshold indicates weakening upward momentum
                if indicators['rsi'].iloc[-1] < self._state['rsi_threshold_lower']:
                    reason = f"Exit long: RSI declined below threshold ({indicators['rsi'].iloc[-1]:.1f})"
                    self._emit_exit_event(data, position, reason)
                    self.log_signal('exit', data, reason)
                    self._record_latency(start_time, "exit_signal", "rsi_exit")
                    return True

    def _calculate_momentum_score(self, data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> Tuple[float, str]:
        """
        Calculate the overall momentum score (0 to 1) and direction.

        Args:
            data: DataFrame with market data
            indicators: Dict of calculated indicators

        Returns:
            Tuple of (momentum_score, direction)
        """
        scores = []
        direction_votes = {'up': 0, 'down': 0}

        # RSI component
        if self.parameters['use_rsi'] and 'rsi' in indicators:
            rsi = indicators['rsi'].iloc[-1]
            # RSI score is between -1 and 1
            rsi_score = (rsi - 50) / 50  # Normalize to -1 to 1
            scores.append(rsi_score)

            # Vote for direction
            if rsi > self._state['rsi_threshold_upper']:
                direction_votes['up'] += 1
            elif rsi < self._state['rsi_threshold_lower']:
                direction_votes['down'] += 1

        # MACD component
        if self.parameters['use_macd'] and 'macd' in indicators and 'macd_signal' in indicators:
            macd = indicators['macd'].iloc[-1]
            macd_signal = indicators['macd_signal'].iloc[-1]
            macd_hist = indicators['macd_hist'].iloc[-1]

            # Get scaling factor based on historical MACD range
            macd_range = max(abs(indicators['macd'].max()), abs(indicators['macd'].min()))
            # Avoid division by zero
            if macd_range < 1e-10:
                macd_range = 1.0

            # MACD score is between -1 and 1
            macd_score = macd / macd_range
            macd_score = max(min(macd_score, 1.0), -1.0)  # Clamp to [-1, 1]
            scores.append(macd_score)

            # Histogram score
            hist_range = max(abs(indicators['macd_hist'].max()), abs(indicators['macd_hist'].min()))
            # Avoid division by zero
            if hist_range < 1e-10:
                hist_range = 1.0

            hist_score = macd_hist / hist_range
            hist_score = max(min(hist_score, 1.0), -1.0)  # Clamp to [-1, 1]
            scores.append(hist_score)

            # Vote for direction
            if macd > macd_signal and macd_hist > 0:
                direction_votes['up'] += 1
            elif macd < macd_signal and macd_hist < 0:
                direction_votes['down'] += 1

        # ROC component
        if self.parameters['use_roc'] and 'roc' in indicators:
            roc = indicators['roc'].iloc[-1]

            # Get scaling factor based on historical ROC range
            roc_range = max(abs(indicators['roc'].max()), abs(indicators['roc'].min()))
            # Avoid division by zero
            if roc_range < 1e-10:
                roc_range = 1.0

            # ROC score is between -1 and 1
            roc_score = roc / roc_range
            roc_score = max(min(roc_score, 1.0), -1.0)  # Clamp to [-1, 1]
            scores.append(roc_score)

            # Vote for direction
            if roc > self.parameters['roc_threshold']:
                direction_votes['up'] += 1
            elif roc < -self.parameters['roc_threshold']:
                direction_votes['down'] += 1

        # Price trend component
        if len(data) >= 20:
            # Calculate short-term trend
            short_trend = data['close'].iloc[-1] / data['close'].iloc[-5] - 1
            medium_trend = data['close'].iloc[-1] / data['close'].iloc[-20] - 1

            # Normalize trends to [-1, 1]
            trend_range = 0.05  # Assume 5% move is significant
            short_trend_score = short_trend / trend_range
            short_trend_score = max(min(short_trend_score, 1.0), -1.0)  # Clamp to [-1, 1]

            medium_trend_score = medium_trend / trend_range
            medium_trend_score = max(min(medium_trend_score, 1.0), -1.0)  # Clamp to [-1, 1]

            scores.append(short_trend_score)
            scores.append(medium_trend_score)

            # Vote for direction
            if short_trend > 0 and medium_trend > 0:
                direction_votes['up'] += 1
            elif short_trend < 0 and medium_trend < 0:
                direction_votes['down'] += 1

        # Calculate overall score as weighted average
        if not scores:
            return 0.5, 'neutral'

        overall_score = sum(scores) / len(scores)

        # Convert to 0-1 scale for threshold comparison
        momentum_score = (overall_score + 1) / 2

        # Determine direction
        if direction_votes['up'] > direction_votes['down']:
            direction = 'up'
        elif direction_votes['down'] > direction_votes['up']:
            direction = 'down'
        else:
            # If tied, use the sign of the overall score
            direction = 'up' if overall_score > 0 else 'down'

        return momentum_score, direction

    def _is_optimal_entry_timing(self, data: pd.DataFrame, indicators: Dict[str, pd.Series], direction: str) -> bool:
        """
        Determine if this is an optimal time to enter the trade.

        Args:
            data: DataFrame with market data
            indicators: Dict of calculated indicators
            direction: Momentum direction ('up' or 'down')

        Returns:
            bool: True if optimal entry time, False otherwise
        """
        # Reset entry optimization type
        self._state['entry_optimization'] = "default"

        # Check if we should wait for a pullback
        if direction == 'up':
            # For upward momentum, look for slight pullbacks in price
            if self.parameters['use_rsi'] and 'rsi' in indicators:
                rsi = indicators['rsi'].iloc[-1]
                rsi_prev = indicators['rsi'].iloc[-2]

                # Look for overbought condition
                if rsi > self.parameters['rsi_overbought']:
                    # Wait for pullback if extremely overbought
                    if rsi > 80:
                        return False

                    # If RSI is declining from an overbought level, it might be a good entry
                    if rsi < rsi_prev and rsi_prev - rsi > 3:
                        self._state['entry_optimization'] = "pullback"
                        return True

            # Check for price pullback
            if len(data) >= 3:
                if (data['close'].iloc[-1] < data['close'].iloc[-2] and
                        data['close'].iloc[-2] > data['close'].iloc[-3]):
                    # Small pullback after uptrend
                    pullback_pct = (data['close'].iloc[-2] - data['close'].iloc[-1]) / data['close'].iloc[-2]
                    if 0.005 < pullback_pct < 0.02:  # 0.5% to 2% pullback
                        self._state['entry_optimization'] = "pullback"
                        return True

            # Check for momentum breakout
            if self.parameters['use_macd'] and 'macd' in indicators and 'macd_signal' in indicators:
                if (indicators['macd'].iloc[-1] > indicators['macd_signal'].iloc[-1] and
                        indicators['macd'].iloc[-2] <= indicators['macd_signal'].iloc[-2]):
                    # MACD just crossed above signal line - strong entry
                    self._state['entry_optimization'] = "breakout"
                    return True

        elif direction == 'down':
            # For downward momentum, look for slight bounces in price
            if self.parameters['use_rsi'] and 'rsi' in indicators:
                rsi = indicators['rsi'].iloc[-1]
                rsi_prev = indicators['rsi'].iloc[-2]

                # Look for oversold condition
                if rsi < self.parameters['rsi_oversold']:
                    # Wait for bounce if extremely oversold
                    if rsi < 20:
                        return False

                    # If RSI is rising from an oversold level, it might be a good entry
                    if rsi > rsi_prev and rsi - rsi_prev > 3:
                        self._state['entry_optimization'] = "pullback"
                        return True

            # Check for price bounce
            if len(data) >= 3:
                if (data['close'].iloc[-1] > data['close'].iloc[-2] and
                        data['close'].iloc[-2] < data['close'].iloc[-3]):
                    # Small bounce after downtrend
                    bounce_pct = (data['close'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2]
                    if 0.005 < bounce_pct < 0.02:  # 0.5% to 2% bounce
                        self._state['entry_optimization'] = "pullback"
                        return True

            # Check for momentum breakout
            if self.parameters['use_macd'] and 'macd' in indicators and 'macd_signal' in indicators:
                if (indicators['macd'].iloc[-1] < indicators['macd_signal'].iloc[-1] and
                        indicators['macd'].iloc[-2] >= indicators['macd_signal'].iloc[-2]):
                    # MACD just crossed below signal line - strong entry
                    self._state['entry_optimization'] = "breakout"
                    return True

        # Default: assume it's an acceptable entry time
        return True

            # Check MACD if used
            if self.parameters['use_macd'] and 'macd_hist' in indicators:
                # MACD histogram turning negative indicates weakening momentum
                if (indicators['macd_hist'].iloc[-1] < 0 and
                        indicators['macd_hist'].iloc[-2] > 0):
                    reason = "Exit long: MACD histogram turned negative"
                    self._emit_exit_event(data, position, reason)
                    self.log_signal('exit', data, reason)
                    self._record_latency(start_time, "exit_signal", "macd_hist_exit")
                    return True

                # MACD crossing below signal line
                if (indicators['macd'].iloc[-1] < indicators['macd_signal'].iloc[-1] and
                        indicators['macd'].iloc[-2] > indicators['macd_signal'].iloc[-2]):
                    reason = "Exit long: MACD crossed below signal line"
                    self._emit_exit_event(data, position, reason)
                    self.log_signal('exit', data, reason)
                    self._record_latency(start_time, "exit_signal", "macd_cross_exit")
                    return True

            # Check ROC if used
            if self.parameters['use_roc'] and 'roc' in indicators:
                # ROC turning negative indicates momentum reversal
                if indicators['roc'].iloc[-1] < self.parameters['roc_threshold']:
                    reason = f"Exit long: ROC turned negative ({indicators['roc'].iloc[-1]:.2f})"
                    self._emit_exit_event(data, position, reason)
                    self.log_signal('exit', data, reason)
                    self._record_latency(start_time, "exit_signal", "roc_exit")
                    return True

        elif position['direction'] == 'short':
            # Exit short position if momentum weakens or reverses

            # Check RSI if used
            if self.parameters['use_rsi'] and 'rsi' in indicators:
                # RSI crossing above threshold indicates weakening downward momentum
                if indicators['rsi'].iloc[-1] > self._state['rsi_threshold_upper']:
                    reason = f"Exit short: RSI increased above threshold ({indicators['rsi'].iloc[-1]:.1f})"
                    self._emit_exit_event(data, position, reason)
                    self.log_signal('exit', data, reason)
                    self._record_latency(start_time, "exit_signal", "rsi_exit")
                    return True

            # Check MACD if used
            if self.parameters['use_macd'] and 'macd_hist' in indicators:
                # MACD histogram turning positive indicates weakening downward momentum
                if (indicators['macd_hist'].iloc[-1] > 0 and
                        indicators['macd_hist'].iloc[-2] < 0):
                    reason = "Exit short: MACD histogram turned positive"
                    self._emit_exit_event(data, position, reason)
                    self.log_signal('exit', data, reason)
                    self._record_latency(start_time, "exit_signal", "macd_hist_exit")
                    return True

                # MACD crossing above signal line
                if (indicators['macd'].iloc[-1] > indicators['macd_signal'].iloc[-1] and
                        indicators['macd'].iloc[-2] < indicators['macd_signal'].iloc[-2]):
                    reason = "Exit short: MACD crossed above signal line"
                    self._emit_exit_event(data, position, reason)
                    self.log_signal('exit', data, reason)
                    self._record_latency(start_time, "exit_signal", "macd_cross_exit")
                    return True

            # Check ROC if used
            if self.parameters['use_roc'] and 'roc' in indicators:
                # ROC turning positive indicates momentum reversal
                if indicators['roc'].iloc[-1] > -self.parameters['roc_threshold']:
                    reason = f"Exit short: ROC turned positive ({indicators['roc'].iloc[-1]:.2f})"
                    self._emit_exit_event(data, position, reason)
                    self.log_signal('exit', data, reason)
                    self._record_latency(start_time, "exit_signal", "roc_exit")
                    return True

        # No exit signal generated
        self._record_latency(start_time, "no_exit_signal")
        return False

    def _emit_exit_event(self, data: pd.DataFrame, position: Dict[str, Any], reason: str) -> None:
        """
        Emit exit event to the event bus.

        Args:
            data: DataFrame with market data
            position: Current position information
            reason: Reason for exit
        """
        EventBus.emit("strategy.exit", {
            'strategy_id': self.id,
            'strategy_type': 'momentum',
            'timestamp': time.time(),
            'position_id': position.get('id', 'unknown'),
            'direction': position.get('direction', 'unknown'),
            'entry_price': position.get('entry_price', 0),
            'current_price': data['close'].iloc[-1],
            'holding_time': time.time() - position.get('entry_time', time.time()),
            'reason': reason
        })

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.

        Args:
            prices: Series of prices
            window: RSI window period

        Returns:
            Series with RSI values
        """
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        rs = avg_gain / avg_loss.replace(0, 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(self, prices: pd.Series, fast_period: int = 12,
                        slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            prices: Series of prices
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period

        Returns:
            Tuple of (MACD, Signal, Histogram)
        """
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()

        macd = fast_ema - slow_ema
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        hist = macd - signal

        return macd, signal, hist

    def _calculate_roc(self, prices: pd.Series, period: int = 10) -> pd.Series:
        """
        Calculate Rate of Change (ROC).

        Args:
            prices: Series of prices
            period: ROC calculation period

        Returns:
            Series with ROC values (percentage)
        """
        roc = prices.pct_change(period) * 100
        return roc

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range.

        Args:
            data: DataFrame with OHLC data
            period: ATR calculation period

        Returns:
            Series with ATR values
        """
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def on_trade_completed(self, trade_result: Dict[str, Any]) -> None:
        """
        Callback for when a trade is completed to adapt strategy parameters.

        Args:
            trade_result: Dict with trade result information
        """
        # Get trade outcome
        pnl = trade_result.get('pnl', 0)
        pnl_pct = trade_result.get('pnl_pct', 0)
        successful = pnl > 0

        # Regime ID that was active during the trade
        regime_id = trade_result.get('regime_id', None)

        # Entry optimization used
        entry_optimization = trade_result.get('entry_optimization', 'default')

        # Store original parameters for tracking changes
        old_momentum_threshold = self._state['momentum_threshold']
        old_stop_loss_multiplier = self._state['stop_loss_atr_multiplier']
        old_take_profit_multiplier = self._state['take_profit_atr_multiplier']

        if successful:
            # If trade was successful, fine-tune the strategy
            if regime_id is not None:  # Check if we have valid regime information
                # If a specific entry optimization worked well, decrease momentum threshold
                if entry_optimization != 'default':
                    self._state['momentum_threshold'] = max(0.5, self._state['momentum_threshold'] * 0.95)
                    logger.debug(
                        f"Lowered momentum threshold to {self._state['momentum_threshold']:.2f} after successful trade with {entry_optimization} entry")

                # For very profitable trades, adjust take profit
                if pnl_pct > 0.05:  # 5% profit
                    self._state['take_profit_atr_multiplier'] = min(4.0,
                                                                    self._state['take_profit_atr_multiplier'] * 1.05)
                    logger.debug(
                        f"Increased take profit multiplier to {self._state['take_profit_atr_multiplier']:.2f} after highly profitable trade")
        else:
            # If trade was unsuccessful, adjust parameters
            if regime_id is not None:  # Check if we have valid regime information
                # Increase momentum threshold to get more selective
                self._state['momentum_threshold'] = min(0.75, self._state['momentum_threshold'] * 1.05)
                logger.debug(
                    f"Increased momentum threshold to {self._state['momentum_threshold']:.2f} after unsuccessful trade")

                # For significant losses, adjust stop loss
                if pnl_pct < -0.03:  # 3% loss
                    self._state['stop_loss_atr_multiplier'] = max(1.5, self._state['stop_loss_atr_multiplier'] * 0.95)
                    logger.debug(
                        f"Decreased stop loss multiplier to {self._state['stop_loss_atr_multiplier']:.2f} after losing trade")

        # Emit parameter adaptation event if any parameters changed
        if (old_momentum_threshold != self._state['momentum_threshold'] or
                old_stop_loss_multiplier != self._state['stop_loss_atr_multiplier'] or
                old_take_profit_multiplier != self._state['take_profit_atr_multiplier']):
            self._emit_parameter_adaptation_event(
                trade_result,
                {
                    'momentum_threshold': {
                        'old': old_momentum_threshold,
                        'new': self._state['momentum_threshold']
                    },
                    'stop_loss_atr_multiplier': {
                        'old': old_stop_loss_multiplier,
                        'new': self._state['stop_loss_atr_multiplier']
                    },
                    'take_profit_atr_multiplier': {
                        'old': old_take_profit_multiplier,
                        'new': self._state['take_profit_atr_multiplier']
                    },
                    'entry_optimization': entry_optimization
                }
            )

    def _emit_parameter_adaptation_event(self, trade_result: Dict[str, Any], param_changes: Dict[str, Any]) -> None:
        """
        Emit parameter adaptation event to the event bus.

        Args:
            trade_result: Trade result information
            param_changes: Parameter changes applied
        """
        EventBus.emit("strategy.parameter_adaptation", {
            'strategy_id': self.id,
            'strategy_type': 'momentum',
            'timestamp': time.time(),
            'trade_id': trade_result.get('id', 'unknown'),
            'trade_pnl': trade_result.get('pnl', 0),
            'trade_pnl_pct': trade_result.get('pnl_pct', 0),
            'regime_id': trade_result.get('regime_id', 'unknown'),
            'parameter_changes': param_changes,
            'adaptation_method': 'online_bayesian'
        })

    def get_required_features(self) -> Set[str]:
        """
        Return the list of features required by the strategy.

        Returns:
            Set of feature names
        """
        features = {'open', 'high', 'low', 'close', 'volume'}

        # Add indicator features based on enabled components
        if self.parameters['use_rsi']:
            features.add('rsi_14')

        if self.parameters['use_macd']:
            features.update({'macd', 'macd_signal', 'macd_hist'})

        if self.parameters['use_roc']:
            features.add('roc_10')

        # ATR is used for risk management
        features.add('atr_14')

        return features

    def update_parameters_online(self, performance_metrics: Dict[str, float],
                                market_conditions: Dict[str, Any]) -> None:
        """
        Update strategy parameters based on recent performance and market conditions.

        Args:
            performance_metrics: Performance metrics
            market_conditions: Current market conditions
        """
        # Extract relevant metrics
        sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
        win_rate = performance_metrics.get('win_rate', 0.5)
        profit_factor = performance_metrics.get('profit_factor', 1.0)

        # Extract market conditions
        adx = market_conditions.get('adx', 20)
        volatility = market_conditions.get('volatility', 0.5)

        # Store original parameters for tracking changes
        old_momentum_threshold = self._state['momentum_threshold']
        old_rsi_threshold_upper = self._state['rsi_threshold_upper']
        old_rsi_threshold_lower = self._state['rsi_threshold_lower']

        # Update momentum threshold based on performance
        if profit_factor > 1.5 and win_rate > 0.5:
            # Successful configuration, slightly reduce threshold for more trades
            new_threshold = self._state['momentum_threshold'] * 0.98
            self._state['momentum_threshold'] = max(0.5, new_threshold)
        elif profit_factor < 1.0 or win_rate < 0.4:
            # Unsuccessful configuration, increase threshold for better quality
            new_threshold = self._state['momentum_threshold'] * 1.02
            self._state['momentum_threshold'] = min(0.75, new_threshold)

        # Adjust RSI thresholds based on ADX (trending vs ranging)
        if adx > 25:  # Stronger trend
            # In trending markets, use more aggressive thresholds
            self._state['rsi_threshold_upper'] = min(60, self._state['rsi_threshold_upper'] + 1)
            self._state['rsi_threshold_lower'] = max(40, self._state['rsi_threshold_lower'] - 1)
        else:  # Weaker trend
            # In ranging markets, use more conservative thresholds
            self._state['rsi_threshold_upper'] = max(53, self._state['rsi_threshold_upper'] - 1)
            self._state['rsi_threshold_lower'] = min(47, self._state['rsi_threshold_lower'] + 1)

        # Adjust stop loss and take profit multipliers based on volatility
        if volatility > 0.7:  # High volatility
            self._state['stop_loss_atr_multiplier'] = min(3.0, self._state['stop_loss_atr_multiplier'] * 1.02)
            self._state['take_profit_atr_multiplier'] = min(4.0, self._state['take_profit_atr_multiplier'] * 1.02)
        elif volatility < 0.3:  # Low volatility
            self._state['stop_loss_atr_multiplier'] = max(1.5, self._state['stop_loss_atr_multiplier'] * 0.98)
            self._state['take_profit_atr_multiplier'] = max(2.5, self._state['take_profit_atr_multiplier'] * 0.98)

        # Log and emit parameter updates
        logger.info(f"Online parameter update: threshold={self._state['momentum_threshold']:.2f}, "
                    f"rsi_upper={self._state['rsi_threshold_upper']:.2f}, "
                    f"rsi_lower={self._state['rsi_threshold_lower']:.2f}")

        # Emit event only if parameters actually changed
        if (old_momentum_threshold != self._state['momentum_threshold'] or
                old_rsi_threshold_upper != self._state['rsi_threshold_upper'] or
                old_rsi_threshold_lower != self._state['rsi_threshold_lower']):
            EventBus.emit("strategy.online_update", {
                'strategy_id': self.id,
                'strategy_type': 'momentum',
                'timestamp': time.time(),
                'new_parameters': {
                    'momentum_threshold': self._state['momentum_threshold'],
                    'rsi_threshold_upper': self._state['rsi_threshold_upper'],
                    'rsi_threshold_lower': self._state['rsi_threshold_lower'],
                    'stop_loss_atr_multiplier': self._state['stop_loss_atr_multiplier'],
                    'take_profit_atr_multiplier': self._state['take_profit_atr_multiplier']
                },
                'performance_metrics': performance_metrics,
                'market_conditions': market_conditions
            })

    @classmethod
    def register(cls) -> None:
        """Register strategy with the strategy registry."""
        try:
            from models.strategies.strategy_registry import StrategyRegistry
            StrategyRegistry.register(
                cls,
                "MomentumStrategy",
                description="Identifies and trades with market momentum using RSI, MACD, and price rate of change",
                default_parameters={
                    'rsi_period': 14,
                    'macd_fast': 12,
                    'macd_slow': 26,
                    'roc_period': 10,
                    'min_momentum_score': 0.6
                },
                compatibility={
                    'min_bars': 50,
                    'timeframes': ['15m', '1h', '4h', 'daily', 'weekly'],
                    'asset_classes': ['forex', 'crypto', 'stocks', 'commodities']
                }
            )
            logger.info("MomentumStrategy registered successfully")
        except ImportError:
            logger.warning("StrategyRegistry not available, skip registration")
        except Exception as e:
            logger.error(f"Failed to register MomentumStrategy: {str(e)}")

    def cluster_fit(self, cluster_metrics: Dict[str, float]) -> float:
        """
        Determine if the momentum strategy is suitable for the given cluster.

        Args:
            cluster_metrics: Dict of cluster characteristics

        Returns:
            float: Fitness score between 0.0 and 1.0 indicating how well the strategy fits
        """
        # Basic validation in case metrics are missing
        if not cluster_metrics or not isinstance(cluster_metrics, dict):
            return 0.5  # Neutral score if no metrics available

        # Start with base fitness
        fitness_score = 0.5

        # Momentum strategy typically works best in trending markets with
        # sufficient trend strength, momentum, and persistence

        # ADX component (Trend strength)
        adx_mean = cluster_metrics.get('ADX_mean', 20)
        if adx_mean < 15:  # Very low ADX indicates ranging/choppy market
            adx_score = max(0, adx_mean / 15)
        elif adx_mean > 40:  # Very high ADX might indicate trend exhaustion
            adx_score = max(0, 1 - (adx_mean - 40) / 20)
        else:
            # Ideal range 15-40 with peak at 25-30
            adx_score = min(1.0, 0.6 + 0.4 * (1 - abs(adx_mean - 27.5) / 12.5))

        # Momentum component
        momentum_score_metric = cluster_metrics.get('momentum_score', 0.5)
        if momentum_score_metric < 0.3:  # Very low momentum
            momentum_fitness = max(0, momentum_score_metric / 0.3)
        else:
            # Higher momentum scores are better for momentum strategy
            momentum_fitness = min(1.0, 0.7 + 0.3 * momentum_score_metric)

        # Trend persistence component
        trend_persistence = cluster_metrics.get('trend_persistence', 0.5)
        if trend_persistence < 0.3:  # Low trend persistence
            persistence_score = max(0, trend_persistence / 0.3)
        else:
            # Higher persistence is better
            persistence_score = min(1.0, 0.5 + 0.5 * trend_persistence)

        # Volatility component
        volatility_rank = cluster_metrics.get('volatility_pct_rank', 0.5)
        if volatility_rank > 0.9:  # Extremely high volatility
            volatility_score = max(0, 1 - (volatility_rank - 0.9) * 10)
        else:
            # Moderate to high volatility is ideal (0.3-0.9)
            volatility_score = min(1.0, volatility_rank / 0.9)

        # Weighted combination
        weights = {
            'adx': 0.3,
            'momentum': 0.3,
            'persistence': 0.3,
            'volatility': 0.1
        }

        fitness_score = (
                weights['adx'] * adx_score +
                weights['momentum'] * momentum_fitness +
                weights['persistence'] * persistence_score +
                weights['volatility'] * volatility_score
        )

        # Log detailed scores for analysis
        logger.debug(f"Momentum cluster fit - ADX: {adx_score:.2f}, Momentum: {momentum_fitness:.2f}, "
                     f"Persistence: {persistence_score:.2f}, Vol: {volatility_score:.2f}, Total: {fitness_score:.2f}")

        # Emit event with fitness score
        EventBus.emit("strategy.cluster_fit", {
            'strategy_id': self.id,
            'strategy_type': 'momentum',
            'cluster_id': cluster_metrics.get('cluster_id', 'unknown'),
            'fitness_score': fitness_score,
            'components': {
                'adx_score': adx_score,
                'momentum_score': momentum_fitness,
                'persistence_score': persistence_score,
                'volatility_score': volatility_score
            }
        })

    def _confirm_volume(self, data: pd.DataFrame, direction: str) -> bool:
        """
        Confirm signal with volume analysis.

        Args:
            data: DataFrame with OHLC and volume data
            direction: Momentum direction ('up' or 'down')

        Returns:
            bool: True if volume confirms signal, False otherwise
        """
        if 'volume' not in data.columns:
            return True  # No volume data to confirm

        # Get recent volume data
        if len(data) < 10:
            return True  # Not enough volume history

        recent_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].iloc[-10:].mean()

        # Volume should be above average for confirmation
        if recent_volume < avg_volume * 0.8:
            return False

        # For upward momentum, volume should increase on up days
        if direction == 'up':
            if data['close'].iloc[-1] > data['open'].iloc[-1]:  # Up day
                up_day_volume = recent_volume
                # Compare to previous up day volume
                prev_up_days = [i for i in range(-5, -1) if data['close'].iloc[i] > data['open'].iloc[i]]
                if prev_up_days:
                    prev_up_volume = data['volume'].iloc[prev_up_days].mean()
                    if up_day_volume < prev_up_volume * 0.9:
                        return False

        # For downward momentum, volume should increase on down days
        elif direction == 'down':
            if data['close'].iloc[-1] < data['open'].iloc[-1]:  # Down day
                down_day_volume = recent_volume
                # Compare to previous down day volume
                prev_down_days = [i for i in range(-5, -1) if data['close'].iloc[i] < data['open'].iloc[i]]
                if prev_down_days:
                    prev_down_volume = data['volume'].iloc[prev_down_days].mean()
                    if down_day_volume < prev_down_volume * 0.9:
                        return False

        return True

        return fitness_score

    def adapt_to_regime(self, regime_data: Dict[str, Any]) -> None:
        """
        Adapt strategy parameters based on market regime.

        Args:
            regime_data: Market regime information
        """
        # Extract regime characteristics
        regime_id = regime_data.get('id', 'unknown')
        directional_bias = regime_data.get('directional_bias', DirectionalBias.NEUTRAL)
        volatility_regime = regime_data.get('volatility_regime', VolatilityRegime.NORMAL)
        peak_hour = regime_data.get('peak_hour', 0)

        logger.info(
            f"Adapting momentum strategy to regime {regime_id}: bias={directional_bias}, volatility={volatility_regime}")

        # Store original parameters before adaptation for tracking changes
        old_momentum_threshold = self._state['momentum_threshold']
        old_rsi_threshold_upper = self._state['rsi_threshold_upper']
        old_rsi_threshold_lower = self._state['rsi_threshold_lower']

        # Apply optimization based on time of day
        if peak_hour in [8, 9]:
            # European market opening optimizations
            self._state['rsi_threshold_upper'] = max(52, self.parameters['rsi_threshold'] - 3)
            self._state['rsi_threshold_lower'] = min(48, 100 - self.parameters['rsi_threshold'] + 3)
            self._state['momentum_threshold'] = max(0.55, self.parameters['min_momentum_score'] - 0.05)
            logger.info(f"Applied European session momentum optimizations")
        elif peak_hour in [13, 14, 15]:
            # US market opening optimizations (typically more volatile)
            self._state['rsi_threshold_upper'] = min(60, self.parameters['rsi_threshold'] + 5)
            self._state['rsi_threshold_lower'] = max(40, 100 - self.parameters['rsi_threshold'] - 5)
            self._state['momentum_threshold'] = min(0.65, self.parameters['min_momentum_score'] + 0.05)
            logger.info(f"Applied US session momentum optimizations")

        # Apply optimization based on directional bias
        if directional_bias == DirectionalBias.UPWARD:
            # In upward bias, lower threshold for long signals
            self._state['rsi_threshold_upper'] = max(52, self.parameters['rsi_threshold'] - 3)
            # Higher threshold for short signals
            self._state['rsi_threshold_lower'] = max(45, 100 - self.parameters['rsi_threshold'] - 5)

            # Create biased signal generator for upward bias
            original_generate = self._original_generate_signal

            def biased_signal(data: pd.DataFrame) -> Optional[str]:
                signal = original_generate(data)

                # In upward bias, prefer long signals
                if signal == 'short':
                    # Additional confirmation for shorts in upward bias
                    indicators = {}
                    if self.parameters['use_rsi'] and 'rsi_14' in data.columns:
                        indicators['rsi'] = data['rsi_14']
                    elif self.parameters['use_rsi']:
                        indicators['rsi'] = self._calculate_rsi(data['close'], self.parameters['rsi_period'])

                    if 'rsi' in indicators and indicators['rsi'].iloc[-1] > 40:
                        # Not oversold enough for a short in upward bias
                        return None

                return signal

            self.generate_signal = biased_signal
            logger.info(f"Applied upward bias optimization to {self.name}")

        elif directional_bias == DirectionalBias.DOWNWARD:
            # In downward bias, lower threshold for short signals
            self._state['rsi_threshold_lower'] = min(48, 100 - self.parameters['rsi_threshold'] + 3)
            # Higher threshold for long signals
            self._state['rsi_threshold_upper'] = min(55, self.parameters['rsi_threshold'] + 5)

            # Create biased signal generator for downward bias
            original_generate = self._original_generate_signal

            def biased_signal(data: pd.DataFrame) -> Optional[str]:
                signal = original_generate(data)

                # In downward bias, prefer short signals
                if signal == 'long':
                    # Additional confirmation for longs in downward bias
                    indicators = {}
                    if self.parameters['use_rsi'] and 'rsi_14' in data.columns:
                        indicators['rsi'] = data['rsi_14']
                    elif self.parameters['use_rsi']:
                        indicators['rsi'] = self._calculate_rsi(data['close'], self.parameters['rsi_period'])

                    if 'rsi' in indicators and indicators['rsi'].iloc[-1] < 60:
                        # Not overbought enough for a long in downward bias
                        return None

                return signal

            self.generate_signal = biased_signal
            logger.info(f"Applied downward bias optimization to {self.name}")
        else:
            # Reset to original signal generator if no directional bias
            self.generate_signal = self._original_generate_signal

        # Apply optimization based on volatility regime
        if volatility_regime == VolatilityRegime.HIGH:
            # In high volatility, require stronger momentum signals
            self._state['momentum_threshold'] = min(0.7, self.parameters['min_momentum_score'] + 0.1)
            self._state['stop_loss_atr_multiplier'] = 2.5  # Wider stops in high volatility
            self._state['take_profit_atr_multiplier'] = 3.5  # Wider targets in high volatility
            logger.info(f"Applied high volatility momentum optimizations")
        elif volatility_regime == VolatilityRegime.LOW:
            # In low volatility, allow weaker momentum signals
            self._state['momentum_threshold'] = max(0.5, self.parameters['min_momentum_score'] - 0.1)
            self._state['stop_loss_atr_multiplier'] = 1.5  # Tighter stops in low volatility
            self._state['take_profit_atr_multiplier'] = 2.5  # Tighter targets in low volatility
            logger.info(f"Applied low volatility momentum optimizations")

        # Emit regime adaptation event
        EventBus.emit("strategy.regime_adaptation", {
            'strategy_id': self.id,
            'strategy_type': 'momentum',
            'timestamp': time.time(),
            'regime_id': regime_id,
            'parameters': {
                'momentum_threshold': {
                    'old': old_momentum_threshold,
                    'new': self._state['momentum_threshold']
                },
                'rsi_threshold_upper': {
                    'old': old_rsi_threshold_upper,
                    'new': self._state['rsi_threshold_upper']
                },
                'rsi_threshold_lower': {
                    'old': old_rsi_threshold_lower,
                    'new': self._state['rsi_threshold_lower']
                },
                'stop_loss_atr_multiplier': self._state['stop_loss_atr_multiplier'],
                'take_profit_atr_multiplier': self._state['take_profit_atr_multiplier']
            }
        })