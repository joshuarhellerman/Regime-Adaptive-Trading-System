"""
Breakout Strategy Module

This module implements a breakout strategy that identifies price movements beyond
consolidation zones. The strategy adapts to different market regimes and uses
multiple confirmation techniques to filter false breakouts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Set, Callable, cast
import logging
import time
from enum import Enum

from models.strategies.strategy_base import TradingStrategy, DirectionalBias, VolatilityRegime
from core.event_bus import EventBus
from core.performance_metrics import PerformanceMetrics

# Configure logger
logger = logging.getLogger(__name__)

class BreakoutStrategy(TradingStrategy):
    """
    Breakout strategy that adapts to market regimes.

    This strategy identifies price breakouts from consolidation patterns and
    applies multiple filters to reduce false signals, including volume confirmation,
    volatility analysis, and multi-timeframe validation.
    """

    def __init__(self, name: str = None, parameters: Dict[str, Any] = None, strategy_id: str = None):
        """
        Initialize the breakout strategy.

        Args:
            name: Strategy name
            parameters: Strategy-specific parameters
            strategy_id: Unique identifier for the strategy instance
        """
        # Default parameters
        default_params = {
            'consolidation_period': 20,
            'breakout_threshold': 1.5,
            'volume_confirmation': True,
            'min_volume_increase': 1.5,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'min_consolidation_length': 10,
            'max_consolidation_width_pct': 0.05,
            'min_consolidation_quality': 0.7,
            'false_breakout_filter': True,
            'confirmation_candles': 2,
            'multi_timeframe_confirmation': True,
            'min_bars': 30  # Minimum bars needed for calculation
        }

        # Merge default parameters with provided parameters
        if parameters:
            for key, value in parameters.items():
                default_params[key] = value

        # Initialize the base class
        super().__init__(name or "Breakout", default_params, strategy_id)

        # Strategy-specific attributes
        self._state = {
            'breakout_threshold_modifier': 1.0,
            'consolidation_quality_threshold': self.parameters['min_consolidation_quality'],
            'support_resistance_levels': {},
            'detected_patterns': [],
            'last_signal_time': 0.0,
            'latency_metrics': [],
            'directional_bias': DirectionalBias.NEUTRAL,
            'volatility_regime': VolatilityRegime.NORMAL,
            'peak_hour': 0
        }

        logger.info(
            f"Initialized BreakoutStrategy with period={self.parameters['consolidation_period']}, "
            f"threshold={self.parameters['breakout_threshold']}, "
            f"volume_confirmation={self.parameters['volume_confirmation']}"
        )

        # Register to central event bus
        self._register_with_event_bus()

    def _register_with_event_bus(self) -> None:
        """Register strategy with the event bus."""
        EventBus.subscribe(f"strategy.{self.id}.parameter_update", self._handle_parameter_update)
        EventBus.subscribe("market.regime_change", self._handle_regime_change)
        return None

    def _handle_parameter_update(self, event_data: Dict[str, Any]) -> None:
        """Handle parameter update events."""
        if 'parameters' in event_data:
            for key, value in event_data['parameters'].items():
                if key in self.parameters:
                    self.parameters[key] = value
                    logger.info(f"Updated parameter {key} to {value} for strategy {self.id}")

            # Re-validate parameters after update
            self._validate_parameters()
        return None

    def _handle_regime_change(self, event_data: Dict[str, Any]) -> None:
        """Handle market regime change events."""
        if 'regime' in event_data:
            regime_id = event_data.get('regime_id')
            logger.info(f"Market regime changed to {regime_id}, adapting parameters")
            # Trigger regime-specific optimizations
            self.adapt_to_regime(event_data['regime'])
        return None

    def _validate_parameters(self) -> None:
        """
        Validate strategy parameters.

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate period values
        for param in ['consolidation_period', 'atr_period', 'min_consolidation_length', 'confirmation_candles']:
            if self.parameters[param] <= 0:
                raise ValueError(f"Parameter {param} must be greater than 0")

        # Validate threshold values
        if self.parameters['breakout_threshold'] <= 0:
            raise ValueError("Breakout threshold must be greater than 0")

        if not 0 < self.parameters['min_consolidation_quality'] <= 1.0:
            raise ValueError("Consolidation quality threshold must be between 0 and 1")

        if self.parameters['max_consolidation_width_pct'] <= 0:
            raise ValueError("Consolidation width percentage must be greater than 0")

        if self.parameters['min_volume_increase'] <= 0:
            raise ValueError("Minimum volume increase must be greater than 0")

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate input data for signal generation.

        Args:
            data: DataFrame with market data

        Returns:
            bool: True if data is valid, False otherwise

        Raises:
            ValueError: If data is invalid
        """
        if data is None or len(data) == 0:
            raise ValueError("Data cannot be empty")

        # Check required columns
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        return True

    def generate_signal(self, data: pd.DataFrame) -> Optional[str]:
        """
        Generate a breakout signal.

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
            if not self.validate_data(data):
                self._record_latency(start_time, "data_validation_failed")
                return None
        except ValueError as e:
            logger.error(f"Data validation failed: {str(e)}")
            self._record_latency(start_time, "data_validation_error")
            return None

        # Ensure we have enough data
        if data is None or len(data) < self.parameters['min_bars']:
            logger.debug(f"Insufficient data for signal generation: {len(data) if data is not None else 0} bars")
            self._record_latency(start_time, "insufficient_data")
            return None

        # Generate the base signal
        signal = self._base_generate_signal(data, start_time)

        # Apply directional bias if applicable
        if signal is not None and self._state['directional_bias'] != DirectionalBias.NEUTRAL:
            signal = self._apply_directional_bias(signal, data)

        # Record performance metrics
        self._record_latency(start_time, "signal_generation", signal)

        return signal

    def _base_generate_signal(self, data: pd.DataFrame, start_time: float) -> Optional[str]:
        """
        Base implementation of signal generation.

        Args:
            data: DataFrame with market data
            start_time: Start time for performance tracking

        Returns:
            str: 'long', 'short', or None for no signal
        """
        # Detect consolidation patterns
        consolidation = self._detect_consolidation(data)
        if not consolidation['is_valid']:
            self._record_latency(start_time, "no_consolidation")
            return None

        # Check for breakout
        breakout = self._detect_breakout(data, consolidation)
        if not breakout['is_breakout']:
            self._record_latency(start_time, "no_breakout")
            return None

        # Check for false breakout if enabled
        if self.parameters['false_breakout_filter'] and self._is_false_breakout(data, breakout):
            logger.debug("False breakout detected, ignoring signal")
            self._record_latency(start_time, "false_breakout")
            return None

        # Apply volume confirmation if enabled
        if self.parameters['volume_confirmation'] and not self._confirm_volume(data, breakout):
            logger.debug("Volume confirmation failed, ignoring signal")
            self._record_latency(start_time, "volume_confirmation_failed")
            return None

        # Apply multi-timeframe confirmation if enabled
        if self.parameters['multi_timeframe_confirmation'] and not self._confirm_multi_timeframe(data, breakout):
            logger.debug("Multi-timeframe confirmation failed, ignoring signal")
            self._record_latency(start_time, "multi_timeframe_confirmation_failed")
            return None

        # Generate signal based on breakout direction
        signal = None
        if breakout['direction'] == 'up':
            self.log_signal('long', data, f"Upward breakout: {breakout['strength']:.2f} strength")
            signal = 'long'
        elif breakout['direction'] == 'down':
            self.log_signal('short', data, f"Downward breakout: {breakout['strength']:.2f} strength")
            signal = 'short'

        # Emit signal event if signal generated
        if signal:
            self._emit_signal_event(signal, data, breakout)
            self._state['last_signal_time'] = time.time()

        return signal

    def _apply_directional_bias(self, signal: str, data: pd.DataFrame) -> Optional[str]:
        """
        Apply directional bias to the generated signal.

        Args:
            signal: Generated signal ('long', 'short')
            data: DataFrame with market data

        Returns:
            str: Modified signal or None if signal should be filtered
        """
        # Get the consolidation and breakout info
        consolidation = self._detect_consolidation(data)
        if not consolidation['is_valid']:
            return signal

        breakout = self._detect_breakout(data, consolidation)
        if not breakout['is_breakout']:
            return signal

        # Apply upward bias
        if self._state['directional_bias'] == DirectionalBias.UPWARD:
            # Keep long signals
            if signal == 'long':
                return 'long'
            # Filter short signals unless they're strong
            elif signal == 'short':
                if breakout['strength'] > 1.5:  # Require stronger breakout
                    return 'short'
                return None

        # Apply downward bias
        elif self._state['directional_bias'] == DirectionalBias.DOWNWARD:
            # Keep short signals
            if signal == 'short':
                return 'short'
            # Filter long signals unless they're strong
            elif signal == 'long':
                if breakout['strength'] > 1.5:  # Require stronger breakout
                    return 'long'
                return None

        return signal

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

        # Keep only the last 100 metrics
        if len(self._state['latency_metrics']) > 100:
            self._state['latency_metrics'] = self._state['latency_metrics'][-100:]

        # Report latency metrics
        PerformanceMetrics.record(
            f"strategy.{self.id}.latency",
            latency,
            tags={'stage': stage, 'signal': signal or 'none'}
        )

    def _emit_signal_event(self, signal: str, data: pd.DataFrame, breakout: Dict[str, Any]) -> None:
        """
        Emit signal event to the event bus.

        Args:
            signal: Signal type ('long' or 'short')
            data: DataFrame with market data
            breakout: Breakout information dictionary
        """
        # Guard against potential KeyError
        last_price = data['close'].iloc[-1] if len(data) > 0 and 'close' in data.columns else 0

        EventBus.emit("strategy.signal", {
            'strategy_id': self.id,
            'strategy_type': 'breakout',
            'timestamp': time.time(),
            'signal': signal,
            'instrument': data.get('symbol', 'unknown'),
            'price': last_price,
            'strength': breakout['strength'],
            'confidence': min(1.0, breakout['strength'] / 2.0),
            'metadata': {
                'breakout_direction': breakout['direction'],
                'consolidation_quality': self._state['consolidation_quality_threshold'],
                'threshold_modifier': self._state['breakout_threshold_modifier'],
                'pattern': 'support_resistance' if self._state['detected_patterns'] else 'unknown'
            }
        })

    def risk_parameters(self, data: pd.DataFrame, entry_price: float) -> Dict[str, float]:
        """
        Calculate risk parameters for the breakout strategy.

        Args:
            data: DataFrame with market data
            entry_price: Entry price for the trade

        Returns:
            Dict with stop loss, take profit, and position size
        """
        # Guard against empty data
        if data is None or len(data) == 0:
            logger.error("Empty data provided to risk_parameters")
            return {
                'stop_loss_pct': 0.01,  # Default fallback
                'take_profit_pct': 0.03,
                'position_size': 0,
                'stop_price': 0,
                'take_profit_price': 0,
                'atr': 0
            }

        # Calculate ATR if not available in data
        if 'atr_14' not in data.columns:
            atr = self._calculate_atr(data, self.parameters['atr_period']).iloc[-1]
        else:
            atr = data['atr_14'].iloc[-1]

        # Detection of most recent pattern for stop placement
        latest_pattern = None
        if self._state['detected_patterns']:
            latest_pattern = self._state['detected_patterns'][-1]

        # Determine stop distance based on pattern and ATR
        if latest_pattern:
            # Use pattern levels for stop placement
            if latest_pattern['type'] == 'support_resistance':
                if entry_price > latest_pattern['resistance']:  # Long position
                    stop_price = max(latest_pattern['support'], entry_price - atr * 2)
                    stop_distance = entry_price - stop_price
                else:  # Short position
                    stop_price = min(latest_pattern['resistance'], entry_price + atr * 2)
                    stop_distance = stop_price - entry_price
            else:
                # Use ATR for channel patterns
                stop_distance = atr * self.parameters['atr_multiplier'] * self.stop_loss_modifier
        else:
            # Default to ATR-based stop
            stop_distance = atr * self.parameters['atr_multiplier'] * self.stop_loss_modifier

        # Calculate stop loss and take profit percentages (with guards against division by zero)
        if entry_price > 0:
            stop_loss_pct = stop_distance / entry_price
        else:
            stop_loss_pct = 0.01  # Default fallback if entry price is zero or negative
            logger.warning(f"Invalid entry price {entry_price}, using default stop loss percentage")

        # Guard against zero stop_loss_pct
        if stop_loss_pct <= 0:
            stop_loss_pct = 0.01  # Default fallback
            logger.warning("Calculated stop loss percentage is zero or negative, using default")

        # Breakout strategies typically target 2-3x the risk
        take_profit_pct = stop_loss_pct * 3.0 * self.profit_target_modifier

        # Calculate position size (safely)
        position_size = self.dynamic_position_size(self.account_balance)

        # Calculate stop and take profit prices
        stop_price = entry_price * (1 - stop_loss_pct) if entry_price > 0 else 0
        take_profit_price = entry_price * (1 + take_profit_pct) if entry_price > 0 else 0

        # Emit risk parameters event
        self._emit_risk_event(entry_price, stop_loss_pct, take_profit_pct, position_size)

        return {
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'position_size': position_size,
            'stop_price': stop_price,
            'take_profit_price': take_profit_price,
            'atr': atr
        }

    def _emit_risk_event(self, entry_price: float, stop_loss_pct: float,
                        take_profit_pct: float, position_size: float) -> None:
        """
        Emit risk parameters event to the event bus.

        Args:
            entry_price: Entry price
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            position_size: Position size
        """
        # Guard against division by zero
        risk_reward_ratio = 0
        if stop_loss_pct > 0:
            risk_reward_ratio = take_profit_pct / stop_loss_pct

        EventBus.emit("strategy.risk_parameters", {
            'strategy_id': self.id,
            'strategy_type': 'breakout',
            'timestamp': time.time(),
            'entry_price': entry_price,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'position_size': position_size,
            'risk_reward_ratio': risk_reward_ratio
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

        # Guard against empty data
        if data is None or len(data) == 0:
            logger.error("Empty data provided to exit_signal")
            return False

        super().exit_signal(data, position)

        # Calculate ATR if not available
        if 'atr_14' not in data.columns:
            atr = self._calculate_atr(data, self.parameters['atr_period'])
        else:
            atr = data['atr_14']

        # Safely get the current ATR
        if len(atr) > 0:
            current_atr = atr.iloc[-1]
        else:
            logger.warning("No ATR data available for exit signal calculation")
            self._record_latency(start_time, "exit_signal", "no_atr_data")
            return False

        # Safely get entry price and current price
        entry_price = position.get('entry_price', None)
        if entry_price is None and len(data) > 0 and 'close' in data.columns:
            entry_price = data['close'].iloc[-1]

        if entry_price is None:
            logger.warning("No entry price available for exit signal calculation")
            self._record_latency(start_time, "exit_signal", "no_entry_price")
            return False

        if len(data) > 0 and 'close' in data.columns:
            current_price = data['close'].iloc[-1]
        else:
            logger.warning("No current price available for exit signal calculation")
            self._record_latency(start_time, "exit_signal", "no_current_price")
            return False

        # Exit if price retraces more than halfway back into the consolidation zone
        if all(k in position for k in ['breakout_level', 'consolidation_low', 'consolidation_high']):
            breakout_level = position['breakout_level']
            consolidation_low = position['consolidation_low']
            consolidation_high = position['consolidation_high']

            # Calculate retracement threshold
            if position.get('direction') == 'long':
                retracement_level = breakout_level - (breakout_level - consolidation_low) * 0.5
                if current_price < retracement_level:
                    self._emit_exit_event(data, position, "Price retraced into consolidation zone (long exit)")
                    self.log_signal('exit', data, "Price retraced into consolidation zone (long exit)")
                    self._record_latency(start_time, "exit_signal", "retracement")
                    return True
            elif position.get('direction') == 'short':
                retracement_level = breakout_level + (consolidation_high - breakout_level) * 0.5
                if current_price > retracement_level:
                    self._emit_exit_event(data, position, "Price retraced into consolidation zone (short exit)")
                    self.log_signal('exit', data, "Price retraced into consolidation zone (short exit)")
                    self._record_latency(start_time, "exit_signal", "retracement")
                    return True

        # Exit if volatility spikes unexpectedly
        if len(atr) > 5:
            avg_atr = atr.iloc[-5:].mean()
            if current_atr > avg_atr * 2:
                self._emit_exit_event(data, position, "Volatility spike detected")
                self.log_signal('exit', data, "Volatility spike detected")
                self._record_latency(start_time, "exit_signal", "volatility_spike")
                return True

        # Exit if price moves against the position significantly after entry
        if 'volume' in data.columns and len(data) >= 5:
            recent_volume_avg = data['volume'].iloc[-5:].mean()
            current_volume = data['volume'].iloc[-1]

            if position.get('direction') == 'long':
                price_change = (current_price - entry_price) / entry_price
                if price_change < -0.01 and current_volume > recent_volume_avg * 1.5:
                    self._emit_exit_event(data, position, "Significant volume on price decline")
                    self.log_signal('exit', data, "Significant volume on price decline")
                    self._record_latency(start_time, "exit_signal", "volume_decline")
                    return True
            elif position.get('direction') == 'short':
                price_change = (entry_price - current_price) / entry_price
                if price_change < -0.01 and current_volume > recent_volume_avg * 1.5:
                    self._emit_exit_event(data, position, "Significant volume on price increase")
                    self.log_signal('exit', data, "Significant volume on price increase")
                    self._record_latency(start_time, "exit_signal", "volume_increase")
                    return True

        self._record_latency(start_time, "exit_signal", "no_exit")
        return False

    def _emit_exit_event(self, data: pd.DataFrame, position: Dict[str, Any], reason: str) -> None:
        """
        Emit exit event to the event bus.

        Args:
            data: DataFrame with market data
            position: Position information
            reason: Exit reason
        """
        # Safely get current price
        current_price = None
        if len(data) > 0 and 'close' in data.columns:
            current_price = data['close'].iloc[-1]
        else:
            logger.warning("No current price available for exit event")
            current_price = 0

        # Safely get entry price
        entry_price = position.get('entry_price', current_price)
        if entry_price is None or entry_price == 0:
            entry_price = current_price

        # Calculate PnL safely
        pnl_pct = 0
        if entry_price and entry_price > 0 and current_price is not None:
            price_change = (current_price / entry_price) - 1
            direction_multiplier = 1 if position.get('direction', 'long') == 'long' else -1
            pnl_pct = price_change * direction_multiplier

        EventBus.emit("strategy.exit", {
            'strategy_id': self.id,
            'strategy_type': 'breakout',
            'timestamp': time.time(),
            'position_id': position.get('id', 'unknown'),
            'instrument': data.get('symbol', 'unknown'),
            'price': current_price,
            'reason': reason,
            'position_duration': time.time() - position.get('entry_time', time.time()),
            'pnl_pct': pnl_pct
        })

    def _detect_consolidation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect consolidation patterns in price data.

        Args:
            data: DataFrame with OHLC data

        Returns:
            Dict with consolidation information
        """
        # Look back period for consolidation detection
        period = self.parameters['consolidation_period']
        min_length = self.parameters['min_consolidation_length']
        max_width_pct = self.parameters['max_consolidation_width_pct']

        # Ensure we have enough data
        if data is None or len(data) < period:
            return {'is_valid': False}

        # Get recent price data (safely)
        try:
            recent_data = data.iloc[-period:]
        except IndexError:
            logger.warning(f"Index error when slicing data in _detect_consolidation. Data length: {len(data)}, Period: {period}")
            return {'is_valid': False}

        # Validate required columns
        if not all(col in recent_data.columns for col in ['high', 'low', 'close']):
            logger.warning("Missing required columns for consolidation detection")
            return {'is_valid': False}

        # Calculate price range
        high = recent_data['high'].max()
        low = recent_data['low'].min()

        # Guard against invalid range
        if high <= low:
            logger.warning(f"Invalid price range detected: high={high}, low={low}")
            return {'is_valid': False}

        # Calculate range width as percentage (safely)
        if low > 0:
            range_width_pct = (high - low) / low
        else:
            logger.warning(f"Invalid low price for percentage calculation: {low}")
            return {'is_valid': False}

        # Calculate average volume if available
        avg_volume = None
        if 'volume' in recent_data.columns:
            avg_volume = recent_data['volume'].mean()

        # Check if price is in consolidation
        if range_width_pct > max_width_pct:
            return {'is_valid': False}

        # Calculate touches of upper and lower bounds
        upper_bound_touches = sum(1 for h in recent_data['high'] if h > high * 0.98)
        lower_bound_touches = sum(1 for l in recent_data['low'] if l < low * 1.02)

        # Calculate consolidation quality
        total_touches = upper_bound_touches + lower_bound_touches
        if total_touches < 4:  # Need at least 4 touches for a valid range
            return {'is_valid': False}

        # Calculate the standard deviation of closes relative to the range (safely)
        range_size = high - low
        if range_size > 0:
            close_std = recent_data['close'].std() / range_size
        else:
            close_std = float('inf')

        # Calculate quality metric (lower std = higher quality)
        quality = 1.0 - min(1.0, close_std * 4)

        # Check overall validity
        is_valid = (
            range_width_pct <= max_width_pct and
            quality >= self._state['consolidation_quality_threshold'] and
            len(recent_data) >= min_length
        )

        # Store support and resistance levels
        if is_valid:
            timestamp = data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else time.time()

            self._state['support_resistance_levels'] = {
                'support': low,
                'resistance': high,
                'mid_point': (high + low) / 2,
                'width': range_size,
                'width_pct': range_width_pct,
                'timestamp': timestamp
            }

            # Add to detected patterns
            pattern = {
                'type': 'support_resistance',
                'support': low,
                'resistance': high,
                'quality': quality,
                'timestamp': timestamp
            }
            self._state['detected_patterns'].append(pattern)

            # Keep detected patterns list manageable
            if len(self._state['detected_patterns']) > 20:
                self._state['detected_patterns'] = self._state['detected_patterns'][-20:]

        return {
            'is_valid': is_valid,
            'support': low,
            'resistance': high,
            'mid_point': (high + low) / 2,
            'width': range_size,
            'width_pct': range_width_pct,
            'quality': quality,
            'avg_volume': avg_volume
        }

    def _detect_breakout(self, data: pd.DataFrame, consolidation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect breakouts from consolidation patterns.

        Args:
            data: DataFrame with OHLC data
            consolidation: Dict with consolidation information

        Returns:
            Dict with breakout information
        """
        if not consolidation['is_valid']:
            return {'is_breakout': False}

        # Guard against empty data
        if data is None or len(data) == 0:
            logger.warning("Empty data provided to _detect_breakout")
            return {'is_breakout': False}

        # Get the last few candles (safely)
        confirmation_candles = self.parameters['confirmation_candles']
        if len(data) >= confirmation_candles:
            recent_data = data.iloc[-confirmation_candles:]
        else:
            logger.warning(f"Insufficient data for breakout detection. Data length: {len(data)}, Required: {confirmation_candles}")
            return {'is_breakout': False}

        # Validate required columns
        if not all(col in recent_data.columns for col in ['close', 'high', 'low']):
            logger.warning("Missing required columns for breakout detection")
            return {'is_breakout': False}

        # Get price information
        close = recent_data['close'].iloc[-1]
        high = recent_data['high'].max()
        low = recent_data['low'].min()

        # Calculate breakout thresholds
        threshold_modifier = self._state['breakout_threshold_modifier'] * self.parameters['breakout_threshold']

        # If we have ATR data, use it for dynamic thresholds
        if 'atr_14' in data.columns and len(data) > 0:
            atr = data['atr_14'].iloc[-1]
            upward_threshold = consolidation['resistance'] + atr * threshold_modifier
            downward_threshold = consolidation['support'] - atr * threshold_modifier
        else:
            # Use fixed percentage of the consolidation width
            width = consolidation['width']
            upward_threshold = consolidation['resistance'] + width * threshold_modifier * 0.1
            downward_threshold = consolidation['support'] - width * threshold_modifier * 0.1

        # Check for breakout
        is_upward_breakout = high > upward_threshold and close > consolidation['resistance']
        is_downward_breakout = low < downward_threshold and close < consolidation['support']

        # Calculate breakout strength
        if is_upward_breakout:
            strength = (close - consolidation['resistance']) / max(consolidation['width'], 0.0001)  # Avoid division by zero
            return {
                'is_breakout': True,
                'direction': 'up',
                'strength': strength,
                'breakout_level': upward_threshold,
                'consolidation_low': consolidation['support'],
                'consolidation_high': consolidation['resistance'],
                'avg_volume': consolidation['avg_volume']
            }
        elif is_downward_breakout:
            strength = (consolidation['support'] - close) / max(consolidation['width'], 0.0001)  # Avoid division by zero
            return {
                'is_breakout': True,
                'direction': 'down',
                'strength': strength,
                'breakout_level': downward_threshold,
                'consolidation_low': consolidation['support'],
                'consolidation_high': consolidation['resistance'],
                'avg_volume': consolidation['avg_volume']
            }

        return {'is_breakout': False}