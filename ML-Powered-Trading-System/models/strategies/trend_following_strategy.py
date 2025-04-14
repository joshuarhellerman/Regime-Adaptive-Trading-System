"""
Trend Following Strategy Module

This module implements a trend following strategy that adapts to different market regimes.
The strategy uses moving averages, ADX, and volatility measures to identify and follow trends.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import logging
import time

from models.strategies.strategy_base import TradingStrategy, DirectionalBias, VolatilityRegime
from core.event_bus import EventBus
from core.performance_metrics import PerformanceMetrics

# Configure logger
logger = logging.getLogger(__name__)

class TrendFollowingStrategy(TradingStrategy):
    """
    Trend following strategy that adapts to market regimes.

    This strategy uses moving averages, ADX, and volatility-based indicators
    to identify and follow market trends with regime-specific optimizations.
    """

    def __init__(self, name: str = None, parameters: Dict[str, Any] = None, strategy_id: str = None):
        """
        Initialize the trend following strategy.

        Args:
            name: Strategy name
            parameters: Strategy-specific parameters
            strategy_id: Unique identifier for the strategy instance
        """
        # Default parameters
        default_params = {
            'fast_ma_period': 20,
            'slow_ma_period': 50,
            'adx_period': 14,
            'adx_threshold': 25,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'trend_filter_strength': 1.0,
            'volatility_filter': True,
            'use_ema': True,
            'min_bars': 50,  # Minimum bars needed for calculation
        }

        # Merge default parameters with provided parameters
        if parameters:
            for key, value in parameters.items():
                default_params[key] = value

        # Initialize the base class
        super().__init__(name or "TrendFollowing", default_params, strategy_id)

        # Strategy-specific attributes
        self._state = {
            'adaptive_lookbacks': (
                self.parameters['fast_ma_period'],
                self.parameters['slow_ma_period']
            ),
            'risk_multipliers': (1.5, 3.0),  # (stop loss, take profit) multipliers
            'last_signal_time': 0.0,
            'latency_metrics': [],
            'recent_signals': []
        }

        # Additional properties
        self.trend_filter_strength = self.parameters['trend_filter_strength']
        self.stop_loss_modifier = 1.0
        self.profit_target_modifier = 1.0
        self.account_balance = 10000.0  # Default value, should be updated from outside
        self.currency_pair = None  # Will be set when trading a specific pair
        self.original_generate_signal = self.generate_signal  # Store original function
        self.regime_characteristics = None  # Will hold current regime info

        logger.info(
            f"Initialized TrendFollowingStrategy with fast_ma={self.parameters['fast_ma_period']}, "
            f"slow_ma={self.parameters['slow_ma_period']}, adx_threshold={self.parameters['adx_threshold']}"
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
        # Validate moving average periods
        if self.parameters['fast_ma_period'] >= self.parameters['slow_ma_period']:
            raise ValueError("Fast MA period must be less than slow MA period")

        # Validate period values
        for param in ['fast_ma_period', 'slow_ma_period', 'adx_period', 'atr_period']:
            if self.parameters[param] <= 0:
                raise ValueError(f"Parameter {param} must be greater than 0")

        # Validate threshold values
        if self.parameters['adx_threshold'] <= 0:
            raise ValueError("ADX threshold must be greater than 0")

        # Validate multipliers
        if self.parameters['atr_multiplier'] <= 0:
            raise ValueError("ATR multiplier must be greater than 0")

    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate the input data for the strategy.

        Args:
            data: DataFrame with market data

        Raises:
            ValueError: If data validation fails
        """
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check for NaN values in key columns
        if data[required_columns].isna().any().any():
            raise ValueError("NaN values found in OHLCV data")

        # Check that data has enough rows
        if len(data) < self.parameters['min_bars']:
            raise ValueError(f"Insufficient data: {len(data)} bars (need {self.parameters['min_bars']})")

    def generate_signal(self, data: pd.DataFrame) -> Optional[str]:
        """
        Generate a trend following signal.

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

        # Use adjusted lookbacks based on regime optimization
        fast_ma_period, slow_ma_period = self._state['adaptive_lookbacks']

        # Calculate moving averages
        if self.parameters['use_ema']:
            ma_fast = data['close'].ewm(span=fast_ma_period, adjust=False).mean()
            ma_slow = data['close'].ewm(span=slow_ma_period, adjust=False).mean()
        else:
            ma_fast = data['close'].rolling(fast_ma_period).mean()
            ma_slow = data['close'].rolling(slow_ma_period).mean()

        # Calculate ADX if not in data
        if 'ADX' not in data.columns and 'adx_14' not in data.columns:
            adx = self._calculate_adx(data, self.parameters['adx_period'])
        else:
            adx = data['ADX'] if 'ADX' in data.columns else data['adx_14']

        # Calculate ATR if not in data
        if 'atr_14' not in data.columns:
            atr = self._calculate_atr(data, self.parameters['atr_period'])
        else:
            atr = data['atr_14']

        # Apply trend filter strength modifier
        adx_threshold = self.parameters['adx_threshold'] * self.trend_filter_strength

        # Check for trend strength
        trend_strength = adx.iloc[-1] >= adx_threshold

        # Apply volatility filter if enabled
        use_volatility_filter = self.parameters['volatility_filter']
        volatility_filtered = True
        vol_ratio = 1.0

        if use_volatility_filter:
            # Calculate volatility ratio
            if len(atr) > 20:
                current_atr = atr.iloc[-1]
                avg_atr = atr.iloc[-20:].mean()

                # Only allow signals when volatility is in a reasonable range
                vol_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
                volatility_filtered = 0.5 <= vol_ratio <= 2.0

            if not volatility_filtered:
                logger.debug(f"Volatility filter blocked signal generation: ratio={vol_ratio:.2f}")
                self._record_latency(start_time, "volatility_filtered")
                return None

        # Get trend direction
        trend_up = ma_fast.iloc[-1] > ma_slow.iloc[-1]
        trend_down = ma_fast.iloc[-1] < ma_slow.iloc[-1]

        # Detect crossovers
        prev_fast = ma_fast.iloc[-2]
        prev_slow = ma_slow.iloc[-2]

        # Check for crossover
        crossover_up = prev_fast <= prev_slow and ma_fast.iloc[-1] > ma_slow.iloc[-1]
        crossover_down = prev_fast >= prev_slow and ma_fast.iloc[-1] < ma_slow.iloc[-1]

        # Generate signal with trend strength filter
        if trend_strength:
            if crossover_up:
                signal = 'long'
                self.log_signal('long', data, "MA crossover up with strong trend")
                self._emit_signal_event(signal, data, {
                    'crossover': True,
                    'adx': float(adx.iloc[-1]),
                    'ma_fast': float(ma_fast.iloc[-1]),
                    'ma_slow': float(ma_slow.iloc[-1]),
                    'trend_strength': 'strong'
                })
                self._record_latency(start_time, "signal_generation", signal)
                return signal
            elif crossover_down:
                signal = 'short'
                self.log_signal('short', data, "MA crossover down with strong trend")
                self._emit_signal_event(signal, data, {
                    'crossover': True,
                    'adx': float(adx.iloc[-1]),
                    'ma_fast': float(ma_fast.iloc[-1]),
                    'ma_slow': float(ma_slow.iloc[-1]),
                    'trend_strength': 'strong'
                })
                self._record_latency(start_time, "signal_generation", signal)
                return signal

        # If no crossover, check for existing trend
        else:
            # If strong ADX and clear trend, give signal in trend direction
            if adx.iloc[-1] >= adx_threshold * 1.5:  # Higher threshold for trend continuation
                if trend_up:
                    signal = 'long'
                    self.log_signal('long', data, "Strong uptrend continuation")
                    self._emit_signal_event(signal, data, {
                        'crossover': False,
                        'adx': float(adx.iloc[-1]),
                        'ma_fast': float(ma_fast.iloc[-1]),
                        'ma_slow': float(ma_slow.iloc[-1]),
                        'trend_strength': 'very_strong'
                    })
                    self._record_latency(start_time, "signal_generation", signal)
                    return signal
                elif trend_down:
                    signal = 'short'
                    self.log_signal('short', data, "Strong downtrend continuation")
                    self._emit_signal_event(signal, data, {
                        'crossover': False,
                        'adx': float(adx.iloc[-1]),
                        'ma_fast': float(ma_fast.iloc[-1]),
                        'ma_slow': float(ma_slow.iloc[-1]),
                        'trend_strength': 'very_strong'
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

        # Keep only the last 100 metrics
        if len(self._state['latency_metrics']) > 100:
            self._state['latency_metrics'] = self._state['latency_metrics'][-100:]

        # Report latency metrics
        PerformanceMetrics.record(
            f"strategy.{self.id}.latency",
            latency,
            tags={'stage': stage, 'signal': signal or 'none'}
        )

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

        # Store original parameters for tracking changes
        old_fast, old_slow = self._state['adaptive_lookbacks']
        old_stop, old_target = self._state['risk_multipliers']

        if successful:
            # If trade was successful, fine-tune the strategy
            if regime_id is not None:  # Check if we have valid regime information
                # Slightly reduce filter strength to get more trades in successful regimes
                self.trend_filter_strength = max(0.8, self.trend_filter_strength * 0.95)

                # Increase profit target if the trade was very successful
                if pnl_pct > 0.05:  # 5% profit
                    self._state['risk_multipliers'] = (
                        old_stop,
                        min(5.0, old_target * 1.05)  # Increase take profit, cap at 5.0
                    )
        else:
            # If trade was unsuccessful, slightly adjust parameters
            if regime_id is not None:  # Check if we have valid regime information
                # Increase filter strength to get fewer but better trades
                self.trend_filter_strength = min(1.5, self.trend_filter_strength * 1.05)

                # Tighten stop loss if the loss was large
                if pnl_pct < -0.03:  # 3% loss
                    self._state['risk_multipliers'] = (
                        max(1.0, old_stop * 0.95),  # Decrease stop loss, floor at 1.0
                        old_target
                    )

        # If parameters changed, emit the event
        if (old_fast, old_slow) != self._state['adaptive_lookbacks'] or \
                (old_stop, old_target) != self._state['risk_multipliers'] or \
                self.trend_filter_strength != 1.0:  # Only emit if different from default

            self._emit_parameter_adaptation_event(
                trade_result,
                {
                    'trend_filter_strength': {
                        'old': self.trend_filter_strength / 0.95 if successful else self.trend_filter_strength / 1.05,
                        'new': self.trend_filter_strength
                    },
                    'risk_multipliers': {
                        'old': (old_stop, old_target),
                        'new': self._state['risk_multipliers']
                    }
                }
            )

        logger.debug(
            f"Adapted trend following parameters after trade: "
            f"filter_strength={self.trend_filter_strength:.2f}, "
            f"risk_multipliers={self._state['risk_multipliers']}"
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
            'strategy_type': 'trend_following',
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

        # Add indicator features
        features.add('adx_14')
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
        old_fast, old_slow = self._state['adaptive_lookbacks']
        old_stop, old_target = self._state['risk_multipliers']
        old_filter_strength = self.trend_filter_strength

        # Update MA periods based on market conditions
        if adx > 30:  # Strong trend
            # In strong trends, shorter lookbacks might be better
            new_fast = int(old_fast * 0.95)
            new_slow = int(old_slow * 0.95)
            self._state['adaptive_lookbacks'] = (new_fast, new_slow)
        elif adx < 15:  # Weak trend
            # In weak trends, longer lookbacks might be better
            new_fast = int(old_fast * 1.05)
            new_slow = int(old_slow * 1.05)
            self._state['adaptive_lookbacks'] = (new_fast, new_slow)

        # Update trend filter strength based on win rate
        if win_rate > 0.6:
            # If win rate is high, we can be less strict with filter
            new_strength = self.trend_filter_strength * 0.98
            self.trend_filter_strength = max(0.7, new_strength)
        elif win_rate < 0.4:
            # If win rate is low, be more strict with filter
            new_strength = self.trend_filter_strength * 1.02
            self.trend_filter_strength = min(1.5, new_strength)

        # Update risk parameters based on volatility and Sharpe
        if volatility > 0.7:  # High volatility
            self._state['risk_multipliers'] = (
                min(3.0, old_stop * 1.02),  # Wider stops
                min(5.0, old_target * 1.02)  # Wider targets
            )
        elif volatility < 0.3:  # Low volatility
            self._state['risk_multipliers'] = (
                max(1.0, old_stop * 0.98),  # Tighter stops
                max(2.0, old_target * 0.98)  # Tighter targets
            )

        # If Sharpe ratio is low, adjust risk/reward
        if sharpe_ratio < 0.5:
            self._state['risk_multipliers'] = (
                old_stop,
                max(old_target * 1.05, old_stop * 2.5)  # Ensure reward at least 2.5x risk
            )

        # Log and emit parameter updates if any changed
        if ((old_fast, old_slow) != self._state['adaptive_lookbacks'] or
                (old_stop, old_target) != self._state['risk_multipliers'] or
                old_filter_strength != self.trend_filter_strength):
            logger.info(f"Online parameter update: lookbacks={self._state['adaptive_lookbacks']}, "
                        f"risk_multipliers={self._state['risk_multipliers']}, "
                        f"trend_filter={self.trend_filter_strength:.2f}")

            EventBus.emit("strategy.online_update", {
                'strategy_id': self.id,
                'strategy_type': 'trend_following',
                'timestamp': time.time(),
                'new_parameters': {
                    'adaptive_lookbacks': self._state['adaptive_lookbacks'],
                    'risk_multipliers': self._state['risk_multipliers'],
                    'trend_filter_strength': self.trend_filter_strength
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
                "TrendFollowingStrategy",
                description="Identifies and follows market trends using moving averages and ADX",
                default_parameters={
                    'fast_ma_period': 20,
                    'slow_ma_period': 50,
                    'adx_threshold': 25,
                    'use_ema': True
                },
                compatibility={
                    'min_bars': 50,
                    'timeframes': ['1h', '4h', 'daily', 'weekly'],
                    'asset_classes': ['forex', 'crypto', 'stocks', 'commodities', 'indices']
                }
            )
            logger.info("TrendFollowingStrategy registered successfully")
        except ImportError:
            logger.warning("StrategyRegistry not available, skip registration")
        except Exception as e:
            logger.error(f"Failed to register TrendFollowingStrategy: {str(e)}")

    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).

        Args:
            data: DataFrame with OHLC data
            period: ADX calculation period

        Returns:
            Series with ADX values
        """
        high = data['high']
        low = data['low']
        close = data['close']

        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate directional movement
        up_move = high - high.shift()
        down_move = low.shift() - low

        # Positive and negative directional movement
        pdm = up_move.copy()
        pdm[pdm < 0] = 0
        pdm[up_move < down_move] = 0

        ndm = down_move.copy()
        ndm[ndm < 0] = 0
        ndm[down_move < up_move] = 0

        # Smooth directional movement and true range
        tr_smoothed = tr.ewm(alpha=1 / period, adjust=False).mean()
        pdm_smoothed = pdm.ewm(alpha=1 / period, adjust=False).mean()
        ndm_smoothed = ndm.ewm(alpha=1 / period, adjust=False).mean()

        # Calculate directional indices
        pdi = 100 * pdm_smoothed / tr_smoothed
        ndi = 100 * ndm_smoothed / tr_smoothed

        # Calculate ADX
        dx = 100 * abs(pdi - ndi) / (pdi + ndi)
        dx = dx.replace([np.inf, -np.inf], np.nan).fillna(0)
        adx = dx.ewm(alpha=1 / period, adjust=False).mean()

        return adx

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).

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

    def dynamic_position_size(self, account_balance: float) -> float:
        """
        Calculate dynamic position size based on account balance and market conditions.

        Args:
            account_balance: Current account balance

        Returns:
            float: Position size as a percentage of account balance
        """
        # Base position size (2% of account)
        base_size = 0.02

        # Adjust based on trend filter strength
        if self.trend_filter_strength > 1.2:
            # More conservative when filters are strict (fewer but higher quality trades)
            size_modifier = 1.2
        elif self.trend_filter_strength < 0.8:
            # More aggressive when filters are relaxed (more signals)
            size_modifier = 0.8
        else:
            size_modifier = 1.0

        # Adjust based on volatility
        if hasattr(self, 'regime_characteristics') and self.regime_characteristics:
            vol_regime = self.regime_characteristics.get('volatility_regime', VolatilityRegime.NORMAL)
            if vol_regime == VolatilityRegime.HIGH:
                # Reduce size in high volatility
                vol_modifier = 0.8
            elif vol_regime == VolatilityRegime.LOW:
                # Increase size in low volatility
                vol_modifier = 1.2
            else:
                vol_modifier = 1.0
        else:
            vol_modifier = 1.0

        # Calculate final position size
        position_size = base_size * size_modifier * vol_modifier

        # Cap at 5% of account
        return min(0.05, position_size)

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

        # Call parent method if it exists
        if hasattr(super(), 'exit_signal'):
            super().exit_signal(data, position)

        # Exit if trend reverses
        fast_ma_period, slow_ma_period = self._state['adaptive_lookbacks']

        # Calculate moving averages
        if self.parameters['use_ema']:
            ma_fast = data['close'].ewm(span=fast_ma_period, adjust=False).mean()
            ma_slow = data['close'].ewm(span=slow_ma_period, adjust=False).mean()
        else:
            ma_fast = data['close'].rolling(fast_ma_period).mean()
            ma_slow = data['close'].rolling(slow_ma_period).mean()

        # Check if trend has reversed
        if position['direction'] == 'long' and ma_fast.iloc[-1] < ma_slow.iloc[-1]:
            reason = "MA crossover down (exit long)"
            self._emit_exit_event(data, position, reason)
            self.log_signal('exit', data, reason)
            self._record_latency(start_time, "exit_signal", "ma_cross_exit")
            return True
        elif position['direction'] == 'short' and ma_fast.iloc[-1] > ma_slow.iloc[-1]:
            reason = "MA crossover up (exit short)"
            self._emit_exit_event(data, position, reason)
            self.log_signal('exit', data, reason)
            self._record_latency(start_time, "exit_signal", "ma_cross_exit")
            return True

        # Check if ADX has significantly weakened
        if 'ADX' in data.columns or 'adx_14' in data.columns:
            adx = data['ADX'] if 'ADX' in data.columns else data['adx_14']

            # If ADX is weak and dropping
            if (adx.iloc[-1] < self.parameters['adx_threshold'] * 0.8 and
                    adx.iloc[-1] < adx.iloc[-2]):
                reason = "ADX weakening (exit trend)"
                self._emit_exit_event(data, position, reason)
                self.log_signal('exit', data, reason)
                self._record_latency(start_time, "exit_signal", "adx_weakening")
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
        EventBus.emit("strategy.exit", {
            'strategy_id': self.id,
            'strategy_type': 'trend_following',
            'timestamp': time.time(),
            'position_id': position.get('id', 'unknown'),
            'instrument': data.get('symbol', 'unknown'),
            'price': data['close'].iloc[-1],
            'reason': reason,
            'position_duration': time.time() - position.get('entry_time', time.time()),
            'pnl_pct': (data['close'].iloc[-1] / position.get('entry_price', data['close'].iloc[-1]) - 1) *
                       (1 if position.get('direction', 'long') == 'long' else -1)
        })

    def risk_parameters(self, data: pd.DataFrame, entry_price: float) -> Dict[str, float]:
        """
        Calculate risk parameters for the trend following strategy.

        Args:
            data: DataFrame with market data
            entry_price: Entry price for the trade

        Returns:
            Dict with stop loss, take profit, and position size
        """
        start_time = time.time()

        # Calculate ATR if not available in data
        if 'atr_14' not in data.columns:
            atr = self._calculate_atr(data, self.parameters['atr_period']).iloc[-1]
        else:
            atr = data['atr_14'].iloc[-1]

        # Apply volatility adjustment based on regime
        volatility_factor = 1.0
        if hasattr(self, 'regime_characteristics') and self.regime_characteristics:
            volatility_zscore = self.regime_characteristics.get('volatility_zscore', 0)
            volatility_factor = 1.0 + volatility_zscore * 0.2

        # Calculate risk parameters
        stop_loss_pct = (atr * self._state['risk_multipliers'][0] *
                         volatility_factor * self.stop_loss_modifier) / entry_price
        take_profit_pct = (atr * self._state['risk_multipliers'][1] *
                           volatility_factor * self.profit_target_modifier) / entry_price

        # Calculate position size
        position_size = self.dynamic_position_size(self.account_balance)

        # Emit risk parameters event
        self._emit_risk_event(entry_price, stop_loss_pct, take_profit_pct, position_size, volatility_factor)

        # Record performance
        self._record_latency(start_time, "risk_parameters")

        return {
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'position_size': position_size,
            'stop_price': entry_price * (1 - stop_loss_pct) if entry_price > 0 else 0,
            'take_profit_price': entry_price * (1 + take_profit_pct) if entry_price > 0 else 0,
            'atr': atr,
            'volatility_factor': volatility_factor
        }

    def _emit_risk_event(self, entry_price: float, stop_loss_pct: float,
                         take_profit_pct: float, position_size: float,
                         volatility_factor: float) -> None:
        """
        Emit risk parameters event to the event bus.

        Args:
            entry_price: Entry price
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            position_size: Position size
            volatility_factor: Volatility adjustment factor
        """
        EventBus.emit("strategy.risk_parameters", {
            'strategy_id': self.id,
            'strategy_type': 'trend_following',
            'timestamp': time.time(),
            'entry_price': entry_price,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'position_size': position_size,
            'risk_reward_ratio': take_profit_pct / stop_loss_pct if stop_loss_pct > 0 else 0,
            'volatility_factor': volatility_factor
        })

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

        # Keep only recent signals
        if len(self._state['recent_signals']) > 20:
            self._state['recent_signals'] = self._state['recent_signals'][-20:]

        # Calculate confidence based on ADX strength and crossover
        adx_value = metadata.get('adx', self.parameters['adx_threshold'])
        is_crossover = metadata.get('crossover', False)

        # Higher ADX gives more confidence, with diminishing returns above 35
        adx_confidence = min(1.0, (adx_value - self.parameters['adx_threshold']) / 20)

        # Crossovers generally give higher confidence signals
        crossover_factor = 1.2 if is_crossover else 1.0

        # Calculate final confidence (capped at 1.0)
        confidence = min(1.0, 0.5 + adx_confidence * crossover_factor)

        EventBus.emit("strategy.signal", {
            'strategy_id': self.id,
            'strategy_type': 'trend_following',
            'timestamp': time.time(),
            'signal': signal,
            'instrument': data.get('symbol', 'unknown'),
            'price': data['close'].iloc[-1],
            'confidence': confidence,
            'metadata': metadata
        })

    def cluster_fit(self, cluster_metrics: Dict[str, float]) -> float:
        """
        Determine if the trend following strategy is suitable for the given cluster.

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

        # Trend following strategy works best in trending markets with moderate to high
        # ADX, strong trend strength, and sufficient directional bias

        # ADX component (Trend strength)
        adx_mean = cluster_metrics.get('ADX_mean', 20)
        if adx_mean > 15:
            # Higher ADX is better for trend following, up to a point (40+)
            adx_score = min(1.0, adx_mean / 40) if adx_mean <= 40 else 1.0
        else:
            # Below ADX 15 is poor for trend following
            adx_score = max(0, adx_mean / 15)

        # Trend strength component
        trend_strength = cluster_metrics.get('trend_strength', 0.5)
        if trend_strength > 0.4:
            # Higher trend strength is better
            trend_score = min(1.0, trend_strength * 1.5)
        else:
            # Low trend strength is poor for trend following
            trend_score = max(0, trend_strength / 0.4)

        # Directional bias component
        directional_bias = cluster_metrics.get('directional_bias', 'neutral')
        if directional_bias != 'neutral':
            # Any directional bias is good for trend following
            bias_score = 1.0
        else:
            # Neutral bias is less ideal but still workable
            bias_score = 0.5

        # Autocorrelation component
        autocorrelation = cluster_metrics.get('autocorrelation', 0)
        if autocorrelation > 0:
            # Positive autocorrelation is ideal for trend following
            autocorr_score = min(1.0, 0.5 + autocorrelation)
        else:
            # Negative autocorrelation is poor for trend following
            autocorr_score = max(0, 0.5 + autocorrelation)

        # Weighted combination
        weights = {
            'adx': 0.3,
            'trend': 0.3,
            'bias': 0.2,
            'autocorr': 0.2
        }

        fitness_score = (
                weights['adx'] * adx_score +
                weights['trend'] * trend_score +
                weights['bias'] * bias_score +
                weights['autocorr'] * autocorr_score
        )

        # Log detailed scores for analysis
        logger.debug(f"Trend following cluster fit - ADX: {adx_score:.2f}, Trend: {trend_score:.2f}, "
                     f"Bias: {bias_score:.2f}, Autocorr: {autocorr_score:.2f}, Total: {fitness_score:.2f}")

        # Emit event with fitness score
        EventBus.emit("strategy.cluster_fit", {
            'strategy_id': self.id,
            'strategy_type': 'trend_following',
            'cluster_id': cluster_metrics.get('cluster_id', 'unknown'),
            'fitness_score': fitness_score,
            'components': {
                'adx_score': adx_score,
                'trend_score': trend_score,
                'bias_score': bias_score,
                'autocorr_score': autocorr_score
            }
        })

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
            f"Adapting trend following strategy to regime {regime_id}: bias={directional_bias}, volatility={volatility_regime}")

        # Store original parameters for tracking changes
        old_lookbacks = self._state['adaptive_lookbacks']
        old_risk_multipliers = self._state['risk_multipliers']

        # Store regime characteristics
        self.regime_characteristics = regime_data

        # Apply optimization based on time of day
        if peak_hour == 11:
            # Faster reaction for volatile morning session
            self._state['adaptive_lookbacks'] = (
                int(self.parameters['fast_ma_period'] * 0.8),
                int(self.parameters['slow_ma_period'] * 0.8)
            )
            self._state['risk_multipliers'] = (1.8, 3.5)
            logger.info(f"Applied 11:00 UTC trend following optimizations")
        elif peak_hour in [8, 9]:
            # Longer lookbacks for more stable morning session
            self._state['adaptive_lookbacks'] = (
                int(self.parameters['fast_ma_period'] * 1.2),
                int(self.parameters['slow_ma_period'] * 1.2)
            )
            self._state['risk_multipliers'] = (1.3, 2.8)
            logger.info(f"Applied 08:00/09:00 UTC trend following optimizations")

        # Apply optimization based on directional bias
        if directional_bias == DirectionalBias.UPWARD:
            # Create biased signal generator for upward bias
            def biased_signal(data: pd.DataFrame) -> Optional[str]:
                signal = self.original_generate_signal(data)

                # Prefer long signals
                if signal == 'long':
                    return 'long'
                elif signal == 'short':
                    # Additional confirmation for shorts in upward bias regime
                    adx = data['ADX'].iloc[-1] if 'ADX' in data.columns else (
                        data['adx_14'].iloc[-1] if 'adx_14' in data.columns else
                        self._calculate_adx(data, self.parameters['adx_period']).iloc[-1]
                    )
                    # Require stronger trend confirmation for shorts
                    if adx > self.parameters['adx_threshold'] * 1.2:
                        return 'short'
                    return None

                return signal

            self.generate_signal = biased_signal
            logger.info(f"Applied upward bias signal optimization to {self.name}")

        elif directional_bias == DirectionalBias.DOWNWARD:
            # Create biased signal generator for downward bias
            def biased_signal(data: pd.DataFrame) -> Optional[str]:
                signal = self.original_generate_signal(data)

                # Prefer short signals
                if signal == 'short':
                    return 'short'
                elif signal == 'long':
                    # Additional confirmation for longs in downward bias regime
                    adx = data['ADX'].iloc[-1] if 'ADX' in data.columns else (
                        data['adx_14'].iloc[-1] if 'adx_14' in data.columns else
                        self._calculate_adx(data, self.parameters['adx_period']).iloc[-1]
                    )
                    # Require stronger trend confirmation for longs
                    if adx > self.parameters['adx_threshold'] * 1.2:
                        return 'long'
                    return None

                return signal

            self.generate_signal = biased_signal
            logger.info(f"Applied downward bias signal optimization to {self.name}")
        else:
            # Reset to original signal generator for neutral bias
            self.generate_signal = self.original_generate_signal
            logger.info(f"Reset to neutral bias signal generation for {self.name}")

        # Apply optimization based on volatility regime
        if volatility_regime == VolatilityRegime.HIGH:
            # Faster lookbacks and wider stops for high volatility
            current_fast, current_slow = self._state['adaptive_lookbacks']
            self._state['adaptive_lookbacks'] = (
                int(current_fast * 0.9),
                int(current_slow * 0.9)
            )
            current_stop, current_target = self._state['risk_multipliers']
            self._state['risk_multipliers'] = (current_stop * 1.2, current_target * 1.1)
            logger.info(f"Applied high volatility trend following optimizations")
        elif volatility_regime == VolatilityRegime.LOW:
            # Slower lookbacks and tighter stops for low volatility
            current_fast, current_slow = self._state['adaptive_lookbacks']
            self._state['adaptive_lookbacks'] = (
                int(current_fast * 1.1),
                int(current_slow * 1.1)
            )
            current_stop, current_target = self._state['risk_multipliers']
            self._state['risk_multipliers'] = (current_stop * 0.9, current_target * 0.9)
            logger.info(f"Applied low volatility trend following optimizations")

        # Currency-pair specific optimizations
        if self.currency_pair is not None:
            currency_pair = self.currency_pair

            if currency_pair == 'GBPUSD':
                # Add volatility filter for GBPUSD
                original_generate = self.generate_signal

                def volatility_filtered_signal(data: pd.DataFrame) -> Optional[str]:
                    # Only generate signals in optimal volatility conditions
                    if 'atr_14' in data.columns:
                        atr = data['atr_14']
                        avg_atr = atr.rolling(50).mean().iloc[-1]
                        vol_ratio = atr.iloc[-1] / avg_atr if avg_atr > 0 else 1.0

                        if 0.8 <= vol_ratio <= 1.5:  # Optimal volatility range
                            return original_generate(data)
                        return None
                    else:
                        return original_generate(data)

                self.generate_signal = volatility_filtered_signal
                logger.info(f"Applied GBPUSD-specific trend following optimization")

        # Emit regime adaptation event
        EventBus.emit("strategy.regime_adaptation", {
            'strategy_id': self.id,
            'strategy_type': 'trend_following',
            'timestamp': time.time(),
            'regime_id': regime_id,
            'parameters': {
                'adaptive_lookbacks': {
                    'old': old_lookbacks,
                    'new': self._state['adaptive_lookbacks']
                },
                'risk_multipliers': {
                    'old': old_risk_multipliers,
                    'new': self._state['risk_multipliers']
                }
            }
        })