"""
Volatility Strategy Module

This module implements a volatility-based strategy that adapts to different market regimes.
The strategy captures price movements during periods of changing volatility, using techniques
like Bollinger Band breakouts, ATR expansions, and volatility breakout patterns.
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

class VolatilityStrategy(TradingStrategy):
    """
    Volatility strategy that adapts to market regimes.

    This strategy identifies periods of volatility expansion and contraction,
    trading breakouts after volatility compression and mean reversion after
    volatility expansion, with regime-specific adaptations.
    """

    def __init__(self, name: str = None, parameters: Dict[str, Any] = None, strategy_id: str = None):
        """
        Initialize the volatility strategy.

        Args:
            name: Strategy name
            parameters: Strategy-specific parameters
            strategy_id: Unique identifier for the strategy instance
        """
        # Default parameters
        default_params = {
            'bb_period': 20,
            'bb_std': 2.0,
            'atr_period': 14,
            'atr_lookback': 100,
            'atr_threshold': 1.5,
            'volatility_period': 20,
            'expansion_threshold': 1.4,
            'contraction_threshold': 0.6,
            'min_volatility_percentile': 0.2,
            'max_volatility_percentile': 0.8,
            'use_bbands': True,
            'use_atr': True,
            'volume_filter': True,
            'min_bars': 50  # Minimum bars needed for calculation
        }

        # Merge default parameters with provided parameters
        if parameters:
            for key, value in parameters.items():
                default_params[key] = value

        # Initialize the base class
        super().__init__(name or "Volatility", default_params, strategy_id)

        # Strategy-specific attributes
        self._state = {
            'volatility_state': "normal",  # "expansion", "contraction", or "normal"
            'volatility_trend': "neutral",  # "increasing", "decreasing", or "neutral"
            'volatility_threshold_modifier': 1.0,
            'trailing_stop_multiplier': 2.0,
            'volatility_history': [],
            'last_signal_time': 0.0,
            'latency_metrics': [],
            'recent_signals': []
        }

        logger.info(
            f"Initialized VolatilityStrategy with bb_period={self.parameters['bb_period']}, "
            f"atr_period={self.parameters['atr_period']}, "
            f"use_bbands={self.parameters['use_bbands']}, use_atr={self.parameters['use_atr']}"
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
        for param in ['bb_period', 'atr_period', 'atr_lookback', 'volatility_period']:
            if self.parameters[param] <= 0:
                raise ValueError(f"Parameter {param} must be greater than 0")

        # Validate threshold values
        if self.parameters['bb_std'] <= 0:
            raise ValueError("Bollinger Band standard deviation must be greater than 0")

        if self.parameters['atr_threshold'] <= 0:
            raise ValueError("ATR threshold must be greater than 0")

        if self.parameters['expansion_threshold'] <= 0:
            raise ValueError("Expansion threshold must be greater than 0")

        if self.parameters['contraction_threshold'] <= 0 or self.parameters['contraction_threshold'] >= 1.0:
            raise ValueError("Contraction threshold must be between 0 and 1")

        if not 0 <= self.parameters['min_volatility_percentile'] < self.parameters['max_volatility_percentile'] <= 1.0:
            raise ValueError("Volatility percentiles must satisfy: 0 <= min < max <= 1")

    def generate_signal(self, data: pd.DataFrame) -> Optional[str]:
        """
        Generate a volatility-based signal.

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

        # Calculate volatility indicators
        indicators = {}
        
        if self.parameters['use_bbands']:
            if 'bb_upper' not in data.columns or 'bb_lower' not in data.columns:
                indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = self._calculate_bollinger_bands(
                    data['close'],
                    self.parameters['bb_period'],
                    self.parameters['bb_std']
                )
            else:
                indicators['bb_upper'] = data['bb_upper']
                indicators['bb_middle'] = data['bb_middle'] if 'bb_middle' in data.columns else (data['bb_upper'] + data['bb_lower']) / 2
                indicators['bb_lower'] = data['bb_lower']

        if self.parameters['use_atr']:
            if 'atr_14' not in data.columns:
                indicators['atr'] = self._calculate_atr(data, self.parameters['atr_period'])
            else:
                indicators['atr'] = data['atr_14']

        # Analyze volatility state
        self._update_volatility_state(data, indicators.get('atr'))

        # Get the current market condition and price
        close = data['close'].iloc[-1]
        prev_close = data['close'].iloc[-2] if len(data) > 1 else close

        # Generate signals based on volatility state and indicators
        signal = None

        # Case 1: Volatility Contraction - Breakout Strategy
        if self._state['volatility_state'] == "contraction":
            if self.parameters['use_bbands']:
                # Bollinger Band breakout after contraction
                bb_upper = indicators['bb_upper']
                bb_lower = indicators['bb_lower']
                bb_middle = indicators['bb_middle']
                
                bb_width = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1]
                prev_bb_width = (bb_upper.iloc[-2] - bb_lower.iloc[-2]) / bb_middle.iloc[-2]

                # Detect squeezed Bollinger Bands followed by a breakout
                expansion_threshold = self.parameters['expansion_threshold'] * self._state['volatility_threshold_modifier']
                if bb_width > prev_bb_width * expansion_threshold:
                    # Direction of breakout
                    if close > bb_upper.iloc[-1]:
                        signal = "long"
                        reason = f"Upward breakout after volatility contraction: BB width expansion {bb_width/prev_bb_width:.2f}x"
                        self.log_signal('long', data, reason)
                    elif close < bb_lower.iloc[-1]:
                        signal = "short"
                        reason = f"Downward breakout after volatility contraction: BB width expansion {bb_width/prev_bb_width:.2f}x"
                        self.log_signal('short', data, reason)

            # Confirm with ATR if configured
            if signal and self.parameters['use_atr']:
                atr = indicators['atr']
                atr_increase = atr.iloc[-1] / atr.iloc[-5:].mean()
                if atr_increase < 1.2:  # Not enough volatility increase
                    signal = None
                    logger.debug(f"Signal canceled: insufficient ATR increase ({atr_increase:.2f}x)")

        # Case 2: Volatility Expansion - Reversal Strategy
        elif self._state['volatility_state'] == "expansion":
            if self.parameters['use_bbands']:
                bb_upper = indicators['bb_upper']
                bb_lower = indicators['bb_lower']
                
                # Look for extreme moves that are likely to reverse
                if close > bb_upper.iloc[-1] * 1.05:  # Significant move above upper band
                    signal = "short"
                    reason = f"Reversal after extreme upward volatility expansion"
                    self.log_signal('short', data, reason)
                elif close < bb_lower.iloc[-1] * 0.95:  # Significant move below lower band
                    signal = "long"
                    reason = f"Reversal after extreme downward volatility expansion"
                    self.log_signal('long', data, reason)

            # Alternative signal using ATR
            if not signal and self.parameters['use_atr']:
                atr = indicators['atr']
                if atr.iloc[-1] > atr.iloc[-20:].mean() * self.parameters['atr_threshold']:
                    # Check for a potential reversal candle
                    if data['high'].iloc[-1] > data['high'].iloc[-5:].max() * 1.01 and close < data['open'].iloc[-1]:
                        signal = "short"
                        reason = f"Reversal after high volatility expansion"
                        self.log_signal('short', data, reason)
                    elif data['low'].iloc[-1] < data['low'].iloc[-5:].min() * 0.99 and close > data['open'].iloc[-1]:
                        signal = "long"
                        reason = f"Reversal after low volatility expansion"
                        self.log_signal('long', data, reason)

        # Case 3: Normal Volatility - Range/Trend Following
        else:  # normal volatility state
            if self.parameters['use_bbands']:
                bb_upper = indicators['bb_upper']
                bb_lower = indicators['bb_lower']
                bb_middle = indicators['bb_middle']
                
                # Range-based strategy during normal volatility
                if self._state['volatility_trend'] == "decreasing":
                    # Approaching contraction - prepare for breakout
                    if close > bb_middle.iloc[-1] and close > prev_close:
                        signal = "long"
                        reason = f"Long bias ahead of expected volatility contraction"
                        self.log_signal('long', data, reason)
                    elif close < bb_middle.iloc[-1] and close < prev_close:
                        signal = "short"
                        reason = f"Short bias ahead of expected volatility contraction"
                        self.log_signal('short', data, reason)
                elif self._state['volatility_trend'] == "increasing":
                    # Approaching expansion - prepare for reversal
                    distance_to_upper = (bb_upper.iloc[-1] - close) / close
                    distance_to_lower = (close - bb_lower.iloc[-1]) / close

                    if distance_to_upper < 0.005:  # Close to upper band
                        signal = "short"
                        reason = f"Short ahead of expected volatility expansion near upper band"
                        self.log_signal('short', data, reason)
                    elif distance_to_lower < 0.005:  # Close to lower band
                        signal = "long"
                        reason = f"Long ahead of expected volatility expansion near lower band"
                        self.log_signal('long', data, reason)

        # Apply volume filter if configured
        if signal and self.parameters['volume_filter'] and 'volume' in data.columns:
            recent_volume = data['volume'].iloc[-1]
            avg_volume = data['volume'].iloc[-20:].mean()

            # Ensure sufficient liquidity
            if recent_volume < avg_volume * 0.8:
                logger.debug(f"Signal canceled: insufficient volume ({recent_volume/avg_volume:.2f}x average)")
                signal = None
                self._record_latency(start_time, "volume_filtered")
        
        # If signal generated, emit event
        if signal:
            metadata = {
                'volatility_state': self._state['volatility_state'],
                'volatility_trend': self._state['volatility_trend'],
                'bb_width': (indicators['bb_upper'].iloc[-1] - indicators['bb_lower'].iloc[-1]) / indicators['bb_middle'].iloc[-1] if 'bb_upper' in indicators else None,
                'atr_ratio': indicators['atr'].iloc[-1] / indicators['atr'].iloc[-20:].mean() if 'atr' in indicators else None,
                'reason': locals().get('reason', 'unspecified')
            }
            self._emit_signal_event(signal, data, metadata)
            self._record_latency(start_time, "signal_generation", signal)
        else:
            self._record_latency(start_time, "no_signal")
            
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

    def get_required_features(self) -> Set[str]:
        """
        Return the list of features required by the strategy.

        Returns:
            Set of feature names
        """
        features = {'open', 'high', 'low', 'close', 'volume'}

        # Add indicator features based on enabled components
        if self.parameters['use_bbands']:
            features.update({'bb_upper', 'bb_lower', 'bb_middle'})

        if self.parameters['use_atr']:
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
        volatility = market_conditions.get('volatility', 0.5)
        volatility_trend = market_conditions.get('volatility_trend', 'neutral')

        # Store original parameters for tracking changes
        old_vol_modifier = self._state['volatility_threshold_modifier']
        old_trailing_stop = self._state['trailing_stop_multiplier']

        # Update volatility threshold based on performance
        if profit_factor > 1.5 and win_rate > 0.6:
            # Successful configuration, slightly reduce threshold for more trades
            self._state['volatility_threshold_modifier'] = max(0.7, self._state['volatility_threshold_modifier'] * 0.98)
        elif profit_factor < 1.0 or win_rate < 0.4:
            # Unsuccessful configuration, increase threshold for better quality
            self._state['volatility_threshold_modifier'] = min(1.5, self._state['volatility_threshold_modifier'] * 1.02)

        # Adjust trailing stop multiplier based on volatility conditions
        if volatility > 0.7:  # High volatility
            self._state['trailing_stop_multiplier'] = min(3.0, self._state['trailing_stop_multiplier'] * 1.02)
        elif volatility < 0.3:  # Low volatility
            self._state['trailing_stop_multiplier'] = max(1.5, self._state['trailing_stop_multiplier'] * 0.98)

        # Adjust based on volatility trend
        if volatility_trend == 'increasing':
            # When volatility is increasing, be more selective with entries
            self._state['volatility_threshold_modifier'] = min(1.5, self._state['volatility_threshold_modifier'] * 1.01)
        elif volatility_trend == 'decreasing':
            # When volatility is decreasing, can be less selective
            self._state['volatility_threshold_modifier'] = max(0.7, self._state['volatility_threshold_modifier'] * 0.99)

        # Log and emit parameter updates if any changed
        if (old_vol_modifier != self._state['volatility_threshold_modifier'] or
                old_trailing_stop != self._state['trailing_stop_multiplier']):
            logger.info(f"Online parameter update: threshold={self._state['volatility_threshold_modifier']:.2f}, "
                        f"trailing_stop={self._state['trailing_stop_multiplier']:.2f}")

            EventBus.emit("strategy.online_update", {
                'strategy_id': self.id,
                'strategy_type': 'volatility',
                'timestamp': time.time(),
                'new_parameters': {
                    'volatility_threshold_modifier': self._state['volatility_threshold_modifier'],
                    'trailing_stop_multiplier': self._state['trailing_stop_multiplier']
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
                "VolatilityStrategy",
                description="Captures price movements during periods of changing volatility",
                default_parameters={
                    'bb_period': 20,
                    'bb_std': 2.0,
                    'atr_period': 14,
                    'expansion_threshold': 1.4,
                    'contraction_threshold': 0.6
                },
                compatibility={
                    'min_bars': 50,
                    'timeframes': ['5m', '15m', '1h', '4h', 'daily'],
                    'asset_classes': ['forex', 'crypto', 'stocks', 'commodities', 'indices']
                }
            )
            logger.info("VolatilityStrategy registered successfully")
        except ImportError:
            logger.warning("StrategyRegistry not available, skip registration")
        except Exception as e:
            logger.error(f"Failed to register VolatilityStrategy: {str(e)}")

    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[
        pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            prices: Series of prices
            window: Bollinger Band window period
            num_std: Number of standard deviations

        Returns:
            Tuple of (Upper Band, Middle Band, Lower Band)
        """
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)

        return upper, middle, lower

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

        # Entry reason to identify volatility state
        entry_reason = trade_result.get('entry_reason', '')

        # Store original parameters for tracking changes
        old_vol_modifier = self._state['volatility_threshold_modifier']
        old_trailing_stop = self._state['trailing_stop_multiplier']
        old_position_size = self.position_size_multiplier

        # Identify which volatility state the trade was taken in
        trade_vol_state = None
        if 'breakout' in entry_reason.lower():
            trade_vol_state = 'contraction'
        elif 'reversal' in entry_reason.lower():
            trade_vol_state = 'expansion'
        else:
            trade_vol_state = 'normal'

        if successful:
            # If trade was successful, fine-tune the strategy
            if regime_id is not None:  # Check if we have valid regime information
                # Adjust parameters based on the volatility state that generated the profitable trade
                if trade_vol_state == 'contraction':
                    # Lower threshold to get more breakout trades
                    self._state['volatility_threshold_modifier'] = max(0.7, self._state[
                        'volatility_threshold_modifier'] * 0.95)
                    logger.debug(
                        f"Lowered volatility threshold to {self._state['volatility_threshold_modifier']:.2f} after successful breakout trade")

                elif trade_vol_state == 'expansion':
                    # Adjust stop multiplier for reversal trades
                    self._state['trailing_stop_multiplier'] = max(1.5, self._state['trailing_stop_multiplier'] * 0.95)
                    logger.debug(
                        f"Adjusted trailing stop to {self._state['trailing_stop_multiplier']:.2f} after successful reversal trade")

                # For very profitable trades, increase position sizing slightly
                if pnl_pct > 0.05:  # 5% profit
                    self.position_size_multiplier = min(1.5, self.position_size_multiplier * 1.05)
                    logger.debug(
                        f"Increased position size multiplier to {self.position_size_multiplier:.2f} after profitable trade")
        else:
            # If trade was unsuccessful, adjust parameters
            if regime_id is not None:  # Check if we have valid regime information
                # Adjust parameters based on the volatility state that generated the losing trade
                if trade_vol_state == 'contraction':
                    # Increase threshold to filter out more breakout trades
                    self._state['volatility_threshold_modifier'] = min(1.5, self._state[
                        'volatility_threshold_modifier'] * 1.05)
                    logger.debug(
                        f"Increased volatility threshold to {self._state['volatility_threshold_modifier']:.2f} after unsuccessful breakout trade")

                elif trade_vol_state == 'expansion':
                    # Adjust stop multiplier for reversal trades
                    self._state['trailing_stop_multiplier'] = min(3.0, self._state['trailing_stop_multiplier'] * 1.05)
                    logger.debug(
                        f"Adjusted trailing stop to {self._state['trailing_stop_multiplier']:.2f} after unsuccessful reversal trade")

                # For losing trades, decrease position sizing slightly
                if pnl_pct < -0.03:  # 3% loss
                    self.position_size_multiplier = max(0.5, self.position_size_multiplier * 0.95)
                    logger.debug(
                        f"Decreased position size multiplier to {self.position_size_multiplier:.2f} after losing trade")

        # Emit parameter adaptation event if any parameters changed
        if (old_vol_modifier != self._state['volatility_threshold_modifier'] or
                old_trailing_stop != self._state['trailing_stop_multiplier'] or
                old_position_size != self.position_size_multiplier):
            self._emit_parameter_adaptation_event(
                trade_result,
                {
                    'volatility_threshold_modifier': {
                        'old': old_vol_modifier,
                        'new': self._state['volatility_threshold_modifier']
                    },
                    'trailing_stop_multiplier': {
                        'old': old_trailing_stop,
                        'new': self._state['trailing_stop_multiplier']
                    },
                    'position_size_multiplier': {
                        'old': old_position_size,
                        'new': self.position_size_multiplier
                    },
                    'volatility_state': trade_vol_state
                }
            )

        # Keep volatility history manageable
        if len(self._state['volatility_history']) > 100:
            self._state['volatility_history'] = self._state['volatility_history'][-100:]

    def _emit_parameter_adaptation_event(self, trade_result: Dict[str, Any], param_changes: Dict[str, Any]) -> None:
        """
        Emit parameter adaptation event to the event bus.

        Args:
            trade_result: Trade result information
            param_changes: Parameter changes applied
        """
        EventBus.emit("strategy.parameter_adaptation", {
            'strategy_id': self.id,
            'strategy_type': 'volatility',
            'timestamp': time.time(),
            'trade_id': trade_result.get('id', 'unknown'),
            'trade_pnl': trade_result.get('pnl', 0),
            'trade_pnl_pct': trade_result.get('pnl_pct', 0),
            'regime_id': trade_result.get('regime_id', 'unknown'),
            'parameter_changes': param_changes,
            'adaptation_method': 'online_bayesian'
        })

        def cluster_fit(self, cluster_metrics: Dict[str, float]) -> float:
            """
            Determine if the volatility strategy is suitable for the given cluster.

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

        # Volatility strategy typically works best with moderate to high volatility
        # but not extreme, and works in both trending and ranging markets

        # Volatility component
        volatility_rank = cluster_metrics.get('volatility_pct_rank', 0.5)
        min_vol = self.parameters['min_volatility_percentile']
        max_vol = self.parameters['max_volatility_percentile']

        # Calculate volatility score
        if min_vol <= volatility_rank <= max_vol:
            # Ideal range with peak in the middle
            mid_range = (min_vol + max_vol) / 2
            volatility_score = 1.0 - 0.5 * abs(volatility_rank - mid_range) / (max_vol - min_vol)
        else:
            # Outside ideal range
            distance = min(abs(volatility_rank - min_vol), abs(volatility_rank - max_vol))
            volatility_score = max(0, 0.5 - distance)

        # ADX component (trend strength) - avoid extreme trending markets
        adx_mean = cluster_metrics.get('ADX_mean', 20)
        if adx_mean > 40:  # Extremely strong trend
            adx_score = max(0, 1 - (adx_mean - 40) / 20)
        else:
            # Moderate ADX is fine
            adx_score = 0.8  # Good baseline score

        # Jump probability component - higher jumps benefit volatility strategy
        jump_probability = cluster_metrics.get('jump_probability', 0)
        jump_score = min(1.0, jump_probability * 5)  # Scale up since jump probability is usually low

        # Volatility change component - changing volatility is ideal
        volatility_zscore = abs(cluster_metrics.get('volatility_zscore', 0))
        if volatility_zscore > 1.0:  # Significant volatility deviation
            vol_change_score = min(1.0, volatility_zscore / 2)
        else:
            vol_change_score = volatility_zscore

        # Weighted combination
        weights = {
            'volatility': 0.35,
            'adx': 0.15,
            'jump': 0.25,
            'vol_change': 0.25
        }

        fitness_score = (
                weights['volatility'] * volatility_score +
                weights['adx'] * adx_score +
                weights['jump'] * jump_score +
                weights['vol_change'] * vol_change_score
        )

        # Log detailed scores for analysis
        logger.debug(f"Volatility cluster fit - Vol: {volatility_score:.2f}, ADX: {adx_score:.2f}, "
                     f"Jump: {jump_score:.2f}, Vol_Change: {vol_change_score:.2f}, Total: {fitness_score:.2f}")

        # Emit event with fitness score
        EventBus.emit("strategy.cluster_fit", {
            'strategy_id': self.id,
            'strategy_type': 'volatility',
            'cluster_id': cluster_metrics.get('cluster_id', 'unknown'),
            'fitness_score': fitness_score,
            'components': {
                'volatility_score': volatility_score,
                'adx_score': adx_score,
                'jump_score': jump_score,
                'vol_change_score': vol_change_score
            }
        })

        return fitness_score

        def _detect_volatility_shift(self, data: pd.DataFrame, atr: pd.Series) -> bool:
            """
            Detect significant shifts in volatility.

            Args:
                data: DataFrame with market data
                atr: ATR series

            Returns:
                bool: True if a volatility shift is detected, False otherwise
            """

        if atr is None or len(atr) < 5:
            return False

        # Calculate volatility change
        atr_change = atr.iloc[-1] / atr.iloc[-5:].mean()

        # Detect significant increase in volatility
        if atr_change > 1.5:
            logger.debug(f"Volatility shift detected: {atr_change:.2f}x increase")
            return True

        # Detect significant decrease in volatility
        if atr_change < 0.6:
            logger.debug(f"Volatility shift detected: {atr_change:.2f}x decrease")
            return True

        return False

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
            f"Adapting volatility strategy to regime {regime_id}: bias={directional_bias}, volatility={volatility_regime}")

        # Store original parameters for tracking changes
        old_vol_modifier = self._state['volatility_threshold_modifier']
        old_trailing_stop = self._state['trailing_stop_multiplier']

        # Apply optimization based on time of day
        if peak_hour in [8, 9]:
            # European market opening optimizations
            self._state['volatility_threshold_modifier'] = 0.9  # Slightly easier to trigger
            self._state['trailing_stop_multiplier'] = 1.8
            logger.info(f"Applied European session volatility optimizations")
        elif peak_hour in [13, 14, 15]:
            # US market opening optimizations (typically more volatile)
            self._state['volatility_threshold_modifier'] = 1.2  # Require stronger signals
            self._state['trailing_stop_multiplier'] = 2.2  # Wider stops during US session
            logger.info(f"Applied US session volatility optimizations")

        # Apply optimization based on directional bias
        if directional_bias == DirectionalBias.UPWARD:
            # Create biased signal generator for upward bias
            original_generate = self.generate_signal

            def biased_signal(data: pd.DataFrame) -> Optional[str]:
                signal = original_generate(data)

                # In upward bias, favor long signals in contraction and short signals in expansion
                if self._state['volatility_state'] == "contraction" and signal == "short":
                    # Require stronger confirmation for shorts during contraction in upward bias
                    if 'bb_lower' in data.columns and data['close'].iloc[-1] < data['bb_lower'].iloc[-1] * 0.95:
                        return "short"
                    return None
                elif self._state['volatility_state'] == "expansion" and signal == "long":
                    # Require stronger confirmation for longs during expansion in upward bias
                    if 'bb_upper' in data.columns and data['close'].iloc[-1] > data['bb_upper'].iloc[-1] * 1.05:
                        return "long"
                    return None

                return signal

            self.generate_signal = biased_signal
            logger.info(f"Applied upward bias signal optimization to {self.name}")

        elif directional_bias == DirectionalBias.DOWNWARD:
            # Create biased signal generator for downward bias
            original_generate = self.generate_signal

            def biased_signal(data: pd.DataFrame) -> Optional[str]:
                signal = original_generate(data)

                # In downward bias, favor short signals in contraction and long signals in expansion
                if self._state['volatility_state'] == "contraction" and signal == "long":
                    # Require stronger confirmation for longs during contraction in downward bias
                    if 'bb_upper' in data.columns and data['close'].iloc[-1] > data['bb_upper'].iloc[-1] * 1.05:
                        return "long"
                    return None
                elif self._state['volatility_state'] == "expansion" and signal == "short":
                    # Require stronger confirmation for shorts during expansion in downward bias
                    if 'bb_lower' in data.columns and data['close'].iloc[-1] < data['bb_lower'].iloc[-1] * 0.95:
                        return "short"
                    return None

                return signal

            self.generate_signal = biased_signal
            logger.info(f"Applied downward bias signal optimization to {self.name}")

        # Apply optimization based on volatility regime
        if volatility_regime == VolatilityRegime.HIGH:
            # High volatility optimizations
            self._state['volatility_threshold_modifier'] = 1.3  # Increase threshold in high vol
            self._state['trailing_stop_multiplier'] = 2.5  # Wider stops in high vol
            logger.info(f"Applied high volatility optimizations")
        elif volatility_regime == VolatilityRegime.LOW:
            # Low volatility optimizations
            self._state['volatility_threshold_modifier'] = 0.7  # Lower threshold in low vol
            self._state['trailing_stop_multiplier'] = 1.5  # Tighter stops in low vol
            logger.info(f"Applied low volatility optimizations")

        # Apply currency pair specific optimizations
        if hasattr(self, 'currency_pair'):
            currency_pair = self.currency_pair

            if currency_pair == 'EURUSD':
                # EUR/USD specific optimizations (typically lower volatility)
                self._state['volatility_threshold_modifier'] = self._state['volatility_threshold_modifier'] * 0.9
                logger.info(f"Applied EURUSD-specific volatility optimization")
            elif currency_pair == 'USDJPY':
                # USD/JPY specific optimizations
                self._state['volatility_threshold_modifier'] = self._state['volatility_threshold_modifier'] * 1.1
                logger.info(f"Applied USDJPY-specific volatility optimization")

        # Emit regime adaptation event
        EventBus.emit("strategy.regime_adaptation", {
            'strategy_id': self.id,
            'strategy_type': 'volatility',
            'timestamp': time.time(),
            'regime_id': regime_id,
            'parameters': {
                'volatility_threshold_modifier': {
                    'old': old_vol_modifier,
                    'new': self._state['volatility_threshold_modifier']
                },
                'trailing_stop_multiplier': {
                    'old': old_trailing_stop,
                    'new': self._state['trailing_stop_multiplier']
                }
            }
        })

        def _update_volatility_state(self, data: pd.DataFrame, atr: Optional[pd.Series] = None) -> None:
            """
            Update the current volatility state and trend.

            Args:
                data: DataFrame with market data
                atr: Optional pre-calculated ATR series
            """

        # If ATR is not provided, calculate it
        if atr is None and self.parameters['use_atr']:
            atr = self._calculate_atr(data, self.parameters['atr_period'])

        if atr is not None and len(atr) >= self.parameters['atr_lookback']:
            # Store the recent ATR value
            self._state['volatility_history'].append(atr.iloc[-1])
            if len(self._state['volatility_history']) > 100:
                self._state['volatility_history'] = self._state['volatility_history'][-100:]

            # Calculate recent vs historical volatility
            recent_atr = atr.iloc[-20:].mean()
            historical_atr = atr.iloc[-self.parameters['atr_lookback']:].mean()

            # Calculate volatility ratio
            vol_ratio = recent_atr / historical_atr if historical_atr > 0 else 1.0

            # Calculate volatility trend
            short_term_atr = atr.iloc[-5:].mean()
            medium_term_atr = atr.iloc[-20:].mean()

            # Determine volatility trend
            if short_term_atr > medium_term_atr * 1.1:
                self._state['volatility_trend'] = "increasing"
            elif short_term_atr < medium_term_atr * 0.9:
                self._state['volatility_trend'] = "decreasing"
            else:
                self._state['volatility_trend'] = "neutral"

            # Determine volatility state
            expansion_threshold = self.parameters['expansion_threshold'] * self._state['volatility_threshold_modifier']
            contraction_threshold = self.parameters['contraction_threshold'] / self._state[
                'volatility_threshold_modifier']

            # Apply regime specific adjustments
            if hasattr(self, 'regime_characteristics') and self.regime_characteristics:
                if self.regime_characteristics.volatility_regime == VolatilityRegime.HIGH:
                    expansion_threshold *= 1.2
                    contraction_threshold *= 1.2
                elif self.regime_characteristics.volatility_regime == VolatilityRegime.LOW:
                    expansion_threshold *= 0.8
                    contraction_threshold *= 0.8

            if vol_ratio > expansion_threshold:
                self._state['volatility_state'] = "expansion"
                logger.debug(f"Volatility state: expansion (ratio: {vol_ratio:.2f})")
            elif vol_ratio < contraction_threshold:
                self._state['volatility_state'] = "contraction"
                logger.debug(f"Volatility state: contraction (ratio: {vol_ratio:.2f})")
            else:
                self._state['volatility_state'] = "normal"
                logger.debug(f"Volatility state: normal (ratio: {vol_ratio:.2f})")
        else:
            # Default to normal volatility if not enough data
            self._state['volatility_state'] = "normal"
            self._state['volatility_trend'] = "neutral"

        # Emit volatility state update event
        EventBus.emit("volatility.state_update", {
            'strategy_id': self.id,
            'timestamp': time.time(),
            'volatility_state': self._state['volatility_state'],
            'volatility_trend': self._state['volatility_trend'],
            'latest_atr': atr.iloc[-1] if atr is not None and len(atr) > 0 else None
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

        # Calculate or get the indicators
        indicators = {}

        if self.parameters['use_bbands']:
            if 'bb_upper' not in data.columns or 'bb_lower' not in data.columns:
                indicators['bb_upper'], indicators['bb_middle'], indicators[
                    'bb_lower'] = self._calculate_bollinger_bands(
                    data['close'],
                    self.parameters['bb_period'],
                    self.parameters['bb_std']
                )
            else:
                indicators['bb_upper'] = data['bb_upper']
                indicators['bb_middle'] = data['bb_middle'] if 'bb_middle' in data.columns else (data['bb_upper'] +
                                                                                                 data['bb_lower']) / 2
                indicators['bb_lower'] = data['bb_lower']

        if self.parameters['use_atr']:
            if 'atr_14' not in data.columns:
                indicators['atr'] = self._calculate_atr(data, self.parameters['atr_period'])
            else:
                indicators['atr'] = data['atr_14']

        # Get current price
        close = data['close'].iloc[-1]

        # Get entry reason to determine exit logic
        entry_reason = position.get('entry_reason', '')

        # Check for exit conditions based on volatility state and position direction
        if position['direction'] == 'long':
            # Exit long position conditions
            if 'contraction' in entry_reason.lower():
                # Exit breakout trade if momentum stalls
                momentum = data['close'].pct_change(3).iloc[-1]
                if momentum < 0 and close < indicators['bb_middle'].iloc[-1]:
                    reason = "Exit long breakout: momentum stalled"
                    self._emit_exit_event(data, position, reason)
                    self.log_signal('exit', data, reason)
                    self._record_latency(start_time, "exit_signal", "momentum_stall")
                    return True

            elif 'expansion' in entry_reason.lower():
                # Exit reversal trade if target reached
                if close > indicators['bb_middle'].iloc[-1]:
                    reason = "Exit long reversal: target reached"
                    self._emit_exit_event(data, position, reason)
                    self.log_signal('exit', data, reason)
                    self._record_latency(start_time, "exit_signal", "target_reached")
                    return True

            else:  # normal volatility
                # Exit on volatility shift
                if self._detect_volatility_shift(data, indicators.get('atr')):
                    reason = "Exit long: volatility shift detected"
                    self._emit_exit_event(data, position, reason)
                    self.log_signal('exit', data, reason)
                    self._record_latency(start_time, "exit_signal", "volatility_shift")
                    return True

        elif position['direction'] == 'short':
            # Exit short position conditions
            if 'contraction' in entry_reason.lower():
                # Exit breakout trade if momentum stalls
                momentum = data['close'].pct_change(3).iloc[-1]
                if momentum > 0 and close > indicators['bb_middle'].iloc[-1]:
                    reason = "Exit short breakout: momentum stalled"
                    self._emit_exit_event(data, position, reason)
                    self.log_signal('exit', data, reason)
                    self._record_latency(start_time, "exit_signal", "momentum_stall")
                    return True

            elif 'expansion' in entry_reason.lower():
                # Exit reversal trade if target reached
                if close < indicators['bb_middle'].iloc[-1]:
                    reason = "Exit short reversal: target reached"
                    self._emit_exit_event(data, position, reason)
                    self.log_signal('exit', data, reason)
                    self._record_latency(start_time, "exit_signal", "target_reached")
                    return True

            else:  # normal volatility
                # Exit on volatility shift
                if self._detect_volatility_shift(data, indicators.get('atr')):
                    reason = "Exit short: volatility shift detected"
                    self._emit_exit_event(data, position, reason)
                    self.log_signal('exit', data, reason)
                    self._record_latency(start_time, "exit_signal", "volatility_shift")
                    return True

        # Check if volatility has dropped significantly (for breakout trades)
        if 'breakout' in entry_reason.lower() and 'atr' in indicators:
            if indicators['atr'].iloc[-1] < indicators['atr'].iloc[-5:].mean() * 0.7:
                reason = "Exit breakout trade: volatility declining"
                self._emit_exit_event(data, position, reason)
                self.log_signal('exit', data, reason)
                self._record_latency(start_time, "exit_signal", "volatility_decline")
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
            'strategy_type': 'volatility',
            'timestamp': time.time(),
            'position_id': position.get('id', 'unknown'),
            'instrument': data.get('symbol', 'unknown'),
            'price': data['close'].iloc[-1],
            'reason': reason,
            'position_duration': time.time() - position.get('entry_time', time.time()),
            'pnl_pct': (data['close'].iloc[-1] / position.get('entry_price', data['close'].iloc[-1]) - 1) *
                       (1 if position.get('direction', 'long') == 'long' else -1),
            'volatility_state': self._state['volatility_state']
        })

        def risk_parameters(self, data: pd.DataFrame, entry_price: float) -> Dict[str, float]:
            """
            Calculate risk parameters for the volatility strategy.

            Args:
                data: DataFrame with market data
                entry_price: Entry price for the trade

            Returns:
                Dict with stop loss, take profit, and position size
            """

        start_time = time.time()

        # Calculate ATR if not available
        if 'atr_14' not in data.columns:
            atr = self._calculate_atr(data, self.parameters['atr_period']).iloc[-1]
        else:
            atr = data['atr_14'].iloc[-1]

        # Adjust risk parameters based on volatility state
        if self._state['volatility_state'] == "contraction":
            # Breakout trades: Wider stops, larger targets
            stop_loss_multiplier = 1.5 * self.stop_loss_modifier
            take_profit_multiplier = 3.0 * self.profit_target_modifier
            position_size_factor = 1.2  # Larger positions due to clearer breakouts

        elif self._state['volatility_state'] == "expansion":
            # Reversal trades: Tighter stops, smaller targets
            stop_loss_multiplier = 1.0 * self.stop_loss_modifier
            take_profit_multiplier = 2.0 * self.profit_target_modifier
            position_size_factor = 0.8  # Smaller positions due to higher risk

        else:  # normal volatility
            # Standard parameters
            stop_loss_multiplier = self._state['trailing_stop_multiplier'] * self.stop_loss_modifier
            take_profit_multiplier = 2.5 * self.profit_target_modifier
            position_size_factor = 1.0

        # Calculate actual risk parameters
        stop_loss_pct = (atr * stop_loss_multiplier) / entry_price
        take_profit_pct = (atr * take_profit_multiplier) / entry_price

        # Calculate position size with adjusted factor
        position_size = self.dynamic_position_size(self.account_balance) * position_size_factor

        # Emit risk parameters event
        self._emit_risk_event(entry_price, stop_loss_pct, take_profit_pct, position_size)

        # Record performance
        self._record_latency(start_time, "risk_parameters")

        return {
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'position_size': position_size,
            'stop_price': entry_price * (1 - stop_loss_pct) if entry_price > 0 else 0,
            'take_profit_price': entry_price * (1 + take_profit_pct) if entry_price > 0 else 0,
            'atr': atr,
            'volatility_state': self._state['volatility_state']
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
        EventBus.emit("strategy.risk_parameters", {
            'strategy_id': self.id,
            'strategy_type': 'volatility',
            'timestamp': time.time(),
            'entry_price': entry_price,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'position_size': position_size,
            'risk_reward_ratio': take_profit_pct / stop_loss_pct if stop_loss_pct > 0 else 0,
            'volatility_state': self._state['volatility_state'],
            'volatility_trend': self._state['volatility_trend']
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

        # Calculate confidence based on volatility state and strength of signal
        vol_state = metadata.get('volatility_state', 'normal')

        # Base confidence on volatility state
        if vol_state == 'contraction':
            # Breakout signals from contraction are higher confidence
            base_confidence = 0.7
        elif vol_state == 'expansion':
            # Reversal signals from expansion are moderate confidence
            base_confidence = 0.6
        else:
            # Normal state signals are lower confidence
            base_confidence = 0.5

        # Adjust based on indicator extremity
        bb_width = metadata.get('bb_width')
        atr_ratio = metadata.get('atr_ratio')

        # If BB width or ATR ratio are available, adjust confidence
        if bb_width is not None:
            # More extreme BB width gives higher confidence
            bb_factor = min(1.5, bb_width * 5)  # Scale factor, caps at 1.5
            base_confidence *= (0.8 + 0.2 * bb_factor)

        if atr_ratio is not None:
            # More extreme ATR ratio gives higher confidence
            atr_factor = min(1.5, atr_ratio)  # Scale factor, caps at 1.5
            base_confidence *= (0.8 + 0.2 * atr_factor)

        # Cap final confidence at 1.0
        confidence = min(1.0, base_confidence)

        EventBus.emit("strategy.signal", {
            'strategy_id': self.id,
            'strategy_type': 'volatility',
            'timestamp': time.time(),
            'signal': signal,
            'instrument': data.get('symbol', 'unknown'),
            'price': data['close'].iloc[-1],
            'confidence': confidence,
            'metadata': metadata
        })