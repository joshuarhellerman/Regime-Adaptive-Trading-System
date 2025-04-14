"""
Mean Reversion Strategy Module

This module implements a mean reversion strategy that adapts to different market regimes.
The strategy uses statistical measures like RSI, Bollinger Bands, and Z-scores to identify
when prices have deviated significantly from their mean and are likely to revert.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import time

from models.strategies.strategy_base import TradingStrategy, DirectionalBias, VolatilityRegime

# Configure logger
logger = logging.getLogger(__name__)

class MeanReversionStrategy(TradingStrategy):
    """
    Mean reversion strategy that adapts to market regimes.

    This strategy identifies situations where prices have deviated substantially
    from their statistical mean and are likely to revert back. It uses indicators
    like RSI, Bollinger Bands, and z-scores with regime-specific adaptations.
    """

    def __init__(self, name: str = None, parameters: Dict[str, Any] = None):
        """
        Initialize the mean reversion strategy.

        Args:
            name: Strategy name
            parameters: Strategy-specific parameters
        """
        # Default parameters
        default_params = {
            'lookback_period': 20,
            'rsi_period': 14,
            'bb_period': 20,
            'bb_std': 2.0,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'zscore_threshold': 2.0,
            'exit_zscore': 0.5,
            'min_liquidity': 1.0,
            'max_holding_period': 10,  # Maximum days to hold position
            'volatility_filter': True,
            'min_bars': 30,
            'epsilon': 1e-8  # Small value to prevent division by zero
        }

        # Merge default parameters with provided parameters
        if parameters:
            for key, value in parameters.items():
                default_params[key] = value

        # Initialize the base class
        super().__init__(name or "MeanReversion", default_params)

        # Strategy-specific attributes
        self.adaptive_thresholds = (
            self.parameters['rsi_oversold'],
            self.parameters['rsi_overbought']
        )
        self.liquidity_factor = 1.0
        self.bb_percentile = 75

        # Cache for calculated indicators
        self.indicator_cache = {}

        logger.info(
            f"Initialized MeanReversionStrategy with lookback={self.parameters['lookback_period']}, "
            f"rsi_thresholds=({self.parameters['rsi_oversold']}, {self.parameters['rsi_overbought']})"
        )

    def _validate_parameters(self) -> None:
        """
        Validate strategy parameters.

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate period values
        for param in ['lookback_period', 'rsi_period', 'bb_period']:
            if self.parameters[param] <= 0:
                raise ValueError(f"Parameter {param} must be greater than 0")

        # Validate threshold values
        if not 0 < self.parameters['rsi_oversold'] < self.parameters['rsi_overbought'] < 100:
            raise ValueError("RSI thresholds must satisfy: 0 < oversold < overbought < 100")

        if self.parameters['zscore_threshold'] <= 0:
            raise ValueError("Z-score threshold must be greater than 0")

        if self.parameters['bb_std'] <= 0:
            raise ValueError("Bollinger Band standard deviation must be greater than 0")

        if self.parameters['max_holding_period'] <= 0:
            raise ValueError("Max holding period must be greater than 0")

        if self.parameters['epsilon'] <= 0:
            raise ValueError("Epsilon must be greater than 0")

    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data for signal generation.

        Args:
            data: DataFrame with market data

        Raises:
            ValueError: If data is invalid
        """
        # Check if data exists and is not empty
        if data is None or len(data) == 0:
            raise ValueError("Input data is empty or None")

        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check for NaN values in critical columns
        critical_columns = ['close'] + [col for col in ['high', 'low', 'volume', 'rsi_14', 'bb_upper', 'bb_lower']
                                       if col in data.columns]

        for col in critical_columns:
            if data[col].isna().any():
                # Fill NaN values for volume if present
                if col == 'volume':
                    data[col] = data[col].fillna(0)
                else:
                    # For price data, forward-fill NaNs
                    data[col] = data[col].fillna(method='ffill')

                    # If there are still NaNs at the beginning, backward-fill
                    if data[col].isna().any():
                        data[col] = data[col].fillna(method='bfill')

                    # If still NaNs, raise error
                    if data[col].isna().any():
                        raise ValueError(f"Column {col} contains NaN values that cannot be filled")

    def generate_signal(self, data: pd.DataFrame) -> Optional[str]:
        """
        Generate a mean reversion signal.

        Args:
            data: DataFrame with market data

        Returns:
            str: 'long', 'short', or None for no signal
        """
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

        # Make a copy of data to avoid modifying the original
        data_copy = data.copy()

        # Use adjusted thresholds based on regime optimization
        rsi_oversold, rsi_overbought = self.adaptive_thresholds

        # Apply signal threshold modifier from regime optimization
        zscore_threshold = self.parameters['zscore_threshold'] * self.signal_threshold_modifier

        # Calculate or get RSI
        if 'rsi_14' not in data_copy.columns:
            rsi = self._get_cached_indicator('rsi', data_copy)
        else:
            rsi = data_copy['rsi_14']

        # Calculate or get Bollinger Bands
        if 'bb_upper' not in data_copy.columns or 'bb_lower' not in data_copy.columns:
            bb_upper, bb_middle, bb_lower = self._get_cached_indicator('bollinger_bands', data_copy)
        else:
            bb_upper = data_copy['bb_upper']
            bb_lower = data_copy['bb_lower']
            bb_middle = (bb_upper + bb_lower) / 2

        # Calculate Z-score (how many standard deviations from mean)
        lookback = self.parameters['lookback_period']
        rolling_mean = data_copy['close'].rolling(lookback).mean()
        rolling_std = data_copy['close'].rolling(lookback).std() + self.parameters['epsilon']  # Prevent division by zero
        z_score = (data_copy['close'] - rolling_mean) / rolling_std

        # Apply volatility filter if enabled
        use_volatility_filter = self.parameters['volatility_filter']
        volatility_filtered = True

        if use_volatility_filter:
            # Calculate volatility percentile
            current_std = rolling_std.iloc[-1]
            historical_std = rolling_std.iloc[-lookback*2:].mean()

            # Only generate signals when volatility is moderate
            vol_ratio = current_std / (historical_std + self.parameters['epsilon'])  # Prevent division by zero
            volatility_filtered = 0.5 <= vol_ratio <= 2.0

        # Check for mean reversion signals
        current_rsi = rsi.iloc[-1]
        current_zscore = z_score.iloc[-1]
        current_price = data_copy['close'].iloc[-1]

        # Check for oversold conditions (long signal)
        if (current_rsi <= rsi_oversold or current_zscore <= -zscore_threshold or
                current_price <= bb_lower.iloc[-1]):

            if volatility_filtered:
                # Apply liquidity filter if volume data available
                if 'volume' in data_copy.columns:
                    recent_volume = data_copy['volume'].iloc[-5:].mean()
                    historical_volume = data_copy['volume'].iloc[-lookback:].mean() + self.parameters['epsilon']

                    if recent_volume >= historical_volume * self.liquidity_factor:
                        self.log_signal('long', data_copy, f"Mean reversion long: RSI={current_rsi:.1f}, Z-score={current_zscore:.2f}")
                        return 'long'
                else:
                    # No volume data, proceed with signal
                    self.log_signal('long', data_copy, f"Mean reversion long: RSI={current_rsi:.1f}, Z-score={current_zscore:.2f}")
                    return 'long'

        # Check for overbought conditions (short signal)
        elif (current_rsi >= rsi_overbought or current_zscore >= zscore_threshold or
                current_price >= bb_upper.iloc[-1]):

            if volatility_filtered:
                # Apply liquidity filter if volume data available
                if 'volume' in data_copy.columns:
                    recent_volume = data_copy['volume'].iloc[-5:].mean()
                    historical_volume = data_copy['volume'].iloc[-lookback:].mean() + self.parameters['epsilon']

                    if recent_volume >= historical_volume * self.liquidity_factor:
                        self.log_signal('short', data_copy, f"Mean reversion short: RSI={current_rsi:.1f}, Z-score={current_zscore:.2f}")
                        return 'short'
                else:
                    # No volume data, proceed with signal
                    self.log_signal('short', data_copy, f"Mean reversion short: RSI={current_rsi:.1f}, Z-score={current_zscore:.2f}")
                    return 'short'

        return None

    def _get_cached_indicator(self, indicator_name: str, data: pd.DataFrame) -> Any:
        """
        Get or calculate indicator with caching.

        Args:
            indicator_name: Name of the indicator
            data: DataFrame with market data

        Returns:
            Calculated indicator
        """
        # Generate a cache key (e.g., combination of indicator name and data length)
        cache_key = f"{indicator_name}_{len(data)}"

        # Check if indicator is already cached
        if cache_key in self.indicator_cache:
            return self.indicator_cache[cache_key]

        # Calculate indicator
        if indicator_name == 'rsi':
            result = self._calculate_rsi(data['close'], self.parameters['rsi_period'])
            self.indicator_cache[cache_key] = result
            return result
        elif indicator_name == 'bollinger_bands':
            result = self._calculate_bollinger_bands(
                data['close'],
                self.parameters['bb_period'],
                self.parameters['bb_std']
            )
            self.indicator_cache[cache_key] = result
            return result
        elif indicator_name == 'atr':
            result = self._calculate_atr(data, 14)
            self.indicator_cache[cache_key] = result
            return result
        else:
            raise ValueError(f"Unknown indicator: {indicator_name}")

    def risk_parameters(self, data: pd.DataFrame, entry_price: float) -> Dict[str, float]:
        """
        Calculate risk parameters for the mean reversion strategy.

        Args:
            data: DataFrame with market data
            entry_price: Entry price for the trade

        Returns:
            Dict with stop loss, take profit, and position size
        """
        # Calculate or get ATR
        if 'atr_14' not in data.columns:
            atr = self._get_cached_indicator('atr', data).iloc[-1]
        else:
            atr = data['atr_14'].iloc[-1]

        # Ensure ATR is positive
        atr = max(atr, self.parameters['epsilon'])

        # Determine stop distance based on average true range
        # Mean reversion requires wider stops than trend following
        stop_distance = atr * 2.5 * self.stop_loss_modifier

        # Take profit at the mean (plus a bit of buffer)
        lookback = self.parameters['lookback_period']
        mean_price = data['close'].rolling(lookback).mean().iloc[-1]

        # Calculate profit target based on direction
        if entry_price < mean_price:
            # Long position
            take_profit_pct = (mean_price * 1.05 - entry_price) / entry_price
        else:
            # Short position
            take_profit_pct = (entry_price - mean_price * 0.95) / entry_price

        # Apply modifier
        take_profit_pct *= self.profit_target_modifier

        # Minimum profit target
        take_profit_pct = max(take_profit_pct, atr / entry_price * 1.5)

        # Maximum profit target
        take_profit_pct = min(take_profit_pct, 0.2)  # 20% max

        return {
            'stop_loss_pct': stop_distance / entry_price,
            'take_profit_pct': take_profit_pct,
            'position_size': self.dynamic_position_size(100000)  # Example equity
        }

    def exit_signal(self, data: pd.DataFrame, position: Dict[str, Any]) -> bool:
        """
        Check if an exit signal is generated for the position.

        Args:
            data: DataFrame with market data
            position: Current position information

        Returns:
            bool: True if should exit, False otherwise
        """
        super().exit_signal(data, position)

        try:
            # Validate data first
            self.validate_data(data)
        except ValueError as e:
            logger.error(f"Data validation failed in exit_signal: {str(e)}")
            return False

        # Exit when price reverts to mean (based on Z-score)
        lookback = self.parameters['lookback_period']
        rolling_mean = data['close'].rolling(lookback).mean()
        rolling_std = data['close'].rolling(lookback).std() + self.parameters['epsilon']  # Prevent division by zero
        z_score = (data['close'] - rolling_mean) / rolling_std

        # Exit threshold from parameters
        exit_zscore = self.parameters['exit_zscore']

        # Check if price has reverted to mean
        if position['direction'] == 'long' and z_score.iloc[-1] >= exit_zscore:
            self.log_signal('exit', data, f"Exit long: Z-score={z_score.iloc[-1]:.2f} crossed exit threshold")
            return True
        elif position['direction'] == 'short' and z_score.iloc[-1] <= -exit_zscore:
            self.log_signal('exit', data, f"Exit short: Z-score={z_score.iloc[-1]:.2f} crossed exit threshold")
            return True

        # Check for holding period exit
        if 'entry_time' in position:
            entry_time = position['entry_time']
            max_holding = self.parameters['max_holding_period'] * 24 * 3600  # Convert days to seconds

            if time.time() - entry_time > max_holding:
                self.log_signal('exit', data, "Maximum holding period reached")
                return True

        return False

    def _optimize_for_time_of_day(self) -> None:
        """Optimize mean reversion for time of day"""
        super()._optimize_for_time_of_day()

        peak_hour = self.regime_characteristics.peak_hour
        if peak_hour in [8, 9]:
            # Tighter bands for low volatility session
            self.adaptive_thresholds = (35, 65)  # Less extreme RSI thresholds
            self.liquidity_factor = 0.8  # Lower volume requirement
            logger.info(f"Applied 08:00/09:00 UTC mean reversion optimizations")

        elif peak_hour == 11:
            # Wider bands for volatile session
            self.adaptive_thresholds = (25, 75)  # More extreme RSI thresholds
            self.liquidity_factor = 1.2  # Higher volume requirement
            logger.info(f"Applied 11:00 UTC mean reversion optimizations")

    def _optimize_for_directional_bias(self) -> None:
        """Optimize mean reversion for directional bias"""
        super()._optimize_for_directional_bias()

        original_generate = self.generate_signal
        bias = self.regime_characteristics.directional_bias

        if bias == DirectionalBias.UPWARD:
            # Asymmetric thresholds for upward bias
            oversold, overbought = self.adaptive_thresholds
            self.adaptive_thresholds = (oversold - 5, overbought)  # Easier to trigger longs (lower oversold)
            logger.info(f"Applied upward bias thresholds to {self.name}")

        elif bias == DirectionalBias.DOWNWARD:
            # Asymmetric thresholds for downward bias
            oversold, overbought = self.adaptive_thresholds
            self.adaptive_thresholds = (oversold, overbought - 5)  # Easier to trigger shorts (lower overbought)

            # Create biased signal generator for downward bias
            def biased_mean_reversion(data: pd.DataFrame) -> Optional[str]:
                signal = original_generate(data)

                # Enhance short signals
                if signal == 'short':
                    return 'short'
                elif signal == 'long':
                    # Add extra confirmation for longs in downward bias regime
                    if 'rsi_14' in data.columns:
                        rsi = data['rsi_14'].iloc[-1]
                    else:
                        rsi = self._get_cached_indicator('rsi', data).iloc[-1]

                    if rsi < 25:  # Only deeply oversold conditions
                        return 'long'
                    return None

                return signal

            self.generate_signal = biased_mean_reversion
            logger.info(f"Applied downward bias signal optimization to {self.name}")

    def _optimize_for_volatility_regime(self) -> None:
        """Optimize mean reversion for volatility regime"""
        super()._optimize_for_volatility_regime()

        vol_regime = self.regime_characteristics.volatility_regime
        oversold, overbought = self.adaptive_thresholds

        if vol_regime == VolatilityRegime.LOW:
            # Tighter thresholds for low volatility
            self.adaptive_thresholds = (oversold + 5, overbought - 5)
            logger.info(f"Applied low volatility threshold adjustment to {self.name}")

        elif vol_regime == VolatilityRegime.HIGH:
            # Wider thresholds for high volatility
            self.adaptive_thresholds = (max(20, oversold - 5), min(80, overbought + 5))
            logger.info(f"Applied high volatility threshold adjustment to {self.name}")

    def _calculate_rsi(self, prices, window=14):
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

        rs = avg_gain / (avg_loss + self.parameters['epsilon'])  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
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

    def _calculate_atr(self, data, window=14):
        """
        Calculate Average True Range.

        Args:
            data: DataFrame with OHLC data
            window: ATR calculation period

        Returns:
            Series with ATR values
        """
        high = data['high']
        low = data['low']
        close = data['close']

        # Handle first row where previous close is not available
        previous_close = close.shift(1)
        previous_close.iloc[0] = close.iloc[0]  # Use current close for first row

        tr1 = high - low
        tr2 = abs(high - previous_close)
        tr3 = abs(low - previous_close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window, min_periods=1).mean()

        return atr

    def cluster_fit(self, cluster_metrics: Dict) -> bool:
        """
        Determine if the mean reversion strategy is suitable for the given cluster.

        Args:
            cluster_metrics: Dict of cluster characteristics

        Returns:
            bool: True if the strategy fits the cluster, False otherwise
        """
        # Basic validation in case metrics are missing
        if not cluster_metrics or not isinstance(cluster_metrics, dict):
            return True

        # Accept if ADX is below 25 (mean reversion works better in less trendy markets)
        adx_mean = cluster_metrics.get('ADX_mean', 0)
        if adx_mean < 25:
            return True

        # Accept if autocorrelation is negative (mean reversion signal)
        autocorrelation = cluster_metrics.get('autocorrelation', 0)
        if autocorrelation < 0:
            return True

        # Accept if volatility is moderate to high (mean reversion needs some volatility)
        volatility_rank = cluster_metrics.get('volatility_rank', 0)
        if volatility_rank > 0.3:
            return True

        # Default to true if we don't have enough information
        return True

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

        if successful:
            # If trade was successful, fine-tune the strategy
            if self.regime_characteristics and self.regime_characteristics.cluster_id == regime_id:
                # Slightly modify thresholds to get more trades in successful regimes
                oversold, overbought = self.adaptive_thresholds

                # If it was a long trade, adjust oversold threshold
                if trade_result.get('direction') == 'long':
                    self.adaptive_thresholds = (min(40, oversold + 1), overbought)

                # If it was a short trade, adjust overbought threshold
                elif trade_result.get('direction') == 'short':
                    self.adaptive_thresholds = (oversold, max(60, overbought - 1))

                # Adjust exit threshold for faster profits
                if pnl_pct > 0.05:  # Good profit
                    self.parameters['exit_zscore'] = max(0.3, self.parameters['exit_zscore'] * 0.95)
        else:
            # If trade was unsuccessful, adjust parameters
            if self.regime_characteristics and self.regime_characteristics.cluster_id == regime_id:
                # Make thresholds more extreme for stronger signals
                oversold, overbought = self.adaptive_thresholds

                # If it was a losing long trade, make oversold more extreme
                if trade_result.get('direction') == 'long':
                    self.adaptive_thresholds = (max(20, oversold - 1), overbought)

                # If it was a losing short trade, make overbought more extreme
                elif trade_result.get('direction') == 'short':
                    self.adaptive_thresholds = (oversold, min(80, overbought + 1))

                # Adjust exit threshold to hold longer
                if pnl_pct < -0.03:  # Significant loss
                    self.parameters['exit_zscore'] = min(0.7, self.parameters['exit_zscore'] * 1.05)

        logger.debug(
            f"Adapted mean reversion parameters after trade: "
            f"adaptive_thresholds={self.adaptive_thresholds}, "
            f"exit_zscore={self.parameters['exit_zscore']:.2f}"
        )

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get strategy-specific performance metrics.

        Returns:
            Dict with performance metrics
        """
        # Basic performance metrics
        metrics = {
            'name': self.name,
            'adaptive_thresholds': self.adaptive_thresholds,
            'exit_zscore': self.parameters['exit_zscore'],
            'liquidity_factor': self.liquidity_factor
        }

        # Add more sophisticated metrics if trading history is available
        if hasattr(self, 'trade_history') and self.trade_history:
            # Calculate win rate
            wins = sum(1 for trade in self.trade_history if trade.get('pnl', 0) > 0)
            total_trades = len(self.trade_history)
            win_rate = wins / total_trades if total_trades > 0 else 0

            # Calculate average profit and loss
            profits = [trade.get('pnl_pct', 0) for trade in self.trade_history if trade.get('pnl', 0) > 0]
            losses = [trade.get('pnl_pct', 0) for trade in self.trade_history if trade.get('pnl', 0) <= 0]

            avg_profit = sum(profits) / len(profits) if profits else 0
            avg_loss = sum(losses) / len(losses) if losses else 0

            # Calculate expectancy
            expectancy = (win_rate * avg_profit) + ((1 - win_rate) * avg_loss)

            # Add to metrics
            metrics.update({
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'expectancy': expectancy,
                'profit_factor': -sum(profits) / sum(losses) if sum(losses) < 0 else 0
            })

        return metrics

    def clear_cache(self) -> None:
        """Clear the indicator cache to free memory"""
        self.indicator_cache = {}
        logger.debug(f"Cleared indicator cache for {self.name}")