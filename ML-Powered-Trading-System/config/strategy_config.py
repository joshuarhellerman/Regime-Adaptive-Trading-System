"""
Strategy configuration module that defines settings for trading strategies.

This module provides configuration classes for the various trading strategies
supported by the system, including breakout, momentum, mean reversion,
trend following, and volatility-based strategies.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Union
import os
from pathlib import Path

from .base_config import BaseConfig, ConfigManager


class StrategyType(Enum):
    """Enumeration of strategy types."""
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    VOLATILITY = "volatility"
    CUSTOM = "custom"


class TimeFrame(Enum):
    """Enumeration of timeframes for strategy execution."""
    TICK = "tick"
    SECOND = "1s"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


@dataclass
class BaseStrategyConfig(BaseConfig):
    """Base configuration for all strategies."""
    # Strategy identification
    strategy_id: str = ""
    strategy_type: StrategyType = StrategyType.CUSTOM
    description: str = ""

    # Activation settings
    enabled: bool = True
    auto_start: bool = False

    # Execution settings
    timeframe: TimeFrame = TimeFrame.MINUTE_5
    instruments: List[str] = field(default_factory=list)
    excluded_instruments: List[str] = field(default_factory=list)

    # Risk settings
    max_position_size: float = 1.0  # As a fraction of total capital
    max_risk_per_trade: float = 0.01  # Max risk per trade (1%)
    max_daily_drawdown: float = 0.03  # Max daily drawdown (3%)
    max_open_positions: int = 10

    # Performance monitoring
    min_sharpe_ratio: float = 0.5
    min_sortino_ratio: float = 0.7
    max_drawdown_threshold: float = 0.2  # Max drawdown threshold (20%)

    # Regime-specific settings
    enable_regime_adaptation: bool = True
    regime_sensitivity: float = 0.5  # How quickly to adapt to regime changes (0-1)

    def __post_init__(self):
        """Initialize the strategy configuration."""
        super().__post_init__()

        # Convert strategy_type from string if needed
        if isinstance(self.strategy_type, str):
            try:
                self.strategy_type = StrategyType(self.strategy_type.lower())
            except ValueError:
                print(f"Warning: Invalid strategy_type '{self.strategy_type}', defaulting to CUSTOM")
                self.strategy_type = StrategyType.CUSTOM

        # Convert timeframe from string if needed
        if isinstance(self.timeframe, str):
            try:
                self.timeframe = TimeFrame(self.timeframe.lower())
            except ValueError:
                print(f"Warning: Invalid timeframe '{self.timeframe}', defaulting to MINUTE_5")
                self.timeframe = TimeFrame.MINUTE_5

    def validate(self) -> List[str]:
        """Validate the strategy configuration."""
        errors = super().validate()

        # Validate strategy_id
        if not self.strategy_id:
            errors.append("strategy_id must not be empty")

        # Validate instruments
        if not self.instruments:
            errors.append("At least one instrument must be specified")

        # Validate risk parameters
        if not (0 < self.max_position_size <= 1):
            errors.append("max_position_size must be between 0 and 1")

        if not (0 < self.max_risk_per_trade <= 0.1):
            errors.append("max_risk_per_trade must be between 0 and 0.1 (10%)")

        if not (0 < self.max_daily_drawdown <= 0.2):
            errors.append("max_daily_drawdown must be between 0 and 0.2 (20%)")

        if self.max_open_positions <= 0:
            errors.append("max_open_positions must be positive")

        # Validate performance thresholds
        if self.min_sharpe_ratio < 0:
            errors.append("min_sharpe_ratio cannot be negative")

        if self.min_sortino_ratio < 0:
            errors.append("min_sortino_ratio cannot be negative")

        if not (0 < self.max_drawdown_threshold <= 1):
            errors.append("max_drawdown_threshold must be between 0 and 1 (100%)")

        # Validate regime settings
        if not (0 <= self.regime_sensitivity <= 1):
            errors.append("regime_sensitivity must be between 0 and 1")

        return errors


@dataclass
class BreakoutStrategyConfig(BaseStrategyConfig):
    """Configuration for breakout strategies."""
    # Specific breakout parameters
    lookback_periods: int = 20
    breakout_threshold: float = 2.0  # Standard deviations
    confirmation_periods: int = 3

    # Volatility adjustment
    volatility_lookback: int = 50
    enable_volatility_adjustment: bool = True
    min_volatility_threshold: float = 0.001  # Minimum volatility for signal generation

    # Signal filtering
    use_volume_confirmation: bool = True
    volume_threshold: float = 1.5  # Volume must be X times average

    def __post_init__(self):
        """Initialize the breakout strategy configuration."""
        super().__post_init__()
        self.strategy_type = StrategyType.BREAKOUT

    def validate(self) -> List[str]:
        """Validate the breakout strategy configuration."""
        errors = super().validate()

        # Validate lookback periods
        if self.lookback_periods <= 0:
            errors.append("lookback_periods must be positive")

        # Validate breakout threshold
        if self.breakout_threshold <= 0:
            errors.append("breakout_threshold must be positive")

        # Validate confirmation periods
        if self.confirmation_periods < 0:
            errors.append("confirmation_periods cannot be negative")

        # Validate volatility settings
        if self.volatility_lookback <= 0:
            errors.append("volatility_lookback must be positive")

        if self.min_volatility_threshold <= 0:
            errors.append("min_volatility_threshold must be positive")

        # Validate volume settings
        if self.volume_threshold <= 0:
            errors.append("volume_threshold must be positive")

        return errors


@dataclass
class MomentumStrategyConfig(BaseStrategyConfig):
    """Configuration for momentum strategies."""
    # Momentum calculation
    lookback_periods: int = 90
    smoothing_periods: int = 14

    # Signal generation
    signal_threshold: float = 0.05  # 5% momentum required for signal
    exit_threshold: float = 0.02  # Exit when momentum drops below 2%

    # Position sizing
    position_scaling: bool = True  # Scale position size with momentum strength
    max_scaling_factor: float = 3.0  # Maximum position sizing multiplier

    # Timing
    avoid_earnings_announcements: bool = True
    pre_earnings_days: int = 5  # Avoid trading this many days before earnings

    def __post_init__(self):
        """Initialize the momentum strategy configuration."""
        super().__post_init__()
        self.strategy_type = StrategyType.MOMENTUM

    def validate(self) -> List[str]:
        """Validate the momentum strategy configuration."""
        errors = super().validate()

        # Validate lookback periods
        if self.lookback_periods <= 0:
            errors.append("lookback_periods must be positive")

        # Validate smoothing periods
        if self.smoothing_periods <= 0:
            errors.append("smoothing_periods must be positive")

        # Validate thresholds
        if self.signal_threshold <= 0:
            errors.append("signal_threshold must be positive")

        if self.exit_threshold < 0:
            errors.append("exit_threshold cannot be negative")

        if self.exit_threshold >= self.signal_threshold:
            errors.append("exit_threshold should be less than signal_threshold")

        # Validate position sizing
        if self.max_scaling_factor <= 0:
            errors.append("max_scaling_factor must be positive")

        # Validate timing
        if self.pre_earnings_days < 0:
            errors.append("pre_earnings_days cannot be negative")

        return errors


@dataclass
class MeanReversionStrategyConfig(BaseStrategyConfig):
    """Configuration for mean reversion strategies."""
    # Mean calculation
    lookback_periods: int = 20
    mean_type: str = "sma"  # "sma" or "ema"

    # Deviation parameters
    entry_std_dev: float = 2.0  # Enter when price is X std devs from mean
    exit_std_dev: float = 0.5  # Exit when price is within X std devs of mean

    # Filters
    trend_filter_enable: bool = True
    trend_filter_periods: int = 200
    oversold_threshold: float = 30.0  # RSI below this is oversold
    overbought_threshold: float = 70.0  # RSI above this is overbought

    # Trade management
    max_days_in_trade: int = 10
    use_stop_loss: bool = True
    stop_loss_std_dev: float = 4.0  # Stop loss at X std devs from mean

    def __post_init__(self):
        """Initialize the mean reversion strategy configuration."""
        super().__post_init__()
        self.strategy_type = StrategyType.MEAN_REVERSION

    def validate(self) -> List[str]:
        """Validate the mean reversion strategy configuration."""
        errors = super().validate()

        # Validate lookback periods
        if self.lookback_periods <= 0:
            errors.append("lookback_periods must be positive")

        # Validate mean type
        if self.mean_type not in ["sma", "ema"]:
            errors.append("mean_type must be either 'sma' or 'ema'")

        # Validate deviation parameters
        if self.entry_std_dev <= 0:
            errors.append("entry_std_dev must be positive")

        if self.exit_std_dev < 0:
            errors.append("exit_std_dev cannot be negative")

        if self.entry_std_dev <= self.exit_std_dev:
            errors.append("entry_std_dev should be greater than exit_std_dev")

        # Validate filter settings
        if self.trend_filter_periods <= 0:
            errors.append("trend_filter_periods must be positive")

        if not (0 <= self.oversold_threshold <= 100):
            errors.append("oversold_threshold must be between 0 and 100")

        if not (0 <= self.overbought_threshold <= 100):
            errors.append("overbought_threshold must be between 0 and 100")

        if self.oversold_threshold >= self.overbought_threshold:
            errors.append("oversold_threshold should be less than overbought_threshold")

        # Validate trade management
        if self.max_days_in_trade <= 0:
            errors.append("max_days_in_trade must be positive")

        if self.use_stop_loss and self.stop_loss_std_dev <= 0:
            errors.append("stop_loss_std_dev must be positive")

        return errors


@dataclass
class TrendFollowingStrategyConfig(BaseStrategyConfig):
    """Configuration for trend following strategies."""
    # Trend identification
    fast_ma_periods: int = 20
    slow_ma_periods: int = 50
    ma_type: str = "ema"  # "sma" or "ema"

    # Signal generation
    entry_confirmation_periods: int = 3  # Consecutive periods above/below MA
    exit_confirmation_periods: int = 2   # Consecutive periods crossing MA

    # Additional filters
    use_atr_filter: bool = True
    atr_periods: int = 14
    min_atr_multiple: float = 1.5  # Minimum ATR multiple for volatility

    # Trend strength
    use_adx_filter: bool = True
    adx_periods: int = 14
    min_adx_value: float = 25  # Minimum ADX for strong trend

    # Position management
    trailing_stop_enable: bool = True
    trailing_stop_atr_multiple: float = 3.0
    profit_target_atr_multiple: float = 5.0

    def __post_init__(self):
        """Initialize the trend following strategy configuration."""
        super().__post_init__()
        self.strategy_type = StrategyType.TREND_FOLLOWING

    def validate(self) -> List[str]:
        """Validate the trend following strategy configuration."""
        errors = super().validate()

        # Validate moving average periods
        if self.fast_ma_periods <= 0:
            errors.append("fast_ma_periods must be positive")

        if self.slow_ma_periods <= 0:
            errors.append("slow_ma_periods must be positive")

        if self.fast_ma_periods >= self.slow_ma_periods:
            errors.append("fast_ma_periods should be less than slow_ma_periods")

        # Validate MA type
        if self.ma_type not in ["sma", "ema"]:
            errors.append("ma_type must be either 'sma' or 'ema'")

        # Validate confirmation periods
        if self.entry_confirmation_periods <= 0:
            errors.append("entry_confirmation_periods must be positive")

        if self.exit_confirmation_periods <= 0:
            errors.append("exit_confirmation_periods must be positive")

        # Validate ATR settings
        if self.use_atr_filter:
            if self.atr_periods <= 0:
                errors.append("atr_periods must be positive")

            if self.min_atr_multiple <= 0:
                errors.append("min_atr_multiple must be positive")

        # Validate ADX settings
        if self.use_adx_filter:
            if self.adx_periods <= 0:
                errors.append("adx_periods must be positive")

            if not (0 <= self.min_adx_value <= 100):
                errors.append("min_adx_value must be between 0 and 100")

        # Validate position management
        if self.trailing_stop_enable and self.trailing_stop_atr_multiple <= 0:
            errors.append("trailing_stop_atr_multiple must be positive")

        if self.profit_target_atr_multiple <= 0:
            errors.append("profit_target_atr_multiple must be positive")

        return errors


@dataclass
class VolatilityStrategyConfig(BaseStrategyConfig):
    """Configuration for volatility-based strategies."""
    # Volatility measurement
    volatility_periods: int = 20
    volatility_metric: str = "atr"  # "atr", "stdev", "parkinson", "garman_klass"
    volatility_lookback: int = 252  # For historical percentile calculation

    # Signal generation
    high_vol_percentile: float = 80.0  # High volatility threshold (percentile)
    low_vol_percentile: float = 20.0   # Low volatility threshold (percentile)
    mean_reversion_threshold: float = 2.0  # Std devs for mean reversion opportunities
    breakout_threshold: float = 1.5  # Volatility multiple for breakout confirmation

    # Strategy adjustments
    position_size_vol_scaling: bool = True  # Scale position size inverse to volatility
    vol_scaling_factor: float = 1.0  # Volatility position scaling factor

    # Risk adjustments
    increase_stops_in_high_vol: bool = True
    high_vol_stop_multiplier: float = 1.5  # Increase stops by this factor in high vol

    def __post_init__(self):
        """Initialize the volatility strategy configuration."""
        super().__post_init__()
        self.strategy_type = StrategyType.VOLATILITY

    def validate(self) -> List[str]:
        """Validate the volatility strategy configuration."""
        errors = super().validate()

        # Validate volatility periods
        if self.volatility_periods <= 0:
            errors.append("volatility_periods must be positive")

        # Validate volatility metric
        valid_metrics = ["atr", "stdev", "parkinson", "garman_klass"]
        if self.volatility_metric not in valid_metrics:
            errors.append(f"volatility_metric must be one of {valid_metrics}")

        # Validate lookback
        if self.volatility_lookback <= 0:
            errors.append("volatility_lookback must be positive")

        # Validate percentiles
        if not (0 <= self.high_vol_percentile <= 100):
            errors.append("high_vol_percentile must be between 0 and 100")

        if not (0 <= self.low_vol_percentile <= 100):
            errors.append("low_vol_percentile must be between 0 and 100")

        if self.low_vol_percentile >= self.high_vol_percentile:
            errors.append("low_vol_percentile should be less than high_vol_percentile")

        # Validate thresholds
        if self.mean_reversion_threshold <= 0:
            errors.append("mean_reversion_threshold must be positive")

        if self.breakout_threshold <= 0:
            errors.append("breakout_threshold must be positive")

        # Validate scaling factor
        if self.vol_scaling_factor <= 0:
            errors.append("vol_scaling_factor must be positive")

        # Validate risk adjustments
        if self.increase_stops_in_high_vol and self.high_vol_stop_multiplier <= 0:
            errors.append("high_vol_stop_multiplier must be positive")

        return errors


@dataclass
class StrategyConfig(BaseConfig):
    """
    Main strategy configuration that contains all strategy-specific configurations.

    This class serves as a container for all strategy configurations and provides
    methods to access and manage them.
    """
    # Strategy management
    max_active_strategies: int = 5
    strategy_optimization_interval_hours: int = 24

    # Default strategy configurations
    breakout: BreakoutStrategyConfig = field(default_factory=BreakoutStrategyConfig)
    momentum: MomentumStrategyConfig = field(default_factory=MomentumStrategyConfig)
    mean_reversion: MeanReversionStrategyConfig = field(default_factory=MeanReversionStrategyConfig)
    trend_following: TrendFollowingStrategyConfig = field(default_factory=TrendFollowingStrategyConfig)
    volatility: VolatilityStrategyConfig = field(default_factory=VolatilityStrategyConfig)

    # Custom strategies dictionary (strategy_id -> config_dict)
    custom_strategies: Dict[str, Dict] = field(default_factory=dict)

    # Strategy allocation weights (strategy_id -> weight)
    allocation_weights: Dict[str, float] = field(default_factory=dict)

    # Strategy restrictions
    excluded_markets: Set[str] = field(default_factory=set)
    trading_hours_restriction: bool = True

    def __post_init__(self):
        """Initialize the strategy configuration."""
        super().__post_init__()

        # Process custom strategies
        self._custom_strategy_configs = {}
        for strategy_id, strategy_dict in self.custom_strategies.items():
            strategy_type = strategy_dict.get("strategy_type", "custom")

            if isinstance(strategy_type, str):
                try:
                    strategy_type = StrategyType(strategy_type.lower())
                except ValueError:
                    strategy_type = StrategyType.CUSTOM

            # Create appropriate strategy config based on type
            if strategy_type == StrategyType.BREAKOUT:
                config = BreakoutStrategyConfig(**strategy_dict)
            elif strategy_type == StrategyType.MOMENTUM:
                config = MomentumStrategyConfig(**strategy_dict)
            elif strategy_type == StrategyType.MEAN_REVERSION:
                config = MeanReversionStrategyConfig(**strategy_dict)
            elif strategy_type == StrategyType.TREND_FOLLOWING:
                config = TrendFollowingStrategyConfig(**strategy_dict)
            elif strategy_type == StrategyType.VOLATILITY:
                config = VolatilityStrategyConfig(**strategy_dict)
            else:
                config = BaseStrategyConfig(**strategy_dict)

            # Ensure strategy_id is set
            config.strategy_id = strategy_id

            # Store in dictionary
            self._custom_strategy_configs[strategy_id] = config

    def get_strategy_config(self, strategy_id: str) -> Optional[BaseStrategyConfig]:
        """
        Get a strategy configuration by ID.

        Args:
            strategy_id: The ID of the strategy

        Returns:
            The strategy configuration if found, None otherwise
        """
        # Check built-in strategies first
        if strategy_id == "breakout":
            return self.breakout
        elif strategy_id == "momentum":
            return self.momentum
        elif strategy_id == "mean_reversion":
            return self.mean_reversion
        elif strategy_id == "trend_following":
            return self.trend_following
        elif strategy_id == "volatility":
            return self.volatility

        # Then check custom strategies
        return self._custom_strategy_configs.get(strategy_id)

    def add_custom_strategy(self, strategy_config: BaseStrategyConfig) -> None:
        """
        Add a custom strategy configuration.

        Args:
            strategy_config: The strategy configuration to add
        """
        strategy_id = strategy_config.strategy_id
        if not strategy_id:
            raise ValueError("Strategy ID cannot be empty")

        # Store in dictionary
        self._custom_strategy_configs[strategy_id] = strategy_config

        # Update custom_strategies for serialization
        self.custom_strategies[strategy_id] = strategy_config.to_dict()

    def remove_custom_strategy(self, strategy_id: str) -> bool:
        """
        Remove a custom strategy configuration.

        Args:
            strategy_id: The ID of the strategy to remove

        Returns:
            True if the strategy was removed, False otherwise
        """
        if strategy_id in self._custom_strategy_configs:
            del self._custom_strategy_configs[strategy_id]
            del self.custom_strategies[strategy_id]
            return True
        return False

    def get_all_strategy_ids(self) -> List[str]:
        """
        Get the IDs of all strategies.

        Returns:
            A list of strategy IDs
        """
        # Start with built-in strategies
        strategy_ids = ["breakout", "momentum", "mean_reversion", "trend_following", "volatility"]

        # Add custom strategies
        strategy_ids.extend(self._custom_strategy_configs.keys())

        return strategy_ids

    def get_enabled_strategy_ids(self) -> List[str]:
        """
        Get the IDs of all enabled strategies.

        Returns:
            A list of enabled strategy IDs
        """
        enabled_ids = []

        # Check built-in strategies
        if self.breakout.enabled:
            enabled_ids.append("breakout")
        if self.momentum.enabled:
            enabled_ids.append("momentum")
        if self.mean_reversion.enabled:
            enabled_ids.append("mean_reversion")
        if self.trend_following.enabled:
            enabled_ids.append("trend_following")
        if self.volatility.enabled:
            enabled_ids.append("volatility")

        # Check custom strategies
        for strategy_id, config in self._custom_strategy_configs.items():
            if config.enabled:
                enabled_ids.append(strategy_id)

        return enabled_ids

    def validate(self) -> List[str]:
        """Validate the strategy configuration."""
        errors = super().validate()

        # Validate max_active_strategies
        if self.max_active_strategies <= 0:
            errors.append("max_active_strategies must be positive")

        # Validate strategy_optimization_interval_hours
        if self.strategy_optimization_interval_hours <= 0:
            errors.append("strategy_optimization_interval_hours must be positive")

        # Validate allocation weights
        total_weight = sum(self.allocation_weights.values())
        if total_weight > 0 and abs(total_weight - 1.0) > 0.001:
            errors.append("Sum of allocation_weights should be close to 1.0")

        # Validate individual strategy configurations
        errors.extend(self.breakout.validate())
        errors.extend(self.momentum.validate())
        errors.extend(self.mean_reversion.validate())
        errors.extend(self.trend_following.validate())
        errors.extend(self.volatility.validate())

        # Validate custom strategy configurations
        for strategy_id, config in self._custom_strategy_configs.items():
            config_errors = config.validate()
            for error in config_errors:
                errors.append(f"In strategy '{strategy_id}': {error}")

        return errors


def get_strategy_config(config_path: Optional[Union[str, Path]] = None) -> StrategyConfig:
    """
    Get the strategy configuration.

    Args:
        config_path: Optional path to a configuration file. If not provided,
                    the default path from the ConfigManager will be used.

    Returns:
        The strategy configuration.
    """
    if config_path is None:
        config_path = ConfigManager.get_config_path("strategy")

    return ConfigManager.load_config(
        StrategyConfig,
        config_path=config_path,
        env_prefix="TRADING_STRATEGY",
        reload=False
    )