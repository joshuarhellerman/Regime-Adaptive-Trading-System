"""
Trading mode configuration module that defines settings for paper trading
and live trading transitions.

This module provides configuration for managing transitions between paper
trading and live trading, including validation rules, capital allocation,
and risk limits.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Union
from pathlib import Path
import os

from .base_config import BaseConfig, ConfigManager


class TradingMode(Enum):
    """Enumeration of trading modes."""
    PAPER = "paper"
    SHADOW = "shadow"
    PILOT = "pilot"
    LIVE = "live"


class TransitionStage(Enum):
    """Enumeration of transition stages."""
    VALIDATION = "validation"
    SHADOW_TRADING = "shadow_trading"
    PILOT_TRADING = "pilot_trading"
    SCALING = "scaling"
    FULL_DEPLOYMENT = "full_deployment"


@dataclass
class PaperTradingConfig(BaseConfig):
    """Configuration for paper trading."""
    # Simulation settings
    initial_capital: float = 100000.0  # $100,000 initial capital
    base_currency: str = "USD"
    enable_margin: bool = False
    margin_multiplier: float = 2.0  # 2x leverage when margin is enabled

    # Realism settings
    simulate_latency: bool = True
    latency_mean_ms: int = 50
    latency_stddev_ms: int = 20

    # Market impact simulation
    simulate_market_impact: bool = True
    market_impact_factor: float = 0.1

    # Fill simulation
    simulate_partial_fills: bool = True
    fill_probability: float = 0.9  # 90% chance of complete fill

    # Slippage simulation
    simulate_slippage: bool = True
    slippage_model: str = "gaussian"  # "gaussian", "fixed", "proportional"
    slippage_mean_bps: float = 2.0    # 2 basis points mean slippage
    slippage_stddev_bps: float = 1.0  # 1 basis point standard deviation

    # Fee simulation
    simulate_fees: bool = True
    maker_fee_rate: float = 0.0010  # 0.10% maker fee
    taker_fee_rate: float = 0.0020  # 0.20% taker fee

    def validate(self) -> List[str]:
        """Validate the paper trading configuration."""
        errors = super().validate()

        # Validate initial capital
        if self.initial_capital <= 0:
            errors.append("initial_capital must be positive")

        # Validate base currency
        if not self.base_currency:
            errors.append("base_currency must not be empty")

        # Validate margin settings
        if self.enable_margin and self.margin_multiplier <= 1:
            errors.append("margin_multiplier must be greater than 1 when enable_margin is True")

        # Validate latency settings
        if self.simulate_latency:
            if self.latency_mean_ms < 0:
                errors.append("latency_mean_ms cannot be negative")

            if self.latency_stddev_ms < 0:
                errors.append("latency_stddev_ms cannot be negative")

        # Validate market impact settings
        if self.simulate_market_impact and self.market_impact_factor < 0:
            errors.append("market_impact_factor cannot be negative")

        # Validate fill simulation settings
        if self.simulate_partial_fills and not (0 <= self.fill_probability <= 1):
            errors.append("fill_probability must be between 0 and 1")

        # Validate slippage settings
        if self.simulate_slippage:
            valid_models = ["gaussian", "fixed", "proportional"]
            if self.slippage_model not in valid_models:
                errors.append(f"slippage_model must be one of {valid_models}")

            if self.slippage_mean_bps < 0:
                errors.append("slippage_mean_bps cannot be negative")

            if self.slippage_model == "gaussian" and self.slippage_stddev_bps < 0:
                errors.append("slippage_stddev_bps cannot be negative")

        # Validate fee settings
        if self.simulate_fees:
            if self.maker_fee_rate < 0:
                errors.append("maker_fee_rate cannot be negative")

            if self.taker_fee_rate < 0:
                errors.append("taker_fee_rate cannot be negative")

        return errors


@dataclass
class LiveTradingConfig(BaseConfig):
    """Configuration for live trading."""
    # Capital allocation
    initial_allocation: float = 10000.0  # $10,000 initial allocation
    max_allocation: float = 100000.0     # $100,000 maximum allocation
    allocation_currency: str = "USD"

    # Risk limits
    max_drawdown_percent: float = 5.0    # 5% maximum drawdown
    emergency_stop_percent: float = 10.0  # 10% emergency stop

    # Position limits
    max_position_size_percent: float = 2.0  # 2% max position size
    max_open_positions: int = 10
    max_open_notional: float = 50000.0  # $50,000 max open notional

    # Order limits
    max_order_size: float = 5000.0  # $5,000 max order size
    max_orders_per_day: int = 100

    # Exchange-specific limits
    exchange_limits: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Emergency contact
    emergency_contact_email: str = ""
    emergency_contact_phone: str = ""

    def validate(self) -> List[str]:
        """Validate the live trading configuration."""
        errors = super().validate()

        # Validate capital allocation
        if self.initial_allocation <= 0:
            errors.append("initial_allocation must be positive")

        if self.max_allocation < self.initial_allocation:
            errors.append("max_allocation must be greater than or equal to initial_allocation")

        if not self.allocation_currency:
            errors.append("allocation_currency must not be empty")

        # Validate risk limits
        if self.max_drawdown_percent <= 0 or self.max_drawdown_percent > 100:
            errors.append("max_drawdown_percent must be between 0 and 100")

        if self.emergency_stop_percent <= 0 or self.emergency_stop_percent > 100:
            errors.append("emergency_stop_percent must be between 0 and 100")

        if self.emergency_stop_percent <= self.max_drawdown_percent:
            errors.append("emergency_stop_percent should be greater than max_drawdown_percent")

        # Validate position limits
        if self.max_position_size_percent <= 0 or self.max_position_size_percent > 100:
            errors.append("max_position_size_percent must be between 0 and 100")

        if self.max_open_positions <= 0:
            errors.append("max_open_positions must be positive")

        if self.max_open_notional <= 0:
            errors.append("max_open_notional must be positive")

        # Validate order limits
        if self.max_order_size <= 0:
            errors.append("max_order_size must be positive")

        if self.max_orders_per_day <= 0:
            errors.append("max_orders_per_day must be positive")

        return errors


@dataclass
class ShadowTradingConfig(BaseConfig):
    """Configuration for shadow trading (paper trading alongside live trading)."""
    # Shadow mode settings
    enable_shadow_mode: bool = True
    shadow_duration_days: int = 14  # Run shadow mode for 14 days

    # Performance metrics for comparison
    metrics_to_compare: List[str] = field(default_factory=lambda: [
        "sharpe_ratio", "max_drawdown", "total_return", "win_rate"
    ])

    # Comparison thresholds
    performance_variance_threshold: float = 0.10  # 10% variance threshold
    slippage_comparison_threshold: float = 0.05  # 5% slippage comparison threshold

    # Reporting
    generate_shadow_reports: bool = True
    report_frequency: str = "daily"  # "hourly", "daily", "weekly"

    def validate(self) -> List[str]:
        """Validate the shadow trading configuration."""
        errors = super().validate()

        # Only validate if shadow mode is enabled
        if not self.enable_shadow_mode:
            return errors

        # Validate shadow duration
        if self.shadow_duration_days <= 0:
            errors.append("shadow_duration_days must be positive")

        # Validate metrics to compare
        if not self.metrics_to_compare:
            errors.append("At least one metric must be specified in metrics_to_compare")

        # Validate comparison thresholds
        if self.performance_variance_threshold < 0:
            errors.append("performance_variance_threshold cannot be negative")

        if self.slippage_comparison_threshold < 0:
            errors.append("slippage_comparison_threshold cannot be negative")

        # Validate reporting
        if self.generate_shadow_reports:
            valid_frequencies = ["hourly", "daily", "weekly"]
            if self.report_frequency not in valid_frequencies:
                errors.append(f"report_frequency must be one of {valid_frequencies}")

        return errors


@dataclass
class PilotTradingConfig(BaseConfig):
    """Configuration for pilot trading (small allocation to live trading)."""
    # Pilot mode settings
    enable_pilot_mode: bool = True
    pilot_duration_days: int = 30  # Run pilot mode for 30 days

    # Allocation settings
    pilot_allocation_percent: float = 10.0  # 10% of max allocation
    increase_allocation_schedule: Dict[int, float] = field(default_factory=lambda: {
        7: 15.0,   # 15% after 7 days
        14: 25.0,  # 25% after 14 days
        21: 50.0,  # 50% after 21 days
        30: 100.0  # 100% after 30 days
    })

    # Performance requirements
    min_sharpe_ratio: float = 0.5
    max_acceptable_drawdown: float = 10.0  # 10% max acceptable drawdown

    # Risk limits (override live trading limits during pilot)
    pilot_max_position_size_percent: float = 1.0  # 1% max position size during pilot
    pilot_max_open_positions: int = 5

    def validate(self) -> List[str]:
        """Validate the pilot trading configuration."""
        errors = super().validate()

        # Only validate if pilot mode is enabled
        if not self.enable_pilot_mode:
            return errors

        # Validate pilot duration
        if self.pilot_duration_days <= 0:
            errors.append("pilot_duration_days must be positive")

        # Validate allocation settings
        if self.pilot_allocation_percent <= 0 or self.pilot_allocation_percent > 100:
            errors.append("pilot_allocation_percent must be between 0 and 100")

        # Validate allocation schedule
        for day, allocation in self.increase_allocation_schedule.items():
            if day <= 0:
                errors.append("Days in increase_allocation_schedule must be positive")

            if allocation <= 0 or allocation > 100:
                errors.append("Allocations in increase_allocation_schedule must be between 0 and 100")

        # Validate sorted schedule
        days = sorted(self.increase_allocation_schedule.keys())
        allocations = [self.increase_allocation_schedule[day] for day in days]
        if allocations != sorted(allocations):
            errors.append("Allocations in increase_allocation_schedule must be non-decreasing")

        # Validate performance requirements
        if self.min_sharpe_ratio < 0:
            errors.append("min_sharpe_ratio cannot be negative")

        if self.max_acceptable_drawdown <= 0 or self.max_acceptable_drawdown > 100:
            errors.append("max_acceptable_drawdown must be between 0 and 100")

        # Validate risk limits
        if self.pilot_max_position_size_percent <= 0 or self.pilot_max_position_size_percent > 100:
            errors.append("pilot_max_position_size_percent must be between 0 and 100")

        if self.pilot_max_open_positions <= 0:
            errors.append("pilot_max_open_positions must be positive")

        return errors


@dataclass
class TransitionValidationConfig(BaseConfig):
    """Configuration for validation requirements before transitioning to live trading."""
    # Performance validation
    min_paper_trading_days: int = 30
    min_paper_trading_trades: int = 100
    min_sharpe_ratio: float = 1.0
    min_profit_factor: float = 1.5
    max_drawdown_percent: float = 15.0

    # System validation
    validate_connectivity: bool = True
    validate_order_execution: bool = True
    validate_data_feeds: bool = True
    validate_risk_limits: bool = True

    # Operational validation
    validate_monitoring_systems: bool = True
    validate_alerting: bool = True
    validate_disaster_recovery: bool = True

    # Capital validation
    validate_sufficient_capital: bool = True
    min_capital_requirement: float = 10000.0  # $10,000 minimum capital

    # Manual approval settings
    require_manual_approval: bool = True
    approvers: List[str] = field(default_factory=list)

    def validate(self) -> List[str]:
        """Validate the transition validation configuration."""
        errors = super().validate()

        # Validate performance validation
        if self.min_paper_trading_days <= 0:
            errors.append("min_paper_trading_days must be positive")

        if self.min_paper_trading_trades <= 0:
            errors.append("min_paper_trading_trades must be positive")

        if self.min_sharpe_ratio <= 0:
            errors.append("min_sharpe_ratio must be positive")

        if self.min_profit_factor <= 0:
            errors.append("min_profit_factor must be positive")

        if self.max_drawdown_percent <= 0 or self.max_drawdown_percent > 100:
            errors.append("max_drawdown_percent must be between 0 and 100")

        # Validate capital validation
        if self.validate_sufficient_capital and self.min_capital_requirement <= 0:
            errors.append("min_capital_requirement must be positive")

        # Validate approvers
        if self.require_manual_approval and not self.approvers:
            errors.append("At least one approver must be specified when require_manual_approval is True")

        return errors


@dataclass
class AutomatedTransitionConfig(BaseConfig):
    """Configuration for automated transitions between trading modes."""
    # Automated transition settings
    enable_automated_transitions: bool = False
    transition_check_interval_hours: int = 24

    # Transition conditions
    auto_promote_to_shadow: bool = True
    auto_promote_to_pilot: bool = False
    auto_promote_to_live: bool = False

    # Demotion conditions
    auto_demote_on_failure: bool = True
    demotion_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "drawdown_percent": 20.0,
        "sharpe_ratio": 0.0,
        "consecutive_losses": 10
    })

    # Emergency circuit breakers
    enable_emergency_circuit_breakers: bool = True
    circuit_breaker_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "daily_loss_percent": 5.0,
        "position_limit_percent": 150.0,
        "order_error_rate": 10.0
    })

    def validate(self) -> List[str]:
        """Validate the automated transition configuration."""
        errors = super().validate()

        # Only validate if automated transitions are enabled
        if not self.enable_automated_transitions:
            return errors

        # Validate transition check interval
        if self.transition_check_interval_hours <= 0:
            errors.append("transition_check_interval_hours must be positive")

        # Validate demotion thresholds
        for threshold_name, threshold_value in self.demotion_thresholds.items():
            if threshold_name == "drawdown_percent" and (threshold_value <= 0 or threshold_value > 100):
                errors.append("demotion_thresholds[drawdown_percent] must be between 0 and 100")

            if threshold_name == "consecutive_losses" and threshold_value <= 0:
                errors.append("demotion_thresholds[consecutive_losses] must be positive")

        # Validate circuit breaker thresholds
        if self.enable_emergency_circuit_breakers:
            for threshold_name, threshold_value in self.circuit_breaker_thresholds.items():
                if threshold_value <= 0:
                    errors.append(f"circuit_breaker_thresholds[{threshold_name}] must be positive")

        return errors


@dataclass
class TradingModeConfig(BaseConfig):
    """
    Main trading mode configuration that contains all trading mode settings.

    This class serves as a container for paper trading, live trading, and
    transition settings.
    """
    # Current trading mode
    current_mode: TradingMode = TradingMode.PAPER
    current_transition_stage: TransitionStage = TransitionStage.VALIDATION

    # Paper trading configuration
    paper_trading: PaperTradingConfig = field(default_factory=PaperTradingConfig)

    # Live trading configuration
    live_trading: LiveTradingConfig = field(default_factory=LiveTradingConfig)

    # Shadow trading configuration
    shadow_trading: ShadowTradingConfig = field(default_factory=ShadowTradingConfig)

    # Pilot trading configuration
    pilot_trading: PilotTradingConfig = field(default_factory=PilotTradingConfig)

    # Transition validation configuration
    transition_validation: TransitionValidationConfig = field(default_factory=TransitionValidationConfig)

    # Automated transition configuration
    automated_transition: AutomatedTransitionConfig = field(default_factory=AutomatedTransitionConfig)

    # Trading schedule
    trading_hours_start: str = "09:30"  # 9:30 AM
    trading_hours_end: str = "16:00"    # 4:00 PM
    trading_timezone: str = "America/New_York"
    trading_days: List[str] = field(default_factory=lambda: [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"
    ])

    # Trading holidays
    holidays: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize the trading mode configuration."""
        super().__post_init__()

        # Convert current_mode from string if needed
        if isinstance(self.current_mode, str):
            try:
                self.current_mode = TradingMode(self.current_mode.lower())
            except ValueError:
                print(f"Warning: Invalid current_mode '{self.current_mode}', defaulting to PAPER")
                self.current_mode = TradingMode.PAPER

        # Convert current_transition_stage from string if needed
        if isinstance(self.current_transition_stage, str):
            try:
                self.current_transition_stage = TransitionStage(self.current_transition_stage.lower())
            except ValueError:
                print(f"Warning: Invalid current_transition_stage '{self.current_transition_stage}', defaulting to VALIDATION")
                self.current_transition_stage = TransitionStage.VALIDATION

        # Ensure paper_trading is a PaperTradingConfig object
        if isinstance(self.paper_trading, dict):
            self.paper_trading = PaperTradingConfig(**self.paper_trading)

        # Ensure live_trading is a LiveTradingConfig object
        if isinstance(self.live_trading, dict):
            self.live_trading = LiveTradingConfig(**self.live_trading)

        # Ensure shadow_trading is a ShadowTradingConfig object
        if isinstance(self.shadow_trading, dict):
            self.shadow_trading = ShadowTradingConfig(**self.shadow_trading)

        # Ensure pilot_trading is a PilotTradingConfig object
        if isinstance(self.pilot_trading, dict):
            self.pilot_trading = PilotTradingConfig(**self.pilot_trading)

        # Ensure transition_validation is a TransitionValidationConfig object
        if isinstance(self.transition_validation, dict):
            self.transition_validation = TransitionValidationConfig(**self.transition_validation)

        # Ensure automated_transition is an AutomatedTransitionConfig object
        if isinstance(self.automated_transition, dict):
            self.automated_transition = AutomatedTransitionConfig(**self.automated_transition)

    def is_live_mode(self) -> bool:
        """
        Check if the system is in live trading mode.

        Returns:
            True if in live mode, False otherwise
        """
        return self.current_mode == TradingMode.LIVE

    def is_paper_mode(self) -> bool:
        """
        Check if the system is in paper trading mode.

        Returns:
            True if in paper mode, False otherwise
        """
        return self.current_mode == TradingMode.PAPER

    def is_shadow_mode(self) -> bool:
        """
        Check if the system is in shadow trading mode.

        Returns:
            True if in shadow mode, False otherwise
        """
        return self.current_mode == TradingMode.SHADOW

    def is_pilot_mode(self) -> bool:
        """
        Check if the system is in pilot trading mode.

        Returns:
            True if in pilot mode, False otherwise
        """
        return self.current_mode == TradingMode.PILOT

    def validate(self) -> List[str]:
        """Validate the trading mode configuration."""
        errors = super().validate()

        # Validate paper trading configuration
        paper_errors = self.paper_trading.validate()
        for error in paper_errors:
            errors.append(f"In paper_trading: {error}")

        # Validate live trading configuration
        live_errors = self.live_trading.validate()
        for error in live_errors:
            errors.append(f"In live_trading: {error}")

        # Validate shadow trading configuration
        shadow_errors = self.shadow_trading.validate()
        for error in shadow_errors:
            errors.append(f"In shadow_trading: {error}")

        # Validate pilot trading configuration
        pilot_errors = self.pilot_trading.validate()
        for error in pilot_errors:
            errors.append(f"In pilot_trading: {error}")

        # Validate transition validation configuration
        validation_errors = self.transition_validation.validate()
        for error in validation_errors:
            errors.append(f"In transition_validation: {error}")

        # Validate automated transition configuration
        transition_errors = self.automated_transition.validate()
        for error in transition_errors:
            errors.append(f"In automated_transition: {error}")

        # Validate trading hours
        import re
        time_pattern = re.compile(r'^([0-1]?[0-9]|2[0-3]):[0-5][0-9]')
        if not time_pattern.match(self.trading_hours_start):
            errors.append("trading_hours_start must be in HH:MM format")

        if not time_pattern.match(self.trading_hours_end):
            errors.append("trading_hours_end must be in HH:MM format")

        # Validate trading days
        valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for day in self.trading_days:
            if day not in valid_days:
                errors.append(f"Invalid trading day: {day}")

        # Validate holidays
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}')
        for holiday in self.holidays:
            if not date_pattern.match(holiday):
                errors.append(f"Holiday date must be in YYYY-MM-DD format: {holiday}")

        return errors


def get_trading_mode_config(config_path: Optional[Union[str, Path]] = None) -> TradingModeConfig:
    """
    Get the trading mode configuration.

    Args:
        config_path: Optional path to a configuration file. If not provided,
                    the default path from the ConfigManager will be used.

    Returns:
        The trading mode configuration.
    """
    if config_path is None:
        config_path = ConfigManager.get_config_path("trading_mode")

    return ConfigManager.load_config(
        TradingModeConfig,
        config_path=config_path,
        env_prefix="TRADING_MODE",
        reload=False
    )