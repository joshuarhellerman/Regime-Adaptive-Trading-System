"""
pre_trade_validator.py - Pre-trade risk validation

This module provides comprehensive risk validation for orders before they are submitted
to the exchange. It enforces position limits, order size constraints, and other risk
management rules to protect against excessive exposure, erroneous orders, and market
impact concerns.

The validation is performed using a series of independent risk checks, each focusing
on a specific aspect of risk control. Each check returns a RiskCheckResult with a
severity level that determines whether the order should be rejected, warned about,
or accepted.
"""

import logging
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass

from core.event_bus import EventTopics, create_event, get_event_bus
from execution.order.order import Order, OrderStatus, OrderType, TimeInForce, OrderSide

# Configure logger
logger = logging.getLogger(__name__)


class RiskCheckLevel(Enum):
    """Risk check severity levels"""
    INFO = "info"  # Informational only
    WARNING = "warning"  # Warning but allows execution  
    ERROR = "error"  # Blocks execution


@dataclass
class RiskCheckResult:
    """Result of a risk check"""
    level: RiskCheckLevel
    message: str
    check_name: str
    details: Optional[Dict[str, Any]] = None


class PreTradeValidator:
    """
    Performs pre-trade risk validation on orders before they are submitted to exchanges.

    Responsibilities:
    - Validate order parameters
    - Enforce position and exposure limits
    - Prevent erroneous orders
    - Check for market impact concerns
    - Validate price levels
    - Verify exchange connectivity and market hours
    - Ensure sufficient capital
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the pre-trade validator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._event_bus = get_event_bus()

        # Initialize risk limits from config
        self._init_risk_limits()

        # Custom checks registry
        self._custom_checks = []

        # Track recent orders for rate limiting
        self._recent_orders = {}

        # Initialize regime-based adjustments
        self._regime_adjustments = self.config.get("regime_adjustments", {
            "high_volatility": {
                "max_order_size_multiplier": 0.7,
                "price_deviation_threshold_multiplier": 1.5
            },
            "medium_volatility": {
                "max_order_size_multiplier": 1.0,
                "price_deviation_threshold_multiplier": 1.0
            },
            "low_volatility": {
                "max_order_size_multiplier": 1.2,
                "price_deviation_threshold_multiplier": 0.8
            }
        })

        logger.info("PreTradeValidator initialized")

    def _init_risk_limits(self) -> None:
        """Initialize risk limits from configuration"""
        # Default risk limits
        self.risk_limits = {
            "max_order_size": self.config.get("max_order_size", 1000000),
            "max_order_value": self.config.get("max_order_value", 1000000),
            "max_position_size": self.config.get("max_position_size", 5000000),
            "max_position_value": self.config.get("max_position_value", 5000000),
            "max_concentration": self.config.get("max_concentration", 0.25),
            "max_overnight_exposure": self.config.get("max_overnight_exposure", 0.75),
            "max_order_count_per_minute": self.config.get("max_order_count_per_minute", 100),
            "max_daily_drawdown": self.config.get("max_daily_drawdown", 0.05),
            "price_deviation_threshold": self.config.get("price_deviation_threshold", 0.1),
            "min_order_size": self.config.get("min_order_size", 0.0001),
            "max_leverage": self.config.get("max_leverage", 50),
            "max_open_orders": self.config.get("max_open_orders", 50),
            "max_total_exposure": self.config.get("max_total_exposure", 5.0)  # 5x equity
        }

        # Per-instrument risk limits
        self.instrument_limits = self.config.get("instrument_limits", {})

        # Per-exchange risk limits
        self.exchange_limits = self.config.get("exchange_limits", {})

        # Strategy-specific limits
        self.strategy_limits = self.config.get("strategy_limits", {})

        logger.debug(f"Risk limits initialized: {self.risk_limits}")

    def register_custom_check(self, check_fn: callable) -> None:
        """
        Register a custom risk check function.

        Args:
            check_fn: Function that takes an Order and returns a RiskCheckResult
        """
        self._custom_checks.append(check_fn)
        logger.info(f"Custom risk check registered: {check_fn.__name__}")

    def validate_order(self, order: Order, position_manager=None, market_data_service=None,
                       account_manager=None, regime_info=None) -> List[RiskCheckResult]:
        """
        Validate an order against all risk checks.

        Args:
            order: Order to validate
            position_manager: Optional position manager for position-aware checks
            market_data_service: Optional market data service for price-aware checks
            account_manager: Optional account manager for margin/capital checks
            regime_info: Optional market regime information for adaptive checks

        Returns:
            List of RiskCheckResults from all checks
        """
        results = []

        # Record start time for performance tracking
        start_time = time.time()

        # Apply regime-specific adjustments to limits
        regime_adjusted_limits = self._apply_regime_adjustments(regime_info)

        # Standard checks
        results.append(self._check_order_parameters(order))
        results.append(self._check_order_size_limits(order, market_data_service, regime_adjusted_limits))
        results.append(self._check_order_frequency(order))
        results.append(self._check_instrument_specific_limits(order, regime_adjusted_limits))
        results.append(self._check_exchange_specific_limits(order, regime_adjusted_limits))
        results.append(self._check_strategy_specific_limits(order, regime_adjusted_limits))

        if market_data_service:
            results.append(self._check_price_deviation(order, market_data_service, regime_adjusted_limits))
            results.append(self._check_market_hours(order, market_data_service))
            results.append(self._check_market_conditions(order, market_data_service, regime_info))

        if position_manager:
            results.append(self._check_position_limits(order, position_manager, regime_adjusted_limits))
            results.append(self._check_concentration_limits(order, position_manager, regime_adjusted_limits))
            results.append(self._check_total_exposure(order, position_manager, account_manager, regime_adjusted_limits))
            results.append(self._check_open_orders_limit(order, position_manager))

        if account_manager:
            results.append(self._check_available_capital(order, account_manager))
            results.append(self._check_leverage_limits(order, account_manager, regime_adjusted_limits))
            results.append(self._check_margin_requirements(order, account_manager))

        # Check for black swan events
        if regime_info and regime_info.get("black_swan_detected", False):
            results.append(self._check_black_swan_limits(order, regime_info))

        # Custom checks
        for check_fn in self._custom_checks:
            try:
                result = check_fn(order, position_manager, market_data_service, account_manager, regime_info)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in custom risk check {check_fn.__name__}: {str(e)}")
                results.append(RiskCheckResult(
                    level=RiskCheckLevel.ERROR,
                    message=f"Custom check error: {str(e)}",
                    check_name=getattr(check_fn, "__name__", "custom_check")
                ))

        # Log execution time for performance monitoring
        execution_time = time.time() - start_time
        if execution_time > 0.1:  # Log if validation takes more than 100ms
            logger.warning(f"Order validation for {order.order_id} took {execution_time:.2f}s")

        # Log results
        error_count = sum(1 for r in results if r.level == RiskCheckLevel.ERROR)
        warning_count = sum(1 for r in results if r.level == RiskCheckLevel.WARNING)

        if error_count > 0:
            logger.warning(f"Order {order.order_id} failed {error_count} risk checks with errors")
        elif warning_count > 0:
            logger.info(f"Order {order.order_id} passed with {warning_count} warnings")
        else:
            logger.debug(f"Order {order.order_id} passed all risk checks")

        # Publish risk check event
        event = create_event(
            EventTopics.RISK_CHECKS_COMPLETED,
            {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "error_count": error_count,
                "warning_count": warning_count,
                "execution_time": execution_time,
                "timestamp": time.time()
            }
        )
        self._event_bus.publish(event)

        # Track order for frequency limits
        self._record_order_for_rate_limiting(order)

        return results

    def _apply_regime_adjustments(self, regime_info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply regime-specific adjustments to risk limits.

        Args:
            regime_info: Market regime information

        Returns:
            Adjusted risk limits
        """
        if not regime_info:
            return self.risk_limits.copy()

        # Copy base limits
        adjusted_limits = self.risk_limits.copy()

        # Determine volatility regime
        volatility_regime = regime_info.get("volatility_regime", "medium_volatility")
        if not volatility_regime.endswith("_volatility"):
            volatility_regime = f"{volatility_regime}_volatility"

        # Apply regime-specific adjustments
        regime_adjustments = self._regime_adjustments.get(volatility_regime, {})
        for key, multiplier in regime_adjustments.items():
            if key.endswith("_multiplier"):
                base_key = key[:-11]  # Remove "_multiplier" suffix
                if base_key in adjusted_limits:
                    adjusted_limits[base_key] *= multiplier

        # Apply black swan adjustments
        if regime_info.get("black_swan_detected", False):
            # In black swan events, reduce size limits by 50%
            for key in ["max_order_size", "max_order_value", "max_position_size", "max_position_value"]:
                adjusted_limits[key] *= 0.5

            # Increase price deviation thresholds to account for volatility
            adjusted_limits["price_deviation_threshold"] *= 2.0

        return adjusted_limits

    def _record_order_for_rate_limiting(self, order: Order) -> None:
        """
        Record order for rate limiting purposes.

        Args:
            order: The order to record
        """
        now = time.time()
        symbol = order.symbol

        # Initialize if needed
        if symbol not in self._recent_orders:
            self._recent_orders[symbol] = []

        # Add current order
        self._recent_orders[symbol].append(now)

        # Remove orders older than 1 minute
        self._recent_orders[symbol] = [t for t in self._recent_orders[symbol] if now - t < 60]

    def _check_order_parameters(self, order: Order) -> RiskCheckResult:
        """
        Validate basic order parameters.

        Args:
            order: Order to validate

        Returns:
            RiskCheckResult
        """
        # Check for required fields
        if not order.symbol:
            return RiskCheckResult(
                level=RiskCheckLevel.ERROR,
                message="Missing symbol",
                check_name="order_parameters"
            )

        if not order.quantity:
            return RiskCheckResult(
                level=RiskCheckLevel.ERROR,
                message="Missing or zero quantity",
                check_name="order_parameters"
            )

        if order.quantity < 0:
            return RiskCheckResult(
                level=RiskCheckLevel.ERROR,
                message="Negative quantity",
                check_name="order_parameters"
            )

        # Check for limit price on limit orders
        if order.order_type == OrderType.LIMIT and not order.limit_price:
            return RiskCheckResult(
                level=RiskCheckLevel.ERROR,
                message="Limit order missing price",
                check_name="order_parameters"
            )

        # Check for stop price on stop orders
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and not order.stop_price:
            return RiskCheckResult(
                level=RiskCheckLevel.ERROR,
                message=f"{order.order_type.value} order missing stop price",
                check_name="order_parameters"
            )

        # Check for minimum order size
        min_size = self.risk_limits["min_order_size"]
        if order.quantity < min_size:
            return RiskCheckResult(
                level=RiskCheckLevel.ERROR,
                message=f"Order size {order.quantity} below minimum {min_size}",
                check_name="order_parameters"
            )

        # All checks passed
        return RiskCheckResult(
            level=RiskCheckLevel.INFO,
            message="Order parameters valid",
            check_name="order_parameters"
        )

    def _check_order_size_limits(self, order: Order, market_data_service=None,
                                 adjusted_limits=None) -> RiskCheckResult:
        """
        Check if order size exceeds configured limits.

        Args:
            order: Order to validate
            market_data_service: Optional market data service for price information
            adjusted_limits: Optional adjusted risk limits based on regime

        Returns:
            RiskCheckResult
        """
        limits = adjusted_limits or self.risk_limits

        # Get max order size limit
        max_size = limits["max_order_size"]

        # Check instrument-specific limit if available
        instrument_limits = self.instrument_limits.get(order.symbol, {})
        if "max_order_size" in instrument_limits:
            max_size = instrument_limits["max_order_size"]

        # Check size
        if order.quantity > max_size:
            return RiskCheckResult(
                level=RiskCheckLevel.ERROR,
                message=f"Order size {order.quantity} exceeds maximum {max_size}",
                check_name="order_size_limits",
                details={
                    "order_size": order.quantity,
                    "max_size": max_size
                }
            )

        # Check value if price is available
        if market_data_service:
            try:
                # Use limit price if available, otherwise get market price
                price = order.limit_price
                if not price:
                    price = market_data_service.get_last_price(order.symbol)

                order_value = order.quantity * price
                max_value = limits["max_order_value"]

                # Check instrument-specific value limit
                if "max_order_value" in instrument_limits:
                    max_value = instrument_limits["max_order_value"]

                if order_value > max_value:
                    return RiskCheckResult(
                        level=RiskCheckLevel.ERROR,
                        message=f"Order value {order_value} exceeds maximum {max_value}",
                        check_name="order_size_limits",
                        details={
                            "order_value": order_value,
                            "max_value": max_value
                        }
                    )
            except Exception as e:
                logger.warning(f"Error checking order value: {str(e)}")
                return RiskCheckResult(
                    level=RiskCheckLevel.WARNING,
                    message=f"Could not validate order value: {str(e)}",
                    check_name="order_size_limits"
                )

        # All checks passed
        return RiskCheckResult(
            level=RiskCheckLevel.INFO,
            message="Order size within limits",
            check_name="order_size_limits"
        )

    def _check_position_limits(self, order: Order, position_manager,
                               adjusted_limits=None) -> RiskCheckResult:
        """
        Check if the order would exceed position limits.

        Args:
            order: Order to validate
            position_manager: Position manager for current positions
            adjusted_limits: Optional adjusted risk limits based on regime

        Returns:
            RiskCheckResult
        """
        limits = adjusted_limits or self.risk_limits

        try:
            # Get current position
            current_position = position_manager.get_position(order.symbol)

            # Calculate new position after order
            new_position = current_position
            if order.side == OrderSide.BUY:
                new_position += order.quantity
            else:
                new_position -= order.quantity

            # Check against position size limit
            max_position_size = limits["max_position_size"]

            # Get instrument-specific limit if available
            instrument_limits = self.instrument_limits.get(order.symbol, {})
            if "max_position_size" in instrument_limits:
                max_position_size = instrument_limits["max_position_size"]

            if abs(new_position) > max_position_size:
                return RiskCheckResult(
                    level=RiskCheckLevel.ERROR,
                    message=f"New position {new_position} would exceed maximum {max_position_size}",
                    check_name="position_limits",
                    details={
                        "current_position": current_position,
                        "new_position": new_position,
                        "max_position_size": max_position_size
                    }
                )

            # All checks passed
            return RiskCheckResult(
                level=RiskCheckLevel.INFO,
                message="Position within limits",
                check_name="position_limits",
                details={
                    "current_position": current_position,
                    "new_position": new_position
                }
            )
        except Exception as e:
            logger.warning(f"Error checking position limits: {str(e)}")
            return RiskCheckResult(
                level=RiskCheckLevel.WARNING,
                message=f"Could not validate position limits: {str(e)}",
                check_name="position_limits"
            )

    def _check_concentration_limits(self, order: Order, position_manager,
                                    adjusted_limits=None) -> RiskCheckResult:
        """
        Check if the order would exceed concentration limits.

        Args:
            order: Order to validate
            position_manager: Position manager for portfolio value
            adjusted_limits: Optional adjusted risk limits based on regime

        Returns:
            RiskCheckResult
        """
        limits = adjusted_limits or self.risk_limits

        try:
            # Get portfolio value
            portfolio_value = position_manager.get_total_position_value()
            if portfolio_value <= 0:
                return RiskCheckResult(
                    level=RiskCheckLevel.WARNING,
                    message="Cannot check concentration with zero portfolio value",
                    check_name="concentration_limits"
                )

            # Get position value
            position_value = position_manager.get_position_value(order.symbol)

            # Calculate order value
            order_price = order.limit_price or position_manager.get_market_price(order.symbol)
            order_value = order.quantity * order_price

            # Calculate new position value
            new_position_value = position_value
            if order.side == OrderSide.BUY:
                new_position_value += order_value
            else:
                new_position_value -= order_value

            # Calculate concentration ratio
            concentration = new_position_value / portfolio_value
            max_concentration = limits["max_concentration"]

            # Get instrument-specific limit if available
            instrument_limits = self.instrument_limits.get(order.symbol, {})
            if "max_concentration" in instrument_limits:
                max_concentration = instrument_limits["max_concentration"]

            if concentration > max_concentration:
                return RiskCheckResult(
                    level=RiskCheckLevel.ERROR,
                    message=f"New concentration {concentration:.2%} would exceed maximum {max_concentration:.2%}",
                    check_name="concentration_limits",
                    details={
                        "current_concentration": position_value / portfolio_value,
                        "new_concentration": concentration,
                        "max_concentration": max_concentration
                    }
                )

            # All checks passed
            return RiskCheckResult(
                level=RiskCheckLevel.INFO,
                message="Concentration within limits",
                check_name="concentration_limits"
            )
        except Exception as e:
            logger.warning(f"Error checking concentration limits: {str(e)}")
            return RiskCheckResult(
                level=RiskCheckLevel.WARNING,
                message=f"Could not validate concentration limits: {str(e)}",
                check_name="concentration_limits"
            )

    def _check_price_deviation(self, order: Order, market_data_service,
                               adjusted_limits=None) -> RiskCheckResult:
        """
        Check if order price deviates significantly from market price.

        Args:
            order: Order to validate
            market_data_service: Market data service for price information
            adjusted_limits: Optional adjusted risk limits based on regime

        Returns:
            RiskCheckResult
        """
        limits = adjusted_limits or self.risk_limits

        # Only check limit orders
        if order.order_type != OrderType.LIMIT or not order.limit_price:
            return RiskCheckResult(
                level=RiskCheckLevel.INFO,
                message="Not a limit order, skipping price deviation check",
                check_name="price_deviation"
            )

        try:
            # Get market price
            market_price = market_data_service.get_last_price(order.symbol)

            # Calculate deviation
            deviation = abs(order.limit_price - market_price) / market_price
            max_deviation = limits["price_deviation_threshold"]

            # Get instrument-specific limit if available
            instrument_limits = self.instrument_limits.get(order.symbol, {})
            if "price_deviation_threshold" in instrument_limits:
                max_deviation = instrument_limits["price_deviation_threshold"]

            if deviation > max_deviation:
                return RiskCheckResult(
                    level=RiskCheckLevel.WARNING if deviation < max_deviation * 2 else RiskCheckLevel.ERROR,
                    message=f"Price deviation {deviation:.2%} exceeds threshold {max_deviation:.2%}",
                    check_name="price_deviation",
                    details={
                        "limit_price": order.limit_price,
                        "market_price": market_price,
                        "deviation": deviation,
                        "max_deviation": max_deviation
                    }
                )

            # All checks passed
            return RiskCheckResult(
                level=RiskCheckLevel.INFO,
                message="Price deviation within limits",
                check_name="price_deviation"
            )
        except Exception as e:
            logger.warning(f"Error checking price deviation: {str(e)}")
            return RiskCheckResult(
                level=RiskCheckLevel.WARNING,
                message=f"Could not validate price deviation: {str(e)}",
                check_name="price_deviation"
            )

    def _check_market_hours(self, order: Order, market_data_service) -> RiskCheckResult:
        """
        Check if the market is open for trading.

        Args:
            order: Order to validate
            market_data_service: Market data service for market hours

        Returns:
            RiskCheckResult
        """
        try:
            # Check if market is open
            is_market_open = market_data_service.is_market_open(order.symbol)

            if not is_market_open:
                # Check if after-hours trading is allowed
                allow_after_hours = self.config.get("allow_after_hours", False)

                # Override for specific instruments
                instrument_config = self.instrument_limits.get(order.symbol, {})
                if "allow_after_hours" in instrument_config:
                    allow_after_hours = instrument_config["allow_after_hours"]

                if not allow_after_hours:
                    return RiskCheckResult(
                        level=RiskCheckLevel.ERROR,
                        message="Market is closed for trading",
                        check_name="market_hours"
                    )
                else:
                    return RiskCheckResult(
                        level=RiskCheckLevel.WARNING,
                        message="Market is closed, but after-hours trading is allowed",
                        check_name="market_hours"
                    )

            # All checks passed
            return RiskCheckResult(
                level=RiskCheckLevel.INFO,
                message="Market is open for trading",
                check_name="market_hours"
            )
        except Exception as e:
            logger.warning(f"Error checking market hours: {str(e)}")
            return RiskCheckResult(
                level=RiskCheckLevel.WARNING,
                message=f"Could not validate market hours: {str(e)}",
                check_name="market_hours"
            )

    def _check_order_frequency(self, order: Order) -> RiskCheckResult:
        """
        Check if order frequency exceeds rate limits.

        Args:
            order: Order to validate

        Returns:
            RiskCheckResult
        """
        symbol = order.symbol
        max_orders_per_minute = self.risk_limits["max_order_count_per_minute"]

        # Get instrument-specific limit if available
        instrument_limits = self.instrument_limits.get(symbol, {})
        if "max_order_count_per_minute" in instrument_limits:
            max_orders_per_minute = instrument_limits["max_order_count_per_minute"]

        # Check recent orders
        recent_orders_count = len(self._recent_orders.get(symbol, []))

        if recent_orders_count >= max_orders_per_minute:
            return RiskCheckResult(
                level=RiskCheckLevel.ERROR,
                message=f"Order frequency of {recent_orders_count} exceeds limit of {max_orders_per_minute} per minute",
                check_name="order_frequency",
                details={
                    "recent_orders_count": recent_orders_count,
                    "max_orders_per_minute": max_orders_per_minute
                }
            )
        elif recent_orders_count >= max_orders_per_minute * 0.8:
            return RiskCheckResult(
                level=RiskCheckLevel.WARNING,
                message=f"Order frequency of {recent_orders_count} approaching limit of {max_orders_per_minute} per minute",
                check_name="order_frequency",
                details={
                    "recent_orders_count": recent_orders_count,
                    "max_orders_per_minute": max_orders_per_minute
                }
            )

        # All checks passed
        return RiskCheckResult(
            level=RiskCheckLevel.INFO,
            message="Order frequency within limits",
            check_name="order_frequency"
        )

    def _check_instrument_specific_limits(self, order: Order, adjusted_limits=None) -> RiskCheckResult:
        """
        Check instrument-specific limits.

        Args:
            order: Order to validate
            adjusted_limits: Optional adjusted risk limits based on regime

        Returns:
            RiskCheckResult
        """
        # Get instrument-specific limits
        instrument_limits = self.instrument_limits.get(order.symbol, {})

        # If no specific limits, pass check
        if not instrument_limits:
            return RiskCheckResult(
                level=RiskCheckLevel.INFO,
                message="No instrument-specific limits configured",
                check_name="instrument_specific_limits"
            )

        # Check if instrument is allowed
        if not instrument_limits.get("allowed", True):
            return RiskCheckResult(
                level=RiskCheckLevel.ERROR,
                message=f"Trading {order.symbol} is not allowed",
                check_name="instrument_specific_limits"
            )

        # All checks passed
        return RiskCheckResult(
            level=RiskCheckLevel.INFO,
            message="Instrument-specific limits passed",
            check_name="instrument_specific_limits"
        )

    def _check_exchange_specific_limits(self, order: Order, adjusted_limits=None) -> RiskCheckResult:
        """
        Check exchange-specific limits.

        Args:
            order: Order to validate
            adjusted_limits: Optional adjusted risk limits based on regime

        Returns:
            RiskCheckResult
        """
        # Get exchange-specific limits
        exchange_limits = self.exchange_limits.get(order.exchange_id, {})

        # If no specific limits, pass check
        if not exchange_limits:
            return RiskCheckResult(
                level=RiskCheckLevel.INFO,
                message="No exchange-specific limits configured",
                check_name="exchange_specific_limits"
            )

        # Check if exchange is allowed
        if not exchange_limits.get("allowed", True):
            return RiskCheckResult(
                level=RiskCheckLevel.ERROR,
                message=f"Trading on {order.exchange_id} is not allowed",
                check_name="exchange_specific_limits"
            )

        # Check exchange-specific order size limits
        if "max_order_size" in exchange_limits and order.quantity > exchange_limits["max_order_size"]:
            return RiskCheckResult(
                level=RiskCheckLevel.ERROR,
                message=f"Order size {order.quantity} exceeds exchange limit {exchange_limits['max_order_size']}",
                check_name="exchange_specific_limits"
            )

        # All checks passed
        return RiskCheckResult(
            level=RiskCheckLevel.INFO,
            message="Exchange-specific limits passed",
            check_name="exchange_specific_limits"
        )

    def _check_strategy_specific_limits(self, order: Order, adjusted_limits=None) -> RiskCheckResult:
        """
        Check strategy-specific limits.

        Args:
            order: Order to validate
            adjusted_limits: Optional adjusted risk limits based on regime

        Returns:
            RiskCheckResult
        """
        # Get strategy from order params
        strategy = order.params.get("strategy")
        if not strategy:
            return RiskCheckResult(
                level=RiskCheckLevel.INFO,
                message="No strategy information available",
                check_name="strategy_specific_limits"
            )

        # Get strategy-specific limits
        strategy_limits = self.strategy_limits.get(strategy, {})

        # If no specific limits, pass check
        if not strategy_limits:
            return RiskCheckResult(
                level=RiskCheckLevel.INFO,
                message=f"No limits configured for strategy: {strategy}",
                check_name="strategy_specific_limits"
            )

        # Check if strategy is allowed
        if not strategy_limits.get("enabled", True):
            return RiskCheckResult(
                level=RiskCheckLevel.ERROR,
                message=f"Strategy {strategy} is currently disabled",
                check_name="strategy_specific_limits"
            )

        # Check strategy-specific max position
        if "max_position_size" in strategy_limits and order.quantity > strategy_limits["max_position_size"]:
            return RiskCheckResult(
                level=RiskCheckLevel.ERROR,
                message=f"Order size {order.quantity} exceeds strategy limit {strategy_limits['max_position_size']}",
                check_name="strategy_specific_limits",
                details={
                    "strategy": strategy,
                    "order_size": order.quantity,
                    "max_position_size": strategy_limits["max_position_size"]
                }
            )

        # Check strategy-specific size as percentage of equity
        if "max_equity_percent" in strategy_limits and "equity" in order.params:
            equity = order.params["equity"]
            max_percent = strategy_limits["max_equity_percent"]
            order_value = order.quantity * (order.limit_price or order.params.get("market_price", 0))
            equity_percent = order_value / equity if equity > 0 else float('inf')

            if equity_percent > max_percent:
                return RiskCheckResult(
                    level=RiskCheckLevel.ERROR,
                    message=f"Order value {equity_percent:.2%} of equity exceeds strategy limit {max_percent:.2%}",
                    check_name="strategy_specific_limits",
                    details={
                        "strategy": strategy,
                        "equity_percent": equity_percent,
                        "max_equity_percent": max_percent
                    }
                )

        # All checks passed
        return RiskCheckResult(
            level=RiskCheckLevel.INFO,
            message="Strategy-specific limits passed",
            check_name="strategy_specific_limits"
        )

    def _check_total_exposure(self, order: Order, position_manager, account_manager=None,
                              adjusted_limits=None) -> RiskCheckResult:
        """
        Check if total exposure would exceed limits.

        Args:
            order: Order to validate
            position_manager: Position manager
            account_manager: Optional account manager for equity information
            adjusted_limits: Optional adjusted risk limits based on regime

        Returns:
            RiskCheckResult
        """
        limits = adjusted_limits or self.risk_limits

        try:
            # Get current total exposure
            current_exposure = position_manager.get_total_position_value()

            # Get equity value
            equity = account_manager.get_equity() if account_manager else order.params.get("equity", current_exposure)
            if equity <= 0:
                return RiskCheckResult(
                    level=RiskCheckLevel.WARNING,
                    message="Cannot calculate exposure with zero equity",
                    check_name="total_exposure"
                )

            # Calculate order value
            order_price = order.limit_price or position_manager.get_market_price(order.symbol)
            order_value = order.quantity * order_price

            # Calculate new total exposure
            new_exposure = current_exposure + order_value if order.side == OrderSide.BUY else current_exposure - order_value

            # Calculate exposure ratio
            exposure_ratio = new_exposure / equity
            max_exposure = limits["max_total_exposure"]

            if exposure_ratio > max_exposure:
                return RiskCheckResult(
                    level=RiskCheckLevel.ERROR,
                    message=f"New exposure ratio {exposure_ratio:.2f} exceeds maximum {max_exposure:.2f}",
                    check_name="total_exposure",
                    details={
                        "current_exposure": current_exposure,
                        "new_exposure": new_exposure,
                        "equity": equity,
                        "exposure_ratio": exposure_ratio,
                        "max_exposure": max_exposure
                    }
                )

            # All checks passed
            return RiskCheckResult(
                level=RiskCheckLevel.INFO,
                message="Total exposure within limits",
                check_name="total_exposure"
            )
        except Exception as e:
            logger.warning(f"Error checking total exposure: {str(e)}")
            return RiskCheckResult(
                level=RiskCheckLevel.WARNING,
                message=f"Could not validate total exposure: {str(e)}",
                check_name="total_exposure"
            )

    def _check_open_orders_limit(self, order: Order, position_manager) -> RiskCheckResult:
        """
        Check if open orders exceed limits.

        Args:
            order: Order to validate
            position_manager: Position manager

        Returns:
            RiskCheckResult
        """
        try:
            # Count open orders
            open_orders = 0
            positions = position_manager.get_all_positions()
            for position in positions.values():
                if hasattr(position, "open_orders"):
                    open_orders += len(position.open_orders)

            max_open_orders = self.risk_limits["max_open_orders"]

            if open_orders >= max_open_orders:
                return RiskCheckResult(
                    level=RiskCheckLevel.ERROR,
                    message=f"Open orders count {open_orders} exceeds maximum {max_open_orders}",
                    check_name="open_orders_limit",
                    details={
                        "open_orders": open_orders,
                        "max_open_orders": max_open_orders
                    }
                )

            # All checks passed
            return RiskCheckResult(
                level=RiskCheckLevel.INFO,
                message="Open orders count within limits",
                check_name="open_orders_limit"
            )
        except Exception as e:
            logger.warning(f"Error checking open orders limit: {str(e)}")
            return RiskCheckResult(
                level=RiskCheckLevel.WARNING,
                message=f"Could not validate open orders limit: {str(e)}",
                check_name="open_orders_limit"
            )

    def _check_available_capital(self, order: Order, account_manager) -> RiskCheckResult:
        """
        Check if there is sufficient capital available.

        Args:
            order: Order to validate
            account_manager: Account manager for capital information

        Returns:
            RiskCheckResult
        """
        try:
            # Check buying power or available balance
            available_capital = account_manager.get_available_capital()

            # Calculate required capital
            order_price = order.limit_price or account_manager.get_market_price(order.symbol)
            required_capital = order.quantity * order_price

            # For margin accounts, adjust by margin requirement
            if hasattr(account_manager, "get_margin_requirement"):
                margin_requirement = account_manager.get_margin_requirement(order.symbol)
                required_capital *= margin_requirement

            if required_capital > available_capital:
                return RiskCheckResult(
                    level=RiskCheckLevel.ERROR,
                    message=f"Required capital {required_capital} exceeds available capital {available_capital}",
                    check_name="available_capital",
                    details={
                        "required_capital": required_capital,
                        "available_capital": available_capital
                    }
                )

            # All checks passed
            return RiskCheckResult(
                level=RiskCheckLevel.INFO,
                message="Sufficient capital available",
                check_name="available_capital"
            )
        except Exception as e:
            logger.warning(f"Error checking available capital: {str(e)}")
            return RiskCheckResult(
                level=RiskCheckLevel.WARNING,
                message=f"Could not validate available capital: {str(e)}",
                check_name="available_capital"
            )

    def _check_leverage_limits(self, order: Order, account_manager, adjusted_limits=None) -> RiskCheckResult:
        """
        Check if leverage exceeds limits.

        Args:
            order: Order to validate
            account_manager: Account manager
            adjusted_limits: Optional adjusted risk limits based on regime

        Returns:
            RiskCheckResult
        """
        limits = adjusted_limits or self.risk_limits

        try:
            # Get current leverage
            current_leverage = account_manager.get_current_leverage()

            # Get max leverage
            max_leverage = limits["max_leverage"]

            # Get instrument-specific limit if available
            instrument_limits = self.instrument_limits.get(order.symbol, {})
            if "max_leverage" in instrument_limits:
                max_leverage = instrument_limits["max_leverage"]

            # Check if leverage would exceed limit
            if current_leverage > max_leverage:
                return RiskCheckResult(
                    level=RiskCheckLevel.ERROR,
                    message=f"Current leverage {current_leverage:.2f}x exceeds maximum {max_leverage:.2f}x",
                    check_name="leverage_limits",
                    details={
                        "current_leverage": current_leverage,
                        "max_leverage": max_leverage
                    }
                )

            # Calculate new leverage after order
            # This would need specific implementation based on account manager capabilities

            # All checks passed
            return RiskCheckResult(
                level=RiskCheckLevel.INFO,
                message="Leverage within limits",
                check_name="leverage_limits"
            )
        except Exception as e:
            logger.warning(f"Error checking leverage limits: {str(e)}")
            return RiskCheckResult(
                level=RiskCheckLevel.WARNING,
                message=f"Could not validate leverage limits: {str(e)}",
                check_name="leverage_limits"
            )

    def _check_margin_requirements(self, order: Order, account_manager) -> RiskCheckResult:
        """
        Check if margin requirements can be satisfied.

        Args:
            order: Order to validate
            account_manager: Account manager for margin information

        Returns:
            RiskCheckResult
        """
        try:
            # Check if account manager supports margin calculations
            if not hasattr(account_manager, "check_margin_requirement"):
                return RiskCheckResult(
                    level=RiskCheckLevel.INFO,
                    message="Margin check not supported by account manager",
                    check_name="margin_requirements"
                )

            # Check margin requirement
            margin_check_result = account_manager.check_margin_requirement(order)

            if not margin_check_result["passed"]:
                return RiskCheckResult(
                    level=RiskCheckLevel.ERROR,
                    message=f"Margin check failed: {margin_check_result.get('reason', 'Insufficient margin')}",
                    check_name="margin_requirements",
                    details=margin_check_result
                )

            # All checks passed
            return RiskCheckResult(
                level=RiskCheckLevel.INFO,
                message="Margin requirements satisfied",
                check_name="margin_requirements"
            )
        except Exception as e:
            logger.warning(f"Error checking margin requirements: {str(e)}")
            return RiskCheckResult(
                level=RiskCheckLevel.WARNING,
                message=f"Could not validate margin requirements: {str(e)}",
                check_name="margin_requirements"
            )

    def _check_market_conditions(self, order: Order, market_data_service, regime_info=None) -> RiskCheckResult:
        """
        Check if market conditions are suitable for the order.

        Args:
            order: Order to validate
            market_data_service: Market data service
            regime_info: Optional market regime information

        Returns:
            RiskCheckResult
        """
        try:
            # Check for extreme volatility
            volatility = market_data_service.get_volatility(order.symbol)
            volatility_threshold = self.config.get("extreme_volatility_threshold", 0.05)  # 5% is extreme

            if volatility > volatility_threshold:
                return RiskCheckResult(
                    level=RiskCheckLevel.WARNING,
                    message=f"Extreme volatility detected: {volatility:.2%}",
                    check_name="market_conditions",
                    details={
                        "volatility": volatility,
                        "threshold": volatility_threshold
                    }
                )

            # Check for liquidity
            liquidity = market_data_service.get_liquidity(order.symbol)
            min_liquidity = self.config.get("min_liquidity_threshold", 1000000)  # Minimum daily volume

            if liquidity < min_liquidity:
                return RiskCheckResult(
                    level=RiskCheckLevel.WARNING,
                    message=f"Low liquidity detected: {liquidity}",
                    check_name="market_conditions",
                    details={
                        "liquidity": liquidity,
                        "threshold": min_liquidity
                    }
                )

            # Check regime-specific warnings
            if regime_info:
                if regime_info.get("volatility_regime") == "high":
                    return RiskCheckResult(
                        level=RiskCheckLevel.WARNING,
                        message="High volatility regime detected",
                        check_name="market_conditions",
                        details=regime_info
                    )

            # All checks passed
            return RiskCheckResult(
                level=RiskCheckLevel.INFO,
                message="Market conditions acceptable",
                check_name="market_conditions"
            )
        except Exception as e:
            logger.warning(f"Error checking market conditions: {str(e)}")
            return RiskCheckResult(
                level=RiskCheckLevel.WARNING,
                message=f"Could not validate market conditions: {str(e)}",
                check_name="market_conditions"
            )

    def _check_black_swan_limits(self, order: Order, regime_info) -> RiskCheckResult:
        """
        Apply special limits during black swan events.

        Args:
            order: Order to validate
            regime_info: Market regime information

        Returns:
            RiskCheckResult
        """
        if not regime_info.get("black_swan_detected", False):
            return RiskCheckResult(
                level=RiskCheckLevel.INFO,
                message="No black swan event detected",
                check_name="black_swan_limits"
            )

        # Get black swan type
        black_swan_type = regime_info.get("black_swan_type", "unknown")

        # Apply special logic based on black swan type
        if black_swan_type == "market_crash":
            # Restrict long positions during crash
            if order.side == OrderSide.BUY:
                return RiskCheckResult(
                    level=RiskCheckLevel.ERROR,
                    message="Long positions restricted during market crash",
                    check_name="black_swan_limits",
                    details={
                        "black_swan_type": black_swan_type
                    }
                )
        elif black_swan_type == "flash_crash":
            # Restrict all trading during flash crash
            return RiskCheckResult(
                level=RiskCheckLevel.ERROR,
                message="All trading restricted during flash crash",
                check_name="black_swan_limits",
                details={
                    "black_swan_type": black_swan_type
                }
            )
        elif black_swan_type == "liquidity_crisis":
            # Check order size during liquidity crisis
            max_order_size = self.risk_limits["max_order_size"] * 0.2  # 80% reduction
            if order.quantity > max_order_size:
                return RiskCheckResult(
                    level=RiskCheckLevel.ERROR,
                    message=f"Order size {order.quantity} exceeds black swan limit {max_order_size}",
                    check_name="black_swan_limits",
                    details={
                        "black_swan_type": black_swan_type,
                        "max_order_size": max_order_size
                    }
                )

        # Warning for all orders during black swan events
        return RiskCheckResult(
            level=RiskCheckLevel.WARNING,
            message=f"Trading during black swan event: {black_swan_type}",
            check_name="black_swan_limits",
            details={
                "black_swan_type": black_swan_type
            }
        )

    def update_limits(self, new_limits: Dict[str, Any]) -> None:
        """
        Update risk limits dynamically.

        Args:
            new_limits: New risk limits
        """
        self.risk_limits.update(new_limits)
        logger.info(f"Risk limits updated: {new_limits}")

    def update_instrument_limits(self, symbol: str, limits: Dict[str, Any]) -> None:
        """
        Update instrument-specific limits.

        Args:
            symbol: Trading symbol
            limits: New limits for the instrument
        """
        if symbol not in self.instrument_limits:
            self.instrument_limits[symbol] = {}

        self.instrument_limits[symbol].update(limits)
        logger.info(f"Instrument limits updated for {symbol}: {limits}")

    def update_exchange_limits(self, exchange_id: str, limits: Dict[str, Any]) -> None:
        """
        Update exchange-specific limits.

        Args:
            exchange_id: Exchange ID
            limits: New limits for the exchange
        """
        if exchange_id not in self.exchange_limits:
            self.exchange_limits[exchange_id] = {}

        self.exchange_limits[exchange_id].update(limits)
        logger.info(f"Exchange limits updated for {exchange_id}: {limits}")

    def update_strategy_limits(self, strategy: str, limits: Dict[str, Any]) -> None:
        """
        Update strategy-specific limits.

        Args:
            strategy: Strategy name
            limits: New limits for the strategy
        """
        if strategy not in self.strategy_limits:
            self.strategy_limits[strategy] = {}

        self.strategy_limits[strategy].update(limits)
        logger.info(f"Strategy limits updated for {strategy}: {limits}")

    def update_regime_adjustments(self, regime_adjustments: Dict[str, Any]) -> None:
        """
        Update regime-specific adjustments.

        Args:
            regime_adjustments: New regime adjustments
        """
        self._regime_adjustments.update(regime_adjustments)
        logger.info(f"Regime adjustments updated: {regime_adjustments}")

    def reset(self) -> None:
        """Reset pre-trade validator state."""
        self._init_risk_limits()
        self._recent_orders = {}
        logger.info("PreTradeValidator state reset")